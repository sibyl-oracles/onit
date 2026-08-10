"""
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Chat function supporting private vLLM, OpenRouter.ai, and Ollama cloud models.
Provider is auto-detected from the host URL.
"""

import asyncio
import base64
import logging
import mimetypes
import os
import json
import re
import time
import types
import uuid
import httpx
from pathlib import Path
from openai import AsyncOpenAI, OpenAIError, APITimeoutError, NotFoundError
from typing import List, Optional, Any

from .verify import VERDICT_MAX_TOKENS, needs_verification, verify_answer

try:
    from ...lib.text import split_instruction
    from ...lib.schema import coerce_arguments, validate_arguments
    from ...learn import describe_tool_call, redact_tool_args
except ImportError:  # imported with src/ itself on sys.path (tests, scripts)
    from lib.text import split_instruction
    from lib.schema import coerce_arguments, validate_arguments
    from learn import describe_tool_call, redact_tool_args

try:
    from ollama import AsyncClient as OllamaAsyncClient
    OLLAMA_SDK_AVAILABLE = True
except ImportError:
    OllamaAsyncClient = None
    OLLAMA_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

# Maximum characters for a tool response stored in conversation history.
# Larger responses are truncated to avoid blowing up the context window.
MAX_TOOL_RESPONSE = 16000

# Stall detection for the "no timeout" (-1) configuration.  Even with no
# overall request timeout, the connect phase and the gap between streamed
# chunks must be bounded — otherwise a wedged server (e.g. vLLM guided
# decoding stalling on tool_choice="required") hangs the chat loop forever
# with no way to recover.
CONNECT_TIMEOUT = 30.0
STREAM_STALL_TIMEOUT = 300.0


def _build_client_timeout(timeout, stream: bool):
    """Return the timeout configuration for the API client.

    A positive ``timeout`` is used as-is (total request timeout).  ``None`` or
    a negative value means "no overall limit": connect time is still bounded,
    and when streaming, the gap between chunks is bounded by
    STREAM_STALL_TIMEOUT.  Streaming read timeouts apply per-chunk, so long
    generations are unaffected — only a genuine stall raises.
    """
    if timeout is not None and timeout >= 0:
        return timeout
    read = STREAM_STALL_TIMEOUT if stream else None
    return httpx.Timeout(connect=CONNECT_TIMEOUT, read=read, write=30.0, pool=30.0)


def _api_tool_payload(tool_items: list) -> list:
    """Project internal tool records onto what the chat-completions API defines.

    ``function`` carries exactly name/description/parameters there.  The
    registry also records ``returns``, captured from each MCP tool's
    outputSchema by ``_build_returns``, which is ours alone: no provider reads
    it, and one that validates the tool schema strictly rejects the request for
    carrying it.  Until then it is dead bytes in the payload of every request
    of every turn, on every tool.

    A projection rather than a deletion, so a field added to the internal
    record later does not reach the wire just because nobody thought about it.
    Anything that is not a recognizable tool record is passed through
    untouched — test doubles hand this whatever they please, and a payload
    builder is the wrong place to start rejecting things.
    """
    payload = []
    for item in tool_items:
        if not isinstance(item, dict) or not isinstance(item.get('function'), dict):
            payload.append(item)
            continue
        fn = item['function']
        projected = {k: fn[k] for k in ('name', 'description', 'parameters')
                     if k in fn}
        payload.append({**{k: v for k, v in item.items() if k != 'function'},
                        'function': projected})
    return payload


def _truncate_tool_response(response: str) -> str:
    """Truncate a tool response if it exceeds MAX_TOOL_RESPONSE characters."""
    if len(response) <= MAX_TOOL_RESPONSE:
        return response
    half = MAX_TOOL_RESPONSE // 2
    return response[:half] + f"\n\n... [truncated {len(response) - MAX_TOOL_RESPONSE} chars] ...\n\n" + response[-half:]


# ── per-turn telemetry ──────────────────────────────────────────────────────

def _as_int(value) -> int:
    """Token counts as an int, whatever the provider actually returned.

    ``usage`` fields are absent on some providers, None on others, and a mock
    in tests; none of those should poison an accumulator.
    """
    return value if isinstance(value, int) else 0


def _as_positive_or_disabled(value, default: int = 50) -> int:
    """A loop ceiling from config: a positive count, or -1 for "no ceiling".

    YAML hands back whatever was typed, so a ceiling can arrive as a string, a
    float, or None.  Anything that is not a usable count falls back to
    ``default`` rather than to the opt-out, because the failure this guards
    against is a typo quietly turning a bound into an unbounded loop.  Values
    below 1 are the explicit opt-out and normalize to -1.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n >= 1 else -1


class TurnMetrics:
    """Timing and token accounting for one ``chat()`` run, turn by turn.

    Wall time for an answer is the sum over turns of prefill + decode + tool
    execution, but the only figure reported was the decode rate of the final
    stream — blind to how many turns ran, to a prompt that grows with every
    tool result, and to the tools themselves.  A run can double in length with
    that number unchanged, which is exactly the case this exists to diagnose.

    The caller owns the dict; it is filled in as the loop runs rather than at
    the end, because the loop returns from a dozen places and a summary built
    on the way out would be missing from most of them.
    """

    def __init__(self, sink: dict):
        self.sink = sink
        self.turns: list = []
        sink.clear()
        sink.update(turns=self.turns, turn_count=0, tool_calls=0,
                    model_s=0.0, tool_s=0.0, prefill_s=0.0, decode_s=0.0,
                    compaction_s=0.0, compactions=0, api_retries=0,
                    prompt_tokens_max=0, completion_tokens=0)
        self._api_start = 0.0
        self._ttft = None

    # -- model call --------------------------------------------------------
    def start_api(self) -> None:
        """Mark the start of an API call.  Called again on each retry, so a
        retried turn is timed from the attempt that actually answered."""
        self._api_start = time.monotonic()
        self._ttft = None

    def first_token(self) -> None:
        """First chunk carrying generated text: the prefill is over."""
        if self._ttft is None and self._api_start:
            self._ttft = time.monotonic() - self._api_start

    def end_api(self, prompt_tokens=0, completion_tokens=0,
                finish_reason=None) -> None:
        elapsed = time.monotonic() - self._api_start if self._api_start else 0.0
        prompt_tokens = _as_int(prompt_tokens)
        completion_tokens = _as_int(completion_tokens)
        self.turns.append({
            "n": len(self.turns) + 1,
            "ttft_s": round(self._ttft, 3) if self._ttft is not None else None,
            "model_s": round(elapsed, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "tools": [],
            "tool_s": 0.0,
        })
        s = self.sink
        s["turn_count"] = len(self.turns)
        s["model_s"] = round(s["model_s"] + elapsed, 3)
        s["completion_tokens"] += completion_tokens
        s["prompt_tokens_max"] = max(s["prompt_tokens_max"], prompt_tokens)
        if self._ttft is not None:
            # Only streamed turns can separate the two: without a first-token
            # timestamp there is no line between waiting and generating.
            s["prefill_s"] = round(s["prefill_s"] + self._ttft, 3)
            s["decode_s"] = round(s["decode_s"] + max(elapsed - self._ttft, 0.0), 3)
        self._api_start = 0.0

    # -- tools and compaction ----------------------------------------------
    def add_tools(self, names: list, seconds: float, runs: list | None = None) -> None:
        """Record the tools a turn ran.

        ``runs`` carries how each call went — ok, duration, result size — which
        is what makes a trajectory diagnosable after the fact.  It is optional
        because the aggregate is what the status line needs, and a caller that
        only has the names still gets correct timing and counts.
        """
        if self.turns:
            self.turns[-1]["tools"] = list(names)
            self.turns[-1]["tool_s"] = round(seconds, 3)
            if runs:
                self.turns[-1]["tool_runs"] = list(runs)
        self.sink["tool_calls"] += len(names)
        self.sink["tool_s"] = round(self.sink["tool_s"] + seconds, 3)

    def add_retry(self) -> None:
        """An API call that had to be made again.  A run that retried twice and
        answered is not the same run as one that answered first time, and the
        turn timings alone cannot tell them apart."""
        self.sink["api_retries"] += 1

    def add_compaction(self, seconds: float) -> None:
        """Compaction is an extra LLM call the user never asked for — counted
        separately so its cost isn't read as the model being slow."""
        self.sink["compactions"] += 1
        self.sink["compaction_s"] = round(self.sink["compaction_s"] + seconds, 3)

    def add_verification(self, seconds: float, issues: int = 0,
                         revised: bool = False) -> None:
        """The fact-check that runs after the answer is written.

        It lands after the last token the user saw, so it is time they wait
        with a finished answer already on screen — a different kind of second
        from the ones before it, and worth telling apart when the total looks
        too high.
        """
        s = self.sink
        s["verify_s"] = round(s.get("verify_s", 0.0) + seconds, 3)
        s["verify_issues"] = s.get("verify_issues", 0) + issues
        s["verify_revisions"] = s.get("verify_revisions", 0) + (1 if revised else 0)


def summarize_metrics(m: dict) -> str:
    """One-line rendering of a TurnMetrics sink, for logs and status lines."""
    if not m or not m.get("turn_count"):
        return "no turns recorded"
    parts = [
        f"{m['turn_count']} turn(s)",
        f"{m['tool_calls']} tool call(s)",
        f"model {m['model_s']:.1f}s (prefill {m['prefill_s']:.1f}s, "
        f"decode {m['decode_s']:.1f}s)",
        f"tools {m['tool_s']:.1f}s",
        f"peak prompt {m['prompt_tokens_max']:,} tok",
        f"generated {m['completion_tokens']:,} tok",
    ]
    if m.get("compactions"):
        parts.append(f"{m['compactions']} compaction(s) {m['compaction_s']:.1f}s")
    if m.get("verify_s"):
        _verdict = (f"{m['verify_issues']} claim(s) corrected"
                    if m.get("verify_revisions") else "clean")
        parts.append(f"fact-check {m['verify_s']:.1f}s ({_verdict})")
    return " | ".join(parts)


def _log_to_ui_or_verbose(message: str, chat_ui, verbose: bool, level: str = "info") -> None:
    if chat_ui:
        chat_ui.add_log(message, level=level)
    elif verbose:
        print(message)


def _is_ollama_host(host: str) -> bool:
    """Return True when host points to an Ollama cloud endpoint."""
    return "ollama.com" in host or "ollama.ai" in host


def _resolve_api_key(host: str, host_key: str = "EMPTY") -> str:
    """Resolve the API key based on the host URL.

    For OpenRouter hosts, use host_key param or OPENROUTER_API_KEY env var
    or OS keychain. For Ollama cloud hosts, use OLLAMA_API_KEY env var or
    keychain. For vLLM and other hosts, use host_key param or VLLM_API_KEY
    env var or keychain, defaulting to "EMPTY" (vLLM without --api-key
    accepts any bearer token).
    """
    if "openrouter.ai" in host:
        if host_key and host_key != "EMPTY":
            return host_key
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            # Try OS keychain via setup module
            try:
                from src.setup import get_secret
                key = get_secret("host_key") or ""
            except Exception:
                pass
        if not key:
            raise ValueError(
                "OpenRouter requires an API key. Set it via:\n"
                "  - onit setup (recommended)\n"
                "  - serving.host_key in the config YAML\n"
                "  - OPENROUTER_API_KEY environment variable"
            )
        return key
    if _is_ollama_host(host):
        if host_key and host_key != "EMPTY":
            return host_key
        key = os.environ.get("OLLAMA_API_KEY", "")
        if not key:
            try:
                from src.setup import get_secret
                key = get_secret("ollama_api_key") or ""
            except Exception:
                pass
        if not key:
            raise ValueError(
                "Ollama cloud requires an API key. Set it via:\n"
                "  - onit setup (recommended)\n"
                "  - serving.host_key in the config YAML\n"
                "  - OLLAMA_API_KEY environment variable"
            )
        return key
    if host_key and host_key != "EMPTY":
        return host_key
    key = os.environ.get("VLLM_API_KEY", "")
    if not key:
        try:
            from src.setup import get_secret
            key = get_secret("vllm_api_key") or ""
        except Exception:
            pass
    return key or host_key


def _create_ollama_client(host: str, api_key: str, timeout, stream: bool = True):
    """Create an OllamaAsyncClient configured for the given host and API key."""
    if not OLLAMA_SDK_AVAILABLE:
        raise ImportError("The 'ollama' package is required for Ollama cloud support.")
    return OllamaAsyncClient(
        host=host,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=_build_client_timeout(timeout, stream),
    )


# ── endpoint metadata caches ────────────────────────────────────────────────
# The model a host serves and the size of its context window are properties of
# a running server, not of a request, but both were re-queried on every chat()
# call — two HTTP round trips sitting in front of the first token of every
# task.  They are cached here.
#
# The caches live inside the helpers rather than at the call site in chat() so
# that a test patching a helper patches the caching away with it, and so the
# fallback path below can invalidate a stale answer in one place.
#
# A successful answer is kept for the life of the process.  A failure is kept
# only briefly: a host that was down or slow during one call is asked again on
# the next, instead of a single bad moment costing context accounting until
# the process restarts.
_MODEL_ID_CACHE: dict = {}        # host -> model id
_MAX_CONTEXT_CACHE: dict = {}     # (host, model) -> (value, expires_at)
_OLLAMA_CONTEXT_CACHE: dict = {}  # model -> (value, expires_at)
_NEGATIVE_CACHE_TTL = 300.0       # seconds a "could not determine" answer sticks

# Hosts that rejected chat_template_kwargs (see _verify_ask).  Learned once per
# process rather than rediscovered on every answer: the fact-check asks the
# template to skip thinking, and a server whose template has no such switch
# should cost one failed call, not one per turn.
_NO_TEMPLATE_KWARGS: set = set()


def reset_endpoint_caches() -> None:
    """Forget every cached model id and context window.

    For tests, and for callers that reconfigure an endpoint in-process.
    """
    _MODEL_ID_CACHE.clear()
    _MAX_CONTEXT_CACHE.clear()
    _OLLAMA_CONTEXT_CACHE.clear()
    _NO_TEMPLATE_KWARGS.clear()


def _cache_get(cache: dict, key) -> tuple:
    """Look up a (value, expires_at) entry. Returns (hit, value).

    A cached success never expires; a cached failure does.
    """
    entry = cache.get(key)
    if entry is None:
        return False, None
    value, expires_at = entry
    if value is not None or expires_at > time.monotonic():
        return True, value
    return False, None


def _cache_put(cache: dict, key, value):
    cache[key] = (value, time.monotonic() + _NEGATIVE_CACHE_TTL)
    return value


async def _ollama_resolve_model_id(client, host: str) -> str:
    """Fetch the first available model from an Ollama endpoint via client.list()."""
    cached = _MODEL_ID_CACHE.get(host)
    if cached:
        return cached
    response = await client.list()
    if not response.models:
        raise ValueError(f"No models available at {host}")
    model_id = response.models[0].model
    logger.info("Auto-detected Ollama model: %s from %s", model_id, host)
    _MODEL_ID_CACHE[host] = model_id
    return model_id


async def _resolve_model_id(client: AsyncOpenAI, host: str) -> str:
    """Fetch the first available model ID from the endpoint.

    vLLM typically serves a single model, so models.data[0].id is used.
    For OpenRouter, the caller should always supply the model name explicitly.
    """
    cached = _MODEL_ID_CACHE.get(host)
    if cached:
        return cached
    models = await client.models.list()
    if not models.data:
        raise ValueError(f"No models available at {host}")
    model_id = models.data[0].id
    logger.info("Auto-detected model: %s from %s", model_id, host)
    _MODEL_ID_CACHE[host] = model_id
    return model_id


async def _autodetect_fallback_model(client, ollama_client, is_ollama: bool,
                                     host: str, current_model: str) -> Optional[str]:
    """After a 404 for the configured model, auto-detect what the host serves.

    A stale ``serving.model``/``model2`` (e.g. after the server was redeployed
    with a different model) 404s on every request even though the host is up.
    Returns the detected model name, or None if detection failed or it matches
    the name that already 404'd.
    """
    # The cached id is what just 404'd if it came from an earlier auto-detect,
    # and a cache hit here would re-detect the dead name, compare it equal to
    # the current model, and report "no fallback" forever.  Ask the host again.
    _MODEL_ID_CACHE.pop(host, None)
    try:
        detected = (await _ollama_resolve_model_id(ollama_client, host) if is_ollama
                    else await _resolve_model_id(client, host))
    except Exception as e:
        logger.error("Model fallback auto-detection failed for %s: %s", host, e)
        return None
    return detected if detected != current_model else None


async def _get_model_max_context(host: str, api_key: str, model: str) -> Optional[int]:
    """Query vLLM for the model's maximum context length.

    vLLM exposes ``max_model_len`` in the ``/v1/models`` response as an extra
    field.  Returns None for OpenRouter or any host that doesn't provide it.
    """
    hit, cached = _cache_get(_MAX_CONTEXT_CACHE, (host, model))
    if hit:
        return cached
    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            resp = await http_client.get(
                f"{host.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code == 200:
                for m in resp.json().get("data", []):
                    if m.get("id") == model:
                        val = m.get("max_model_len")
                        if val:
                            return _cache_put(_MAX_CONTEXT_CACHE, (host, model), int(val))
    except Exception as e:
        logger.debug("Could not query max_model_len from vLLM: %s", e)
    return _cache_put(_MAX_CONTEXT_CACHE, (host, model), None)


async def _get_ollama_max_context(client, model: str) -> Optional[int]:
    """Query Ollama for the model's native maximum context length.

    Ollama exposes per-architecture metadata via ``client.show(model)``; the
    context window lives in ``modelinfo`` under the key ``<arch>.context_length``
    (e.g. ``qwen3.context_length``).  Returns None if it can't be determined.

    This matters because Ollama defaults ``num_ctx`` to 2048 unless told
    otherwise — far smaller than ``max_tokens`` — which silently truncates long
    responses mid-stream.  Knowing the native window lets us size ``num_ctx`` and
    drive the same context-compaction accounting used for vLLM.

    Cached by model name alone: the context window is a property of the model,
    so two hosts serving the same model agree on it.
    """
    hit, cached = _cache_get(_OLLAMA_CONTEXT_CACHE, model)
    if hit:
        return cached
    try:
        resp = await client.show(model)
        modelinfo = getattr(resp, "modelinfo", None) or {}
        for key, val in modelinfo.items():
            if key.endswith(".context_length") and val:
                return _cache_put(_OLLAMA_CONTEXT_CACHE, model, int(val))
    except Exception as e:
        logger.debug("Could not query context_length from Ollama: %s", e)
    return _cache_put(_OLLAMA_CONTEXT_CACHE, model, None)


def _parse_commands_format(obj: dict, tool_registry) -> Optional[list[dict]]:
    """Parse the commands-style response format used by some models.

    Expects a dict like:
        {"state_analysis": "...", "explanation": "...",
         "commands": [{"keystrokes": "tool_name arg1 arg2\n", ...}],
         "is_task_complete": false}

    Returns a list of {"name": ..., "arguments": ...} dicts, or None if
    the format doesn't match.
    """
    if not isinstance(obj, dict) or "commands" not in obj:
        return None
    commands = obj.get("commands", [])
    if not isinstance(commands, list) or not commands:
        return None
    results = []
    for cmd in commands:
        keystrokes = cmd.get("keystrokes", "").strip()
        if not keystrokes:
            continue
        # keystrokes is "tool_name [args...]\n" — split into name and the rest
        parts = keystrokes.split(None, 1)
        tool_name = parts[0]
        if tool_name not in tool_registry.tools:
            continue
        # If there's text after the tool name, treat it as a single positional
        # argument (the tool schema decides how to interpret it).
        arguments = {}
        if len(parts) > 1:
            arguments = {"input": parts[1]}
        results.append({
            "name": tool_name,
            "arguments": arguments,
            "_timeout_sec": cmd.get("timeout_sec"),
            "_is_blocking": cmd.get("is_blocking", True),
        })
    return results if results else None


def _parse_tool_call_from_content(content: str, tool_registry) -> Optional[dict | list[dict]]:
    """Detect a raw JSON tool call in message content.

    Some models return tool calls as plain JSON in the response body instead of
    using the structured tool_calls field.  This function tries to parse the
    content and, if it looks like a valid tool call for a known tool, returns
    a dict with 'name' and 'arguments' (or a list of such dicts for the
    commands format).
    """
    if not content or not tool_registry:
        return None
    # Strip thinking tags if present
    text = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
    # Try to find a JSON object in the text
    start = text.find("{")
    if start == -1:
        return None
    # Find the matching closing brace, respecting JSON string literals
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        # JSON may be truncated (e.g. max_tokens cut it off).
        # Try regex fallback to extract tool name and arguments.
        return _parse_truncated_tool_call(text[start:], tool_registry)
    try:
        obj = json.loads(text[start:end])
    except json.JSONDecodeError:
        return _parse_truncated_tool_call(text[start:end], tool_registry)
    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
        if obj["name"] not in tool_registry.tools:
            return None
        return obj
    # Try commands-style format: {"commands": [{"keystrokes": "tool\n", ...}]}
    commands_result = _parse_commands_format(obj, tool_registry)
    if commands_result:
        return commands_result
    return None


def _parse_truncated_tool_call(text: str, tool_registry) -> Optional[dict]:
    """Attempt to extract a tool call from truncated/malformed JSON.

    When the model's response is cut off (e.g. by max_tokens), the JSON may be
    incomplete.  This function uses regex to extract the tool name and any
    parseable arguments from the partial JSON.
    """
    # Extract the tool name
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if not name_match:
        return None
    tool_name = name_match.group(1)
    if tool_name not in tool_registry.tools:
        return None
    # Try to extract arguments object - find where "arguments" value starts
    args_match = re.search(r'"arguments"\s*:\s*\{', text)
    if not args_match:
        return {"name": tool_name, "arguments": {}}
    args_start = args_match.end() - 1  # include the opening brace
    # Try progressively larger substrings, closing any open braces
    # First try parsing as-is with closing braces appended
    args_text = text[args_start:]
    # Count unclosed braces (string-aware)
    depth = 0
    in_str = False
    esc = False
    last_valid = -1
    for i, ch in enumerate(args_text):
        if esc:
            esc = False
            continue
        if ch == '\\' and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                last_valid = i + 1
                break
    if last_valid > 0:
        try:
            args = json.loads(args_text[:last_valid])
            return {"name": tool_name, "arguments": args}
        except json.JSONDecodeError:
            pass
    # Arguments JSON is truncated - return with empty args so the tool can be
    # re-invoked by the model on the next iteration
    return {"name": tool_name, "arguments": {}}


def _looks_like_raw_tool_call(content: str) -> bool:
    """Check if content looks like a raw tool-call JSON that wasn't parsed.

    Returns True if the text contains patterns like {"name": "...", "arguments": ...}
    that indicate the model emitted a tool call as plain text.
    """
    if not content:
        return False
    text = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
    # Quick heuristic: standard format or commands format
    has_standard = bool(re.search(r'"name"\s*:\s*"[^"]+"', text) and re.search(r'"arguments"\s*:', text))
    has_commands = bool(re.search(r'"commands"\s*:\s*\[', text) and re.search(r'"keystrokes"\s*:', text))
    return has_standard or has_commands


def _resolve_sandbox_download_locally(args: dict, data_path: str) -> str | None:
    """Handle sandbox_download_file locally when the file already exists on the host.

    The sandbox container may volume-mount data_path as /workspace.  If so,
    files written there already exist on the host and we can skip the remote
    server call entirely.  Returns a JSON result string on success, or None
    to fall through to the remote server.
    """
    src_path = args.get("path", "")
    if not src_path:
        return None

    abs_data = os.path.abspath(data_path)

    # Resolve to a relative path under data_path
    if src_path.startswith("/workspace/"):
        relative = src_path[len("/workspace/"):]
    elif src_path.startswith("/workspace"):
        relative = src_path[len("/workspace"):]
    elif os.path.abspath(src_path).startswith(abs_data):
        relative = os.path.relpath(os.path.abspath(src_path), abs_data)
    else:
        return None

    if not relative or relative == ".":
        return None

    local_path = os.path.join(data_path, relative)
    if not os.path.exists(local_path):
        return None  # not on host — must go through remote server

    try:
        size_bytes = os.path.getsize(local_path)
    except OSError:
        size_bytes = None
    return json.dumps({
        "status": "ok",
        "filename": os.path.basename(relative),
        "dest": local_path,
        "size_bytes": size_bytes,
    }, indent=2)


def _extract_base64_file(tool_response: str, data_path: str) -> tuple[str, str | None, str | None]:
    """Detect base64-encoded file data in a tool response and save it to disk.

    If the response is JSON containing a 'file_data_base64' field, decode it,
    write the file to data_path, and return a tuple:
      (cleaned_json_str, image_base64_or_None, mime_type_or_None)

    When the file is an image (mime_type starts with 'image/'), the base64 data
    and mime_type are returned so callers can inject the image into the
    conversation for VLM processing.  Otherwise return
    (original_response, None, None).
    """
    try:
        data = json.loads(tool_response)
    except (json.JSONDecodeError, TypeError):
        return tool_response, None, None

    if not isinstance(data, dict) or "file_data_base64" not in data:
        return tool_response, None, None

    file_data_b64 = data.pop("file_data_base64")
    # Tools are inconsistent about the key: vlm_web uses 'file_name',
    # bash send_file uses 'filename'.
    file_name = data.get("file_name") or data.get("filename")
    mime_type = data.get("mime_type") \
        or (mimetypes.guess_type(file_name)[0] if file_name else None) \
        or "application/octet-stream"

    if not file_name:
        # Pick a sensible extension from the mime type
        _ext_map = {
            "image/jpeg": ".jpg", "image/png": ".png",
            "image/gif": ".gif", "image/webp": ".webp",
        }
        ext = _ext_map.get(mime_type, ".bin")
        file_name = f"{uuid.uuid4()}{ext}"
    safe_name = os.path.basename(file_name)
    filepath = os.path.join(data_path, safe_name)
    os.makedirs(data_path, exist_ok=True)

    file_bytes = base64.b64decode(file_data_b64)
    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(file_bytes)

    data["saved_path"] = filepath
    data["download_url"] = f"/uploads/{safe_name}"
    data["file_size_bytes"] = len(file_bytes)

    # Return image base64 + mime for VLM injection when the file is an image
    image_b64 = file_data_b64 if mime_type.startswith("image/") else None
    return json.dumps(data), image_b64, mime_type if image_b64 else None


# Tool results kept at full length; older ones are trimmed to their opening.
TOOL_RESULT_KEEP_FULL = 3
# Enough for the head of a search page to be evidence rather than just a table
# of contents: a chunk is DEFAULT_CHUNK_SIZE (1600) characters, so this holds
# the top three ranked passages once JSON escaping is paid for. At 1500 the
# surviving head of a local_search result was its document summaries alone —
# every matched passage fell outside it, leaving the model a list of titles and
# no quotes, which is the one thing worse than dropping the result entirely.
# Still trims a maximum-size (MAX_TOOL_RESPONSE) result to ~37%.
TOOL_RESULT_DECAY_CHARS = 6000
_DECAY_MARKER = "… [trimmed: older tool result, call the tool again for the rest]"


def _decay_old_tool_results(messages: list,
                            keep_full: int = TOOL_RESULT_KEEP_FULL,
                            head_chars: int = TOOL_RESULT_DECAY_CHARS) -> None:
    """Trim tool results the conversation has moved past.

    Each result may be up to MAX_TOOL_RESPONSE characters, and every one of
    them is re-sent on every later turn: a research loop that opens six
    documents pays for all six on each subsequent request, so the prompt grows
    quadratically in the number of tool calls while the answer is still being
    assembled.

    Only the most recent results are working memory. Older ones are cut to
    their opening, which is enough to remember what was consulted and what it
    was about, and the marker tells the model the rest is retrievable rather
    than gone.

    This is deliberately cheaper than compaction, which only fires near the
    context ceiling and costs a whole extra LLM call when it does. Trimming
    here is meant to keep the conversation from reaching that ceiling at all.
    """
    tool_indices = [
        i for i, msg in enumerate(messages)
        if isinstance(msg, dict) and msg.get("role") == "tool"
        and isinstance(msg.get("content"), str)
    ]
    for i in tool_indices[:-keep_full] if keep_full else tool_indices:
        content = messages[i]["content"]
        if len(content) <= head_chars or content.endswith(_DECAY_MARKER):
            continue
        messages[i] = {**messages[i],
                       "content": content[:head_chars].rstrip() + "\n\n" + _DECAY_MARKER}


def _strip_old_images(messages: list) -> None:
    """Replace base64 image payloads in stale tool messages with a short placeholder.

    The image is kept intact in the most-recently-added image-bearing tool message
    so the model sees it once, then stripped from all older messages to avoid
    re-sending large base64 blobs on every subsequent turn.
    """
    last_image_idx = -1
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool" and isinstance(msg.get("content"), list):
            if any(part.get("type") == "image_url" for part in msg["content"]):
                last_image_idx = i

    if last_image_idx == -1:
        return

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if i == last_image_idx:
            continue  # keep the latest image intact
        if msg.get("role") == "tool" and isinstance(msg.get("content"), list):
            if any(part.get("type") == "image_url" for part in msg["content"]):
                text = next((p["text"] for p in msg["content"] if p.get("type") == "text"), "")
                msg["content"] = text + "\n[image omitted — already analyzed]"


async def _execute_tool(function_name: str, function_arguments: dict,
                        tool_call_id: str, tool_registry, timeout, data_path,
                        chat_ui, verbose, messages: list,
                        tool_call_history: list,
                        max_repeated: int,
                        is_structured: bool = False,
                        session_id: str = "",
                        tool_log: list | None = None) -> Optional[str]:
    """Execute a single tool call and append the result to messages.

    Returns a bail-out message string if repeated-call limit is hit,
    otherwise returns None (caller should continue).

    ``tool_log``, when given, collects one record per call describing how it
    went.  Every path out of this function that answers the call records one,
    including the failures — a tool that errors is the case worth keeping.
    """
    _t0 = time.monotonic()

    def _log(ok: bool, result_chars: int) -> None:
        if tool_log is None:
            return
        tool_log.append(describe_tool_call(
            function_name, function_arguments, ok=ok,
            ms=int((time.monotonic() - _t0) * 1000),
            result_chars=result_chars,
            redact=redact_tool_args(),
        ))

    # Inject session_id / data_path into tool calls whose schema declares
    # these parameters, so callers (e.g. sandbox MCP servers) receive them
    # automatically without hardcoding tool names. These are trust boundaries:
    # servers scope filesystem access to the session's data_path, so the
    # harness value must always win over anything the model put in the call.
    if session_id and tool_registry.tool_accepts_param(function_name, "session_id"):
        function_arguments["session_id"] = session_id
    if data_path and tool_registry.tool_accepts_param(function_name, "data_path"):
        function_arguments["data_path"] = data_path
    # Intercept sandbox_download_file for /workspace/ paths.  The container
    # volume-mounts data_path as /workspace, so those files already exist on
    # the host — no need to call the remote server (which may be containerized
    # and unable to write to host paths).
    if function_name == "sandbox_download_file" and data_path:
        result = _resolve_sandbox_download_locally(function_arguments, data_path)
        if result is not None:
            if chat_ui:
                chat_ui.add_tool_call(function_name, function_arguments)
                chat_ui.show_tool_start(function_name, function_arguments)
            tool_message = {'role': 'tool', 'content': result, 'name': function_name,
                            'parameters': function_arguments, "tool_call_id": tool_call_id}
            messages.append(tool_message)
            if chat_ui:
                chat_ui.add_tool_result(function_name, result)
                if is_structured:
                    chat_ui.show_tool_done(function_name, result)
            _log(True, len(result))
            return None
    if chat_ui:
        chat_ui.add_tool_call(function_name, function_arguments)
        chat_ui.show_tool_start(function_name, function_arguments)
        chat_ui.start_tool_spinner(function_name, function_arguments)
    elif verbose:
        print(f"{function_name}({function_arguments})")

    # A call whose required arguments are blank is refused here rather than
    # dispatched.  Sent on, it reaches a server that runs nothing and returns
    # nothing, and the model — with an empty result and no memory of why —
    # reports the emptiness back as the answer ("the command ran, `ready` was
    # printed with no errors"), ending the task on a non-answer.  An explicit
    # error names the missing arguments, which the model can act on.
    #
    # getattr, not a direct call: the registry is duck-typed at this boundary
    # and test doubles implement only what they exercise.
    _blank_args = []
    if function_name in tool_registry.tools:
        _blank = getattr(tool_registry, "blank_required_args", None)
        if callable(_blank):
            try:
                _result = _blank(function_name, function_arguments)
            except Exception:  # a malformed schema must not break dispatch
                _result = None
            # The return value is checked, not trusted: on a MagicMock registry
            # getattr yields a callable whose result is a truthy Mock, and taking
            # that for a list of blank arguments would refuse every call.
            if isinstance(_result, (list, tuple)):
                _blank_args = [str(n) for n in _result]

    # Then the declared schema.  A blank argument is one specific mistake; this
    # catches the rest — a string where a number was asked for, a value outside
    # an enum — which would otherwise reach the server and come back as a
    # traceback for the model to interpret.  Same duck-typed posture as above,
    # and same rule: anything unexpected means dispatch, never refuse.
    _schema_problems: list[str] = []
    _coercions: list[str] = []
    if not _blank_args and function_name in tool_registry.tools:
        _schema_of = getattr(tool_registry, "parameters_schema", None)
        if callable(_schema_of):
            try:
                _schema = _schema_of(function_name)
                if isinstance(_schema, dict) and _schema:
                    # Coerce first: a model that sends "3" for an integer has
                    # made an unambiguous mistake, and refusing it costs a
                    # round trip to fix something we can fix here.
                    _coerced, _coercions = coerce_arguments(_schema, function_arguments)
                    if _coercions:
                        # In place — the injection above sets the precedent, and
                        # the tool message records this same dict.
                        function_arguments.clear()
                        function_arguments.update(_coerced)
                    _schema_problems = validate_arguments(_schema, function_arguments)
            except Exception:  # a malformed schema must not break dispatch
                _schema_problems, _coercions = [], []
    if _coercions:
        logger.debug("%s: coerced %s", function_name, "; ".join(_coercions))

    if function_name not in tool_registry.tools:
        tool_message = {'role': 'tool', 'content': f'Error: tool {function_name} not found',
                        'name': function_name, 'parameters': function_arguments,
                        "tool_call_id": tool_call_id}
        messages.append(tool_message)
        _log(False, 0)
    elif _blank_args:
        _names = ", ".join(_blank_args)
        _error = (f"Error: {function_name} was called with no value for: {_names}. "
                  "The call was not run. Supply a real value for each of them and "
                  "call the tool again, or use a different tool — do not repeat "
                  "this call unchanged and do not report this error as the answer.")
        _log_to_ui_or_verbose(
            f"{function_name} called with blank required argument(s): {_names}",
            chat_ui, verbose, level="warning",
        )
        tool_message = {'role': 'tool', 'content': _error, 'name': function_name,
                        'parameters': function_arguments, "tool_call_id": tool_call_id}
        messages.append(tool_message)
        if chat_ui:
            chat_ui.add_tool_result(function_name, _error)
            if is_structured:
                chat_ui.stop_tool_spinner()
                chat_ui.show_tool_done(function_name, _error, success=False)
        _log(False, 0)
    elif _schema_problems:
        _detail = "; ".join(_schema_problems)
        _error = (f"Error: {function_name} was called with arguments that do not "
                  f"match its schema: {_detail}. The call was not run. Fix the "
                  "arguments and call the tool again, or use a different tool — "
                  "do not repeat this call unchanged and do not report this "
                  "error as the answer.")
        _log_to_ui_or_verbose(
            f"{function_name} called with invalid arguments: {_detail}",
            chat_ui, verbose, level="warning",
        )
        tool_message = {'role': 'tool', 'content': _error, 'name': function_name,
                        'parameters': function_arguments, "tool_call_id": tool_call_id}
        messages.append(tool_message)
        if chat_ui:
            chat_ui.add_tool_result(function_name, _error)
            if is_structured:
                chat_ui.stop_tool_spinner()
                chat_ui.show_tool_done(function_name, _error, success=False)
        _log(False, 0)
    else:
        tool_handler = tool_registry[function_name]
        _timed_out = False
        try:
            try:
                # Build a log handler to forward MCP notifications/message
                # to the UI in real-time (e.g. sandbox stdout/stderr).
                _log_handler = None
                if chat_ui and hasattr(chat_ui, 'tool_log'):
                    async def _log_handler(msg):
                        chat_ui.tool_log(function_name, msg.data, level=msg.level)
                tool_task = asyncio.ensure_future(tool_handler(log_handler=_log_handler, **function_arguments))

                async def _heartbeat(interval=10):
                    """Send periodic progress events to keep SSE alive."""
                    elapsed = 0
                    while True:
                        await asyncio.sleep(interval)
                        elapsed += interval
                        if chat_ui:
                            chat_ui.tool_progress(function_name, elapsed)

                heartbeat_task = asyncio.ensure_future(_heartbeat())
                try:
                    tool_response = await asyncio.wait_for(tool_task, timeout=timeout)
                finally:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
            except asyncio.TimeoutError:
                _timed_out = True
                tool_response = (f"- tool call timed out after {timeout} seconds. "
                                 "Tool might have succeeded but no response was received. "
                                 "Check expected output.")
                _log_to_ui_or_verbose(f"{function_name} timed out after {timeout}s", chat_ui, verbose, level="warning")
            if is_structured and chat_ui:
                chat_ui.stop_tool_spinner()
            tool_response = "" if tool_response is None else str(tool_response)
            _vision_b64, _vision_mime = None, None
            if data_path and "file_data_base64" in tool_response:
                tool_response, _vision_b64, _vision_mime = _extract_base64_file(tool_response, data_path)
            tool_response = _truncate_tool_response(tool_response)
            if _vision_b64:
                tool_content = [
                    {"type": "text", "text": tool_response},
                    {"type": "image_url", "image_url": {"url": f"data:{_vision_mime};base64,{_vision_b64}"}},
                ]
            else:
                tool_content = tool_response
            tool_message = {'role': 'tool', 'content': tool_content, 'name': function_name,
                            'parameters': function_arguments, "tool_call_id": tool_call_id}
            messages.append(tool_message)
            if chat_ui:
                chat_ui.add_tool_result(function_name, tool_response)
                if is_structured:
                    chat_ui.show_tool_done(function_name, tool_response)
            elif verbose:
                truncated = tool_response[:500] + "..." if len(tool_response) > 500 else tool_response
                print(f"{function_name}({function_arguments}) returned: {truncated}")
            # A timeout answered the call — with a message saying the tool may
            # well have succeeded — but it is not a call that went well.
            _log(not _timed_out, len(tool_response))
        except Exception as e:
            if chat_ui:
                if is_structured:
                    chat_ui.stop_tool_spinner()
                    chat_ui.show_tool_done(function_name, str(e), success=False)
            _log_to_ui_or_verbose(f"{function_name} error: {e}", chat_ui, verbose, level="error")
            tool_message = {'role': 'tool', 'content': f'Error: {e}', 'name': function_name,
                            'parameters': function_arguments, "tool_call_id": tool_call_id}
            messages.append(tool_message)
            _log(False, 0)

    # Check for repeated tool calls
    call_key = (function_name, json.dumps(function_arguments, sort_keys=True))
    tool_call_history.append(call_key)
    if tool_call_history.count(call_key) >= max_repeated:
        msg = f"I am sorry 😊. Could you try to rephrase or provide additional details?"
        _log_to_ui_or_verbose(f"Repeated tool call detected: {function_name} called {tool_call_history.count(call_key)} times with same args", chat_ui, verbose, level="warning")
        return msg
    return None


def _load_images(images: List[str] | str | None, chat_ui, verbose: bool) -> list[str]:
    """Read image files from disk and return their base64-encoded bytes."""
    images_bytes: list[str] = []
    if isinstance(images, list):
        for image_path in images:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as image_file:
                    images_bytes.append(base64.b64encode(image_file.read()).decode('utf-8'))
            else:
                _log_to_ui_or_verbose(f"Image file {image_path} not found, proceeding without this image.", chat_ui, verbose, level="warning")
    elif isinstance(images, str):
        image_path = images
        if os.path.exists(image_path):
            with open(image_path, 'rb') as image_file:
                images_bytes = [base64.b64encode(image_file.read()).decode('utf-8')]
        else:
            _log_to_ui_or_verbose(f"Image file {image_path} not found, proceeding without this image.", chat_ui, verbose, level="warning")
    return images_bytes


# Replayed history: exchanges kept verbatim, and the size an older answer is
# cut down to.  A research answer runs well past a thousand tokens, and every
# prior one is re-sent on every turn of every later task — ten of them is a
# five-figure token bill paid before the model reads the current question.
# What an older exchange is actually for is the thread of the conversation:
# what was asked, and roughly what came back.  Its opening carries that; the
# body and the references list do not, and they are the bulk of it.
HISTORY_KEEP_FULL = 3
HISTORY_DECAY_CHARS = 1200


def _trim_history(session_history: list,
                  keep_full: int = HISTORY_KEEP_FULL,
                  head_chars: int = HISTORY_DECAY_CHARS) -> list:
    """Replayed history with older answers cut to their opening.

    Questions are left whole — they are one line each, and they are what makes
    a follow-up intelligible.  Only answers are trimmed, and only those older
    than the last ``keep_full`` exchanges.
    """
    if len(session_history) <= keep_full:
        return session_history
    trimmed = []
    for entry in session_history[:-keep_full] if keep_full else session_history:
        response = entry.get("response") or ""
        if len(response) > head_chars:
            response = (response[:head_chars]
                        + f"\n... [earlier answer, {len(response) - head_chars} chars trimmed] ...")
        trimmed.append({**entry, "response": response})
    return trimmed + (session_history[-keep_full:] if keep_full else [])


def _build_messages(instruction: str, images_bytes: list[str],
                    prompt_intro: str, session_history: list | None,
                    memories: Any, system_rules: str = "") -> list[dict]:
    """Assemble the initial message list for the API call.

    Includes the system message, session history, the current user instruction,
    and the empty tool sentinel when appropriate.

    ``system_rules`` is the half of the instruction that does not vary between
    requests (see ``split_instruction``).  It belongs in the system message and
    nowhere else: the system message is the only one that sits ahead of the
    session history, so it is the only place where the same bytes appear at the
    same offset on every request and a prefix-caching server can skip
    prefilling them.  Carried in the trailing user message instead — where it
    used to live — it shifted by a turn's worth of history each time and was
    re-prefilled on every request of every session.
    """
    if images_bytes:
        system_content = (
            f"{prompt_intro} "
            "You are an expert vision-language assistant. Your task is to analyze images with high precision, "
            "reasoning step-by-step about visual elements and their spatial relationships (e.g., coordinates, "
            "relative positions like left/right/center). Always verify visual evidence before concluding. "
            "If a task requires external data, calculation, or specific actions beyond visual description, "
            "use the provided tools. Be concise, objective, and format your tool calls strictly according to schema."
        )
    else:
        system_content = prompt_intro
    if system_rules:
        system_content = f"{system_content}\n{system_rules}"
    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Inject session history BEFORE the current instruction so the model
    # sees prior context first and treats the latest user message as the
    # one to respond to.
    if session_history:
        for entry in _trim_history(session_history):
            messages.append({"role": "user", "content": entry["task"]})
            messages.append({"role": "assistant", "content": entry["response"]})

    # Current instruction goes last so the model responds to it
    if images_bytes:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images_bytes[0]}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": instruction})

    # Note: previously an empty tool sentinel message was appended here when
    # there was no session history or memories.  This violated the OpenAI API
    # spec (tool messages must follow an assistant message with tool_calls) and
    # caused some vLLM versions to reject the request outright.

    return messages


def _adapt_ollama_tool_calls(tool_calls) -> list:
    """Convert Ollama tool_calls (dict arguments) to OpenAI-compatible SimpleNamespace objects.

    Ollama's function.arguments is a dict; _handle_structured_tool_calls expects a JSON string.
    """
    adapted = []
    for tc in tool_calls:
        adapted.append(
            types.SimpleNamespace(
                id=f"call_{uuid.uuid4().hex[:24]}",
                function=types.SimpleNamespace(
                    name=tc.function.name,
                    arguments=json.dumps(dict(tc.function.arguments)),
                ),
            )
        )
    return adapted


async def _process_streaming_response(
    chat_completion, safety_queue: asyncio.Queue,
    chat_ui, think: bool, on_first_token=None,
) -> tuple[str, str, dict, bool, Any, str | None] | None:
    """Consume a streaming chat completion and return accumulated results.

    Returns (full_content, full_reasoning, full_tool_calls_dict, ui_was_streaming, usage, finish_reason)
    or None if the safety queue fired mid-stream.  ``usage`` is the
    CompletionUsage object from the final chunk (requires stream_options
    include_usage=True), or None if not available.  ``finish_reason`` is the
    stop reason from the final chunk (e.g. "stop", "tool_calls", "length").

    ``on_first_token`` fires once, on the first chunk carrying generated text
    — reasoning, content, or a tool-call delta.  That instant separates the
    prefill from the decode, which is the split telemetry needs and which the
    chunk loop is the only place that can see.
    """
    full_content = ""
    full_reasoning = ""
    full_tool_calls: dict = {}  # index -> {id, name, arguments}
    ui_streaming = False
    usage = None
    finish_reason: str | None = None
    in_think = think  # True if we expect <think>...</think> in delta.content

    async for chunk in chat_completion:
        if not safety_queue.empty():
            if ui_streaming and chat_ui:
                chat_ui.stream_end()
            return None
        # Capture usage from the final chunk (stream_options include_usage=True).
        # This must be read before stream_end() so set_context_usage can fire first.
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        # Capture finish_reason when it arrives (usually the final chunk)
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason
        delta = choice.delta

        if on_first_token and (delta.tool_calls or delta.content
                               or getattr(delta, 'reasoning_content', None)):
            on_first_token()
            on_first_token = None

        # Accumulate structured tool-call deltas
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in full_tool_calls:
                    full_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.id:
                    full_tool_calls[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        full_tool_calls[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        full_tool_calls[idx]["arguments"] += tc.function.arguments

        # vLLM/OpenAI reasoning_content: thinking tokens in a dedicated field
        reasoning_tok = getattr(delta, 'reasoning_content', None)
        if reasoning_tok and not full_tool_calls:
            full_reasoning += reasoning_tok
            if chat_ui:
                if not ui_streaming:
                    chat_ui.stream_start()
                    ui_streaming = True
                chat_ui.stream_think_token(reasoning_tok)

        # Stream content (answer) tokens to UI
        if delta.content:
            token = delta.content
            full_content += token
            if chat_ui and not full_tool_calls:
                if in_think:
                    # Three cases:
                    # 1. vLLM sends thinking via reasoning_content -> no <think> in content
                    # 2. Model chose not to think -> no <think> in content
                    # 3. Model embeds <think>...</think> inside content
                    if "<think>" not in full_content:
                        # Cases 1 & 2 -- stream answer directly
                        in_think = False
                        if not ui_streaming:
                            chat_ui.stream_start()
                            ui_streaming = True
                        chat_ui.stream_token(token)
                    elif "</think>" in full_content:
                        # Case 3 end: think block closed, stream the answer part
                        in_think = False
                        post_think = full_content.split("</think>", 1)[1]
                        if post_think:
                            if not ui_streaming:
                                chat_ui.stream_start()
                                ui_streaming = True
                            chat_ui.stream_token(post_think)
                    else:
                        # Case 3 mid: inside inline <think> block
                        if not ui_streaming:
                            chat_ui.stream_start()
                            ui_streaming = True
                        chat_ui.stream_think_token(token.replace("<think>", ""))
                else:
                    if not ui_streaming:
                        chat_ui.stream_start()
                        ui_streaming = True
                    chat_ui.stream_token(token)

    return full_content, full_reasoning, full_tool_calls, ui_streaming, usage, finish_reason


async def _ollama_process_streaming_response(
    chat_completion,
    safety_queue: asyncio.Queue,
    chat_ui,
    think: bool,
    on_first_token=None,
) -> tuple[str, str, list | None, bool, int, str | None] | None:
    """Consume an Ollama streaming chat completion and return accumulated results.

    Returns (full_content, full_thinking, tool_calls_or_None, ui_was_streaming,
    prompt_eval_count, done_reason, eval_count) or None if the safety queue
    fired mid-stream.
    ``done_reason`` is Ollama's stop reason from the final chunk (e.g. "stop",
    "length"); it lets the caller detect token-budget truncation and resume.

    Ollama streaming differences from OpenAI:
    - Chunks have .message.content (str|None), no .choices
    - Tool calls arrive complete in the final chunk, not as deltas
    - No .usage field in streaming chunks
    """
    full_content = ""
    full_thinking = ""
    tool_calls = None
    ui_streaming = False
    prompt_eval_count = 0
    eval_count = 0
    done_reason = None

    try:
        async for chunk in chat_completion:
            if not safety_queue.empty():
                if ui_streaming and chat_ui:
                    chat_ui.stream_end()
                return None

            # Capture the stop reason from the final chunk (done=True).
            dr = getattr(chunk, "done_reason", None)
            if dr:
                done_reason = dr

            if on_first_token and (chunk.message.tool_calls
                                   or chunk.message.content
                                   or getattr(chunk.message, "thinking", None)):
                on_first_token()
                on_first_token = None

            # Tool calls arrive complete in the final chunk
            if chunk.message.tool_calls:
                tool_calls = chunk.message.tool_calls

            # Capture token counts from final chunk. prompt_eval_count drives
            # context tracking; eval_count is what the turn actually generated,
            # and without it every streamed Ollama run reports zero output
            # tokens — the non-streaming path below has always read it.
            pec = getattr(chunk, "prompt_eval_count", None)
            if pec:
                prompt_eval_count = pec
            ec = getattr(chunk, "eval_count", None)
            if ec:
                eval_count = ec

            # Thinking tokens (Ollama native think support)
            thinking_tok = getattr(chunk.message, "thinking", None)
            if thinking_tok and not tool_calls:
                full_thinking += thinking_tok
                if chat_ui:
                    if not ui_streaming:
                        chat_ui.stream_start()
                        ui_streaming = True
                    chat_ui.stream_think_token(thinking_tok)

            # Content tokens
            content_tok = chunk.message.content
            if content_tok and not tool_calls:
                full_content += content_tok
                if chat_ui:
                    if not ui_streaming:
                        chat_ui.stream_start()
                        ui_streaming = True
                    chat_ui.stream_token(content_tok)
    except Exception as e:  # noqa: BLE001 — httpx.RemoteProtocolError or similar mid-stream disconnect
        logging.getLogger(__name__).warning("Ollama stream interrupted: %s", e)

    return (full_content, full_thinking, tool_calls, ui_streaming,
            prompt_eval_count, done_reason, eval_count)


def _prose_alongside_tool_calls(full_content: str) -> str:
    """Return the answer text a model wrote before emitting tool calls.

    Streaming shows this text to the user (it arrives before the tool-call
    deltas), so it must survive into the conversation history — otherwise the
    model has no record of what it already said and follows the tool result
    with a bare "Done.", which is all the caller gets back.
    """
    if not full_content:
        return ""
    text = full_content.split("</think>", 1)[1] if "</think>" in full_content else full_content
    if "<think>" in text:  # unterminated think block — nothing was said out loud
        text = text.split("<think>", 1)[0]
    return text.strip()


def _unify_streaming_result(
    full_content: str, full_tool_calls: dict,
) -> tuple[str | None, list | None, dict]:
    """Convert accumulated streaming data into unified content/tool_calls/history variables."""
    if full_tool_calls:
        tool_call_objs = [
            types.SimpleNamespace(
                id=v["id"],
                function=types.SimpleNamespace(name=v["name"], arguments=v["arguments"])
            )
            for v in full_tool_calls.values()
        ]
        message_for_history = {
            "role": "assistant",
            "content": _prose_alongside_tool_calls(full_content) or None,
            "tool_calls": [
                {"id": v["id"], "type": "function",
                 "function": {"name": v["name"], "arguments": v["arguments"]}}
                for v in full_tool_calls.values()
            ]
        }
        return None, tool_call_objs, message_for_history
    return full_content, None, {"role": "assistant", "content": full_content}


_PLANNING_PREFIXES = (
    "let me ", "i will ", "i'll ", "i'm going to ", "i am going to ",
    "now i'll ", "now i will ", "first i'll ", "first i will ",
    "next i'll ", "next i will ", "then i'll ", "then i will ",
    "the user wants me to ",
)

# A plan announcement is forward-looking from the outset.  Once this much prose
# has gone by, a "let me ..." sentence is a follow-up step appended to an answer
# that has already been given, and the answer is the part that matters.
_PLANNING_LEAD_MAX_CHARS = 200


def _is_planning_response(content: str) -> bool:
    """Return True if the response looks like a plan announcement rather than a final answer.

    Detects patterns like "Let me create X and then push it" where the model
    states its intent in future tense but stops before executing tool calls.
    Only returns True when tools are available (caller's responsibility).

    A trailing planning sentence does not make the whole response a plan.  Models
    routinely answer at length and then close with "Let me verify that link" ahead
    of one last tool call; treating that as a plan would discard the answer, so the
    planning phrase only counts while it is still near the start of the text.
    """
    if not content:
        return False
    # Strip thinking blocks
    text = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
    lower = text.lower()
    # Check sentence-start planning phrases
    if any(lower.startswith(p) for p in _PLANNING_PREFIXES):
        return True
    # Check mid-sentence planning phrases (after ". " or "\n")
    for p in _PLANNING_PREFIXES:
        for sep in (f". {p}", f"\n{p}"):
            idx = lower.find(sep)
            if idx != -1 and idx <= _PLANNING_LEAD_MAX_CHARS:
                return True
    return False


# Filler a model emits when it treats a continuation prompt as an announcement to
# accept rather than work to resume.  Matched only against short replies, so an
# answer that happens to open with "Understood" still reaches the user intact.
_ACK_PHRASES = (
    "ready for the next message",
    "ready for your next",
    "awaiting further instructions",
    "awaiting your next",
    "context window management acknowledged",
    "acknowledged. ready",
    "understood. continuing",
    "understood, continuing",
    "continuing based on the context summary",
    # Mechanics reported as if they were the answer.  After a long tool run a
    # model sometimes signs off on the plumbing — "Done. Tool called
    # successfully." — which says the call returned, not what it found.  The
    # user asked about the task, never about the tool, so this is filler
    # however the run went.  Anchored on "tool" on purpose: a bare "Task
    # completed successfully" is a real answer for a task that was an action.
    "tool called successfully",
    "tool call succeeded",
    "tool call was successful",
    "tool call completed successfully",
    "tool executed successfully",
    "tool ran successfully",
    "called the tool successfully",
    "tools called successfully",
)
_ACK_MAX_CHARS = 200


def _is_acknowledgment_response(content: str) -> bool:
    """Return True if the response is content-free filler rather than an answer.

    A compaction or continuation prompt asks the model to resume work.  Weaker
    models sometimes reply "Understood. Ready for the next message." and stop,
    which the caller would otherwise hand back as the final answer — ending a
    long task on a non-answer.  Only short replies qualify: real work that opens
    with an acknowledgment goes on to say something.
    """
    if not content:
        return False
    text = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
    if not text or len(text) > _ACK_MAX_CHARS:
        return False
    lower = text.lower()
    return any(p in lower for p in _ACK_PHRASES)


# ── content-free replies, detected structurally ────────────────────────────
#
# _ACK_PHRASES and _META_PHRASES catch filler by recognising its exact wording,
# which means each new phrasing of the same failure needs its own entry — and
# the failure has unlimited phrasings.  "Working directory confirmed: <uuid>"
# matched nothing; neither did "Ready.", nor "Ready. What would you like me to
# do?", nor "The user asked me to call a tool" (the list had "you want me to
# call a tool", one pronoun away).
#
# The test below is about content rather than wording: strip everything the
# prompt itself supplied — the working directory, the date, the session uuid,
# stock status words, stock hand-backs, claims that the plumbing worked — and
# see whether anything is left.  A reply that says nothing the prompt did not
# already say is filler however it is phrased.
#
# Deliberately conservative: a single surviving content word is enough to keep
# a reply.  "Done — 3 files updated." keeps "3", "files", "updated" and lives;
# "Working directory confirmed: 9f1fd947-..." keeps nothing and does not.

# Session metadata the prompt states and a stuck model reads back.
_METADATA_PHRASES = (
    "working directory", "today's date", "todays date", "current date",
    "agent local filesystem", "local filesystem", "sandbox filesystem",
    "staging area for file transfers", "file operations policy",
    "context compacted", "original instruction", "summary of prior work",
    "reference only",
)

# Stock openers that report on the exchange instead of contributing to it.
#
# "done" is deliberately absent.  It reports the work rather than the exchange —
# the same distinction that keeps "Task completed successfully." an answer — and
# a run that finished may legitimately sign off with nothing more than "All
# done!".  Dropping a reply on the strength of that word alone would end such a
# run on silence, which is worse than the filler this catches.
_STATUS_WORDS = frozenset((
    "ready", "ok", "okay", "understood", "acknowledged", "confirmed",
    "confirm", "noted", "sure", "certainly", "affirmative", "yes", "yep",
))

# Stock closings that hand the turn back without saying anything.
_HANDBACK_PHRASES = (
    "what would you like me to do", "what would you like to do",
    "what would you like next", "what would you like me to",
    "how can i help", "how may i help", "anything else",
    "let me know", "please specify", "could you specify", "please provide",
    "awaiting", "standing by", "next instruction", "next message",
    "waiting for", "ready when you are",
)

# Claims that the plumbing worked.  Stripped rather than matched outright, so a
# reply that reports success *and* says what came of it keeps its content:
# "The command ran successfully and printed 42" loses the clause but keeps "42".
_MECHANICS_RE = re.compile(
    r"\b(?:the\s+|all\s+)?(?:tool(?:\s+call)?s?|command|script|code|call)\b"
    r"[^.;\n]{0,40}"
    r"\b(?:succeeded|successful|successfully|completed|complete|executed|"
    r"ran|run|finished|printed|returned|errors?)\b",
    re.I,
)
# The tail of a mechanics claim, which routinely falls outside the window above:
# "...was printed with no errors".  Its own pattern so the claim is stripped
# whole rather than leaving "errors" behind as if it were a finding.
_NO_ERRORS_RE = re.compile(
    r"\b(?:with\s+)?no\s+errors?\b|\bwithout\s+(?:any\s+)?errors?\b"
    r"|\berror[\s-]free\b",
    re.I,
)

# A session uuid and an absolute path are the two shapes of prompt metadata no
# static phrase list can hold: both are minted per session.
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_ABS_PATH_RE = re.compile(r"(?:/[\w.\-]+){2,}/?")

# Words carrying no information wherever they appear.
_FILLER_WORDS = frozenset((
    "the", "a", "an", "and", "or", "but", "so", "then", "now", "is", "are",
    "was", "were", "be", "been", "am", "i", "you", "it", "its", "my", "your",
    "me", "we", "us", "this", "that", "these", "those", "to", "of", "in", "on",
    "at", "for", "with", "from", "as", "by", "have", "has", "had", "do", "did",
    "does", "will", "would", "should", "can", "could", "no", "not", "any",
    "all", "up", "out", "if", "there", "here", "what", "which", "who", "how",
    "user", "assistant", "task", "next", "further", "instructions",
    "instruction", "message", "please", "thanks", "thank",
))

# Long enough for a short paragraph of filler, short enough that real work runs
# past it.  Matches _META_MAX_CHARS in spirit: length is what separates a filler
# reply from an answer that merely opens with filler.
_CONTENT_FREE_MAX_CHARS = 400


def _content_residue(text: str, data_path: str | None = None) -> list[str]:
    """Words left in ``text`` once everything the prompt supplied is removed."""
    lower = text.lower()
    # The session's own directory first: its basename is a uuid the model quotes
    # verbatim, and stripping the path before the generic patterns keeps a real
    # path in an answer ("the bug is in /etc/hosts") from being mistaken for it.
    if data_path:
        for token in (str(data_path).lower(), Path(str(data_path)).name.lower()):
            if token:
                lower = lower.replace(token, " ")
    lower = _UUID_RE.sub(" ", lower)
    lower = _ABS_PATH_RE.sub(" ", lower)
    lower = _MECHANICS_RE.sub(" ", lower)
    lower = _NO_ERRORS_RE.sub(" ", lower)
    for phrase in _METADATA_PHRASES + _HANDBACK_PHRASES:
        lower = lower.replace(phrase, " ")
    return [w for w in re.findall(r"[a-z0-9]+", lower)
            if w not in _FILLER_WORDS and w not in _STATUS_WORDS]


def _is_content_free_response(content: str, data_path: str | None = None) -> bool:
    """Return True if the reply adds nothing the prompt did not already contain.

    The wording-independent counterpart to ``_is_acknowledgment_response`` and
    ``_is_meta_commentary_response``: those two name the filler they know, this
    one recognises filler by the absence of anything else.

    Meaningful only about a reply to a synthetic prompt — see
    ``_is_answering_a_nudge``.  "All done!" is filler as an answer to "resume the
    task" and a fair sign-off after a run that did the work, and nothing in the
    text tells the two apart; what tells them apart is the question.
    """
    if not content:
        return False
    text = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
    if not text or len(text) > _CONTENT_FREE_MAX_CHARS:
        return False
    return not _content_residue(text, data_path)


def _is_answering_a_nudge(messages: list) -> bool:
    """True if the most recent user turn is one the harness wrote, not the user.

    The continuation prompts and the compacted prompt all say the same thing —
    resume the task — so a content-free reply to one of them is a refusal to
    resume.  After a genuine user turn the same reply may be a real answer, so
    the structural test is scoped to this case.
    """
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            return False
        return (content in (_ACK_CONTINUATION_PROMPT, _FINAL_CONTINUATION_PROMPT)
                or content.startswith("[CONTEXT COMPACTED]")
                or "Use this exact JSON format:" in content)
    return False


# Shell tools make a poor example for a stuck model: `bash` is the one call it
# can satisfy without doing anything ("echo ok"), which passes the format check,
# resets the continuation counter, and advances the task not at all.  Lead with
# them only when the task itself reads like shell work.
_SHELL_TOOLS = ("bash", "shell", "run_command")
_FILE_TOOLS = ("write_file", "create_file", "read_file", "list_files")
_SHELL_TASK_HINTS = (
    "bash", "shell", "terminal", "command", "run ", "execute", "script",
    "install", "compile", "build", "git ", "npm", "pip ", "docker", "make ",
)


# A reply that discusses the instructions instead of following them.  Weaker
# models answer a continuation prompt by narrating it back — "I understand you
# want me to call a tool without any text" — which is neither a plan nor an
# acknowledgment, so without this it falls through every guard and is handed to
# the user as the final answer.
_META_PHRASES = (
    "i understand you want me to",
    "i understand that you want me to",
    "you want me to call a tool",
    "you may be testing",
    "testing my ability",
    "let me know what specific",
    "let me know what command",
    "i'll call it immediately",
    "i will call it immediately",
    "without any additional text",
    "without any extra text",
)
# Looser than _ACK_MAX_CHARS: this filler runs to a short paragraph, but real
# work still runs longer than one.
_META_MAX_CHARS = 700


def _is_meta_commentary_response(content: str) -> bool:
    """Return True if the reply is about the prompt rather than about the task.

    Only short replies qualify, on the same reasoning as
    ``_is_acknowledgment_response``: an answer that mentions the instructions in
    passing goes on to say something, and length is what separates the two.
    """
    if not content:
        return False
    text = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
    if not text or len(text) > _META_MAX_CHARS:
        return False
    lower = text.lower()
    return any(p in lower for p in _META_PHRASES)


def _build_tool_example(tool_registry, task: str = "") -> str:
    """Return a filled-in JSON tool-call example using the first available tool's schema.

    Prefers common action tools (write_file, create_file, bash) so the example
    contains real argument names rather than an empty ``{}``.  Shell tools lead
    only for shell-shaped tasks (see ``_SHELL_TASK_HINTS``); otherwise they trail
    the file tools but stay in the list, so a registry holding nothing else still
    produces a usable example.
    """
    lower_task = (task or "").lower()
    if any(hint in lower_task for hint in _SHELL_TASK_HINTS):
        preferred = _SHELL_TOOLS + _FILE_TOOLS
    else:
        preferred = _FILE_TOOLS + _SHELL_TOOLS
    tool_names = sorted(tool_registry.tools) if tool_registry else []
    # Pick preferred first, otherwise fall back to first alphabetically
    chosen = next((t for t in preferred if t in tool_registry.tools), None) if tool_registry else None
    if not chosen:
        chosen = tool_names[0] if tool_names else "bash"

    # Try to pull argument names from the schema
    sample_args: dict = {}
    try:
        items = tool_registry.get_tool_items()
        for item in items:
            fn = item.get("function", {}) if isinstance(item, dict) else {}
            if fn.get("name") == chosen:
                props = fn.get("parameters", {}).get("properties", {})
                for param_name, param_schema in list(props.items())[:2]:
                    ptype = param_schema.get("type", "string")
                    sample_args[param_name] = "<string>" if ptype == "string" else f"<{ptype}>"
                break
    except Exception:
        pass

    # Fall back to a sensible shape for well-known tools
    if not sample_args:
        _defaults = {
            "bash": {"command": "<shell command>"},
            "shell": {"command": "<shell command>"},
            "run_command": {"command": "<shell command>"},
            "write_file": {"path": "<file path>", "content": "<file content>"},
            "create_file": {"path": "<file path>", "content": "<file content>"},
            "read_file": {"path": "<file path>"},
        }
        sample_args = _defaults.get(chosen, {"input": "<value>"})

    return json.dumps({"name": chosen, "arguments": sample_args})


# Commands that report nothing about the world and change nothing in it.  A model
# cornered by tool_choice=required reaches for one of these to satisfy the
# requirement: the call succeeds, the task stands still.
_NOOP_COMMANDS = ("echo", "printf", "true", ":")
# Shell metacharacters that make a command consequential no matter how it starts:
# `echo x > f.py` writes a file, `echo $(git rev-parse HEAD)` reads a repo.
_CONSEQUENTIAL_SHELL_CHARS = (">", "|", "&", ";", "$(", "`", "<")


def _is_noop_tool_call(name: str, arguments: dict) -> bool:
    """Return True if this call cannot advance the task.

    Only shell tools qualify.  Every other tool does something observable by
    definition — a read returns content, a write lands a file — so there is no
    no-op shape to recognise.  A shell call is a no-op only when its whole
    command is one of ``_NOOP_COMMANDS`` used plainly: any redirect, pipe,
    substitution, or chaining means it is doing real work.
    """
    if name not in _SHELL_TOOLS:
        return False
    command = ""
    for key in ("command", "cmd", "script"):
        value = arguments.get(key)
        if isinstance(value, str):
            command = value.strip()
            break
    if not command:
        # A shell tool called with no command at all is the emptiest call there is.
        return True
    if any(c in command for c in _CONSEQUENTIAL_SHELL_CHARS):
        return False
    first = command.split(maxsplit=1)[0]
    return first in _NOOP_COMMANDS


# The task restated in the continuation prompt is trimmed to this much: enough to
# name the work again, short enough not to re-inflate a context the model is
# already struggling in.
_CONTINUATION_TASK_MAX_CHARS = 300


def _build_planning_continuation_prompt(tool_registry, continuation_count: int,
                                        task: str = "") -> str:
    """Build a direct continuation prompt for models stuck in planning mode.

    Includes a concrete JSON tool-call example (with real argument shapes) so the
    raw-tool-call parser can catch it even on models that don't honour
    tool_choice=required (e.g. Ollama).

    The task is restated alongside the example because that is what keeps the
    reply on the work.  A bare format command ("Do not write any text") leaves a
    weak model nothing to act on but the instruction itself, which it answers
    instead of working: "I understand you want me to call a tool without any
    text."  Naming the task gives the next call something to be about.
    """
    tool_names = sorted(tool_registry.tools)[:6] if tool_registry else []
    example = _build_tool_example(tool_registry, task)
    tools_list = ", ".join(tool_names)
    task_text = " ".join((task or "").split())
    if len(task_text) > _CONTINUATION_TASK_MAX_CHARS:
        task_text = task_text[:_CONTINUATION_TASK_MAX_CHARS].rstrip() + "..."
    task_line = f"Task: {task_text}\n" if task_text else ""
    if continuation_count > 1:
        lead = ("Your entire reply must be one JSON object: the tool call that "
                "takes the next step on this task, and nothing else.")
    else:
        lead = ("Take the next step on this task now by calling a tool. Reply "
                "with the tool call only.")
    return (
        f"{task_line}{lead}\nUse this exact JSON format:\n"
        f"{example}\n"
        f"Available tools: {tools_list}"
    )


# Prompt used to resume a final answer that hit the output token limit mid-stream.
_FINAL_CONTINUATION_PROMPT = (
    "Your previous reply was cut off mid-sentence because it reached the output "
    "length limit. Continue from exactly where it stopped. Do not repeat anything "
    "you already wrote, do not restate the question, and do not add any preamble — "
    "just continue the text seamlessly."
)

# Prompt used to push past a content-free acknowledgment.
_ACK_CONTINUATION_PROMPT = (
    "Do not acknowledge or restate the status. Resume the task now: either call "
    "the next tool you need, or give the final answer if the task is already done."
)


def _extract_final_response(content: str | None, full_reasoning: str, full_content: str) -> str:
    """Clean up the final text response, stripping think tags and applying fallbacks."""
    last_response = content or ""
    if "</think>" in last_response:
        last_response = last_response.split("</think>")[1]
    # Fallback: if delta.content was empty but reasoning_content had the answer,
    # surface the reasoning so the user gets a non-empty reply.
    if not last_response or not last_response.strip():
        if full_reasoning and full_reasoning.strip():
            last_response = full_reasoning.strip()
        elif full_content and "<think>" in full_content:
            # Model put entire answer inside inline <think> tags
            think_body = full_content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
            if think_body:
                last_response = think_body
    return last_response


# Prose only counts as the dropped answer if there is enough of it to be one.
_ANSWER_MIN_CHARS = 200
# A post-tool reply is an acknowledgment rather than an answer when the prose
# before it is at least this many times longer.
_ACK_RATIO = 3

# Closings that point at the answer instead of being it. However long these
# run, they are a reference to prose the caller is about to drop, so they can
# never stand in for it.
_ACK_REFERENCE_RE = re.compile(
    r"\b(?:provided|described|detailed|outlined|listed|covered|answered|shown"
    r"|summarized|summarised)\s+(?:in\s+full\s+)?above\b"
    r"|\bas\s+(?:described|detailed|outlined|noted|stated)\s+above\b"
    r"|\bsee\s+(?:the\s+)?(?:answer|response|details|list|breakdown)\s+above\b",
    re.IGNORECASE,
)


def _message_content(message) -> str:
    """Read the content of a history message, dict or SDK object alike."""
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", None) or ""


def _recover_dropped_answer(final: str, prose: str) -> str:
    """Restore answer text the model wrote before its last tool call.

    Models routinely write out the whole answer, call one more tool (save the
    file, verify a link), then close with "Done." — leaving the caller holding
    only the acknowledgment while the answer scrolls away.  When the earlier
    prose is substantial and the closing line is not, return both.

    "Not substantial" is judged against the prose, not against a fixed length:
    a closing that summarizes what it replaces ("the full answer was provided
    above, covering ...") runs to a few hundred characters and would clear any
    absolute cap while still being an acknowledgment.  Such a closing names the
    answer as being elsewhere, so _ACK_REFERENCE_RE recovers the prose whatever
    the lengths work out to.
    """
    prose = (prose or "").strip()
    if len(prose) < _ANSWER_MIN_CHARS:
        return final
    closing = (final or "").strip()
    if not closing:
        return prose
    if (not _ACK_REFERENCE_RE.search(closing)
            and len(closing) * _ACK_RATIO > len(prose)):
        return final
    return f"{prose}\n\n{closing}"


async def _handle_raw_tool_call(
    last_response: str, tool_registry, timeout, data_path,
    chat_ui, verbose: bool, messages: list,
    tool_call_history: list, max_repeated: int,
    session_id: str = "", tool_log: list | None = None,
) -> tuple[bool, str | None]:
    """Handle a raw JSON tool call embedded in model content.

    Returns (should_continue, bail_message).
    If should_continue is True, the caller should loop back for another iteration.
    If bail_message is not None, the caller should return it immediately.
    """
    raw_tool = _parse_tool_call_from_content(last_response, tool_registry)
    if raw_tool:
        # Normalize to a list (commands format returns a list, legacy returns a dict)
        tool_calls = raw_tool if isinstance(raw_tool, list) else [raw_tool]
        messages.append({"role": "assistant", "content": last_response})
        for tc in tool_calls:
            function_name = tc["name"]
            function_arguments = {k: v for k, v in tc.get("arguments", {}).items()}
            synthetic_id = f"call_{uuid.uuid4().hex[:24]}"
            bail = await _execute_tool(
                function_name, function_arguments, synthetic_id,
                tool_registry, timeout, data_path, chat_ui, verbose,
                messages, tool_call_history, max_repeated,
                is_structured=False, session_id=session_id,
                tool_log=tool_log,
            )
            if bail:
                return False, bail
        return True, None  # loop back for the model to generate the final response

    # Guard against returning raw tool-call JSON to the user.
    # If the content looks like a tool call but couldn't be parsed,
    # ask the model to retry without tools.
    if _looks_like_raw_tool_call(last_response):
        _log_to_ui_or_verbose("Model returned unparseable raw tool-call JSON, retrying without tools.", chat_ui, verbose, level="warning")
        messages.append({"role": "assistant", "content": last_response})
        messages.append({"role": "user", "content": "Please provide your answer as plain text, not as a JSON tool call."})
        return True, None

    return False, None


_SAFETY_ABORT = object()  # sentinel distinct from None


def _verify_content(message) -> str:
    """The text of a fact-check reply, wherever the server put it.

    A thinking model's verdict lands in ``content`` on one server and in
    ``reasoning`` or ``reasoning_content`` on the next, and an endpoint that
    splits them can leave ``content`` empty with the JSON sitting in the other
    field.  The same fallback the streaming path already makes for a final
    answer (see _extract_final_response), for the same reason.
    """
    text = getattr(message, "content", None) or ""
    if text.strip():
        return text
    for field in ("reasoning_content", "reasoning", "thinking"):
        alt = getattr(message, field, None)
        if isinstance(alt, str) and alt.strip():
            return alt
    return ""


class _VerifyStopped(Exception):
    """The user stopped the run while the fact-check was still in flight.

    Raised rather than returned so that it unwinds through the verification
    pass the same way a failed call does — the draft the user already read is
    what they keep, and a stop request never costs them an answer.
    """


async def _await_with_safety(awaitable, safety_queue: asyncio.Queue, poll: float = 0.25):
    """Await *awaitable* while polling the safety queue.

    The chat loop only checks the safety queue between streamed chunks, so a
    user stop request goes unnoticed while blocked on the initial API call
    (e.g. during prompt processing or guided-decoding grammar compilation).
    This races the awaitable against the queue: if the queue fires first, the
    in-flight request is cancelled and _SAFETY_ABORT is returned.
    """
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll)
            if done:
                return task.result()
            if not safety_queue.empty():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                return _SAFETY_ABORT
    except asyncio.CancelledError:
        task.cancel()
        raise


# Tools that only read.  A batch of these can run at once: none of them can
# observe another's effect, so the messages they produce are the same whatever
# order the calls finish in.  Everything else — writing a file, running a
# command, driving a sandbox — stays sequential, because a batch of those is a
# script, and a script has an order.
_READ_ONLY_TOOLS = frozenset({
    "local_search", "search_document", "get_document_context",
    "read_file", "grep", "find_files", "search_directory",
    "search", "fetch_content", "extract_tables", "extract_pdf_images",
    "get_weather",
})


def _parse_tool_arguments(tool, verbose: bool) -> dict:
    """Read a structured tool call's arguments, repairing sloppy JSON."""
    try:
        return json.loads(tool.function.arguments)
    except json.JSONDecodeError:
        # Try fixing common issues: single quotes, trailing commas
        fixed = tool.function.arguments.strip()
        # Replace single quotes with double quotes
        fixed = fixed.replace("'", '"')
        # Remove trailing commas before closing braces/brackets
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            if verbose:
                print(f"Failed to parse tool arguments for {tool.function.name}: {e}")
                print(f"Raw arguments: {tool.function.arguments}")
            return {}


async def _execute_tools_in_parallel(
    calls: list, tool_registry, timeout, data_path, chat_ui, verbose: bool,
    messages: list, tool_call_history: list, max_repeated: int,
    safety_queue: asyncio.Queue, session_id: str, tool_log: list | None = None,
) -> str | object | None:
    """Run a batch of read-only tool calls concurrently.

    A model that asks for six documents in one turn was made to wait for the
    sum of six round trips, when the turn is only as useful as its slowest
    call.  Reading documents is the case that matters: the research prompt
    asks for several at once, and they are all network round trips to the same
    MCP server, idle while each other waits.

    Each call writes into its own buffer and the buffers are drained in call
    order, because the API requires one tool message per tool_call id in the
    order the assistant asked for them — completion order is not that order.
    """
    buffers: list = [[] for _ in calls]
    if chat_ui and hasattr(chat_ui, "start_tool_batch"):
        # Arguments included: what five concurrent calls are looking for is
        # more use to someone waiting than the fact that there are five.
        chat_ui.start_tool_batch([(name, args) for name, args, _ in calls])

    async def _run(index: int, name: str, args: dict, call_id: str):
        return await _execute_tool(
            name, args, call_id, tool_registry, timeout, data_path,
            chat_ui, verbose, buffers[index], tool_call_history,
            max_repeated, is_structured=True, session_id=session_id,
            tool_log=tool_log,
        )

    gathered = _await_with_safety(
        asyncio.gather(*(_run(i, name, args, call_id)
                         for i, (name, args, call_id) in enumerate(calls)),
                       return_exceptions=True),
        safety_queue,
    )
    results = await gathered
    if results is _SAFETY_ABORT:
        if chat_ui and hasattr(chat_ui, "end_tool_batch"):
            chat_ui.end_tool_batch()
        return _SAFETY_ABORT

    bail = None
    for (name, args, call_id), buffer, result in zip(calls, buffers, results):
        if not buffer:
            # _execute_tool answers every call, including failures, so an empty
            # buffer means it did not run at all (cancelled, or raised before
            # appending).  The id still needs an answer or the next request is
            # rejected for having a tool_call nothing replied to.
            reason = result if isinstance(result, BaseException) else "no result"
            buffer.append({'role': 'tool', 'content': f'Error: {reason}',
                           'name': name, 'parameters': args,
                           "tool_call_id": call_id})
        messages.extend(buffer)
        if bail is None and isinstance(result, str):
            bail = result
    if chat_ui and hasattr(chat_ui, "end_tool_batch"):
        chat_ui.end_tool_batch()
    return bail


async def _handle_structured_tool_calls(
    tool_calls: list, message_for_history, tool_registry,
    timeout, data_path, chat_ui, verbose: bool,
    messages: list, tool_call_history: list,
    max_repeated: int, safety_queue: asyncio.Queue,
    session_id: str = "", tool_log: list | None = None,
) -> str | object | None:
    """Execute structured tool calls and append results to messages.

    A batch of read-only calls runs concurrently; anything else runs in the
    order the model asked for it.

    Returns:
      - A bail-out message string if a repeated-call limit is hit.
      - _SAFETY_ABORT sentinel if the safety queue fired.
      - None when all tools completed normally.
    """
    messages.append(message_for_history)
    calls = [(tool.function.name, _parse_tool_arguments(tool, verbose), tool.id)
             for tool in tool_calls]

    if len(calls) > 1 and all(name in _READ_ONLY_TOOLS for name, _, _ in calls):
        return await _execute_tools_in_parallel(
            calls, tool_registry, timeout, data_path, chat_ui, verbose,
            messages, tool_call_history, max_repeated, safety_queue, session_id,
            tool_log=tool_log,
        )

    for function_name, function_arguments, call_id in calls:
        await asyncio.sleep(0.1)
        if not safety_queue.empty():
            if verbose:
                print("Safety queue triggered, exiting chat loop.")
            return _SAFETY_ABORT
        bail = await _execute_tool(
            function_name, function_arguments, call_id,
            tool_registry, timeout, data_path, chat_ui, verbose,
            messages, tool_call_history, max_repeated,
            is_structured=True, session_id=session_id,
            tool_log=tool_log,
        )
        if bail:
            return bail
    return None


async def _compact_context(
    messages: list, client, model: str,
    max_tokens: int, chat_ui, verbose: bool,
    is_ollama: bool = False,
    instruction: str = "",
) -> list:
    """Summarize the conversation and return a compacted messages list.

    Keeps the system message, generates a dense LLM summary of all other
    messages, and returns [system_msg, compacted_user_msg] — ending on the user
    turn so the model resumes the task rather than acknowledging the summary.
    When ``instruction`` is given it is restated verbatim in the compacted
    user message, so it survives the lossy summary.

    Callers pass only the volatile half of the instruction — the session
    context and the task.  The standing rules (tool routing, citations,
    sandbox) live in the system message, which compaction preserves untouched,
    so restating them here would duplicate them in every compacted prompt.

    Falls back to the original messages list if the summarization call fails.
    """
    system_msg = (
        messages[0]
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system"
        else None
    )
    messages_to_summarize = messages[1:] if system_msg else messages[:]
    if not messages_to_summarize:
        return messages

    # Build plain-text conversation transcript for the summarization prompt
    parts: list[str] = []
    for msg in messages_to_summarize:
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = str(content)
        name = msg.get("name", "")
        if role == "user":
            parts.append(f"User: {content[:600]}")
        elif role == "assistant":
            tcs = msg.get("tool_calls")
            if tcs:
                tc_names = []
                for tc in tcs:
                    if isinstance(tc, dict):
                        tc_names.append(tc.get("function", {}).get("name", "?"))
                    else:
                        tc_names.append(getattr(getattr(tc, "function", None), "name", "?"))
                parts.append(f"Assistant called tools: {', '.join(tc_names)}")
            elif content:
                parts.append(f"Assistant: {content[:600]}")
        elif role == "tool":
            parts.append(f"Tool({name}): {content[:400]}")

    compaction_prompt = (
        "Summarize the following agent conversation. Include: the original task, "
        "key tool results and findings, decisions made, and what still needs to be "
        "done. Be thorough but concise — this summary replaces the full history.\n\n"
        + "\n".join(parts)
    )
    try:
        if is_ollama:
            resp = await client.chat(
                model=model,
                messages=[{"role": "user", "content": compaction_prompt}],
                options={"num_predict": min(2048, max_tokens)},
                stream=False,
            )
            summary = (resp.message.content or "").strip()
        else:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": compaction_prompt}],
                max_tokens=min(2048, max_tokens),
                stream=False,
            )
            summary = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Context compaction failed: %s", e)
        return messages

    _log_to_ui_or_verbose(
        f"Context compacted: {len(messages_to_summarize)} messages → {len(summary):,} char summary",
        chat_ui, verbose, level="info",
    )
    if chat_ui and hasattr(chat_ui, "show_context_compaction"):
        chat_ui.show_context_compaction(len(messages_to_summarize), len(summary))

    new_messages: list = []
    if system_msg:
        new_messages.append(system_msg)
    # The closing line says explicitly not to acknowledge.  A compacted prompt
    # restates the session context — the working directory among it — and frames
    # it as rules still in effect, which is exactly the cue a weak model answers
    # with "Working directory confirmed: <uuid>" instead of resuming work.
    _resume = ("[Resume the task now. Do not acknowledge this message, do not "
               "restate the context or the summary — either call the next tool "
               "you need, or give the final answer if the task is already done.]")
    if instruction:
        compacted_content = (
            "[CONTEXT COMPACTED]\n"
            "The original task instruction is restated below and all of its "
            "rules remain in effect, followed by a summary of prior work.\n\n"
            "## Original instruction\n"
            + instruction
            + "\n\n## Summary of prior work\n"
            + summary
            + "\n\n" + _resume
        )
    else:
        compacted_content = (
            "[CONTEXT COMPACTED]\nThe following is a summary of prior work:\n\n"
            + summary
            + "\n\n" + _resume
        )
    new_messages.append({
        "role": "user",
        "content": compacted_content,
    })
    # The compacted user message is deliberately last.  Appending a synthetic
    # assistant acknowledgment here would leave the model generating a second
    # consecutive assistant turn with nothing new to respond to — it answers
    # with filler ("Ready for the next message.") and no tool call, which the
    # loop then returns as the final answer, silently ending a long task.
    return new_messages


async def chat(host: str = "http://127.0.0.1:8001/v1",
         host_key: str = "EMPTY",
         model: str = None,
         instruction: str = "Tell me more about yourself.",
         images: List[str]|str = None,
         tool_registry: Optional[Any] = None,
         timeout: int = None,
         stream: bool = False,
         think: bool = False,
         safety_queue: Optional[asyncio.Queue] = None,
         **kwargs) -> Optional[str]:

    tools = _api_tool_payload(tool_registry.get_tool_items()) if tool_registry else []
    chat_ui = kwargs['chat_ui'] if 'chat_ui' in kwargs else None
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    data_path = kwargs.get('data_path', '')
    session_id = kwargs.get('session_id', '')
    max_tokens = kwargs.get('max_tokens', 32768)
    temperature = kwargs.get('temperature', 0.6)
    top_p = kwargs.get('top_p', 0.95)
    top_k = kwargs.get('top_k', 20)
    min_p = kwargs.get('min_p', 0.0)
    presence_penalty = kwargs.get('presence_penalty', 0.0)
    repetition_penalty = kwargs.get('repetition_penalty', 1.0 if think else 1.05)
    memories = kwargs.get('memories', None)
    prompt_intro = kwargs.get('prompt_intro', "I am a helpful AI assistant. My name is OnIt.")
    max_context_tokens: Optional[int] = kwargs.get('max_context_tokens', None)
    num_ctx: Optional[int] = kwargs.get('num_ctx', None)  # Ollama context window override
    # Optional caller-owned dict, filled in turn by turn (see TurnMetrics).
    # Accounting always runs: with no caller sink it fills a throwaway dict, so
    # every recording site is unconditional rather than each one remembering a
    # null guard.  The timers it wraps run either way.
    _metrics_sink = kwargs.get('metrics', None)
    _m = TurnMetrics(_metrics_sink if _metrics_sink is not None else {})
    # Whether thinking, when enabled, is also spent on the turns that only
    # pick a tool.  A thinking model emits its full reasoning before every
    # tool-call JSON, so in a loop of N tool calls the reasoning is paid for N
    # times to produce N pieces of JSON.  Turning this off keeps thinking for
    # the opening turn — where the approach is actually decided — and drops it
    # for the rest of the loop, the final answer included.  That last part is
    # the cost: it is a trade of answer deliberation for latency, which is why
    # it is opt-in.
    think_tool_turns = kwargs.get('think_tool_turns', True)
    # Fact-checking the finished answer (see verify.py).  On by default: the
    # draft still streams at full speed, so what it costs is the pause between
    # the last token and the answer being final, and what it buys is that the
    # figure on screen is the figure the tool returned.  verify_max_tool_turns
    # bounds how far the checker may go looking for evidence the run did not
    # already gather — 0 checks against the transcript alone.
    verify_answers = kwargs.get('verify_answers', True)
    verify_max_tool_turns = max(0, _as_int(kwargs.get('verify_max_tool_turns', 2)))

    images_bytes = _load_images(images, chat_ui, verbose)
    # The standing rules go to the system message, where they stay put across
    # requests; only the session context and the task ride in the user message.
    system_rules, task_instruction = split_instruction(instruction)
    messages = _build_messages(task_instruction, images_bytes, prompt_intro,
                               kwargs.get('session_history', None), memories,
                               system_rules=system_rules)

    api_key = _resolve_api_key(host, host_key)
    # Explicit positive timeout applies to the whole request; -1/None means no
    # overall limit but connect and per-chunk stall timeouts still apply (see
    # _build_client_timeout) so a wedged server can't hang the loop forever.
    is_ollama = _is_ollama_host(host)
    if is_ollama:
        ollama_client = _create_ollama_client(host, api_key, timeout, stream)
        client = None
    else:
        client = AsyncOpenAI(base_url=host, api_key=api_key,
                             timeout=_build_client_timeout(timeout, stream))
        ollama_client = None

    # Resolve model: use explicit name if provided, otherwise auto-detect.
    if not model:
        _MODEL_RETRIES = 3
        for _attempt in range(1, _MODEL_RETRIES + 1):
            try:
                if is_ollama:
                    model = await _ollama_resolve_model_id(ollama_client, host)
                else:
                    model = await _resolve_model_id(client, host)
                break
            except Exception as e:
                _err = f"Failed to resolve model from {host} (attempt {_attempt}/{_MODEL_RETRIES}): {e}"
                logger.error(_err)
                _log_to_ui_or_verbose(_err, chat_ui, verbose, level="error")
                if _attempt < _MODEL_RETRIES:
                    await asyncio.sleep(min(2 ** _attempt, 10))
        if model is None:
            return None

    if chat_ui:
        chat_ui.model_name = model
    _log_to_ui_or_verbose(f"Starting chat with model: {model}", chat_ui, verbose, level="info")

    # Query vLLM for the model's maximum context window if not provided in config.
    # Skip for OpenRouter and Ollama (neither exposes max_model_len via vLLM endpoint).
    if max_context_tokens is None and "openrouter.ai" not in host and not is_ollama:
        max_context_tokens = await _get_model_max_context(host, api_key, model)
        if max_context_tokens:
            _log_to_ui_or_verbose(
                f"Model max context: {max_context_tokens:,} tokens", chat_ui, verbose, level="info"
            )

    # Ollama: resolve the context window (num_ctx).  Ollama defaults num_ctx to
    # 2048, which silently truncates any response longer than ~prompt+2048 tokens
    # regardless of num_predict.  Use the explicit override if given, otherwise the
    # model's native window, capped to a sane ceiling so a huge-context model
    # doesn't try to allocate a giant KV cache and fail to load.
    if is_ollama:
        if num_ctx is None:
            _native_ctx = await _get_ollama_max_context(ollama_client, model)
            # Ceiling keeps auto-sizing safe; raise serving.num_ctx to go higher.
            _ctx_ceiling = max(max_tokens * 2, 262144)
            if _native_ctx:
                num_ctx = min(_native_ctx, _ctx_ceiling)
            else:
                num_ctx = _ctx_ceiling
        _log_to_ui_or_verbose(
            f"Ollama context window (num_ctx): {num_ctx:,} tokens", chat_ui, verbose, level="info"
        )
        # Keep compaction accounting consistent with the window Ollama allocates.
        if max_context_tokens is None:
            max_context_tokens = num_ctx

    # ── loop policy ─────────────────────────────────────────────────────────
    #
    # Every one of these is a ceiling on a way the loop can fail to terminate,
    # and every one is reachable from ``serving:`` in the config (see
    # SERVING_PASSTHROUGH in onit.py).  The default lives here, in the
    # signature's kwargs, so there is exactly one place each is defined — a
    # caller that does not set it does not mention it.
    #
    # Total turns before the loop gives up.  This was -1 (disabled) and the
    # only thing bounding a run was MAX_REPEATED_TOOL_CALLS, which requires the
    # same tool name *and* byte-identical arguments.  A model alternating
    # between two calls that differ in any character — a page number, a
    # timestamp, a retry suffix — never trips that check and never stops.  50
    # is above any legitimate run observed so far and far below "forever";
    # -1 restores the old unbounded behavior for anyone who wants it.
    MAX_CHAT_ITERATIONS = _as_positive_or_disabled(
        kwargs.get('max_chat_iterations', 50))
    MAX_REPEATED_TOOL_CALLS = kwargs.get('max_repeated_tool_calls', 30)
    MAX_API_RETRIES = kwargs.get('max_api_retries', 3)
    MAX_PLANNING_CONTINUATIONS = kwargs.get('max_planning_continuations', 2)
    # Max times to push past a content-free acknowledgment before accepting the
    # reply as-is.  Bounded low: a model that keeps acknowledging is stuck, and
    # looping on it burns the context that compaction just freed.
    MAX_ACK_CONTINUATIONS = kwargs.get('max_ack_continuations', 2)
    # Max times to resume a final answer that was cut off by the output token
    # budget (finish_reason=length).  Each resume grants another max_tokens of
    # output, so this bounds a very long answer at ~(N+1)*max_tokens tokens.
    MAX_FINAL_CONTINUATIONS = kwargs.get('max_final_continuations', 3)
    # Continuation token budget: thinking models can emit thousands of reasoning tokens
    # before the tool-call JSON, so give them the full max_tokens when think=True.
    # Without thinking, 512 is still enough for any tool-call JSON payload.
    CONTINUATION_MAX_TOKENS = max_tokens if think else 512
    # Compaction threshold: fire early enough that prompt_tokens + max_tokens still fits
    # within the context window after new messages are added (tool results, etc.).
    # Reserve max_tokens output budget + 5% of the window as a safety buffer.
    if max_context_tokens:
        _reserved = max_tokens + int(max_context_tokens * 0.05)
        CONTEXT_COMPACT_THRESHOLD = min(0.90, max(0.50, 1.0 - _reserved / max_context_tokens))
    else:
        CONTEXT_COMPACT_THRESHOLD = 0.90
    iteration_count = 0
    planning_continuation_count = 0
    ack_continuation_count = 0
    tool_call_history: list = []  # list of (name, args_json) tuples
    _last_prompt_tokens: int = 0  # prompt token count from the last API response
    _force_tool_call: bool = False  # set after a planning-only response to require tool use
    _active_max_tokens: int = max_tokens  # may be reduced for continuation calls
    _force_compact: bool = False  # set when finish_reason=length to compact on next iteration
    _final_continuation_count = 0  # times the final answer was resumed after truncation
    _final_answer_prefix = ""      # accumulated text from prior truncated final-answer turns
    _prose_before_tools = ""       # answer text written just before the last tool call

    async def _compact(msgs: list) -> list:
        """Compact the conversation, timing it.  Compaction is itself an LLM
        call, so its cost belongs in the telemetry rather than hidden inside
        whichever turn happened to trigger it."""
        _t0 = time.monotonic()
        out = await _compact_context(
            msgs, ollama_client if is_ollama else client,
            model, max_tokens, chat_ui, verbose,
            is_ollama=is_ollama, instruction=task_instruction,
        )
        _m.add_compaction(time.monotonic() - _t0)
        return out

    # ── fact-checking the answer ────────────────────────────────────────────
    #
    # Tools the check may use: the read-only ones, and only those.  A checker
    # able to write a file or run a command would be a second agent acting
    # after the user was told the work was done; this one may look things up
    # and nothing else.
    _verify_tools = [
        t for t in (tools or [])
        if isinstance(t, dict)
        and isinstance(t.get('function'), dict)
        and t['function'].get('name') in _READ_ONLY_TOOLS
    ]

    async def _verify_ask(msgs: list, tools: list | None = None,
                          max_tokens: int = VERDICT_MAX_TOKENS):
        """One non-streaming completion for the fact-check.

        Deliberate at temperature 0: the check is a judgment about text that
        already exists, and sampling it is how a clean answer acquires an
        imaginary error on the second run.

        And explicitly without thinking, which is the whole cost of the pass.
        A hybrid model reasons its way through a verdict by default — measured
        on Qwen3.6-27B at 18s and 1,277 tokens to report two wrong figures, and
        6.7s to report that a correct answer was correct.  The same two calls
        with thinking off returned the same verdicts in 1.2s and 0.1s.  Reading
        a claim off a source and comparing it to a sentence is recognition, not
        deliberation, and paying for a chain of thought to do it turns a check
        that runs behind a finished answer into a wait the user notices.
        """
        if is_ollama:
            _kw: dict = dict(model=model, messages=msgs, stream=False,
                             think=False,
                             options={"temperature": 0.0, "num_ctx": num_ctx,
                                      "num_predict": max_tokens})
            if tools:
                _kw["tools"] = tools
            resp = await _await_with_safety(ollama_client.chat(**_kw), safety_queue)
            if resp is _SAFETY_ABORT:
                raise _VerifyStopped()
            _msg = resp.message
            _raw = getattr(_msg, "tool_calls", None)
            return (_verify_content(_msg),
                    _adapt_ollama_tool_calls(_raw) if _raw else None)
        _kw = dict(model=model, messages=msgs, max_tokens=max_tokens,
                   temperature=0.0, stream=False)
        if tools:
            _kw["tools"] = tools
        # Asked of the chat template, which is where a hybrid model's thinking
        # is switched.  A template without the switch ignores an unknown
        # variable, but a server that validates them rejects the request — so
        # a rejection is retried plainly and remembered for the process.
        _no_think = {"chat_template_kwargs": {"enable_thinking": False}}
        for _attempt in (_no_think, {}):
            if _attempt and host in _NO_TEMPLATE_KWARGS:
                continue
            try:
                resp = await _await_with_safety(
                    client.chat.completions.create(**_kw, extra_body=_attempt),
                    safety_queue)
                break
            except OpenAIError as e:
                if not _attempt:
                    raise
                logger.info("Host %s rejected chat_template_kwargs (%s); "
                            "running the fact-check without it.", host, e)
                _NO_TEMPLATE_KWARGS.add(host)
        if resp is _SAFETY_ABORT:
            raise _VerifyStopped()
        _msg = resp.choices[0].message
        return _verify_content(_msg), (_msg.tool_calls or None)

    async def _verify_run_tools(tool_calls: list, msgs: list) -> None:
        """Run the lookups the checker asked for, into its own message list.

        Its own list, not the run's: the check is a separate conversation
        about the answer, and folding its detour back into the history the
        answer was written from would rewrite the evidence under it.
        """
        _args = ((lambda a: json.loads(a)) if is_ollama else (lambda a: a))
        _history = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": _args(tc.function.arguments)}}
                for tc in tool_calls
            ],
        }
        _t0 = time.monotonic()
        _tool_log: list = []
        # A fresh call history, so a lookup the run already made during the
        # task is not refused here as a repeat — checking a claim against the
        # same source the answer used is the point, not a loop.
        bail = await _handle_structured_tool_calls(
            tool_calls, _history, tool_registry, timeout, data_path,
            chat_ui, verbose, msgs, [], MAX_REPEATED_TOOL_CALLS,
            safety_queue, session_id=session_id, tool_log=_tool_log,
        )
        _m.add_tools([tc.function.name for tc in tool_calls],
                     time.monotonic() - _t0, runs=_tool_log)
        if bail is _SAFETY_ABORT:
            raise _VerifyStopped()

    async def _fact_check(answer: str, msgs: list) -> str:
        """The answer the user keeps: the draft, or a corrected version of it.

        Runs after the draft has finished streaming, so the user has already
        read it — this decides whether what they read stands.
        """
        if not verify_answers or not answer or not needs_verification(answer):
            return answer
        if not safety_queue.empty():
            return answer
        if chat_ui and hasattr(chat_ui, "verification_start"):
            chat_ui.verification_start()
        _t0 = time.monotonic()
        try:
            checked, note, issues = await verify_answer(
                task=task_instruction, answer=answer, messages=msgs,
                ask=_verify_ask,
                run_tools=_verify_run_tools if _verify_tools else None,
                tools=_verify_tools,
                max_tool_turns=verify_max_tool_turns,
                # A thinking model reasons through every claim before writing
                # the verdict, and a budget it overruns yields no verdict at
                # all rather than a shorter one.
                verdict_max_tokens=min(max_tokens, 8192),
                revision_max_tokens=max_tokens,
                log=lambda message, level="info": _log_to_ui_or_verbose(
                    message, chat_ui, verbose, level=level),
            )
        except Exception as e:
            logger.warning("Fact-check pass failed: %s", e)
            checked, note, issues = answer, "", []
        _m.add_verification(time.monotonic() - _t0, len(issues), bool(note))
        if chat_ui and hasattr(chat_ui, "verification_end"):
            chat_ui.verification_end(checked, note)
        return checked

    while True:
        iteration_count += 1
        if MAX_CHAT_ITERATIONS > 0 and iteration_count > MAX_CHAT_ITERATIONS:
            # Distinct from the repeated-tool-call bail-out below, which used
            # to return this same sentence.  The two are different failures —
            # one is a model stuck on one call, the other a model making
            # progress too slowly to finish — and a log that cannot tell them
            # apart cannot be used to pick a better ceiling.
            _log_to_ui_or_verbose(
                f"Chat loop hit its turn limit ({MAX_CHAT_ITERATIONS} turns), stopping. "
                f"Raise serving.max_chat_iterations if this task legitimately needs more.",
                chat_ui, verbose, level="warning")
            _partial = _final_answer_prefix or _prose_before_tools
            if _partial.strip():
                # Work was done and some of it was written down; handing back
                # nothing would throw away an answer the user already watched
                # stream past.
                return _partial
            return (f"I stopped after {MAX_CHAT_ITERATIONS} steps without finishing 😅. "
                    "Could you narrow the task, or break it into smaller pieces?")

        # Forced compaction: previous response was truncated (finish_reason=length).
        # Compact unconditionally so the next call fits within the context window.
        if _force_compact:
            _force_compact = False
            _log_to_ui_or_verbose(
                "Compacting context after truncated response (finish_reason=length)...",
                chat_ui, verbose, level="warning",
            )
            messages = await _compact(messages)
            _last_prompt_tokens = 0

        # Context compaction: check if the previous call used ≥90% of the context window.
        if _last_prompt_tokens > 0 and max_context_tokens:
            usage_pct = _last_prompt_tokens / max_context_tokens
            if chat_ui and hasattr(chat_ui, "set_context_usage"):
                chat_ui.set_context_usage(usage_pct * 100, max_context_tokens)
            if usage_pct >= CONTEXT_COMPACT_THRESHOLD:
                _log_to_ui_or_verbose(
                    f"Context at {usage_pct:.0%} ({_last_prompt_tokens:,}/{max_context_tokens:,} tokens). Compacting...",
                    chat_ui, verbose, level="warning",
                )
                messages = await _compact(messages)
                _last_prompt_tokens = 0

        _strip_old_images(messages)
        _decay_old_tool_results(messages)
        if chat_ui and hasattr(chat_ui, "set_turn_context"):
            # Prose written on a turn that follows tool calls is the answer
            # being written; the UI shows it as such rather than as one more
            # indistinguishable phase.
            chat_ui.set_turn_context(tools_run=len(tool_call_history))

        # Track streaming state across the try block for the final-response path
        _full_content = ""
        _full_reasoning = ""
        _finish_reason: str | None = None
        _completion_tokens = 0  # set from usage where the provider reports it
        # Reasoning on the opening turn is the plan; on later turns it is spent
        # deciding which tool to call next.  See think_tool_turns.
        _turn_think = think and (think_tool_turns or iteration_count == 1)

        # Pre-call compaction: _last_prompt_tokens is from the previous API call and may
        # underestimate the true prompt size after large tool results were appended.
        # If the estimated output budget is critically small, compact now rather than
        # letting the model hit finish_reason=length with an empty/tag-only response.
        if max_context_tokens and _last_prompt_tokens > 0:
            _growth_buffer_pre = max(MAX_TOOL_RESPONSE // 2, 1024)
            _available_est = max_context_tokens - _last_prompt_tokens - _growth_buffer_pre
            _min_useful_output = max(256, min(max_tokens // 4, 1024))
            if _available_est < _min_useful_output:
                _log_to_ui_or_verbose(
                    f"Estimated output budget ({max(_available_est, 0):,} tokens) is below "
                    f"minimum ({_min_useful_output:,}). Compacting context before API call...",
                    chat_ui, verbose, level="warning",
                )
                messages = await _compact(messages)
                _last_prompt_tokens = 0

        # Retry loop for transient API errors — preserves accumulated messages/tool history
        api_error = None
        for api_attempt in range(1, MAX_API_RETRIES + 1):
            api_error = None
            try:
                if not safety_queue.empty():
                    logger.warning("Safety queue triggered before API call, exiting chat loop.")
                    return None

                _m.start_api()

                # Cap output tokens so that prompt + output never approaches the context limit.
                # Never expand beyond the configured max_tokens — only shrink when the
                # remaining context window is tighter than max_tokens.
                _api_max_tokens = _active_max_tokens
                if max_context_tokens:
                    _prompt_est = _last_prompt_tokens if _last_prompt_tokens > 0 else 0
                    # Use MAX_TOOL_RESPONSE // 2 because code/JSON can be ~1.5 chars/token,
                    # so the same char limit costs more tokens than a /3 estimate implies.
                    _growth_buffer = max(MAX_TOOL_RESPONSE // 2, 1024)
                    _available = max(max_context_tokens - _prompt_est - _growth_buffer, 64)
                    _api_max_tokens = min(_api_max_tokens, _available)

                if is_ollama:
                    ollama_kwargs = dict(
                        model=model,
                        messages=messages,
                        stream=stream,
                        options={
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                            "num_ctx": num_ctx,
                            "num_predict": _api_max_tokens,
                            "presence_penalty": presence_penalty,
                            "repeat_penalty": repetition_penalty,
                        },
                    )
                    if _turn_think:
                        ollama_kwargs["think"] = True
                    if tools:
                        ollama_kwargs["tools"] = tools
                    # Force JSON output when the model keeps generating planning prose
                    # instead of tool calls.  Ollama's format="json" ensures the response
                    # is parseable by _parse_tool_call_from_content even though Ollama has
                    # no tool_choice="required" equivalent.
                    if _force_tool_call and tools:
                        ollama_kwargs["format"] = "json"
                    chat_completion = await _await_with_safety(
                        ollama_client.chat(**ollama_kwargs), safety_queue)
                    if chat_completion is _SAFETY_ABORT:
                        logger.warning("Safety queue triggered during API call, exiting chat loop.")
                        return None
                else:
                    _extra_body = {
                        "top_k": top_k,          # vLLM extension, important for Qwen3
                        "min_p": min_p,
                        "presence_penalty": presence_penalty,
                        "repetition_penalty": repetition_penalty,
                    }
                    if _turn_think:
                        _chat_template_kwargs: dict = {"enable_thinking": True}
                        # preserve_thinking is only supported on Qwen3.6+
                        _model_lower = model.lower()
                        _qwen3_ver = next(
                            (float(p.removeprefix("qwen"))
                             for p in _model_lower.replace("/", "-").split("-")
                             if p.startswith("qwen") and p[4:].replace(".", "", 1).isdigit()),
                            None,
                        )
                        if _qwen3_ver is not None and _qwen3_ver >= 3.6:
                            _chat_template_kwargs["preserve_thinking"] = True
                        _extra_body["chat_template_kwargs"] = _chat_template_kwargs
                    completion_kwargs = dict(
                        model=model,
                        messages=messages,
                        stream=stream,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=_api_max_tokens,
                        extra_body=_extra_body,
                    )
                    if tools: # and not images_bytes:  # vLLM doesn't support tools + images in the same message, so only include tools if no images are present
                        completion_kwargs["tools"] = tools
                        # vLLM rejects tool_choice when tools is unset, so only
                        # send it alongside tools.
                        completion_kwargs["tool_choice"] = "required" if _force_tool_call else "auto"
                    if stream:
                        completion_kwargs["stream_options"] = {"include_usage": True}
                    chat_completion = await _await_with_safety(
                        client.chat.completions.create(**completion_kwargs), safety_queue)
                    if chat_completion is _SAFETY_ABORT:
                        logger.warning("Safety queue triggered during API call, exiting chat loop.")
                        return None

                # Streaming path: iterate chunks, populate shared variables
                if stream:
                    if is_ollama:
                        stream_result = await _ollama_process_streaming_response(
                            chat_completion, safety_queue, chat_ui, _turn_think,
                            on_first_token=_m.first_token,
                        )
                        if stream_result is None:
                            return None
                        (_full_content, _full_reasoning, _ollama_tcs,
                         _ui_was_streaming, _ollama_prompt_tokens, _finish_reason,
                         _ollama_eval_count) = stream_result
                        _completion_tokens = _ollama_eval_count
                        if _ollama_prompt_tokens:
                            _last_prompt_tokens = _ollama_prompt_tokens
                            if max_context_tokens and chat_ui and hasattr(chat_ui, "set_context_usage"):
                                chat_ui.set_context_usage(_last_prompt_tokens / max_context_tokens * 100, max_context_tokens)
                        if _ui_was_streaming and chat_ui:
                            chat_ui.stream_end()
                        if _finish_reason == "length":
                            _log_to_ui_or_verbose(
                                "Model response truncated (finish_reason=length). "
                                "Consider increasing num_ctx/max_tokens.",
                                chat_ui, verbose, level="warning",
                            )
                            _force_compact = True
                        if _ollama_tcs:
                            _tool_calls = _adapt_ollama_tool_calls(_ollama_tcs)
                            # arguments must be dict in the history message (Ollama validates this)
                            _message_for_history = {
                                "role": "assistant",
                                "content": _prose_alongside_tool_calls(_full_content),
                                "tool_calls": [
                                    {"id": tc.id, "type": "function",
                                     "function": {"name": tc.function.name,
                                                  "arguments": json.loads(tc.function.arguments)}}
                                    for tc in _tool_calls
                                ],
                            }
                            _content = None
                        else:
                            _tool_calls = None
                            _content = _full_content
                            _message_for_history = {"role": "assistant", "content": _full_content}
                    else:
                        stream_result = await _process_streaming_response(
                            chat_completion, safety_queue, chat_ui, _turn_think,
                            on_first_token=_m.first_token,
                        )
                        if stream_result is None:
                            return None
                        _full_content, _full_reasoning, _full_tool_calls, _ui_was_streaming, _stream_usage, _finish_reason = stream_result
                        if _stream_usage is not None:
                            _last_prompt_tokens = _stream_usage.prompt_tokens
                            _completion_tokens = getattr(_stream_usage, "completion_tokens", 0)
                            if max_context_tokens and chat_ui and hasattr(chat_ui, "set_context_usage"):
                                chat_ui.set_context_usage(_last_prompt_tokens / max_context_tokens * 100, max_context_tokens)
                        if _ui_was_streaming and chat_ui:
                            chat_ui.stream_end()
                        if _finish_reason == "length":
                            _log_to_ui_or_verbose(
                                "Model response truncated (finish_reason=length). "
                                "Consider increasing max_tokens.",
                                chat_ui, verbose, level="warning",
                            )
                            _force_compact = True
                        elif _finish_reason == "tool_calls" and not _full_tool_calls:
                            _log_to_ui_or_verbose(
                                f"Model signaled finish_reason=tool_calls but no tool calls received "
                                f"(model={model}). Checking content for raw tool calls.",
                                chat_ui, verbose, level="warning",
                            )
                        _content, _tool_calls, _message_for_history = _unify_streaming_result(
                            _full_content, _full_tool_calls,
                        )

                await asyncio.sleep(0.1)
                if not safety_queue.empty():
                    logger.warning("Safety queue triggered after API call, exiting chat loop.")
                    return None
                break  # success — exit retry loop
            except (APITimeoutError, httpx.ReadTimeout) as e:
                api_error = f"Request to {host} timed out (read timeout during streaming)."
                logger.error(api_error)
                _log_to_ui_or_verbose(api_error, chat_ui, verbose, level="error")
                # tool_choice="required" engages guided decoding on vLLM, which
                # can stall on large tool schemas.  If that's what we asked for,
                # retry without it rather than wedging the server again.
                if _force_tool_call:
                    _force_tool_call = False
                    _active_max_tokens = max_tokens
                    _log_to_ui_or_verbose(
                        "Dropping tool_choice=required for retry (possible guided-decoding stall).",
                        chat_ui, verbose, level="warning",
                    )
            except NotFoundError as e:
                api_error = f"Model {model!r} not found at {host}: {e}."
                logger.error(api_error)
                _log_to_ui_or_verbose(api_error, chat_ui, verbose, level="warning")
                _detected = await _autodetect_fallback_model(
                    client, ollama_client, is_ollama, host, model)
                if _detected:
                    model = _detected
                    if chat_ui:
                        chat_ui.model_name = model
                    _log_to_ui_or_verbose(
                        f"Falling back to auto-detected model: {model}",
                        chat_ui, verbose, level="warning")
            except OpenAIError as e:
                api_error = f"Error communicating with {host}: {e}."
                logger.error(api_error)
                _log_to_ui_or_verbose(api_error, chat_ui, verbose, level="warning")
            except Exception as e:
                api_error = f"Unexpected error ({type(e).__name__}): {e}"
                logger.error(api_error, exc_info=True)
                _log_to_ui_or_verbose(api_error, chat_ui, verbose, level="error")

            # Log retry attempt if we haven't exhausted retries
            if api_attempt < MAX_API_RETRIES:
                _m.add_retry()
                retry_msg = f"Retrying API call (attempt {api_attempt + 1}/{MAX_API_RETRIES})..."
                logger.info(retry_msg)
                _log_to_ui_or_verbose(retry_msg, chat_ui, verbose, level="info")
                await asyncio.sleep(min(2 ** api_attempt, 10))  # exponential backoff

        if api_error is not None:
            # All retries exhausted
            return None

        # Non-streaming: extract from response object into unified variables
        if not stream:
            if is_ollama:
                _msg = chat_completion.message
                _content = _msg.content
                _raw_tcs = getattr(_msg, "tool_calls", None)
                _tool_calls = _adapt_ollama_tool_calls(_raw_tcs) if _raw_tcs else None
                _finish_reason = getattr(chat_completion, "done_reason", None)
                if _tool_calls:
                    _message_for_history = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": tc.id, "type": "function",
                             "function": {"name": tc.function.name,
                                          "arguments": json.loads(tc.function.arguments)}}
                            for tc in _tool_calls
                        ],
                    }
                else:
                    _message_for_history = {"role": "assistant", "content": _content}
                # Ollama exposes prompt_eval_count for token tracking
                _completion_tokens = getattr(chat_completion, "eval_count", 0)
                _pec = getattr(chat_completion, "prompt_eval_count", None)
                if _pec is not None:
                    _last_prompt_tokens = _pec
                    if max_context_tokens and chat_ui and hasattr(chat_ui, "set_context_usage"):
                        chat_ui.set_context_usage(_last_prompt_tokens / max_context_tokens * 100, max_context_tokens)
            else:
                _choice = chat_completion.choices[0]
                _msg = _choice.message
                _content = _msg.content
                _tool_calls = _msg.tool_calls if _msg.tool_calls else None
                _finish_reason = _choice.finish_reason
                _message_for_history = _msg
                # Warn on unexpected finish reasons
                if _finish_reason == "length":
                    _log_to_ui_or_verbose(
                        "Model response truncated (finish_reason=length). "
                        "Consider increasing max_tokens.",
                        chat_ui, verbose, level="warning",
                    )
                    _force_compact = True
                elif _finish_reason == "tool_calls" and not _tool_calls:
                    _log_to_ui_or_verbose(
                        f"Model signaled finish_reason=tool_calls but no tool calls received "
                        f"(model={model}). Checking content for raw tool calls.",
                        chat_ui, verbose, level="warning",
                    )
                # Capture token usage for context tracking
                if chat_completion.usage is not None:
                    _last_prompt_tokens = chat_completion.usage.prompt_tokens
                    _completion_tokens = getattr(chat_completion.usage, "completion_tokens", 0)
                    if max_context_tokens and chat_ui and hasattr(chat_ui, "set_context_usage"):
                        chat_ui.set_context_usage(_last_prompt_tokens / max_context_tokens * 100, max_context_tokens)

        # Both paths have converged: the model call for this turn is done.
        _m.end_api(_last_prompt_tokens, _completion_tokens, _finish_reason)

        tool_calls = _tool_calls
        if tool_calls is None or len(tool_calls) == 0:
            # No structured tool calls -- check for raw JSON tool calls in content
            _t_tools = time.monotonic()
            _tool_log: list = []
            should_continue, bail = await _handle_raw_tool_call(
                _content, tool_registry, timeout, data_path,
                chat_ui, verbose, messages, tool_call_history,
                MAX_REPEATED_TOOL_CALLS, session_id=session_id,
                tool_log=_tool_log,
            )
            if should_continue or bail:
                # A tool ran; its name was parsed out of the content inside the
                # handler, so record it under the shape it arrived in.
                _m.add_tools(["<raw tool call>"], time.monotonic() - _t_tools,
                             runs=_tool_log)
            if bail:
                return bail
            if should_continue:
                continue

            # Truncated final answer: the model produced plain-text output (not a
            # tool call) that was cut off by the output token budget
            # (finish_reason=length).  Resume generation from the partial text so
            # the user gets the complete response, stitching the pieces together.
            # Compaction would be wrong here — there is context room to spare; the
            # answer simply exceeded max_tokens — and it would summarize away the
            # exact partial text we need to continue from.
            if (_finish_reason == "length"
                    and _final_continuation_count < MAX_FINAL_CONTINUATIONS):
                _final_continuation_count += 1
                _force_compact = False  # don't discard the partial answer
                _partial = _extract_final_response(_content, _full_reasoning, _full_content)
                _final_answer_prefix += _partial
                messages.append({"role": "assistant", "content": _partial})
                messages.append({"role": "user", "content": _FINAL_CONTINUATION_PROMPT})
                _log_to_ui_or_verbose(
                    f"Final response truncated (finish_reason=length); resuming "
                    f"({_final_continuation_count}/{MAX_FINAL_CONTINUATIONS})...",
                    chat_ui, verbose, level="info",
                )
                continue

            # Detect planning responses: the model announced intent ("Let me create X")
            # but stopped without calling any tools.  Inject a concrete JSON-format
            # continuation prompt and cap tokens to limit time waste on stuck models.
            # Skip when finish_reason=length: a token-truncated response is not a plan —
            # "Let me" appearing mid-response would cause a false positive continuation loop.
            if (tools and tool_registry
                    and _finish_reason != "length"
                    and _is_planning_response(_content)
                    and planning_continuation_count < MAX_PLANNING_CONTINUATIONS):
                planning_continuation_count += 1
                _force_tool_call = True
                _active_max_tokens = CONTINUATION_MAX_TOKENS
                _log_to_ui_or_verbose(
                    f"Model announced a plan without calling tools (continuation "
                    f"{planning_continuation_count}/{MAX_PLANNING_CONTINUATIONS}, "
                    f"capping tokens={CONTINUATION_MAX_TOKENS}).",
                    chat_ui, verbose, level="info",
                )
                continuation_prompt = _build_planning_continuation_prompt(
                    tool_registry, planning_continuation_count, task_instruction
                )
                messages.append({"role": "assistant", "content": _content})
                messages.append({"role": "user", "content": continuation_prompt})
                continue

            # Continuations exhausted — model cannot call tools
            if (tools and tool_registry
                    and _finish_reason != "length"
                    and _is_planning_response(_content)
                    and planning_continuation_count >= MAX_PLANNING_CONTINUATIONS):
                _log_to_ui_or_verbose(
                    f"Model ({model}) failed to call tools after "
                    f"{MAX_PLANNING_CONTINUATIONS} continuation attempts. "
                    "It may not support agentic tool use.",
                    chat_ui, verbose, level="warning",
                )
                return (
                    f"This model ({model}) was unable to complete the task — it repeatedly "
                    f"described a plan but did not call any tools after "
                    f"{MAX_PLANNING_CONTINUATIONS} attempts. "
                    "Try a model with stronger tool-calling support "
                    "(e.g. qwen3, mistral-nemo, llama3.1, deepseek-r1)."
                )

            # Content-free acknowledgment, or commentary on the prompt itself:
            # the model answered a compaction or continuation prompt with filler
            # instead of resuming work.  Handing that back would end the task on a
            # non-answer, so nudge it once.  Both share one budget — a model doing
            # either is stuck the same way.
            _is_ack = _is_acknowledgment_response(_content)
            _is_meta = _is_meta_commentary_response(_content)
            # Third test, and the one that does not depend on knowing the
            # wording in advance: the reply carries nothing the prompt did not
            # already state.  data_path goes in because the session directory is
            # the metadata most often read back.
            _is_empty = (not _is_ack and not _is_meta
                         and _is_answering_a_nudge(messages)
                         and _is_content_free_response(_content, data_path))
            if (_finish_reason != "length"
                    and (_is_ack or _is_meta or _is_empty)
                    and ack_continuation_count < MAX_ACK_CONTINUATIONS):
                ack_continuation_count += 1
                _what = ("acknowledged" if _is_ack
                         else "commented on the prompt" if _is_meta
                         else "replied with nothing the prompt did not supply")
                _log_to_ui_or_verbose(
                    f"Model {_what} instead of resuming the task (continuation "
                    f"{ack_continuation_count}/{MAX_ACK_CONTINUATIONS}).",
                    chat_ui, verbose, level="info",
                )
                messages.append({"role": "assistant", "content": _content})
                messages.append({"role": "user", "content": _ACK_CONTINUATION_PROMPT})
                continue

            _force_tool_call = False
            _active_max_tokens = max_tokens
            _final = _extract_final_response(_content, _full_reasoning, _full_content)
            # Stitch on any text from earlier turns that were truncated and resumed.
            if _final_answer_prefix:
                _final = _final_answer_prefix + _final
            # The one exit that hands back an answer about the world.  Every
            # other return above is the loop reporting on itself — a limit hit,
            # a model that cannot call tools — and there is nothing in those to
            # check against evidence.
            return await _fact_check(
                _recover_dropped_answer(_final, _prose_before_tools), messages)

        # Keep the answer text written ahead of this tool call.  A model that
        # answers and then calls one last tool often signs off with "Done." —
        # the user already saw the real answer stream by, so don't lose it.
        _prose = _prose_alongside_tool_calls(_message_content(_message_for_history))
        if _prose and not _is_planning_response(_prose):
            _prose_before_tools = _prose

        # Structured tool calls: execute them and loop back.
        # Reset force/token/planning flags — the model successfully called a tool.
        # A batch of nothing but no-ops does not clear the planning budget: that is
        # how a stuck model launders "echo ok" into another two continuations and
        # loops indefinitely.  The call still runs — it may be the model's own way
        # of orienting — it just doesn't count as progress.
        _force_tool_call = False
        _active_max_tokens = max_tokens
        _all_noop = all(
            _is_noop_tool_call(tc.function.name, _parse_tool_arguments(tc, verbose))
            for tc in tool_calls
        )
        if _all_noop:
            _log_to_ui_or_verbose(
                "Tool call does nothing (no-op shell command); keeping the "
                f"planning budget at {planning_continuation_count}/"
                f"{MAX_PLANNING_CONTINUATIONS}.",
                chat_ui, verbose, level="info",
            )
        else:
            # Only a call that did something counts as progress.  The ack budget
            # is reset on the same condition and for the same reason: a model
            # that acknowledges, runs "echo ok", then acknowledges again has not
            # advanced, and refreshing its budget here lets it do that forever.
            planning_continuation_count = 0
            ack_continuation_count = 0
        _t_tools = time.monotonic()
        _tool_log = []
        bail = await _handle_structured_tool_calls(
            tool_calls, _message_for_history, tool_registry,
            timeout, data_path, chat_ui, verbose,
            messages, tool_call_history, MAX_REPEATED_TOOL_CALLS,
            safety_queue, session_id=session_id, tool_log=_tool_log,
        )
        _m.add_tools([tc.function.name for tc in tool_calls],
                     time.monotonic() - _t_tools, runs=_tool_log)
        if bail is _SAFETY_ABORT:
            return None
        if bail:
            return bail
