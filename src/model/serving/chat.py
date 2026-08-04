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
from openai import AsyncOpenAI, OpenAIError, APITimeoutError, NotFoundError
from typing import List, Optional, Any

try:
    from ...lib.text import split_instruction
    from ...learn import describe_tool_call, redact_tool_args
except ImportError:  # imported with src/ itself on sys.path (tests, scripts)
    from lib.text import split_instruction
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


def reset_endpoint_caches() -> None:
    """Forget every cached model id and context window.

    For tests, and for callers that reconfigure an endpoint in-process.
    """
    _MODEL_ID_CACHE.clear()
    _MAX_CONTEXT_CACHE.clear()
    _OLLAMA_CONTEXT_CACHE.clear()


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

    if function_name not in tool_registry.tools:
        tool_message = {'role': 'tool', 'content': f'Error: tool {function_name} not found',
                        'name': function_name, 'parameters': function_arguments,
                        "tool_call_id": tool_call_id}
        messages.append(tool_message)
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
    prompt_eval_count, done_reason) or None if the safety queue fired mid-stream.
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

            # Capture prompt token count from final chunk for context tracking.
            pec = getattr(chunk, "prompt_eval_count", None)
            if pec:
                prompt_eval_count = pec

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

    return full_content, full_thinking, tool_calls, ui_streaming, prompt_eval_count, done_reason


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


def _build_tool_example(tool_registry) -> str:
    """Return a filled-in JSON tool-call example using the first available tool's schema.

    Prefers common action tools (bash, write_file, create_file) so the example
    contains real argument names rather than an empty ``{}``.
    """
    preferred = ("bash", "shell", "run_command", "write_file", "create_file",
                 "read_file", "list_files")
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


def _build_planning_continuation_prompt(tool_registry, continuation_count: int) -> str:
    """Build a direct continuation prompt for models stuck in planning mode.

    Includes a concrete JSON tool-call example (with real argument shapes) so the
    raw-tool-call parser can catch it even on models that don't honour
    tool_choice=required (e.g. Ollama).
    """
    tool_names = sorted(tool_registry.tools)[:6] if tool_registry else []
    example = _build_tool_example(tool_registry)
    tools_list = ", ".join(tool_names)
    header = "OUTPUT ONLY JSON — no prose, no explanation." if continuation_count > 1 else "Do not write any text."
    return (
        f"{header} Call a tool RIGHT NOW using this exact JSON format:\n"
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
        chat_ui.start_tool_batch([name for name, _, _ in calls])

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
    if instruction:
        compacted_content = (
            "[CONTEXT COMPACTED]\n"
            "The original task instruction is restated below and all of its "
            "rules remain in effect, followed by a summary of prior work.\n\n"
            "## Original instruction\n"
            + instruction
            + "\n\n## Summary of prior work\n"
            + summary
            + "\n\n[Continue the task, following the original instruction and "
            "building on the summary above.]"
        )
    else:
        compacted_content = (
            "[CONTEXT COMPACTED]\nThe following is a summary of prior work:\n\n"
            + summary
            + "\n\n[Continue the task based on the summary above.]"
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

    tools = tool_registry.get_tool_items() if tool_registry else []
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

    MAX_CHAT_ITERATIONS = -1
    MAX_REPEATED_TOOL_CALLS = 30
    MAX_API_RETRIES = 3
    MAX_PLANNING_CONTINUATIONS = 2
    # Max times to push past a content-free acknowledgment before accepting the
    # reply as-is.  Bounded low: a model that keeps acknowledging is stuck, and
    # looping on it burns the context that compaction just freed.
    MAX_ACK_CONTINUATIONS = 2
    # Max times to resume a final answer that was cut off by the output token
    # budget (finish_reason=length).  Each resume grants another max_tokens of
    # output, so this bounds a very long answer at ~(N+1)*max_tokens tokens.
    MAX_FINAL_CONTINUATIONS = 3
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

    while True:
        iteration_count += 1
        if MAX_CHAT_ITERATIONS >= 0 and iteration_count > MAX_CHAT_ITERATIONS:
            msg = f"I am sorry 😊. Could you try to rephrase or provide additional details?"
            _log_to_ui_or_verbose(f"Chat loop exceeded {MAX_CHAT_ITERATIONS} iterations, stopping.", chat_ui, verbose, level="warning")
            return msg

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
                        _full_content, _full_reasoning, _ollama_tcs, _ui_was_streaming, _ollama_prompt_tokens, _finish_reason = stream_result
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
                    tool_registry, planning_continuation_count
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

            # Content-free acknowledgment: the model answered a compaction or
            # continuation prompt with filler instead of resuming work.  Handing
            # that back would end the task on a non-answer, so nudge it once.
            if (_finish_reason != "length"
                    and _is_acknowledgment_response(_content)
                    and ack_continuation_count < MAX_ACK_CONTINUATIONS):
                ack_continuation_count += 1
                _log_to_ui_or_verbose(
                    f"Model acknowledged instead of resuming the task (continuation "
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
            return _recover_dropped_answer(_final, _prose_before_tools)

        # Keep the answer text written ahead of this tool call.  A model that
        # answers and then calls one last tool often signs off with "Done." —
        # the user already saw the real answer stream by, so don't lose it.
        _prose = _prose_alongside_tool_calls(_message_content(_message_for_history))
        if _prose and not _is_planning_response(_prose):
            _prose_before_tools = _prose

        # Structured tool calls: execute them and loop back.
        # Reset force/token/planning flags — the model successfully called a tool.
        _force_tool_call = False
        _active_max_tokens = max_tokens
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
