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
import os
import json
import re
import types
import uuid
import httpx
from openai import AsyncOpenAI, OpenAIError, APITimeoutError
from typing import List, Optional, Any

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


def _truncate_tool_response(response: str) -> str:
    """Truncate a tool response if it exceeds MAX_TOOL_RESPONSE characters."""
    if len(response) <= MAX_TOOL_RESPONSE:
        return response
    half = MAX_TOOL_RESPONSE // 2
    return response[:half] + f"\n\n... [truncated {len(response) - MAX_TOOL_RESPONSE} chars] ...\n\n" + response[-half:]


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
    keychain. For vLLM and other local hosts, default to "EMPTY".
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
    return host_key


def _create_ollama_client(host: str, api_key: str, timeout):
    """Create an OllamaAsyncClient configured for the given host and API key."""
    if not OLLAMA_SDK_AVAILABLE:
        raise ImportError("The 'ollama' package is required for Ollama cloud support.")
    _timeout = None if (timeout is None or timeout < 0) else timeout
    return OllamaAsyncClient(
        host=host,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=_timeout,
    )


async def _ollama_resolve_model_id(client, host: str) -> str:
    """Fetch the first available model from an Ollama endpoint via client.list()."""
    response = await client.list()
    if not response.models:
        raise ValueError(f"No models available at {host}")
    model_id = response.models[0].model
    logger.info("Auto-detected Ollama model: %s from %s", model_id, host)
    return model_id


async def _resolve_model_id(client: AsyncOpenAI, host: str) -> str:
    """Fetch the first available model ID from the endpoint.

    vLLM typically serves a single model, so models.data[0].id is used.
    For OpenRouter, the caller should always supply the model name explicitly.
    """
    models = await client.models.list()
    if not models.data:
        raise ValueError(f"No models available at {host}")
    model_id = models.data[0].id
    logger.info("Auto-detected model: %s from %s", model_id, host)
    return model_id


async def _get_model_max_context(host: str, api_key: str, model: str) -> Optional[int]:
    """Query vLLM for the model's maximum context length.

    vLLM exposes ``max_model_len`` in the ``/v1/models`` response as an extra
    field.  Returns None for OpenRouter or any host that doesn't provide it.
    """
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
                            return int(val)
    except Exception as e:
        logger.debug("Could not query max_model_len from vLLM: %s", e)
    return None


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
    mime_type = data.get("mime_type", "application/octet-stream")

    # Pick a sensible extension from the mime type
    _ext_map = {
        "image/jpeg": ".jpg", "image/png": ".png",
        "image/gif": ".gif", "image/webp": ".webp",
    }
    ext = _ext_map.get(mime_type, ".bin")
    file_name = data.get("file_name", f"{uuid.uuid4()}{ext}")
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
                        session_id: str = "") -> Optional[str]:
    """Execute a single tool call and append the result to messages.

    Returns a bail-out message string if repeated-call limit is hit,
    otherwise returns None (caller should continue).
    """
    # Inject session_id / data_path into tool calls whose schema declares
    # these parameters, so callers (e.g. sandbox MCP servers) receive them
    # automatically without hardcoding tool names.
    if session_id and tool_registry.tool_accepts_param(function_name, "session_id"):
        function_arguments.setdefault("session_id", session_id)
    if data_path and tool_registry.tool_accepts_param(function_name, "data_path"):
        function_arguments.setdefault("data_path", data_path)
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
    else:
        tool_handler = tool_registry[function_name]
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
        except Exception as e:
            if chat_ui:
                if is_structured:
                    chat_ui.stop_tool_spinner()
                    chat_ui.show_tool_done(function_name, str(e), success=False)
            _log_to_ui_or_verbose(f"{function_name} error: {e}", chat_ui, verbose, level="error")
            tool_message = {'role': 'tool', 'content': f'Error: {e}', 'name': function_name,
                            'parameters': function_arguments, "tool_call_id": tool_call_id}
            messages.append(tool_message)

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


def _build_messages(instruction: str, images_bytes: list[str],
                    prompt_intro: str, session_history: list | None,
                    memories: Any) -> list[dict]:
    """Assemble the initial message list for the API call.

    Includes the system message, session history, the current user instruction,
    and the empty tool sentinel when appropriate.
    """
    if images_bytes:
        messages: list[dict] = [{
            "role": "system",
            "content": (
                f"{prompt_intro} "
                "You are an expert vision-language assistant. Your task is to analyze images with high precision, "
                "reasoning step-by-step about visual elements and their spatial relationships (e.g., coordinates, "
                "relative positions like left/right/center). Always verify visual evidence before concluding. "
                "If a task requires external data, calculation, or specific actions beyond visual description, "
                "use the provided tools. Be concise, objective, and format your tool calls strictly according to schema."
            )
        }]
    else:
        messages = [{"role": "system", "content": prompt_intro}]

    # Inject session history BEFORE the current instruction so the model
    # sees prior context first and treats the latest user message as the
    # one to respond to.
    if session_history:
        for entry in session_history:
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
    chat_ui, think: bool,
) -> tuple[str, str, dict, bool, Any, str | None] | None:
    """Consume a streaming chat completion and return accumulated results.

    Returns (full_content, full_reasoning, full_tool_calls_dict, ui_was_streaming, usage, finish_reason)
    or None if the safety queue fired mid-stream.  ``usage`` is the
    CompletionUsage object from the final chunk (requires stream_options
    include_usage=True), or None if not available.  ``finish_reason`` is the
    stop reason from the final chunk (e.g. "stop", "tool_calls", "length").
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
) -> tuple[str, str, list | None, bool] | None:
    """Consume an Ollama streaming chat completion and return accumulated results.

    Returns (full_content, full_thinking, tool_calls_or_None, ui_was_streaming)
    or None if the safety queue fired mid-stream.

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

    try:
        async for chunk in chat_completion:
            if not safety_queue.empty():
                if ui_streaming and chat_ui:
                    chat_ui.stream_end()
                return None

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

    return full_content, full_thinking, tool_calls, ui_streaming, prompt_eval_count


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
            "content": None,
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

def _is_planning_response(content: str) -> bool:
    """Return True if the response looks like a plan announcement rather than a final answer.

    Detects patterns like "Let me create X and then push it" where the model
    states its intent in future tense but stops before executing tool calls.
    Only returns True when tools are available (caller's responsibility).
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
        if f". {p}" in lower or f"\n{p}" in lower:
            return True
    return False


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


def _extract_final_response(content: str, full_reasoning: str, full_content: str) -> str:
    """Clean up the final text response, stripping think tags and applying fallbacks."""
    last_response = content
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


async def _handle_raw_tool_call(
    last_response: str, tool_registry, timeout, data_path,
    chat_ui, verbose: bool, messages: list,
    tool_call_history: list, max_repeated: int,
    session_id: str = "",
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


async def _handle_structured_tool_calls(
    tool_calls: list, message_for_history, tool_registry,
    timeout, data_path, chat_ui, verbose: bool,
    messages: list, tool_call_history: list,
    max_repeated: int, safety_queue: asyncio.Queue,
    session_id: str = "",
) -> str | object | None:
    """Execute structured tool calls and append results to messages.

    Returns:
      - A bail-out message string if a repeated-call limit is hit.
      - _SAFETY_ABORT sentinel if the safety queue fired.
      - None when all tools completed normally.
    """
    messages.append(message_for_history)
    for tool in tool_calls:
        await asyncio.sleep(0.1)
        if not safety_queue.empty():
            if verbose:
                print("Safety queue triggered, exiting chat loop.")
            return _SAFETY_ABORT
        function_name = tool.function.name
        try:
            function_arguments = json.loads(tool.function.arguments)
        except json.JSONDecodeError:
            # Try fixing common issues: single quotes, trailing commas
            import re
            fixed = tool.function.arguments.strip()
            # Replace single quotes with double quotes
            fixed = fixed.replace("'", '"')
            # Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            try:
                function_arguments = json.loads(fixed)
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Failed to parse tool arguments for {function_name}: {e}")
                    print(f"Raw arguments: {tool.function.arguments}")
                function_arguments = {}
        bail = await _execute_tool(
            function_name, function_arguments, tool.id,
            tool_registry, timeout, data_path, chat_ui, verbose,
            messages, tool_call_history, max_repeated,
            is_structured=True, session_id=session_id,
        )
        if bail:
            return bail
    return None


async def _compact_context(
    messages: list, client, model: str,
    max_tokens: int, chat_ui, verbose: bool,
    is_ollama: bool = False,
) -> list:
    """Summarize the conversation and return a compacted messages list.

    Keeps the system message, generates a dense LLM summary of all other
    messages, and returns [system_msg, compacted_user_msg, ack_assistant_msg].
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
    new_messages.append({
        "role": "user",
        "content": (
            "[CONTEXT COMPACTED]\nThe following is a summary of prior work:\n\n"
            + summary
            + "\n\n[Continue the task based on the summary above.]"
        ),
    })
    new_messages.append({
        "role": "assistant",
        "content": "Understood. Continuing based on the context summary.",
    })
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
    max_tokens = kwargs.get('max_tokens', 262144)
    temperature = kwargs.get('temperature', 0.6)
    top_p = kwargs.get('top_p', 0.95)
    top_k = kwargs.get('top_k', 20)
    min_p = kwargs.get('min_p', 0.0)
    presence_penalty = kwargs.get('presence_penalty', 0.0)
    repetition_penalty = kwargs.get('repetition_penalty', 1.0 if think else 1.05)
    memories = kwargs.get('memories', None)
    prompt_intro = kwargs.get('prompt_intro', "I am a helpful AI assistant. My name is OnIt.")
    max_context_tokens: Optional[int] = kwargs.get('max_context_tokens', None)

    images_bytes = _load_images(images, chat_ui, verbose)
    messages = _build_messages(instruction, images_bytes, prompt_intro,
                               kwargs.get('session_history', None), memories)

    api_key = _resolve_api_key(host, host_key)
    # Use explicit timeout if provided; -1 or None means no timeout (infinite wait).
    _client_timeout = None if (timeout is None or timeout < 0) else timeout
    is_ollama = _is_ollama_host(host)
    if is_ollama:
        ollama_client = _create_ollama_client(host, api_key, timeout)
        client = None
    else:
        client = AsyncOpenAI(base_url=host, api_key=api_key, timeout=_client_timeout)
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

    MAX_CHAT_ITERATIONS = -1
    MAX_REPEATED_TOOL_CALLS = 30
    MAX_API_RETRIES = 3
    MAX_PLANNING_CONTINUATIONS = 2
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
    tool_call_history: list = []  # list of (name, args_json) tuples
    _last_prompt_tokens: int = 0  # prompt token count from the last API response
    _force_tool_call: bool = False  # set after a planning-only response to require tool use
    _active_max_tokens: int = max_tokens  # may be reduced for continuation calls

    while True:
        iteration_count += 1
        if MAX_CHAT_ITERATIONS >= 0 and iteration_count > MAX_CHAT_ITERATIONS:
            msg = f"I am sorry 😊. Could you try to rephrase or provide additional details?"
            _log_to_ui_or_verbose(f"Chat loop exceeded {MAX_CHAT_ITERATIONS} iterations, stopping.", chat_ui, verbose, level="warning")
            return msg

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
                messages = await _compact_context(
                    messages,
                    ollama_client if is_ollama else client,
                    model, max_tokens, chat_ui, verbose,
                    is_ollama=is_ollama,
                )
                _last_prompt_tokens = 0

        _strip_old_images(messages)

        # Track streaming state across the try block for the final-response path
        _full_content = ""
        _full_reasoning = ""
        _finish_reason: str | None = None

        # Retry loop for transient API errors — preserves accumulated messages/tool history
        api_error = None
        for api_attempt in range(1, MAX_API_RETRIES + 1):
            api_error = None
            try:
                if not safety_queue.empty():
                    logger.warning("Safety queue triggered before API call, exiting chat loop.")
                    return None

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
                            "num_predict": _api_max_tokens,
                            "presence_penalty": presence_penalty,
                            "repeat_penalty": repetition_penalty,
                        },
                    )
                    if think:
                        ollama_kwargs["think"] = True
                    if tools:
                        ollama_kwargs["tools"] = tools
                    # Force JSON output when the model keeps generating planning prose
                    # instead of tool calls.  Ollama's format="json" ensures the response
                    # is parseable by _parse_tool_call_from_content even though Ollama has
                    # no tool_choice="required" equivalent.
                    if _force_tool_call and tools:
                        ollama_kwargs["format"] = "json"
                    chat_completion = await ollama_client.chat(**ollama_kwargs)
                else:
                    _extra_body = {
                        "top_k": top_k,          # vLLM extension, important for Qwen3
                        "min_p": min_p,
                        "presence_penalty": presence_penalty,
                        "repetition_penalty": repetition_penalty,
                    }
                    if think:
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
                        tool_choice="required" if (_force_tool_call and tools) else "auto",
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=_api_max_tokens,
                        extra_body=_extra_body,
                    )
                    if tools: # and not images_bytes:  # vLLM doesn't support tools + images in the same message, so only include tools if no images are present
                        completion_kwargs["tools"] = tools
                    if stream:
                        completion_kwargs["stream_options"] = {"include_usage": True}
                    chat_completion = await client.chat.completions.create(**completion_kwargs)

                # Streaming path: iterate chunks, populate shared variables
                if stream:
                    if is_ollama:
                        stream_result = await _ollama_process_streaming_response(
                            chat_completion, safety_queue, chat_ui, think,
                        )
                        if stream_result is None:
                            return None
                        _full_content, _full_reasoning, _ollama_tcs, _ui_was_streaming, _ollama_prompt_tokens = stream_result
                        if _ollama_prompt_tokens:
                            _last_prompt_tokens = _ollama_prompt_tokens
                            if max_context_tokens and chat_ui and hasattr(chat_ui, "set_context_usage"):
                                chat_ui.set_context_usage(_last_prompt_tokens / max_context_tokens * 100, max_context_tokens)
                        if _ui_was_streaming and chat_ui:
                            chat_ui.stream_end()
                        if _ollama_tcs:
                            _tool_calls = _adapt_ollama_tool_calls(_ollama_tcs)
                            # arguments must be dict in the history message (Ollama validates this)
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
                            _content = None
                        else:
                            _tool_calls = None
                            _content = _full_content
                            _message_for_history = {"role": "assistant", "content": _full_content}
                    else:
                        stream_result = await _process_streaming_response(
                            chat_completion, safety_queue, chat_ui, think,
                        )
                        if stream_result is None:
                            return None
                        _full_content, _full_reasoning, _full_tool_calls, _ui_was_streaming, _stream_usage, _finish_reason = stream_result
                        if _stream_usage is not None:
                            _last_prompt_tokens = _stream_usage.prompt_tokens
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
                elif _finish_reason == "tool_calls" and not _tool_calls:
                    _log_to_ui_or_verbose(
                        f"Model signaled finish_reason=tool_calls but no tool calls received "
                        f"(model={model}). Checking content for raw tool calls.",
                        chat_ui, verbose, level="warning",
                    )
                # Capture token usage for context tracking
                if chat_completion.usage is not None:
                    _last_prompt_tokens = chat_completion.usage.prompt_tokens
                    if max_context_tokens and chat_ui and hasattr(chat_ui, "set_context_usage"):
                        chat_ui.set_context_usage(_last_prompt_tokens / max_context_tokens * 100, max_context_tokens)

        tool_calls = _tool_calls
        if tool_calls is None or len(tool_calls) == 0:
            # No structured tool calls -- check for raw JSON tool calls in content
            should_continue, bail = await _handle_raw_tool_call(
                _content, tool_registry, timeout, data_path,
                chat_ui, verbose, messages, tool_call_history,
                MAX_REPEATED_TOOL_CALLS, session_id=session_id,
            )
            if bail:
                return bail
            if should_continue:
                continue

            # Detect planning responses: the model announced intent ("Let me create X")
            # but stopped without calling any tools.  Inject a concrete JSON-format
            # continuation prompt and cap tokens to limit time waste on stuck models.
            if (tools and tool_registry
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

            _force_tool_call = False
            _active_max_tokens = max_tokens
            return _extract_final_response(_content, _full_reasoning, _full_content)

        # Structured tool calls: execute them and loop back.
        # Reset force/token/planning flags — the model successfully called a tool.
        _force_tool_call = False
        _active_max_tokens = max_tokens
        planning_continuation_count = 0
        bail = await _handle_structured_tool_calls(
            tool_calls, _message_for_history, tool_registry,
            timeout, data_path, chat_ui, verbose,
            messages, tool_call_history, MAX_REPEATED_TOOL_CALLS,
            safety_queue, session_id=session_id,
        )
        if bail is _SAFETY_ABORT:
            return None
        if bail:
            return bail
