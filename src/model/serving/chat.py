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

Chat function supporting private vLLM and OpenRouter.ai models via OpenAI-compatible API.
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
from openai import AsyncOpenAI, OpenAIError, APITimeoutError
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


def _resolve_api_key(host: str, host_key: str = "EMPTY") -> str:
    """Resolve the API key based on the host URL.

    For OpenRouter hosts, use host_key param or OPENROUTER_API_KEY env var.
    For vLLM and other local hosts, default to "EMPTY".
    """
    if "openrouter.ai" in host:
        if host_key and host_key != "EMPTY":
            return host_key
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError(
                "OpenRouter requires an API key. Set it via:\n"
                "  - serving.host_key in the config YAML\n"
                "  - OPENROUTER_API_KEY environment variable"
            )
        return key
    return host_key


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


def _parse_tool_call_from_content(content: str, tool_registry) -> Optional[dict]:
    """Detect a raw JSON tool call in message content.

    Some models return tool calls as plain JSON in the response body instead of
    using the structured tool_calls field.  This function tries to parse the
    content and, if it looks like a valid tool call for a known tool, returns
    a dict with 'name' and 'arguments'.
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
    # Quick heuristic: must contain both "name" and "arguments" keys in JSON-like syntax
    return bool(re.search(r'"name"\s*:\s*"[^"]+"', text) and re.search(r'"arguments"\s*:', text))


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


async def chat(host: str = "http://127.0.0.1:8001/v1",
         host_key: str = "EMPTY",
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
    max_tokens = kwargs.get('max_tokens', 8192)
    memories = kwargs.get('memories', None)
    prompt_intro = kwargs.get('prompt_intro', "I am a helpful AI assistant. My name is OnIt.")

    images_bytes = []
    if isinstance(images, list):
        for image_path in images:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as image_file:
                    images_bytes.append(base64.b64encode(image_file.read()).decode('utf-8'))
            else:
                if chat_ui:
                    chat_ui.add_log(f"Image file {image_path} not found, proceeding without this image.", level="warning")
                elif verbose:
                    print(f"Image file {image_path} not found, proceeding without this image.")
    elif isinstance(images, str):
        image_path = images
        if os.path.exists(image_path):
            with open(image_path, 'rb') as image_file:
                images_bytes = [base64.b64encode(image_file.read()).decode('utf-8')]
        else:
            if chat_ui:
                chat_ui.add_log(f"Image file {image_path} not found, proceeding without this image.", level="warning")
            elif verbose:
                print(f"Image file {image_path} not found, proceeding without this image.")

    if images_bytes:
        messages = [{
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
    session_history = kwargs.get('session_history', None)
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
        
    if not memories and not session_history:
        message = {'role': 'tool', 'content': '', 'name': '', 'parameters': {}, "tool_call_id": ''}
        messages.append(message)

    api_key = _resolve_api_key(host, host_key)
    client = AsyncOpenAI(base_url=host, api_key=api_key)

    model = await _resolve_model_id(client, host)

    if chat_ui:
        chat_ui.model_name = model
        chat_ui.add_log(f"Starting chat with model: {model}", level="info")
    elif verbose:
        print(f"Starting chat with model: {model}")

    MAX_CHAT_ITERATIONS = 100
    MAX_REPEATED_TOOL_CALLS = 30
    iteration_count = 0
    tool_call_history = []  # list of (name, args_json) tuples

    while True:
        iteration_count += 1
        if iteration_count > MAX_CHAT_ITERATIONS:
            msg = f"I am sorry 😊. Could you try to rephrase or provide additional details?"
            if chat_ui:
                chat_ui.add_log(f"Chat loop exceeded {MAX_CHAT_ITERATIONS} iterations, stopping.", level="warning")
            elif verbose:
                print(f"Chat loop exceeded {MAX_CHAT_ITERATIONS} iterations, stopping.")
            return msg

        _strip_old_images(messages)

        try:
            if not safety_queue.empty():
                logger.warning("Safety queue triggered before API call, exiting chat loop.")
                return None

            completion_kwargs = dict(
                model=model,
                messages=messages,
                stream=stream,
                tool_choice="auto",          # never "required"
                temperature=0.6,             # official recommendation
                top_p=0.95,
                max_tokens=max_tokens,             # cap to prevent runaway generation
                extra_body={
                    "top_k": 20,             # vLLM extension, important for Qwen3
                    "repetition_penalty": 1.05,  # helps break repetition loops
                    "chat_template_kwargs": {"enable_thinking": think},  # if not using CoT
                },
            )
            if tools: # and not images_bytes:  # vLLM doesn't support tools + images in the same message, so only include tools if no images are present
                completion_kwargs["tools"] = tools
                
            chat_completion = await client.chat.completions.create(**completion_kwargs)

            # Streaming path: iterate chunks, populate shared variables
            if stream:
                _full_content = ""
                _full_reasoning = ""
                _full_tool_calls: dict = {}  # index -> {id, name, arguments}
                _ui_streaming = False
                _in_think = think  # True if we expect <think>...</think> in delta.content

                async for chunk in chat_completion:
                    if not safety_queue.empty():
                        if _ui_streaming and chat_ui:
                            chat_ui.stream_end()
                        return None
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    # Accumulate structured tool-call deltas
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in _full_tool_calls:
                                _full_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc.id:
                                _full_tool_calls[idx]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    _full_tool_calls[idx]["name"] += tc.function.name
                                if tc.function.arguments:
                                    _full_tool_calls[idx]["arguments"] += tc.function.arguments

                    # vLLM/OpenAI reasoning_content: thinking tokens in a dedicated field
                    reasoning_tok = getattr(delta, 'reasoning_content', None)
                    if reasoning_tok and not _full_tool_calls:
                        _full_reasoning += reasoning_tok
                        if chat_ui:
                            if not _ui_streaming:
                                chat_ui.stream_start()
                                _ui_streaming = True
                            chat_ui.stream_think_token(reasoning_tok)

                    # Stream content (answer) tokens to UI
                    if delta.content:
                        token = delta.content
                        _full_content += token
                        if chat_ui and not _full_tool_calls:
                            if _in_think:
                                # Three cases:
                                # 1. vLLM sends thinking via reasoning_content → no <think> in content
                                # 2. Model chose not to think → no <think> in content
                                # 3. Model embeds <think>...</think> inside content
                                if "<think>" not in _full_content:
                                    # Cases 1 & 2 — stream answer directly
                                    _in_think = False
                                    if not _ui_streaming:
                                        chat_ui.stream_start()
                                        _ui_streaming = True
                                    chat_ui.stream_token(token)
                                elif "</think>" in _full_content:
                                    # Case 3 end: think block closed, stream the answer part
                                    _in_think = False
                                    post_think = _full_content.split("</think>", 1)[1]
                                    if post_think:
                                        if not _ui_streaming:
                                            chat_ui.stream_start()
                                            _ui_streaming = True
                                        chat_ui.stream_token(post_think)
                                else:
                                    # Case 3 mid: inside inline <think> block
                                    if not _ui_streaming:
                                        chat_ui.stream_start()
                                        _ui_streaming = True
                                    chat_ui.stream_think_token(token.replace("<think>", ""))
                            else:
                                if not _ui_streaming:
                                    chat_ui.stream_start()
                                    _ui_streaming = True
                                chat_ui.stream_token(token)

                if _ui_streaming and chat_ui:
                    chat_ui.stream_end()

                # Reconstruct into unified variables
                if _full_tool_calls:
                    _tool_call_objs = [
                        types.SimpleNamespace(
                            id=v["id"],
                            function=types.SimpleNamespace(name=v["name"], arguments=v["arguments"])
                        )
                        for v in _full_tool_calls.values()
                    ]
                    _content = None
                    _tool_calls = _tool_call_objs
                    _message_for_history = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": v["id"], "type": "function",
                             "function": {"name": v["name"], "arguments": v["arguments"]}}
                            for v in _full_tool_calls.values()
                        ]
                    }
                else:
                    _content = _full_content
                    _tool_calls = None
                    _message_for_history = {"role": "assistant", "content": _full_content}

            await asyncio.sleep(0.1)
            if not safety_queue.empty():
                logger.warning("Safety queue triggered after API call, exiting chat loop.")
                return None
        except APITimeoutError as e:
            error_message = f"Request to {host} timed out after {timeout} seconds."
            logger.error(error_message)
            if chat_ui:
                chat_ui.add_log(error_message, level="error")
            elif verbose:
                print(error_message)
            return None
        except OpenAIError as e:
            error_message = f"Error communicating with {host}: {e}."
            logger.error(error_message)
            if chat_ui:
                chat_ui.add_log(error_message, level="warning")
            elif verbose:
                print(error_message)
            return None
        except Exception as e:
            error_message = f"Unexpected error: {e}"
            logger.error(error_message)
            if chat_ui:
                chat_ui.add_log(error_message, level="error")
            elif verbose:
                print(error_message)
            return None

        # Non-streaming: extract from response object into unified variables
        if not stream:
            _msg = chat_completion.choices[0].message
            _content = _msg.content
            _tool_calls = _msg.tool_calls if _msg.tool_calls else None
            _message_for_history = _msg

        tool_calls = _tool_calls
        if tool_calls is None or len(tool_calls) == 0:
            last_response = _content
            # Check if the model returned a tool call as raw JSON in the content
            raw_tool = _parse_tool_call_from_content(last_response, tool_registry)
            if raw_tool:
                function_name = raw_tool["name"]
                function_arguments = raw_tool["arguments"]
                synthetic_id = f"call_{uuid.uuid4().hex[:24]}"
                if chat_ui:
                    chat_ui.add_tool_call(function_name, function_arguments)
                    chat_ui.show_tool_start(function_name, function_arguments)
                    chat_ui.start_tool_spinner(function_name, function_arguments)
                elif verbose:
                    print(f"{function_name}({function_arguments})")
                messages.append({"role": "assistant", "content": last_response})
                for tool_name in tool_registry.tools:
                    if tool_name == function_name:
                        try:
                            tool_handler = tool_registry[tool_name]
                            try:
                                tool_response = await asyncio.wait_for(
                                    tool_handler(**function_arguments),
                                    timeout=timeout,
                                )
                            except asyncio.TimeoutError:
                                tool_response = f"- tool call timed out after {timeout} seconds. Tool might have succeeded but no response was received. Check expected output."
                                if chat_ui:
                                    chat_ui.add_log(f"{function_name} timed out after {timeout}s", level="warning")
                                elif verbose:
                                    print(f"{function_name} timed out after {timeout}s")
                            tool_response = "" if tool_response is None else str(tool_response)
                            _vision_b64, _vision_mime = None, None
                            if data_path and "file_data_base64" in tool_response:
                                tool_response, _vision_b64, _vision_mime = _extract_base64_file(tool_response, data_path)
                            if _vision_b64:
                                tool_content = [
                                    {"type": "text", "text": tool_response},
                                    {"type": "image_url", "image_url": {"url": f"data:{_vision_mime};base64,{_vision_b64}"}},
                                ]
                            else:
                                tool_content = tool_response
                            tool_message = {'role': 'tool', 'content': tool_content, 'name': function_name, 'parameters': function_arguments, "tool_call_id": synthetic_id}
                            messages.append(tool_message)
                            if chat_ui:
                                chat_ui.add_tool_result(function_name, tool_response)
                            elif verbose:
                                truncated = tool_response[:500] + "..." if len(tool_response) > 500 else tool_response
                                print(f"{function_name}({function_arguments}) returned: {truncated}")
                        except Exception as e:
                            if chat_ui:
                                chat_ui.add_log(f"{function_name}({function_arguments}) error: {e}", level="error")
                            elif verbose:
                                print(f"{function_name}({function_arguments}) encountered an error: {e}")
                            tool_message = {'role': 'tool', 'content': f'Error: {e}', 'name': function_name, 'parameters': function_arguments, "tool_call_id": synthetic_id}
                            messages.append(tool_message)
                        break
                else:
                    # Tool not found in registry
                    tool_message = {'role': 'tool', 'content': f'Error: tool {function_name} not found', 'name': function_name, 'parameters': function_arguments, "tool_call_id": synthetic_id}
                    messages.append(tool_message)
                # Check for repeated tool calls
                call_key = (function_name, json.dumps(function_arguments, sort_keys=True))
                tool_call_history.append(call_key)
                if tool_call_history.count(call_key) >= MAX_REPEATED_TOOL_CALLS:
                    msg = f"I am sorry 😊. Could you try to rephrase or provide additional details?"
                    if chat_ui:
                        chat_ui.add_log(f"Repeated tool call detected: {function_name} called {tool_call_history.count(call_key)} times with same args", level="warning")
                    elif verbose:
                        print(f"Repeated tool call detected: {function_name} called {tool_call_history.count(call_key)} times with same args")
                    return msg
                continue  # loop back for the model to generate the final response

            # Guard against returning raw tool-call JSON to the user.
            # If the content looks like a tool call but couldn't be parsed,
            # ask the model to retry without tools.
            if _looks_like_raw_tool_call(last_response):
                if chat_ui:
                    chat_ui.add_log("Model returned unparseable raw tool-call JSON, retrying without tools.", level="warning")
                elif verbose:
                    print("Model returned unparseable raw tool-call JSON, retrying without tools.")
                messages.append({"role": "assistant", "content": last_response})
                messages.append({"role": "user", "content": "Please provide your answer as plain text, not as a JSON tool call."})
                continue

            if "</think>" in last_response:
                last_response = last_response.split("</think>")[1]
            # Fallback: if delta.content was empty but reasoning_content had the answer,
            # surface the reasoning so the user gets a non-empty reply.
            if not last_response or not last_response.strip():
                if _full_reasoning and _full_reasoning.strip():
                    last_response = _full_reasoning.strip()
                elif _full_content and "<think>" in _full_content:
                    # Model put entire answer inside inline <think> tags
                    think_body = _full_content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
                    if think_body:
                        last_response = think_body
            return last_response

        messages.append(_message_for_history)
        for tool in tool_calls:
            await asyncio.sleep(0.1)
            if not safety_queue.empty():
                if verbose:
                    print("Safety queue triggered, exiting chat loop.")
                return None
            function_name = tool.function.name
            function_arguments = json.loads(tool.function.arguments)
            if chat_ui:
                chat_ui.add_tool_call(function_name, function_arguments)
                chat_ui.show_tool_start(function_name, function_arguments)
                chat_ui.start_tool_spinner(function_name, function_arguments)
            elif verbose:
                print(f"{function_name}({function_arguments})")
            # Ensure the function is available, and then call it
            # FIXME: Possible that 2 or more tools have the same name?
            for tool_name in tool_registry.tools:
                if tool_name == function_name:
                    try:
                        tool_handler = tool_registry[tool_name]
                        try:
                            tool_response = await asyncio.wait_for(
                                tool_handler(**function_arguments),
                                timeout=timeout,
                            )
                        except asyncio.TimeoutError:
                            tool_response = f"- tool call timed out after {timeout} seconds. Tool might have succeeded but no response was received. Check expected output."
                            if chat_ui:
                                chat_ui.add_log(f"{function_name} timed out after {timeout}s", level="warning")
                            elif verbose:
                                print(f"{function_name} timed out after {timeout}s")
                        if chat_ui:
                            chat_ui.stop_tool_spinner()
                        tool_response = "" if tool_response is None else str(tool_response)
                        # Extract base64 file data from tool response and save to disk
                        _vision_b64, _vision_mime = None, None
                        if data_path and "file_data_base64" in tool_response:
                            tool_response, _vision_b64, _vision_mime = _extract_base64_file(tool_response, data_path)
                        if _vision_b64:
                            tool_content = [
                                {"type": "text", "text": tool_response},
                                {"type": "image_url", "image_url": {"url": f"data:{_vision_mime};base64,{_vision_b64}"}},
                            ]
                        else:
                            tool_content = tool_response
                        tool_message = {'role': 'tool', 'content': tool_content, 'name': tool.function.name, 'parameters': function_arguments, "tool_call_id": tool.id,}
                        messages.append(tool_message)

                        if chat_ui:
                            chat_ui.add_tool_result(function_name, tool_response)
                            chat_ui.show_tool_done(function_name, tool_response)
                        elif verbose:
                            truncated = tool_response[:500] + "..." if len(tool_response) > 500 else tool_response
                            print(f"{function_name} returned: {truncated}")
                    except Exception as e:
                        if chat_ui:
                            chat_ui.stop_tool_spinner()
                            chat_ui.show_tool_done(function_name, str(e), success=False)
                            chat_ui.add_log(f"{tool_name} error: {e}", level="error")
                        elif verbose:
                            print(f"{tool_name} encountered an error: {e}")
                        tool_message = {'role': 'tool', 'content': f'Error: {e}', 'name': tool.function.name, 'parameters': function_arguments, "tool_call_id": tool.id}
                        messages.append(tool_message)
                    break
            else:
                # Tool not found in registry
                tool_message = {'role': 'tool', 'content': f'Error: tool {function_name} not found', 'name': function_name, 'parameters': function_arguments, "tool_call_id": tool.id}
                messages.append(tool_message)
            # Check for repeated tool calls
            call_key = (function_name, json.dumps(function_arguments, sort_keys=True))
            tool_call_history.append(call_key)
            if tool_call_history.count(call_key) >= MAX_REPEATED_TOOL_CALLS:
                msg = f"I am sorry 😊. Could you try to rephrase or provide additional details?"
                if chat_ui:
                    chat_ui.add_log(f"Repeated tool call detected: {function_name} called {tool_call_history.count(call_key)} times with same args", level="warning")
                elif verbose:
                    print(f"Repeated tool call detected: {function_name} called {tool_call_history.count(call_key)} times with same args")
                return msg
