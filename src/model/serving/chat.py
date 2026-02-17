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
import os
import json
import uuid
from openai import AsyncOpenAI, OpenAIError, APITimeoutError
from typing import List, Optional, Any


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
    # Find the matching closing brace
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        obj = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
        if obj["name"] in tool_registry.tools:
            return obj
    return None


async def chat(host: str = "http://127.0.0.1:8001/v1",
         host_key: str = "EMPTY",
         model: str = "Qwen/Qwen3-8B",
         instruction: str = "Tell me more about yourself.",
         images: List[str]|str = None,
         tool_registry: Optional[Any] = None,
         timeout: int = None,
         stream: bool = False,
         think: bool = True,
         safety_queue: Optional[asyncio.Queue] = None,
         **kwargs) -> Optional[str]:

    tools = tool_registry.get_tool_items() if tool_registry else []
    chat_ui = kwargs['chat_ui'] if 'chat_ui' in kwargs else None
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    max_tokens = kwargs.get('max_tokens', 262144)
    memories = kwargs.get('memories', None)
    prompt_intro = kwargs.get('prompt_intro', "I am a helpful AI assistant. My name is OnIt.")

    images_bytes = []
    if isinstance(images, list):
        for image_path in images:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as image_file:
                    images_bytes.append(base64.b64encode(image_file.read()).decode('utf-8'))
            else:
                if chat_ui and verbose:
                    chat_ui.add_log(f"Image file {image_path} not found, proceeding without this image.", level="warning")
    elif isinstance(images, str):
        image_path = images
        if os.path.exists(image_path):
            with open(image_path, 'rb') as image_file:
                images_bytes = [base64.b64encode(image_file.read()).decode('utf-8')]
        else:
            if chat_ui and verbose:
                chat_ui.add_log(f"Image file {image_path} not found, proceeding without this image.", level="warning")

    session_history = kwargs.get('session_history', None)

    # FIXME: see do_image_task() correct API use for messages with images
    if memories:
        assistant_prompt = f"{prompt_intro} Answer the question based on query and memories.\nMemories:\n{memories}\n"
        messages = [{"role": "assistant", "content": assistant_prompt}]
    else:
        messages = [{"role": "assistant", "content": prompt_intro}]

    # inject session history as prior conversation turns
    if session_history:
        for entry in session_history:
            messages.append({"role": "user", "content": entry["task"]})
            messages.append({"role": "assistant", "content": entry["response"]})

    if images_bytes:
        messages.append({
            "role": "system", 
            "content": (
                "You are an expert vision-language assistant. Your task is to analyze images with high precision, "
                "reasoning step-by-step about visual elements and their spatial relationships (e.g., coordinates, "
                "relative positions like left/right/center). Always verify visual evidence before concluding. "
                "If a task requires external data, calculation, or specific actions beyond visual description, "
                "use the provided tools. Be concise, objective, and format your tool calls strictly according to schema."
            )})
        messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images_bytes[0]}"}}
                    ],})
    else:
        messages.append({"role": "user", "content": instruction})
        
    if not memories and not session_history:
        message = {'role': 'tool', 'content': '', 'name': '', 'parameters': {}, "tool_call_id": ''}
        messages.append(message)

    api_key = _resolve_api_key(host, host_key)
    client = AsyncOpenAI(base_url=host, api_key=api_key)

    if chat_ui:
        chat_ui.add_log(f"Starting chat with model: {model}", level="info")
    while True:
        try:
            if not safety_queue.empty():
                return None

            completion_kwargs = dict(
                model=model,
                messages=messages,
                stream=stream,
                timeout=60,
            )
            if tools: # and not images_bytes:  # vLLM doesn't support tools + images in the same message, so only include tools if no images are present
                completion_kwargs["tools"] = tools
            chat_completion = await client.chat.completions.create(**completion_kwargs)

            await asyncio.sleep(0.1)
            if not safety_queue.empty():
                return None
        except APITimeoutError as e:
            error_message = f"Request to {host} timed out after {timeout} seconds."
            if chat_ui and verbose:
                chat_ui.add_log(error_message, level="error")
            elif verbose:
                print(error_message)
            return None
        except OpenAIError as e:
            error_message = f"Error communicating with {host}: {e}."
            if chat_ui and verbose:
                chat_ui.add_log(error_message, level="warning")
            elif verbose:
                print(error_message)
            return None
        except Exception as e:
            error_message = f"Unexpected error: {e}"
            if chat_ui and verbose:
                chat_ui.add_log(error_message, level="error")
            elif verbose:
                print(error_message)
            return None
            
        tool_calls = chat_completion.choices[0].message.tool_calls
        if tool_calls is None or len(tool_calls) == 0:
            last_response = chat_completion.choices[0].message.content
            # Check if the model returned a tool call as raw JSON in the content
            raw_tool = _parse_tool_call_from_content(last_response, tool_registry)
            if raw_tool:
                function_name = raw_tool["name"]
                function_arguments = raw_tool["arguments"]
                synthetic_id = f"call_{uuid.uuid4().hex[:24]}"
                if chat_ui:
                    chat_ui.add_log(f"Calling: {function_name} (parsed from content)", level="info")
                    chat_ui.render()
                else:
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
                            tool_response = "" if tool_response is None else str(tool_response)
                            tool_message = {'role': 'tool', 'content': tool_response, 'name': function_name, 'parameters': function_arguments, "tool_call_id": synthetic_id}
                            messages.append(tool_message)
                            truncated = tool_response[:200] + "..." if len(tool_response) > 200 else tool_response
                            if chat_ui:
                                chat_ui.add_log(f"{function_name} returned: {truncated}", level="debug")
                        except Exception as e:
                            if chat_ui and verbose:
                                chat_ui.add_log(f"{function_name} error: {e}", level="error")
                            elif verbose:
                                print(f"{function_name} encountered an error: {e}")
                        break
                continue  # loop back for the model to generate the final response

            if "</think>" in last_response:
                last_response = last_response.split("</think>")[1]
            return last_response

        messages.append(chat_completion.choices[0].message)
        for tool in tool_calls:
            await asyncio.sleep(0.1)
            if not safety_queue.empty():
                return None
            function_name = tool.function.name
            function_arguments = json.loads(tool.function.arguments)
            if chat_ui:
                chat_ui.add_log(f"Calling: {function_name}", level="info")
                chat_ui.render()
            else:
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
                        tool_response = "" if tool_response is None else str(tool_response)
                        tool_message = {'role': 'tool', 'content': tool_response, 'name': tool.function.name, 'parameters': function_arguments, "tool_call_id": tool.id,}
                        messages.append(tool_message)

                        # Log tool response (truncated for display)
                        truncated_response = tool_response[:200] + "..." if len(tool_response) > 200 else tool_response
                        if chat_ui:
                            chat_ui.add_log(f"{function_name} returned: {truncated_response}", level="debug")
                    except Exception as e:
                        if chat_ui and verbose:
                            chat_ui.add_log(f"{tool_name} error: {e}", level="error")
                        elif verbose:
                            print(f"{tool_name} encountered an error: {e}")
                        continue
                    break
