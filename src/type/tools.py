'''
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

ToolRegistry - where all tools are registered and managed.
RequestHandler - abstract base class for all tools.
ToolHandler - concrete implementation of RequestHandler that calls a tool via FastMCP client.
'''

import tempfile
import os
import base64
import mimetypes
import wave
import random
from fastmcp import Client
from abc import ABC, abstractmethod
from mcp.types import ImageContent, TextContent, AudioContent
from typing import TypedDict

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Default WAV parameters — used when AudioContent metadata does not specify them.
# These defaults match common speech audio: mono channel, 16-bit samples, 16 kHz sample rate.
_DEFAULT_WAV_CHANNELS = 1
_DEFAULT_WAV_SAMPLE_WIDTH = 2
_DEFAULT_WAV_FRAME_RATE = 16000

# Map MIME types to file extensions for audio content
_MIME_TO_EXT: dict[str, str] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "audio/aac": "aac",
}


class FunctionSpec(TypedDict):
    name: str
    description: str
    parameters: dict[str, object]
    returns: dict[str, object]


class ToolItem(TypedDict):
    type: str
    function: FunctionSpec


def get_tools(tool_registry: 'ToolRegistry') -> list[ToolItem]:
    """Get the tools from the tool registry"""
    tools: list[ToolItem] = []
    for key in tool_registry.handlers:
        tools.append(tool_registry[key].get_tool())
    return tools

class RequestHandler(ABC):
    def __init__(self,
                 url: str | None = None,
                 tool_item: ToolItem | None = None,
                 **kwargs: object):
        super().__init__()
        self.url = url
        self.tool_item = tool_item
        self.kwargs = kwargs

    @abstractmethod
    async def __call__(self, **kwargs: object) -> str | None:
        pass

    def get_tool(self) -> ToolItem | None:
        return self.tool_item


class ToolHandler(RequestHandler):
    def __init__(self,
                 url: str | None = None,
                 tool_item: ToolItem | None = None,
                 **kwargs: object):
        super().__init__(url=url, tool_item=tool_item, **kwargs)

    async def __call__(self, **kwargs: object) -> str | None:
        # if there is an image in the kwargs, convert it to base64
        media_types = ['images', 'audios']
        for media_type in media_types:
            if media_type in kwargs:
                if isinstance(kwargs[media_type], list) and len(kwargs[media_type]) == 0:
                    if media_type == 'images':
                        return f"No {media_type} provided. For images, use the camera tool to capture an image first."
                    else:
                        return f"No {media_type} provided. For audios, use the microphone tool to record an audio first."
                kwargs[media_type] = kwargs[media_type][0] if isinstance(kwargs[media_type], list) else kwargs[media_type]
                if isinstance(kwargs[media_type], str):
                    # check if the path is a valid file path
                    if not os.path.exists(kwargs[media_type]):
                        # try removing leading dot
                        kwargs[media_type] = kwargs[media_type].lstrip('.')
                        if not os.path.exists(kwargs[media_type]):
                            logger.error(f"File not found: {kwargs[media_type]}")
                            return f"File not found: {kwargs[media_type]}"
                    with open(kwargs[media_type], 'rb') as image_file:
                        kwargs[media_type] = [base64.b64encode(image_file.read()).decode('utf-8')]
                elif isinstance(kwargs[media_type], bytes):
                    # if the image is already in bytes, convert to base64
                    kwargs[media_type] = [base64.b64encode(kwargs[media_type]).decode('utf-8')]
                # if dictionary, extract the value and encode it
                elif isinstance(kwargs[media_type], dict):
                    for key, value in kwargs[media_type].items():
                        if isinstance(value, str):
                            with open(value, 'rb') as image_file:
                                kwargs[media_type] = [base64.b64encode(image_file.read()).decode('utf-8')]
                        elif isinstance(value, bytes):
                            kwargs[media_type] = [base64.b64encode(value).decode('utf-8')]
                        else:
                            logger.error(f"Unsupported image type: {type(value)}")
                            return "Unsupported image type provided."
                else:
                    logger.error(f"Unsupported images type: {type(kwargs[media_type])}")
                    return f"Unsupported {media_type} type provided. Please provide a list of base64-encoded {media_types} or file paths."


        async with Client(self.url) as client:
            keys_kwargs = list(kwargs.keys())
            logger.info(f"Calling tool: {self.tool_item['function']['name']} with arguments: {keys_kwargs}")
            tool_response = await client.call_tool(self.tool_item['function']['name'], kwargs)

        if isinstance(tool_response, str):
            return tool_response

        content = tool_response.content
        content = content[0] if isinstance(content, list) else content

        # MCP data types: https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/types.py
        if isinstance(content, ImageContent):
            image_data = base64.b64decode(content.data)
            mime_ext = _mime_to_extension(content.mimeType) if content.mimeType else "png"
            suffix = f".{mime_ext}"
            fd, image_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(image_data)
            return image_path
        elif isinstance(content, TextContent):
            return content.text
        elif isinstance(content, AudioContent):
            audio_data = content.data
            if len(audio_data) == 0:
                logger.warning("No audio data returned from the tool.")
                return None
            if isinstance(audio_data, str):
                logger.info(f"Audio data is a base64 string of length {len(audio_data)}")

            # Detect format from mimeType, falling back to 'wav'
            audio_format = "wav"
            if hasattr(content, 'mimeType') and content.mimeType:
                audio_format = _mime_to_extension(content.mimeType)
            elif hasattr(content, 'format') and content.format:
                audio_format = content.format

            audio_data = base64.b64decode(content.data)

            meta: dict[str, object] = content.metadata if hasattr(content, 'metadata') else {}
            logger.info(f"Audio data format: {audio_format}, metadata: {meta}")

            suffix = f".{audio_format}"
            fd, audio_path = tempfile.mkstemp(suffix=suffix)
            try:
                os.close(fd)
                # WAV parameters sourced from metadata when available, otherwise defaults.
                channels = int(meta.get('channels', _DEFAULT_WAV_CHANNELS)) if meta else _DEFAULT_WAV_CHANNELS
                sample_width = int(meta.get('sample_width', _DEFAULT_WAV_SAMPLE_WIDTH)) if meta else _DEFAULT_WAV_SAMPLE_WIDTH
                frame_rate = int(meta.get('frame_rate', _DEFAULT_WAV_FRAME_RATE)) if meta else _DEFAULT_WAV_FRAME_RATE
                with wave.open(audio_path, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(frame_rate)
                    wf.writeframes(audio_data)
            except Exception:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                raise
            return audio_path

        return "Undefined content type returned from the tool."


def _mime_to_extension(mime_type: str) -> str:
    """Convert a MIME type string to a file extension (without dot).

    Uses the lookup table first, then falls back to the mimetypes stdlib module.
    Returns a sensible default ('bin') when the MIME type is unrecognised.
    """
    if mime_type in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime_type]
    ext = mimetypes.guess_extension(mime_type)
    if ext:
        return ext.lstrip('.')
    return 'bin'


# create a tool registry
class ToolRegistry:
    def __init__(self) -> None:
        self.tools: set[str] = set()
        self.urls: dict[str, list[str]] = {}
        self.handlers: dict[str, ToolHandler] = {}

    def register(self, tool: ToolHandler) -> None:
        # many need to modify to support one to many tools
        # self.tools[tool.tool_item['function']['name']] = tool

        tool_name: str = tool.tool_item['function']['name']
        tool_url: str = tool.url
        self.tools.add(tool_name)
        if tool_name not in self.urls:
            self.urls[tool_name] = [tool_url]
        else:
            self.urls[tool_name].append(tool_url)

        key = f"{tool_name}@{tool_url}"
        self.handlers[key] = tool

    def get_url(self, tool_name: str) -> str | None:
        """Get a random URL for the tool"""
        if tool_name not in self.urls:
            return None
        return random.choice(self.urls[tool_name])

    def get_tool_items(self) -> list[ToolItem]:
        """Get the tool items from the tool registry"""
        tool_items: list[ToolItem] = []
        for key in self.handlers:
            tool_items.append(self.handlers[key].get_tool())
        return tool_items

    def get_handler_by(self, tool_name: str, url: str) -> ToolHandler | None:
        """Get a tool handler by tool name and URL"""
        if tool_name is None or url is None:
            return None

        key = f"{tool_name}@{url}"
        if key in self.handlers:
            return self.handlers[key]
        return None


    def __getitem__(self, tool_name: str) -> ToolHandler | None:
        if tool_name not in self.tools:
            return None
        url = random.choice(self.urls[tool_name])
        return self.handlers[f"{tool_name}@{url}"]

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self):
        return iter(self.tools)
