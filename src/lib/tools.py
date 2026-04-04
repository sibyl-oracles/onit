'''

Tool related utility functions for On-it agent.

'''

import asyncio
import logging
import os
import sys
from typing import Any, Optional
from urllib.parse import urlparse

from fastmcp import Client
from rich.status import Status

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from type.tools import ToolHandler, ToolRegistry

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _build_parameters(tool_item: Any) -> dict:
    """Extract parameter schema from a tool or resource item.

    Args:
        tool_item: An MCP tool or resource object.

    Returns:
        A JSON-schema-style dict describing the tool's parameters.
    """
    if hasattr(tool_item, 'inputSchema'):
        return {
            'type': 'object',
            'properties': tool_item.inputSchema['properties'],
        }

    if hasattr(tool_item, 'arguments') and tool_item.arguments:
        properties: dict[str, dict[str, str]] = {}
        for arg in tool_item.arguments:
            prop: dict[str, str] = {'type': 'string'}
            if arg.description:
                prop['description'] = arg.description
            properties[arg.name] = prop
        return {
            'type': 'object',
            'properties': properties,
        }

    # Fallback: serialize all attributes
    parameters: dict[str, Any] = {}
    for attr, value in tool_item.__dict__.items():
        if hasattr(value, 'model_dump'):
            parameters[attr] = value.model_dump()
        elif isinstance(value, list):
            parameters[attr] = [
                v.model_dump() if hasattr(v, 'model_dump') else v
                for v in value
            ]
        else:
            parameters[attr] = value
    return parameters


def _build_returns(tool_item: Any) -> dict:
    """Extract the output/return schema from a tool item.

    Args:
        tool_item: An MCP tool or resource object.

    Returns:
        A dict describing the tool's return schema properties.
    """
    output_schema = getattr(tool_item, 'outputSchema', {})
    if output_schema and 'properties' in output_schema:
        return output_schema['properties']
    return {}


async def _wait_for_port(host: str, port: int, timeout: float = 30.0, poll_interval: float = 0.5) -> bool:
    """Poll a TCP port until it accepts connections or the timeout expires.

    Args:
        host: Hostname or IP address to connect to.
        port: TCP port number.
        timeout: Maximum seconds to wait before giving up.
        poll_interval: Seconds between connection attempts.

    Returns:
        True if the port became reachable within the timeout, False otherwise.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=1.0,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            await asyncio.sleep(poll_interval)
    return False


async def _discover_server_tools(
    server: dict,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
) -> list[ToolHandler]:
    """Discover tools from a single MCP server.

    Args:
        server: MCP server configuration dict.
        max_retries: Maximum number of connection retry attempts.
        base_delay: Initial delay in seconds (doubles each attempt, capped at max_delay).
        max_delay: Maximum delay in seconds between retries.

    Returns:
        List of ToolHandler instances discovered from this server.
    """
    import random

    name = server.get('name', 'Unknown')
    enabled = server.get('enabled', True)
    if not enabled:
        logger.info("[discover_tools] Skipping disabled MCP server: %s", name)
        return []

    url: Optional[str] = server.get('url', None)
    if not url:
        logger.error("[discover_tools] MCP server URL is not provided.")
        return []

    logger.info("[discover_tools] Discovering tools from MCP server: %s", url)

    # Wait for the server's TCP port to accept connections before attempting
    # MCP protocol discovery.  This eliminates spurious connection-refused
    # warnings that occur when discovery races ahead of server startup.
    parsed = urlparse(url)
    host = parsed.hostname or '127.0.0.1'
    port = parsed.port or 80
    if not await _wait_for_port(host, port):
        logger.error(
            "[discover_tools] %s did not become ready at %s:%d within timeout",
            name, host, port,
        )
        return []

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            async with Client(url) as client:
                tools_list = await client.list_tools()
                resources_list = await client.list_resources()
                tools_list.extend(resources_list)

                # Retry if server connected but returned no tools — it may still be initializing
                if not tools_list:
                    raise ValueError(f"Server {name} returned empty tool list (may still be initializing)")

                handlers: list[ToolHandler] = []
                for tool_item in tools_list:
                    parameters = _build_parameters(tool_item)
                    returns = _build_returns(tool_item)

                    tool_entry: dict[str, Any] = {
                        'type': 'function',
                        'function': {
                            'name': tool_item.name,
                            'description': tool_item.description,
                            'parameters': parameters,
                            'returns': returns,
                        }
                    }
                    handler = ToolHandler(url=url, tool_item=tool_entry)
                    handlers.append(handler)
                    logger.info("[discover_tools] %s", tool_entry)

                logger.info("[discover_tools] %s", tools_list)
                return handlers

        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                jitter = random.uniform(0, delay * 0.2)
                wait = delay + jitter
                logger.warning(
                    "[discover_tools] Attempt %d/%d failed for %s: %s. "
                    "Retrying in %.1fs...",
                    attempt, max_retries, name, exc, wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "[discover_tools] All %d attempts failed for %s: %s",
                    max_retries, name, last_error,
                )

    raise last_error  # type: ignore[misc]


async def discover_tools(mcp_servers: list[dict]) -> ToolRegistry:
    """
    Automatically discover tools from all MCP servers in parallel.

    Args:
        mcp_servers: a list of MCP server configurations.

    Returns:
        A registry of all discovered tools.
    """
    tool_registry = ToolRegistry()

    # Discover tools from all servers concurrently
    results = await asyncio.gather(
        *[_discover_server_tools(server) for server in mcp_servers],
        return_exceptions=True
    )

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            name = mcp_servers[i].get('name', 'Unknown')
            logger.error("[discover_tools] Failed to discover tools from %s: %s", name, result)
            continue
        for handler in result:
            tool_registry.register(handler)

    logger.info("[discover_tools] Registered %d tools", len(tool_registry.tools))
    return tool_registry


async def listen(
    tool_registry: ToolRegistry,
    status: Optional[Status] = None,
    prompt_color: str = "bold dark_blue",
    is_safety_task: bool = False,
) -> Optional[str]:
    """Record audio from the microphone and transcribe it via ASR.

    The microphone MCP tool returns a temporary wav file path, which is then
    passed to the ASR tool for transcription.  The intermediate file is a
    limitation of the current MCP audio pipeline — both tools communicate
    via file paths rather than in-memory byte streams.  We clean up the temp
    file after ASR to avoid leaking disk space.

    Args:
        tool_registry: Registry containing microphone, ASR, TTS, and speaker tools.
        status: Optional Rich Status widget for progress feedback.
        prompt_color: Rich markup colour string for the status message.
        is_safety_task: When True, return an empty string immediately after recording.

    Returns:
        The transcribed text, an empty string for safety tasks, or None on failure.
    """
    microphone = tool_registry['microphone'] if 'microphone' in tool_registry.tools else None
    tts = tool_registry['speech_synthesis'] if 'speech_synthesis' in tool_registry.tools else None
    speaker = tool_registry['speaker'] if 'speaker' in tool_registry.tools else None
    asr = tool_registry['speech_recognition'] if 'speech_recognition' in tool_registry.tools else None

    if not microphone or not asr or not tts or not speaker:
        logger.error("[%s] Microphone, ASR, TTS or Speaker tool is not available.", listen.__name__)
        return None

    # Record audio — the microphone tool writes a temporary wav file and
    # returns its path.  This intermediate file is unavoidable given the
    # current MCP tool interface which communicates via file paths.
    wav_file: Optional[str] = await microphone()

    if is_safety_task:
        return ""

    if wav_file is None:
        wav_file = await tts(text="Sorry! I did not hear you. Please say it again.")
        await speaker(audios=[wav_file])
        return None

    if status is not None:
        status.update(f"[{prompt_color}] 🤔 Hold on...[/]")

    logger.info("[%s] wav file: %s", __name__, wav_file)

    message: Optional[str] = await asr(audios=[wav_file])

    # Clean up the intermediate wav file to avoid leaking temp disk space
    if wav_file:
        try:
            os.unlink(wav_file)
        except OSError:
            logger.debug("Could not remove temporary wav file: %s", wav_file, exc_info=True)

    return message
