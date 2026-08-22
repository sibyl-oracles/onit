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

import asyncio
import json
import tempfile
import os
import base64
import mimetypes
import wave
import random
import httpx
from fastmcp import Client
from abc import ABC, abstractmethod
from mcp.types import ImageContent, TextContent, AudioContent
from typing import Callable, TypedDict

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


# ── MCP connection pooling ──────────────────────────────────────────────────
#
# A tool call used to open its own client: a TCP connect, an SSE stream, an
# initialize handshake and a notifications/initialized round trip — all before
# the tool began doing anything, and all torn down the moment it finished. A
# research answer makes six to ten calls, so that setup was paid six to ten
# times per answer, in series, while the user waited.
#
# One client per (server URL, event loop) instead, opened on first use and kept
# open. MCP multiplexes concurrent requests over a single session by request
# id, so tool calls issued in parallel share it safely.

# How long to wait for a server's reply before giving up on the session.
#
# A pooled session can go half-dead: the SSE transport's writer task logs
# "Error in post_writer" and closes the write stream when a POST fails, but the
# reader keeps running, so nothing tears the session down and the client still
# reports itself connected.  Nothing can be sent on it again, yet fastmcp
# defaults to no request timeout — the call below would wait for a reply that
# can never arrive, hanging the answer instead of failing it.  Bounding the
# wait turns that into an error the retry can act on.
#
# Sized so a working tool never reaches it: commands that genuinely run long
# belong on the serve tool, which returns immediately and reports progress.
_REQUEST_TIMEOUT = 300.0

# How long the client may reuse an idle HTTP connection to an MCP server.
#
# This is the other half of the half-dead session described above, and the
# reason one keeps appearing.  httpx expires a pooled connection after 5s and
# uvicorn closes an idle keep-alive connection after 5s, so the two timers fire
# at the same mark: when the gap between two tool calls lands near five seconds
# — an ordinary model turn — the client hands out a connection it still counts
# as fresh while the server has already sent a FIN for it.  The POST goes into
# that socket, nothing comes back, and the SDK's post_writer dies with
# "httpx.ReadError" and closes the session's write stream for good.
#
# Expiring our end well before any server's timer removes the overlap.  OnIt's
# own servers hold theirs open for KEEPALIVE_TIMEOUT (see
# src/mcp/servers/tasks/shared.py), which covers third-party servers left on
# uvicorn's 5s default too.  A reconnect costs a localhost TCP handshake; a
# lost session costs the answer.
_KEEPALIVE_EXPIRY = 2.0

# How often a call in flight checks that its session can still be written to.
#
# The request timeout above is a backstop, not a recovery: a call that was in
# flight when the write stream closed waits out the full five minutes before
# anything notices.  A ping on a closed write stream raises at once, so asking
# for one periodically turns that wait into a retry within seconds.  A ping on
# a *busy* server simply doesn't come back, and that is the correct reading —
# a server working on a long tool is not a broken one.
_HEALTH_INTERVAL = 15.0

# How long an event stream may go quiet before the client gives up on it.
#
# The SDK reads the SSE stream with httpx's read timeout set to
# sse_read_timeout, 300s by default, and that clock measures silence on the
# stream — not on the session.  A pooled client outlives many turns, so the gap
# between two tool calls is however long the user takes to type the next
# message; on a server that sends no keepalives, five quiet minutes ends the
# stream.  The SDK's reader then logs its whole stack at ERROR and pushes the
# exception into the read stream, which kills a session that was working.
#
# Long enough here that only a genuinely abandoned connection reaches it.  The
# liveness this timeout was standing in for is already covered better: a call
# in flight is bounded by _REQUEST_TIMEOUT and watched by the ping above, and a
# connection that died while idle surfaces as a write error on the next call,
# which the retry handles.  POSTs keep the shorter timeout the SDK asked for —
# only the stream gets this one.
_STREAM_READ_TIMEOUT = 86400.0


class _StreamingReadTimeoutClient(httpx.AsyncClient):
    """An httpx client that lets streamed responses stay quiet far longer.

    One client serves both halves of an MCP HTTP transport: short POSTs that
    must fail fast when a server stops answering, and a long-lived event stream
    whose whole job is to sit idle between messages.  They need different read
    timeouts, and httpx only has one per client — so the streaming request gets
    its own via the per-request extension.
    """

    async def send(self, request: httpx.Request, *, stream: bool = False,
                   **kwargs: object) -> httpx.Response:
        if stream:
            timeout = dict(request.extensions.get("timeout") or {})
            timeout["read"] = _STREAM_READ_TIMEOUT
            request.extensions = {**request.extensions, "timeout": timeout}
        return await super().send(request, stream=stream, **kwargs)


def _http_client_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """The httpx client MCP transports use, with connection reuse bounded.

    Mirrors the SDK's own defaults (redirects followed, 30s connect / 300s
    read) and adds two things the defaults get wrong for a pooled session: the
    keepalive expiry that keeps a POST off a connection the server is about to
    close, and a read timeout for streamed responses that outlasts an idle
    session (see _STREAM_READ_TIMEOUT).
    """
    return _StreamingReadTimeoutClient(
        follow_redirects=True,
        timeout=timeout if timeout is not None else httpx.Timeout(30.0, read=300.0),
        headers=headers or {},
        auth=auth,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20,
                            keepalive_expiry=_KEEPALIVE_EXPIRY),
    )


# Launch specs for MCP servers this process runs over stdio, keyed by the
# ``stdio://<name>`` pseudo-URL that stands in for them everywhere a server is
# identified by URL — the pooled client, the tool registry's per-tool URL
# lists, collision reporting. Keeping the identity a string means none of that
# machinery has to learn what a transport is.
STDIO_SCHEME = "stdio://"
_STDIO_SPECS: dict[str, dict] = {}


def stdio_url(name: str) -> str:
    """The pseudo-URL identifying the stdio server called ``name``."""
    return f"{STDIO_SCHEME}{name}"


def register_stdio_server(url: str, command: str, args: list[str],
                          env: dict[str, str] | None = None,
                          cwd: str | None = None,
                          log_file: str | None = None) -> None:
    """Record how to spawn the stdio server addressed by ``url``.

    ``env`` is merged over the current environment rather than replacing it.
    The MCP SDK hands a subprocess only HOME, LOGNAME, PATH, SHELL and USER
    when it is given an explicit environment, which would strip every
    ONIT_* setting and API key the tools rely on.
    """
    _STDIO_SPECS[url] = {
        "command": command,
        "args": list(args),
        "env": {**os.environ, **(env or {})},
        "cwd": cwd,
        "log_file": log_file,
    }


def is_stdio_url(url: object) -> bool:
    return isinstance(url, str) and url.startswith(STDIO_SCHEME)


def _transport_for(url: str):
    """The transport to connect to ``url`` with.

    Returns the URL unchanged — letting fastmcp infer the transport at its own
    connection defaults — for anything that is not an HTTP MCP server, or if
    this fastmcp cannot be told which httpx client to use.
    """
    spec = _STDIO_SPECS.get(url) if isinstance(url, str) else None
    if spec is not None:
        from pathlib import Path
        from fastmcp.client.transports import StdioTransport
        return StdioTransport(
            command=spec["command"],
            args=spec["args"],
            env=spec["env"],
            cwd=spec["cwd"],
            # One subprocess per OnIt process, held across the open/close
            # cycles of the pooled client rather than respawned per call.
            keep_alive=True,
            log_file=Path(spec["log_file"]) if spec["log_file"] else None,
        )
    if is_stdio_url(url):
        raise ValueError(
            f"No launch spec registered for {url}; call register_stdio_server() "
            f"before connecting.")
    if not isinstance(url, str) or not url.startswith("http"):
        return url
    try:
        from fastmcp.client.transports import (SSETransport,
                                               StreamableHttpTransport)
        # Matches fastmcp's own inference: a /sse endpoint is SSE, the rest is
        # streamable HTTP.
        cls = SSETransport if url.rstrip("/").endswith("/sse") else StreamableHttpTransport
        return cls(url, httpx_client_factory=_http_client_factory)
    except Exception:
        logger.debug("using inferred transport for %s", url, exc_info=True)
        return url


class _TransportNoise(logging.Filter):
    """Replace the SDK's transport tracebacks with one line each.

    Both halves of the SSE transport report a broken session by logging the
    whole stack at ERROR: post_writer when a POST fails, sse_reader when the
    event stream ends or goes quiet past its read timeout.  Either lands on the
    user's terminal looking like a crash, while the actual handling — discard
    the session, reconnect, retry — happens here and works.  Keep the cause,
    drop the wall of frames; the frames stay available at DEBUG.
    """

    # Prefix of the SDK's log line -> what it means for the session.
    _CAUSES = {
        "Error in post_writer": "MCP write channel closed",
        "Error in sse_reader": "MCP event stream closed",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for prefix, summary in self._CAUSES.items():
            if message.startswith(prefix):
                break
        else:
            return True
        exc = record.exc_info[1] if record.exc_info else None
        logger.warning(
            "%s (%s); reconnecting on the next call", summary,
            type(exc).__name__ if exc is not None else "unknown cause")
        logger.debug("%s failure", prefix, exc_info=record.exc_info)
        return False


logging.getLogger("mcp.client.sse").addFilter(_TransportNoise())


class _PooledClient:
    """A long-lived MCP client for one server URL on one event loop."""

    def __init__(self, url: str, loop: asyncio.AbstractEventLoop) -> None:
        self.url = url
        # Held so the loop cannot be garbage collected while this entry is
        # pooled under its id(), which would let a later loop reuse the address
        # and inherit a session belonging to a dead one.
        self.loop = loop
        self._client: Client | None = None
        self._connect_lock = asyncio.Lock()
        self._log_handlers: dict[int, Callable] = {}
        self._next_token = 0

    async def _dispatch_log(self, message: object) -> None:
        """Forward a server log notification to the call that caused it.

        MCP log notifications carry no request id, so a line can only be
        attributed while a single call is in flight on this connection. With
        several running, labelling sandbox output with whichever tool happened
        to be picked would be worse than not showing it: the UI reports batch
        progress in that case anyway.
        """
        handlers = list(self._log_handlers.values())
        if len(handlers) != 1:
            return
        try:
            result = handlers[0](message)
            if hasattr(result, "__await__"):
                await result
        except Exception:
            logger.debug("log handler failed for %s", self.url, exc_info=True)

    async def _connect(self) -> Client:
        if self._client is not None:
            return self._client
        async with self._connect_lock:
            if self._client is None:
                client = Client(_transport_for(self.url),
                                log_handler=self._dispatch_log,
                                timeout=_REQUEST_TIMEOUT)
                await client.__aenter__()
                self._client = client
            return self._client

    async def _discard(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass

    async def _watch_session(self, client: Client, call: asyncio.Future) -> None:
        """Ping until ``call`` finishes, raising once the session is unwritable.

        The SSE transport closes its write stream when a POST fails and leaves
        everything else standing, so the session still reports itself
        connected while nothing can be sent on it again.  A ping is the
        cheapest thing that has to travel that stream: it raises immediately
        once the stream is closed, and on a server that is merely busy it just
        never answers, which leaves the call alone as it should.
        """
        while not call.done():
            await asyncio.sleep(_HEALTH_INTERVAL)
            if call.done():
                return
            await client.ping()

    async def _call_watched(self, client: Client, name: str, arguments: dict):
        """Run one tool call, failing it as soon as its session stops working."""
        call = asyncio.ensure_future(client.call_tool(name, arguments))
        watchdog = asyncio.ensure_future(self._watch_session(client, call))
        try:
            await asyncio.wait({call, watchdog},
                               return_when=asyncio.FIRST_COMPLETED)
            if call.done():
                return call.result()
            # Only a failed ping ends the watchdog before the call: the reply
            # this call is waiting for can no longer arrive, so fail now rather
            # than sit out the request timeout.
            exc = watchdog.exception()
            if exc is None:
                return await call
            raise ConnectionError(
                f"MCP session to {self.url} is no longer usable: {exc!r}") from exc
        finally:
            for task in (call, watchdog):
                if not task.done():
                    task.cancel()
            await asyncio.gather(call, watchdog, return_exceptions=True)

    async def call_tool(self, name: str, arguments: dict,
                        log_handler: Callable | None = None):
        token = self._next_token
        self._next_token += 1
        if log_handler is not None:
            self._log_handlers[token] = log_handler
        try:
            for attempt in (1, 2):
                client = await self._connect()
                try:
                    return await self._call_watched(client, name, arguments)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A pooled session can be closed underneath us: the server
                    # restarted, an idle SSE stream was reaped, or a failed POST
                    # left it unwritable and the request timed out. Reconnect and
                    # try once more — a second failure is the tool's own error
                    # and belongs to the caller.
                    await self._discard()
                    if attempt == 2:
                        raise
        finally:
            self._log_handlers.pop(token, None)


_CLIENT_POOL: dict[tuple[str, int], _PooledClient] = {}


def _pooled_client(url: str) -> _PooledClient:
    """The pooled client for ``url`` on the running loop, creating it if needed."""
    loop = asyncio.get_running_loop()
    key = (url, id(loop))
    entry = _CLIENT_POOL.get(key)
    if entry is None or entry.loop is not loop:
        entry = _PooledClient(url, loop)
        _CLIENT_POOL[key] = entry
    return entry


class ToolHandler(RequestHandler):
    def __init__(self,
                 url: str | None = None,
                 tool_item: ToolItem | None = None,
                 **kwargs: object):
        super().__init__(url=url, tool_item=tool_item, **kwargs)

    async def __call__(self, log_handler: Callable | None = None, **kwargs: object) -> str | None:
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


        # Normalize arguments where a string-typed parameter received a dict.
        # Some models incorrectly nest values, e.g.:
        #   {"query": {"query": "..."}}  → {"query": "..."}   (same-key wrapping)
        #   {"query": {"type": "news"}}  → {"query": None}    (unrelated keys; let
        #                                                       the tool's required
        #                                                       check return an error)
        props = (self.tool_item.get('function', {})
                 .get('parameters', {})
                 .get('properties', {}))
        for k in list(kwargs.keys()):
            v = kwargs[k]
            if not isinstance(v, dict):
                continue
            prop = props.get(k, {})
            # Determine whether the schema expects a scalar string for this param.
            prop_type = prop.get('type', '')
            is_string_param = prop_type == 'string' or any(
                s.get('type') == 'string' for s in prop.get('anyOf', [])
            )
            if not is_string_param:
                continue
            # If the dict contains the same key, unwrap it; otherwise set to None
            # so the tool's own required-argument check returns a clean error.
            kwargs[k] = v[k] if k in v else None

        tool_name = self.tool_item['function']['name']
        logger.info(f"Calling tool: {tool_name} with arguments: {list(kwargs.keys())}")
        tool_response = await _pooled_client(self.url).call_tool(
            tool_name, kwargs, log_handler=log_handler)

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


def _contract_of(tool_item: ToolItem | None) -> str:
    """A stable identity for the contract a tool offers.

    Only the parameter schema: that is what decides whether a call written
    against one server's copy of a tool is valid against another's.  Two
    servers may word the description differently and still be the same tool.
    """
    fn = (tool_item or {}).get('function') or {}
    return json.dumps(fn.get('parameters') or {}, sort_keys=True, default=str)


# create a tool registry
class ToolRegistry:
    """Tool name → handler, across every MCP server that offers it.

    Two servers can offer the same tool name for two very different reasons,
    and conflating them is a bug:

    * **Replicas** — the same tool, same parameters, on more than one host.
      Calls rotate across them, which is the point.
    * **Collisions** — the same *name* for tools that take different
      parameters.  OnIt ships one: ``read_file`` is a text reader on the bash
      server and a text/tables/images reader on the consolidated Tools server,
      and ``fetch_content`` is defined twice as well.  Rotating across those
      makes an identical call succeed or fail depending on a coin flip, and
      — worse — advertised *both* schemas to the model under one name, which
      is an excellent way to teach it to invent parameters that exist on
      neither.

    Registration order decides which copy of a colliding name wins, so the
    order servers appear in ``mcp.servers`` is the way to choose.
    """

    def __init__(self) -> None:
        self.tools: set[str] = set()
        self.urls: dict[str, list[str]] = {}
        self.handlers: dict[str, ToolHandler] = {}
        # Registration order, so the tool list handed to a model is stable from
        # run to run: a set's iteration order is not, and reordering the tools
        # both unsettles the model and breaks the server's prefix cache.
        self.order: list[str] = []
        # name → contract of the copy that won, for detecting collisions.
        self._contracts: dict[str, str] = {}
        # (name, url, winning_url) for every copy that lost, so discovery can
        # report a misconfiguration rather than silently resolving it.
        self.collisions: list[tuple[str, str, str]] = []

    def register(self, tool: ToolHandler) -> None:
        tool_name: str = tool.tool_item['function']['name']
        tool_url: str = tool.url

        # Always addressable by name@url, winner or not: get_handler_by() is
        # how a caller reaches a specific server's copy on purpose.
        self.handlers[f"{tool_name}@{tool_url}"] = tool
        contract = _contract_of(tool.tool_item)

        if tool_name not in self.tools:
            self.tools.add(tool_name)
            self.order.append(tool_name)
            self.urls[tool_name] = [tool_url]
            self._contracts[tool_name] = contract
            return

        if contract == self._contracts[tool_name]:
            # A replica. Same call, same result, wherever it lands.
            if tool_url not in self.urls[tool_name]:
                self.urls[tool_name].append(tool_url)
            return

        # A collision. The first registration keeps the name; this copy stays
        # reachable by name@url and is kept out of the rotation, so dispatch
        # and the advertised schema agree with each other.
        self.collisions.append((tool_name, tool_url, self.urls[tool_name][0]))
        logger.warning(
            "Tool name collision: %r is offered by %s with different parameters "
            "than %s. Using the first; the other is reachable only by URL. "
            "Reorder mcp.servers to change which one wins.",
            tool_name, tool_url, self.urls[tool_name][0])

    def get_url(self, tool_name: str) -> str | None:
        """A URL serving this tool — any of them, since they are replicas."""
        if tool_name not in self.urls:
            return None
        return random.choice(self.urls[tool_name])

    def get_tool_items(self) -> list[ToolItem]:
        """One entry per tool name, in registration order.

        Iterating ``handlers`` instead listed a replicated tool once per host
        and a colliding name once per *contract* — so a model was shown
        ``read_file`` twice with different parameters and no way to tell which
        it would get.  This returns exactly what ``__getitem__`` will dispatch
        to, which is the only description of a tool that can be acted on.

        The full internal record, ``returns`` included.  What goes to a model
        server is a projection of this — see ``_api_tool_payload`` in
        ``model/serving/chat.py``, which is where knowledge of the
        chat-completions schema belongs.
        """
        tool_items: list[ToolItem] = []
        for tool_name in self.order:
            handler = self.handlers.get(f"{tool_name}@{self.urls[tool_name][0]}")
            if handler is not None:
                tool_items.append(handler.get_tool())
        return tool_items

    def get_handler_by(self, tool_name: str, url: str) -> ToolHandler | None:
        """Get a tool handler by tool name and URL"""
        if tool_name is None or url is None:
            return None

        key = f"{tool_name}@{url}"
        if key in self.handlers:
            return self.handlers[key]
        return None


    def tool_accepts_param(self, tool_name: str, param_name: str) -> bool:
        """Check if a registered tool declares a given parameter in its schema."""
        handler = self[tool_name]
        if not handler or not handler.tool_item:
            return False
        props = handler.tool_item.get('function', {}).get('parameters', {}).get('properties', {})
        return param_name in props

    def parameters_schema(self, tool_name: str) -> dict:
        """The JSON Schema a tool declares for its arguments, or ``{}``.

        Empty for an unknown tool or a server that declared nothing, and the
        validator treats empty as "nothing to check" — so a tool without a
        schema keeps dispatching exactly as it did before.
        """
        handler = self[tool_name]
        if not handler or not handler.tool_item:
            return {}
        params = handler.tool_item.get('function', {}).get('parameters')
        return params if isinstance(params, dict) else {}

    def blank_required_args(self, tool_name: str, arguments: dict) -> list[str]:
        """Required parameters the caller left out or supplied as empty.

        A model that has lost the thread emits the shape of a call without its
        content — ``bash(command="")`` — and the server, asked to run nothing,
        answers with nothing.  The model then reports on that empty result as
        though it were the task ("the command ran, `ready` was printed"), which
        is how a session ends on a non-answer.  Naming the blank parameters lets
        the caller hand back a usable error instead of dispatching the call.

        Only blank *strings* count.  ``0``, ``False`` and ``[]`` are legitimate
        values that happen to be falsy, and rejecting them would break calls
        that are perfectly well formed.
        """
        handler = self[tool_name]
        if not handler or not handler.tool_item:
            return []
        params = handler.tool_item.get('function', {}).get('parameters', {}) or {}
        required = params.get('required') or []
        blank = []
        for name in required:
            if name not in arguments:
                blank.append(name)
                continue
            value = arguments[name]
            if value is None or (isinstance(value, str) and not value.strip()):
                blank.append(name)
        return blank

    def __getitem__(self, tool_name: str) -> ToolHandler | None:
        """The handler for a tool name.

        ``urls[tool_name]`` holds only copies that agree on their parameters,
        so choosing among them at random is load balancing rather than a coin
        flip over which tool the caller gets.
        """
        if tool_name not in self.tools:
            return None
        url = random.choice(self.urls[tool_name])
        return self.handlers[f"{tool_name}@{url}"]

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self):
        return iter(self.order)
