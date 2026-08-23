"""Tests for src/type/tools.py — ToolRegistry, ToolHandler, RequestHandler."""

import asyncio
import inspect
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from type.tools import ToolRegistry, ToolHandler, RequestHandler


# ── RequestHandler (abstract) ───────────────────────────────────────────────

class TestRequestHandler:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            RequestHandler()


# ── ToolRegistry ────────────────────────────────────────────────────────────

def _make_handler(name="tool", url="http://localhost:9000/x", properties=None,
                  description=None):
    item = {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"{name} tool",
            "parameters": {"type": "object", "properties": properties or {}},
            "returns": {},
        },
    }
    return ToolHandler(url=url, tool_item=item)


class TestToolRegistry:
    def test_register_and_len(self):
        reg = ToolRegistry()
        reg.register(_make_handler("a", "http://h1"))
        assert len(reg) == 1
        assert "a" in reg.tools

    def test_register_multiple_urls_for_same_tool(self):
        reg = ToolRegistry()
        reg.register(_make_handler("a", "http://h1"))
        reg.register(_make_handler("a", "http://h2"))
        # Still one unique tool name
        assert len(reg) == 1
        assert len(reg.urls["a"]) == 2
        # Two handler entries
        assert len(reg.handlers) == 2


# ── replicas vs. name collisions ────────────────────────────────────────────

class TestNameCollisions:
    """OnIt ships one real collision: `read_file` is a text reader on the bash
    server and a text/tables/images reader on the consolidated Tools server."""

    _TEXT_ONLY = {"path": {"type": "string"}, "encoding": {"type": "string"}}
    _WITH_MODE = {"path": {"type": "string"}, "mode": {"type": "string"},
                  "encoding": {"type": "string"}}

    def _colliding_registry(self):
        reg = ToolRegistry()
        reg.register(_make_handler("read_file", "http://tools", self._WITH_MODE))
        reg.register(_make_handler("read_file", "http://bash", self._TEXT_ONLY))
        return reg

    def test_replicas_share_the_rotation(self):
        """Same parameters on two hosts is load balancing, not a conflict."""
        reg = ToolRegistry()
        reg.register(_make_handler("search", "http://h1", {"q": {"type": "string"}}))
        reg.register(_make_handler("search", "http://h2", {"q": {"type": "string"}}))
        assert reg.urls["search"] == ["http://h1", "http://h2"]
        assert reg.collisions == []

    def test_description_differences_are_not_a_collision(self):
        """Two servers may word a tool differently and still be the same tool."""
        reg = ToolRegistry()
        reg.register(_make_handler("search", "http://h1", {"q": {"type": "string"}},
                                   description="Search the web"))
        reg.register(_make_handler("search", "http://h2", {"q": {"type": "string"}},
                                   description="Web search"))
        assert reg.collisions == []
        assert len(reg.urls["search"]) == 2

    def test_different_parameters_is_a_collision(self):
        reg = self._colliding_registry()
        assert len(reg.collisions) == 1
        tool_name, losing, winning = reg.collisions[0]
        assert (tool_name, losing, winning) == ("read_file", "http://bash", "http://tools")

    def test_the_loser_is_kept_out_of_the_rotation(self):
        """Regression: dispatch used to pick at random, so the same call
        succeeded or failed depending on a coin flip."""
        reg = self._colliding_registry()
        assert reg.urls["read_file"] == ["http://tools"]
        for _ in range(20):
            assert reg[  # never resolves to the bash copy
                "read_file"].url == "http://tools"

    def test_the_model_is_shown_exactly_one_schema(self):
        """Regression: both schemas were advertised under one name, which is an
        excellent way to teach a model to invent parameters."""
        reg = self._colliding_registry()
        items = [i for i in reg.get_tool_items() if i["function"]["name"] == "read_file"]
        assert len(items) == 1
        assert "mode" in items[0]["function"]["parameters"]["properties"]

    def test_the_advertised_schema_is_the_one_that_dispatches(self):
        reg = self._colliding_registry()
        advertised = reg.get_tool_items()[0]["function"]["parameters"]
        assert reg["read_file"].tool_item["function"]["parameters"] == advertised

    def test_registration_order_decides_the_winner(self):
        """Which is how an operator picks: reorder mcp.servers."""
        reg = ToolRegistry()
        reg.register(_make_handler("read_file", "http://bash", self._TEXT_ONLY))
        reg.register(_make_handler("read_file", "http://tools", self._WITH_MODE))
        assert reg["read_file"].url == "http://bash"
        assert "mode" not in reg.get_tool_items()[0]["function"]["parameters"]["properties"]

    def test_the_loser_stays_reachable_by_url(self):
        """get_handler_by() is how a caller reaches a specific server on purpose."""
        reg = self._colliding_registry()
        assert reg.get_handler_by("read_file", "http://bash") is not None

    def test_replicas_are_not_listed_twice(self):
        reg = ToolRegistry()
        reg.register(_make_handler("search", "http://h1", {"q": {"type": "string"}}))
        reg.register(_make_handler("search", "http://h2", {"q": {"type": "string"}}))
        assert len(reg.get_tool_items()) == 1

    def test_tool_listing_order_is_registration_order(self):
        """A set's iteration order is not stable; reordering the tool list both
        unsettles the model and breaks the server's prefix cache."""
        reg = ToolRegistry()
        for name in ("zebra", "apple", "mango", "kiwi"):
            reg.register(_make_handler(name, f"http://{name}"))
        assert [i["function"]["name"] for i in reg.get_tool_items()] == \
            ["zebra", "apple", "mango", "kiwi"]
        assert list(reg) == ["zebra", "apple", "mango", "kiwi"]

    def test_get_url_returns_url(self):
        reg = ToolRegistry()
        reg.register(_make_handler("x", "http://only"))
        assert reg.get_url("x") == "http://only"

    def test_get_url_unknown_returns_none(self):
        reg = ToolRegistry()
        assert reg.get_url("nope") is None

    def test_get_tool_items(self):
        reg = ToolRegistry()
        reg.register(_make_handler("a"))
        reg.register(_make_handler("b"))
        items = reg.get_tool_items()
        assert len(items) == 2
        names = {i["function"]["name"] for i in items}
        assert names == {"a", "b"}

    def test_get_tool_items_keeps_returns(self):
        """The internal record keeps outputSchema; only the wire payload drops it."""
        reg = ToolRegistry()
        reg.register(_make_handler("a"))
        assert "returns" in reg.get_tool_items()[0]["function"]


# ── parameters_schema ───────────────────────────────────────────────────────

class TestParametersSchema:
    def _reg_with_schema(self, params):
        reg = ToolRegistry()
        item = {"type": "function",
                "function": {"name": "search", "description": "d",
                             "parameters": params, "returns": {}}}
        reg.register(ToolHandler(url="http://h", tool_item=item))
        return reg

    def test_returns_the_declared_schema(self):
        params = {"type": "object", "properties": {"q": {"type": "string"}},
                  "required": ["q"]}
        assert self._reg_with_schema(params).parameters_schema("search") == params

    def test_unknown_tool_returns_empty(self):
        assert self._reg_with_schema({}).parameters_schema("nope") == {}

    def test_non_dict_parameters_returns_empty(self):
        """Empty means "nothing to check", so a tool without a usable schema
        keeps dispatching exactly as it did before."""
        assert self._reg_with_schema(None).parameters_schema("search") == {}

    def test_get_handler_by(self):
        reg = ToolRegistry()
        h = _make_handler("t", "http://u")
        reg.register(h)
        assert reg.get_handler_by("t", "http://u") is h
        assert reg.get_handler_by("t", "http://other") is None

    def test_getitem_returns_handler(self):
        reg = ToolRegistry()
        h = _make_handler("t", "http://u")
        reg.register(h)
        assert reg["t"] is h

    def test_getitem_unknown_returns_none(self):
        reg = ToolRegistry()
        assert reg["missing"] is None

    def test_iter(self):
        reg = ToolRegistry()
        reg.register(_make_handler("x"))
        reg.register(_make_handler("y"))
        assert set(reg) == {"x", "y"}


# ── ToolHandler ─────────────────────────────────────────────────────────────

class TestToolHandler:
    def test_get_tool(self):
        h = _make_handler("t")
        item = h.get_tool()
        assert item["function"]["name"] == "t"

    @pytest.mark.asyncio
    async def test_call_text_content(self):
        """Mocked MCP client returns TextContent."""
        from mcp.types import TextContent

        mock_response = MagicMock()
        mock_response.content = [TextContent(type="text", text="hello world")]

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        h = _make_handler("t")
        with patch("type.tools.Client", return_value=mock_client):
            result = await h(query="test")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_call_image_content(self):
        """Mocked MCP client returns ImageContent — should save a temp PNG."""
        import base64
        from mcp.types import ImageContent

        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        b64 = base64.b64encode(fake_png).decode()

        mock_response = MagicMock()
        mock_response.content = [ImageContent(type="image", data=b64, mimeType="image/png")]

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        h = _make_handler("t")
        with patch("type.tools.Client", return_value=mock_client):
            result = await h(query="img")

        assert result.endswith(".png")
        assert os.path.isfile(result)
        os.unlink(result)

    @pytest.mark.asyncio
    async def test_empty_images_list(self):
        h = _make_handler("t")
        result = await h(images=[])
        assert "No images provided" in result

    @pytest.mark.asyncio
    async def test_empty_audios_list(self):
        h = _make_handler("t")
        result = await h(audios=[])
        assert "No audios provided" in result

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        h = _make_handler("t")
        result = await h(images="/nonexistent/path.png")
        assert "File not found" in result


# ── pooled session recovery ─────────────────────────────────────────────────

class TestPooledClient:
    """A pooled MCP session that stops working must fail, then heal."""

    @pytest.mark.asyncio
    async def test_session_gets_a_request_timeout(self):
        """Without one, a half-dead session hangs the call instead of failing it.

        The SSE transport closes its write stream when a POST fails but leaves
        the session looking connected, so an untimed request waits forever.
        """
        from mcp.types import TextContent
        from type.tools import _REQUEST_TIMEOUT

        mock_response = MagicMock()
        mock_response.content = [TextContent(type="text", text="ok")]

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        h = _make_handler("t")
        with patch("type.tools.Client", return_value=mock_client) as ctor:
            await h(query="test")

        assert ctor.call_args.kwargs["timeout"] == _REQUEST_TIMEOUT

    @pytest.mark.asyncio
    async def test_unwritable_session_is_replaced(self):
        """The timed-out call discards the session and retries on a fresh one."""
        from mcp.types import TextContent

        mock_response = MagicMock()
        mock_response.content = [TextContent(type="text", text="recovered")]

        dead = AsyncMock()
        dead.call_tool = AsyncMock(side_effect=TimeoutError("no reply"))
        dead.__aenter__ = AsyncMock(return_value=dead)
        dead.__aexit__ = AsyncMock(return_value=False)

        live = AsyncMock()
        live.call_tool = AsyncMock(return_value=mock_response)
        live.__aenter__ = AsyncMock(return_value=live)
        live.__aexit__ = AsyncMock(return_value=False)

        h = _make_handler("t")
        with patch("type.tools.Client", side_effect=[dead, live]):
            result = await h(query="test")

        assert result == "recovered"
        assert dead.__aexit__.await_count == 1  # the dead session was closed
        assert live.call_tool.await_count == 1

    @pytest.mark.asyncio
    async def test_second_failure_reaches_the_caller(self):
        """A tool that fails twice is the tool's own error, not a stale session."""
        failing = AsyncMock()
        failing.call_tool = AsyncMock(side_effect=RuntimeError("tool exploded"))
        failing.__aenter__ = AsyncMock(return_value=failing)
        failing.__aexit__ = AsyncMock(return_value=False)

        h = _make_handler("t")
        with patch("type.tools.Client", return_value=failing):
            with pytest.raises(RuntimeError, match="tool exploded"):
                await h(query="test")

        assert failing.call_tool.await_count == 2


# ── connection hygiene ──────────────────────────────────────────────────────

# Long enough that a hung call would fail the test's own timeout first.
_REQUEST_TIMEOUT_STANDIN = 30


def _fake_stdio_spec():
    """A launch spec for a server no test ever starts."""
    return {"command": sys.executable, "args": ["-c", "pass"],
            "env": {}, "cwd": None, "log_file": None}

class TestConnectionHygiene:
    """The client must not POST into a connection the server is closing.

    httpx and uvicorn both expire an idle connection after 5s by default, so a
    tool call arriving near that mark could be written into a socket the server
    had already closed — "Error in post_writer ... httpx.ReadError", and a
    session that looked connected but could never be written to again.
    """

    def test_client_gives_up_a_connection_before_any_server_does(self):
        import httpx
        from type.tools import _http_client_factory, _KEEPALIVE_EXPIRY

        with patch("type.tools._StreamingReadTimeoutClient") as ctor:
            _http_client_factory()

        limits = ctor.call_args.kwargs["limits"]
        assert limits.keepalive_expiry == _KEEPALIVE_EXPIRY
        # uvicorn's default idle close is 5s; ours has to land clear of it.
        assert limits.keepalive_expiry < httpx._config.DEFAULT_LIMITS.keepalive_expiry

    def test_server_holds_connections_longer_than_the_client_reuses_them(self):
        """The two timers that used to collide, checked against each other."""
        from src.mcp.servers.tasks.shared import uvicorn_config
        from type.tools import _KEEPALIVE_EXPIRY

        assert uvicorn_config()["timeout_keep_alive"] > _KEEPALIVE_EXPIRY
        # Quiet mode must not drop the setting.
        assert uvicorn_config(quiet=False)["timeout_keep_alive"] > _KEEPALIVE_EXPIRY

    def test_sse_url_gets_a_transport_carrying_our_http_client(self):
        from fastmcp.client.transports import SSETransport
        from type.tools import _transport_for, _http_client_factory

        transport = _transport_for("http://127.0.0.1:18201/sse")
        assert isinstance(transport, SSETransport)
        assert transport.httpx_client_factory is _http_client_factory

    def test_non_http_target_is_left_for_fastmcp_to_infer(self):
        from type.tools import _transport_for

        assert _transport_for("server.py") == "server.py"

    @pytest.mark.asyncio
    async def test_one_stdio_subprocess_is_shared_by_every_pooled_call(self):
        """A transport dropped rather than closed is torn down by __del__.

        __del__ signals the connect task without awaiting it, and whatever the
        server writes during that unwind hits a stream the session has already
        closed — an unretrieved BrokenResourceError printed over the UI, and a
        subprocess that outlived the client.  Handing back the same transport
        is what stops one being dropped in the first place.
        """
        from type.tools import _transport_for, _STDIO_SPECS

        _STDIO_SPECS["stdio://Shared"] = _fake_stdio_spec()
        try:
            first = _transport_for("stdio://Shared")
            assert _transport_for("stdio://Shared") is first
            assert first.keep_alive is True
        finally:
            _STDIO_SPECS.pop("stdio://Shared", None)

    @pytest.mark.asyncio
    async def test_a_one_shot_caller_gets_a_subprocess_it_can_close(self):
        """Discovery must not borrow the session the tool calls will run on.

        The log handler and timeout belong to whoever connects first, so a
        borrowed session would route the pooled client's tool output nowhere.
        keep_alive=False is what lets the client context close this one.
        """
        from type.tools import _transport_for, _STDIO_SPECS

        _STDIO_SPECS["stdio://OneShot"] = _fake_stdio_spec()
        try:
            shared = _transport_for("stdio://OneShot")
            one_shot = _transport_for("stdio://OneShot", shared=False)
            assert one_shot is not shared
            assert one_shot.keep_alive is False
            # Nor may it displace the shared one.
            assert _transport_for("stdio://OneShot") is shared
        finally:
            _STDIO_SPECS.pop("stdio://OneShot", None)

    @pytest.mark.asyncio
    async def test_a_discarded_stdio_session_takes_its_subprocess_with_it(self):
        """keep_alive holds the subprocess past the client's own close.

        Closing the transport is what reaps it, and close() awaits the connect
        task, so a teardown-race error is retrieved here instead of surfacing
        on the terminal later.
        """
        from type.tools import _PooledClient

        client = MagicMock()
        client.__aexit__ = AsyncMock(return_value=False)
        client.transport = MagicMock()
        client.transport.close = AsyncMock()

        pooled = _PooledClient("stdio://Discarded", asyncio.get_running_loop())
        pooled._client = client
        await pooled._discard()

        client.transport.close.assert_awaited_once()
        assert pooled._client is None

    @pytest.mark.asyncio
    async def test_an_http_session_is_discarded_without_touching_its_transport(self):
        """Only a stdio transport owns a process that outlives its client."""
        from type.tools import _PooledClient

        client = MagicMock()
        client.__aexit__ = AsyncMock(return_value=False)
        client.transport = MagicMock()
        client.transport.close = AsyncMock()

        pooled = _PooledClient("http://127.0.0.1:18201/sse",
                               asyncio.get_running_loop())
        pooled._client = client
        await pooled._discard()

        client.transport.close.assert_not_awaited()

    def test_transport_tracebacks_are_collapsed_to_one_line(self):
        """The SDK logs the whole stack at ERROR; recovery happens here."""
        import logging
        from type.tools import _TransportNoise

        def record(msg):
            return logging.LogRecord(
                "mcp.client.sse", logging.ERROR, __file__, 1, msg, None,
                (RuntimeError, RuntimeError("read error"), None))

        # Both halves of the transport report a dead session this way.
        assert _TransportNoise().filter(record("Error in post_writer")) is False
        assert _TransportNoise().filter(record("Error in sse_reader")) is False
        # Anything else the SDK has to say still reaches the user.
        assert _TransportNoise().filter(record("Unknown SSE event: x")) is True

    @pytest.mark.asyncio
    async def test_idle_event_stream_outlives_the_gap_between_tool_calls(self):
        """A quiet stream must not expire while the user types the next turn.

        The SDK reads the stream on a 300s timeout, which is shorter than an
        ordinary pause between two tool calls on a pooled session.  Streamed
        responses get their own, far longer, read timeout; POSTs keep the short
        one so a wedged server still fails fast.
        """
        import httpx
        from mcp.client.sse import sse_client  # for its default read timeout
        from type.tools import _http_client_factory, _STREAM_READ_TIMEOUT

        sdk_default = inspect.signature(sse_client).parameters[
            "sse_read_timeout"].default
        assert _STREAM_READ_TIMEOUT > sdk_default

        client = _http_client_factory(timeout=httpx.Timeout(30.0, read=300.0))
        try:
            streamed = client.build_request("GET", "http://127.0.0.1/sse")
            posted = client.build_request("POST", "http://127.0.0.1/messages/")
            with patch.object(httpx.AsyncClient, "send", new=AsyncMock()):
                await client.send(streamed, stream=True)
                await client.send(posted)
        finally:
            await client.aclose()

        assert streamed.extensions["timeout"]["read"] == _STREAM_READ_TIMEOUT
        assert posted.extensions["timeout"]["read"] == 300.0


class TestSessionWatchdog:
    """A call in flight when the session dies must not wait out the timeout."""

    @pytest.mark.asyncio
    async def test_unwritable_session_is_caught_mid_call(self):
        import asyncio
        from anyio import ClosedResourceError
        from mcp.types import TextContent

        mock_response = MagicMock()
        mock_response.content = [TextContent(type="text", text="recovered")]

        async def never_answers(*args, **kwargs):
            await asyncio.sleep(_REQUEST_TIMEOUT_STANDIN)

        dead = AsyncMock()
        dead.call_tool = AsyncMock(side_effect=never_answers)
        # What a ping does once post_writer has closed the write stream.
        dead.ping = AsyncMock(side_effect=ClosedResourceError())
        dead.__aenter__ = AsyncMock(return_value=dead)
        dead.__aexit__ = AsyncMock(return_value=False)

        live = AsyncMock()
        live.call_tool = AsyncMock(return_value=mock_response)
        live.__aenter__ = AsyncMock(return_value=live)
        live.__aexit__ = AsyncMock(return_value=False)

        h = _make_handler("t")
        with patch("type.tools.Client", side_effect=[dead, live]), \
             patch("type.tools._HEALTH_INTERVAL", 0.01):
            result = await asyncio.wait_for(h(query="test"), timeout=5)

        assert result == "recovered"
        assert dead.__aexit__.await_count == 1  # the dead session was closed

    @pytest.mark.asyncio
    async def test_a_busy_server_is_not_treated_as_a_dead_one(self):
        """A ping that never answers means the server is working, not gone."""
        import asyncio
        from mcp.types import TextContent

        mock_response = MagicMock()
        mock_response.content = [TextContent(type="text", text="slow but fine")]

        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(0.05)
            return mock_response

        async def unanswered_ping(*args, **kwargs):
            await asyncio.sleep(_REQUEST_TIMEOUT_STANDIN)

        busy = AsyncMock()
        busy.call_tool = AsyncMock(side_effect=slow_tool)
        busy.ping = AsyncMock(side_effect=unanswered_ping)
        busy.__aenter__ = AsyncMock(return_value=busy)
        busy.__aexit__ = AsyncMock(return_value=False)

        h = _make_handler("t")
        with patch("type.tools.Client", return_value=busy), \
             patch("type.tools._HEALTH_INTERVAL", 0.01):
            result = await asyncio.wait_for(h(query="test"), timeout=5)

        assert result == "slow but fine"
        assert busy.__aexit__.await_count == 0  # session kept

