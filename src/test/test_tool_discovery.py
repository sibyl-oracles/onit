"""Tests for src/lib/tools.py — discover_tools, _discover_server_tools."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.tools import discover_tools, _discover_server_tools, _build_parameters


# ── helpers ─────────────────────────────────────────────────────────────────

def _fake_tool(name="search", description="Search the web"):
    """Create a mock tool object that has inputSchema."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = {"properties": {"query": {"type": "string"}}}
    tool.outputSchema = {"properties": {"result": {"type": "string"}}}
    return tool


def _fake_prompt(name="assistant", description="Assistant prompt"):
    """Create a mock prompt object that has arguments instead of inputSchema."""
    prompt = MagicMock(spec=[])
    prompt.name = name
    prompt.description = description
    # Remove inputSchema so code falls to the arguments branch
    del prompt.inputSchema
    arg = MagicMock()
    arg.name = "task"
    arg.description = "The task"
    prompt.arguments = [arg]
    return prompt


def _mock_client(tools=None, resources=None, prompts=None):
    """Build a mock fastmcp.Client context manager."""
    client = AsyncMock()
    client.list_tools = AsyncMock(return_value=tools or [])
    client.list_resources = AsyncMock(return_value=resources or [])
    client.list_prompts = AsyncMock(return_value=prompts or [])
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# ── _build_parameters ───────────────────────────────────────────────────────

class TestBuildParameters:
    """What survives the trip from an MCP inputSchema into the tool record."""

    @staticmethod
    def _tool(schema):
        tool = MagicMock()
        tool.inputSchema = schema
        return tool

    def test_required_is_preserved(self):
        """Regression: `required` used to be dropped, which silently disabled
        blank_required_args for every MCP-discovered tool and left the model
        without any signal about which parameters are mandatory."""
        params = _build_parameters(self._tool({
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }))
        assert params["required"] == ["command"]

    def test_properties_are_preserved(self):
        params = _build_parameters(self._tool(
            {"properties": {"query": {"type": "string"}}}))
        assert params["properties"] == {"query": {"type": "string"}}

    def test_type_is_pinned_to_object(self):
        params = _build_parameters(self._tool({"type": "string", "properties": {}}))
        assert params["type"] == "object"

    def test_ref_definitions_come_along(self):
        """A $ref whose referent was dropped dangles, and nothing — model
        included — can read the property that used it."""
        params = _build_parameters(self._tool({
            "properties": {"cfg": {"$ref": "#/$defs/Cfg"}},
            "$defs": {"Cfg": {"type": "object"}},
        }))
        assert params["$defs"] == {"Cfg": {"type": "object"}}

    def test_absent_keys_are_not_invented(self):
        params = _build_parameters(self._tool({"properties": {}}))
        assert "required" not in params
        assert "$defs" not in params

    def test_missing_properties_defaults_to_empty(self):
        assert _build_parameters(self._tool({}))["properties"] == {}

    def test_additional_properties_is_preserved(self):
        """FastMCP sets it false and means it. Without it the harness cannot
        tell "extras allowed" from "extras rejected", and has to let every such
        call make the round trip to find out."""
        params = _build_parameters(self._tool(
            {"properties": {}, "additionalProperties": False}))
        assert params["additionalProperties"] is False

    def test_mcp_extras_are_dropped(self):
        params = _build_parameters(self._tool(
            {"properties": {}, "title": "X", "$schema": "http://json-schema.org/"}))
        assert set(params) == {"type", "properties"}


# ── one tool, one contract, across the servers that offer it ────────────────

class TestConsolidatedToolContracts:
    """`read_file` and `search_document` are each registered by two servers.

    They used to be *different tools sharing a name*: the bash server's
    read_file had no `mode`, so `read_file(mode="tables")` succeeded or failed
    depending which server the registry picked, and both schemas were shown to
    the model at once. Both now register one shared definition, which makes
    them replicas — see TestNameCollisions in test_tool_registry.py.
    """

    @staticmethod
    def _schemas(name):
        import importlib
        from lib.tools import _build_parameters
        out = {}
        for label, mod_path in (
                ("tools", "src.mcp.servers.tasks.tools.mcp_server"),
                ("bash", "src.mcp.servers.tasks.os.bash.mcp_server")):
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, name)
            fn = getattr(fn, "fn", fn)
            import inspect
            out[label] = list(inspect.signature(fn).parameters)
        return out

    def test_read_file_signatures_match(self):
        tools, bash = self._schemas("read_file").values()
        assert tools == bash

    def test_search_document_signatures_match(self):
        tools, bash = self._schemas("search_document").values()
        assert tools == bash

    def test_read_file_covers_all_three_modes(self):
        params = self._schemas("read_file")["bash"]
        # tables and images used to be reachable only through the Tools server
        for p in ("mode", "table_index", "output_format", "output_dir", "min_size"):
            assert p in params

    def test_search_document_covers_both_modes(self):
        params = self._schemas("search_document")["bash"]
        for p in ("mode", "query", "keywords", "context_chars", "max_sections"):
            assert p in params

    def test_descriptions_are_the_same_object(self):
        """Two servers offering one name with different description text is a
        collision to ToolRegistry, so the text is shared, not copied."""
        import importlib
        shared = importlib.import_module("src.mcp.servers.tasks.shared")
        tools = importlib.import_module("src.mcp.servers.tasks.tools.mcp_server")
        bash = importlib.import_module("src.mcp.servers.tasks.os.bash.mcp_server")
        assert tools.READ_FILE_DESCRIPTION is shared.READ_FILE_DESCRIPTION
        assert bash.READ_FILE_DESCRIPTION is shared.READ_FILE_DESCRIPTION
        assert tools.SEARCH_DOCUMENT_DESCRIPTION is shared.SEARCH_DOCUMENT_DESCRIPTION
        assert bash.SEARCH_DOCUMENT_DESCRIPTION is shared.SEARCH_DOCUMENT_DESCRIPTION

    def test_read_file_description_documents_every_parameter(self):
        """The `offset`/`limit` hallucination in the recorded trajectories was a
        model filling a documentation gap with another harness's signature."""
        import importlib
        shared = importlib.import_module("src.mcp.servers.tasks.shared")
        text = shared.READ_FILE_DESCRIPTION
        for p in self._schemas("read_file")["bash"]:
            assert p in text, f"{p} is undocumented"
        assert "offset" in text and "limit" in text  # says they do not exist


# ── _discover_server_tools ──────────────────────────────────────────────────

class TestDiscoverServerTools:
    @pytest.mark.asyncio
    async def test_discovers_tools_with_input_schema(self):
        server = {"name": "ToolsMCPServer", "url": "http://127.0.0.1:18201/sse", "enabled": True}
        mock = _mock_client(tools=[_fake_tool("search")])
        with patch("lib.tools._wait_for_port", new=AsyncMock(return_value=True)), \
             patch("lib.tools.Client", return_value=mock):
            handlers = await _discover_server_tools(server)
        assert len(handlers) == 1
        assert handlers[0].tool_item["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_prompts_only_server_raises_after_retries(self):
        # A server that returns no tools (e.g. a prompts-only server) will be
        # retried and ultimately raise.  In practice onit.py filters out
        # PromptsMCPServer before calling discover_tools, so this path is
        # defensive.  discover_tools handles the exception gracefully.
        server = {"name": "Prompts", "url": "http://127.0.0.1:18200/sse", "enabled": True}
        mock = _mock_client(prompts=[_fake_prompt("assistant")])
        with patch("lib.tools._wait_for_port", new=AsyncMock(return_value=True)), \
             patch("lib.tools.Client", return_value=mock):
            with pytest.raises(ValueError, match="empty tool list"):
                await _discover_server_tools(server, max_retries=1)

    @pytest.mark.asyncio
    async def test_skips_disabled_server(self):
        server = {"name": "Disabled", "url": "http://x", "enabled": False}
        handlers = await _discover_server_tools(server)
        assert handlers == []

    @pytest.mark.asyncio
    async def test_skips_server_without_url(self):
        server = {"name": "NoURL", "enabled": True}
        handlers = await _discover_server_tools(server)
        assert handlers == []


# ── discover_tools ──────────────────────────────────────────────────────────

def _target_url(target) -> str:
    """URL of whatever discover_tools handed to Client.

    Discovery builds a transport rather than passing the URL string, so that
    stdio servers and HTTP servers go through one code path. Both SSE and
    streamable-HTTP transports keep the URL on ``.url``.
    """
    return target if isinstance(target, str) else str(getattr(target, "url", target))


class TestDiscoverTools:
    @pytest.mark.asyncio
    async def test_discovers_from_multiple_servers(self):
        servers = [
            {"name": "Prompts", "url": "http://127.0.0.1:18200/sse", "enabled": True},
            {"name": "Tools", "url": "http://127.0.0.1:18201/sse", "enabled": True},
        ]

        mock_a = _mock_client(tools=[_fake_tool("search")])
        mock_b = _mock_client(tools=[_fake_tool("bash")])

        def client_factory(target):
            return mock_a if "18200" in _target_url(target) else mock_b

        with patch("lib.tools._wait_for_port", new=AsyncMock(return_value=True)), \
             patch("lib.tools.Client", side_effect=client_factory):
            registry = await discover_tools(servers)

        assert len(registry) == 2
        assert "search" in registry.tools
        assert "bash" in registry.tools

    @pytest.mark.asyncio
    async def test_handles_connection_error(self):
        servers = [
            {"name": "Good", "url": "http://127.0.0.1:18201/sse", "enabled": True},
            {"name": "Bad", "url": "http://127.0.0.1:9999/bad", "enabled": True},
        ]

        mock_good = _mock_client(tools=[_fake_tool("search")])

        def client_factory(target):
            if "9999" in _target_url(target):
                raise ConnectionError("Cannot connect")
            return mock_good

        with patch("lib.tools._wait_for_port", new=AsyncMock(return_value=True)), \
             patch("lib.tools.Client", side_effect=client_factory):
            registry = await discover_tools(servers)

        # Only the good server's tool should be registered
        assert len(registry) == 1
        assert "search" in registry.tools

    @pytest.mark.asyncio
    async def test_empty_servers_list(self):
        registry = await discover_tools([])
        assert len(registry) == 0
