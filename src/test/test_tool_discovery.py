"""Tests for src/lib/tools.py — discover_tools, _discover_server_tools."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.tools import discover_tools, _discover_server_tools


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

class TestDiscoverTools:
    @pytest.mark.asyncio
    async def test_discovers_from_multiple_servers(self):
        servers = [
            {"name": "Prompts", "url": "http://127.0.0.1:18200/sse", "enabled": True},
            {"name": "Tools", "url": "http://127.0.0.1:18201/sse", "enabled": True},
        ]

        mock_a = _mock_client(tools=[_fake_tool("search")])
        mock_b = _mock_client(tools=[_fake_tool("bash")])

        def client_factory(url):
            return mock_a if "18200" in url else mock_b

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

        def client_factory(url):
            if "9999" in url:
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
