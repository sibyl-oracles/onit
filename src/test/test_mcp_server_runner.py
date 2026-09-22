"""Tests for src/mcp/servers/run.py — the stdio server entry point.

The server pool (port search, claims, spawn, watchdog) is gone: every built-in
MCP server is spawned by the MCP client over a pipe. What remains here is the
one process an MCP client launches, and the tool profiles it serves.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.servers.run import run_server


# ── run_server ───────────────────────────────────────────────────────────────

class TestRunServer:
    def test_run_server_success(self):
        mock_module = MagicMock()
        mock_module.run = MagicMock()

        with patch("builtins.__import__", return_value=mock_module):
            result = run_server(name="Test", module="tasks.test", options={})
        assert result is True
        # stdio is the only transport: the module's run() gets it, and the
        # caller's options ride through unchanged.
        mock_module.run.assert_called_once_with(transport='stdio', options={})

    def test_the_tasks_shorthand_resolves_into_the_package(self):
        mock_module = MagicMock()
        with patch("builtins.__import__", return_value=mock_module) as imp:
            run_server(name="Test", module="tasks.test", options={})
        assert imp.call_args.args[0] == "src.mcp.servers.tasks.test"

    def test_the_src_prefix_is_taken_literally(self):
        mock_module = MagicMock()
        with patch("builtins.__import__", return_value=mock_module) as imp:
            run_server(name="Test", module="src.mcp.prompts.prompts", options={})
        assert imp.call_args.args[0] == "src.mcp.prompts.prompts"

    def test_run_server_import_error(self):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = run_server(
                name="Bad", module="tasks.nonexistent", options={})
        assert result is False

    def test_run_server_no_module(self):
        result = run_server(name="Empty", module="", options={})
        assert result is False


# ── tool profiles ────────────────────────────────────────────────────────────

class TestToolProfiles:
    """The aggregate server is subset by profile so the tools that touch a
    session's data_path can be served over a pipe, away from any socket."""

    @pytest.fixture(autouse=True)
    def _all_tools_registered(self, monkeypatch):
        """Register the full toolset regardless of ambient environment.

        Several tools are registered only when their ONIT_DISABLE_* variable
        is unset, so a value left in the environment by another test would
        otherwise change what these profiles contain.
        """
        for var in ("ONIT_DISABLE_WEB_SEARCH", "ONIT_DISABLE_WEATHER",
                    "ONIT_DISABLE_LOCAL_SEARCH"):
            monkeypatch.delenv(var, raising=False)

    @staticmethod
    def _names(profile):
        import asyncio
        import importlib
        import src.mcp.servers.tasks.tools.mcp_server as mod
        # Each case needs the full registration set, and _apply_profile
        # mutates it, so rebuild the module per call.
        mod = importlib.reload(mod)
        mod._apply_profile(profile)
        return {t.name for t in asyncio.run(mod.mcp.list_tools())}

    def test_local_profile_keeps_every_data_path_tool(self):
        names = self._names("local")
        assert {"bash", "read_file", "write_file", "edit_file", "serve",
                "grep", "send_file", "search_document", "fetch_content",
                "index_documents", "local_search"} <= names

    def test_local_profile_drops_the_stateless_tools(self):
        assert self._names("local").isdisjoint({"search", "get_weather"})

    def test_net_profile_is_exactly_the_stateless_tools(self):
        assert self._names("net") == {"search", "get_weather"}

    def test_the_credential_bearing_tool_stays_per_user(self):
        """github_repo acts under a token and can delete repos, so it belongs
        to the user who started it, not to whoever reaches the socket first."""
        assert "github_repo" in self._names("local")
        assert "github_repo" not in self._names("net")

    def test_the_two_profiles_partition_the_toolset(self):
        local, net = self._names("local"), self._names("net")
        assert local.isdisjoint(net)
        assert local | net == self._names("all")

    def test_net_profile_never_exceeds_the_declared_stateless_set(self):
        import importlib
        import src.mcp.servers.tasks.tools.mcp_server as mod
        assert self._names("net") <= set(importlib.reload(mod).NET_TOOLS)

    def test_unknown_profile_raises(self):
        import importlib
        import src.mcp.servers.tasks.tools.mcp_server as mod
        mod = importlib.reload(mod)
        with pytest.raises(ValueError, match="Unknown tool profile"):
            mod._apply_profile("sideways")

    def test_tool_enumeration_survives_old_fastmcp(self, monkeypatch):
        """The stdio child must boot on fastmcp 2.x, where FastMCP has no
        list_tools(); enumeration falls back to get_tools() and then to
        the tool manager.  An AttributeError here killed the child at
        startup on the RPi environment (mcp_ToolsLocalMCPServer_0.log:
        "'FastMCP' object has no attribute 'list_tools'")."""
        import importlib
        import src.mcp.servers.tasks.tools.mcp_server as mod
        mod = importlib.reload(mod)

        class OldFastMCP:
            """The fastmcp 2.x surface: enumeration via get_tools()."""

            async def get_tools(self):
                return [type("T", (), {"name": "bash"})(),
                        type("T", (), {"name": "grep"})()]

        monkeypatch.setattr(mod, "mcp", OldFastMCP())
        assert mod._registered_tool_names() == ["bash", "grep"]

        class AncientFastMCP:
            """A pre-2.x surface: only the private tool manager."""

            def __init__(self):
                manager = type("TM", (), {
                    "list_tools": lambda self: [type("T", (), {"name": n})()
                                                for n in ("bash", "grep")]})()
                self._tool_manager = manager

        monkeypatch.setattr(mod, "mcp", AncientFastMCP())
        assert mod._registered_tool_names() == ["bash", "grep"]