"""Tests for src/mcp/servers/run.py — load_config, prepare_server_args, run_server."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.servers.run import (load_config, prepare_server_args, run_server,
                                 find_free_ports, DEFAULT_PORT_BASE)


# ── load_config ─────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        config = {
            "servers": [
                {"name": "TestServer", "module": "tasks.test", "enabled": True, "port": 9000}
            ]
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))
        result = load_config(str(config_file))
        assert "servers" in result
        assert result["servers"][0]["name"] == "TestServer"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_default_config_servers(self):
        """The built-in default config splits the toolset by isolation need."""
        default_path = os.path.join(
            os.path.dirname(__file__), "..", "mcp", "servers", "configs", "default.yaml"
        )
        if os.path.exists(default_path):
            result = load_config(default_path)
            assert "servers" in result
            by_name = {s["name"]: s for s in result["servers"]}
            assert set(by_name) == {
                "PromptsMCPServer", "ToolsLocalMCPServer",
                "ToolsNetMCPServer", "VLMToolsMCPServer",
            }

            # The tools that touch a session's data_path are served over a
            # pipe, so no other account on the machine can reach them.
            local = by_name["ToolsLocalMCPServer"]
            assert local["transport"] == "stdio"
            assert local["options"]["profile"] == "local"
            assert "port" not in local

            # The rest keep a socket, on loopback only.
            for name in ("PromptsMCPServer", "ToolsNetMCPServer", "VLMToolsMCPServer"):
                assert by_name[name]["transport"] == "sse"
                assert by_name[name]["host"] == "127.0.0.1"


# ── prepare_server_args ─────────────────────────────────────────────────────

class TestPrepareServerArgs:
    def test_extracts_enabled_servers(self):
        config = {
            "servers": [
                {"name": "A", "module": "tasks.a", "enabled": True, "port": 9000,
                 "host": "0.0.0.0", "path": "/a", "transport": "sse"},
                {"name": "B", "module": "tasks.b", "enabled": False, "port": 9001},
                {"name": "C", "module": "tasks.c", "port": 18200,
                 "host": "0.0.0.0", "path": "/c"},
            ]
        }
        args = prepare_server_args(config)
        # A and C should be included (C defaults to enabled=True)
        assert len(args) == 2
        names = [a[0] for a in args]
        assert "A" in names
        assert "C" in names
        assert "B" not in names

    def test_skips_server_without_name(self):
        config = {"servers": [{"module": "tasks.x", "port": 9000}]}
        args = prepare_server_args(config)
        assert len(args) == 0

    def test_skips_server_without_module(self):
        config = {"servers": [{"name": "NoModule", "port": 9000}]}
        args = prepare_server_args(config)
        assert len(args) == 0

    def test_empty_config(self):
        args = prepare_server_args({})
        assert args == []

    def test_options_passed_through(self):
        config = {
            "servers": [
                {"name": "A", "module": "tasks.a", "port": 9000,
                 "host": "0.0.0.0", "path": "/a",
                 "options": {"verbose": True}}
            ]
        }
        args = prepare_server_args(config)
        assert len(args) == 1
        assert args[0][6] == {"verbose": True}


# ── run_server ──────────────────────────────────────────────────────────────

class TestRunServer:
    def test_run_server_success(self):
        mock_module = MagicMock()
        mock_module.run = MagicMock()

        with patch("builtins.__import__", return_value=mock_module):
            result = run_server(
                name="Test", transport="sse",
                host="0.0.0.0", port=9000, path="/test",
                module="tasks.test", options={}
            )
        # run_server returns True on success
        assert result is True

    def test_run_server_import_error(self):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = run_server(
                name="Bad", transport="sse",
                host="0.0.0.0", port=9000, path="/bad",
                module="tasks.nonexistent", options={}
            )
        assert result is False

    def test_run_server_no_module(self):
        result = run_server(
            name="Empty", transport="sse",
            host="0.0.0.0", port=9000, path="/empty",
            module="", options={}
        )
        assert result is False


# ── port allocation ─────────────────────────────────────────────────────────

class TestFindFreePorts:
    """Every OnIt process gets its own ports so users never share servers."""

    def test_returns_requested_count_at_or_above_base(self):
        ports = find_free_ports(3)
        assert len(ports) == 3
        assert all(p >= DEFAULT_PORT_BASE for p in ports)

    def test_ports_are_distinct(self):
        ports = find_free_ports(4)
        assert len(set(ports)) == 4

    def test_ports_are_actually_bindable(self):
        import socket
        for port in find_free_ports(2):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("127.0.0.1", port))
            finally:
                sock.close()

    def test_skips_a_port_already_taken(self):
        """A port held by another user's OnIt is stepped over, not stolen."""
        import socket
        base = DEFAULT_PORT_BASE
        squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            try:
                squatter.bind(("127.0.0.1", base))
            except OSError:
                pytest.skip(f"port {base} is already in use on this machine")
            squatter.listen(1)
            assert base not in find_free_ports(2, base=base)
        finally:
            squatter.close()

    def test_raises_when_the_range_is_exhausted(self):
        with pytest.raises(RuntimeError, match="No free port"):
            find_free_ports(2, base=DEFAULT_PORT_BASE, limit=0)


# ── transport partitioning ──────────────────────────────────────────────────

class TestStdioServersAreNotPooled:
    def test_stdio_server_is_skipped(self):
        """The client spawns stdio servers itself, so the pool must not."""
        config = {"servers": [
            {"name": "Local", "module": "tasks.tools", "transport": "stdio"},
            {"name": "Net", "module": "tasks.tools", "port": 9001,
             "host": "127.0.0.1", "path": "/sse", "transport": "sse"},
        ]}
        args = prepare_server_args(config)
        assert [a[0] for a in args] == ["Net"]

    def test_port_override_wins_over_config(self):
        config = {"servers": [
            {"name": "Net", "module": "tasks.tools", "port": 18201,
             "host": "127.0.0.1", "path": "/sse", "transport": "sse"},
        ]}
        args = prepare_server_args(config, port_overrides={"Net": 18377})
        assert args[0][3] == 18377

    def test_host_defaults_to_loopback(self):
        """These servers have no auth; they must not be reachable off-box."""
        config = {"servers": [
            {"name": "Net", "module": "tasks.tools", "port": 9001, "path": "/sse"},
        ]}
        assert prepare_server_args(config)[0][2] == "127.0.0.1"


# ── tool profiles ───────────────────────────────────────────────────────────

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
