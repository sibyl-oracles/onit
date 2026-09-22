"""Two people running OnIt on one machine must not share tool servers.

Before the stdio split these tests describe, the second OnIt to start found
ports 18200/18201 already bound, treated that as "my servers are already up",
and connected to the *first* user's server process. Their tools then ran under
the first user's account, and every filesystem call was rejected because the
session directory sat outside that server's data root.

The fix removed the socket entirely: every built-in MCP server is spawned by
the MCP client, over a pipe, as the user who started OnIt. There is no port to
contend for and no server process to adopt by mistake — which is why the port
tests that used to live here are gone with the machinery they covered.
"""

import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastmcp import Client  # noqa: E402  (before src/ shadows the mcp package)

from src.cli import _ensure_mcp_servers  # noqa: E402

# Imported without the ``src.`` prefix, the way lib/tools.py does at runtime.
# Reaching the same module by both names would give this test a second, empty
# copy of the stdio launch-spec registry.
from lib.tools import register_stdio_servers  # noqa: E402
from type.tools import _transport_for  # noqa: E402


# ── defaults ─────────────────────────────────────────────────────────────────

class TestDefaultsExistBeforeTheClientNeedsThem:
    """A config naming no MCP servers still gets the default ones.

    They used to be added downstream, by OnIt, after the CLI had already
    decided which servers to start. The CLI saw an empty list, registered
    nothing; OnIt then added servers with no launch spec, and half the toolset
    silently went missing.
    """

    def test_an_empty_config_still_gets_the_default_servers(self):
        config = {}
        _ensure_mcp_servers(config)

        names = {s["name"] for s in config["mcp"]["servers"]}
        assert names == {"PromptsMCPServer", "ToolsLocalMCPServer",
                         "ToolsNetMCPServer"}

    def test_every_default_is_stdio(self):
        """No socket-served default remains: each server is a subprocess of
        this process, running as this user."""
        config = {}
        _ensure_mcp_servers(config)
        for server in config["mcp"]["servers"]:
            assert server["transport"] == "stdio"
            assert server.get("module"), server["name"]

    def test_the_stdio_default_is_registered_from_an_empty_config(self):
        from type.tools import _STDIO_SPECS
        config = {}
        _ensure_mcp_servers(config)
        assert "stdio://ToolsLocalMCPServer" in _STDIO_SPECS
        assert "stdio://ToolsNetMCPServer" in _STDIO_SPECS
        assert "stdio://PromptsMCPServer" in _STDIO_SPECS

    def test_the_data_path_is_pinned_in_every_launch_spec(self):
        from type.tools import _STDIO_SPECS
        config = {}
        _ensure_mcp_servers(config)
        for name in ("ToolsLocalMCPServer", "ToolsNetMCPServer",
                     "PromptsMCPServer"):
            spec = _STDIO_SPECS[f"stdio://{name}"]
            assert spec["env"]["ONIT_DATA_PATH"] == os.environ["ONIT_DATA_PATH"]

    def test_the_net_profile_is_in_the_net_server_spec(self):
        from type.tools import _STDIO_SPECS
        config = {}
        _ensure_mcp_servers(config)
        spec = _STDIO_SPECS["stdio://ToolsNetMCPServer"]
        assert "--profile" in spec["args"] and "net" in spec["args"]


# ── stdio ────────────────────────────────────────────────────────────────────

def _spawn_spec(name, data_path, profile="local"):
    """Register a per-user stdio tools server and return its pseudo-URL."""
    servers = [{"name": name, "transport": "stdio", "module": "tasks.tools",
                "profile": profile, "enabled": True}]
    register_stdio_servers(servers, str(data_path))
    return servers[0]["url"]


@pytest.mark.asyncio
class TestStdioToolsAreNotShared:
    """These spawn real subprocesses — the point is that they are separate."""

    async def test_each_user_works_in_their_own_sandbox(self, tmp_path):
        alice, bob = tmp_path / "alice", tmp_path / "bob"
        alice.mkdir()
        bob.mkdir()

        async def cwd_of(url, data_path):
            async with Client(_transport_for(url, shared=False)) as client:
                result = await client.call_tool(
                    "bash", {"command": "pwd", "data_path": str(data_path)})
                return json.loads(str(result.content[0].text))["cwd"]

        # Concurrently, the way two logged-in users would be.
        got = await asyncio.gather(
            cwd_of(_spawn_spec("ToolsLocal_alice", alice), alice),
            cwd_of(_spawn_spec("ToolsLocal_bob", bob), bob),
        )

        assert os.path.realpath(got[0]) == os.path.realpath(str(alice))
        assert os.path.realpath(got[1]) == os.path.realpath(str(bob))

    async def test_one_user_cannot_read_anothers_files(self, tmp_path):
        alice, bob = tmp_path / "alice", tmp_path / "bob"
        alice.mkdir()
        bob.mkdir()
        secret = alice / "secret.txt"
        secret.write_text("alice's notes")

        async with Client(_transport_for(_spawn_spec("ToolsLocal_bob2", bob), shared=False)) as client:
            result = await client.call_tool(
                "read_file", {"path": str(secret), "data_path": str(bob)})

        assert "Read access denied" in str(result.content[0].text)

    async def test_the_local_server_serves_only_the_local_tools(self, tmp_path):
        """The stateless tools stay on the net server; none reach this pipe."""
        async with Client(_transport_for(_spawn_spec("ToolsLocal_p", tmp_path), shared=False)) as client:
            names = {t.name for t in await client.list_tools()}

        assert "bash" in names
        assert names.isdisjoint({"search", "get_weather"})
        # Credential-bearing, so it rides with the per-user tools.
        assert "github_repo" in names

    async def test_the_net_server_serves_only_the_stateless_tools(self, tmp_path):
        """The net profile is spawnable over stdio too, and stays stateless."""
        async with Client(_transport_for(
                _spawn_spec("ToolsNet_p", tmp_path, profile="net"),
                shared=False)) as client:
            names = {t.name for t in await client.list_tools()}

        assert names == {"search", "get_weather"}
        assert "bash" not in names
        assert "github_repo" not in names