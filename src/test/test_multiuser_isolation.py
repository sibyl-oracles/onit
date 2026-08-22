"""Two people running OnIt on one machine must not share tool servers.

Before the local/net split these tests describe, the second OnIt to start
found ports 18200/18201 already bound, treated that as "my servers are
already up", and connected to the *first* user's server process. Their tools
then ran under the first user's account, and every filesystem call was
rejected because the session directory sat outside that server's data root.

The two halves of the fix are checked separately: the tools that touch a
session's files are spawned per user over a pipe, and whatever still wants a
socket gets a port found free at startup rather than a fixed one.
"""

import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastmcp import Client  # noqa: E402  (before src/ shadows the mcp package)

from src.cli import _assign_free_ports, _ensure_mcp_servers  # noqa: E402

# Imported without the ``src.`` prefix, the way lib/tools.py does at runtime.
# Reaching the same module by both names would give this test a second, empty
# copy of the stdio launch-spec registry.
from lib.tools import register_stdio_servers  # noqa: E402
from type.tools import _transport_for  # noqa: E402


# ── ports ───────────────────────────────────────────────────────────────────

class TestPortsAreNotShared:
    @staticmethod
    def _config(port=18200):
        return {"mcp": {"servers": [
            {"name": "Net", "url": f"http://127.0.0.1:{port}/sse", "enabled": True},
        ]}}

    def test_two_processes_get_different_ports(self):
        """Alice starts first; Bob must not land on Alice's port."""
        import socket

        alice = self._config()
        _assign_free_ports(alice["mcp"]["servers"], alice)
        alice_port = int(alice["mcp"]["servers"][0]["url"].rsplit(":", 1)[1].split("/")[0])

        # Alice's server is now listening there.
        held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            held.bind(("127.0.0.1", alice_port))
            held.listen(1)

            bob = self._config()
            _assign_free_ports(bob["mcp"]["servers"], bob)
            bob_port = int(bob["mcp"]["servers"][0]["url"].rsplit(":", 1)[1].split("/")[0])
        finally:
            held.close()

        assert bob_port != alice_port

    def test_a_bound_port_is_never_reused_as_someone_elses(self):
        """Several servers in one process also never collide with each other."""
        config = {"mcp": {"servers": [
            {"name": f"S{i}", "url": "http://127.0.0.1:18200/sse", "enabled": True}
            for i in range(4)
        ]}}
        assigned = _assign_free_ports(config["mcp"]["servers"], config)
        assert len(set(assigned.values())) == 4

    def test_startup_no_longer_adopts_a_running_server(self):
        """A port that answers belongs to another user, not to us.

        Startup used to read "18200 is listening" as "my servers are already
        up" and connect to them. Now it starts its own, somewhere else.
        """
        import socket

        squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            try:
                squatter.bind(("127.0.0.1", 18200))
            except OSError:
                pytest.skip("port 18200 is already in use on this machine")
            squatter.listen(1)

            config = self._config(port=18200)
            thread = MagicMock()
            with patch("src.cli.threading.Thread", return_value=thread), \
                 patch("src.cli._mcp_servers_ready", return_value=True):
                _ensure_mcp_servers(config)
        finally:
            squatter.close()

        thread.start.assert_called_once()
        assert ":18200/" not in config["mcp"]["servers"][0]["url"]


class TestPortClaimsSurviveTheBindGap:
    """The bind test alone cannot reserve a port.

    A test socket has to be closed before the server can bind, and two OnIt
    processes that tested the same port in that gap both take it — one server
    then serves both users. The claim is what holds across the gap.
    """

    def test_a_claimed_port_is_refused_to_a_second_caller(self):
        from src.mcp.servers.run import _claim_port, DEFAULT_PORT_BASE
        pytest.importorskip("fcntl")

        port = DEFAULT_PORT_BASE + 300
        assert _claim_port(port) is True
        # Same process, same lock: a re-entrant flock succeeds, so prove the
        # exclusion where it matters — from a separate process.
        import subprocess
        import textwrap
        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))!r})
            from src.mcp.servers.run import _claim_port
            sys.exit(0 if _claim_port({port}) else 3)
        """)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True)
        assert result.returncode == 3, result.stderr.decode()[-400:]

    def test_the_claim_directory_is_shared_not_per_user(self):
        """$TMPDIR is per-user on some hosts; a claim dir under it excludes
        nobody, while still looking like it works."""
        from src.mcp.servers.run import _port_claim_dir
        if os.name != "nt" and os.path.isdir("/tmp"):
            assert _port_claim_dir() == "/tmp/onit-mcp-ports"

    def test_claim_files_stay_world_writable(self):
        """A released claim leaves the file behind. If the umask narrowed it,
        the first user to touch a port would own it for good."""
        from src.mcp.servers.run import _claim_port, _port_claim_dir, DEFAULT_PORT_BASE
        pytest.importorskip("fcntl")
        import stat

        port = DEFAULT_PORT_BASE + 302
        assert _claim_port(port) is True
        mode = os.stat(os.path.join(_port_claim_dir(), str(port))).st_mode
        assert mode & stat.S_IWGRP and mode & stat.S_IWOTH

    def test_a_read_only_claim_file_can_still_be_claimed(self):
        """Simulates another user's leftover file: flock needs no write access."""
        from src.mcp.servers.run import _claim_port, _port_claim_dir, DEFAULT_PORT_BASE
        pytest.importorskip("fcntl")

        port = DEFAULT_PORT_BASE + 303
        path = os.path.join(_port_claim_dir(), str(port))
        open(path, "a").close()
        os.chmod(path, 0o444)
        try:
            assert _claim_port(port) is True
        finally:
            os.chmod(path, 0o666)

    def test_the_claim_is_released_when_the_process_ends(self):
        """No stale reservations after a crash: the kernel drops the lock."""
        from src.mcp.servers.run import _claim_port, DEFAULT_PORT_BASE
        pytest.importorskip("fcntl")

        import subprocess
        import textwrap
        port = DEFAULT_PORT_BASE + 301
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {root!r})
            from src.mcp.servers.run import _claim_port
            sys.exit(0 if _claim_port({port}) else 3)
        """)
        assert subprocess.run([sys.executable, "-c", code]).returncode == 0
        # That process is gone, so the port is claimable again.
        assert _claim_port(port) is True


# ── stdio ───────────────────────────────────────────────────────────────────

def _spawn_spec(name, data_path):
    """Register a per-user stdio tools server and return its pseudo-URL."""
    servers = [{"name": name, "transport": "stdio", "module": "tasks.tools",
                "profile": "local", "enabled": True}]
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
            async with Client(_transport_for(url)) as client:
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

        async with Client(_transport_for(_spawn_spec("ToolsLocal_bob2", bob))) as client:
            result = await client.call_tool(
                "read_file", {"path": str(secret), "data_path": str(bob)})

        assert "Read access denied" in str(result.content[0].text)

    async def test_the_stdio_server_serves_only_the_local_tools(self, tmp_path):
        """The stateless tools stay on the socket server; none reach the pipe."""
        async with Client(_transport_for(_spawn_spec("ToolsLocal_p", tmp_path))) as client:
            names = {t.name for t in await client.list_tools()}

        assert "bash" in names
        assert names.isdisjoint({"search", "get_weather"})
        # Credential-bearing, so it rides with the per-user tools.
        assert "github_repo" in names
