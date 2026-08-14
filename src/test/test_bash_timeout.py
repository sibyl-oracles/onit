"""Timeout handling for the bash tool.

Nothing above this tool bounds a command: the pooled MCP client calls tools
without a timeout and OnIt's request timeout defaults to none, so the ceiling
enforced here is the only thing between a command that never returns and a
session that hangs forever. These tests pin that down — including that a
timeout kills the whole process group, not just the shell it started.
"""

import asyncio
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import src.mcp.servers.tasks.os.bash.mcp_server as bash_mod

# A sleep long enough that surviving it is unambiguous, and distinctive enough
# to find with pgrep without matching anything else on the machine.
ORPHAN_SLEEP = "sleep 6100"


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Point the sandbox at tmp_path and clear any timeout env overrides."""
    monkeypatch.setattr(bash_mod, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(bash_mod, "DOCUMENTS_PATH", None)
    monkeypatch.setattr(bash_mod, "_SANDBOX_ENV", None)
    monkeypatch.setattr(bash_mod, "_VIOLATIONS", [])
    monkeypatch.setattr(bash_mod, "_CONTAINED", False)
    monkeypatch.delenv("ONIT_BASH_TIMEOUT", raising=False)
    monkeypatch.delenv("ONIT_BASH_MAX_TIMEOUT", raising=False)


class _Ctx:
    """Minimal Context stand-in; its presence selects the streaming exec path."""

    def __init__(self):
        self.lines = []

    async def log(self, level=None, message=None):
        self.lines.append(message)


def run_bash(tmp_path, command, **kwargs):
    fn = getattr(bash_mod.bash, "fn", bash_mod.bash)
    return json.loads(asyncio.run(fn(
        command=command, cwd=str(tmp_path), data_path=str(tmp_path), **kwargs)))


def _live_orphans():
    out = subprocess.run(["pgrep", "-f", ORPHAN_SLEEP], capture_output=True, text=True)
    return [p for p in out.stdout.split() if p]


class TestTimeoutResolution:
    def test_defaults(self):
        assert bash_mod._default_timeout() == 300
        assert bash_mod._max_timeout() == 1800

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("ONIT_BASH_TIMEOUT", "600")
        monkeypatch.setenv("ONIT_BASH_MAX_TIMEOUT", "3600")
        assert bash_mod._default_timeout() == 600
        assert bash_mod._max_timeout() == 3600

    def test_unparseable_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("ONIT_BASH_TIMEOUT", "soon")
        monkeypatch.setenv("ONIT_BASH_MAX_TIMEOUT", "")
        assert bash_mod._default_timeout() == 300
        assert bash_mod._max_timeout() == 1800

    def test_ceiling_is_never_zero(self, monkeypatch):
        """A zero ceiling would make every command time out instantly."""
        monkeypatch.setenv("ONIT_BASH_MAX_TIMEOUT", "0")
        assert bash_mod._max_timeout() >= 1


class TestTimeoutEnforcement:
    """The tool must stop a command it cannot finish, and say that it did."""

    @pytest.mark.parametrize("streaming", [False, True], ids=["sync", "streaming"])
    def test_timeout_is_reported_as_a_timeout(self, tmp_path, streaming):
        # Not "error": the model needs to tell "too slow" from "broken".
        result = run_bash(tmp_path, "sleep 30", timeout=1,
                          ctx=_Ctx() if streaming else None)
        assert result["status"] == "timeout"
        assert "timed out" in result["error"]

    def test_caller_may_exceed_the_old_300s_cap(self, tmp_path):
        """The point of the change: a slow install is no longer capped at 5min."""
        assert run_bash(tmp_path, "echo ok", timeout=1200)["status"] == "success"

    def test_request_above_the_ceiling_is_clamped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONIT_BASH_MAX_TIMEOUT", "1")
        result = run_bash(tmp_path, "sleep 30", timeout=99999, ctx=_Ctx())
        assert result["status"] == "timeout"
        assert "after 1 seconds" in result["error"]

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_non_positive_timeout_does_not_mean_unlimited(self, tmp_path, monkeypatch, timeout):
        """-1 disables OnIt's *request* timeout; here it must not disable this
        one, since no other layer would ever stop the command."""
        monkeypatch.setenv("ONIT_BASH_MAX_TIMEOUT", "1")
        result = run_bash(tmp_path, "sleep 30", timeout=timeout, ctx=_Ctx())
        assert result["status"] == "timeout"

    def test_normal_command_still_succeeds(self, tmp_path):
        ctx = _Ctx()
        result = run_bash(tmp_path, "echo hello", ctx=ctx)
        assert result["status"] == "success"
        assert result["stdout"] == "hello"
        assert ctx.lines == ["hello"]  # still streamed line by line


class TestTimeoutKillsTheProcessGroup:
    """A killed shell used to leave its children running, holding the sandbox
    directory and any ports they had opened."""

    @pytest.fixture(autouse=True)
    def _no_leaks(self):
        assert not _live_orphans(), "stale test processes from an earlier run"
        yield
        for pid in _live_orphans():
            subprocess.run(["kill", "-9", pid], capture_output=True)

    @pytest.mark.skipif(bash_mod.IS_WINDOWS, reason="POSIX process groups")
    @pytest.mark.parametrize("streaming", [False, True], ids=["sync", "streaming"])
    def test_background_child_does_not_survive(self, tmp_path, streaming):
        result = run_bash(tmp_path, f"{ORPHAN_SLEEP} & sleep 30", timeout=1,
                          ctx=_Ctx() if streaming else None)
        assert result["status"] == "timeout"
        time_to_die = 0.5
        asyncio.run(asyncio.sleep(time_to_die))
        assert _live_orphans() == [], "background child outlived the timeout"
