"""Tests for src/ui/doctor.py — the session's live self-check battery.

The doctor's checks run against a *live* session, so these tests stand in
for the network at the seams the module itself defines: ``_call_tool`` for
MCP round trips, ``list_models`` for the endpoint, and a fake agent for
everything that reads attributes.  No MCP server, no model, no keychain.

What the tests pin down:

* every check reports pass/fail/skip and never raises out of ``run_checks``
* a hung check is a timed-out failure, not a hung session
* a check that cannot run here (no data_path, tool missing) skips, and a
  skip never reads as a failure in the report
* the report renders marks, counts and the closing line the transcript shows
* the ``\\doctor`` command is wired: listed, parsed, dispatched, and its
  report lands as the reply
"""

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.serving.balancer import LoadBalancer, ServerEndpoint
from src.ui import commands, doctor


# ── fakes ────────────────────────────────────────────────────────────────────

class FakeRegistry:
    """Stands in for ToolRegistry where the checks only read membership."""

    def __init__(self, tools=(), collisions=()):
        self.tools = set(tools)
        self.collisions = list(collisions)
        self.calls = []

    def __len__(self):
        return len(self.tools)

    def __getitem__(self, name):
        if name not in self.tools:
            return None

        async def handler(**kwargs):
            self.calls.append((name, kwargs))
            return self.replies.get(name, "{}")

        return handler

    replies: dict = {}


def _agent(**over):
    """A stand-in for OnIt carrying what the checks read — the same shape
    test_commands.py's ``agent`` fixture uses."""
    endpoints = [ServerEndpoint(host="http://localhost:8000/v1",
                                model="Qwen3-30B", name="vllm-a")]
    base = dict(
        config_data={"serving": {"host": "http://localhost:8000/v1"},
                     "mcp": {"servers": [{"name": "s", "url": "http://x/sse"}]}},
        mcp_servers=[{"name": "ToolsNetMCPServer",
                      "url": "http://127.0.0.1:18201/sse", "enabled": True}],
        tool_registry=FakeRegistry(doctor.DEFAULT_TOOLS),
        load_balancer=LoadBalancer(endpoints, "sticky"),
        session_id="sess-1",
        session_path="/tmp/does-not-exist.jsonl",
        data_path="",
        chat_ui=None,
        model_serving={"host": "http://localhost:8000/v1"},
        prompt_in_process=True,
        load_session_history=lambda max_turns=None, session_path=None: [],
    )
    base.update(over)
    return SimpleNamespace(**base)


def _endpoint_up(monkeypatch):
    """Patch the endpoint's model listing with a fixed answer."""
    async def fake(host, host_key="EMPTY", timeout=15.0):
        return ["Qwen3-30B"]
    monkeypatch.setattr("src.model.serving.chat.list_models", fake)


# ── result plumbing ──────────────────────────────────────────────────────────

class TestCheckResult:
    def test_marks(self):
        assert doctor.CheckResult("x", "pass").mark == "✓"
        assert doctor.CheckResult("x", "fail").mark == "✗"
        assert doctor.CheckResult("x", "skip").mark == "–"

    def test_short_collapses_whitespace(self):
        assert doctor._short("a\n  b\tc") == "a b c"

    def test_short_truncates_with_ellipsis(self):
        out = doctor._short("x" * 300, limit=50)
        assert len(out) == 50 and out.endswith("…")


# ── individual checks ────────────────────────────────────────────────────────

class TestConfigCheck:
    async def test_a_loaded_config_passes(self):
        agent = _agent()
        r = await doctor.check_config(agent)
        assert r.state == "pass"
        assert "serving.host" in r.detail or "http://localhost:8000/v1" in r.detail

    async def test_no_config_fails(self):
        agent = _agent()
        agent.config_data = {}
        r = await doctor.check_config(agent)
        assert r.state == "fail"

    async def test_missing_serving_host_fails(self):
        agent = _agent()
        agent.config_data = {"mcp": {"servers": [{}]}}
        r = await doctor.check_config(agent)
        assert r.state == "fail" and "serving.host" in r.detail


class TestMcpServersCheck:
    async def test_a_dead_port_fails_with_the_server_named(self):
        agent = _agent()
        agent.mcp_servers = [{"name": "ToolsNetMCPServer",
                              "url": "http://127.0.0.1:1/sse", "enabled": True}]
        r = await doctor.check_mcp_servers(agent)
        assert r.state == "fail"
        assert "ToolsNetMCPServer" in r.detail

    async def test_a_stdio_server_needs_only_its_launch_spec(self):
        """A stdio server has no port: reachable means spawnable."""
        from type.tools import _STDIO_SPECS, register_stdio_server, stdio_url
        url = stdio_url("DoctorFake")
        register_stdio_server(url, command="/bin/true", args=[])
        try:
            agent = _agent()
            agent.mcp_servers = [{"name": "DoctorFake", "url": url,
                                  "enabled": True}]
            r = await doctor.check_mcp_servers(agent)
            assert r.state == "pass"
        finally:
            _STDIO_SPECS.pop(url, None)

    async def test_disabled_servers_are_not_checked(self):
        agent = _agent()
        agent.mcp_servers = [{"name": "off", "url": "http://127.0.0.1:1/sse",
                              "enabled": False}]
        r = await doctor.check_mcp_servers(agent)
        assert r.state == "fail"  # no *enabled* servers left


class TestToolRegistryCheck:
    async def test_empty_registry_fails_with_the_discovery_hint(self):
        agent = _agent()
        agent.tool_registry = FakeRegistry()
        r = await doctor.check_tool_registry(agent)
        assert r.state == "fail" and "0 tools" in r.detail

    async def test_no_registry_fails(self):
        agent = _agent()
        agent.tool_registry = None
        assert (await doctor.check_tool_registry(agent)).state == "fail"

    async def test_collisions_are_reported_but_not_a_failure(self):
        """A collision is a configuration fact the operator controls, not a
        broken registry — the check names it and stays green."""
        agent = _agent()
        agent.tool_registry = FakeRegistry(
            doctor.DEFAULT_TOOLS, collisions=[("read_file", "u2", "u1")])
        r = await doctor.check_tool_registry(agent)
        assert r.state == "pass" and "read_file" in r.detail


class TestDefaultToolsCheck:
    async def test_a_missing_tool_is_named(self):
        reg = FakeRegistry(doctor.DEFAULT_TOOLS[:-1])   # drop github_repo
        agent = _agent()
        agent.tool_registry = reg
        r = await doctor.check_default_tools(agent)
        assert r.state == "fail" and "github_repo" in r.detail

    async def test_the_full_toolset_passes(self):
        agent = _agent()
        r = await doctor.check_default_tools(agent)
        assert r.state == "pass"


class TestToolBashCheck:
    async def test_a_successful_echo_passes(self):
        agent = _agent()

        async def fake_call(reg, name, **kw):
            agent.tool_registry.calls.append((name, kw))
            return json.dumps({"returncode": 0,
                               "stdout": kw["command"].split("echo ", 1)[1] + "\n"})

        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_tool_bash(agent)
        assert r.state == "pass", r.detail
        # the probe really went through the registry's bash handler
        assert agent.tool_registry.calls[0][0] == "bash"
        assert agent.tool_registry.calls[0][1]["command"].startswith("echo onit-doctor-")

    async def test_a_nonzero_exit_fails_with_stderr(self):
        agent = _agent()
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps({"returncode": 2,
                                                   "stderr": "boom"})):
            r = await doctor.check_tool_bash(agent)
        assert r.state == "fail" and "boom" in r.detail

    async def test_non_json_output_fails_without_raising(self):
        agent = _agent()
        with patch.object(doctor, "_call_tool", return_value="not json"):
            r = await doctor.check_tool_bash(agent)
        assert r.state == "fail" and "non-JSON" in r.detail

    async def test_missing_bash_tool_skips(self):
        agent = _agent()
        agent.tool_registry = FakeRegistry(["search"])
        r = await doctor.check_tool_bash(agent)
        assert r.state == "skip"


class TestToolFilesCheck:
    async def test_no_data_path_skips(self):
        """Nowhere to probe is a fact about the config, not a broken tool."""
        agent = _agent()
        agent.data_path = ""
        r = await doctor.check_tool_files(agent)
        assert r.state == "skip"

    async def test_a_full_round_trip_passes_and_cleans_up(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        marker = "DOCTOR-PROBE-TEST"

        async def fake_call(reg, name, **kw):
            path = os.path.join(agent.data_path, kw["path"])
            if name == "write_file":
                with open(path, "w") as f:
                    f.write(kw["content"])
                return json.dumps({"status": "success", "path": path})
            if name == "read_file":
                with open(path) as f:
                    return f.read()
            if name == "edit_file":
                with open(path) as f:
                    text = f.read()
                if kw["old_string"] not in text:
                    return json.dumps({"replacements": 0})
                with open(path, "w") as f:
                    f.write(text.replace(kw["old_string"], kw["new_string"]))
                return json.dumps({"replacements": 1})
            raise AssertionError(name)

        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_tool_files(agent)
        assert r.state == "pass", r.detail
        # nothing left behind
        assert not list(tmp_path.iterdir())

    async def test_edit_that_replaces_nothing_fails(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)

        async def fake_call(reg, name, **kw):
            if name == "write_file":
                with open(os.path.join(agent.data_path, kw["path"]), "w") as f:
                    f.write(kw["content"])
                return json.dumps({"status": "success"})
            if name == "read_file":
                with open(os.path.join(agent.data_path, kw["path"])) as f:
                    return f.read()
            return json.dumps({"replacements": 0})   # edit_file lies

        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_tool_files(agent)
        assert r.state == "fail" and "replaced nothing" in r.detail



class TestHarnessToolsCheck:
    async def test_notes_round_trip_passes_and_cleans_up(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        r = await doctor.check_harness_tools(agent)
        assert r.state == "pass", r.detail
        note = os.path.join(tmp_path, ".onit", "notes", "doctor_probe.md")
        assert not os.path.exists(note), "probe note not removed"

    async def test_works_without_a_data_path(self, tmp_path):
        """The temp-dir fallback is legitimate here: this check is in-process."""
        agent = _agent()
        agent.data_path = ""
        r = await doctor.check_harness_tools(agent)
        assert r.state == "pass", r.detail


class TestPromptsCheck:
    async def test_instruction_builds(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        r = await doctor.check_prompts(agent)
        assert r.state == "pass", r.detail
        assert "chars" in r.detail

    async def test_works_without_a_data_path(self):
        agent = _agent()
        agent.data_path = ""
        r = await doctor.check_prompts(agent)
        assert r.state == "pass", r.detail


class TestLoadBalancerCheck:
    async def test_an_assigned_endpoint_passes(self):
        agent = _agent()
        r = await doctor.check_load_balancer(agent)
        assert r.state == "pass" and "vllm-a" in r.detail

    async def test_no_endpoints_fails(self):
        agent = _agent()
        # LoadBalancer itself refuses an empty list, so the shape that reaches
        # the check is a balancer whose endpoints were lost some other way.
        agent.load_balancer = SimpleNamespace(endpoints=[],
                                              assigned=lambda key=None: None)
        r = await doctor.check_load_balancer(agent)
        assert r.state == "fail"

    async def test_all_cooling_down_fails(self):
        import time as _time
        agent = _agent()
        ep = agent.load_balancer.endpoints[0]
        ep.failed_at = _time.monotonic() - 1   # inside the cooldown window
        r = await doctor.check_load_balancer(agent)
        assert r.state == "fail" and "cooling down" in r.detail


class TestEndpointCheck:
    async def test_a_listed_model_passes(self, monkeypatch):
        _endpoint_up(monkeypatch)
        agent = _agent()
        r = await doctor.check_endpoint(agent)
        assert r.state == "pass" and "Qwen3-30B" in r.detail

    async def test_an_unlisted_model_fails_by_name(self, monkeypatch):
        async def fake(host, host_key="EMPTY", timeout=15.0):
            return ["other-model"]
        monkeypatch.setattr("src.model.serving.chat.list_models", fake)
        r = await doctor.check_endpoint(_agent())
        assert r.state == "fail" and "Qwen3-30B" in r.detail

    async def test_a_dead_endpoint_fails_with_the_error(self, monkeypatch):
        async def fake(host, host_key="EMPTY", timeout=15.0):
            raise ConnectionError("refused")
        monkeypatch.setattr("src.model.serving.chat.list_models", fake)
        r = await doctor.check_endpoint(_agent())
        assert r.state == "fail" and "ConnectionError" in r.detail


class TestSessionHistoryCheck:
    async def test_a_missing_session_file_still_passes(self):
        """A fresh session has no history file yet — that is not a failure."""
        r = await doctor.check_session_history(_agent())
        assert r.state == "pass"


class TestCommandsCheck:
    async def test_every_command_listed_and_parsed(self):
        r = await doctor.check_commands(_agent())
        assert r.state == "pass", r.detail


class TestLearnCheck:
    async def test_recording_on_and_writable(self, tmp_path, monkeypatch):
        from src.learn import config as learn_config
        monkeypatch.setattr(learn_config, "trajectory_dir",
                            lambda cfg: str(tmp_path / "learned"))
        agent = _agent()
        r = await doctor.check_learn(agent)
        assert r.state == "pass", r.detail

    async def test_recording_off_skips(self, monkeypatch):
        from src.learn import config as learn_config
        monkeypatch.setattr(learn_config, "autonomy", lambda cfg: 0)
        r = await doctor.check_learn(_agent())
        assert r.state == "skip"

    async def test_an_unwritable_dir_fails(self, tmp_path, monkeypatch):
        from src.learn import config as learn_config
        blocked = tmp_path / "file-not-dir"
        blocked.write_text("x")
        monkeypatch.setattr(learn_config, "trajectory_dir",
                            lambda cfg: str(blocked / "learned"))
        r = await doctor.check_learn(_agent())
        assert r.state == "fail"


# ── the deep checks ──────────────────────────────────────────────────────────

class TestModelReplyCheck:
    async def test_an_echoing_model_passes(self, monkeypatch):
        _endpoint_up(monkeypatch)
        async def fake_chat(**kw):
            return "DOCTOR-OK"
        monkeypatch.setattr("src.model.serving.chat.chat", fake_chat)
        r = await doctor.check_model_reply(_agent())
        assert r.state == "pass"

    async def test_a_silent_model_fails_with_what_came_back(self, monkeypatch):
        async def fake_chat(**kw):
            return "I am sorry 😊. Could you rephrase?"
        monkeypatch.setattr("src.model.serving.chat.chat", fake_chat)
        r = await doctor.check_model_reply(_agent())
        assert r.state == "fail" and "rephrase" in r.detail

    async def test_the_probe_is_sent_without_tools(self, monkeypatch):
        """A probe with a tool payload would let the model wander into tools."""
        seen = {}

        async def fake_chat(**kw):
            seen.update(kw)
            return "DOCTOR-OK"

        monkeypatch.setattr("src.model.serving.chat.chat", fake_chat)
        await doctor.check_model_reply(_agent())
        assert seen["tool_registry"] is None
        assert "DOCTOR-OK" in seen["instruction"]


class TestModelToolTurnCheck:
    async def test_a_tool_calling_turn_passes(self, monkeypatch):
        async def fake_chat(**kw):
            return "DOCTOR-TOOL-OK"
        monkeypatch.setattr("src.model.serving.chat.chat", fake_chat)
        r = await doctor.check_model_tool_turn(_agent())
        assert r.state == "pass"

    async def test_needs_the_bash_tool(self):
        agent = _agent()
        agent.tool_registry = FakeRegistry(["search"])
        r = await doctor.check_model_tool_turn(agent)
        assert r.state == "skip"


# ── the runner ───────────────────────────────────────────────────────────────

class TestRunChecks:
    async def test_every_check_reports_and_none_raise(self):
        results = await doctor.run_checks(_agent(), deep=False)
        names = [r.name for r in results]
        assert names == [c[0] for c in doctor.CHECKS]
        for r in results:
            assert r.state in ("pass", "fail", "skip")

    async def test_deep_adds_the_model_checks(self, monkeypatch):
        async def fake_chat(**kw):
            return "DOCTOR-OK"
        monkeypatch.setattr("src.model.serving.chat.chat", fake_chat)
        results = await doctor.run_checks(_agent(), deep=True)
        assert [r.name for r in results] == \
            [c[0] for c in doctor.CHECKS + doctor.DEEP_CHECKS]

    async def test_a_raising_check_becomes_a_failure(self):
        async def boom(agent):
            raise RuntimeError("exploded")
        with patch.object(doctor, "CHECKS", (("boom", boom, 5.0),)):
            results = await doctor.run_checks(_agent())
        assert results[0].state == "fail" and "exploded" in results[0].detail

    async def test_a_hung_check_times_out_instead_of_hanging(self):
        async def stall(agent):
            await asyncio.sleep(60)
        with patch.object(doctor, "CHECKS", (("stall", stall, 0.1),)):
            results = await doctor.run_checks(_agent(), deep=False)
        assert results[0].state == "fail" and "timed out" in results[0].detail

    async def test_on_start_is_called_for_each_check(self):
        seen = []
        with patch.object(doctor, "CHECKS", (("a", lambda ag: _pass(), 5.0),)):
            await doctor.run_checks(_agent(), deep=False, on_start=seen.append)
        assert seen == ["a"]


async def _pass():
    return doctor.CheckResult("a", "pass")


class TestRenderReport:
    def test_counts_and_marks(self):
        results = [
            doctor.CheckResult("a", "pass", "fine", 0.1),
            doctor.CheckResult("b", "fail", "broken", 0.2),
            doctor.CheckResult("c", "skip", "n/a", 0.0),
        ]
        out = doctor.render_report(results)
        assert "1 passed" in out and "1 failed" and "1 skipped" in out
        assert "✓ a" in out and "✗ b" in out and "– c" in out

    def test_all_green_says_so(self):
        results = [doctor.CheckResult("a", "pass", "fine", 0.1)]
        assert "Nothing is broken" in doctor.render_report(results)

    def test_failures_point_at_deep(self):
        results = [doctor.CheckResult("a", "fail", "bad", 0.1)]
        out = doctor.render_report(results, deep=False)
        assert "\\doctor deep" in out

    def test_deep_report_does_not_advertise_itself(self):
        results = [doctor.CheckResult("a", "fail", "bad", 0.1)]
        assert "\\doctor deep" not in doctor.render_report(results, deep=True)

    def test_skips_are_not_failures(self):
        results = [doctor.CheckResult("a", "skip", "n/a", 0.0)]
        out = doctor.render_report(results)
        assert "failed" not in out.splitlines()[0]


# ── the command wiring ───────────────────────────────────────────────────────

class TestDoctorCommand:
    def test_listed_in_help(self):
        text = commands.render_help()
        entry = next(c for c in commands.COMMANDS if c.name == "doctor")
        assert entry.usage in text and entry.summary in text

    @pytest.mark.asyncio
    async def test_dispatch_runs_the_battery(self):
        with patch("src.ui.doctor.run_checks", side_effect=fake_run_checks()):
            out = await commands.dispatch(_agent(), "\\doctor")
        assert "Self-check" in out and "1 passed" in out

    @pytest.mark.asyncio
    async def test_deep_flag_is_passed_through(self):
        seen = {}

        async def fake_run(agent, deep=False, on_start=None):
            seen["deep"] = deep
            return []

        with patch("src.ui.doctor.run_checks", side_effect=fake_run):
            await commands.dispatch(_agent(), "\\doctor deep")
        assert seen["deep"] is True

    @pytest.mark.asyncio
    async def test_a_bad_argument_is_answered_not_run(self):
        out = await commands.dispatch(_agent(), "\\doctor wat")
        assert "takes no argument" in out

    @pytest.mark.asyncio
    async def test_a_crashing_battery_costs_a_message_not_the_session(self):
        with patch("src.ui.doctor.run_checks",
                   side_effect=RuntimeError("no event loop")):
            out = await commands.dispatch(_agent(), "\\doctor")
        assert "\\doctor failed" in out


def fake_run_checks():
    async def _run(agent, deep=False, on_start=None):
        return [doctor.CheckResult("config", "pass", "ok", 0.1)]
    return _run