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

    async def test_endpoints_list_passes(self):
        """/doctor must not flag the documented endpoints-list config shape."""
        agent = _agent()
        agent.config_data = {
            "serving": {"endpoints": [
                {"host": "http://a:8000/v1", "priority": 1},
                {"host": "https://api.ollama.com", "model": "x:cloud"},
            ]},
            "mcp": {"servers": [{}]},
        }
        r = await doctor.check_config(agent)
        assert r.state == "pass", r.detail
        assert "2 endpoint(s) from serving.endpoints" in r.detail

    async def test_endpoints_list_with_bare_url_strings_passes(self):
        agent = _agent()
        agent.config_data = {
            "serving": {"endpoints": ["http://a:8000/v1"]},
            "mcp": {"servers": [{}]},
        }
        r = await doctor.check_config(agent)
        assert r.state == "pass", r.detail

    async def test_endpoints_list_without_usable_host_fails(self):
        agent = _agent()
        agent.config_data = {
            "serving": {"endpoints": [{"model": "x"}]},
            "mcp": {"servers": [{}]},
        }
        r = await doctor.check_config(agent)
        assert r.state == "fail"
        assert "no entry has a host" in r.detail


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



class TestToolGrepCheck:
    async def test_a_found_marker_passes_and_cleans_up(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        marker = "DOCTOR-GREP-TEST"

        async def fake_call(reg, name, **kw):
            assert name == "grep"
            # the probe really greps the planted directory
            hits = []
            root = kw["path"]
            for dirpath, _, files in os.walk(root):
                for fn in files:
                    with open(os.path.join(dirpath, fn)) as f:
                        for i, line in enumerate(f, 1):
                            if kw["pattern"] in line:
                                hits.append({"file": fn, "line_number": i})
            return json.dumps({"results": hits, "total_matches": len(hits),
                               "status": "success"})

        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_tool_grep(agent)
        assert r.state == "pass", r.detail
        assert not list(tmp_path.iterdir())          # probe dir removed

    async def test_zero_matches_fails(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps({"results": [],
                                                   "total_matches": 0})):
            r = await doctor.check_tool_grep(agent)
        assert r.state == "fail" and "0 times" in r.detail

    async def test_no_data_path_skips(self, tmp_path):
        agent = _agent()
        agent.data_path = ""
        r = await doctor.check_tool_grep(agent)
        assert r.state == "skip"

    async def test_missing_tool_skips(self):
        agent = _agent()
        agent.tool_registry = FakeRegistry(["bash"])
        r = await doctor.check_tool_grep(agent)
        assert r.state == "skip"


class TestToolSearchDocumentCheck:
    async def test_a_found_marker_passes_and_cleans_up(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        marker = "DOCTOR-DOC-TEST"

        async def fake_call(reg, name, **kw):
            assert name == "search_document" and kw["mode"] == "pattern"
            with open(kw["path"]) as f:
                text = f.read()
            hits = [{"line": i} for i, line in enumerate(text.splitlines(), 1)
                    if kw["pattern"] in line]
            return json.dumps({"matches": hits, "total_matches": len(hits)})

        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_tool_search_document(agent)
        assert r.state == "pass", r.detail
        assert not list(tmp_path.iterdir())          # probe file removed

    async def test_a_miss_fails(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps({"total_matches": 0})):
            r = await doctor.check_tool_search_document(agent)
        assert r.state == "fail" and "missed" in r.detail


class TestToolLocalSearchCheck:
    async def test_index_then_search_passes_and_cleans_up(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        calls = []

        async def fake_call(reg, name, **kw):
            calls.append((name, kw))
            if name == "index_documents":
                corpus = kw["path"]
                docs = [f for f in os.listdir(corpus) if f.endswith(".md")]
                return json.dumps({"status": "success", "scope": "session",
                                   "total_documents": len(docs),
                                   "total_chunks": len(docs)})
            # local_search: a working engine retrieves what was indexed —
            # the probe file's text, whatever marker the check planted in it
            with open(os.path.join(kw["path"], "probe.md")) as f:
                text = f.read()
            hit = {"file": os.path.join(kw["path"], "probe.md"), "text": text}
            return json.dumps({"results": [hit], "total_results": 1})

        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_tool_index_and_search(agent)
        assert r.state == "pass", r.detail
        # probe corpus removed and a cleanup re-index ran (3 calls: index,
        # search, re-index of the emptied corpus)
        assert [c[0] for c in calls] == ["index_documents", "local_search",
                                         "index_documents"]
        assert not list(tmp_path.iterdir())

    async def test_indexing_nothing_fails(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps({"status": "success",
                                                   "total_documents": 0})):
            r = await doctor.check_tool_index_and_search(agent)
        assert r.state == "fail" and "indexed nothing" in r.detail

    async def test_a_miss_after_indexing_fails(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)

        async def fake_call(reg, name, **kw):
            if name == "index_documents":
                return json.dumps({"status": "success", "total_documents": 1})
            return json.dumps({"results": [], "total_results": 0})

        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_tool_index_and_search(agent)
        assert r.state == "fail" and "did not retrieve" in r.detail


class TestToolSendFileCheck:
    async def test_base64_round_trip_passes_and_cleans_up(self, tmp_path):
        import base64
        agent = _agent()
        agent.data_path = str(tmp_path)
        payload = "DOCTOR-SEND-TEST"

        async def fake_call(reg, name, **kw):
            assert name == "send_file"
            with open(os.path.join(agent.data_path, kw["path"]), "rb") as f:
                data = f.read()
            return json.dumps({"status": "success",
                               "file_data_base64": base64.b64encode(data).decode()})

        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_tool_send_file(agent)
        assert r.state == "pass", r.detail
        assert not list(tmp_path.iterdir())

    async def test_a_corrupted_payload_fails(self, tmp_path):
        import base64
        agent = _agent()
        agent.data_path = str(tmp_path)
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps(
                              {"status": "success",
                               "file_data_base64": base64.b64encode(
                                   b"something else").decode()})):
            r = await doctor.check_tool_send_file(agent)
        assert r.state == "fail" and "does not decode" in r.detail

    async def test_a_failed_status_fails(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps({"status": "failed"})):
            r = await doctor.check_tool_send_file(agent)
        assert r.state == "fail" and "failed" in r.detail


class TestToolServeCheck:
    async def test_the_full_cycle_passes_and_stops_the_probe(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        calls = []

        async def fake_call(reg, tool, **kw):
            calls.append((tool, kw))
            action = kw["action"]
            if action in ("start", "status"):
                return json.dumps({"name": kw["name"], "pid": 4242,
                                   "status": "started" if action == "start"
                                   else "running"})
            return json.dumps({"name": kw["name"], "pid": 4242,
                               "status": "stopped"})

        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_tool_serve(agent)
        assert r.state == "pass", r.detail
        # start, status, then the cleanup stop
        assert [c[1]["action"] for c in calls] == ["start", "status", "stop"]

    async def test_a_leftover_running_probe_is_not_a_failure(self, tmp_path):
        """A crashed earlier run may have left the probe process up; serve
        then reports already_running, which is not a broken serve."""
        agent = _agent()
        agent.data_path = str(tmp_path)

        async def fake_call(reg, tool, **kw):
            if kw["action"] in ("start", "status"):
                return json.dumps({"name": kw["name"], "pid": 4242,
                                   "status": "already_running"})
            return json.dumps({"name": kw["name"], "pid": 4242,
                               "status": "stopped"})

        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_tool_serve(agent)
        assert r.state == "pass", r.detail

    async def test_start_that_never_runs_fails(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps({"status": "failed"})):
            r = await doctor.check_tool_serve(agent)
        assert r.state == "fail" and "did not report started" in r.detail

    async def test_status_that_loses_the_process_fails(self, tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)

        async def fake_call(reg, tool, **kw):
            if kw["action"] == "start":
                return json.dumps({"name": kw["name"], "pid": 4242,
                                   "status": "running"})
            return json.dumps({"error": "No managed process found",
                               "status": "error"})

        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_tool_serve(agent)
        assert r.state == "fail" and "status" in r.detail


class TestToolSearchCheck:
    async def test_results_pass(self):
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps(
                              [{"title": "x", "url": "u"}])):
            r = await doctor.check_tool_search(_agent())
        assert r.state == "pass", r.detail

    async def test_no_results_fails(self):
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps([])):
            r = await doctor.check_tool_search(_agent())
        assert r.state == "fail" and "no results" in r.detail

    async def test_an_error_object_fails(self):
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps(
                              {"error": "Search failed - no results from any "
                                        "provider"})):
            r = await doctor.check_tool_search(_agent())
        assert r.state == "fail" and "no results from any" in r.detail


class TestToolFetchContentCheck:
    async def test_the_known_page_passes(self):
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps(
                              {"title": "Example Domain",
                               "url": "https://example.com",
                               "content": "Example Domain\nThis domain is "
                                          "for use in illustrative "
                                          "examples."})):
            r = await doctor.check_tool_fetch_content(_agent())
        assert r.state == "pass", r.detail

    async def test_wrong_content_fails(self):
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps(
                              {"title": "404", "content": "not found"})):
            r = await doctor.check_tool_fetch_content(_agent())
        assert r.state == "fail" and "did not come back" in r.detail

    async def test_an_error_object_fails(self):
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps(
                              {"error": "connection refused"})):
            r = await doctor.check_tool_fetch_content(_agent())
        assert r.state == "fail" and "connection refused" in r.detail

    async def test_missing_tool_skips(self):
        agent = _agent()
        agent.tool_registry = FakeRegistry(["bash"])
        r = await doctor.check_tool_fetch_content(agent)
        assert r.state == "skip"


class TestKeyedToolChecks:
    """The keyed tools skip without their credential and probe with it."""

    async def test_weather_without_a_key_skips(self, monkeypatch):
        for v in ("OPENWEATHER_API_KEY", "OPENWEATHERMAP_API_KEY"):
            monkeypatch.delenv(v, raising=False)
        r = await doctor.check_tool_weather(_agent())
        assert r.state == "skip" and "OPENWEATHER" in r.detail

    async def test_github_without_a_token_skips(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        r = await doctor.check_tool_github(_agent())
        assert r.state == "skip" and "GITHUB_TOKEN" in r.detail

    async def test_weather_with_a_key_and_conditions_passes(self, monkeypatch):
        monkeypatch.setenv("OPENWEATHER_API_KEY", "k")
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps(
                              {"location": "Tokyo",
                               "current": {"description": "clear sky"}})):
            r = await doctor.check_tool_weather(_agent())
        assert r.state == "pass", r.detail

    async def test_weather_without_conditions_fails(self, monkeypatch):
        monkeypatch.setenv("OPENWEATHER_API_KEY", "k")
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps({"current": {}})):
            r = await doctor.check_tool_weather(_agent())
        assert r.state == "fail" and "no current conditions" in r.detail

    async def test_an_error_object_fails(self, monkeypatch):
        monkeypatch.setenv("OPENWEATHER_API_KEY", "k")
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps(
                              {"error": "city not found"})):
            r = await doctor.check_tool_weather(_agent())
        assert r.state == "fail" and "city not found" in r.detail

    async def test_github_listing_passes(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "t")
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps(
                              {"repos": [], "count": 0, "status": "ok"})):
            r = await doctor.check_tool_github(_agent())
        assert r.state == "pass", r.detail

    async def test_github_failure_fails(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "t")
        with patch.object(doctor, "_call_tool",
                          return_value=json.dumps(
                              {"status": "error", "error": "Bad credentials"})):
            r = await doctor.check_tool_github(_agent())
        assert r.state == "fail" and "Bad credentials" in r.detail

    async def test_an_empty_string_key_counts_as_absent(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "")
        r = await doctor.check_tool_github(_agent())
        assert r.state == "skip"


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

    async def test_ollama_cloud_suffix_passes_when_base_name_is_listed(self, monkeypatch):
        """Ollama cloud lists 'x' but serves 'x:cloud' in a chat request."""
        async def fake(host, host_key="EMPTY", timeout=15.0):
            return ["glm-5.3-flash", "gemma4:31b"]
        monkeypatch.setattr("src.model.serving.chat.list_models", fake)
        agent = _agent()
        agent.load_balancer = LoadBalancer(
            [ServerEndpoint(host="https://api.ollama.com",
                            model="glm-5.3-flash:cloud", name="ollama")],
            "sticky")
        r = await doctor.check_endpoint(agent)
        assert r.state == "pass", r.detail

    async def test_an_unknown_model_still_fails(self, monkeypatch):
        """The :cloud equivalence must not loosen the check into a no-op."""
        async def fake(host, host_key="EMPTY", timeout=15.0):
            return ["other-model"]
        monkeypatch.setattr("src.model.serving.chat.list_models", fake)
        r = await doctor.check_endpoint(_agent())
        assert r.state == "fail" and "Qwen3-30B" in r.detail


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


class TestModelFileTurnCheck:
    async def test_a_file_reading_turn_passes_and_cleans_up(self, monkeypatch,
                                                            tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)
        seen = {}

        async def fake_call(reg, name, **kw):
            assert name == "write_file"
            with open(os.path.join(agent.data_path, kw["path"]), "w") as f:
                f.write(kw["content"])
            return json.dumps({"status": "success"})

        async def fake_chat(**kw):
            seen.update(kw)
            # the model would have read the file: echo what was planted
            with open(kw["instruction"].rsplit("Path: ", 1)[1]) as f:
                return f.read()

        monkeypatch.setattr("src.model.serving.chat.chat", fake_chat)
        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_model_file_turn(agent)
        assert r.state == "pass", r.detail
        # the turn carried the registry (the model must call read_file itself)
        assert seen["tool_registry"] is agent.tool_registry
        assert "doctor-read-" in seen["instruction"]
        assert not list(tmp_path.iterdir())          # probe file removed

    async def test_a_model_that_ignores_the_file_fails(self, monkeypatch,
                                                       tmp_path):
        agent = _agent()
        agent.data_path = str(tmp_path)

        async def fake_call(reg, name, **kw):
            return json.dumps({"status": "success"})

        async def fake_chat(**kw):
            return "I am sorry 😊. Could you rephrase?"

        monkeypatch.setattr("src.model.serving.chat.chat", fake_chat)
        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            r = await doctor.check_model_file_turn(agent)
        assert r.state == "fail" and "did not land" in r.detail

    async def test_needs_the_file_tools(self):
        agent = _agent()
        agent.tool_registry = FakeRegistry(["bash"])
        r = await doctor.check_model_file_turn(agent)
        assert r.state == "skip"

    async def test_no_data_path_skips(self):
        agent = _agent()
        agent.data_path = ""
        r = await doctor.check_model_file_turn(agent)
        assert r.state == "skip"


# ── the runner ───────────────────────────────────────────────────────────────

class TestRunChecks:
    async def test_every_check_reports_and_none_raise(self, monkeypatch):
        """The whole battery runs hermetically: the network-bound probes are
        stood in for at the module's own seam, so the suite never touches
        the web, the weather, or GitHub."""
        async def fake_call(reg, name, **kw):
            return json.dumps({"status": "success", "results": [],
                               "total_matches": 0, "total_documents": 0,
                               "content": "", "current": {}, "repos": []})
        monkeypatch.setattr("src.model.serving.chat.chat",
                            lambda **kw: _async("DOCTOR-OK"))
        with patch.object(doctor, "_call_tool", side_effect=fake_call):
            results = await doctor.run_checks(_agent(), deep=False)
        names = [r.name for r in results]
        assert names == [c[0] for c in doctor.CHECKS]
        for r in results:
            assert r.state in ("pass", "fail", "skip")

    async def test_the_fast_battery_covers_every_default_tool(self):
        """The point of the expansion: every shipped tool has a check that
        exercises it, not merely lists it.  One check may cover several
        tools (the file trio, the retrieval pair)."""
        covered = {
            "bash": "tool-bash",
            "read_file": "tool-files",
            "write_file": "tool-files",
            "edit_file": "tool-files",
            "grep": "tool-grep",
            "search_document": "tool-search-document",
            "local_search": "tool-local-search",
            "index_documents": "tool-local-search",
            "send_file": "tool-send-file",
            "serve": "tool-serve",
            "search": "tool-search",
            "fetch_content": "tool-fetch-content",
            "get_weather": "tool-weather",
            "github_repo": "tool-github",
        }
        assert set(covered) == set(doctor.DEFAULT_TOOLS)   # mapping is complete
        check_names = {name for name, _, _ in doctor.CHECKS}
        for tool, check in covered.items():
            assert check in check_names, \
                f"no check exercises the {tool!r} tool (expected {check!r})"

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


async def _async(value):
    return value


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