"""Tests for src/onit.py — OnIt, OnItA2AExecutor, ClientDisconnectMiddleware."""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.onit import (OnIt, OnItA2AExecutor, ClientDisconnectMiddleware,
                      STOP_TAG, StreamingAdapter, friendly_tool_status)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_config(tmp_path, overrides=None):
    """Build a minimal config dict."""
    cfg = {
        "serving": {
            "host": "http://localhost:8000/v1",
            "think": False,
            "max_tokens": 1024,
        },
        "mcp": {
            "servers": [
                {
                    "name": "PromptsMCPServer",
                    "url": "http://127.0.0.1:18200/sse",
                    "enabled": True,
                },
            ],
        },
        "session_path": str(tmp_path / "sessions"),
        "theme": "white",
        "verbose": False,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def _mock_discover():
    """Patch discover_tools to return an empty registry."""
    from type.tools import ToolRegistry
    return patch("src.onit.discover_tools", return_value=ToolRegistry())


# ── friendly_tool_status / StreamingAdapter.tool_log ────────────────────────

class TestFriendlyToolStatus:
    def test_unwraps_mcp_dict_payload(self):
        data = {"msg": "Collecting opencv-python-headless (from easyocr)",
                "extra": None}
        assert friendly_tool_status("bash", data) == "Downloading required files…"

    def test_common_operations_are_humanized(self):
        assert friendly_tool_status("bash", "Installing collected packages: numpy") \
            == "Installing components…"
        assert friendly_tool_status("bash", "Cloning into 'repo'...") \
            == "Downloading source code…"

    def test_unrecognized_line_kept_one_line_with_tool_name(self):
        out = friendly_tool_status("search", "first line\nsecond line")
        assert out == "search: first line"

    def test_bash_output_is_never_echoed(self):
        """Command output describes how the agent works, not the task."""
        assert friendly_tool_status("bash", "gcc: error: no input files") == ""

    def test_long_line_truncated(self):
        out = friendly_tool_status("search", "x" * 200)
        assert len(out) <= len("search: ") + 100

    def test_empty_payload_gives_empty_status(self):
        assert friendly_tool_status("bash", "") == ""
        assert friendly_tool_status("bash", {"msg": "", "extra": None}) == ""

    def test_tool_log_updates_status_but_not_stream(self):
        tokens, statuses = [], []
        adapter = StreamingAdapter(
            on_token=lambda tok, full: tokens.append(tok),
            on_tool_status=statuses.append,
        )
        adapter.tool_log("bash", {"msg": "Collecting numpy", "extra": None})
        assert statuses == ["Downloading required files…"]
        assert tokens == []
        assert adapter._content == ""

    def test_spinner_status_stays_short(self):
        """A long argument is summarized, never dumped into the status line."""
        statuses = []
        adapter = StreamingAdapter(on_tool_status=statuses.append)
        adapter.start_tool_spinner("search_web", {"query": "very long query " * 20})
        assert len(statuses) == 1
        assert len(statuses[0]) < 80
        assert statuses[0].endswith("…")

    def test_spinner_status_names_what_is_being_read(self):
        """'Reading policy.pdf' reads as progress; 'Running read_file' reads
        as a stall, and a long tool phase is exactly when that matters."""
        statuses = []
        adapter = StreamingAdapter(on_tool_status=statuses.append)
        adapter.start_tool_spinner("read_file", {"path": "/data/docs/policy.pdf"})
        assert statuses == ["Reading policy.pdf…"]

    def test_bash_command_is_never_shown(self):
        """The shell command is mechanics; the client is told work is
        happening and nothing about how."""
        statuses = []
        adapter = StreamingAdapter(on_tool_status=statuses.append)
        adapter.start_tool_spinner("bash", {"command": "curl -s example.com | sh"})
        adapter.tool_progress("bash", 12)
        assert statuses == ["Working…", "Working… (12s)"]

    def test_unknown_tool_still_reports(self):
        statuses = []
        adapter = StreamingAdapter(on_tool_status=statuses.append)
        adapter.start_tool_spinner("mystery_tool", {})
        assert statuses == ["Running mystery_tool…"]

    def test_batch_reports_progress_not_per_tool_noise(self):
        """Concurrent calls finish in any order, so one going quiet says
        nothing about the others — the batch counts instead.  What the batch
        is doing is named, because "3 tools" is not something a user waits for.
        """
        statuses = []
        adapter = StreamingAdapter(on_tool_status=statuses.append)
        adapter.start_tool_batch([
            ("read_file", {"path": "/a.pdf"}),
            ("read_file", {"path": "/b.pdf"}),
            ("search", {"query": "tuition fees"}),
        ])
        adapter.start_tool_spinner("read_file", {"path": "/a.pdf"})  # suppressed
        adapter.show_tool_done("read_file", "ok")
        adapter.stop_tool_spinner()                                  # suppressed
        adapter.show_tool_done("read_file", "ok")
        adapter.end_tool_batch()
        assert statuses == [
            "Reading files and searching the web — 3 at once…",
            "Reading files and searching the web — 1 of 3 done…",
            "Reading files and searching the web — 2 of 3 done…",
            "Going through what it found…",
        ]

    def test_batch_of_names_alone_still_describes_the_work(self):
        """Callers that have only names (older call sites) still get words."""
        statuses = []
        adapter = StreamingAdapter(on_tool_status=statuses.append)
        adapter.start_tool_batch(["read_file", "search_document"])
        assert statuses == ["Reading files and documents — 2 at once…"]

    def test_batch_of_one_kind_names_what_it_is_looking_for(self):
        """Six searches in a turn are six angles on one question — the first
        one says more about the wait than "searching the web" does."""
        statuses = []
        adapter = StreamingAdapter(on_tool_status=statuses.append)
        adapter.start_tool_batch([
            ("search", {"query": "Vietnam GDP 2024"}),
            ("search", {"query": "Indonesia GDP 2024"}),
        ])
        adapter.show_tool_done("search", "ok")
        assert statuses == [
            "Searching the web for Vietnam GDP 2024 (+1 more)…",
            "Searching the web — 1 of 2 done…",
        ]

    def test_gap_between_tools_says_what_the_model_is_doing(self):
        """The model reading tool output is most of a long run's wall time;
        going blank there is what makes a run look stalled."""
        statuses = []
        adapter = StreamingAdapter(on_tool_status=statuses.append)
        adapter.start_tool_spinner("search", {"query": "peso rate"})
        adapter.show_tool_done("search", "ok")
        assert statuses == ["Searching the web for peso rate…",
                            "Going through what it found…"]

    def test_url_is_reported_as_its_site(self):
        """A tracking-laden URL fills the line without saying anything."""
        statuses = []
        adapter = StreamingAdapter(on_tool_status=statuses.append)
        adapter.start_tool_spinner(
            "fetch_content", {"url": "https://www.msn.com/en-ph/money/x?ocid=abc"})
        assert statuses == ["Fetching msn.com…"]

    def test_unnamed_tool_during_fact_check_reports_the_phase(self):
        """The verifier calls things this map has no words for; "Running
        issues…" is worse than saying which phase the run is in."""
        statuses = []
        adapter = StreamingAdapter(on_tool_status=statuses.append)
        adapter.verification_start()
        adapter.start_tool_spinner("issues", {})
        adapter.show_tool_done("issues", "ok")
        assert statuses == ["Checking the facts…"] * 3

    def test_answer_start_fires_once_after_tools_have_run(self):
        starts, tokens = [], []
        adapter = StreamingAdapter(
            on_token=lambda tok, full: tokens.append(tok),
            on_answer_start=lambda: starts.append(1),
        )
        # A prose phase before any tool ran is not the answer.
        adapter.stream_start()
        adapter.stream_token("Let me look that up.")
        assert starts == []

        adapter.set_turn_context(tools_run=2)
        adapter.stream_start()
        adapter.stream_token("The scholarship ")
        adapter.stream_token("covers tuition.")
        assert starts == [1]

    def test_reasoning_is_forwarded_apart_from_the_answer(self):
        """The client shows the working while it happens and folds it away
        when the answer lands — which it can only do if the two never arrive
        on the same channel."""
        thoughts, tokens = [], []
        adapter = StreamingAdapter(
            on_token=lambda tok, full: tokens.append(tok),
            on_think=thoughts.append,
        )
        adapter.stream_start()
        adapter.stream_think_token("Weighing ")
        adapter.stream_think_token("the options.")
        adapter.stream_token("The answer.")
        assert "".join(thoughts) == "Weighing the options."
        assert "".join(tokens) == "The answer."

    def test_inline_think_tags_are_stripped_from_the_reasoning(self):
        """A model that keeps its reasoning in `content` brings the delimiters
        with it; the panel shows reasoning, not markup."""
        thoughts = []
        adapter = StreamingAdapter(on_think=thoughts.append)
        adapter.stream_start()
        adapter.stream_think_token("<think>planning")
        adapter.stream_think_token("</think>")
        assert "".join(thoughts) == "planning"

    def test_reasoning_is_dropped_when_no_one_asked_for_it(self):
        """A caller with nowhere to show the working pays nothing for it."""
        adapter = StreamingAdapter(on_token=lambda tok, full: None)
        adapter.stream_start()
        adapter.stream_think_token("thinking hard")  # must not raise
        assert adapter._pending == []

    def test_fact_check_reports_as_status_not_as_answer_text(self):
        """The draft is already on the client's screen; the check is work
        happening behind it, so it must not append to the message."""
        statuses, tokens = [], []
        adapter = StreamingAdapter(
            on_token=lambda tok, full: tokens.append(tok),
            on_tool_status=statuses.append,
        )
        adapter.verification_start()
        adapter.verification_end("The corrected answer.", "3.1M, not 4.2M")
        assert statuses == ["Checking the facts…", ""]
        assert tokens == []


# ── OnIt.__init__ ───────────────────────────────────────────────────────────

class TestOnItInit:
    def test_init_from_dict(self, tmp_path):
        cfg = _make_config(tmp_path)
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.status == "initialized"
        assert onit.status == "initialized"

    def test_init_from_yaml_path(self, tmp_path):
        import yaml
        cfg = _make_config(tmp_path)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))
        with _mock_discover():
            onit = OnIt(config=str(config_file))
        assert onit.status == "initialized"

    def test_init_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            with _mock_discover():
                OnIt(config="/nonexistent/path.yaml")

    def test_init_invalid_type_raises(self):
        with pytest.raises(TypeError):
            with _mock_discover():
                OnIt(config=12345)

    def test_init_prompts_server_disabled_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg["mcp"]["servers"] = [
            {"name": "PromptsMCPServer", "url": "http://x", "enabled": False},
        ]
        with _mock_discover():
            with pytest.raises(ValueError, match="PromptsMCPServer"):
                OnIt(config=cfg)

    def test_init_no_host_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ONIT_HOST", raising=False)
        cfg = _make_config(tmp_path)
        del cfg["serving"]["host"]
        with _mock_discover():
            with pytest.raises(ValueError, match="No serving host"):
                OnIt(config=cfg)


# ── OnIt.initialize ────────────────────────────────────────────────────────

class TestOnItInitialize:
    def test_mcp_host_override(self, tmp_path):
        """mcp_host moves the socket-served servers, and only those.

        A stdio server is a subprocess of this process reached over a pipe, so
        there is no host to point elsewhere; rewriting its address would leave
        it unreachable.
        """
        cfg = _make_config(tmp_path)
        cfg["mcp"]["mcp_host"] = "192.168.1.100"
        with _mock_discover():
            onit = OnIt(config=cfg)

        moved = [s for s in onit.mcp_servers if not s["url"].startswith("stdio://")]
        assert moved, "expected at least one socket-served MCP server"
        for server in moved:
            assert "192.168.1.100" in server["url"]

        for server in onit.mcp_servers:
            if server["url"].startswith("stdio://"):
                assert server["url"] == f"stdio://{server['name']}"

    def test_env_host_fallback(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        del cfg["serving"]["host"]
        monkeypatch.setenv("ONIT_HOST", "http://env-host:8000/v1")
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.model_serving["host"] == "http://env-host:8000/v1"

    def test_single_host_yields_one_endpoint_balancer(self, tmp_path):
        cfg = _make_config(tmp_path)
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.load_balancer.hosts == ["http://localhost:8000/v1"]
        assert onit.load_balancer.acquire().host == "http://localhost:8000/v1"

    def test_host2_enables_load_balancing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONIT_HOST2_KEY", "sk-second-key")
        cfg = _make_config(tmp_path)
        cfg["serving"]["host2"] = "https://api.ollama.com"
        cfg["serving"]["model2"] = "glm-5.1:cloud"
        cfg["serving"]["load_balancer"] = "least_busy"
        with _mock_discover():
            onit = OnIt(config=cfg)
        lb = onit.load_balancer
        assert lb.algorithm == "least_busy"
        assert lb.hosts == ["http://localhost:8000/v1", "https://api.ollama.com"]
        second = lb.endpoints[1]
        assert second.host_key == "sk-second-key"
        assert second.model == "glm-5.1:cloud"

    def test_host2_from_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONIT_HOST2", "http://gpu2:8000/v1")
        cfg = _make_config(tmp_path)
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.load_balancer.hosts == [
            "http://localhost:8000/v1", "http://gpu2:8000/v1"]

    def test_duplicate_host2_ignored(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg["serving"]["host2"] = cfg["serving"]["host"]
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert len(onit.load_balancer.endpoints) == 1

    def test_ollama_fallback_only_defaults_true(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg["serving"]["host2"] = "https://api.ollama.com"
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.load_balancer.ollama_fallback_only is True

    def test_ollama_fallback_only_config_reaches_balancer(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg["serving"]["host2"] = "https://api.ollama.com"
        cfg["serving"]["ollama_fallback_only"] = False
        with _mock_discover():
            onit = OnIt(config=cfg)
        lb = onit.load_balancer
        assert lb.ollama_fallback_only is False
        # Ollama is now a normal rotation member, not a fallback.
        seen = {lb.acquire(key=f"s{i}").host for i in range(50)}
        assert seen == {"http://localhost:8000/v1", "https://api.ollama.com"}

    def test_endpoints_list_replaces_host_pair(self, tmp_path):
        cfg = _make_config(tmp_path)
        del cfg["serving"]["host"]
        cfg["serving"]["endpoints"] = [
            {"name": "gpu-a", "host": "http://gpu-a:8000/v1", "priority": 1},
            {"name": "gpu-b", "host": "http://gpu-b:8000/v1", "priority": 1},
            {"name": "ollama", "host": "https://ollama.com",
             "model": "glm-5.1:cloud", "priority": 2},
        ]
        with _mock_discover():
            onit = OnIt(config=cfg)
        lb = onit.load_balancer
        assert lb.uses_priority is True
        assert [ep.name for ep in lb.endpoints] == ["gpu-a", "gpu-b", "ollama"]
        assert lb.endpoints[2].model == "glm-5.1:cloud"
        # serving.host is backfilled from the preferred endpoint for the
        # config readers that still expect a single host.
        assert onit.model_serving["host"] == "http://gpu-a:8000/v1"

    def test_endpoints_list_routes_by_priority(self, tmp_path):
        cfg = _make_config(tmp_path)
        del cfg["serving"]["host"]
        cfg["serving"]["load_balancer"] = "round_robin"
        cfg["serving"]["endpoints"] = [
            {"name": "primary", "host": "http://gpu-a:8000/v1", "priority": 1},
            {"name": "backup", "host": "http://gpu-b:8000/v1", "priority": 2},
        ]
        with _mock_discover():
            onit = OnIt(config=cfg)
        lb = onit.load_balancer
        for _ in range(4):
            ep = lb.acquire()
            assert ep.name == "primary"
            lb.release(ep, success=True)

    def test_endpoints_list_ranks_ollama_first_when_asked(self, tmp_path):
        """The exact case ollama_fallback_only used to make inexpressible."""
        cfg = _make_config(tmp_path)
        del cfg["serving"]["host"]
        cfg["serving"]["endpoints"] = [
            {"host": "https://ollama.com", "model": "glm-5.1:cloud",
             "priority": 1},
            {"host": "http://gpu-a:8000/v1", "priority": 2},
        ]
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.load_balancer.acquire().host == "https://ollama.com"

    def test_endpoints_list_accepts_url_strings(self, tmp_path):
        cfg = _make_config(tmp_path)
        del cfg["serving"]["host"]
        cfg["serving"]["endpoints"] = ["http://gpu-a:8000/v1",
                                       "http://gpu-b:8000/v1"]
        with _mock_discover():
            onit = OnIt(config=cfg)
        lb = onit.load_balancer
        assert lb.hosts == ["http://gpu-a:8000/v1", "http://gpu-b:8000/v1"]
        # No priorities given — all endpoints share one tier.
        assert lb.uses_priority is False

    def test_endpoints_list_skips_bad_and_duplicate_entries(self, tmp_path):
        cfg = _make_config(tmp_path)
        del cfg["serving"]["host"]
        cfg["serving"]["endpoints"] = [
            {"host": "http://gpu-a:8000/v1"},
            {"model": "no-host-here"},
            {"host": "http://gpu-a:8000/v1"},   # duplicate
            "http://gpu-b:8000/v1",
        ]
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.load_balancer.hosts == ["http://gpu-a:8000/v1",
                                            "http://gpu-b:8000/v1"]

    def test_endpoints_list_takes_precedence_over_host_pair(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg["serving"]["host2"] = "http://legacy2:8000/v1"
        cfg["serving"]["endpoints"] = [{"host": "http://gpu-a:8000/v1"}]
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.load_balancer.hosts == ["http://gpu-a:8000/v1"]

    def test_endpoints_list_with_no_usable_host_raises(self, tmp_path,
                                                       monkeypatch):
        monkeypatch.delenv("ONIT_HOST", raising=False)
        cfg = _make_config(tmp_path)
        del cfg["serving"]["host"]
        cfg["serving"]["endpoints"] = [{"model": "no-host-here"}]
        with _mock_discover():
            with pytest.raises(ValueError, match="no entry has a usable"):
                OnIt(config=cfg)

    def test_endpoints_list_bad_priority_falls_back_to_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        del cfg["serving"]["host"]
        cfg["serving"]["endpoints"] = [
            {"host": "http://gpu-a:8000/v1", "priority": "high"},
            {"host": "http://gpu-b:8000/v1", "priority": 1},
        ]
        with _mock_discover():
            onit = OnIt(config=cfg)
        lb = onit.load_balancer
        assert lb.endpoints[0].priority == 0
        assert lb.acquire().name == "server1"  # default 0 outranks 1

    def test_session_path_created(self, tmp_path):
        cfg = _make_config(tmp_path)
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert os.path.exists(onit.session_path)

    def test_negative_timeout_becomes_none(self, tmp_path):
        cfg = _make_config(tmp_path, {"timeout": -1})
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.timeout is None

    def test_prompt_intro_from_config(self, tmp_path):
        cfg = _make_config(tmp_path, {"prompt_intro": "I am a custom bot."})
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.prompt_intro == "I am a custom bot."

    def test_prompt_intro_default_none(self, tmp_path):
        cfg = _make_config(tmp_path)
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.prompt_intro is None

    def test_placeholder_credentials_nullified(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg["web_google_client_id"] = "YOUR_GOOGLE_CLIENT_ID_HERE"
        cfg["web_google_client_secret"] = "YOUR_SECRET_HERE"
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.web_google_client_id is None
        assert onit.web_google_client_secret is None


# ── OnIt.load_session_history ───────────────────────────────────────────────

class TestLoadSessionHistory:
    def test_reads_jsonl(self, tmp_path):
        cfg = _make_config(tmp_path)
        with _mock_discover():
            onit = OnIt(config=cfg)
        # Write entries to the session file
        with open(onit.session_path, "w") as f:
            f.write(json.dumps({"task": "q1", "response": "a1"}) + "\n")
            f.write(json.dumps({"task": "q2", "response": "a2"}) + "\n")

        history = onit.load_session_history()
        assert len(history) == 2
        assert history[0]["task"] == "q1"

    def test_skips_malformed_lines(self, tmp_path):
        cfg = _make_config(tmp_path)
        with _mock_discover():
            onit = OnIt(config=cfg)
        with open(onit.session_path, "w") as f:
            f.write("not json\n")
            f.write(json.dumps({"task": "ok", "response": "yes"}) + "\n")
            f.write(json.dumps({"unrelated": "data"}) + "\n")

        history = onit.load_session_history()
        assert len(history) == 1

    def test_returns_last_n(self, tmp_path):
        cfg = _make_config(tmp_path)
        with _mock_discover():
            onit = OnIt(config=cfg)
        with open(onit.session_path, "w") as f:
            for i in range(30):
                f.write(json.dumps({"task": f"q{i}", "response": f"a{i}"}) + "\n")

        history = onit.load_session_history(max_turns=5)
        assert len(history) == 5
        assert history[0]["task"] == "q25"

    def test_empty_file(self, tmp_path):
        cfg = _make_config(tmp_path)
        with _mock_discover():
            onit = OnIt(config=cfg)
        history = onit.load_session_history()
        assert history == []


# ── OnIt.process_task ───────────────────────────────────────────────────────

def _make_onit_for_async(tmp_path, overrides=None):
    """Create an OnIt instance safe for use within async tests.

    OnIt.__init__ calls asyncio.run(discover_tools(...)) which conflicts with
    the running event loop in pytest-asyncio.  We patch asyncio.run to simply
    return an empty ToolRegistry (the discover_tools mock is never actually
    awaited in this path).
    """
    from type.tools import ToolRegistry

    cfg = _make_config(tmp_path, overrides)
    empty_registry = ToolRegistry()

    with patch("src.onit.discover_tools", return_value=empty_registry), \
         patch("src.onit.asyncio.run", return_value=empty_registry):
        onit = OnIt(config=cfg)
    return onit


class TestProcessTask:
    @pytest.mark.asyncio
    async def test_returns_response(self, tmp_path):
        onit = _make_onit_for_async(tmp_path)
        onit.safety_queue = asyncio.Queue()

        # Mock prompt client
        mock_prompt_msg = MagicMock()
        mock_prompt_msg.content.text = "Instruction text"
        mock_prompt_result = MagicMock()
        mock_prompt_result.messages = [mock_prompt_msg]

        mock_client = AsyncMock()
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.onit.Client", return_value=mock_client), \
             patch("src.onit.chat", new_callable=AsyncMock, return_value="The answer"):
            result = await onit.process_task("What is 2+2?")

        assert result == "The answer"

    @pytest.mark.asyncio
    async def test_returns_error_on_none(self, tmp_path):
        onit = _make_onit_for_async(tmp_path)
        onit.safety_queue = asyncio.Queue()

        mock_prompt_msg = MagicMock()
        mock_prompt_msg.content.text = "Instruction"
        mock_prompt_result = MagicMock()
        mock_prompt_result.messages = [mock_prompt_msg]

        mock_client = AsyncMock()
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.onit.Client", return_value=mock_client), \
             patch("src.onit.chat", new_callable=AsyncMock, return_value=None):
            result = await onit.process_task("fail")

        assert "rephrase" in result

    @pytest.mark.asyncio
    async def test_prompt_intro_passed_to_chat(self, tmp_path):
        onit = _make_onit_for_async(tmp_path, {"prompt_intro": "I am a custom bot."})
        onit.safety_queue = asyncio.Queue()

        mock_prompt_msg = MagicMock()
        mock_prompt_msg.content.text = "Instruction text"
        mock_prompt_result = MagicMock()
        mock_prompt_result.messages = [mock_prompt_msg]

        mock_client = AsyncMock()
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_chat = AsyncMock(return_value="answer")
        with patch("src.onit.Client", return_value=mock_client), \
             patch("src.onit.chat", mock_chat):
            await onit.process_task("test")

        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs.get("prompt_intro") == "I am a custom bot."

    async def _chat_kwargs_for(self, tmp_path, overrides):
        """Run one task and return the kwargs process_task handed to chat()."""
        onit = _make_onit_for_async(tmp_path, overrides)
        onit.safety_queue = asyncio.Queue()

        mock_prompt_msg = MagicMock()
        mock_prompt_msg.content.text = "Instruction text"
        mock_prompt_result = MagicMock()
        mock_prompt_result.messages = [mock_prompt_msg]

        mock_client = AsyncMock()
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_chat = AsyncMock(return_value="answer")
        with patch("src.onit.Client", return_value=mock_client), \
             patch("src.onit.chat", mock_chat):
            await onit.process_task("test")
        return mock_chat.call_args.kwargs

    @pytest.mark.asyncio
    async def test_loop_policy_reaches_chat(self, tmp_path):
        """serving.max_chat_iterations et al. are forwarded, not dropped."""
        kwargs = await self._chat_kwargs_for(tmp_path, {"serving": {
            "host": "http://localhost:8000/v1",
            "max_chat_iterations": 25,
            "max_repeated_tool_calls": 8,
            "max_planning_continuations": 4,
        }})
        assert kwargs.get("max_chat_iterations") == 25
        assert kwargs.get("max_repeated_tool_calls") == 8
        assert kwargs.get("max_planning_continuations") == 4

    @pytest.mark.asyncio
    async def test_fact_check_keys_are_forwarded(self, tmp_path):
        kwargs = await self._chat_kwargs_for(tmp_path, {"serving": {
            "host": "http://localhost:8000/v1",
            "verify_answers": False,
            "verify_max_tool_turns": 0,
            "verify_timeout_s": 1.5,
            "verify_background": False,
            "verify_trusted_domains": ["docs.internal.corp"],
        }})
        assert kwargs.get("verify_answers") is False
        assert kwargs.get("verify_max_tool_turns") == 0
        assert kwargs.get("verify_timeout_s") == 1.5
        assert kwargs.get("verify_background") is False
        assert kwargs.get("verify_trusted_domains") == ["docs.internal.corp"]

    @pytest.mark.asyncio
    async def test_unset_loop_policy_keys_are_not_forwarded(self, tmp_path):
        """An unset key must not appear at all, so chat()'s signature stays the
        single place each default is defined."""
        kwargs = await self._chat_kwargs_for(tmp_path, {"serving": {
            "host": "http://localhost:8000/v1",
            "max_chat_iterations": 25,
        }})
        assert kwargs.get("max_chat_iterations") == 25
        for absent in ("max_repeated_tool_calls", "max_api_retries",
                       "max_planning_continuations", "max_ack_continuations",
                       "max_final_continuations", "verify_answers",
                       "verify_max_tool_turns"):
            assert absent not in kwargs


# ── fact-checks that outlive the answer ─────────────────────────────────────

class TestBackgroundChecks:
    """The deep check runs after process_task returns, so what matters is who
    owns it: it must be cancellable, it must correct the saved answer, and it
    must not exist at all for a caller that cannot show what it finds."""

    async def _run(self, tmp_path, correction_callback=None, answer="The answer"):
        onit = _make_onit_for_async(tmp_path)
        onit.safety_queue = asyncio.Queue()
        mock_prompt_msg = MagicMock()
        mock_prompt_msg.content.text = "Instruction"
        mock_prompt_result = MagicMock()
        mock_prompt_result.messages = [mock_prompt_msg]
        mock_client = AsyncMock()
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_chat = AsyncMock(return_value=answer)
        with patch("src.onit.Client", return_value=mock_client), \
             patch("src.onit.chat", mock_chat):
            result = await onit.process_task(
                "q", correction_callback=correction_callback)
        return onit, mock_chat, result

    @pytest.mark.asyncio
    async def test_a_caller_that_cannot_report_gets_no_deep_check(self, tmp_path):
        _, mock_chat, _ = await self._run(tmp_path)
        assert "background_verify" not in mock_chat.call_args.kwargs

    @pytest.mark.asyncio
    async def test_a_caller_that_can_report_is_offered_one(self, tmp_path):
        _, mock_chat, _ = await self._run(tmp_path, correction_callback=lambda *a: None)
        assert callable(mock_chat.call_args.kwargs.get("background_verify"))

    @pytest.mark.asyncio
    async def test_the_next_task_ends_the_check_behind_the_last_answer(self, tmp_path):
        """A correction to an answer the user has moved past is not worth the
        tokens, and would land under the wrong answer."""
        onit = _make_onit_for_async(tmp_path)
        onit.safety_queue = asyncio.Queue()
        started = asyncio.Event()

        async def _never_finishes():
            started.set()
            await asyncio.sleep(3600)

        onit._schedule_background_check(onit.session_id, _never_finishes(),
                                        onit.session_path)
        await started.wait()
        task = onit.background_checks[onit.session_id]

        mock_prompt_result = MagicMock()
        mock_prompt_result.messages = [MagicMock()]
        mock_prompt_result.messages[0].content.text = "Instruction"
        mock_client = AsyncMock()
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("src.onit.Client", return_value=mock_client), \
             patch("src.onit.chat", new_callable=AsyncMock, return_value="next"):
            await onit.process_task("something else")

        await asyncio.sleep(0)
        assert task.cancelled() or task.done()
        assert onit.session_id not in onit.background_checks

    @pytest.mark.asyncio
    async def test_a_correction_rewrites_the_saved_answer(self, tmp_path):
        """The session file is what the next turn is built from. A correction
        that is not written there is one the model contradicts later, citing
        itself."""
        onit, _, _ = await self._run(tmp_path, correction_callback=lambda *a: None)
        saved = [json.loads(line) for line
                 in open(onit.session_path, encoding="utf-8").read().splitlines()
                 if line.strip()]
        assert saved[-1]["response"] == "The answer"

        async def _found():
            return "The corrected answer", "3.1M, not 4.2M"

        onit._schedule_background_check(onit.session_id, _found(), onit.session_path)
        await onit.background_checks[onit.session_id]

        saved = [json.loads(line) for line
                 in open(onit.session_path, encoding="utf-8").read().splitlines()
                 if line.strip()]
        assert len(saved) == 1                      # amended, not appended
        assert saved[-1]["response"] == "The corrected answer"
        assert saved[-1]["fact_checked"] is True

    @pytest.mark.asyncio
    async def test_a_clean_check_leaves_the_saved_answer_alone(self, tmp_path):
        onit, _, _ = await self._run(tmp_path, correction_callback=lambda *a: None)

        async def _clean():
            return "The answer", ""

        onit._schedule_background_check(onit.session_id, _clean(), onit.session_path)
        await onit.background_checks[onit.session_id]
        saved = [json.loads(line) for line
                 in open(onit.session_path, encoding="utf-8").read().splitlines()
                 if line.strip()]
        assert saved[-1]["response"] == "The answer"
        assert "fact_checked" not in saved[-1]

    @pytest.mark.asyncio
    async def test_a_failed_check_costs_nothing(self, tmp_path):
        onit, _, _ = await self._run(tmp_path, correction_callback=lambda *a: None)

        async def _boom():
            raise RuntimeError("endpoint down")

        onit._schedule_background_check(onit.session_id, _boom(), onit.session_path)
        await onit.background_checks[onit.session_id]
        saved = [json.loads(line) for line
                 in open(onit.session_path, encoding="utf-8").read().splitlines()
                 if line.strip()]
        assert saved[-1]["response"] == "The answer"



# ── OnItA2AExecutor ─────────────────────────────────────────────────────────

class TestOnItA2AExecutor:
    @pytest.mark.asyncio
    async def test_execute_calls_process_task(self, tmp_path):
        mock_onit = MagicMock()
        mock_onit.process_task = AsyncMock(return_value="result text")
        mock_onit.session_path = str(tmp_path / "sessions" / "test.jsonl")
        mock_onit.config_data = {"data_path": str(tmp_path / "data")}
        os.makedirs(os.path.dirname(mock_onit.session_path), exist_ok=True)

        executor = OnItA2AExecutor(mock_onit)

        context = MagicMock()
        context.get_user_input.return_value = "test task"
        context.context_id = "ctx-123"
        context.task_id = "task-456"
        context.message = MagicMock()
        context.message.parts = []

        event_queue = MagicMock()
        event_queue.enqueue_event = AsyncMock()

        await executor.execute(context, event_queue)

        mock_onit.process_task.assert_awaited_once()
        call_kwargs = mock_onit.process_task.call_args
        assert call_kwargs[0][0] == "test task"
        assert call_kwargs[1]["images"] is None
        assert "session_path" in call_kwargs[1]
        assert "data_path" in call_kwargs[1]
        assert "safety_queue" in call_kwargs[1]
        event_queue.enqueue_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_raises_on_no_message(self):
        mock_onit = MagicMock()
        executor = OnItA2AExecutor(mock_onit)

        context = MagicMock()
        context.message = None

        with pytest.raises(Exception, match="No message"):
            await executor.execute(context, MagicMock())

    @pytest.mark.asyncio
    async def test_cancel_signals_safety_queue(self, tmp_path):
        mock_onit = MagicMock()
        mock_onit.session_path = str(tmp_path / "sessions" / "test.jsonl")
        mock_onit.config_data = {"data_path": str(tmp_path / "data")}
        os.makedirs(os.path.dirname(mock_onit.session_path), exist_ok=True)

        executor = OnItA2AExecutor(mock_onit)

        context = MagicMock()
        context.context_id = "ctx-123"
        context.task_id = "task-456"

        await executor.cancel(context, MagicMock())

        # The per-session safety_queue should have the stop signal
        session = executor._sessions["ctx-123"]
        assert not session["safety_queue"].empty()
        assert session["safety_queue"].get_nowait() == STOP_TAG

    @pytest.mark.asyncio
    async def test_sessions_isolated_by_context(self, tmp_path):
        """Different context_ids get different sessions."""
        mock_onit = MagicMock()
        mock_onit.session_path = str(tmp_path / "sessions" / "test.jsonl")
        mock_onit.config_data = {"data_path": str(tmp_path / "data")}
        os.makedirs(os.path.dirname(mock_onit.session_path), exist_ok=True)

        executor = OnItA2AExecutor(mock_onit)

        ctx1 = MagicMock()
        ctx1.context_id = "ctx-aaa"
        ctx1.task_id = "task-1"

        ctx2 = MagicMock()
        ctx2.context_id = "ctx-bbb"
        ctx2.task_id = "task-2"

        s1 = executor._get_session(ctx1)
        s2 = executor._get_session(ctx2)

        assert s1["session_id"] != s2["session_id"]
        assert s1["session_path"] != s2["session_path"]
        assert s1["data_path"] != s2["data_path"]

    @pytest.mark.asyncio
    async def test_same_context_reuses_session(self, tmp_path):
        """Same context_id returns the same session."""
        mock_onit = MagicMock()
        mock_onit.session_path = str(tmp_path / "sessions" / "test.jsonl")
        mock_onit.config_data = {"data_path": str(tmp_path / "data")}
        os.makedirs(os.path.dirname(mock_onit.session_path), exist_ok=True)

        executor = OnItA2AExecutor(mock_onit)

        ctx = MagicMock()
        ctx.context_id = "ctx-same"
        ctx.task_id = "task-1"

        s1 = executor._get_session(ctx)
        s2 = executor._get_session(ctx)

        assert s1["session_id"] == s2["session_id"]


# ── ClientDisconnectMiddleware ──────────────────────────────────────────────

class TestClientDisconnectMiddleware:
    @pytest.mark.asyncio
    async def test_passes_through_non_http(self):
        mock_app = AsyncMock()
        mock_executor = MagicMock(spec=OnItA2AExecutor)
        mock_executor._active_safety_queues = {}
        mw = ClientDisconnectMiddleware(mock_app, mock_executor)

        scope = {"type": "websocket"}
        await mw(scope, AsyncMock(), AsyncMock())
        mock_app.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_buffers_body_and_forwards(self):
        calls = []

        async def fake_app(scope, receive, send):
            msg = await receive()
            calls.append(msg)

        mock_executor = MagicMock(spec=OnItA2AExecutor)
        mock_executor._active_safety_queues = {}
        mw = ClientDisconnectMiddleware(fake_app, mock_executor)

        body_content = b'{"test": true}'
        messages = [
            {"type": "http.request", "body": body_content, "more_body": False},
        ]
        msg_iter = iter(messages)

        async def receive():
            return next(msg_iter)

        scope = {"type": "http"}
        await mw(scope, receive, AsyncMock())

        assert len(calls) == 1
        assert calls[0]["body"] == body_content


class TestEnterKeyListener:
    """The Enter-key stop listener shares stdin with the raw text-UI input
    reader (``os.read(fd, 1)``). It must therefore be non-blocking and read
    *unbuffered* — otherwise it either freezes the event loop or strands the
    start of the user's next message in Python's stdin buffer, which surfaces
    as dropped input characters in the text UI.
    """

    def _make_stub(self, safety_queue):
        stub = MagicMock()
        stub.web = False
        stub.messages = {}
        stub.safety_queue = safety_queue
        return stub

    @pytest.mark.asyncio
    async def test_newline_signals_stop_without_buffered_readline(self, monkeypatch):
        r, w = os.pipe()
        try:
            fake_stdin = MagicMock()
            fake_stdin.fileno.return_value = r
            # If the implementation ever falls back to buffered readline() this
            # raises, failing the test loudly instead of silently stranding bytes.
            fake_stdin.readline.side_effect = AssertionError("must not use buffered readline()")
            monkeypatch.setattr(sys, "stdin", fake_stdin)

            loop = asyncio.get_running_loop()
            q = asyncio.Queue()
            cb = OnIt._setup_enter_key_listener(self._make_stub(q), loop)
            assert cb is not None
            try:
                os.write(w, b"\n")
                await asyncio.wait_for(q.get(), timeout=1.0)  # STOP_TAG enqueued
            finally:
                loop.remove_reader(r)
            assert q.empty()
        finally:
            os.close(r)
            os.close(w)

    @pytest.mark.asyncio
    async def test_partial_line_does_not_stop(self, monkeypatch):
        """Bytes without a newline (type-ahead) must not trigger a stop, and the
        callback must return promptly rather than block waiting for a newline."""
        r, w = os.pipe()
        try:
            fake_stdin = MagicMock()
            fake_stdin.fileno.return_value = r
            monkeypatch.setattr(sys, "stdin", fake_stdin)

            loop = asyncio.get_running_loop()
            q = asyncio.Queue()
            cb = OnIt._setup_enter_key_listener(self._make_stub(q), loop)
            try:
                os.write(w, b"hello")  # no newline
                await asyncio.sleep(0.1)  # give the reader a chance to fire
                assert q.empty()  # no false stop, and it did not block the loop
            finally:
                loop.remove_reader(r)
        finally:
            os.close(r)
            os.close(w)


# ── trajectory recording ────────────────────────────────────────────────────

class TestRecordTrajectory:
    """What a task leaves behind beyond its answer. Best-effort by contract:
    a trajectory that fails to write must never turn a completed task into a
    failed one."""

    def _onit(self, tmp_path, autonomy="observe"):
        cfg = _make_config(tmp_path, {
            "data_path": str(tmp_path / "data"),
            "learn": {"autonomy": autonomy, "path": str(tmp_path / "learned")},
        })
        with _mock_discover():
            onit = OnIt(config=cfg)
        onit.tool_registry = MagicMock(tools={"search", "bash"})
        return onit

    def _session_with_one_turn(self, tmp_path):
        path = tmp_path / "s.jsonl"
        path.write_text(json.dumps({"task": "t", "response": "r"}) + "\n")
        return str(path)

    def test_records_the_run_alongside_the_answer(self, tmp_path):
        from learn import read_session
        onit = self._onit(tmp_path)
        metrics = {"turns": [{"n": 1, "prompt_tokens": 300, "finish_reason": "stop",
                              "tool_runs": [{"name": "search", "ok": False, "ms": 9}]}],
                   "turn_count": 1, "tool_calls": 1, "api_retries": 1}
        onit._record_trajectory("find the policy", "here it is", metrics,
                                self._session_with_one_turn(tmp_path), "sess-1", None)
        records = read_session("sess-1", onit.config_data)
        assert len(records) == 1
        assert records[0]["task"] == "find the policy"
        assert records[0]["turn"] == 1
        assert records[0]["tools_available"] == ["bash", "search"]
        assert records[0]["signals"] == {
            "tool_errors": 1, "retries": 1, "truncations": 0, "compactions": 0,
            # No run state was supplied, so how the loop ended is unknown
            # rather than assumed to have been fine.
            "stop_reason": "",
            "user_rating": None, "verifier": None,
        }

    def test_how_the_run_ended_is_taken_from_its_state(self, tmp_path):
        """The metrics cannot say whether the loop finished or gave up; the
        run state can, and it is the same object either record reads."""
        from learn import read_session
        from src.model.serving.state import RunState, STOP_TURN_LIMIT

        onit = self._onit(tmp_path)
        onit._record_trajectory(
            "an impossible task", "partial", {"turns": [], "turn_count": 9},
            self._session_with_one_turn(tmp_path), "sess-1", None,
            run_state=RunState(iteration_count=9, stop_reason=STOP_TURN_LIMIT))
        records = read_session("sess-1", onit.config_data)
        assert records[0]["signals"]["stop_reason"] == STOP_TURN_LIMIT

    def test_turn_number_follows_the_session_file(self, tmp_path):
        from learn import read_session
        onit = self._onit(tmp_path)
        path = tmp_path / "s.jsonl"
        path.write_text("".join(
            json.dumps({"task": f"t{n}", "response": "r"}) + "\n" for n in range(3)))
        onit._record_trajectory("t2", "r", {}, str(path), "sess-1", None)
        assert read_session("sess-1", onit.config_data)[0]["turn"] == 3

    def test_off_writes_nothing(self, tmp_path):
        onit = self._onit(tmp_path, autonomy="off")
        onit._record_trajectory("t", "r", {},
                                self._session_with_one_turn(tmp_path), "sess-1", None)
        assert not (tmp_path / "learned").exists()

    def test_a_broken_store_does_not_propagate(self, tmp_path, monkeypatch):
        onit = self._onit(tmp_path)
        import learn.trajectory as traj
        monkeypatch.setattr(traj.os, "makedirs",
                            MagicMock(side_effect=OSError("read-only")))
        onit._record_trajectory("t", "r", {},
                                self._session_with_one_turn(tmp_path), "sess-1", None)

    def test_a_missing_registry_does_not_propagate(self, tmp_path):
        onit = self._onit(tmp_path)
        onit.tool_registry = None
        onit._record_trajectory("t", "r", {},
                                self._session_with_one_turn(tmp_path), "sess-1", None)
        from learn import read_session
        assert read_session("sess-1", onit.config_data)[0]["tools_available"] == []


class TestHarnessToolsFlag:
    """The prompt may only claim tools chat() will actually offer."""

    @staticmethod
    def _flag(cfg, tmp_path):
        """The value ``_assistant_instruction`` passes to the prompt builder.

        OnIt is constructed outside the loop on purpose: ``initialize()`` calls
        ``asyncio.run`` for tool discovery, which cannot run inside one.
        """
        captured = {}

        async def _fake(**args):
            captured.update(args)
            return "instruction"

        with _mock_discover():
            onit = OnIt(config=cfg)
        with patch("src.onit.build_assistant_instruction", new=_fake):
            asyncio.run(onit._assistant_instruction("task", str(tmp_path / "data")))
        return captured["harness_tools_available"]

    def test_claimed_by_default(self, tmp_path):
        assert self._flag(_make_config(tmp_path), tmp_path) is True

    def test_withdrawn_when_the_config_turns_them_off(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg["serving"]["harness_tools"] = False
        assert self._flag(cfg, tmp_path) is False

    def test_reaches_chat_only_when_the_config_sets_it(self, tmp_path):
        from src.onit import SERVING_PASSTHROUGH
        assert "harness_tools" in SERVING_PASSTHROUGH
        cfg = _make_config(tmp_path)
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert "harness_tools" not in onit.model_serving



# ── run state across the tasks of a session ─────────────────────────────────

class TestRunState:
    """The other half of a session: what was done, not what was said.

    A resumed session used to replay only the conversation, with no record of
    which tools ran or what was tried and failed.
    """

    @staticmethod
    def _config(tmp_path):
        return _make_config(tmp_path, {"data_path": str(tmp_path / "data")})

    @staticmethod
    def _prompt_client():
        msg = MagicMock()
        msg.content.text = "Instruction text"
        result = MagicMock()
        result.messages = [msg]
        client = AsyncMock()
        client.get_prompt = AsyncMock(return_value=result)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        return client

    # ── a fresh session ─────────────────────────────────────────────────────

    def test_a_new_session_starts_with_an_empty_state(self, tmp_path):
        with _mock_discover():
            onit = OnIt(config=self._config(tmp_path))
        assert onit.run_state is not None
        assert onit.run_state.tool_call_history == []
        assert onit.run_state.task_count == 0

    # ── persistence ─────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_a_finished_task_is_written_beside_the_session_file(self, tmp_path):
        from src.model.serving.state import STOP_ANSWERED, state_path_for

        onit = _make_onit_for_async(tmp_path, {"data_path": str(tmp_path / "data")})
        onit.safety_queue = asyncio.Queue()

        async def _fake_chat(**kwargs):
            # Stand in for the loop: fill the caller's state the way chat does.
            state = kwargs["run_state"]
            state.iteration_count = 2
            state.tool_call_history.append(("bash", '{"command": "ls"}'))
            state.stop_reason = STOP_ANSWERED
            return "The answer"

        with patch("src.onit.Client", return_value=self._prompt_client()), \
             patch("src.onit.chat", new=_fake_chat):
            await onit.process_task("List the files.")

        path = state_path_for(onit.session_path)
        assert os.path.exists(path)
        saved = json.loads(Path(path).read_text())
        assert saved["tool_call_history"] == [["bash", '{"command": "ls"}']]
        assert saved["stop_reason"] == STOP_ANSWERED
        assert saved["total_turns"] == 2
        assert saved["task_count"] == 1

    @pytest.mark.asyncio
    async def test_a_run_that_answered_nothing_is_still_recorded(self, tmp_path):
        """A run that spent its turns and came back empty is exactly the one
        the next task should know about."""
        from src.model.serving.state import STOP_TURN_LIMIT, state_path_for

        onit = _make_onit_for_async(tmp_path, {"data_path": str(tmp_path / "data")})
        onit.safety_queue = asyncio.Queue()

        async def _fake_chat(**kwargs):
            state = kwargs["run_state"]
            state.iteration_count = 5
            state.tool_call_history.append(("bash", "{}"))
            state.stop_reason = STOP_TURN_LIMIT
            return None

        with patch("src.onit.Client", return_value=self._prompt_client()), \
             patch("src.onit.chat", new=_fake_chat):
            result = await onit.process_task("Impossible task.")

        assert "rephrase" in result
        saved = json.loads(Path(state_path_for(onit.session_path)).read_text())
        assert saved["stop_reason"] == STOP_TURN_LIMIT

    @pytest.mark.asyncio
    async def test_history_accumulates_across_tasks(self, tmp_path):
        onit = _make_onit_for_async(tmp_path, {"data_path": str(tmp_path / "data")})
        onit.safety_queue = asyncio.Queue()

        async def _fake_chat(**kwargs):
            kwargs["run_state"].tool_call_history.append(("bash", "{}"))
            kwargs["run_state"].iteration_count = 1
            return "ok"

        with patch("src.onit.Client", return_value=self._prompt_client()), \
             patch("src.onit.chat", new=_fake_chat):
            await onit.process_task("one")
            await onit.process_task("two")

        assert len(onit.run_state.tool_call_history) == 2
        assert onit.run_state.task_count == 2
        assert onit.run_state.total_turns == 2

    @pytest.mark.asyncio
    async def test_each_task_gets_a_fresh_budget(self, tmp_path):
        """The repeated-call ceiling counts against the run's own history: a
        session's accumulated calls would spend that budget before the task
        made its first one."""
        seen = []

        onit = _make_onit_for_async(tmp_path, {"data_path": str(tmp_path / "data")})
        onit.safety_queue = asyncio.Queue()

        async def _fake_chat(**kwargs):
            seen.append(list(kwargs["run_state"].tool_call_history))
            kwargs["run_state"].tool_call_history.append(("bash", "{}"))
            return "ok"

        with patch("src.onit.Client", return_value=self._prompt_client()), \
             patch("src.onit.chat", new=_fake_chat):
            await onit.process_task("one")
            await onit.process_task("two")

        assert seen == [[], []]

    # ── resume ──────────────────────────────────────────────────────────────

    def test_resume_restores_the_tool_call_history(self, tmp_path):
        from src.model.serving.state import (RunState, STOP_TURN_LIMIT,
                                             state_path_for)

        cfg = self._config(tmp_path)
        with _mock_discover():
            first = OnIt(config=cfg)

        state = RunState(iteration_count=4, stop_reason=STOP_TURN_LIMIT,
                         task_count=1, total_turns=4)
        state.tool_call_history = [("local_search", '{"query": "x"}'),
                                   ("bash", "{}")]
        state.save(state_path_for(first.session_path))

        resumed_cfg = dict(cfg)
        resumed_cfg["resume_session_id"] = first.session_id
        with _mock_discover():
            second = OnIt(config=resumed_cfg)

        assert second.session_id == first.session_id
        assert second.run_state.tool_call_history == [
            ("local_search", '{"query": "x"}'), ("bash", "{}")]
        assert second.run_state.stop_reason == STOP_TURN_LIMIT
        assert second.run_state.total_turns == 4

    def test_a_resumed_session_tells_the_model_what_it_tried(self, tmp_path):
        """Step 5 of the phase: the history is fed into the instruction, so a
        continued session knows what has already been attempted."""
        from src.model.serving.state import (RunState, STOP_TURN_LIMIT,
                                             state_path_for)

        cfg = self._config(tmp_path)
        with _mock_discover():
            first = OnIt(config=cfg)

        state = RunState(stop_reason=STOP_TURN_LIMIT, total_turns=9)
        state.tool_call_history = [("local_search", "{}"), ("local_search", '{"q":1}'),
                                   ("bash", "{}")]
        state.save(state_path_for(first.session_path))

        resumed_cfg = dict(cfg)
        resumed_cfg["resume_session_id"] = first.session_id
        with _mock_discover():
            second = OnIt(config=resumed_cfg)

        captured = {}

        async def _fake(**args):
            captured.update(args)
            return "instruction"

        with patch("src.onit.build_assistant_instruction", new=_fake):
            asyncio.run(second._assistant_instruction("carry on",
                                                      str(tmp_path / "data")))

        note = captured["prior_attempts"]
        assert "local_search×2" in note
        assert "bash" in note
        assert "turn limit" in note

    def test_a_new_session_adds_nothing_to_the_instruction(self, tmp_path):
        """The common case, and it must cost nothing."""
        with _mock_discover():
            onit = OnIt(config=self._config(tmp_path))

        captured = {}

        async def _fake(**args):
            captured.update(args)
            return "instruction"

        with patch("src.onit.build_assistant_instruction", new=_fake):
            asyncio.run(onit._assistant_instruction("hello", str(tmp_path / "data")))

        assert captured["prior_attempts"] == ""

    # ── one OnIt, many sessions ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_another_session_writes_its_own_file(self, tmp_path):
        """The web UI runs several sessions through one OnIt; a task run under
        an overridden session path must not fold into this process's own."""
        from src.model.serving.state import state_path_for

        onit = _make_onit_for_async(tmp_path, {"data_path": str(tmp_path / "data")})
        onit.safety_queue = asyncio.Queue()

        other = tmp_path / "sessions" / "other.jsonl"
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_text("")

        async def _fake_chat(**kwargs):
            kwargs["run_state"].tool_call_history.append(("bash", "{}"))
            return "ok"

        with patch("src.onit.Client", return_value=self._prompt_client()), \
             patch("src.onit.chat", new=_fake_chat):
            await onit.process_task("hi", session_path=str(other),
                                    session_id="other")

        assert os.path.exists(state_path_for(str(other)))
        assert not os.path.exists(state_path_for(onit.session_path))
        assert onit.run_state.tool_call_history == []

    @pytest.mark.asyncio
    async def test_a_state_file_that_cannot_be_written_does_not_fail_the_task(
            self, tmp_path):
        onit = _make_onit_for_async(tmp_path, {"data_path": str(tmp_path / "data")})
        onit.safety_queue = asyncio.Queue()

        with patch("src.onit.Client", return_value=self._prompt_client()), \
             patch("src.onit.chat", new_callable=AsyncMock, return_value="The answer"), \
             patch("src.model.serving.state.RunState.save",
                   side_effect=OSError("read-only")):
            result = await onit.process_task("hi")

        assert result == "The answer"


class TestResultStoreFlag:
    """The prompt may only claim tools chat() will actually offer."""

    @staticmethod
    def _args(cfg, tmp_path):
        captured = {}

        async def _fake(**args):
            captured.update(args)
            return "instruction"

        with _mock_discover():
            onit = OnIt(config=cfg)
        with patch("src.onit.build_assistant_instruction", new=_fake):
            asyncio.run(onit._assistant_instruction("task", str(tmp_path / "data")))
        return captured

    def test_claimed_by_default(self, tmp_path):
        assert self._args(_make_config(tmp_path), tmp_path)["result_store_available"] is True

    def test_withdrawn_when_the_store_is_off(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg["serving"]["result_store"] = False
        args = self._args(cfg, tmp_path)
        assert args["result_store_available"] is False
        # The note tools are unaffected — the two switches are independent.
        assert args["harness_tools_available"] is True

    def test_the_harness_switch_withdraws_it_too(self, tmp_path):
        """No harness tools means no dispatch for result_read either."""
        cfg = _make_config(tmp_path)
        cfg["serving"]["harness_tools"] = False
        assert self._args(cfg, tmp_path)["result_store_available"] is False

    def test_reaches_chat_only_when_the_config_sets_it(self, tmp_path):
        from src.onit import SERVING_PASSTHROUGH
        assert "result_store" in SERVING_PASSTHROUGH
        with _mock_discover():
            onit = OnIt(config=_make_config(tmp_path))
        assert "result_store" not in onit.model_serving


class TestCodeExecutionFlag:
    """Off unless a deployment asked for it, everywhere at once."""

    @staticmethod
    def _args(cfg, tmp_path):
        captured = {}

        async def _fake(**args):
            captured.update(args)
            return "instruction"

        with _mock_discover():
            onit = OnIt(config=cfg)
        with patch("src.onit.build_assistant_instruction", new=_fake):
            asyncio.run(onit._assistant_instruction("task", str(tmp_path / "data")))
        return captured

    def test_off_by_default(self, tmp_path):
        assert self._args(_make_config(tmp_path), tmp_path)["code_execution_available"] is False

    def test_claimed_when_switched_on(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg["serving"]["code_execution"] = True
        assert self._args(cfg, tmp_path)["code_execution_available"] is True

    def test_the_harness_switch_withdraws_it(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg["serving"]["code_execution"] = True
        cfg["serving"]["harness_tools"] = False
        assert self._args(cfg, tmp_path)["code_execution_available"] is False


class TestSandboxAvailable:
    """Derived from the registry, so it cannot claim tools nobody registered."""

    @staticmethod
    def _onit(tmp_path, tools=None):
        with _mock_discover():
            onit = OnIt(config=_make_config(tmp_path))
        onit.tool_registry = MagicMock(tools=tools) if tools is not None else None
        return onit

    def test_no_registry_means_no_sandbox(self, tmp_path):
        assert self._onit(tmp_path).sandbox_available is False

    def test_absent_without_a_provider(self, tmp_path):
        assert self._onit(tmp_path, {"bash", "read_file"}).sandbox_available is False

    def test_present_when_a_provider_registered_it(self, tmp_path):
        onit = self._onit(tmp_path, {"bash", "sandbox_run_code", "sandbox_stop"})
        assert onit.sandbox_available is True

    def test_no_config_key_can_assert_it(self, tmp_path):
        """The old --sandbox/config flag is gone; a stale key must stay inert."""
        with _mock_discover():
            onit = OnIt(config=_make_config(tmp_path, {"sandbox": True}))
        onit.tool_registry = MagicMock(tools={"bash"})
        assert onit.sandbox_available is False

    def test_reaches_chat_only_when_the_config_sets_it(self, tmp_path):
        from src.onit import SERVING_PASSTHROUGH
        assert "code_execution" in SERVING_PASSTHROUGH
        assert "code_timeout" in SERVING_PASSTHROUGH
        with _mock_discover():
            onit = OnIt(config=_make_config(tmp_path))
        assert "code_execution" not in onit.model_serving

    @pytest.mark.asyncio
    async def test_stopping_a_session_stops_its_interpreter(self, tmp_path):
        """A child process left behind by a stopped session is one nobody will
        ever stop."""
        from src.onit import _call_sandbox_stop
        from src.model.serving.interpreter import get_interpreter, shutdown_all

        try:
            interp = get_interpreter("doomed", tool_items=[], dispatch=None)
            await interp.run("x = 1")
            assert interp.alive
            # No registry, so no sandbox to stop: the interpreter goes anyway.
            await _call_sandbox_stop(None, "doomed")
            assert not interp.alive
        finally:
            await shutdown_all()
