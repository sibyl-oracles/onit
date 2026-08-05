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
        nothing about the others — the batch counts instead."""
        statuses = []
        adapter = StreamingAdapter(on_tool_status=statuses.append)
        adapter.start_tool_batch(["read_file", "read_file", "search_document"])
        adapter.start_tool_spinner("read_file", {"path": "/a.pdf"})  # suppressed
        adapter.show_tool_done("read_file", "ok")
        adapter.stop_tool_spinner()                                  # suppressed
        adapter.show_tool_done("read_file", "ok")
        adapter.end_tool_batch()
        assert statuses == [
            "Running 3 tools together…",
            "1 of 3 tools done…",
            "2 of 3 tools done…",
            "",
        ]

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
        cfg = _make_config(tmp_path)
        cfg["mcp"]["mcp_host"] = "192.168.1.100"
        with _mock_discover():
            onit = OnIt(config=cfg)
        for server in onit.mcp_servers:
            assert "192.168.1.100" in server["url"]

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
            "user_rating": None, "verifier": None,
        }

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
