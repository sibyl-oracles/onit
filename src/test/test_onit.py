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

from src.onit import OnIt, OnItA2AExecutor, ClientDisconnectMiddleware, STOP_TAG


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_config(tmp_path, overrides=None):
    """Build a minimal config dict."""
    cfg = {
        "serving": {
            "host": "http://localhost:8000/v1",
            "model": "test-model",
            "think": False,
            "max_tokens": 1024,
        },
        "mcp": {
            "servers": [
                {
                    "name": "PromptsMCPServer",
                    "url": "http://127.0.0.1:9002/prompts",
                    "enabled": True,
                },
            ],
        },
        "session_path": str(tmp_path / "sessions"),
        "theme": "white",
        "persona": "assistant",
        "verbose": False,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def _mock_discover():
    """Patch discover_tools to return an empty registry."""
    from type.tools import ToolRegistry
    return patch("src.onit.discover_tools", return_value=ToolRegistry())


# ── OnIt.__init__ ───────────────────────────────────────────────────────────

class TestOnItInit:
    def test_init_from_dict(self, tmp_path):
        cfg = _make_config(tmp_path)
        with _mock_discover():
            onit = OnIt(config=cfg)
        assert onit.status == "initialized"
        assert onit.persona == "assistant"

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

    def test_init_no_prompts_server_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg["mcp"]["servers"] = [{"name": "Other", "url": "http://x", "enabled": True}]
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

        assert "Error" in result


# ── OnItA2AExecutor ─────────────────────────────────────────────────────────

class TestOnItA2AExecutor:
    @pytest.mark.asyncio
    async def test_execute_calls_process_task(self):
        mock_onit = MagicMock()
        mock_onit.process_task = AsyncMock(return_value="result text")

        executor = OnItA2AExecutor(mock_onit)

        context = MagicMock()
        context.get_user_input.return_value = "test task"
        context.message = MagicMock()
        context.message.parts = []

        event_queue = MagicMock()
        event_queue.enqueue_event = AsyncMock()

        await executor.execute(context, event_queue)

        mock_onit.process_task.assert_awaited_once_with("test task", images=None)
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
    async def test_cancel_signals_safety_queue(self):
        mock_onit = MagicMock()
        mock_onit.safety_queue = asyncio.Queue()

        executor = OnItA2AExecutor(mock_onit)
        await executor.cancel(MagicMock(), MagicMock())

        assert not mock_onit.safety_queue.empty()
        assert mock_onit.safety_queue.get_nowait() == STOP_TAG


# ── ClientDisconnectMiddleware ──────────────────────────────────────────────

class TestClientDisconnectMiddleware:
    @pytest.mark.asyncio
    async def test_passes_through_non_http(self):
        mock_app = AsyncMock()
        mock_onit = MagicMock()
        mw = ClientDisconnectMiddleware(mock_app, mock_onit)

        scope = {"type": "websocket"}
        await mw(scope, AsyncMock(), AsyncMock())
        mock_app.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_buffers_body_and_forwards(self):
        calls = []

        async def fake_app(scope, receive, send):
            msg = await receive()
            calls.append(msg)

        mock_onit = MagicMock()
        mock_onit.safety_queue = asyncio.Queue()
        mw = ClientDisconnectMiddleware(fake_app, mock_onit)

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
