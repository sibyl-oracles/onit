"""Tests for the legacy A2A executor and disconnect middleware.

Moved from src/test/test_onit.py when A2A support moved to legacy/.
The classes under test live in legacy/a2a_server/server.py.
"""

import asyncio
import os
import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from legacy.a2a_server.server import (OnItA2AExecutor, ClientDisconnectMiddleware,
                               STOP_TAG)


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
