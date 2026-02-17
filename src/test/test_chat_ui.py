"""Tests for src/ui/text.py — Message, ChatUI."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest
from rich.panel import Panel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.text import Message, ChatUI


# ── Message dataclass ───────────────────────────────────────────────────────

class TestMessage:
    def test_create_with_required_fields(self):
        m = Message(role="user", content="hello", timestamp="12:00 PM")
        assert m.role == "user"
        assert m.content == "hello"
        assert m.elapsed == ""

    def test_create_with_elapsed(self):
        m = Message(role="assistant", content="hi", timestamp="12:01 PM", elapsed="1.5s")
        assert m.elapsed == "1.5s"


# ── ChatUI ──────────────────────────────────────────────────────────────────

@pytest.fixture
def chat_ui():
    """Create a ChatUI instance without clearing the real terminal."""
    with patch.object(ChatUI, "initialize"):
        ui = ChatUI(theme="white", max_messages=5, max_logs=5)
    return ui


class TestChatUI:
    def test_init_defaults(self, chat_ui):
        assert chat_ui.messages.maxlen == 5
        assert chat_ui.execution_logs.maxlen == 5
        assert chat_ui.show_logs is False

    def test_set_theme_dark(self, chat_ui):
        chat_ui.set_theme("dark")
        assert "prompt" in chat_ui.theme.styles

    def test_set_theme_white(self, chat_ui):
        chat_ui.set_theme("white")
        assert "prompt" in chat_ui.theme.styles

    def test_add_message(self, chat_ui):
        chat_ui.add_message("user", "hello")
        assert len(chat_ui.messages) == 1
        msg = chat_ui.messages[0]
        assert isinstance(msg, Message)
        assert msg.content == "hello"

    def test_add_message_respects_maxlen(self, chat_ui):
        for i in range(10):
            chat_ui.add_message("user", f"msg {i}")
        assert len(chat_ui.messages) == 5  # maxlen=5
        assert chat_ui.messages[0].content == "msg 5"

    def test_update_last_message(self, chat_ui):
        chat_ui.add_message("assistant", "start")
        chat_ui.update_last_message(" end")
        assert chat_ui.messages[-1].content == "start end"

    def test_update_last_message_empty(self, chat_ui):
        # No crash when no messages
        chat_ui.update_last_message("nothing")
        assert len(chat_ui.messages) == 0

    def test_clear_messages_all(self, chat_ui):
        chat_ui.add_message("user", "a")
        chat_ui.add_message("user", "b")
        chat_ui.clear_messages()
        assert len(chat_ui.messages) == 0

    def test_clear_messages_keep_last(self, chat_ui):
        chat_ui.add_message("user", "a")
        chat_ui.add_message("user", "b")
        chat_ui.add_message("user", "c")
        chat_ui.clear_messages(keep_last=1)
        assert len(chat_ui.messages) == 1
        assert chat_ui.messages[0].content == "c"

    def test_add_log(self, chat_ui):
        chat_ui.add_log("something happened", level="warning")
        assert len(chat_ui.execution_logs) == 1
        assert chat_ui.execution_logs[0]["level"] == "warning"

    def test_clear_logs_all(self, chat_ui):
        chat_ui.add_log("a")
        chat_ui.add_log("b")
        chat_ui.clear_logs()
        assert len(chat_ui.execution_logs) == 0

    def test_clear_logs_keep_last(self, chat_ui):
        for i in range(5):
            chat_ui.add_log(f"log {i}")
        chat_ui.clear_logs(keep_last=2)
        assert len(chat_ui.execution_logs) == 2

    def test_set_show_logs(self, chat_ui):
        chat_ui.set_show_logs(True)
        assert chat_ui.show_logs is True
        chat_ui.set_show_logs(False)
        assert chat_ui.show_logs is False

    def test_render_messages_empty(self, chat_ui):
        panel = chat_ui.render_messages()
        assert isinstance(panel, Panel)

    def test_render_messages_with_content(self, chat_ui):
        chat_ui.add_message("user", "hello")
        chat_ui.add_message("assistant", "hi", elapsed="0.5s")
        panel = chat_ui.render_messages()
        assert isinstance(panel, Panel)

    def test_render_logs_panel_empty(self, chat_ui):
        panel = chat_ui.render_logs_panel()
        assert isinstance(panel, Panel)

    def test_render_logs_panel_with_content(self, chat_ui):
        chat_ui.add_log("test log", level="info")
        chat_ui.add_log("warning log", level="warning")
        panel = chat_ui.render_logs_panel()
        assert isinstance(panel, Panel)

    def test_render_returns_panel(self, chat_ui):
        panel = chat_ui.render()
        assert isinstance(panel, Panel)

    def test_render_with_show_logs(self, chat_ui):
        chat_ui.set_show_logs(True)
        chat_ui.add_log("visible log")
        panel = chat_ui.render()
        assert isinstance(panel, Panel)

    def test_stop_status_no_error(self, chat_ui):
        # Should not raise even if status is already stopped
        chat_ui.stop_status()
        chat_ui.stop_status()  # double stop is safe
