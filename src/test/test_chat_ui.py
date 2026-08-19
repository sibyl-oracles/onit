"""Tests for src/ui/text.py — Message, ChatUI."""

import io
import os
import re
import sys
from unittest.mock import patch, MagicMock

import pytest
from rich.console import Console, Group
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
        result = chat_ui.render_messages()
        assert isinstance(result, Group)

    def test_render_messages_with_content(self, chat_ui):
        chat_ui.add_message("user", "hello")
        chat_ui.add_message("assistant", "hi", elapsed="0.5s")
        result = chat_ui.render_messages()
        assert isinstance(result, Group)

    def test_render_logs_panel_empty(self, chat_ui):
        panel = chat_ui.render_logs_panel()
        assert isinstance(panel, Panel)

    def test_render_logs_panel_with_content(self, chat_ui):
        chat_ui.add_log("test log", level="info")
        chat_ui.add_log("warning log", level="warning")
        panel = chat_ui.render_logs_panel()
        assert isinstance(panel, Panel)

    def test_render_returns_panel(self, chat_ui):
        result = chat_ui.render()
        assert isinstance(result, Group)

    def test_render_with_show_logs(self, chat_ui):
        chat_ui.set_show_logs(True)
        chat_ui.add_log("visible log")
        result = chat_ui.render()
        assert isinstance(result, Group)

    def test_stop_status_no_error(self, chat_ui):
        # Should not raise even if status is already stopped
        chat_ui.stop_status()
        chat_ui.stop_status()  # double stop is safe


# ── fact-check display ──────────────────────────────────────────────────────

class TestVerificationDisplay:
    def test_clean_check_leaves_the_answer_alone(self, chat_ui):
        chat_ui.add_message("assistant", "Revenue was 4.2M in 2019.")
        chat_ui.verification_end("Revenue was 4.2M in 2019.", "")
        assert chat_ui.messages[-1].content == "Revenue was 4.2M in 2019."

    def test_revision_replaces_the_stored_answer(self, chat_ui):
        """The panel, the session file and the block the user scrolls back to
        must not disagree about what the answer was."""
        chat_ui.add_message("user", "what was the revenue?")
        chat_ui.add_message("assistant", "Revenue was 4.2M in 2019.")
        chat_ui.verification_end("Revenue was 3.1M in 2019.", "3.1M, not 4.2M")
        assert chat_ui.messages[-1].content == "Revenue was 3.1M in 2019."
        assert chat_ui.messages[-1].role == "assistant"
        assert chat_ui.messages[0].content == "what was the revenue?"

    def test_replacement_is_normalized_like_a_streamed_answer(self, chat_ui):
        """onit.py compares the returned answer against what is stored to
        decide whether to append it again; the two must match byte for byte."""
        chat_ui.add_message("assistant", "draft")
        chat_ui.verification_end("<answer>Revenue was 3.1M.</answer>", "corrected")
        assert chat_ui.messages[-1].content == "Revenue was 3.1M."

    def test_revision_with_no_assistant_message_does_not_raise(self, chat_ui):
        chat_ui.add_message("user", "hello")
        chat_ui.verification_end("Revenue was 3.1M.", "corrected")
        assert chat_ui.messages[-1].role == "user"

    def test_verification_start_does_not_raise(self, chat_ui):
        chat_ui.verification_start()


class TestLateCorrection:
    """A correction that lands after the answer was final. The user is at the
    prompt by then and this loop reads keystrokes in raw mode, so nothing may
    be printed under them until the panel redraws."""

    def test_the_stored_answer_is_corrected_immediately(self, chat_ui):
        chat_ui.add_message("user", "what was the revenue?")
        chat_ui.add_message("assistant", "Revenue was 4.2M in 2019.")
        chat_ui.verification_correction("Revenue was 3.1M in 2019.", "3.1M, not 4.2M")
        assert chat_ui.messages[-1].content == "Revenue was 3.1M in 2019."
        assert chat_ui.messages[0].content == "what was the revenue?"

    def test_the_note_waits_for_the_next_turn(self, chat_ui):
        chat_ui.add_message("assistant", "Revenue was 4.2M in 2019.")
        chat_ui.verification_correction("Revenue was 3.1M in 2019.", "3.1M, not 4.2M")
        assert chat_ui.pending_corrections == ["3.1M, not 4.2M"]
        chat_ui.flush_corrections()
        assert chat_ui.pending_corrections == []
        chat_ui.flush_corrections()  # nothing left to say, and saying it twice

    def test_an_empty_note_is_not_a_correction(self, chat_ui):
        chat_ui.add_message("assistant", "Revenue was 4.2M in 2019.")
        chat_ui.verification_correction("Revenue was 4.2M in 2019.", "")
        assert chat_ui.pending_corrections == []

    def test_a_correction_with_no_answer_to_amend_does_not_raise(self, chat_ui):
        chat_ui.add_message("user", "hello")
        chat_ui.verification_correction("Revenue was 3.1M.", "corrected")
        assert chat_ui.pending_corrections == ["corrected"]


# ── turn timing ─────────────────────────────────────────────────────────────

class TestTurnTiming:
    def test_meta_matches_the_web_ui_shape(self):
        """The browser prints '12.35s · 21.5 tok/s'; the terminal must agree."""
        assert ChatUI.format_meta(12.345, 21.47) == "12.35s · 21.5 tok/s"

    def test_meta_drops_fields_it_has_no_number_for(self):
        assert ChatUI.format_meta(3.0, 0.0) == "3.00s"
        assert ChatUI.format_meta(0.0, 0.0) == ""

    def test_turn_clock_covers_the_time_before_the_first_token(self, chat_ui):
        """Tool calls and thinking run before streaming starts, and the user
        waited through them — the web UI counts them, so this must too."""
        chat_ui.turn_start()
        chat_ui._turn_start_time -= 5.0  # 5s of tool calls
        chat_ui.stream_start()           # first token only now
        assert chat_ui._turn_elapsed() >= 5.0

    def test_turn_clock_falls_back_to_the_stream(self, chat_ui):
        """Entry points that never call turn_start() still get a number."""
        chat_ui.stream_start()
        chat_ui._stream_start_time -= 2.0
        assert 2.0 <= chat_ui._turn_elapsed() < 3.0

    def test_streamed_block_footer_carries_elapsed_and_rate(self, chat_ui):
        buf = io.StringIO()
        chat_ui.console = Console(file=buf, width=120)
        chat_ui.set_metrics({"completion_tokens": 300, "decode_s": 10.0})
        chat_ui.turn_start()
        chat_ui.stream_start()
        chat_ui.stream_token("hello")
        chat_ui._turn_start_time -= 3.0
        chat_ui._stream_start_time -= 3.0
        chat_ui.stream_end()
        footer = buf.getvalue().splitlines()[-1]
        assert re.search(r"\d+\.\d\ds · 30\.0 tok/s", footer), footer

    def test_footer_rate_counts_the_thinking_the_model_streamed(self, chat_ui):
        """Reasoning tokens never reach stream_token(), but the clock has been
        running since the first one — counting only the answer reports a
        thinking model at a fraction of the speed it actually generated."""
        buf = io.StringIO()
        chat_ui.console = Console(file=buf, width=120)
        # 900 thinking + 100 answer tokens in 10s.
        chat_ui.set_metrics({"completion_tokens": 1000, "decode_s": 10.0})
        chat_ui.turn_start()
        chat_ui.stream_start()
        for _ in range(100):
            chat_ui.stream_think_token("x")
        chat_ui.stream_token("hello")
        chat_ui.stream_end()
        footer = buf.getvalue().splitlines()[-1]
        assert "100.0 tok/s" in footer, footer

    def test_explicit_elapsed_wins_over_the_measured_one(self, chat_ui):
        buf = io.StringIO()
        chat_ui.console = Console(file=buf, width=120)
        chat_ui.turn_start()
        chat_ui.stream_start()
        chat_ui.stream_token("hello")
        chat_ui.stream_end(elapsed="9.99s")
        assert "9.99s" in buf.getvalue()


# ── streamed answer text ────────────────────────────────────────────────────

def _stream(chat_ui, tokens, capsys):
    """Feed tokens through a full stream and return everything printed."""
    chat_ui.console = Console(file=sys.stdout, width=120)
    chat_ui.stream_start()
    for t in tokens:
        chat_ui.stream_token(t)
    chat_ui.stream_end()
    out = capsys.readouterr().out
    # Drop ANSI escapes and the blinking block cursor so text is comparable.
    out = re.sub(r"\x1b\[[0-9;? ]*[a-zA-Z]|[\r\x08]", "", out)
    return out.replace("\u2588 ", "").replace("\u2588", "")


class TestStreamedAnswerIsComplete:
    """The link/tag filters buffer across tokens; nothing they hold may be lost."""

    def test_unclosed_link_paren_does_not_eat_the_rest(self, chat_ui, capsys):
        out = _stream(chat_ui, ["The plot ", "f[i](t ", "keeps rising. ",
                                "Final answer: 42.\n"], capsys)
        assert "Final answer: 42." in out
        assert "f[i](t keeps rising." in out

    def test_trailing_bracket_survives_the_end_of_stream(self, chat_ui, capsys):
        out = _stream(chat_ui, ["Answer: see note ", "[1]"], capsys)
        assert "see note [1]" in out

    def test_unterminated_label_survives_the_end_of_stream(self, chat_ui, capsys):
        out = _stream(chat_ui, ["Done. ", "[TODO"], capsys)
        assert "Done. [TODO" in out

    def test_lone_angle_bracket_survives_the_end_of_stream(self, chat_ui, capsys):
        out = _stream(chat_ui, ["Result is a ", "<"], capsys)
        assert "Result is a <" in out

    def test_real_link_still_prints_label_only(self, chat_ui, capsys):
        out = _stream(chat_ui, ["See ", "[docs](http://x)", " for more.\n"], capsys)
        assert "See docs for more." in out
        assert "http://x" not in out

    def test_answer_tags_still_stripped(self, chat_ui, capsys):
        out = _stream(chat_ui, ["<answer>", "hi there", "</answer>"], capsys)
        assert "hi there" in out
        assert "<answer>" not in out

    def test_reference_link_brackets_are_preserved(self, chat_ui, capsys):
        out = _stream(chat_ui, ["Cited ", "[label][ref]", " here.\n"], capsys)
        assert "Cited [label][ref] here." in out


class TestNotice:
    def test_a_warning_is_printed_not_just_logged(self, chat_ui):
        buf = io.StringIO()
        chat_ui.console = Console(file=buf, width=120)
        chat_ui.notice("Answer is still incomplete", level="warning")
        assert "Answer is still incomplete" in buf.getvalue()

    def test_info_notices_print_too(self, chat_ui):
        buf = io.StringIO()
        chat_ui.console = Console(file=buf, width=120)
        chat_ui.notice("resuming (1/3)")
        assert "resuming (1/3)" in buf.getvalue()
