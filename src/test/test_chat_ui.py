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


# ── approval prompt ─────────────────────────────────────────────────────────


class _NotATTY:
    """A stdin that is honest about not being a terminal.

    Wraps whatever stdin the suite was started with — captured or real — so a
    test drives the prompt's non-terminal path regardless.
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class TestAskApproval:
    """The terminal prompt for a command policy will not run on its own.

    Both defects this pins were reported as "the approve UI is not
    responding", and neither was a hang: the key hints were invisible and the
    word the user typed was not recognised, so the prompt looked inert while
    quietly refusing.
    """

    REQUEST = {
        "command": "ls /home/rowel; grep '[0-9]' notes.txt",
        "reason": "Command blocked: references path '/home/rowel' outside "
                  "allowed directories.",
        "subjects": ["path:/home/rowel"],
    }

    def _ask(self, chat_ui, answers):
        """Run the prompt with ``answers`` queued on stdin.

        Returns the choice, what the console rendered, and the prompts that
        were put to the reader — the re-prompt goes to input()'s own prompt
        argument rather than through Rich, so it is only visible here.
        """
        import asyncio

        buf = io.StringIO()
        chat_ui.console = Console(file=buf, width=100, force_terminal=False)
        chat_ui.stop_status = lambda: None
        chat_ui.stop_thinking = lambda: None
        queued = list(answers)
        prompts = []

        def _input(prompt=""):
            prompts.append(prompt)
            return queued.pop(0)

        # isatty pinned false, so the prompt takes its non-terminal path and
        # reads through input() no matter how the suite was started. Under
        # `pytest -s` stdin is the real terminal, and without this these tests
        # would poll it for five minutes instead of reading the queue.
        with patch("builtins.input", _input), \
                patch.object(sys, "stdin", _NotATTY(sys.stdin)):
            choice = asyncio.run(chat_ui.ask_approval(self.REQUEST))
        return choice, buf.getvalue(), prompts

    def test_the_keys_are_actually_on_screen(self, chat_ui):
        """Rich reads square brackets as style tags.

        Written as "[y] run once", the hints were parsed as markup and
        rendered as blank space, leaving a prompt that asked a question and
        showed no way to answer it.
        """
        _, out, _prompts = self._ask(chat_ui, ["1"])
        assert "1) run once" in out
        assert "2) allow for this session" in out
        assert "3) refuse" in out

    def test_the_command_is_shown_as_it_would_run(self, chat_ui):
        """Same markup problem, but load-bearing: a character class or a glob
        in the command must reach the screen intact, or the person is
        approving something other than what they were shown."""
        _, out, _prompts = self._ask(chat_ui, ["3"])
        assert "grep '[0-9]' notes.txt" in out

    @pytest.mark.parametrize("typed,expected", [
        ("1", "once"), ("y", "once"), ("yes", "once"), ("run", "once"),
        # Ambiguous — the screen offers "allow" as the session label, but on
        # its own it just means yes, so it takes the narrower reading.
        ("allow", "once"),
        ("2", "session"), ("a", "session"), ("always", "session"),
        ("3", "deny"), ("n", "deny"), ("no", "deny"),
    ])
    def test_words_people_actually_type(self, chat_ui, typed, expected):
        choice, _, _prompts = self._ask(chat_ui, [typed])
        assert choice == expected

    def test_silence_is_a_refusal(self, chat_ui):
        choice, _, _prompts = self._ask(chat_ui, [""])
        assert choice == "deny"

    def test_an_unrecognised_answer_is_asked_again(self, chat_ui):
        """Not the same as silence.

        Someone who typed something is present and answering; refusing them on
        a spelling is how the prompt earned its reputation for ignoring input.
        """
        choice, _out, prompts = self._ask(chat_ui, ["huh?", "1"])
        assert choice == "once"
        assert len(prompts) == 2
        assert "please answer" in prompts[1]

    def test_it_gives_up_rather_than_looping(self, chat_ui):
        choice, _out, prompts = self._ask(chat_ui, ["huh?", "what", "eh"])
        assert len(prompts) == 3
        assert choice == "deny"

    def test_a_closed_stdin_refuses_at_once(self, chat_ui):
        import asyncio

        buf = io.StringIO()
        chat_ui.console = Console(file=buf, width=100, force_terminal=False)
        chat_ui.stop_status = lambda: None
        chat_ui.stop_thinking = lambda: None

        def _eof(prompt=""):
            raise EOFError

        with patch("builtins.input", _eof), \
                patch.object(sys, "stdin", _NotATTY(sys.stdin)):
            choice = asyncio.run(chat_ui.ask_approval(self.REQUEST))
        assert choice == "deny"

    def test_a_timed_out_prompt_does_not_swallow_the_next_answer(self, chat_ui):
        """The hang this replaced input() to fix.

        The prompt used to read stdin with input() on a pooled worker thread.
        A thread blocked in a read cannot be cancelled, so once the question
        timed out that thread stayed on stdin and consumed whatever was typed
        next — the late "2", and then every line meant for the prompt after
        it. The terminal looked frozen for the rest of the session.

        So: give up at the deadline, and leave the answer typed a moment later
        sitting in the terminal where the next reader will find it.
        """
        import asyncio
        import pty
        import time as _time

        master, slave = pty.openpty()
        try:
            stdin = os.fdopen(slave, "r")
            with patch.object(sys, "stdin", stdin):
                late = asyncio.run(chat_ui._read_answer_tty(
                    "  ➤ approve? ", _time.monotonic() + 0.2))
                assert late is None, "the prompt should stop at its deadline"

                # Typed just after the question expired — the keystrokes the
                # abandoned thread used to eat.
                os.write(master, b"2\n")
                after = asyncio.run(chat_ui._read_answer_tty(
                    "  ➤ ", _time.monotonic() + 5))
            assert after == "2"
        finally:
            os.close(master)
            try:
                stdin.close()
            except Exception:
                os.close(slave)

    def test_an_unanswered_prompt_refuses_and_says_so(self, chat_ui):
        """Silence ends the question rather than standing on screen forever.

        And it is called out: a refusal the person never chose reads as the
        run ignoring them unless the screen says the question expired.
        """
        import asyncio
        import threading as _threading

        buf = io.StringIO()
        chat_ui.console = Console(file=buf, width=100, force_terminal=False)
        chat_ui.stop_status = lambda: None
        chat_ui.stop_thinking = lambda: None
        answered = _threading.Event()

        def _never(prompt=""):
            answered.wait(30)  # a person who is not at the keyboard
            return "2"

        with patch("ui.text.APPROVAL_TIMEOUT", 0.3), \
                patch("builtins.input", _never), \
                patch.object(sys, "stdin", _NotATTY(sys.stdin)):
            try:
                choice = asyncio.run(chat_ui.ask_approval(self.REQUEST))
            finally:
                answered.set()  # let the reader thread go

        assert choice == "deny"
        assert "no answer in" in buf.getvalue()

    def test_the_abandoned_reader_cannot_hold_the_process_open(self, chat_ui):
        """A reader left waiting on a pipe must not keep the interpreter up.

        The default executor's threads are not daemons and are joined at exit,
        so a prompt nobody answers would have made shutdown hang too.
        """
        import asyncio
        import threading as _threading

        release = _threading.Event()

        def _never(prompt=""):
            release.wait(30)
            return "1"

        async def _run():
            with patch("builtins.input", _never), \
                    patch.object(sys, "stdin", _NotATTY(sys.stdin)):
                import time as _time
                return await chat_ui._read_answer_piped(
                    "? ", _time.monotonic() + 0.3)

        try:
            assert asyncio.run(_run()) is None
            readers = [t for t in _threading.enumerate()
                       if t.name == "onit-approval-read"]
            assert readers, "the reader should still be the one blocked"
            assert all(t.daemon for t in readers)
        finally:
            release.set()
