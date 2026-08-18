"""Tests for src/sessions.py — the session index and its auto-naming."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sessions import (
    _auto_tag_candidates,
    _make_auto_tag,
    _TAG_MAX_CHARS,
    _TAG_MAX_WORDS,
    register_session,
    update_session,
    list_sessions,
)


# ── Auto-tag wording ────────────────────────────────────────────────────────

class TestAutoTag:

    @pytest.mark.parametrize("task,expected", [
        ("can you please help me fix the login bug in the auth module",
         "fix-login-bug-auth"),
        ("How do I add a new user to the database?", "add-new-user-database"),
        ("why does the voice bridge drop the first word",
         "voice-bridge-drop-first"),
    ])
    def test_filler_is_dropped_so_the_topic_leads(self, task, expected):
        assert _make_auto_tag(task) == expected

    def test_names_stay_short_enough_for_the_sidebar(self):
        tag = _make_auto_tag(
            "investigate the intermittent websocket disconnection happening "
            "whenever the model server restarts under load")
        assert len(tag) <= _TAG_MAX_CHARS
        assert len(tag.split("-")) <= _TAG_MAX_WORDS

    def test_only_the_first_sentence_is_used(self):
        task = ("Explain the token limit error.\n\nIt happens when the "
                "context grows past 8k and everything breaks.")
        assert _make_auto_tag(task) == "explain-token-limit-error"

    def test_urls_are_reduced_to_readable_parts(self):
        tag = _make_auto_tag("summarize https://example.com/blog/2026/roadmap")
        assert tag == "summarize-example-blog-2026"

    def test_repeated_words_are_not_repeated_in_the_name(self):
        assert _make_auto_tag("cache the cache warmup cache results") == \
            "cache-warmup-results"

    def test_an_all_filler_task_still_gets_a_name(self):
        assert _make_auto_tag("what is this?") == "what-is-this"

    def test_an_empty_task_falls_back_to_unnamed(self):
        assert _make_auto_tag("   ") == "unnamed"

    def test_punctuation_never_leaks_into_the_slug(self):
        tag = _make_auto_tag("Fix the parser: it chokes on (nested) braces!")
        assert all(c.isalnum() or c == "-" for c in tag)
        assert tag == "fix-parser-chokes-nested"

    def test_candidates_slide_the_window_forward(self):
        options = _auto_tag_candidates(
            "fix the login bug in the auth module")
        assert options[0] == "fix-login-bug-auth"
        # Later options are differently worded, not numbered variants.
        assert len(set(options)) == len(options)
        assert all(not o[-1].isdigit() for o in options)


# ── Uniqueness in the index ─────────────────────────────────────────────────

class TestSessionNaming:

    def test_a_second_session_on_the_same_topic_gets_a_distinct_name(self, tmp_path):
        d = str(tmp_path)
        task = "fix the login bug in the auth module"
        for sid in ("s1", "s2"):
            register_session(sid, sessions_dir=d)
            update_session(sid, task=task, sessions_dir=d)

        tags = {s["session_id"]: s["tag"] for s in list_sessions(sessions_dir=d)}
        assert tags["s1"] == "fix-login-bug-auth"
        assert tags["s2"] != tags["s1"]
        assert not tags["s2"].endswith("-2")

    def test_identical_short_tasks_fall_back_to_a_numeric_suffix(self, tmp_path):
        d = str(tmp_path)
        for sid in ("s1", "s2"):
            register_session(sid, sessions_dir=d)
            update_session(sid, task="deploy", sessions_dir=d)

        tags = {s["session_id"]: s["tag"] for s in list_sessions(sessions_dir=d)}
        assert tags["s1"] == "deploy"
        assert tags["s2"] == "deploy-2"

    def test_a_later_turn_does_not_rename_the_session(self, tmp_path):
        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        update_session("s1", task="fix the login bug", sessions_dir=d)
        update_session("s1", task="now also update the docs", sessions_dir=d)

        (meta,) = list_sessions(sessions_dir=d)
        assert meta["tag"] == "fix-login-bug"
        assert meta["turns"] == 2


class TestRunStateSidecar:
    """The run state file is part of a session and goes when it does.

    Session ids are recycled by nothing today, but a state file that outlives
    its history would tell a future session it had already run tools it never
    ran.
    """

    def _session(self, tmp_path, sid="sid-1"):
        from model.serving.state import RunState, state_path_for
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(exist_ok=True)
        jsonl = sessions_dir / f"{sid}.jsonl"
        jsonl.write_text('{"task": "t", "response": "r"}\n')
        state = RunState()
        state.tool_call_history = [("bash", "{}")]
        state.save(state_path_for(str(jsonl)))
        return str(sessions_dir), str(jsonl)

    def test_delete_session_takes_the_state_with_it(self, tmp_path):
        from model.serving.state import state_path_for
        from sessions import delete_session

        sessions_dir, jsonl = self._session(tmp_path)
        assert os.path.exists(state_path_for(jsonl))

        delete_session("sid-1", sessions_dir=sessions_dir)

        assert not os.path.exists(jsonl)
        assert not os.path.exists(state_path_for(jsonl))

    def test_clear_sessions_takes_them_all(self, tmp_path):
        from model.serving.state import state_path_for
        from sessions import clear_sessions

        sessions_dir, jsonl = self._session(tmp_path)
        _, jsonl2 = self._session(tmp_path, "sid-2")

        assert clear_sessions(sessions_dir=sessions_dir) == 2
        assert not os.path.exists(state_path_for(jsonl))
        assert not os.path.exists(state_path_for(jsonl2))
