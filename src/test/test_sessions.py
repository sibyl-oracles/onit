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


# ── Naming and renaming ──────────────────────────────────────────────────────
#
# A session is found by its name long after it was written, so the name has to
# be something a person chose or a message that says what the session was
# about.  The cases below are the ones that made sessions unfindable.

class TestExplicitNaming:

    def test_a_session_can_be_named_as_it_is_created(self, tmp_path):
        d = str(tmp_path)
        register_session("s1", sessions_dir=d, tag="Quarterly Review")
        assert list_sessions(sessions_dir=d)[0]["tag"] == "Quarterly Review"

    def test_a_name_keeps_its_spaces_and_case(self, tmp_path):
        d = str(tmp_path)
        register_session("s1", sessions_dir=d, tag="PhD Chapter Three")
        assert list_sessions(sessions_dir=d)[0]["tag"] == "PhD Chapter Three"

    def test_surrounding_and_repeated_whitespace_is_collapsed(self, tmp_path):
        d = str(tmp_path)
        register_session("s1", sessions_dir=d, tag="  my   session  ")
        assert list_sessions(sessions_dir=d)[0]["tag"] == "my session"

    def test_a_name_cannot_carry_a_newline_into_the_index(self, tmp_path):
        d = str(tmp_path)
        register_session("s1", sessions_dir=d, tag="line one\nline two")
        tag = list_sessions(sessions_dir=d)[0]["tag"]
        assert "\n" not in tag and tag == "line one line two"

    def test_two_sessions_named_the_same_get_distinct_names(self, tmp_path):
        d = str(tmp_path)
        register_session("s1", sessions_dir=d, tag="notes")
        register_session("s2", sessions_dir=d, tag="notes")
        tags = [s["tag"] for s in list_sessions(sessions_dir=d)]
        assert len(set(tags)) == 2

    def test_an_empty_name_falls_back_to_no_name(self, tmp_path):
        d = str(tmp_path)
        register_session("s1", sessions_dir=d, tag="   ")
        assert list_sessions(sessions_dir=d)[0]["tag"] is None

    def test_a_typed_name_is_never_replaced_by_a_later_message(self, tmp_path):
        d = str(tmp_path)
        register_session("s1", sessions_dir=d, tag="My Work")
        update_session("s1", task="fix the login bug", sessions_dir=d)
        assert list_sessions(sessions_dir=d)[0]["tag"] == "My Work"

    def test_registering_returns_the_name_the_session_got(self, tmp_path):
        """The caller prints this name back and expects --resume-session to
        find the session, so a name the index had to change comes back
        changed rather than silently different from what was announced."""
        d = str(tmp_path)
        assert register_session("s1", sessions_dir=d, tag="chapter 3") == "chapter 3"
        assert register_session("s2", sessions_dir=d, tag="chapter 3") == "chapter 3-2"

    def test_the_name_a_session_will_get_can_be_asked_for_first(self, tmp_path):
        """A caller that says "call it X" must be able to print the name the
        session will actually have, before creating it."""
        from src.sessions import name_for_new_session
        d = str(tmp_path)
        assert name_for_new_session("chapter 3", d) == "chapter 3"
        register_session("s1", sessions_dir=d, tag="chapter 3")
        assert name_for_new_session("chapter 3", d) == "chapter 3-2"

    def test_the_announced_name_is_the_one_that_resolves(self, tmp_path):
        """The whole point of returning the stored name: the printed "return
        to it with" name must resolve to this session, not the one that
        already held the name."""
        from src.sessions import resolve_session_choice, name_for_new_session
        d = str(tmp_path)
        register_session("older", sessions_dir=d, tag="chapter 3")
        announced = name_for_new_session("chapter 3", d)
        register_session("newer", sessions_dir=d, tag="chapter 3")
        assert announced == "chapter 3-2"
        assert resolve_session_choice(announced, d) == "newer"
        # and the original still answers to the plain name
        assert resolve_session_choice("chapter 3", d) == "older"

    def test_a_typed_name_that_reads_like_a_placeholder_is_kept(self, tmp_path):
        """The auto-namer treats "untitled" as "no name yet". A person typing
        it means the opposite, and the name given to --new-session must be the
        name the session keeps -- otherwise the requirement "when a name is
        given it must be used" fails on exactly the words that look like
        defaults."""
        for name in ("unnamed", "untitled", "new-session"):
            d = str(tmp_path / name)
            register_session("s1", sessions_dir=d, tag=name)
            update_session("s1", task="fix the login bug", sessions_dir=d)
            assert list_sessions(sessions_dir=d)[0]["tag"] == name

    def test_a_rename_to_a_placeholder_word_also_sticks(self, tmp_path):
        from src.sessions import tag_session
        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        assert tag_session("s1", "untitled", d) is True
        update_session("s1", task="fix the login bug", sessions_dir=d)
        assert list_sessions(sessions_dir=d)[0]["tag"] == "untitled"

    def test_the_auto_namer_still_replaces_its_own_placeholder(self, tmp_path):
        """The fix above must not freeze the auto-namer's own placeholders:
        a session it could not name from the first message is still waiting
        for a real one."""
        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        index = {"s1": {"tag": "unnamed", "named_by_user": False}}
        from src.sessions import _needs_name
        assert _needs_name("unnamed", "fix the login bug",
                           named_by_user=index["s1"]["named_by_user"]) is True
        assert _needs_name("untitled", "fix the login bug",
                           named_by_user=True) is False

    def test_an_unnamed_session_returns_no_name(self, tmp_path):
        d = str(tmp_path)
        assert register_session("s1", sessions_dir=d) == ""
        assert list_sessions(sessions_dir=d)[0]["tag"] is None


class TestAutoNameWaitsForATopic:

    def test_a_greeting_does_not_become_the_session_name(self, tmp_path):
        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        update_session("s1", task="hi", sessions_dir=d)
        assert list_sessions(sessions_dir=d)[0]["tag"] is None

    def test_the_first_real_message_names_it_after_a_greeting(self, tmp_path):
        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        update_session("s1", task="hi", sessions_dir=d)
        update_session("s1", task="fix the login bug in auth", sessions_dir=d)
        assert list_sessions(sessions_dir=d)[0]["tag"] == "fix-login-bug-auth"

    def test_a_substantive_first_message_still_names_it_at_once(self, tmp_path):
        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        update_session("s1", task="fix the login bug", sessions_dir=d)
        assert list_sessions(sessions_dir=d)[0]["tag"] == "fix-login-bug"

    def test_a_greeting_does_not_become_the_preview_either(self, tmp_path):
        """The preview is how a session is recognised in the list, so it is
        worth no more than the name: "hi" says nothing about the session."""
        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        update_session("s1", task="hi", sessions_dir=d)
        update_session("s1", task="analyze the sales data", sessions_dir=d)
        assert list_sessions(sessions_dir=d)[0]["preview"] == \
            "analyze the sales data"

    def test_a_typed_name_keeps_its_preview_current(self, tmp_path):
        """Naming a session by hand says what to call it, not what it was
        about -- so the preview still follows the work."""
        d = str(tmp_path)
        register_session("s1", sessions_dir=d, tag="My Chosen Name")
        update_session("s1", task="hey", sessions_dir=d)
        update_session("s1", task="real work about databases", sessions_dir=d)
        (meta,) = list_sessions(sessions_dir=d)
        assert meta["tag"] == "My Chosen Name"
        assert meta["preview"] == "real work about databases"

    @pytest.mark.parametrize("greeting", ["hi", "hello", "thanks", "ok", "hey"])
    def test_no_courtesy_phrase_becomes_a_name(self, tmp_path, greeting):
        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        update_session("s1", task=greeting, sessions_dir=d)
        assert list_sessions(sessions_dir=d)[0]["tag"] is None


class TestRename:

    def test_renaming_changes_the_name_it_is_found_by(self, tmp_path):
        from sessions import tag_session, find_session_by_tag

        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        update_session("s1", task="fix the login bug", sessions_dir=d)

        assert tag_session("s1", "auth work", d) is True
        assert find_session_by_tag("auth work", d) == "s1"

    def test_an_exact_name_wins_over_a_longer_name_starting_with_it(self, tmp_path):
        from sessions import tag_session, resolve_session

        d = str(tmp_path)
        for sid in ("old", "new"):
            register_session(sid, sessions_dir=d)
        tag_session("old", "auth", d)
        tag_session("new", "auth-fix-2", d)
        # "new" was touched last, so recency alone would pick it.
        assert resolve_session("auth", d) == "old"

    def test_a_prefix_still_resolves_when_nothing_matches_exactly(self, tmp_path):
        from sessions import tag_session, resolve_session

        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        tag_session("s1", "auth-fix-2", d)
        assert resolve_session("auth", d) == "s1"

    def test_a_name_cannot_be_taken_from_another_session(self, tmp_path):
        from sessions import tag_session

        d = str(tmp_path)
        for sid in ("s1", "s2"):
            register_session(sid, sessions_dir=d)
        tag_session("s1", "taken", d)
        result = tag_session("s2", "taken", d)
        assert isinstance(result, str) and "already" in result

    def test_names_are_unique_regardless_of_case(self, tmp_path):
        from sessions import tag_session

        d = str(tmp_path)
        for sid in ("s1", "s2"):
            register_session(sid, sessions_dir=d)
        tag_session("s1", "Notes", d)
        assert isinstance(tag_session("s2", "NOTES", d), str)

    def test_renaming_to_the_same_name_is_allowed(self, tmp_path):
        from sessions import tag_session

        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        tag_session("s1", "keep", d)
        assert tag_session("s1", "keep", d) is True

    @pytest.mark.parametrize("bad", ["", "   ", "\n", "\t\n "])
    def test_an_unusable_name_is_refused_with_a_reason(self, tmp_path, bad):
        from sessions import tag_session

        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        result = tag_session("s1", bad, d)
        assert isinstance(result, str) and "empty" in result

    def test_renaming_an_unknown_session_reports_not_found(self, tmp_path):
        from sessions import tag_session

        assert tag_session("nope", "x", str(tmp_path)) is False


class TestRowNumbers:

    """`onit sessions` prints a # column, so that number must address the row
    it was printed next to -- including when empty sessions are hidden."""

    def test_a_row_number_resolves_to_the_row_shown(self, tmp_path):
        from sessions import resolve_session_choice

        d = str(tmp_path)
        register_session("empty", sessions_dir=d)          # no turns, hidden
        for sid, task in (("a", "write the report"), ("b", "debug cuda")):
            register_session(sid, sessions_dir=d)
            update_session(sid, task=task, sessions_dir=d)

        shown = list_sessions(sessions_dir=d, include_empty=False)
        for i, row in enumerate(shown, 1):
            assert resolve_session_choice(str(i), d) == row["session_id"]

    def test_a_row_number_past_the_end_resolves_to_nothing(self, tmp_path):
        from sessions import resolve_session_choice

        d = str(tmp_path)
        register_session("s1", sessions_dir=d)
        update_session("s1", task="do the thing", sessions_dir=d)
        assert resolve_session_choice("9", d) is None

    def test_last_still_means_the_most_recent(self, tmp_path):
        from sessions import resolve_session_choice

        d = str(tmp_path)
        for sid in ("s1", "s2"):
            register_session(sid, sessions_dir=d)
            update_session(sid, task="work", sessions_dir=d)
        assert resolve_session_choice("last", d) == "s2"


class TestEmptySessionsAreHidden:

    def test_sessions_with_no_turns_are_left_out_by_default(self, tmp_path):
        d = str(tmp_path)
        register_session("empty", sessions_dir=d)
        register_session("used", sessions_dir=d)
        update_session("used", task="real work", sessions_dir=d)

        shown = [s["session_id"] for s in list_sessions(sessions_dir=d,
                                                       include_empty=False)]
        assert shown == ["used"]

    def test_they_are_still_there_when_asked_for(self, tmp_path):
        d = str(tmp_path)
        register_session("empty", sessions_dir=d)
        shown = [s["session_id"] for s in list_sessions(sessions_dir=d,
                                                       include_empty=True)]
        assert shown == ["empty"]

    def test_the_count_ignores_the_limit(self, tmp_path):
        from sessions import count_sessions

        d = str(tmp_path)
        for i in range(5):
            register_session(f"s{i}", sessions_dir=d)
            update_session(f"s{i}", task=f"task number {i}", sessions_dir=d)
        assert count_sessions(d) == 5
        assert len(list_sessions(sessions_dir=d, limit=2)) == 2
