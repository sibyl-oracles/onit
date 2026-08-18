"""Tests for src/model/serving/state.py — RunState and its session file."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.state import (MAX_PERSISTED_CALLS, MAX_RESUME_TOOLS,
                                 RunState, STOP_ANSWERED, STOP_TURN_LIMIT,
                                 state_path_for)


# ── the shape of a fresh state ──────────────────────────────────────────────

class TestDefaults:
    def test_a_fresh_state_has_spent_nothing(self):
        state = RunState()
        assert state.iteration_count == 0
        assert state.tool_call_history == []
        assert state.planning_continuation_count == 0
        assert state.ack_continuation_count == 0
        assert state.final_continuation_count == 0
        assert state.stop_reason == ""

    def test_active_max_tokens_starts_unset(self):
        """None, not a number: its opening value is chat()'s max_tokens, which
        this object has no way to know."""
        assert RunState().active_max_tokens is None

    def test_two_states_do_not_share_a_history(self):
        a, b = RunState(), RunState()
        a.tool_call_history.append(("bash", "{}"))
        assert b.tool_call_history == []


# ── tool_counts ─────────────────────────────────────────────────────────────

class TestToolCounts:
    def test_counts_by_name_in_first_call_order(self):
        state = RunState()
        state.tool_call_history = [("bash", "{}"), ("read_file", '{"p":1}'),
                                   ("bash", '{"c":"ls"}')]
        assert list(state.tool_counts().items()) == [("bash", 2), ("read_file", 1)]

    def test_empty_history_counts_nothing(self):
        assert RunState().tool_counts() == {}


# ── resume_note ─────────────────────────────────────────────────────────────

class TestResumeNote:
    def test_a_session_that_did_nothing_says_nothing(self):
        assert RunState().resume_note() == ""

    def test_names_the_tools_and_their_counts(self):
        state = RunState()
        state.tool_call_history = [("bash", "{}"), ("bash", '{"c":"ls"}'),
                                   ("local_search", "{}")]
        note = state.resume_note()
        assert "bash×2" in note
        assert "local_search" in note and "local_search×" not in note

    def test_marked_reference_only(self):
        """Same hazard as the context block: a section describing past work is
        exactly what a weak model restates instead of acting on."""
        state = RunState()
        state.tool_call_history = [("bash", "{}")]
        assert "Reference only" in state.resume_note()

    def test_an_early_stop_is_reported(self):
        state = RunState(stop_reason=STOP_TURN_LIMIT)
        state.tool_call_history = [("bash", "{}")]
        assert "turn limit" in state.resume_note()

    def test_a_run_that_answered_is_not_flagged_as_early(self):
        state = RunState(stop_reason=STOP_ANSWERED)
        state.tool_call_history = [("bash", "{}")]
        assert "ended early" not in state.resume_note()

    def test_the_tool_list_is_bounded(self):
        state = RunState()
        state.tool_call_history = [(f"tool_{i}", "{}") for i in range(40)]
        note = state.resume_note()
        assert "tool_0" in note
        assert f"and {40 - MAX_RESUME_TOOLS} more" in note

    def test_turns_spent_are_reported_when_known(self):
        state = RunState(total_turns=7)
        state.tool_call_history = [("bash", "{}")]
        assert "Turns spent so far: 7" in state.resume_note()


# ── merge ───────────────────────────────────────────────────────────────────

class TestMerge:
    def test_history_accumulates_and_counters_describe_the_last_run(self):
        session = RunState()
        session.tool_call_history = [("bash", "{}")]
        session.total_turns = 4

        finished = RunState(iteration_count=3, planning_continuation_count=2,
                            stop_reason=STOP_TURN_LIMIT)
        finished.tool_call_history = [("search", "{}")]

        session.merge(finished)

        assert session.tool_call_history == [("bash", "{}"), ("search", "{}")]
        assert session.planning_continuation_count == 2
        assert session.stop_reason == STOP_TURN_LIMIT
        assert session.total_turns == 7  # 4 + 3
        assert session.task_count == 1

    def test_answer_text_is_not_carried_between_tasks(self):
        """A half-written answer belongs to the task that was writing it."""
        session = RunState()
        finished = RunState(final_answer_prefix="half an answer",
                            prose_before_tools="Now let me")
        session.merge(finished)
        assert session.final_answer_prefix == ""
        assert session.prose_before_tools == ""

    def test_history_is_trimmed_from_the_front(self):
        session = RunState()
        finished = RunState()
        finished.tool_call_history = [("bash", str(i))
                                      for i in range(MAX_PERSISTED_CALLS + 25)]
        session.merge(finished)
        assert len(session.tool_call_history) == MAX_PERSISTED_CALLS
        # The most recent survive; the oldest are what went.
        assert session.tool_call_history[-1] == ("bash", str(MAX_PERSISTED_CALLS + 24))
        assert ("bash", "0") not in session.tool_call_history


# ── serialization ───────────────────────────────────────────────────────────

class TestSerialization:
    def test_round_trip_preserves_history_and_counters(self):
        state = RunState(iteration_count=5, ack_continuation_count=1,
                         stop_reason=STOP_ANSWERED, task_count=2, total_turns=9)
        state.tool_call_history = [("bash", '{"command":"ls"}')]
        back = RunState.from_dict(json.loads(json.dumps(state.to_dict())))
        assert back.iteration_count == 5
        assert back.ack_continuation_count == 1
        assert back.stop_reason == STOP_ANSWERED
        assert back.task_count == 2
        assert back.total_turns == 9
        assert back.tool_call_history == [("bash", '{"command":"ls"}')]

    def test_history_comes_back_as_tuples(self):
        """JSON has no tuples.  The live list is counted against tuple keys, so
        a list read back would never match one and the repeated-call detector
        would silently stop detecting."""
        back = RunState.from_dict({"tool_call_history": [["bash", "{}"]]})
        assert back.tool_call_history == [("bash", "{}")]
        assert isinstance(back.tool_call_history[0], tuple)

    def test_answer_text_is_not_persisted(self):
        state = RunState(final_answer_prefix="draft", prose_before_tools="Now let me")
        assert "final_answer_prefix" not in state.to_dict()
        assert "prose_before_tools" not in state.to_dict()

    @pytest.mark.parametrize("junk", [None, [], "nonsense", 42])
    def test_junk_yields_a_fresh_state_rather_than_an_error(self, junk):
        assert RunState.from_dict(junk).tool_call_history == []

    def test_a_partial_record_loads_what_it_has(self):
        back = RunState.from_dict({"iteration_count": "not a number",
                                   "stop_reason": STOP_TURN_LIMIT})
        assert back.iteration_count == 0
        assert back.stop_reason == STOP_TURN_LIMIT


# ── the file beside the session ─────────────────────────────────────────────

class TestStateFile:
    def test_path_sits_beside_the_session_file(self):
        assert (state_path_for("/tmp/sessions/abc.jsonl")
                == "/tmp/sessions/abc.state.json")

    def test_no_session_no_path(self):
        assert state_path_for("") == ""

    def test_save_then_load(self, tmp_path):
        path = state_path_for(str(tmp_path / "s.jsonl"))
        state = RunState(iteration_count=3, stop_reason=STOP_TURN_LIMIT)
        state.tool_call_history = [("bash", "{}")]
        assert state.save(path) is True

        back = RunState.load(path)
        assert back.iteration_count == 3
        assert back.stop_reason == STOP_TURN_LIMIT
        assert back.tool_call_history == [("bash", "{}")]

    def test_loading_nothing_gives_an_empty_state(self, tmp_path):
        back = RunState.load(str(tmp_path / "never-written.state.json"))
        assert back.tool_call_history == []
        assert back.task_count == 0

    def test_a_torn_file_does_not_raise(self, tmp_path):
        path = tmp_path / "torn.state.json"
        path.write_text('{"iteration_count": 3, "tool_call')
        assert RunState.load(str(path)).iteration_count == 0

    def test_save_leaves_no_temp_file_behind(self, tmp_path):
        path = state_path_for(str(tmp_path / "s.jsonl"))
        RunState(iteration_count=1).save(path)
        assert not os.path.exists(f"{path}.tmp")

    def test_an_unwritable_path_is_reported_not_raised(self, tmp_path):
        """Bookkeeping that cannot be written is a session that resumes with
        less, never a task that failed."""
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        assert RunState().save(str(blocker / "sub" / "s.state.json")) is False
