"""
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

The implicit outcome signals, and the text UI's rating path.

Two things are being tested here, and they are different kinds of test.

The signals are a *heuristic*, so their tests are about the cases that
would make them lie: a follow-up that merely shares vocabulary with the
task before it, a rephrase that is also a thank-you, a session's last turn
where nothing follows.  A heuristic tested only on its happy path is a
heuristic nobody can trust.

The rating path is *code*, so its tests are about the contract: which turn
gets rated, what happens when recording is off, what a malformed argument
does.  It has to survive contact with a real terminal.
"""

import json
import os

import pytest

from src.learn import signals as sig
from src.learn.corpus import (format_summary, label_corpus, read_labels,
                               write_labels)
from src.ui import rating as rating_mod


# ── the rephrase test ─────────────────────────────────────────────────────

class TestRephrase:

    def test_the_same_task_twice_is_a_rephrase(self):
        assert sig.is_rephrase("run the unit tests",
                               "run the unit tests") is True

    def test_a_reworded_task_is_a_rephrase(self):
        # Different words, same request.  This is the case a keyword
        # matcher misses and a similarity score catches.
        assert sig.is_rephrase("please run the unit tests for the parser",
                               "run the parser unit tests please") is True

    def test_a_correction_cue_is_a_rephrase(self):
        assert sig.is_rephrase("summarize the report",
                               "no, summarize the report") is True
        assert sig.is_rephrase("summarize the report",
                               "I said summarize, not summarize") is True

    def test_a_completely_different_task_is_not_a_rephrase(self):
        assert sig.is_rephrase("run the unit tests",
                               "now write a haiku about rain") is False

    def test_a_follow_up_that_shares_a_word_or_two_is_not_a_rephrase(self):
        # The threshold has to be high.  A multi-step task that mentions
        # the same noun as the previous one is a continuation, and calling
        # it a failure would label every real workflow as broken.
        assert sig.is_rephrase("summarize the quarterly report",
                               "now email it to the team") is False

    def test_an_empty_task_is_not_a_rephrase(self):
        assert sig.is_rephrase("", "run the tests") is False
        assert sig.is_rephrase("run the tests", "") is False

    def test_a_one_word_task_is_never_compared(self):
        # "yes" after "yes" shares everything and means nothing.
        assert sig.is_rephrase("yes", "yes") is False
        assert sig.similarity("hi", "hi") == 0.0

    def test_similarity_is_symmetric(self):
        a = "add a test for the parser"
        b = "add parser tests"
        assert sig.similarity(a, b) == pytest.approx(sig.similarity(b, a))


class TestSubstitution:

    def test_a_swapped_subject_is_a_substitution(self):
        # The corpus's most common false positive: "now do the same for
        # <other subject>".  Same shape, different task.
        a = ("make a complete mbpp benchmarking using Qwen/Qwen3.8-27B. "
             "Update the results table of scores.")
        b = ("make a complete bigcodebench benchmarking using "
             "Qwen/Qwen3.8-27B. Update the results table of scores.")
        assert sig.is_substitution(a, b) is True
        assert sig.is_rephrase(a, b) is False

    def test_a_reworded_task_is_not_a_substitution(self):
        # Adding a word and changing no subject is the same request.
        assert sig.is_substitution("run the unit tests",
                                   "run the parser unit tests") is False
        assert sig.is_rephrase("run the unit tests",
                               "run the parser unit tests") is True

    def test_the_same_task_is_not_a_substitution(self):
        assert sig.is_substitution("run the unit tests",
                                   "run the unit tests") is False

    def test_an_unrelated_task_is_not_a_substitution(self):
        assert sig.is_substitution("summarize the report",
                                   "write a haiku about rain") is False

    def test_a_very_different_task_is_not_a_substitution(self):
        # Too many differences to be a one-word swap.
        a = "benchmark mbpp with qwen and update the results table"
        b = ("deploy the service to staging with a canary and watch the "
             "latency for an hour")
        assert sig.is_substitution(a, b) is False


# ── the acceptance test ───────────────────────────────────────────────────

class TestAcceptance:

    def test_an_explicit_thanks_is_an_acceptance(self):
        assert sig.is_acceptance("summarize the report",
                                  "thanks, that worked") is True

    def test_a_thanks_followed_by_more_work_is_an_acceptance(self):
        assert sig.is_acceptance("summarize the report",
                                  "thanks, now do the same for Q3") is True

    def test_a_rephrase_is_not_also_an_acceptance(self):
        # "thanks, no, do it again" is a rephrase.  The order of the two
        # checks matters and this is where it shows.
        assert sig.is_acceptance("summarize the report",
                                  "thanks, no, do it again") is False

    def test_an_unrelated_next_task_is_not_an_acceptance(self):
        # Silence is not consent.  A follow-up that says nothing about the
        # previous answer must not be read as approval of it.
        assert sig.is_acceptance("summarize the report",
                                  "now list the files in the repo") is False

    def test_an_empty_task_is_not_an_acceptance(self):
        assert sig.is_acceptance("summarize the report", "") is False


# ── the pair label ────────────────────────────────────────────────────────

class TestLabelPair:

    def test_a_rephrase_labels_bad(self):
        assert sig.label_pair("run the tests", "no, run the tests") == "bad"

    def test_an_acceptance_labels_good(self):
        assert sig.label_pair("run the tests", "thanks, that worked") == "good"

    def test_anything_else_is_unknown(self):
        assert sig.label_pair("run the tests", "list the files") == "unknown"

    def test_unknown_is_the_common_case(self):
        # The honest default: most follow-ups say nothing.
        assert sig.label_pair("a", "b") == "unknown"


# ── the session label ─────────────────────────────────────────────────────

def _rec(task, turn=1):
    return {"task": task, "turn": turn}


class TestSessionLabelling:

    def test_an_empty_session_labels_nothing(self):
        assert sig.label_session([]) == []

    def test_a_one_task_session_labels_it_good(self):
        # The session ended on it and nobody objected.  It is the weakest
        # signal there is and it is the only one a one-task session has.
        assert sig.label_session([_rec("do the thing")]) == ["good"]

    def test_the_last_turn_is_labelled_from_the_session_end(self):
        labels = sig.label_session([_rec("first"), _rec("second")])
        assert labels == ["unknown", "good"]

    def test_a_middle_turn_is_labelled_from_what_followed(self):
        records = [
            _rec("run the tests"),
            _rec("no, run the tests again"),
            _rec("thanks"),
        ]
        assert sig.label_session(records) == ["bad", "good", "good"]

    def test_an_explicit_rating_is_never_overwritten(self):
        records = [
            _rec("run the tests"),
            _rec("no, run them again"),
        ]
        records[0]["signals"] = {"user_rating": 1}
        sig.label_records(records)
        assert records[0]["signals"].get("implicit") is None
        assert records[0]["signals"]["implicit_source"] == "explicit-rating"

    def test_the_source_says_where_the_label_came_from(self):
        records = [_rec("first"), _rec("no, first again")]
        sig.label_records(records)
        assert records[0]["signals"]["implicit_source"] == "follow-up"
        assert records[1]["signals"]["implicit_source"] == "session-end"

    def test_an_unlabelled_record_says_none(self):
        records = [_rec("first"), _rec("list the files")]
        sig.label_records(records)
        assert records[0]["signals"]["implicit"] == "unknown"
        assert records[0]["signals"]["implicit_source"] == "none"


# ── the summary ───────────────────────────────────────────────────────────

class TestSummary:

    def test_an_empty_corpus_summarises_to_nothing(self):
        summary = sig.summarize([])
        assert summary["total"] == 0
        assert summary["coverage"] == 0.0
        assert summary["positive_rate"] == 0.0

    def test_the_counts_include_the_unknowns(self):
        # A report that says "9 good, 1 bad" without saying how many were
        # unlabelled reads as a much stronger corpus than it is.
        records = [
            {"signals": {"implicit": "good"}},
            {"signals": {"implicit": "bad"}},
            {"signals": {"implicit": "unknown"}},
            {"signals": {}},
        ]
        summary = sig.summarize(records)
        assert summary["total"] == 4
        assert summary["good"] == 1
        assert summary["bad"] == 1
        assert summary["unknown"] == 2
        assert summary["labeled"] == 2
        assert summary["coverage"] == 0.5

    def test_the_positive_rate_is_over_the_labelled_part_only(self):
        records = [{"signals": {"implicit": "good"}}] * 3
        records += [{"signals": {"implicit": "bad"}}]
        records += [{"signals": {"implicit": "unknown"}}] * 6
        summary = sig.summarize(records)
        assert summary["labeled"] == 4
        assert summary["positive_rate"] == 0.75

    def test_explicit_ratings_are_counted_separately(self):
        records = [{"signals": {"user_rating": 1, "implicit": "good"}}]
        summary = sig.summarize(records)
        assert summary["explicit_ratings"] == 1


# ── the corpus, retroactively ─────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """A learned store with two sessions of recorded tasks."""
    config = {"learn": {"autonomy": "observe", "path": str(tmp_path / "learned")}}
    return config


def _write_session(config, session_id, tasks):
    from src.learn import record_task
    for turn, task in enumerate(tasks, start=1):
        record_task(session_id=session_id, turn=turn, task=task,
                    response="an answer", metrics={}, config_data=config)


class TestCorpusLabelling:

    def test_an_empty_store_summarises_to_nothing(self, store):
        summary = label_corpus(store)
        assert summary["total"] == 0
        assert summary["sessions"] == 0

    def test_a_session_is_labelled_from_its_own_follow_ups(self, store):
        _write_session(store, "s1", ["run the tests",
                                     "no, run the tests",
                                     "thanks"])
        summary = label_corpus(store)
        assert summary["total"] == 3
        assert summary["sessions"] == 1
        # first is bad (rephrased), second is good (thanked), third is good
        # (session end).
        assert summary["good"] == 2
        assert summary["bad"] == 1

    def test_sessions_are_not_labelled_against_each_other(self, store):
        # The bug this guards: a flat list of every record would label one
        # session's last turn against another session's first task.
        _write_session(store, "s1", ["run the tests"])
        _write_session(store, "s2", ["no, run the tests"])
        summary = label_corpus(store)
        # s1's only task ends its session -> good.  s2's only task also
        # ends its session -> good.  Neither is a rephrase of the other.
        assert summary["bad"] == 0
        assert summary["good"] == 2

    def test_the_trajectories_are_never_rewritten(self, store):
        # The store is the audit trail.  A backfill that rewrote it would
        # destroy the evidence a later audit needs.
        _write_session(store, "s1", ["run the tests", "no, run the tests"])
        from src.learn import read_session
        before = json.dumps(read_session("s1", store), sort_keys=True)
        label_corpus(store)
        after = json.dumps(read_session("s1", store), sort_keys=True)
        assert before == after

    def test_the_derived_labels_can_be_written_and_read_back(self, store):
        _write_session(store, "s1", ["run the tests", "no, run the tests"])
        path = write_labels(store)
        assert path and os.path.isfile(path)
        rows = read_labels(store)
        assert len(rows) == 2
        assert rows[0]["signals"]["implicit"] == "bad"

    def test_writing_is_refused_when_recording_is_off(self, tmp_path):
        config = {"learn": {"autonomy": "off", "path": str(tmp_path / "l")}}
        assert write_labels(config) is None

    def test_reading_labels_from_a_missing_file_is_empty(self, store):
        assert read_labels(store) == []


class TestFormatSummary:

    def test_an_empty_store_says_so(self):
        text = format_summary({"total": 0, "sessions": 0})
        assert "nothing to label" in text

    def test_the_report_shows_the_coverage(self):
        text = format_summary({"total": 10, "labeled": 4, "good": 3,
                                "bad": 1, "unknown": 6, "sessions": 1,
                                "coverage": 0.4, "positive_rate": 0.75,
                                "explicit_ratings": 0})
        assert "40.0%" in text
        assert "10" in text

    def test_the_report_names_the_weakest_signal(self):
        # A reader has to know which of the three signals they are looking
        # at, because the end-of-session one is the weakest.
        text = format_summary({"total": 1, "labeled": 1, "good": 1,
                                "bad": 0, "unknown": 0, "sessions": 1,
                                "coverage": 1.0, "positive_rate": 1.0,
                                "explicit_ratings": 0})
        assert "weakest" in text


# ── the text UI rating path ───────────────────────────────────────────────

class _Agent:
    """The minimum an agent has to look like for \\rate to work.

    The session file holds the conversation and the trajectory store holds
    the runs, and both have to exist: the rating is appended to the
    trajectory store, and it names a turn by the session file's count.
    """

    def __init__(self, tmp_path, turns=2, config=None):
        self.session_id = "sess-1"
        self.session_path = str(tmp_path / "session.jsonl")
        self.config_data = config if config is not None else {
            "learn": {"autonomy": "observe", "path": str(tmp_path / "learned")}
        }
        with open(self.session_path, "w", encoding="utf-8") as f:
            for index in range(turns):
                f.write(json.dumps({"task": "t%d" % index,
                                    "response": "r%d" % index}) + "\n")
        if turns:
            from src.learn import record_task
            for index in range(turns):
                record_task(session_id=self.session_id, turn=index + 1,
                            task="t%d" % index, response="r%d" % index,
                            metrics={}, config_data=self.config_data)


class TestRatingCommand:

    def test_rating_the_newest_answer(self, tmp_path):
        agent = _Agent(tmp_path)
        out = rating_mod.cmd_rate(agent, "up")
        assert "Recorded" in out
        assert "answer 2" in out

    def test_rating_a_specific_answer(self, tmp_path):
        agent = _Agent(tmp_path)
        out = rating_mod.cmd_rate(agent, "1 down")
        assert "answer 1" in out
        assert "wrong" in out

    def test_the_rating_lands_in_the_trajectory_store(self, tmp_path):
        from src.learn import read_session
        agent = _Agent(tmp_path)
        rating_mod.cmd_rate(agent, "down")
        records = read_session(agent.session_id, agent.config_data)
        assert len(records) == 2
        # The newest answer, and only that one, carries the verdict.
        assert records[1]["signals"]["user_rating"] == -1
        assert records[0]["signals"].get("user_rating") is None

    def test_the_rating_carries_its_source(self, tmp_path):
        from src.learn import read_session
        agent = _Agent(tmp_path)
        rating_mod.cmd_rate(agent, "up")
        records = read_session(agent.session_id, agent.config_data)
        # The source distinguishes a terminal verdict from a web thumb, so
        # a loop can weight the evidence by where it came from.
        assert records[1]["signals"]["user_rating"] == 1
        assert records[1]["signals"]["user_rating_source"] == "text"

    def test_no_argument_rates_the_newest_answer_up(self, tmp_path):
        agent = _Agent(tmp_path)
        out = rating_mod.cmd_rate(agent, "")
        assert "answer 2" in out
        assert "good" in out

    def test_clearing_removes_the_verdict(self, tmp_path):
        from src.learn import read_session
        agent = _Agent(tmp_path)
        rating_mod.cmd_rate(agent, "up")
        out = rating_mod.cmd_rate(agent, "clear")
        assert "Cleared" in out
        records = read_session(agent.session_id, agent.config_data)
        assert records[1]["signals"].get("user_rating") is None

    def test_an_out_of_range_turn_is_refused(self, tmp_path):
        agent = _Agent(tmp_path, turns=2)
        out = rating_mod.cmd_rate(agent, "5 up")
        assert "no answer 5" in out

    def test_a_turn_below_one_is_refused(self, tmp_path):
        agent = _Agent(tmp_path)
        out = rating_mod.cmd_rate(agent, "0 up")
        assert "no answer 0" in out

    def test_recording_off_is_reported_honestly(self, tmp_path):
        agent = _Agent(tmp_path, config={
            "learn": {"autonomy": "off", "path": str(tmp_path / "learned")}})
        out = rating_mod.cmd_rate(agent, "up")
        assert "not being recorded" in out

    def test_a_session_with_no_answers_says_so(self, tmp_path):
        agent = _Agent(tmp_path, turns=0)
        out = rating_mod.cmd_rate(agent, "up")
        assert "No answer" in out

    def test_a_missing_session_file_says_so(self, tmp_path):
        agent = _Agent(tmp_path)
        agent.session_path = str(tmp_path / "nope.jsonl")
        out = rating_mod.cmd_rate(agent, "up")
        assert "No answer" in out

    def test_an_unknown_word_is_refused_with_the_accepted_ones(self, tmp_path):
        agent = _Agent(tmp_path)
        out = rating_mod.cmd_rate(agent, "sideways")
        assert "up" in out and "down" in out and "clear" in out

    def test_a_missing_verdict_is_refused(self, tmp_path):
        agent = _Agent(tmp_path)
        out = rating_mod.cmd_rate(agent, "3")
        assert "needs a verdict" in out

    def test_two_turn_numbers_are_refused(self, tmp_path):
        agent = _Agent(tmp_path)
        out = rating_mod.cmd_rate(agent, "1 2 up")
        assert "one turn number" in out

    def test_the_command_is_reachable_as_a_backslash_command(self):
        from src.ui import commands
        assert "\\rate" in commands.render_help()
        assert commands.parse("\\rate up") == ("rate", "up")
        assert commands.parse("\\rate 3 down") == ("rate", "3 down")


class TestRatingArgumentParsing:

    def test_the_default_is_up(self):
        assert rating_mod._parse_arg("") == (None, 1, "")

    def test_the_spellings(self):
        for word in ("up", "good", "yes", "+"):
            assert rating_mod._parse_arg(word)[1] == 1
        for word in ("down", "bad", "no", "-"):
            assert rating_mod._parse_arg(word)[1] == -1
        for word in ("clear", "none", "null", "undo"):
            assert rating_mod._parse_arg(word)[1] is None

    def test_a_bare_number_is_a_turn_not_a_verdict(self):
        # "1" is a turn number.  If it also meant "up", then "\rate 1"
        # would be two different things depending on which the user meant.
        turn, rating, error = rating_mod._parse_arg("1")
        assert turn == 1
        assert rating is None
        assert error != ""

    def test_the_order_of_the_parts_does_not_matter(self):
        assert rating_mod._parse_arg("down 2") == (2, -1, "")
        assert rating_mod._parse_arg("2 down") == (2, -1, "")

    def test_an_empty_argument_is_the_default(self):
        assert rating_mod._parse_arg(None) == (None, 1, "")


# ── the CLI surface ───────────────────────────────────────────────────────

class TestLearnSignalsCommand:

    def test_the_flags_exist(self):
        from src.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["learn", "--signals"])
        assert args.command == "learn"
        assert args.signals is True
        assert args.write_signals is False
        args = parser.parse_args(["learn", "--signals", "--write-signals"])
        assert args.write_signals is True
