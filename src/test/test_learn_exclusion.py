"""
The exclusion boundary, tested adversarially.

The holdout is the one thing in the learning system that the agent must
never see, and there are two ways it can leak:

1. **By record id.**  The obvious one: the holdout item's own recorded
   run is recalled back to the agent.
2. **By question.**  The one that actually bites.  The corpus holds the
   same question asked many times -- "weather today" was asked 21
   times, "marikina river water level latest" 6 -- and every one of
   those runs carries an answer to it.  A holdout drawn from that corpus
   excludes exactly one of the twenty-one runs, and the other twenty
   are still in the recall pool, each carrying the answer to the
   question the agent is about to be asked.

Measured on the real corpus before the fix: 158 of 160 holdout items
recalled a run of their own question.  After: 0.

This file pins that number, and pins the other half -- that the
exclusion is not so broad that recall stops working, because an agent
with amnesia and an agent whose memory does not help produce the same
null result and mean different things.
"""

import json
import os

import pytest

from src.learn import eval as ruler
from src.learn import playbook, recall, trajectory


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def learned(tmp_path):
    """An empty learned store and an isolated ruler."""
    root = str(tmp_path / "learned")
    ruler_root = str(tmp_path / "ruler")
    os.makedirs(root, exist_ok=True)
    os.makedirs(ruler_root, exist_ok=True)
    return {"learn": {"path": root, "eval": {"path": ruler_root}},
            "data_path": str(tmp_path / "data")}


def _record(session="s1", turn=1, task="a question worth recording",
            response="a long enough answer", model="m1", owner=None,
            verdict=None, tools=None):
    """One trajectory record, shaped like the real ones."""
    return {
        "kind": "task",
        "session_id": session,
        "turn": turn,
        "task": task,
        "response": response,
        "model": model,
        "owner": owner,
        "ts": "2026-10-20T10:00:00",
        "trajectory": [{"turn": turn, "tools": tools or []}],
        "signals": {"verifier": verdict} if verdict else {},
    }


def _write(learned, record):
    path = os.path.join(trajectory.trajectory_dir(learned),
                        "%s.jsonl" % record["session_id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def _holdout(learned, items):
    """Write a holdout file directly."""
    directory = ruler.eval_dir(learned)
    os.makedirs(directory, exist_ok=True)
    with open(ruler.holdout_path(learned), "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


# ── task_key: the identity of a question ────────────────────────────────────

class TestTaskKey:
    """The question key is what makes the exclusion work at all."""

    def test_the_same_question_has_the_same_key(self):
        assert ruler.task_key("weather today") == ruler.task_key("weather today")

    def test_case_and_punctuation_do_not_make_a_new_question(self):
        assert ruler.task_key("Weather Today?") == ruler.task_key("weather today")

    def test_extra_whitespace_does_not_make_a_new_question(self):
        assert ruler.task_key("  weather   today  ") == ruler.task_key("weather today")

    def test_different_questions_have_different_keys(self):
        assert ruler.task_key("weather today") != ruler.task_key("weather tomorrow")

    def test_an_empty_question_has_an_empty_key(self):
        assert ruler.task_key("") == ""
        assert ruler.task_key(None) == ""


# ── the exclusion set ───────────────────────────────────────────────────────

class TestExclusions:
    """The exclusion set holds both ids and questions."""

    def test_the_exclusions_carry_the_ids(self, learned):
        _holdout(learned, [{"id": "s1#1", "task": "a question"}])
        assert "s1#1" in ruler.excluded_record_ids(learned)

    def test_the_exclusions_carry_the_questions(self, learned):
        _holdout(learned, [{"id": "s1#1", "task": "A Question?"}])
        assert ruler.task_key("a question") in ruler.excluded_task_keys(learned)

    def test_no_holdout_means_no_exclusions(self, learned):
        exclusions = ruler.holdout_exclusions(learned)
        assert exclusions["ids"] == set()
        assert exclusions["tasks"] == set()

    def test_a_missing_holdout_file_is_not_an_error(self, learned):
        assert ruler.holdout_exclusions(learned) == {"ids": set(), "tasks": set()}

    def test_a_torn_holdout_line_does_not_break_the_exclusions(self, learned):
        directory = ruler.eval_dir(learned)
        os.makedirs(directory, exist_ok=True)
        with open(ruler.holdout_path(learned), "w", encoding="utf-8") as f:
            f.write('{"id": "s1#1", "task": "a question"}\n')
            f.write("{not json at all\n")
        assert "s1#1" in ruler.excluded_record_ids(learned)


# ── the predicate ───────────────────────────────────────────────────────────

class TestIsExcluded:
    """One predicate, so the boundary is in one place."""

    def test_a_holdout_record_is_excluded(self, learned):
        _holdout(learned, [{"id": "s1#1", "task": "a question"}])
        record = _record(session="s1", turn=1, task="a question")
        assert ruler.is_excluded(record, learned)

    def test_a_sibling_run_of_the_holdout_question_is_excluded(self, learned):
        """The leak that actually bites."""
        _holdout(learned, [{"id": "s1#1", "task": "a question"}])
        sibling = _record(session="s2", turn=1, task="a question")
        assert ruler.is_excluded(sibling, learned)

    def test_a_sibling_run_differing_only_in_punctuation_is_excluded(self, learned):
        _holdout(learned, [{"id": "s1#1", "task": "a question"}])
        sibling = _record(session="s2", turn=1, task="A Question!")
        assert ruler.is_excluded(sibling, learned)

    def test_an_unrelated_record_is_not_excluded(self, learned):
        _holdout(learned, [{"id": "s1#1", "task": "a question"}])
        other = _record(session="s3", turn=1, task="a completely different one")
        assert not ruler.is_excluded(other, learned)

    def test_nothing_is_excluded_when_no_holdout_exists(self, learned):
        record = _record()
        assert not ruler.is_excluded(record, learned)


# ── the filter ──────────────────────────────────────────────────────────────

class TestFilterExcluded:
    def test_the_filter_drops_the_holdout_and_its_siblings(self, learned):
        _holdout(learned, [{"id": "s1#1", "task": "a question"}])
        records = [
            _record(session="s1", turn=1, task="a question"),
            _record(session="s2", turn=1, task="a question"),
            _record(session="s3", turn=1, task="a different question here"),
        ]
        kept = ruler.filter_excluded(records, learned)
        assert [r["session_id"] for r in kept] == ["s3"]

    def test_the_filter_keeps_everything_when_no_holdout_exists(self, learned):
        records = [_record(), _record(session="s2")]
        assert len(ruler.filter_excluded(records, learned)) == 2


# ── recall: the real path ───────────────────────────────────────────────────

class TestRecallExcludesTheHoldout:
    """The measurement path, not the predicate."""

    def test_recall_does_not_return_the_holdout_question(self, learned):
        _write(learned, _record(session="s1", turn=1, task="weather today",
                                response="the answer"))
        _holdout(learned, [{"id": "s1#1", "task": "weather today"}])
        episodes = recall.recall("weather today", config_data=learned)
        assert all(e["id"] != "s1#1" for e in episodes)

    def test_recall_does_not_return_a_sibling_of_the_holdout_question(self,
                                                                      learned):
        """The twenty siblings.

        This is the test that would have caught the original bug.  The
        holdout names one record; the store holds another run of the
        same question; the agent must not be shown the second one.
        """
        _write(learned, _record(session="s1", turn=1, task="weather today",
                                response="the answer"))
        _write(learned, _record(session="s2", turn=1, task="weather today",
                                response="the answer"))
        _holdout(learned, [{"id": "s1#1", "task": "weather today"}])
        episodes = recall.recall("weather today", config_data=learned)
        assert all(e["id"] != "s2#1" for e in episodes), (
            "recall returned a sibling run of the holdout's question")

    def test_recall_still_works_for_questions_it_may_learn_from(self, learned):
        _write(learned, _record(session="s1", turn=1,
                                task="check the weather in manila",
                                response="a long enough answer"))
        _holdout(learned, [{"id": "h1", "task": "something else entirely"}])
        episodes = recall.recall("check the weather in manila",
                                 config_data=learned)
        assert episodes, "the exclusion swallowed the corpus"

    def test_the_exclusion_can_be_turned_off_for_inspection(self, learned):
        """A measurement tool has to be able to show the leak.

        The exclusion is on by default, but the diagnostic that proved
        the leak existed needs to be able to reproduce it.
        """
        _write(learned, _record(session="s1", turn=1, task="weather today",
                                response="the answer"))
        _holdout(learned, [{"id": "s1#1", "task": "weather today"}])
        episodes = recall.recall("weather today", config_data=learned,
                                 exclude_holdout=False)
        assert any(e["id"] == "s1#1" for e in episodes), (
            "the leak could not be reproduced, so the diagnostic is broken")


# ── reflect: the other path ─────────────────────────────────────────────────

class TestReflectExcludesTheHoldout:
    """A bullet written from the exam is a bullet written from the exam."""

    def test_reflect_does_not_use_a_holdout_run(self, learned):
        good = _record(session="s1", turn=1, task="the held out question",
                       response="a long enough answer", verdict="clean",
                       tools=[{"name": "search", "ok": True}])
        bad = _record(session="s2", turn=1, task="the held out question",
                      response="a long enough answer", verdict="issues",
                      tools=[{"name": "search", "ok": False}])
        _write(learned, good)
        _write(learned, bad)
        _holdout(learned, [{"id": "s1#1", "task": "the held out question"}])
        deltas = playbook.reflect([good, bad], scope=playbook.SCOPE_GLOBAL,
                                  config_data=learned)
        assert not deltas, "reflect produced a bullet from a holdout run"

    def test_reflect_still_works_outside_the_holdout(self, learned):
        good = _record(session="s1", turn=1, task="an ordinary question",
                       response="a long enough answer", verdict="clean",
                       tools=[{"name": "search", "ok": True}])
        bad = _record(session="s2", turn=1, task="an ordinary question",
                      response="a long enough answer", verdict="issues",
                      tools=[{"name": "search", "ok": False}])
        _write(learned, good)
        _write(learned, bad)
        _holdout(learned, [{"id": "h1", "task": "something else entirely"}])
        deltas = playbook.reflect([good, bad], scope=playbook.SCOPE_GLOBAL,
                                  config_data=learned)
        assert deltas, "reflect found no contrast it should have found"


# ── the provenance ──────────────────────────────────────────────────────────

class TestProvenance:
    """A bullet must point at a run, not at a session.

    The provenance is what an audit follows: "where did this lesson
    come from?"  A session id is not precise enough, because a session
    file holds one record per turn, and a session can contain a turn
    whose question is in the holdout and a turn whose question is not.

    The exclusion drops the holdout turn correctly.  But the surviving
    turn's *session* id still points at a session that contains one --
    so the holdout's answer would not be in the lesson, but its
    existence would be in the pointer.
    """

    def test_the_provenance_names_the_record_not_the_session(self, learned):
        """The provenance must be checkable against the exclusion set.

        It has to be the record id -- ``session_id#turn`` -- because
        that is the identity the holdout's exclusion set is keyed on.
        A session id cannot be checked at all.
        """
        good = _record(session="s1", turn=1, task="an ordinary question",
                       response="a long enough answer", verdict="clean",
                       tools=[{"name": "search", "ok": True}])
        bad = _record(session="s2", turn=1, task="an ordinary question",
                      response="a long enough answer", verdict="issues",
                      tools=[{"name": "search", "ok": False}])
        _write(learned, good)
        _write(learned, bad)
        deltas = playbook.reflect([good, bad], scope=playbook.SCOPE_GLOBAL,
                                  config_data=learned)
        assert deltas
        for delta in deltas:
            for ref in delta.get("provenance") or []:
                assert "#" in ref, (
                    "the provenance names a session (%r), which cannot be "
                    "checked against the exclusion set" % ref)

    def test_a_bullet_never_points_at_a_holdout_run(self, learned):
        """The whole point, as a test.

        A session holds both a holdout turn and a surviving turn; the
        bullet is distilled from the surviving one; the provenance must
        not name the holdout turn.
        """
        holdout_turn = _record(session="s1", turn=2,
                               task="the held out question",
                               response="a long enough answer", verdict="clean",
                               tools=[{"name": "search", "ok": True}])
        kept_turn = _record(session="s1", turn=1,
                             task="a different question here",
                             response="a long enough answer", verdict="clean",
                             tools=[{"name": "search", "ok": True}])
        bad = _record(session="s2", turn=1, task="a different question here",
                      response="a long enough answer", verdict="issues",
                      tools=[{"name": "search", "ok": False}])
        for record in (holdout_turn, kept_turn, bad):
            _write(learned, record)
        _holdout(learned, [{"id": "s1#2", "task": "the held out question"}])

        deltas = playbook.reflect([holdout_turn, kept_turn, bad],
                                  scope=playbook.SCOPE_GLOBAL,
                                  config_data=learned)
        excluded = ruler.holdout_exclusions(learned)
        for delta in deltas:
            for ref in delta.get("provenance") or []:
                assert ref not in excluded["ids"], (
                    "a bullet points at the holdout run %r" % ref)


# ── the split filter ────────────────────────────────────────────────────────

class TestSplitFilter:
    """``iter_records`` can be asked for one split."""

    def test_the_holdout_split_yields_only_holdout_records(self, learned):
        for i in range(6):
            _write(learned, _record(session="s%d" % i, turn=1,
                                    task="a question number %d here" % i,
                                    response="a long enough answer"))
        split = ruler.build_split(config_data=learned, holdout_ratio=0.5)
        ruler.write_split(split, learned)
        holdout_ids = {i["id"] for i in split["holdout"]}
        records = list(trajectory.iter_records(learned, split="holdout"))
        assert records
        assert all("%s#%s" % (r["session_id"], r["turn"]) in holdout_ids
                   for r in records)

    def test_the_dev_split_yields_only_dev_records(self, learned):
        for i in range(6):
            _write(learned, _record(session="s%d" % i, turn=1,
                                    task="a question number %d here" % i,
                                    response="a long enough answer"))
        split = ruler.build_split(config_data=learned, holdout_ratio=0.5)
        ruler.write_split(split, learned)
        dev_ids = {i["id"] for i in split["dev"]}
        records = list(trajectory.iter_records(learned, split="dev"))
        assert records
        assert all("%s#%s" % (r["session_id"], r["turn"]) in dev_ids
                   for r in records)

    def test_exclude_holdout_removes_the_holdout(self, learned):
        for i in range(6):
            _write(learned, _record(session="s%d" % i, turn=1,
                                    task="a question number %d here" % i,
                                    response="a long enough answer"))
        split = ruler.build_split(config_data=learned, holdout_ratio=0.5)
        ruler.write_split(split, learned)
        everything = list(trajectory.iter_records(learned))
        without = list(trajectory.iter_records(learned, exclude_holdout=True))
        assert len(without) < len(everything)
        assert len(without) == len(split["dev"])

    def test_the_two_splits_partition_the_corpus(self, learned):
        for i in range(6):
            _write(learned, _record(session="s%d" % i, turn=1,
                                    task="a question number %d here" % i,
                                    response="a long enough answer"))
        split = ruler.build_split(config_data=learned, holdout_ratio=0.5)
        ruler.write_split(split, learned)
        holdout = list(trajectory.iter_records(learned, split="holdout"))
        dev = list(trajectory.iter_records(learned, split="dev"))
        assert len(holdout) + len(dev) == len(list(trajectory.iter_records(learned)))
