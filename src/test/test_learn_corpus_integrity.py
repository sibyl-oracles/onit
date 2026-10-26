"""Three bugs that made the loops measure the wrong thing.

Each test here was written after the bug was found and fixed against
the real corpus, and each one is a *regression* test: revert the fix
and this file fails.

The three are related, and they compound:

  A  the walk read the derived files as if they were recorded runs, so
     the corpus reported twice the tasks it held
  B  the labels never reached the reflector, so a contrast had no bad
     side and the reflector produced nothing
  C  a human rating reached no loop at all, so the strongest evidence
     in the corpus was invisible

Together they mean the self-improvement machinery was running, and
reporting, on a corpus that was not the corpus.
"""

import json
import os

import pytest

from src.learn import corpus, playbook, recall, signals as sig
from src.learn import trajectory


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def learned(tmp_path):
    """An empty learned store, isolated from the real one."""
    root = str(tmp_path / "learned")
    ruler_root = str(tmp_path / "ruler")
    os.makedirs(root, exist_ok=True)
    os.makedirs(ruler_root, exist_ok=True)
    yield {"learn": {"path": root, "eval": {"path": ruler_root}}}


def _task_record(session, turn, task, response="an answer", **extra):
    record = {
        "kind": "task",
        "session_id": session,
        "turn": turn,
        "task": task,
        "response": response,
        "model": "m1",
        "owner": None,
        "ts": "2026-10-20T10:00:00",
        "trajectory": [],
        "signals": {},
    }
    record.update(extra)
    return record


def _write(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# ── A: the derived files are not sessions ────────────────────────────────

class TestTheWalkDoesNotReadDerivedFiles:
    """A store of N tasks must report N tasks.

    The bug: ``labels.jsonl`` and ``loop.jsonl`` sit in the trajectory
    directory and end in ".jsonl", and the walk took every one of them
    for a session.  A store of 1,034 tasks reported 2,056 - the label
    rows were read back as if they were recorded runs.
    """

    def test_a_label_file_beside_the_trajectories_is_not_a_session(self, learned):
        tdir = trajectory.trajectory_dir(learned)
        _write(os.path.join(tdir, "s1.jsonl"),
               [_task_record("s1", 1, "run the tests")])
        # The derived labels, in the old place.
        _write(os.path.join(tdir, "labels.jsonl"),
               [{"session_id": "s1", "turn": 1, "task": "run the tests",
                 "signals": {"implicit": "good"}}])

        records = list(trajectory.iter_records(learned))
        assert len(records) == 1, \
            "the label rows were read back as recorded runs"
        assert records[0]["task"] == "run the tests"

    def test_the_event_log_beside_the_trajectories_is_not_a_session(self, learned):
        tdir = trajectory.trajectory_dir(learned)
        _write(os.path.join(tdir, "s1.jsonl"),
               [_task_record("s1", 1, "run the tests")])
        _write(os.path.join(tdir, "loop.jsonl"),
               [{"kind": "event", "event": "started", "ts": "x"}])

        records = list(trajectory.iter_records(learned))
        assert len(records) == 1, "the event log was read back as a run"

    def test_a_record_with_no_response_is_not_a_run(self, learned):
        """A label row has a task and a turn but no response.

        The response is the only thing that makes a line a *run*, and
        it is the one field a derived label cannot have.
        """
        tdir = trajectory.trajectory_dir(learned)
        path = os.path.join(tdir, "s1.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_task_record("s1", 1, "a real run")) + "\n")
            f.write(json.dumps({"session_id": "s1", "turn": 2,
                                "task": "a label, not a run"}) + "\n")

        records = list(trajectory.iter_records(learned))
        assert len(records) == 1
        assert records[0]["task"] == "a real run"

    def test_the_labels_file_lives_outside_the_trajectory_directory(self, learned):
        """A derived file belongs outside the record it is derived from."""
        path = corpus.labels_path(learned)
        assert os.path.dirname(path) == corpus.learn_root(learned)
        assert path != os.path.join(trajectory.trajectory_dir(learned),
                                    "labels.jsonl")

    def test_a_derived_file_is_excluded_by_name_even_if_it_looks_like_a_run(self, learned):
        """The name screen defends the case the shape check cannot.

        The shape check rejects a derived row that carries no response,
        which is what today's label rows look like.  A derived file
        whose rows *do* carry a response would pass the shape check,
        and only the name screen keeps it out.  Both are kept: the
        screen is cheap, and the two do not depend on each other.
        """
        tdir = trajectory.trajectory_dir(learned)
        _write(os.path.join(tdir, "s1.jsonl"),
               [_task_record("s1", 1, "run the tests")])
        # A derived file that looks exactly like a session.
        _write(os.path.join(tdir, "labels.jsonl"),
               [_task_record("s1", 1, "run the tests", response="an answer")])

        records = list(trajectory.iter_records(learned))
        assert len(records) == 1, \
            "a derived file was read as a session"

    def test_a_session_id_is_not_required_to_be_a_uuid(self, learned):
        """The store accepts any session name, so the walk must too.

        The first attempt at this fix filtered on "is the name a
        UUID".  That silently dropped every session the tests wrote
        ("s1") and every session written by a caller that names its
        own.  The name is not the discriminator; the record's shape
        is.
        """
        tdir = trajectory.trajectory_dir(learned)
        _write(os.path.join(tdir, "s1.jsonl"),
               [_task_record("s1", 1, "run the tests")])
        _write(os.path.join(tdir, "not-a-uuid.jsonl"),
               [_task_record("not-a-uuid", 1, "another task")])

        records = list(trajectory.iter_records(learned))
        assert len(records) == 2, "a session was dropped for its name"


# ── B: the labels reach the reader ───────────────────────────────────────

class TestTheLabelsReachTheReader:
    """A record read from disk must carry the label the loops read.

    The bug: ``label_corpus`` annotated the records it had in memory
    and returned.  Nothing wrote them back, and the one production
    caller re-read the corpus from disk, so the reflector saw records
    with no ``signals.implicit`` at all.
    """

    def test_a_record_read_from_disk_carries_the_implicit_label(self, learned):
        tdir = trajectory.trajectory_dir(learned)
        _write(os.path.join(tdir, "s1.jsonl"), [
            _task_record("s1", 1, "run the tests"),
            _task_record("s1", 2, "no, run the tests"),
        ])

        records = list(trajectory.iter_records(learned))
        assert records, "nothing was read"
        for record in records:
            assert record["signals"].get("implicit") is not None, \
                "a record read from disk has no label"

    def test_the_labels_survive_a_second_read(self, learned):
        """The label must be a property of the record, not of one read.

        This is the shape the bug took: the first read was labelled,
        a second read was not.  A loop that reads the corpus twice -
        once to report, once to reflect - got two different corpora.
        """
        tdir = trajectory.trajectory_dir(learned)
        _write(os.path.join(tdir, "s1.jsonl"), [
            _task_record("s1", 1, "run the tests"),
            _task_record("s1", 2, "no, run the tests"),
        ])

        first = list(trajectory.iter_records(learned))
        second = list(trajectory.iter_records(learned))
        assert [r["signals"].get("implicit") for r in first] == \
               [r["signals"].get("implicit") for r in second]

    def test_a_rephrase_is_labelled_bad_on_read(self, learned):
        tdir = trajectory.trajectory_dir(learned)
        _write(os.path.join(tdir, "s1.jsonl"), [
            _task_record("s1", 1, "run the tests"),
            _task_record("s1", 2, "no, run the tests"),
        ])
        records = list(trajectory.iter_records(learned))
        assert records[0]["signals"]["implicit"] == "bad"
        assert records[0]["signals"]["implicit_source"] == "follow-up"

    def test_the_contrast_has_both_sides_after_a_read(self, learned):
        """The end of the chain the bug broke.

        A contrast with no bad side is not a contrast, and a reflector
        given one produces nothing and reports nothing wrong.
        """
        tdir = trajectory.trajectory_dir(learned)
        _write(os.path.join(tdir, "s1.jsonl"), [
            _task_record("s1", 1, "run the tests"),
            _task_record("s1", 2, "no, run the tests"),
        ])
        records = list(trajectory.iter_records(learned))
        result = playbook.contrast(records)
        assert result["good"], "the contrast has no good side"
        assert result["bad"], "the contrast has no bad side"

    def test_the_recorder_still_writes_derived_labels_where_they_belong(self, learned):
        """``write_labels`` still works, and still writes derived rows."""
        tdir = trajectory.trajectory_dir(learned)
        _write(os.path.join(tdir, "s1.jsonl"), [
            _task_record("s1", 1, "run the tests"),
            _task_record("s1", 2, "no, run the tests"),
        ])
        path = corpus.write_labels(learned)
        assert path and os.path.isfile(path)
        rows = corpus.read_labels(learned)
        assert len(rows) == 2
        assert rows[0]["signals"]["implicit"] == "bad"


# ── C: a human rating reaches the loops ─────────────────────────────────

class TestAHumanRatingReachesTheLoops:
    """The strongest evidence in the corpus must not be invisible.

    The bug: ``label_records`` deliberately leaves a rated record's
    ``signals.implicit`` alone - a click beats a heuristic - and
    ``contrast`` read only the heuristic field.  A thumbs-down was
    written, shown in the status report, and consumed by nothing.
    """

    def test_a_thumbs_down_is_on_the_bad_side_of_a_contrast(self, learned):
        record = _task_record("s1", 1, "a run the user rejected",
                              signals={"user_rating": -1})
        result = playbook.contrast([record])
        assert len(result["bad"]) == 1, \
            "a thumbs-down reached no side of the contrast"
        assert not result["good"]

    def test_a_thumbs_up_is_on_the_good_side_of_a_contrast(self, learned):
        record = _task_record("s1", 1, "a run the user accepted",
                              signals={"user_rating": 1})
        result = playbook.contrast([record])
        assert len(result["good"]) == 1
        assert not result["bad"]

    def test_a_click_outranks_a_clean_verdict(self, learned):
        """The user saw the answer; the fact-checker read it.

        When they disagree, the click wins - it is the one verdict
        the agent could not influence.
        """
        record = _task_record("s1", 1, "a run",
                              signals={"user_rating": -1, "verifier": "clean"})
        result = playbook.contrast([record])
        assert len(result["bad"]) == 1, \
            "a clean fact-check overruled a thumbs-down"

    def test_a_rating_reaches_the_recall_ranking(self, learned):
        """The recall path ranked on the verifier and the heuristic.

        It did not rank on the click, which is why a rejected answer
        could be recalled as a good example.
        """
        rejected = _task_record("s1", 1, "how do I list the local files",
                                signals={"user_rating": -1, "verifier": "clean"})
        assert recall._implicit_of(rejected) == "bad"

    def test_a_rating_is_not_overwritten_by_the_heuristic(self, learned):
        """A rated record keeps its rating, and gains no heuristic label."""
        record = _task_record("s1", 1, "a run the user rejected",
                              signals={"user_rating": -1})
        sig.label_records([record])
        signals = record["signals"]
        assert signals.get("user_rating") == -1
        assert signals.get("implicit_source") == "explicit-rating"
        assert signals.get("implicit") is None, \
            "a heuristic label was written over a click"

    def test_the_rating_verdict_helper_reads_either_spelling(self):
        assert sig.rating_verdict({"signals": {"user_rating": 1}}) == "good"
        assert sig.rating_verdict({"signals": {"user_rating": -1}}) == "bad"
        assert sig.rating_verdict({"signals": {"user_rating": None}}) == ""
        assert sig.rating_verdict({"signals": {}}) == ""
        assert sig.rating_verdict({}) == ""


# ── the session-end label is not a verdict ───────────────────────────────

class TestSilenceIsNotAcceptance:
    """"good" inferred from a session that ended is the weakest signal.

    It must not outrank a fact-check that found something.  Before this
    rule, a clean answer and a corrected one both ranked the same when
    the clean one's session simply ended, and the ranking the whole
    recall path is built on stopped discriminating.
    """

    def test_a_session_end_good_does_not_mask_a_verifier_verdict(self, learned):
        record = _task_record("s1", 1, "a run",
                              signals={"verifier": "issues"})
        sig.label_records([record])
        assert record["signals"]["implicit_source"] == "session-end"
        # The label is there, but it does not decide the band.
        assert recall._implicit_of(record) == "unknown"

    def test_a_follow_up_good_does_beat_a_verifier_verdict(self, learned):
        """An acceptance read off a message is the user saying something."""
        tdir = trajectory.trajectory_dir(learned)
        _write(os.path.join(tdir, "s1.jsonl"), [
            _task_record("s1", 1, "run the tests",
                         signals={"verifier": "issues"}),
            _task_record("s1", 2, "thanks, that worked"),
        ])
        records = list(trajectory.iter_records(learned))
        assert records[0]["signals"]["implicit_source"] == "follow-up"
        assert recall._implicit_of(records[0]) == "good"

    def test_a_clean_episode_outranks_a_corrected_one(self, learned):
        """The regression that the session-end rule exists to prevent."""
        tdir = trajectory.trajectory_dir(learned)
        _write(os.path.join(tdir, "good.jsonl"), [
            _task_record("good", 1, "deploy the service to staging",
                         signals={"verifier": "clean"}),
        ])
        _write(os.path.join(tdir, "bad.jsonl"), [
            _task_record("bad", 1, "deploy the service to staging",
                         signals={"verifier": "issues"}),
        ])
        episodes = recall.recall("deploy the service to staging",
                                  config_data=learned)
        outcomes = [e["outcome"] for e in episodes]
        assert outcomes.index("clean") < outcomes.index("issues"), \
            "the clean episode did not come first: %r" % outcomes
