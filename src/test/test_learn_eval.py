"""
The ruler: a frozen holdout, a dev split, and the boundary that keeps them honest.

What is being protected here, in order of how badly it would hurt if it broke:

1. **The holdout cannot be edited by the agent** (I7).  If the thing being
   measured can rewrite its own exam, every number it produces is a number
   about the rewriting, and the whole self-improvement track becomes
   self-certification.  This is the invariant most likely to rot silently,
   because a holdout that quietly moved still produces plausible numbers.
2. **A split is per model and per owner.**  The corpus holds twelve models
   and, in web mode, several tenants.  A split that mixes either is not a
   measurement of one agent.
3. **The split is deterministic.**  A split that changed on every run would
   make every comparison against it meaningless — and it would look fine.
4. **The holdout is only read by the gate** (I1).  A loop iterating on a
   candidate must read the dev split; if it could read the holdout, the
   holdout would stop being held out.
5. **The holdout is big enough to measure against.**  A three-item holdout
   has a stderr wide enough to promote anything, which makes the gate
   decorative.
"""

import json
import os

import pytest

from learn import eval as lev
from learn import trajectory as traj


# ─── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def learned(tmp_path):
    """A learned store under tmp_path, recording on."""
    return {"learn": {"autonomy": "observe", "path": str(tmp_path / "learned")}}


@pytest.fixture
def eval_dir(tmp_path, monkeypatch):
    """An eval tree outside the agent's writable surface.

    Under tmp_path but not under the learned store, so the I7 check has
    something to actually distinguish.
    """
    target = tmp_path / "ruler"
    monkeypatch.setenv("ONIT_EVAL_PATH", str(target))
    return target


def _record(learned, *, session="s1", turn=1, task="a question worth replaying",
            response="an answer worth comparing", model="m1", owner=None):
    return traj.record_task(session_id=session, turn=turn, task=task,
                            response=response, model=model, owner=owner,
                            config_data=learned)


def _seed(learned, n, *, model="m1", owner=None):
    """Record ``n`` distinct tasks, one per session.

    One per session so each has a unique id: the split's identity is
    session#turn, and two tasks sharing one would be indistinguishable.
    """
    for i in range(n):
        _record(learned, session=f"s{i}", turn=1,
                task=f"question number {i} about something worth asking",
                response=f"answer number {i} worth comparing", model=model,
                owner=owner)
    return n


# ─── I7: the boundary ───────────────────────────────────────────────────────

class TestOwnershipBoundary:
    """The ruler lives outside the agent's writable surface."""

    def test_the_default_eval_dir_is_outside_the_learned_store(self, learned):
        # No ONIT_EVAL_PATH: the default is the repository's benchmarks/eval.
        assert lev.is_outside_agent_surface(lev.eval_dir(learned), learned)

    def test_a_path_inside_the_learned_store_is_refused(self, learned):
        inside = os.path.join(learned["learn"]["path"], "eval")
        with pytest.raises(PermissionError) as excinfo:
            lev.assert_outside_agent_surface(inside, learned)
        assert "I7" in str(excinfo.value)

    def test_a_path_inside_data_path_is_refused(self, learned, tmp_path):
        cfg = dict(learned)
        cfg["data_path"] = str(tmp_path / "workspace")
        inside = os.path.join(cfg["data_path"], "eval")
        with pytest.raises(PermissionError):
            lev.assert_outside_agent_surface(inside, cfg)

    def test_a_path_that_resolves_into_the_root_is_caught(self, learned):
        # The check resolves the path, so a ".." that lands *inside* the
        # root is caught even though the string does not start with it.
        sneaky = os.path.join(learned["learn"]["path"], "sub", "..", "eval")
        with pytest.raises(PermissionError):
            lev.assert_outside_agent_surface(sneaky, learned)

    def test_a_path_that_resolves_out_of_the_root_is_allowed(self, learned):
        # The converse: "learned/../eval" really is outside the learned
        # store, so refusing it would be the check being wrong rather than
        # right.  A boundary that refuses everything is not a boundary.
        outside = os.path.join(learned["learn"]["path"], "..", "eval")
        assert lev.is_outside_agent_surface(outside, learned)

    def test_a_symlink_into_the_learned_store_is_refused(self, learned, tmp_path):
        # A symlink is the obvious way around a string-prefix check.
        outside = tmp_path / "outside"
        outside.mkdir()
        link = outside / "link"
        os.symlink(learned["learn"]["path"], link)
        with pytest.raises(PermissionError):
            lev.assert_outside_agent_surface(str(link), learned)

    def test_a_sibling_with_a_shared_prefix_is_allowed(self, learned, tmp_path):
        # "learned-evil" starts with "learned" but is not under it.  A naive
        # startswith() check would refuse a legitimate path.
        sibling = tmp_path / "learned-evil"
        sibling.mkdir()
        assert lev.is_outside_agent_surface(str(sibling), learned)

    def test_writing_the_split_inside_the_agent_surface_is_refused(self, learned):
        # The check has to fire on the write, not only on a helper.  The
        # eval path is pointed inside the learned store to make it so.
        cfg = dict(learned)
        cfg["learn"] = dict(learned["learn"])
        cfg["learn"]["eval"] = {"path": learned["learn"]["path"] + "/eval"}
        split = {"holdout": [{"id": "x"}], "dev": []}
        with pytest.raises(PermissionError):
            lev.write_split(split, cfg)

    def test_writing_the_baseline_inside_the_agent_surface_is_refused(self,
                                                                      learned):
        cfg = dict(learned)
        cfg["learn"] = dict(learned["learn"])
        cfg["learn"]["eval"] = {"path": learned["learn"]["path"] + "/eval"}
        with pytest.raises(PermissionError):
            lev.write_baseline({"model": "m1"}, cfg)

    def test_an_eval_path_override_is_honoured(self, learned, tmp_path,
                                               monkeypatch):
        target = tmp_path / "ruler"
        monkeypatch.setenv("ONIT_EVAL_PATH", str(target))
        assert lev.eval_dir(learned) == str(target)


# ─── drawing the split ──────────────────────────────────────────────────────

class TestBuildSplit:
    """A split drawn from the real corpus."""

    def test_a_split_needs_real_questions_and_answers(self, learned):
        # A task with no text, or no answer, measures the harness.
        traj.record_task(session_id="s1", turn=1, task="hi", response="",
                         config_data=learned)
        traj.record_task(session_id="s1", turn=2, task="", response="hello",
                         config_data=learned)
        split = lev.build_split(config_data=learned)
        assert split["counts"]["candidates"] == 0

    def test_a_usable_task_becomes_a_candidate(self, learned):
        _record(learned)
        split = lev.build_split(config_data=learned)
        assert split["counts"]["candidates"] == 1

    def test_the_two_splits_partition_the_candidates(self, learned):
        _seed(learned, 10)
        split = lev.build_split(config_data=learned, holdout_ratio=0.4)
        assert split["counts"]["holdout"] + split["counts"]["dev"] == \
            split["counts"]["candidates"]

    def test_the_ratio_is_respected(self, learned):
        _seed(learned, 10)
        split = lev.build_split(config_data=learned, holdout_ratio=0.3)
        assert split["counts"]["holdout"] == 3
        assert split["counts"]["dev"] == 7

    def test_a_nonsense_ratio_is_refused(self, learned):
        with pytest.raises(ValueError):
            lev.build_split(config_data=learned, holdout_ratio=1.5)
        with pytest.raises(ValueError):
            lev.build_split(config_data=learned, holdout_ratio=0.0)

    def test_the_split_is_deterministic(self, learned):
        _seed(learned, 12)
        first = lev.build_split(config_data=learned)
        second = lev.build_split(config_data=learned)
        assert [i["id"] for i in first["holdout"]] == \
            [i["id"] for i in second["holdout"]]
        assert [i["id"] for i in first["dev"]] == \
            [i["id"] for i in second["dev"]]

    def test_no_item_appears_in_both_splits(self, learned):
        _seed(learned, 12)
        split = lev.build_split(config_data=learned)
        holdout_ids = {i["id"] for i in split["holdout"]}
        dev_ids = {i["id"] for i in split["dev"]}
        assert not (holdout_ids & dev_ids)

    def test_a_model_filter_restricts_the_draw(self, learned):
        _record(learned, session="a", model="m1")
        _record(learned, session="b", model="m2")
        split = lev.build_split(config_data=learned, model="m1")
        assert split["counts"]["candidates"] == 1
        drawn = split["holdout"] + split["dev"]
        assert [i["model"] for i in drawn] == ["m1"]

    def test_an_owner_filter_restricts_the_draw(self, learned):
        _record(learned, session="a", owner="alice@example.com")
        _record(learned, session="b", owner="bob@example.com")
        split = lev.build_split(config_data=learned,
                                owner=traj.owner_hash("alice@example.com"))
        assert split["counts"]["candidates"] == 1
        drawn = split["holdout"] + split["dev"]
        assert [i["owner"] for i in drawn] == \
            [traj.owner_hash("alice@example.com")]

    def test_a_tenant_owned_task_never_enters_a_shared_split(self, learned):
        # The default draw is owner-less only.  A task belonging to a tenant
        # must not be replayed into a measurement that gets shared.
        _record(learned, session="a", owner="alice@example.com")
        _record(learned, session="b", owner=None)
        split = lev.build_split(config_data=learned)
        assert split["counts"]["candidates"] == 1
        drawn = split["holdout"] + split["dev"]
        assert [i["owner"] for i in drawn] == [None]

    def test_the_max_items_cap_is_respected(self, learned):
        _seed(learned, 30)
        split = lev.build_split(config_data=learned, max_items=10)
        assert split["counts"]["candidates"] == 10
        assert len(split["holdout"]) + len(split["dev"]) == 10

    def test_a_split_entry_carries_what_a_replay_needs(self, learned):
        _record(learned, model="m1", task="a question worth replaying here",
                response="an answer worth comparing here")
        split = lev.build_split(config_data=learned)
        entry = (split["holdout"] + split["dev"])[0]
        assert entry["task"] == "a question worth replaying here"
        assert entry["response"] == "an answer worth comparing here"
        assert entry["model"] == "m1"
        assert entry["session_id"] == "s1"
        assert entry["turn"] == 1
        assert "tools_available" in entry


# ─── freezing and reading ───────────────────────────────────────────────────

class TestFreezeAndRead:
    """What gets written, and what comes back."""

    def test_a_frozen_split_round_trips(self, learned, eval_dir):
        _seed(learned, 6)
        split = lev.build_split(config_data=learned)
        written = lev.write_split(split, learned)
        assert os.path.isfile(written["holdout"])
        assert os.path.isfile(written["dev"])
        assert os.path.isfile(written["split"])

        holdout = lev.read_holdout(learned)
        assert len(holdout) == split["counts"]["holdout"]
        assert holdout[0]["task"] == split["holdout"][0]["task"]

    def test_the_holdout_and_the_dev_split_are_separate_files(self, learned,
                                                               eval_dir):
        # A process handed the holdout must not also receive the dev split.
        _seed(learned, 6)
        lev.write_split(lev.build_split(config_data=learned), learned)
        holdout_ids = {i["id"] for i in lev.read_holdout(learned)}
        dev_ids = {i["id"] for i in lev.read_dev(learned)}
        assert holdout_ids and dev_ids
        assert not (holdout_ids & dev_ids)

    def test_reading_a_missing_holdout_is_empty_not_an_error(self, learned,
                                                              eval_dir):
        assert lev.read_holdout(learned) == []
        assert lev.read_dev(learned) == []

    def test_a_torn_line_costs_one_item_not_the_file(self, learned, eval_dir):
        _seed(learned, 4)
        lev.write_split(lev.build_split(config_data=learned), learned)
        path = lev.holdout_path(learned)
        with open(path, "a", encoding="utf-8") as f:
            f.write("{not json\n")
        # The good items still read.
        assert len(lev.read_holdout(learned)) >= 1

    def test_a_split_with_no_holdout_and_no_dev_reads_as_empty(self, learned,
                                                               eval_dir):
        assert lev.read_split(learned)["holdout"] == []
        assert lev.read_split(learned)["dev"] == []

    def test_the_baseline_round_trips(self, learned, eval_dir):
        payload = {"model": "m1", "learn": "off", "score": 0.94}
        lev.write_baseline(payload, learned)
        assert lev.read_baseline(learned)["score"] == 0.94

    def test_reading_a_missing_baseline_is_empty(self, learned, eval_dir):
        assert lev.read_baseline(learned) == {}


# ─── the gate ───────────────────────────────────────────────────────────────

class TestGate:
    """What the promotion gate is allowed to see."""

    def test_a_small_holdout_is_not_sealed(self, learned, eval_dir):
        _seed(learned, 5)
        lev.write_split(lev.build_split(config_data=learned), learned)
        assert not lev.holdout_is_sealed(learned)

    def test_a_holdout_of_enough_items_is_sealed(self, learned, eval_dir):
        # Enough items that the 40% split clears the minimum on its own.
        _seed(learned, lev.MIN_HOLDOUT_ITEMS * 3)
        lev.write_split(lev.build_split(config_data=learned), learned)
        assert lev.holdout_is_sealed(learned)

    def test_the_gate_refuses_to_run_without_a_holdout(self, learned,
                                                       eval_dir):
        with pytest.raises(FileNotFoundError) as excinfo:
            lev.assert_readable_by_gate(learned)
        assert "holdout" in str(excinfo.value).lower()

    def test_the_gate_gets_the_holdout_when_there_is_one(self, learned,
                                                          eval_dir):
        _seed(learned, lev.MIN_HOLDOUT_ITEMS * 3)
        lev.write_split(lev.build_split(config_data=learned), learned)
        assert len(lev.assert_readable_by_gate(learned)) >= \
            lev.MIN_HOLDOUT_ITEMS

    def test_the_dev_split_is_readable_without_the_holdout(self, learned,
                                                            eval_dir):
        # I1: iteration is on dev.  Dev must be readable on its own, because
        # a loop iterating on a candidate never touches the holdout.
        _seed(learned, 6)
        lev.write_split(lev.build_split(config_data=learned), learned)
        assert len(lev.read_dev(learned)) > 0


# ─── the summary ────────────────────────────────────────────────────────────

class TestSummary:
    """What `onit learn --eval` reports."""

    def test_an_empty_ruler_reports_nothing_honestly(self, learned, eval_dir):
        summary = lev.summarize(learned)
        assert summary["holdout"] == 0
        assert summary["dev"] == 0
        assert summary["holdout_sealed"] is False
        assert summary["outside_agent_surface"] is True

    def test_the_summary_reports_the_split(self, learned, eval_dir):
        _seed(learned, 6)
        lev.write_split(lev.build_split(config_data=learned), learned)
        summary = lev.summarize(learned)
        assert summary["holdout"] + summary["dev"] == 6
        assert summary["eval_dir"] == str(eval_dir)

    def test_the_summary_flags_a_ruler_inside_the_agent_surface(self, learned):
        cfg = dict(learned)
        cfg["learn"] = dict(learned["learn"])
        cfg["learn"]["eval"] = {"path": learned["learn"]["path"] + "/eval"}
        summary = lev.summarize(cfg)
        assert summary["outside_agent_surface"] is False
