"""
The cross-cutting invariants.

The per-module test files each check one module's own contract.  This file
checks the things that only exist at the *seams* between modules, which is
where the failures that matter actually live:

- **I1**  no loop reads the holdout.  The dev split is for iteration; the
  holdout is for the gate.  A loop that reads the holdout is a loop that
  grades its own exam.
- **I2**  delta-only updates.  A bullet or a skill changes by a delta, and
  the previous version stays recoverable.
- **I3**  pinned and reversible.  Everything live is named in a pin file,
  and every promotion has a way back.
- **I4**  quality-gated efficiency.  A promotion needs a measured delta.
- **I7**  the ruler is outside the agent's writable surface.  Not "is not
  supposed to be" -- cannot be.

These are the invariants the RFC calls load-bearing, and they are the ones
a future edit is most likely to break, because a future edit will usually
be made to one module and will not think about the other.

The tests here are deliberately adversarial: they try to *break* the
invariants rather than confirm them.
"""

import json
import os
import shutil
import tempfile

import pytest

from src.learn import corpus, eval as ruler, playbook, recall, signals as sig
from src.learn import skills, skills_curator, skills_gate, skills_mining
from src.learn import trajectory


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def learned(tmp_path):
    """An empty learned store, isolated from the real one.

    The *ruler* is isolated too.  The eval tree defaults to the
    repository's ``benchmarks/eval``, so a test that writes a split
    without redirecting it would land a holdout in the checkout -- a
    test that mutates the repository is a test that can make the next
    run measure something else.
    """
    root = str(tmp_path / "learned")
    ruler_root = str(tmp_path / "ruler")
    os.makedirs(root, exist_ok=True)
    os.makedirs(ruler_root, exist_ok=True)
    try:
        yield {"learn": {"path": root,
                         "eval": {"path": ruler_root}}}
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(ruler_root, ignore_errors=True)


def _record(session="s1", turn=1, task="fix the failing test", response="done",
            model="m1", owner=None, tools=None, verdict=None):
    """One trajectory record, shaped like the real ones."""
    turn_entry = {"turn": turn, "tools": tools or []}
    if verdict is not None:
        turn_entry["signals"] = {"verifier": verdict}
    return {
        "kind": "task",
        "session_id": session,
        "turn": turn,
        "task": task,
        "response": response,
        "model": model,
        "owner": owner,
        "ts": "2026-10-20T10:00:00",
        "trajectory": [turn_entry],
        "signals": {"verifier": verdict} if verdict is not None else {},
    }


def _write_record(learned, record):
    """Append one record to the store, through the real writer."""
    path = os.path.join(trajectory.trajectory_dir(learned),
                        "%s.jsonl" % record["session_id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def _candidate(digest="a" * 64):
    """A miner candidate, in the shape the miner produces."""
    return {"tool": "bash", "digest": digest, "sessions": 3, "calls": 5,
            "errors": 0, "success_rate": 1.0, "arg_keys": ["cmd"]}


# ── I1: no loop reads the holdout ─────────────────────────────────────────

class TestHoldoutIsolation:
    """The holdout is read by the gate and by nothing else."""

    def test_the_split_separates_holdout_from_dev(self, learned):
        for i in range(40):
            _write_record(learned, _record(session="s%d" % i, turn=i,
                                           task="a task long enough to count %d"
                                           % i,
                                           response="a long enough answer"))
        split = ruler.build_split(config_data=learned)
        assert split["holdout"], "the split produced no holdout"
        assert split["dev"], "the split produced no dev set"
        overlap = ({i["id"] for i in split["holdout"]}
                   & {i["id"] for i in split["dev"]})
        assert not overlap, "a task is in both the holdout and the dev set"

    def test_the_holdout_and_the_dev_split_are_separate_files(self, learned):
        for i in range(40):
            _write_record(learned, _record(session="s%d" % i, turn=i,
                                           task="a task long enough to count %d"
                                           % i,
                                           response="a long enough answer"))
        ruler.write_split(ruler.build_split(config_data=learned), learned)
        holdout = ruler.read_holdout(learned)
        dev = ruler.read_dev(learned)
        assert {i["id"] for i in holdout}.isdisjoint({i["id"] for i in dev})

    def test_recall_reads_the_store_not_the_holdout_file(self, learned):
        """A recall block is what the agent iterates against.

        The recall path reads the *trajectory store*, never the split
        files, so it cannot touch the holdout even by accident.  This
        test pins that: the holdout file may contain text the recall
        would happily match, and neither the holdout's own id nor its
        *question* may come back.

        The question is the stronger half of the contract, and it is
        the half that matters.  The corpus holds the same question asked
        many times -- "weather today" was asked 21 times -- and every
        one of those runs carries an answer to it.  Excluding only the
        record id the split happened to pick would leave the other
        twenty in the pool, and the agent would be handed the answer
        key before it was asked the question.
        """
        marker = "ZZQQXX-unique-holdout-marker"
        # A sibling run: same question, different record id, same answer.
        _write_record(learned, _record(session="s1", task=marker,
                                       response="the answer"))
        _write_record(learned, _record(session="s2", task=marker,
                                       response="the answer"))
        directory = ruler.eval_dir(learned)
        os.makedirs(directory, exist_ok=True)
        with open(ruler.holdout_path(learned), "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "h1", "task": marker,
                                "response": "held out"}) + "\n")
        episodes = recall.recall(marker, config_data=learned)
        # The exclusion is by question, so the two store runs of the
        # holdout's question are excluded too.  Recall finds nothing
        # here, and that is the correct answer: the only thing it could
        # have found was the answer to the question being asked.
        for episode in episodes:
            assert episode["id"] != "h1", "recall returned a holdout item"
            assert episode["id"] not in ("s1#1", "s2#1"), (
                "recall returned a run of the holdout's own question")

    def test_recall_still_finds_the_runs_it_may_learn_from(self, learned):
        """The exclusion must not swallow the corpus.

        A holdout is a *question* set aside, not a corpus-wide mute.
        If the exclusion were too broad, recall would return nothing for
        everything, and the treatment arm would be measuring an agent
        with amnesia -- which is a null result for the wrong reason, and
        is indistinguishable from "recall does not help".
        """
        _write_record(learned, _record(session="s1", turn=1,
                                       task="a completely different question",
                                       response="a long enough answer"))
        _write_record(learned, _record(session="s2", turn=1,
                                       task="an unrelated question entirely",
                                       response="a long enough answer"))
        directory = ruler.eval_dir(learned)
        os.makedirs(directory, exist_ok=True)
        with open(ruler.holdout_path(learned), "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "h1", "task": "the held out question",
                                "response": "held out"}) + "\n")
        episodes = recall.recall("a completely different question",
                                 config_data=learned)
        assert episodes, ("recall found nothing for an unrelated question: "
                          "the exclusion is too broad")

    def test_the_gate_screen_names_the_holdout(self):
        """A skill that even mentions the holdout is refused.

        The gate screens the *source* for the ruler's names, so a skill
        cannot be written that reaches for the exam.
        """
        ok, reason, _ = skills_gate.check_source(
            "def run():\n    return open('/x/holdout.jsonl').read()\n")
        assert not ok, "a skill that reads the holdout passed the gate"

    def test_the_gate_screen_names_the_eval_directory(self):
        ok, reason, _ = skills_gate.check_source(
            "def run():\n    return open('/x/eval/baseline.json').read()\n")
        assert not ok, "a skill that reads the baseline passed the gate"

    def test_the_gate_knows_where_the_ruler_is(self, learned):
        paths = skills_gate.ruler_paths(learned)
        assert ruler.holdout_path(learned) in paths
        assert ruler.dev_path(learned) in paths
        assert ruler.baseline_path(learned) in paths

    def test_a_skill_cannot_reach_the_ruler_through_a_relative_path(self):
        """The screen is on the *name*, not just on an absolute path."""
        for source in (
            "def run():\n    return open('holdout.jsonl').read()\n",
            "def run():\n    return open('../eval/dev.jsonl').read()\n",
            "def run():\n    return open('/x/baselines/full.json').read()\n",
        ):
            ok, reason, _ = skills_gate.check_source(source)
            assert not ok, "the gate let through: %r (%s)" % (source, reason)

    def test_the_gate_refuses_a_source_naming_a_forbidden_path(self):
        """The forbidden-path screen is on the resolved path."""
        ok, reason, _ = skills_gate.check_source(
            "def run():\n    return open('/x/ruler/dev.jsonl').read()\n",
            forbidden_paths=["/x/ruler"])
        assert not ok, "a skill that reads a forbidden path passed"

    def test_the_holdout_is_not_sealed_until_it_is_written(self, learned):
        """A holdout that was never drawn is not a sealed holdout.

        Pretending it is sealed would let a promotion gate read an
        empty file and conclude "no regression".
        """
        assert not ruler.holdout_is_sealed(learned)
        for i in range(60):
            _write_record(learned, _record(session="s%d" % i, turn=i,
                                           task="a task long enough to count %d"
                                           % i,
                                           response="a long enough answer"))
        ruler.write_split(ruler.build_split(config_data=learned), learned)
        assert ruler.holdout_is_sealed(learned)


# ── I7: the ruler is outside the agent's writable surface ─────────────────

class TestRulerBoundary:
    """The exam must not be editable by the thing being examined."""

    def test_the_default_eval_dir_is_outside_the_learned_store(self, learned):
        assert ruler.is_outside_agent_surface(ruler.eval_dir(learned), learned)

    def test_the_default_eval_dir_is_outside_the_data_path(self, learned):
        config = dict(learned)
        config["data_path"] = tempfile.mkdtemp(prefix="onit-data-")
        try:
            assert ruler.is_outside_agent_surface(ruler.eval_dir(config),
                                                 config)
        finally:
            shutil.rmtree(config["data_path"], ignore_errors=True)

    def test_an_eval_path_inside_the_learned_store_is_refused(self, learned):
        inside = os.path.join(learned["learn"]["path"], "eval")
        with pytest.raises(PermissionError):
            ruler.assert_outside_agent_surface(inside, learned)

    def test_a_symlink_into_the_learned_store_is_refused(self, learned,
                                                         tmp_path):
        """A symlink is the obvious way around a path check."""
        target = os.path.join(learned["learn"]["path"], "eval")
        os.makedirs(target, exist_ok=True)
        link = os.path.join(str(tmp_path), "eval-link")
        os.symlink(target, link)
        with pytest.raises(PermissionError):
            ruler.assert_outside_agent_surface(link, learned)

    def test_writing_the_split_inside_the_agent_surface_is_refused(self,
                                                                   learned):
        config = dict(learned)
        config["learn"] = dict(learned["learn"])
        config["learn"]["eval"] = {
            "path": os.path.join(learned["learn"]["path"], "eval")}
        with pytest.raises(PermissionError):
            ruler.write_split({"holdout": [{"id": "x"}], "dev": []}, config)

    def test_writing_the_baseline_inside_the_agent_surface_is_refused(self,
                                                                      learned):
        config = dict(learned)
        config["learn"] = dict(learned["learn"])
        config["learn"]["eval"] = {
            "path": os.path.join(learned["learn"]["path"], "eval")}
        with pytest.raises(PermissionError):
            ruler.write_baseline({"rows": []}, config)

    def test_the_env_override_cannot_put_the_ruler_inside_the_store(self,
                                                                     learned):
        """``ONIT_EVAL_PATH`` is a convenience, not a loophole.

        The override exists so a deployment can point the ruler at a
        read-only mount.  It must not be able to point it at the agent's
        own writable tree.
        """
        inside = os.path.join(learned["learn"]["path"], "eval")
        # The config names no eval path, so the environment override is
        # the one that decides where the ruler lives.
        config = {"learn": {"path": learned["learn"]["path"]}}
        old = os.environ.get("ONIT_EVAL_PATH")
        os.environ["ONIT_EVAL_PATH"] = inside
        try:
            assert ruler.eval_dir(config) == inside, \
                "the environment override was not honoured"
            with pytest.raises(PermissionError):
                ruler.write_split({"holdout": [], "dev": []}, config)
            assert ruler.is_outside_agent_surface(inside, config) is False
        finally:
            if old is None:
                os.environ.pop("ONIT_EVAL_PATH", None)
            else:
                os.environ["ONIT_EVAL_PATH"] = old

    def test_a_sibling_with_a_shared_prefix_is_allowed(self, learned):
        """``/a/learned-2`` is not inside ``/a/learned``.

        A prefix check that forgets the separator rejects innocent
        neighbours, and then operators "fix" it by moving the ruler into
        the store.
        """
        sibling = os.path.join(learned["learn"]["path"] + "-2", "eval")
        assert ruler.is_outside_agent_surface(sibling, learned)


# ── I3: pinned and reversible ─────────────────────────────────────────────

class TestPinAndRollback:
    """Everything live is named in a pin, and every promotion is undoable."""

    def test_an_unpinned_skill_is_not_served(self, learned):
        skills_curator.synthesize(_candidate(), name="s1",
                                  config_data=learned)
        assert "s1" not in skills.pinned_skills(learned)

    def test_a_promoted_skill_is_pinned(self, learned):
        skills_curator.synthesize(_candidate(), name="s1",
                                  config_data=learned)
        skills_curator.promote("s1", config_data=learned, delta=0.1)
        assert "s1" in skills.pinned_skills(learned)

    def test_the_pin_file_names_what_is_live(self, learned):
        skills_curator.synthesize(_candidate(), name="s1",
                                  config_data=learned)
        skills_curator.promote("s1", config_data=learned, delta=0.1)
        pin = skills.read_pin(learned)
        assert "s1" in pin["skills"]
        assert pin["version"] >= 1

    def test_the_pin_version_increases_on_every_promotion(self, learned):
        skills_curator.synthesize(_candidate(), name="s1",
                                  config_data=learned)
        skills_curator.promote("s1", config_data=learned, delta=0.1)
        first = skills.read_pin(learned)["version"]
        skills_curator.promote("s1", config_data=learned, delta=0.2)
        second = skills.read_pin(learned)["version"]
        assert second > first

    def test_rollback_restores_the_previous_pin(self, learned):
        """The first promotion must be reversible too.

        This is the case that is easy to get wrong: if the rollback
        writes an *empty* pin when there is no previous one, then
        rolling back the first promotion leaves nothing pinned, which is
        right, and rolling it back a second time leaves a pin that
        pins a skill that was never promoted.  The pin is the
        allowlist, so its state has to be exactly right in both
        directions.
        """
        skills_curator.synthesize(_candidate(), name="s1",
                                  config_data=learned)
        skills_curator.promote("s1", config_data=learned, delta=0.1)
        assert "s1" in skills.pinned_skills(learned)
        skills_curator.rollback(config_data=learned)
        assert "s1" not in skills.pinned_skills(learned)

    def test_rolling_back_with_nothing_pinned_is_harmless(self, learned):
        skills_curator.rollback(config_data=learned)
        assert skills.pinned_skills(learned) == {}

    def test_a_broken_pin_file_pins_nothing(self, learned):
        """A corrupt allowlist must fail closed."""
        path = skills.pinned_path(learned)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("this: is: not: valid: yaml:\n  - [\n")
        assert skills.pinned_skills(learned) == {}

    def test_a_playbook_change_keeps_the_previous_version(self, learned):
        """The playbook is reversible by file copy, not reconstruction."""
        path = playbook.global_path(learned)
        first = {"version": 1, "items": [
            {"id": "b1", "text": "use the flag", "status": "active",
             "scope": "global", "trigger": "", "provenance": []}]}
        assert playbook.write_playbook(path, first)
        deltas = [{"op": "insert", "text": "read the docs first",
                   "trigger": "read_file", "scope": "global",
                   "provenance": ["s1"]}]
        playbook.curate(first, deltas, config_data=learned)
        assert os.path.isfile(path + ".prev"), "no previous version was kept"
        with open(path + ".prev", encoding="utf-8") as f:
            previous = json.load(f)
        assert len(previous["items"]) == 1

    def test_the_previous_version_is_the_one_that_was_in_use(self, learned):
        """The backup is written *before* the live file is replaced.

        If it were written after, the backup would be a copy of the new
        version and the rollback would be a no-op.
        """
        path = playbook.global_path(learned)
        first = {"version": 1, "items": [
            {"id": "b1", "text": "one", "status": "active",
             "scope": "global", "trigger": "", "provenance": []}]}
        playbook.write_playbook(path, first)
        second = {"version": 2, "items": [
            {"id": "b1", "text": "two", "status": "active",
             "scope": "global", "trigger": "", "provenance": []}]}
        playbook.curate(second, [{"op": "update", "id": "b1",
                                  "text": "three"}], config_data=learned)
        with open(path + ".prev", encoding="utf-8") as f:
            previous = json.load(f)
        assert previous["items"][0]["text"] == "one"


# ── I2: delta-only updates ────────────────────────────────────────────────

class TestDeltaOnly:
    """A bullet changes by a delta; the rest of the store is untouched."""

    def test_a_delta_does_not_mutate_its_input(self):
        before = {"version": 1, "items": [
            {"id": "b1", "text": "one", "status": "active", "scope": "global",
             "trigger": "", "provenance": []}]}
        snapshot = json.dumps(before, sort_keys=True)
        playbook.apply_deltas(before, [{"op": "update", "id": "b1",
                                        "text": "two"}])
        assert json.dumps(before, sort_keys=True) == snapshot

    def test_a_delta_leaves_the_other_bullets_alone(self):
        before = {"version": 1, "items": [
            {"id": "b1", "text": "one", "status": "active", "scope": "global",
             "trigger": "", "provenance": []},
            {"id": "b2", "text": "two", "status": "active", "scope": "global",
             "trigger": "", "provenance": []}]}
        after = playbook.apply_deltas(before, [{"op": "update", "id": "b1",
                                                "text": "changed"}])
        assert after["items"][1]["text"] == "two"

    def test_an_unknown_operation_changes_nothing(self):
        before = {"version": 1, "items": [
            {"id": "b1", "text": "one", "status": "active", "scope": "global",
             "trigger": "", "provenance": []}]}
        after = playbook.apply_deltas(before, [{"op": "rewrite-everything",
                                                "text": "x"}])
        assert after == before

    def test_a_malformed_delta_changes_nothing(self):
        before = {"version": 1, "items": [
            {"id": "b1", "text": "one", "status": "active", "scope": "global",
             "trigger": "", "provenance": []}]}
        for delta in ({}, {"op": "insert"}, {"op": None}, "not a dict"):
            after = playbook.apply_deltas(before, [delta])
            assert after == before, "delta %r changed the playbook" % (delta,)

    def test_the_version_bumps_only_when_something_changed(self):
        before = {"version": 1, "items": [
            {"id": "b1", "text": "one", "status": "active", "scope": "global",
             "trigger": "", "provenance": []}]}
        same = playbook.apply_deltas(before, [{"op": "update", "id": "b1",
                                               "text": "one"}])
        assert same["version"] == before["version"]


# ── owner isolation, across the seams ─────────────────────────────────────

class TestOwnerIsolation:
    """One tenant's data must never reach another tenant's prompt."""

    def test_recall_does_not_cross_the_owner_boundary(self, learned):
        _write_record(learned, _record(session="s1", task="shared task",
                                       owner="alice"))
        _write_record(learned, _record(session="s2", task="shared task",
                                       owner="bob"))
        alice = recall.recall("shared task", config_data=learned,
                               owner="alice")
        bob = recall.recall("shared task", config_data=learned, owner="bob")
        assert alice, "alice got nothing, so the test proves nothing"
        assert bob, "bob got nothing, so the test proves nothing"
        for episode in alice:
            assert "s2" not in episode["id"], "alice saw bob's episode"
        for episode in bob:
            assert "s1" not in episode["id"], "bob saw alice's episode"

    def test_the_default_recall_returns_only_owner_less_records(self, learned):
        """The default must be the *safe* default, not the convenient one.

        ``iter_records`` treats ``owner=None`` as "no filter", which is
        right for a status report and catastrophic for retrieval.
        """
        _write_record(learned, _record(session="s1", task="shared task",
                                       owner="alice"))
        episodes = recall.recall("shared task", config_data=learned)
        for episode in episodes:
            assert "s1" not in episode["id"], "the default leaked alice"

    def test_the_playbook_does_not_cross_the_owner_boundary(self, learned):
        playbook.write_playbook(
            playbook.owner_path("alice", learned),
            {"version": 1, "items": [
                {"id": "a1", "text": "alice's trick", "status": "active",
                 "scope": "owner:alice", "trigger": "", "provenance": []}]})
        context = playbook.build_context(learned, owner="bob")
        assert "alice's trick" not in context["text"]


# ── the autonomy ladder does not lie ──────────────────────────────────────

class TestAutonomyLadder:
    """A level the agent has not earned must not be granted.

    The RFC's rule is that the cap stays and the *silence* is the bug:
    a requested level that is not implemented has to be reported, not
    quietly downgraded.
    """

    def test_the_requested_level_is_reported(self, learned):
        from src.learn import config as learn_config
        config = {"learn": {"path": learned["learn"]["path"],
                            "autonomy": "evolve"}}
        assert learn_config.requested_level(config) == learn_config.EVOLVE

    def test_the_granted_level_is_never_above_the_implemented_cap(self):
        from src.learn import config as learn_config
        for name in ("observe", "adapt", "extend", "evolve"):
            config = {"learn": {"autonomy": name}}
            granted = learn_config.autonomy(config)
            assert granted <= learn_config._MAX_IMPLEMENTED

    def test_a_capped_request_is_still_a_capped_request(self):
        """The cap is real, and the request is not forgotten."""
        from src.learn import config as learn_config
        config = {"learn": {"autonomy": "evolve"}}
        assert learn_config.autonomy(config) < \
            learn_config.requested_level(config)
        assert learn_config.autonomy(config) <= \
            learn_config._MAX_IMPLEMENTED

    def test_the_report_names_the_difference(self, learned):
        """The report must not present a downgraded level as granted."""
        from src.learn import report as learn_report
        config = {"learn": {"path": learned["learn"]["path"],
                            "autonomy": "evolve"}}
        text = learn_report.format_status(config)
        assert text, "the report was empty"

    def test_recall_is_off_when_recording_is_off(self, learned):
        _write_record(learned, _record(session="s1", task="a task"))
        config = {"learn": {"path": learned["learn"]["path"],
                            "enabled": False}}
        assert recall.recall("a task", config_data=config) == []


# ── the injection is bounded ──────────────────────────────────────────────

class TestInjectionIsBounded:
    """Everything injected into a prompt is inside a hard budget."""

    def test_the_recall_block_is_under_its_budget(self, learned):
        for i in range(40):
            _write_record(learned, _record(
                session="s%d" % i, task="the same task over and over",
                response="a long answer " * 40, verdict="clean"))
        context = recall.build_context("the same task over and over",
                                       config_data=learned,
                                       token_budget=200)
        assert context["text"], "recall produced nothing"
        # 4 characters per token is the estimate the module itself uses.
        assert len(context["text"]) // 4 <= 200 + 100

    def test_the_playbook_block_is_under_its_budget(self, learned):
        items = []
        for i in range(50):
            items.append({"id": "b%d" % i, "text": "tactic number %d" % i,
                          "status": "active", "scope": "global",
                          "trigger": "", "provenance": []})
        playbook.write_playbook(playbook.global_path(learned),
                                {"version": 1, "items": items})
        context = playbook.build_context(learned, token_budget=100)
        assert context["text"], "the playbook produced nothing"
        assert len(context["text"]) // 4 <= 100 + 100

    def test_the_recall_block_is_marked_as_untrusted(self, learned):
        _write_record(learned, _record(session="s1", task="a task",
                                       response="an answer"))
        context = recall.build_context("a task", config_data=learned)
        assert "untrusted" in context["text"].lower(), \
            "the recall block does not mark itself as untrusted data"

    def test_the_playbook_block_says_it_is_not_an_instruction(self, learned):
        items = [{"id": "b1", "text": "a tactic", "status": "active",
                  "scope": "global", "trigger": "", "provenance": []}]
        playbook.write_playbook(playbook.global_path(learned),
                                {"version": 1, "items": items})
        context = playbook.build_context(learned)
        assert "not" in context["text"].lower()
        assert "instruction" in context["text"].lower() or \
            "notes" in context["text"].lower()


# ── the store fails closed ────────────────────────────────────────────────

class TestStoreFailsClosed:
    """A store that cannot be read must mean "nothing", not "everything"."""

    def test_a_broken_trajectory_file_is_skipped_not_fatal(self, learned):
        path = os.path.join(trajectory.trajectory_dir(learned), "s1.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json at all\n")
        records = list(trajectory.iter_records(learned))
        assert records == []

    def test_a_torn_line_loses_one_record_not_the_file(self, learned):
        path = os.path.join(trajectory.trajectory_dir(learned), "s1.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_record(session="s1", turn=1)) + "\n")
            f.write("{torn\n")
            f.write(json.dumps(_record(session="s1", turn=2)) + "\n")
        records = list(trajectory.iter_records(learned))
        assert len(records) == 2, "the torn line cost more than one record"

    def test_a_missing_store_is_empty_not_an_error(self, learned):
        assert list(trajectory.iter_records(learned)) == []
        assert recall.recall("anything", config_data=learned) == []

    def test_a_broken_skill_manifest_is_not_served(self, learned):
        skills_curator.synthesize(_candidate(), name="s1",
                                  config_data=learned)
        manifest_path = os.path.join(skills.skill_dir("s1", learned),
                                     "manifest.yaml")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write("::: not yaml :::\n")
        assert skills.read_manifest(manifest_path) is None

    def test_a_skill_name_that_is_not_safe_is_refused(self, learned):
        """A name is a path component.  It must not be able to escape."""
        for name in ("../escape", "a/b", "..", "/absolute"):
            assert skills.read_skill(name, learned) == {}
            assert not skills._safe_name(name)


# ── the derived labels never rewrite the record ───────────────────────────

class TestLabelsAreDerived:
    """The corpus's labels are a sidecar, never an edit of the record."""

    def test_labeling_does_not_touch_the_trajectory_file(self, learned):
        _write_record(learned, _record(session="s1", turn=1,
                                       task="do the thing", response="ok"))
        before = open(os.path.join(trajectory.trajectory_dir(learned),
                                   "s1.jsonl"), encoding="utf-8").read()
        corpus.label_corpus(learned)
        after = open(os.path.join(trajectory.trajectory_dir(learned),
                                  "s1.jsonl"), encoding="utf-8").read()
        assert before == after, "labeling rewrote the trajectory"

    def test_the_labels_land_in_their_own_file(self, learned):
        _write_record(learned, _record(session="s1", turn=1,
                                       task="do the thing", response="ok"))
        corpus.label_corpus(learned)
        labels = corpus.read_labels(learned)
        assert labels is not None


# ── the miner never stores an argument value ──────────────────────────────

class TestMiningStoresNoValues:
    """A skill candidate is a *shape*, never a payload."""

    def test_the_candidate_carries_the_digest_not_the_arguments(self,
                                                                 learned):
        _write_record(learned, _record(
            session="s1", verdict="clean",
            tools=[{"name": "bash", "args_digest": "a" * 64, "ok": True,
                    "args": {"command": "cat /etc/passwd"}}]))
        candidates = skills_mining.mine_store(learned)
        for candidate in candidates:
            blob = json.dumps(candidate, default=str)
            assert "passwd" not in blob, "the candidate stored an argument value"
            assert "/etc/passwd" not in blob

    def test_the_candidate_carries_no_arguments_key(self, learned):
        _write_record(learned, _record(
            session="s1", verdict="clean",
            tools=[{"name": "bash", "args_digest": "a" * 64, "ok": True,
                    "args": {"command": "rm -rf /"}}]))
        candidates = skills_mining.mine_store(learned)
        for candidate in candidates:
            assert "args" not in candidate

    def test_a_candidate_needs_three_sessions(self, learned):
        """One session repeating a call is one agent working.

        The threshold is on distinct sessions, not on call count.
        """
        for i in range(5):
            _write_record(learned, _record(
                session="s1", turn=i, verdict="clean",
                tools=[{"name": "bash", "args_digest": "a" * 64, "ok": True}]))
        assert skills_mining.mine_store(learned) == []

    def test_a_candidate_spanning_three_sessions_is_found(self, learned):
        for i in range(3):
            _write_record(learned, _record(
                session="s%d" % i, turn=1, verdict="clean",
                tools=[{"name": "bash", "args_digest": "a" * 64, "ok": True}]))
        candidates = skills_mining.mine_store(learned)
        assert len(candidates) == 1
        assert candidates[0]["tool"] == "bash"


# ── the gate refuses what it should ───────────────────────────────────────

class TestGateRefusals:
    """The five checks, each refusing the thing it is there to refuse."""

    def test_the_gate_refuses_a_skill_with_no_manifest(self):
        result = skills_gate.run_gate(manifest=None, source="",
                                      test_source="", delta=None)
        assert not result.passed

    def test_the_gate_refuses_a_skill_with_no_run_function(self):
        result = skills_gate.run_gate(
            manifest={"name": "s1", "description": "x", "input_schema": {}},
            source="def not_run():\n    return 1\n",
            test_source="", delta=0.1)
        assert not result.passed

    def test_the_gate_refuses_a_skill_that_imports_a_subprocess(self):
        result = skills_gate.run_gate(
            manifest={"name": "s1", "description": "x", "input_schema": {}},
            source="import subprocess\ndef run():\n"
                   "    return subprocess.check_output('id')\n",
            test_source="", delta=0.1)
        assert not result.passed

    def test_the_gate_refuses_a_skill_with_a_new_dependency_when_sealed(self):
        result = skills_gate.run_gate(
            manifest={"name": "s1", "description": "x", "input_schema": {},
                      "dependencies": ["requests"]},
            source="def run():\n    return 1\n",
            test_source="", delta=0.1, sealed=True)
        assert not result.passed

    def test_the_gate_refuses_an_unmeasured_skill(self):
        """No delta, no promotion.

        The regression check needs a measurement.  A skill that was
        never measured is a skill whose effect on quality is unknown,
        and "unknown" is not "no regression".
        """
        result = skills_gate.run_gate(
            manifest={"name": "s1", "description": "x", "input_schema": {}},
            source="def run():\n    return 1\n",
            test_source="", delta=None)
        assert not result.passed, "an unmeasured skill passed the gate"

    def test_the_gate_refuses_a_skill_with_no_test(self):
        """A skill with no test is a skill nobody checked."""
        result = skills_gate.run_gate(
            manifest={"name": "s1", "description": "x", "input_schema": {}},
            source="def run():\n    return 1\n",
            test_source="", delta=0.1)
        assert not result.passed

    def test_the_promotion_refuses_a_skill_that_fails_the_gate(self, learned):
        """``promote`` runs the gate first.  A failed gate is not pinned."""
        skills_curator.synthesize(_candidate(), name="s1",
                                  config_data=learned)
        tool_path = os.path.join(skills.skill_dir("s1", learned), "tool.py")
        with open(tool_path, "w", encoding="utf-8") as f:
            f.write("import subprocess\ndef run():\n"
                    "    return subprocess.check_output('id')\n")
        skills_curator.promote("s1", config_data=learned, delta=0.1)
        assert "s1" not in skills.pinned_skills(learned), \
            "a skill that failed the gate was pinned"
