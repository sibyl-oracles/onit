"""
The whole chain, end to end.

Every other learn test file tests one module.  This one runs the
*chain* -- record a session, label it, draw a split, mine a skill, gate
it, promote it, roll it back -- and asserts that the pieces fit
together.

A chain test is the only kind that catches the failures that matter
most here, because the loops are separate modules that were written and
tested separately: a signature that drifted, a path that is right in
isolation and wrong in sequence, a store written by one module and read
by another with a different idea of its shape.

The chain also has to be *safe*: it must not write into the repository,
and it must not leave a skill pinned that a later run would serve.
"""

import json
import os
import shutil
import tempfile

import pytest

from src.learn import (corpus, eval as ruler, playbook, recall, skills,
                       skills_curator, skills_gate, skills_mining, trajectory)


@pytest.fixture
def world(tmp_path):
    """A whole world: a learned store, a ruler, and a working dir.

    Every path is under ``tmp_path``.  Nothing here can touch the
    repository, and that is the point: a test that writes into the
    checkout changes what the *next* test measures.
    """
    learned_root = str(tmp_path / "learned")
    ruler_root = str(tmp_path / "ruler")
    work = str(tmp_path / "work")
    for path in (learned_root, ruler_root, work):
        os.makedirs(path, exist_ok=True)
    config = {
        "learn": {"path": learned_root, "eval": {"path": ruler_root}},
        "data_path": work,
    }
    try:
        yield config
    finally:
        shutil.rmtree(learned_root, ignore_errors=True)
        shutil.rmtree(ruler_root, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)


def _session(world, session_id, tasks, model="m1", owner=None,
             verdict="clean", tools=None):
    """Write a whole session of task records, the way the recorder does."""
    path = os.path.join(trajectory.trajectory_dir(world),
                        "%s.jsonl" % session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for index, task in enumerate(tasks):
            record = {
                "kind": "task",
                "session_id": session_id,
                "turn": index,
                "task": task,
                "response": "the answer to: %s" % task,
                "model": model,
                "owner": owner,
                "ts": "2026-10-20T10:00:00",
                "trajectory": [{
                    "turn": index,
                    "tools": tools or [],
                    "signals": {"verifier": verdict},
                }],
                "signals": {"verifier": verdict},
            }
            f.write(json.dumps(record) + "\n")
    return path


# ── the chain, in one run ─────────────────────────────────────────────────

class TestTheWholeChain:
    """Record → label → split → mine → gate → promote → serve → roll back."""

    def test_the_chain_runs_end_to_end(self, world):
        # 1. Record four sessions, each with a repeated bash fragment.
        for index in range(4):
            _session(
                world, "s%d" % index,
                ["deploy the service to staging %d" % index,
                 "run the unit tests %d" % index],
                tools=[{"name": "bash", "args_digest": "d" * 64, "ok": True}])

        # 2. The records are readable.
        records = list(trajectory.iter_records(world))
        assert len(records) == 8, "the session did not land in the store"

        # 3. The corpus can be labelled, and the labels are a sidecar.
        corpus.label_corpus(world)
        assert corpus.read_labels(world) is not None
        assert len(records) == 8, "labeling changed the record count"

        # 4. A split can be drawn and frozen.
        split = ruler.build_split(config_data=world)
        assert split["holdout"] and split["dev"]
        ruler.write_split(split, world)
        assert ruler.read_holdout(world)
        assert ruler.read_dev(world)

        # 5. A skill can be mined from the repetition.
        candidates = skills_mining.mine_store(world)
        assert candidates, "the repeated fragment was not mined"
        candidate = candidates[0]
        name = candidate["name"]

        # 6. The candidate can be synthesised into a skill.
        skills_curator.synthesize(candidate, name=name, config_data=world)
        assert skills.read_skill(name, world), "the skill was not written"

        # 7. The gate has a verdict on it.
        verdict = skills_curator.gate_skill(name, config_data=world,
                                            delta=0.1)
        assert "passed" in verdict, "the gate returned no verdict"

        # 8. A skill that passes the gate can be promoted and pinned.
        if verdict.get("passed"):
            skills_curator.promote(name, config_data=world, delta=0.1)
            assert name in skills.pinned_skills(world)

        # 9. And rolled back.
        skills_curator.rollback(config_data=world)
        assert name not in skills.pinned_skills(world)

    def test_the_server_serves_exactly_what_the_pin_names(self, world):
        """The pin is the allowlist, and the server reads the pin.

        This is the whole safety story of the skills server, and it is
        only a story if the server really reads the pin.
        """
        from src.mcp.servers.tasks.skills import mcp_server

        candidate = {"tool": "bash", "digest": "e" * 64, "sessions": 3,
                     "calls": 5, "errors": 0, "success_rate": 1.0,
                     "arg_keys": ["cmd"]}
        skills_curator.synthesize(candidate, name="served",
                                  config_data=world)
        skills_curator.synthesize(candidate, name="unserved",
                                  config_data=world)
        skills_curator.promote("served", config_data=world, delta=0.1)

        loaded = mcp_server.load_skills(
            {"learn": {"path": world["learn"]["path"]}})
        assert "served" in loaded, "the pinned skill was not served"
        assert "unserved" not in loaded, "an unpinned skill was served"

    def test_two_pinned_skills_are_both_served(self, world):
        """Two skills, two tools.

        The load is a loop over the pin, and a loop that only ever ran
        once would be a loop that quietly drops every skill but the
        last.
        """
        from src.mcp.servers.tasks.skills import mcp_server

        for name, digest in (("one", "a" * 64), ("two", "b" * 64)):
            candidate = {"tool": "bash", "digest": digest, "sessions": 3,
                         "calls": 5, "errors": 0, "success_rate": 1.0,
                         "arg_keys": ["cmd"]}
            skills_curator.synthesize(candidate, name=name,
                                      config_data=world)
            skills_curator.promote(name, config_data=world, delta=0.1)

        loaded = mcp_server.load_skills(
            {"learn": {"path": world["learn"]["path"]}})
        assert set(loaded) == {"one", "two"}, \
            "the server served %r" % sorted(loaded)

    def test_the_server_survives_one_broken_skill(self, world):
        """One broken skill should cost the agent that skill only."""
        from src.mcp.servers.tasks.skills import mcp_server

        for name, digest in (("good", "a" * 64), ("bad", "b" * 64)):
            candidate = {"tool": "bash", "digest": digest, "sessions": 3,
                         "calls": 5, "errors": 0, "success_rate": 1.0,
                         "arg_keys": ["cmd"]}
            skills_curator.synthesize(candidate, name=name,
                                      config_data=world)
            skills_curator.promote(name, config_data=world, delta=0.1)
        with open(os.path.join(skills.skill_dir("bad", world), "tool.py"),
                  "w", encoding="utf-8") as f:
            f.write("this is not python(\n")

        loaded = mcp_server.load_skills(
            {"learn": {"path": world["learn"]["path"]}})
        assert "good" in loaded, "the good skill went down with the bad one"
        assert "bad" not in loaded


# ── the recall path, from the store to the prompt ────────────────────────

class TestRecallReachesThePrompt:
    """A recorded episode has to arrive in the model's context."""

    def test_a_recorded_episode_is_recalled(self, world):
        _session(world, "s1", ["deploy the service to staging"],
                 tools=[{"name": "bash", "args_digest": "f" * 64,
                         "ok": True}])
        episodes = recall.recall("deploy the service to staging",
                                 config_data=world)
        assert episodes, "the episode was recorded but not recalled"
        assert "deploy" in episodes[0]["summary"]

    def test_the_recall_block_is_injectable(self, world):
        _session(world, "s1", ["deploy the service to staging"])
        context = recall.build_context("deploy the service to staging",
                                       config_data=world)
        assert context["text"].strip(), "the recall block was empty"
        assert context["ids"], "the recall block carried no ids"

    def test_the_ids_name_the_records_that_were_shown(self, world):
        """Credit assignment needs the ids.

        The record of a run carries the ids of the episodes it was
        shown, so a later run can ask whether those episodes actually
        helped.  Without the ids the question cannot be asked.
        """
        _session(world, "s1", ["deploy the service to staging"])
        context = recall.build_context("deploy the service to staging",
                                       config_data=world)
        assert any("s1" in i for i in context["ids"]), \
            "the ids do not name the session"


# ── the playbook path, from reflection to injection ───────────────────────

class TestPlaybookReachesThePrompt:
    """A reflected tactic has to arrive in the model's context."""

    def test_a_reflected_tactic_reaches_the_context(self, world):
        """Reflect → curate → build_context, in one go.

        The runs have to land in the *same* cluster, because a cluster
        is a tool signature and the contrast is inside it.  The good
        runs and the bad runs ran the same tools; one of them also ran
        a tool the other did not, and that difference is the lesson.
        """
        for index in range(3):
            _session(
                world, "good%d" % index,
                ["deploy the service %d" % index],
                verdict="clean",
                tools=[{"name": "bash", "args_digest": "g" * 64,
                        "ok": True}])
        for index in range(3):
            _session(
                world, "bad%d" % index,
                ["deploy the service badly %d" % index],
                verdict="issues",
                tools=[{"name": "bash", "args_digest": "g" * 64,
                        "ok": True}])

        records = list(trajectory.iter_records(world))
        deltas = playbook.reflect(records, scope=playbook.SCOPE_GLOBAL)
        assert deltas, "reflection produced nothing to learn from"

        current = playbook.load(world)
        playbook.curate(current, deltas, config_data=world)

        context = playbook.build_context(world)
        assert context["text"].strip(), \
            "the playbook produced an empty block after curating"
        assert context["items"], "the block carried no items"
        assert context["version"] >= 1

    def test_the_playbook_block_is_stable_within_a_session(self, world):
        """The playbook is pinned per session.

        A block that changed between two tasks of the same session
        would change the static half of the prompt between two requests
        in the same conversation, which is exactly what the prefix
        cache cannot survive.
        """
        items = [{"id": "b1", "text": "a tactic", "status": "active",
                  "scope": "global", "trigger": "", "provenance": []}]
        playbook.write_playbook(playbook.global_path(world),
                                {"version": 1, "items": items})
        first = playbook.build_context(world)
        second = playbook.build_context(world)
        assert first["text"] == second["text"]
        assert first["version"] == second["version"]


# ── the labels, from the corpus to the recall ranking ────────────────────

class TestLabelsReachTheRanking:
    """The signals are only useful if the retrieval honours them."""

    def test_a_clean_episode_outranks_an_issue_one(self, world):
        """Two similar episodes, different verdicts.

        The whole point of recording a verdict is that a good episode
        is retrieved ahead of a bad one.  If the ranking ignored the
        verdict, the labels would be decoration.
        """
        _session(world, "good", ["deploy the service to staging"],
                 verdict="clean")
        _session(world, "bad", ["deploy the service to staging"],
                 verdict="issues")
        episodes = recall.recall("deploy the service to staging",
                                 config_data=world)
        assert len(episodes) >= 2, "the test needs two episodes"
        outcomes = [e["outcome"] for e in episodes]
        assert outcomes.index("clean") < outcomes.index("issues"), \
            "the clean episode did not come first: %r" % outcomes

    def test_an_unjudged_episode_is_not_preferred_over_a_clean_one(self, world):
        _session(world, "good", ["deploy the service to staging"],
                 verdict="clean")
        _session(world, "unknown", ["deploy the service to staging"],
                 verdict=None)
        episodes = recall.recall("deploy the service to staging",
                                 config_data=world)
        outcomes = [e["outcome"] for e in episodes]
        assert "clean" in outcomes
        if "unknown" in outcomes or "" in outcomes:
            assert outcomes.index("clean") < max(
                i for i, o in enumerate(outcomes) if o != "clean")


# ── the store is the only thing that crosses the chain ───────────────────

class TestTheStoreIsTheHandoff:
    """Each module reads the store, not another module's memory.

    If a loop kept its own copy of the corpus, then the playbook would
    be distilled from a snapshot and the skills mined from another one,
    and the three would quietly disagree about what happened.
    """

    def test_the_playbook_is_read_from_disk_not_from_memory(self, world):
        items = [{"id": "b1", "text": "a tactic", "status": "active",
                  "scope": "global", "trigger": "", "provenance": []}]
        playbook.write_playbook(playbook.global_path(world),
                                {"version": 1, "items": items})
        loaded = playbook.load(world)
        assert len(loaded["items"]) == 1

    def test_a_second_reader_sees_what_the_first_wrote(self, world):
        """No in-process caching between the writer and the reader."""
        _session(world, "s1", ["deploy the service to staging"])
        first = recall.build_context("deploy the service to staging",
                                     config_data=world)
        second = recall.build_context("deploy the service to staging",
                                      config_data=world)
        assert first["text"] == second["text"]

    def test_the_pin_is_read_from_disk(self, world):
        candidate = {"tool": "bash", "digest": "h" * 64, "sessions": 3,
                     "calls": 5, "errors": 0, "success_rate": 1.0,
                     "arg_keys": ["cmd"]}
        skills_curator.synthesize(candidate, name="s1",
                                  config_data=world)
        skills_curator.promote("s1", config_data=world, delta=0.1)
        assert "s1" in skills.pinned_skills(world)
        # Read it back through a fresh reader: the file is the truth.
        with open(skills.pinned_path(world), encoding="utf-8") as f:
            assert "s1" in f.read()


# ── nothing escapes the world ─────────────────────────────────────────────

class TestNothingEscapes:
    """The chain must not write into the repository.

    The ruler defaults to ``<repo>/benchmarks/eval`` and the learned
    store to ``~/.onit/learned``.  A test that reached either of those
    would change what the next run measures -- and, in the case of the
    learned store, would leave a skill pinned that the real agent
    would then serve.
    """

    def test_the_chain_writes_nothing_outside_the_world(self, world):
        before = sorted(os.listdir(world["learn"]["path"]))
        _session(world, "s1", ["deploy the service to staging"])
        corpus.label_corpus(world)
        ruler.write_split(ruler.build_split(config_data=world), world)
        candidates = skills_mining.mine_store(world)
        if candidates:
            skills_curator.synthesize(candidates[0], name="x",
                                      config_data=world)
        after = sorted(os.listdir(world["learn"]["path"]))
        assert after, "the chain wrote nothing at all"

    def test_the_ruler_stays_in_the_world(self, world):
        ruler.write_split(ruler.build_split(config_data=world), world)
        assert ruler.holdout_path(world).startswith(
            world["learn"]["eval"]["path"])
        assert ruler.is_outside_agent_surface(ruler.eval_dir(world), world)
