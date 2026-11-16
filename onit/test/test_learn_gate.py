"""
The wiki gate: the measurement before the memory wiki's default can move.

What is being protected here:

1. **The isolation.**  The two arms sit at the same autonomy level and
   differ only in ``learn.memory.enabled``.  A comparison whose arms
   also differ in recall, playbooks or the dream trigger measures the
   whole learn stack, and the wiki's own effect is unmeasurable.
2. **The freeze.**  A replay must not dream mid-run: a pass that fired
   between items would extract facts from the very items being scored,
   which is training on the test set.  The trigger is deferred, not
   disabled, so the gate's own seeding still works.
3. **The seeding.**  The treatment's wiki is distilled from the
   control's recorded runs -- the same experience both arms recall --
   by the same dream pass production uses, hallucination check
   included.  A fact the control never established does not enter.
4. **The verdict.**  The delta is compared against its error bar, and
   the printed verdict is the decision the plan ties to it: raise the
   default, keep it off, or look for poisoning.
"""

import asyncio
import json
import os
import unittest.mock

import pytest

from onit.learn import dream, memory as mem, runner


@pytest.fixture
def world(tmp_path):
    """A learned store, autonomy adapt, memory on."""
    learned = tmp_path / "learned"
    (learned / "trajectories").mkdir(parents=True)
    return {
        "learn": {
            "autonomy": "adapt",
            "path": str(learned),
            "memory": {"path": str(learned / "memory"), "enabled": True},
        }
    }


def _session(world, session_id, tasks, ts="2026-11-16T10:00:00"):
    """Write a session of task records, the way the recorder does."""
    from onit.learn import trajectory as trajectory_module

    path = os.path.join(
        trajectory_module.trajectory_dir(world),
        "%s.jsonl" % session_id,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for index, (task, response) in enumerate(tasks):
            record = {
                "kind": "task",
                "session_id": session_id,
                "turn": index,
                "task": task,
                "response": response,
                "model": "m1",
                "owner": None,
                "ts": ts,
                "trajectory": [],
                "signals": {},
            }
            f.write(json.dumps(record) + "\n")
    return path


def _ask(items):
    """A stub model client that returns the given extraction items."""

    async def ask(messages):
        return json.dumps({"items": items})

    return ask


class StubAgent:
    """An agent that answers with a fixed string and records its calls."""

    def __init__(self, answer="the right answer", config=None):
        self.answer = answer
        self.config = config or {}
        self.calls = []

    async def process_task(self, task, **kwargs):
        self.calls.append({"task": task, "kwargs": kwargs})
        return self.answer


def _item(task="a question", response="the right answer", item_id="s1#1"):
    return {
        "id": item_id,
        "task": task,
        "response": response,
        "model": "m1",
        "owner": None,
        "session_id": "s1",
        "turn": 1,
    }


async def _ask_ok(messages):
    return "GRADE: C"


# ── the switch ────────────────────────────────────────────────────────────


class TestEnabled:
    def test_the_default_is_off(self):
        assert mem.enabled({}) is False

    def test_the_config_switches_it(self):
        assert mem.enabled({"learn": {"memory": {"enabled": True}}}) is True
        assert mem.enabled({"learn": {"memory": {"enabled": False}}}) is False

    def test_a_quoted_false_is_false(self):
        assert mem.enabled({"learn": {"memory": {"enabled": "false"}}}) is False
        assert mem.enabled({"learn": {"memory": {"enabled": "no"}}}) is False
        assert mem.enabled({"learn": {"memory": {"enabled": "true"}}}) is True

    def test_the_env_switches_it(self, monkeypatch):
        monkeypatch.setenv("ONIT_MEMORY_ENABLED", "1")
        assert mem.enabled({}) is True
        monkeypatch.setenv("ONIT_MEMORY_ENABLED", "0")
        assert mem.enabled({}) is False
        monkeypatch.delenv("ONIT_MEMORY_ENABLED")
        assert mem.enabled({}) is False

    def test_the_config_wins_over_the_env(self, monkeypatch):
        monkeypatch.setenv("ONIT_MEMORY_ENABLED", "1")
        assert mem.enabled({"learn": {"memory": {"enabled": False}}}) is False


# ── the arm configs ──────────────────────────────────────────────────────


class TestWikiArmConfigs:
    def test_the_arms_differ_only_in_the_switch(self, world):
        on = runner._config_for_wiki_arm(world, True)
        off = runner._config_for_wiki_arm(world, False)
        assert on["learn"]["memory"]["enabled"] is True
        assert off["learn"]["memory"]["enabled"] is False
        assert on["learn"]["autonomy"] == off["learn"]["autonomy"] == "adapt"
        assert on["learn"]["path"] == off["learn"]["path"] == world["learn"]["path"]

    def test_the_replay_is_frozen_against_dreaming(self, world):
        config = runner._config_for_wiki_arm(world, True)
        assert config["learn"]["memory"]["min_idle_episodes"] == runner.REPLAY_MIN_IDLE_EPISODES
        assert dream.should_dream(config) is False

    def test_the_seeding_is_not_blocked_by_the_freeze(self, world):
        # A forced pass consults neither the switch nor the idle count,
        # so the same frozen config that blocks the trigger mid-run
        # still admits the seed between arms.
        frozen = runner._config_for_wiki_arm(world, False)
        result = asyncio.run(dream.run_dream(frozen, ask=_ask([]), force=True))
        assert result.get("ran") is True

    def test_a_frozen_disabled_store_owes_no_pass(self, world):
        frozen = runner._config_for_wiki_arm(world, False)
        assert dream.dream_enabled(frozen) is False
        result = asyncio.run(dream.run_dream(frozen, ask=_ask([])))
        assert result.get("ran") is False

    def test_a_label_arm_is_not_coerced_to_a_level(self, world):
        # "wiki-on" is a label, not an autonomy level; _config_for_arm
        # must not overwrite the pinned level with it.
        config = runner._config_for_arm(world, "wiki-on", "/store")
        assert config["learn"]["autonomy"] == "adapt"
        assert config["learn"]["path"] == "/store"
        classic = runner._config_for_arm(world, "adapt", "/store")
        assert classic["learn"]["autonomy"] == "adapt"


# ── the seeding ──────────────────────────────────────────────────────────


class TestSeeding:
    def test_the_control_runs_become_the_treatments_wiki(self, world):
        _session(
            world,
            "s1",
            [
                ("deploy the service", "deployed; host is stg-7.internal"),
                ("check the queue", "queue is clear; broker is mq-2.internal"),
                ("rotate the key", "rotated; new fingerprint starts ab12"),
            ],
        )
        treatment = runner._config_for_wiki_arm(world, True)
        result = asyncio.run(
            dream.run_dream(
                treatment,
                ask=_ask(
                    [
                        {
                            "text": "the staging host is stg-7.internal",
                            "kind": "fact",
                            "topic": "staging",
                            "cites": [2],
                        },
                        {
                            "text": "the broker is mq-2.internal",
                            "kind": "fact",
                            "topic": "staging",
                            "cites": [4],
                        },
                        {
                            "text": "the new fingerprint starts ab12",
                            "kind": "fact",
                            "topic": "staging",
                            "cites": [6],
                        },
                    ]
                ),
                force=True,
            )
        )
        assert result.get("ran") is True
        assert result.get("pages_created") == 1
        pages = mem.list_pages(world)
        assert len(pages) == 1
        assert len(pages[0]["facts"]) >= 3

    def test_a_hallucinated_seed_does_not_enter_the_wiki(self, world):
        _session(world, "s1", [("deploy the service", "deployed; host is stg-7.internal")])
        treatment = runner._config_for_wiki_arm(world, True)
        asyncio.run(
            dream.run_dream(
                treatment,
                ask=_ask(
                    [
                        {
                            "text": "the production host is prd-9.internal",
                            "kind": "fact",
                            "topic": "staging",
                            "cites": [2],
                        },
                    ]
                ),
                force=True,
            )
        )
        assert mem.list_pages(world) == []

    def test_an_empty_store_seeds_an_empty_wiki(self, world):
        treatment = runner._config_for_wiki_arm(world, True)
        result = asyncio.run(dream.run_dream(treatment, ask=_ask([]), force=True))
        assert result.get("ran") is True
        assert mem.list_pages(world) == []


# ── the inheritance ──────────────────────────────────────────────────────


class TestInheritance:
    def test_the_control_trajectories_reach_the_treatment_store(self, tmp_path):
        src = tmp_path / "control"
        dst = tmp_path / "treatment"
        (src / "trajectories").mkdir(parents=True)
        dst.mkdir()
        (src / "trajectories" / "s1.jsonl").write_text('{"kind": "task"}\n')
        (src / "trajectories" / "loop.jsonl").write_text('{"kind": "event"}\n')
        copied = runner._inherit_trajectories(str(src), str(dst))
        assert copied == 1
        assert (dst / "trajectories" / "s1.jsonl").exists()
        assert not (dst / "trajectories" / "loop.jsonl").exists()

    def test_an_existing_file_is_not_overwritten(self, tmp_path):
        src = tmp_path / "control"
        dst = tmp_path / "treatment"
        (src / "trajectories").mkdir(parents=True)
        (dst / "trajectories").mkdir(parents=True)
        (src / "trajectories" / "s1.jsonl").write_text('{"from": "control"}\n')
        (dst / "trajectories" / "s1.jsonl").write_text('{"from": "treatment"}\n')
        runner._inherit_trajectories(str(src), str(dst))
        assert json.loads((dst / "trajectories" / "s1.jsonl").read_text())["from"] == "treatment"

    def test_a_store_with_no_trajectories_inherits_nothing(self, tmp_path):
        src = tmp_path / "control"
        dst = tmp_path / "treatment"
        src.mkdir()
        dst.mkdir()
        assert runner._inherit_trajectories(str(src), str(dst)) == 0


# ── the arms ─────────────────────────────────────────────────────────────


class TestWikiArms:
    async def _run(self, tmp_path, items, **kwargs):
        kwargs.setdefault("make_agent", lambda config: StubAgent(config=config))
        return await runner.run_wiki_ab(
            items,
            root=str(tmp_path),
            base_config={"learn": {"memory": {"path": str(tmp_path / "mem")}}},
            ask=_ask_ok,
            seed_ask=_ask([]),
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_both_arms_run_and_are_compared(self, tmp_path):
        result = await self._run(tmp_path, [_item()] * 4)
        assert result["results"]["wiki-off"]
        assert result["results"]["wiki-on"]
        assert result["summaries"]["wiki-off"]["n"] == 4
        assert result["summaries"]["wiki-on"]["n"] == 4
        assert "delta" in result

    @pytest.mark.asyncio
    async def test_the_arms_sit_at_the_same_autonomy(self, tmp_path):
        seen = []

        def factory(config):
            seen.append(dict(config))
            return StubAgent(config=config)

        await runner.run_wiki_ab(
            [_item()],
            root=str(tmp_path),
            base_config={"learn": {"memory": {"path": str(tmp_path / "mem")}}},
            make_agent=factory,
            ask=_ask_ok,
            seed_ask=_ask([]),
        )
        assert len(seen) == 2
        assert seen[0]["learn"]["autonomy"] == seen[1]["learn"]["autonomy"] == "adapt"
        assert seen[0]["learn"]["memory"]["enabled"] is False
        assert seen[1]["learn"]["memory"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_each_arm_gets_its_own_store(self, tmp_path):
        # The stores are separate directories, and the treatment's
        # memory path is pinned under its own store -- an operator
        # config that names learn.memory.path must not point both arms
        # at one wiki.
        await self._run(tmp_path, [_item()] * 2)
        assert os.path.isdir(os.path.join(str(tmp_path), "store", "wiki-off"))
        assert os.path.isdir(os.path.join(str(tmp_path), "store", "wiki-on"))
        assert not os.path.exists(
            os.path.join(str(tmp_path), "store", "wiki-off", "memory", "wiki")
        )
        assert not os.path.exists(os.path.join(str(tmp_path), "store", "wiki-on", "memory", "wiki"))

    @pytest.mark.asyncio
    async def test_a_limit_caps_both_arms(self, tmp_path):
        result = await self._run(tmp_path, [_item()] * 6, limit=2)
        assert result["summaries"]["wiki-off"]["n"] == 2
        assert result["summaries"]["wiki-on"]["n"] == 2

    @pytest.mark.asyncio
    async def test_no_make_agent_still_runs_the_real_factory(self, tmp_path):
        """The CLI calls run_wiki_ab without make_agent.

        The factory must fall back to the real one the way run_ab's
        does -- a caller who passes none gets arms that run the agent,
        not a ValueError before the first item.  The fallback itself is
        exercised by the CLI; here the contract under test is that a
        make_agent-less call runs to a verdict instead of raising.
        """
        called = []

        def real_factory(config):
            called.append(config)
            return StubAgent(config=config)

        with unittest.mock.patch.object(runner, "_wiki_agent_factory", real_factory):
            result = await self._run(tmp_path, [_item()] * 2, make_agent=None)
        assert called, "the fallback factory never built an agent"
        assert result["summaries"]["wiki-off"]["n"] == 2
        assert result["summaries"]["wiki-on"]["n"] == 2
        assert result["summaries"]["wiki-off"]["correct"] == 2
        assert result["summaries"]["wiki-on"]["correct"] == 2


# ── the verdict ──────────────────────────────────────────────────────────


class TestVerdict:
    def _result(self, c_correct, t_correct, n=40):
        control = [dict(grade="C" if i < c_correct else "I", error="") for i in range(n)]
        treatment = [dict(grade="C" if i < t_correct else "I", error="") for i in range(n)]
        return runner.compare(control, treatment)

    def test_a_significant_win_raises_the_default(self):
        text = runner.format_wiki_verdict(self._result(10, 25))
        assert "earns its default" in text

    def test_a_significant_loss_is_the_poisoning_signal(self):
        text = runner.format_wiki_verdict(self._result(25, 10))
        assert "poisoning" in text

    def test_a_tie_keeps_the_default_off(self):
        text = runner.format_wiki_verdict(self._result(20, 20))
        assert "no measurable effect" in text

    def test_the_verdict_carries_the_numbers(self):
        text = runner.format_wiki_verdict(self._result(10, 25))
        assert "wiki-off" in text and "wiki-on" in text
        assert "delta" in text


class TestAnUngradedRunIsNotAMeasurement:
    """A run whose every grade came back empty is not a tie.

    The grader used to swallow its failures at debug level, and a run
    against a dead endpoint printed 0.000 vs 0.000 with the verdict
    "no measurable effect" -- which reads as a clean null result about
    the wiki.  It is not: nothing was measured.  The summary counts the
    grader's failures separately, and the verdict refuses to speak.
    """

    def _raw(self, n=20):
        return [dict(grade="", error="", grading_error="APIConnectionError: x") for _ in range(n)]

    def _result(self, n=20):
        control = self._raw(n)
        treatment = [
            dict(grade="", error="", grading_error="APIConnectionError: x") for _ in range(n)
        ]
        return runner.compare(control, treatment)

    def test_the_verdict_says_not_measured(self):
        text = runner.format_wiki_verdict(self._result())
        assert "NOT MEASURED" in text
        assert "no measurable effect" not in text

    def test_the_summary_counts_grading_errors(self):
        summary = runner.summarise(self._raw())
        assert summary["grading_errors"] == 20
        assert summary["ungraded"] == 20
        assert summary["accuracy"] == 0.0

    def test_the_comparison_prints_the_grading_errors(self):
        text = runner.format_comparison(self._result(), label="Wiki gate:")
        assert "grade-err=20" in text
        assert "NOTHING WAS GRADED" in text

    def test_a_partially_graded_run_still_gets_a_normal_verdict(self):
        """One dead grader call must not void a run that measured."""
        results = [dict(grade="C", error="", grading_error="") for _ in range(19)]
        results.append(dict(grade="", error="", grading_error="timeout"))
        text = runner.format_wiki_verdict(runner.compare(results, results))
        assert "NOT MEASURED" not in text

    def test_the_grader_failure_is_recorded_on_the_item(self):
        """The exception text lands in results.json, not just the log."""

        async def dead_ask(messages):
            raise ConnectionError("connection refused")

        verdict = asyncio.run(runner.grade(dead_ask, "task", "ref", "an answer"))
        assert verdict["grade"] == ""
        assert "ConnectionError" in verdict["grading_error"]


# ── the arm announces its seeding ────────────────────────────────────


class TestSeedingAnnouncesItself:
    """Seeding is minutes of model calls; silence reads as a hang."""

    async def _run(self, tmp_path, items, **kwargs):
        kwargs.setdefault("make_agent", lambda config: StubAgent(config=config))
        return await runner.run_wiki_ab(
            items,
            root=str(tmp_path),
            base_config={"learn": {"memory": {"path": str(tmp_path / "mem")}}},
            ask=_ask_ok,
            seed_ask=_ask([]),
            **kwargs,
        )

    async def test_seeding_is_announced_when_it_runs(self, tmp_path, capsys):
        # The stub records no trajectories of its own, so stage one in
        # the control's store -- what a real control run leaves behind.
        trajectories = tmp_path / "store" / "wiki-off" / "trajectories"
        trajectories.mkdir(parents=True, exist_ok=True)
        (trajectories / "123e4567-e89b-42d3-a456-426614174000.jsonl").write_text("{}\n")
        await self._run(tmp_path, [_item()])
        out = capsys.readouterr().out
        assert "seeding: inherited" in out
        assert "seeding: distilling" in out
        assert "seeding: done" in out

    async def test_a_skipped_seeding_is_announced(self, tmp_path, capsys):
        """An empty wiki must never masquerade as a measured treatment.

        ``seed_ask=None`` means "default the extraction client to the
        grader", so the skip only happens when there is no client at
        all -- no grader, no seeding client.
        """
        result = await runner.run_wiki_ab(
            [_item()],
            root=str(tmp_path),
            base_config={"learn": {"memory": {"path": str(tmp_path / "mem")}}},
            make_agent=lambda config: StubAgent(config=config),
            ask=None,
            seed_ask=None,
            with_grader=False,
        )
        out = capsys.readouterr().out
        assert "seeding skipped" in out
        assert "empty wiki" in out
        # The run itself still completes and reports.
        assert result["summaries"]["wiki-on"]["n"] == 1
