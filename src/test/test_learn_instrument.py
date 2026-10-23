"""End-to-end smoke: the runner and the A/B, with a stub agent.

No network, no model server, no endpoint.  The point is to prove that
the pieces fit: the CLI's flags reach the runner, the runner's arms
write their results where they said they would, the comparison comes
out with an error bar, and the reflector produces bullets a curator
will accept.
"""

import asyncio
import json
import os
import shutil
import tempfile

import pytest

from src.learn import ab, playbook, reflector, runner
from src.learn import trajectory


class StubAgent:
    """Answers a fixed string, and records the config it was built with."""

    instances = []

    def __init__(self, config=None):
        self.config = config or {}
        self.calls = []
        StubAgent.instances.append(self)

    async def process_task(self, task, **kwargs):
        self.calls.append(task)
        return "the right answer"


def _stub_factory(config):
    """The agent factory the tests inject."""
    return StubAgent(config)


def _item(task="a question", response="the right answer", item_id="s1#1"):
    return {"id": item_id, "task": task, "response": response,
            "model": "m1", "owner": None, "session_id": "s1", "turn": 1}


async def _ask_ok(messages):
    return "GRADE: C"


# ── the A/B ─────────────────────────────────────────────────────────────────

class TestRunAb:
    """Two arms, one comparison."""

    async def test_the_ab_produces_a_comparison(self, tmp_path):
        StubAgent.instances = []
        items = [_item("q1", "the reference", "s1#1"),
                 _item("q2", "the reference", "s1#2")]
        result = await ab.run_ab(items, root=str(tmp_path),
                                 base_config={"serving": {"host": "x"}},
                                 limit=2, with_grader=False,
                                 make_agent=_stub_factory)
        assert result["control"]["n"] == 2
        assert result["treatment"]["n"] == 2
        assert "delta" in result
        assert "stderr" in result

    async def test_the_two_arms_are_two_agents(self, tmp_path):
        """One agent per arm.

        The arms differ in the agent's own configuration, and a shared
        agent would carry the first arm's configuration into the second
        -- so the second arm would be the first arm wearing a label.
        """
        StubAgent.instances = []
        items = [_item()]
        await ab.run_ab(items, root=str(tmp_path),
                        base_config={"serving": {"host": "x"}},
                        limit=1, with_grader=False,
                        make_agent=_stub_factory)
        assert len(StubAgent.instances) == 2
        autonomies = {a.config["learn"]["autonomy"]
                       for a in StubAgent.instances}
        assert autonomies == {"off", "adapt"}

    async def test_the_results_are_on_disk(self, tmp_path):
        StubAgent.instances = []
        items = [_item()]
        await ab.run_ab(items, root=str(tmp_path),
                        base_config={"serving": {"host": "x"}},
                        limit=1, with_grader=False,
                        make_agent=_stub_factory)
        for arm in ("off", "adapt"):
            path = os.path.join(str(tmp_path), "arms", arm, "results.json")
            assert os.path.exists(path), "no results for the %s arm" % arm
            with open(path, encoding="utf-8") as f:
                assert json.load(f)

    async def test_the_comparison_is_formattable(self, tmp_path):
        StubAgent.instances = []
        items = [_item()]
        result = await ab.run_ab(items, root=str(tmp_path),
                                 base_config={"serving": {"host": "x"}},
                                 limit=1, with_grader=False,
                                 make_agent=_stub_factory)
        text = runner.format_comparison(result, label="A/B:")
        assert "control" in text
        assert "treatment" in text
        assert "delta" in text


# ── the base config ─────────────────────────────────────────────────────────

class TestBaseConfigForRun:
    def test_the_ui_is_off(self):
        config = ab.base_config_for_run({"verbose": True})
        assert config["loop"] is True
        assert config["stream"] is False

    def test_the_operators_config_is_carried_over(self):
        config = ab.base_config_for_run({"serving": {"host": "http://x/v1"}})
        assert config["serving"]["host"] == "http://x/v1"


# ── the reflector, end to end ───────────────────────────────────────────────

class TestReflectorToPlaybook:
    """A model bullet has to survive the curator and reach the context."""

    async def test_a_model_bullet_reaches_the_playbook(self, tmp_path):
        """The whole path: model → delta → curate → build_context.

        A bullet the model wrote has to be a bullet the curator
        accepts, because a delta the curator rejects is a lesson
        silently thrown away.
        """
        config = {"learn": {"path": str(tmp_path / "learned")},
                  "data_path": str(tmp_path / "work")}

        good = {"session_id": "s1", "turn": 1,
                "task": "find the weather forecast",
                "response": "an answer", "signals": {"verifier": "clean"},
                "trajectory": [{"turn": 1, "tools": [{"name": "search",
                                                      "ok": True}]}]}
        bad = {"session_id": "s2", "turn": 1,
               "task": "find the weather forecast",
               "response": "an answer", "signals": {"verifier": "issues"},
               "trajectory": [{"turn": 1, "tools": [{"name": "search",
                                                     "ok": False}]}]}

        reply = json.dumps({"bullets": [
            {"text": "Check the date on the page before quoting it",
             "trigger": "fetch"}]})

        async def ask(messages):
            return reply

        deltas = await reflector.reflect_corpus(ask, [good, bad])
        assert deltas, "the reflector produced no bullets"

        before = playbook.load(config)
        after = playbook.curate(before, deltas, config_data=config)
        assert len(after["items"]) > len(before["items"])

        context = playbook.build_context(config)
        assert "Check the date" in context["text"]
        assert context["items"]

    async def test_a_bullet_the_curator_would_reject_is_caught(self, tmp_path):
        """A model bullet with no text is not a bullet.

        The curator would drop it, and the loop would report that it
        reflected when it had not.  The reflector should not produce it
        in the first place.
        """
        reply = json.dumps({"bullets": [{"trigger": "search"}]})

        async def ask(messages):
            return reply

        good = {"session_id": "s1", "turn": 1, "task": "find the weather",
                "response": "an answer", "signals": {"verifier": "clean"},
                "trajectory": [{"turn": 1, "tools": []}]}
        bad = {"session_id": "s2", "turn": 1, "task": "find the weather",
               "response": "an answer", "signals": {"verifier": "issues"},
               "trajectory": [{"turn": 1, "tools": []}]}
        deltas = await reflector.reflect_corpus(ask, [good, bad])
        assert deltas == [], "a bullet with no text was produced"
