"""
The memory chain, end to end: dream → wiki → inject → fast path → check.

Every other memory test file tests one module.  This one runs the
*chain* and asserts the pieces fit: a dream pass writes a page the
injection path serves, the served answer carries the wiki flag that
forces the deep check, and a contradiction the checker records closes
the fast path until the next pass resolves it.

The serving-side pieces are tested through the same seams the real
calls run through -- ``_build_messages`` for the injection, the
``HarnessTools`` dispatch for the tool -- so a signature that drifted
fails here and not in production.
"""

import asyncio
import json
import os

import pytest

from onit.learn import dream, memory as mem


@pytest.fixture
def world(tmp_path):
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
    from onit.learn.trajectory import trajectory_dir

    path = os.path.join(trajectory_dir(world), "%s.jsonl" % session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for index, (task, response) in enumerate(tasks):
            f.write(
                json.dumps(
                    {
                        "kind": "task",
                        "session_id": session_id,
                        "turn": index,
                        "task": task,
                        "response": response,
                        "model": "m1",
                        "owner": None,
                        "ts": ts,
                        "trajectory": [],
                        "signals": {"verifier": "clean"},
                    }
                )
                + "\n"
            )


def _ask(items):
    async def ask(messages):
        return json.dumps({"items": items})

    return ask


_ITEMS = [
    {
        "text": "the staging host is stg-7.internal",
        "kind": "fact",
        "topic": "staging hosts",
        "cites": [2],
    },
    {
        "text": "deploys go through the pipeline",
        "kind": "rule",
        "topic": "staging hosts",
        "cites": [1],
    },
    {
        "text": "staging deploys are verified clean",
        "kind": "fact",
        "topic": "staging hosts",
        "cites": [3],
    },
]


class TestTheChain:
    def test_dream_then_inject_serves_the_page(self, world):
        _session(
            world,
            "s1",
            [
                ("deploy one", "deployed one; the staging host is stg-7.internal"),
                ("deploy two", "deployed two; the staging host is stg-7.internal"),
                ("deploy three", "deployed three; the staging host is stg-7.internal"),
            ],
        )
        out = asyncio.run(dream.run_dream(world, ask=_ask(_ITEMS)))
        assert out["pages_created"] == 1
        ctx = mem.build_context("what is the staging host", world)
        assert ctx["hit"] is True
        assert "stg-7.internal" in ctx["text"]
        assert "provisional" in ctx["text"]
        # The checker's evidence pool carries the same page.
        assert "stg-7.internal" in ctx["evidence"]

    def test_the_block_rides_in_the_volatile_half(self, world):
        from onit.model.serving.chat import _build_messages

        _session(
            world,
            "s1",
            [
                ("deploy one", "deployed one; the staging host is stg-7.internal"),
                ("deploy two", "deployed two; the staging host is stg-7.internal"),
                ("deploy three", "deployed three; the staging host is stg-7.internal"),
            ],
        )
        asyncio.run(dream.run_dream(world, ask=_ask(_ITEMS)))
        ctx = mem.build_context("what is the staging host", world)
        messages = _build_messages(
            "## Task\ndeploy four",
            [],
            "I am OnIt.",
            session_history=None,
            system_rules="rules",
            learned_context=ctx["text"],
        )
        # The block is data for the run, never part of the cached prefix.
        assert "stg-7.internal" not in messages[0]["content"]
        assert "stg-7.internal" in messages[-1]["content"]
        # The task still leads.
        content = messages[-1]["content"]
        assert content.index("stg-7.internal") < content.index("## Task")

    def test_no_memory_means_the_same_bytes(self, world):
        from onit.model.serving.chat import _build_messages

        messages = _build_messages(
            "## Task\ndeploy four",
            [],
            "I am OnIt.",
            session_history=None,
            system_rules="rules",
            learned_context="",
        )
        assert messages[-1]["content"] == "## Task\ndeploy four"

    def test_the_wiki_tool_serves_the_page(self, world):
        from onit.model.serving.harness import HarnessTools

        _session(
            world,
            "s1",
            [
                ("deploy one", "deployed one; the staging host is stg-7.internal"),
                ("deploy two", "deployed two; the staging host is stg-7.internal"),
                ("deploy three", "deployed three; the staging host is stg-7.internal"),
            ],
        )
        asyncio.run(dream.run_dream(world, ask=_ask(_ITEMS)))
        # Point the tool at this world's store.
        harness = HarnessTools(data_path="/tmp", config_data=world)
        harness.wiki_available = True
        out = harness.wiki_search("what is the staging host")
        assert "stg-7.internal" in out
        assert "untrusted" in out

    def test_a_contradiction_closes_then_reopens_the_fast_path(self, world):
        _session(
            world,
            "s1",
            [
                ("deploy one", "deployed one; the staging host is stg-7.internal"),
                ("deploy two", "deployed two; the staging host is stg-7.internal"),
                ("deploy three", "deployed three; the staging host is stg-7.internal"),
            ],
        )
        asyncio.run(dream.run_dream(world, ask=_ask(_ITEMS)))
        assert mem.build_context("what is the staging host", world)["hit"] is True
        # The checker finds the world disagrees.
        mem.record_contradiction(
            "staging-hosts",
            claim="host is stg-7",
            correction="host is stg-9",
            config_data=world,
        )
        ctx = mem.build_context("what is the staging host", world)
        assert ctx["hit"] is False
        assert "lead" in ctx["text"]
        # The next dream pass resolves it.
        _session(world, "s2", [("another", "answer")], ts="2026-11-16T12:00:00")
        asyncio.run(dream.run_dream(world, ask=_ask([])))
        assert mem.build_context("what is the staging host", world)["hit"] is True

    def test_the_evidence_is_labeled_unverified(self, world):
        _session(
            world,
            "s1",
            [
                ("deploy one", "deployed one; the staging host is stg-7.internal"),
                ("deploy two", "deployed two; the staging host is stg-7.internal"),
                ("deploy three", "deployed three; the staging host is stg-7.internal"),
            ],
        )
        asyncio.run(dream.run_dream(world, ask=_ask(_ITEMS)))
        ev = mem.evidence_for("what is the staging host", world)
        assert "unverified" in ev["text"]
        assert ev["slug"] == "staging-hosts"

    def test_the_evidence_never_clears_a_claim(self, world):
        """The wiki is distilled, not primary: matching it proves nothing.

        ``covered_by_trusted_sources`` is the free pass that skips the
        fact-check when every figure came from an authoritative source.
        Wiki evidence must never trigger it, or a memory answer would
        verify itself.
        """
        from onit.model.serving.verify import covered_by_trusted_sources

        _session(
            world,
            "s1",
            [
                ("deploy one", "deployed one; the staging host is stg-7.internal"),
                ("deploy two", "deployed two; the staging host is stg-7.internal"),
                ("deploy three", "deployed three; the staging host is stg-7.internal"),
            ],
        )
        asyncio.run(dream.run_dream(world, ask=_ask(_ITEMS)))
        ev = mem.evidence_for("what is the staging host", world)
        answer = "the staging host is stg-7.internal and 42% of deploys"
        messages = [{"role": "tool", "name": "wiki_search", "content": ev["text"]}]
        assert not covered_by_trusted_sources(answer, messages)

    def test_an_empty_store_changes_nothing(self, world):
        ctx = mem.build_context("anything at all", world)
        assert ctx == {"text": "", "pages": [], "hit": False, "slug": "", "evidence": ""}
