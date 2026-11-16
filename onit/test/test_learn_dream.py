"""
Dreaming: the consolidation pass that maintains the wiki.

What is being protected here:

1. **The verification.**  A candidate whose citations do not appear in
   the transcript is a hallucination, and a hallucination that reached
   the wiki would be served to every future session.  The check is
   textual and exact: the cited line must exist and the candidate must
   share a token with it.
2. **The creation policy.**  A page is something a topic *earns* --
   fewer verified facts than the threshold stay staged.  A pass that
   created a page per session built the Sonnet-3.7 store, and the
   article measured what that costs.
3. **The watermark.**  A pass processes what it has not seen and
   advances over what it did; a re-run of the same pass is a no-op, not
   a double-write.
4. **The lock.**  Two passes must not run at once, and a task that
   starts mid-dream preempts the pass rather than waiting for it.
5. **The trigger.**  Episodes, not wall-clock: a weekend of zero use
   produces zero new records and no pass, which is the correct outcome.
"""

import asyncio
import json
import os

import pytest

from onit.learn import dream, memory as mem


@pytest.fixture
def world(tmp_path):
    """A learned store with trajectories, autonomy adapt."""
    learned = tmp_path / "learned"
    (learned / "trajectories").mkdir(parents=True)
    return {
        "learn": {
            "autonomy": "adapt",
            "path": str(learned),
            "memory": {"path": str(learned / "memory"), "enabled": True},
        }
    }


def _session(world, session_id, tasks, ts="2026-11-16T10:00:00", verdict="clean"):
    """Write a session of task records, the way the recorder does."""
    path = os.path.join(
        __import__("onit.learn.trajectory", fromlist=["x"]).trajectory_dir(world),
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
                "signals": {"verifier": verdict},
            }
            f.write(json.dumps(record) + "\n")
    return path


def _ask(items):
    """A stub model client that returns the given extraction items."""

    async def ask(messages):
        return json.dumps({"items": items})

    return ask


# ── extraction ────────────────────────────────────────────────────────────────


class TestExtraction:
    def test_a_cited_candidate_survives(self, world):
        _session(world, "s1", [("deploy the service", "deployed; host is stg-7.internal")])
        records = _read(world, "s1")
        found = asyncio.run(
            dream.extract(
                _ask(
                    [
                        {
                            "text": "the staging host is stg-7.internal",
                            "kind": "fact",
                            "topic": "staging",
                            "cites": [2],
                        },
                    ]
                ),
                records,
            )
        )
        assert len(found) == 1
        assert found[0]["provenance"] == ["s1#2"]

    def test_an_uncited_candidate_is_rejected(self, world):
        _session(world, "s1", [("deploy the service", "deployed; host is stg-7.internal")])
        records = _read(world, "s1")
        found = asyncio.run(
            dream.extract(
                _ask(
                    [
                        {
                            "text": "the staging host is stg-9.internal",
                            "kind": "fact",
                            "topic": "staging",
                            "cites": [99],
                        },
                    ]
                ),
                records,
            )
        )
        assert found == []

    def test_a_candidate_that_shares_no_token_with_its_cite_is_rejected(self, world):
        _session(world, "s1", [("deploy the service", "deployed; host is stg-7.internal")])
        records = _read(world, "s1")
        found = asyncio.run(
            dream.extract(
                _ask(
                    [
                        {
                            "text": "the database password is hunter2",
                            "kind": "fact",
                            "topic": "credentials",
                            "cites": [1],
                        },
                    ]
                ),
                records,
            )
        )
        assert found == []

    def test_a_meta_comment_is_rejected(self, world):
        _session(world, "s1", [("deploy the service", "deployed")])
        records = _read(world, "s1")
        found = asyncio.run(
            dream.extract(
                _ask(
                    [
                        {
                            "text": "the session established the deployment",
                            "kind": "fact",
                            "topic": "meta",
                            "cites": [1],
                        },
                    ]
                ),
                records,
            )
        )
        assert found == []

    def test_a_model_failure_yields_nothing(self, world):
        _session(world, "s1", [("deploy the service", "deployed")])
        records = _read(world, "s1")

        async def broken(messages):
            raise RuntimeError("down")

        assert asyncio.run(dream.extract(broken, records)) == []

    def test_an_unparseable_reply_yields_nothing(self, world):
        _session(world, "s1", [("deploy the service", "deployed")])
        records = _read(world, "s1")

        async def prose(messages):
            return "I found three facts worth keeping."

        assert asyncio.run(dream.extract(prose, records)) == []


def _read(world, session_id):
    from onit.learn.trajectory import read_session

    return read_session(session_id, world)


# ── the pass ──────────────────────────────────────────────────────────────────


class TestThePass:
    def test_three_verified_facts_earn_a_page(self, world):
        _session(
            world,
            "s1",
            [
                ("deploy one", "deployed one; the staging host is stg-7.internal"),
                ("deploy two", "deployed two; the staging host is stg-7.internal"),
                ("deploy three", "deployed three; the staging host is stg-7.internal"),
            ],
        )
        out = asyncio.run(
            dream.run_dream(
                world,
                ask=_ask(
                    [
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
                            "text": "staging deploys are verified",
                            "kind": "fact",
                            "topic": "staging hosts",
                            "cites": [3],
                        },
                    ]
                ),
            )
        )
        assert out["pages_created"] == 1
        page = mem.read_page(mem.page_path("staging-hosts", world))
        assert len(page["facts"]) == 3

    def test_fewer_than_three_stay_staged(self, world):
        _session(world, "s1", [("deploy one", "deployed one; host is stg-7")])
        out = asyncio.run(
            dream.run_dream(
                world,
                ask=_ask(
                    [
                        {
                            "text": "the staging host is stg-7",
                            "kind": "fact",
                            "topic": "staging hosts",
                            "cites": [2],
                        },
                    ]
                ),
            )
        )
        assert out["pages_created"] == 0
        assert not mem.list_pages(world)

    def test_the_watermark_makes_a_rerun_a_noop(self, world):
        _session(
            world,
            "s1",
            [
                ("deploy one", "deployed one; host is stg-7"),
                ("deploy two", "deployed two; host is stg-7"),
                ("deploy three", "deployed three; host is stg-7"),
            ],
        )
        first = asyncio.run(
            dream.run_dream(
                world,
                ask=_ask(
                    [
                        {
                            "text": "the staging host is stg-7",
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
                            "text": "staging deploys are verified",
                            "kind": "fact",
                            "topic": "staging hosts",
                            "cites": [3],
                        },
                    ]
                ),
            )
        )
        assert first["ran"]
        second = asyncio.run(
            dream.run_dream(
                world,
                ask=_ask(
                    [
                        {
                            "text": "a brand new fact about nothing",
                            "kind": "fact",
                            "topic": "staging hosts",
                            "cites": [2],
                        },
                    ]
                ),
            )
        )
        # Nothing new since the watermark: the trigger does not fire,
        # which is the idempotence the watermark buys.
        assert second["ran"] is False
        page = mem.read_page(mem.page_path("staging-hosts", world))
        assert len(page["facts"]) == 3

    def test_the_log_records_what_changed(self, world):
        _session(
            world,
            "s1",
            [
                ("deploy one", "deployed one; host is stg-7"),
                ("deploy two", "deployed two; host is stg-7"),
                ("deploy three", "deployed three; host is stg-7"),
            ],
        )
        asyncio.run(
            dream.run_dream(
                world,
                ask=_ask(
                    [
                        {
                            "text": "the staging host is stg-7",
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
                            "text": "staging deploys are verified",
                            "kind": "fact",
                            "topic": "staging hosts",
                            "cites": [3],
                        },
                    ]
                ),
            )
        )
        log = open(mem.log_path(world), encoding="utf-8").read()
        assert "created page staging-hosts" in log
        assert "added fact" in log

    def test_the_index_is_rebuilt(self, world):
        _session(
            world,
            "s1",
            [
                ("deploy one", "deployed one; host is stg-7"),
                ("deploy two", "deployed two; host is stg-7"),
                ("deploy three", "deployed three; host is stg-7"),
            ],
        )
        asyncio.run(
            dream.run_dream(
                world,
                ask=_ask(
                    [
                        {
                            "text": "the staging host is stg-7",
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
                            "text": "staging deploys are verified",
                            "kind": "fact",
                            "topic": "staging hosts",
                            "cites": [3],
                        },
                    ]
                ),
            )
        )
        index = open(mem.index_path(world), encoding="utf-8").read()
        assert "staging-hosts" in index

    def test_a_second_pass_merges_into_the_existing_page(self, world):
        _session(
            world,
            "s1",
            [
                ("deploy one", "deployed one; host is stg-7"),
                ("deploy two", "deployed two; host is stg-7"),
                ("deploy three", "deployed three; host is stg-7"),
            ],
        )
        asyncio.run(
            dream.run_dream(
                world,
                ask=_ask(
                    [
                        {
                            "text": "the staging host is stg-7",
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
                            "text": "staging deploys are verified",
                            "kind": "fact",
                            "topic": "staging hosts",
                            "cites": [3],
                        },
                    ]
                ),
            )
        )
        # A second session, new facts on the same topic.
        _session(
            world,
            "s2",
            [
                ("rollout one", "rolled out; host is stg-7"),
                ("rollout two", "rolled out; host is stg-7"),
            ],
            ts="2026-11-16T11:00:00",
        )
        out = asyncio.run(
            dream.run_dream(
                world,
                ask=_ask(
                    [
                        {
                            "text": "the staging host is stg-7",
                            "kind": "fact",
                            "topic": "staging hosts",
                            "cites": [2],
                        },
                        {
                            "text": "rollouts follow deploys",
                            "kind": "rule",
                            "topic": "staging hosts",
                            "cites": [1],
                        },
                    ]
                ),
            )
        )
        page = mem.read_page(mem.page_path("staging-hosts", world))
        assert page is not None
        assert out["pages_created"] == 0

    def test_open_contradictions_are_resolved_by_a_pass(self, world):
        _session(
            world,
            "s1",
            [
                ("deploy one", "deployed one; host is stg-7"),
                ("deploy two", "deployed two; host is stg-7"),
                ("deploy three", "deployed three; host is stg-7"),
            ],
        )
        asyncio.run(
            dream.run_dream(
                world,
                ask=_ask(
                    [
                        {
                            "text": "the staging host is stg-7",
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
                            "text": "staging deploys are verified",
                            "kind": "fact",
                            "topic": "staging hosts",
                            "cites": [3],
                        },
                    ]
                ),
            )
        )
        # The fact-checker leaves a contradiction open.
        mem.record_contradiction(
            "staging-hosts",
            claim="host is stg-7",
            correction="host is stg-9",
            config_data=world,
        )
        assert mem.lookup("what is the staging host", world)["hit"] is False
        # The next pass resolves it (the transcript supports the page).
        _session(world, "s3", [("another task", "another answer")], ts="2026-11-16T12:00:00")
        out = asyncio.run(dream.run_dream(world, ask=_ask([])))
        assert out["contradictions_cleared"] == 1
        assert mem.lookup("what is the staging host", world)["hit"] is True


# ── the trigger and the lock ──────────────────────────────────────────────────


class TestTriggerAndLock:
    def test_nothing_below_adapt(self, tmp_path):
        world = {"learn": {"autonomy": "observe", "path": str(tmp_path / "l")}}
        assert not dream.should_dream(world)
        out = asyncio.run(dream.run_dream(world, ask=_ask([])))
        assert out["ran"] is False

    def test_an_idle_store_owes_a_pass(self, world):
        _session(world, "s1", [("deploy one", "deployed one")])
        assert dream.should_dream(world)

    def test_a_fresh_watermark_owes_nothing(self, world):
        _session(world, "s1", [("deploy one", "deployed one")])
        state = {"watermark_ts": "2026-11-16T10:00:00"}
        mem.save_state(state, world)
        assert not dream.should_dream(world)

    def test_the_lock_excludes_a_second_pass(self, world):
        with dream._Lock(world):
            with pytest.raises(dream.DreamBusy):
                with dream._Lock(world):
                    pass

    def test_the_lock_is_released_on_exit(self, world):
        with dream._Lock(world):
            pass
        with dream._Lock(world):
            pass  # a second pass can take it now

    def test_a_stale_lock_is_not_stolen(self, world):
        os.makedirs(mem.memory_dir(world), exist_ok=True)
        with open(mem.lock_path(world), "w") as f:
            f.write("999999")
        # A stale lock blocks the pass -- it does not get stolen.  The
        # operator removes it, or the next reboot does; a stolen lock is
        # two writers, and one writer is the whole point.
        with pytest.raises(dream.DreamBusy):
            with dream._Lock(world):
                pass
