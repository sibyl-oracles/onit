"""
The owner boundary at the *harness* seam.

``test_learn_invariants.py`` proves that ``playbook.build_context`` and
``recall.recall`` respect the owner boundary.  That is the library's
contract, and it holds.

It is not enough on its own, because the harness does not call those
functions the way a test calls them.  The harness **caches** the
playbook on the ``OnIt`` instance, keyed by nothing, and the web UI puts
**one** ``OnIt`` instance behind **every** session.  So the library's
boundary is bypassed by the caller's cache: the first session to pin a
playbook decides what every other session is shown.

This file tests the harness, not the library.  Each test here is a way
the seam could leak, and each one fails if the seam leaks.

The fix is in ``OnIt._pin_playbook``: the pin is keyed by owner, not by
instance, so two tenants on one instance get two playbooks.
"""

import json
import os

import pytest

from src.learn import playbook


@pytest.fixture
def learned(tmp_path):
    """An empty learned store, isolated from the real one."""
    root = str(tmp_path / "learned")
    sessions = str(tmp_path / "sessions")
    os.makedirs(root, exist_ok=True)
    os.makedirs(sessions, exist_ok=True)
    yield {"learn": {"autonomy": "adapt", "path": root},
           "_sessions_dir": sessions}


def _write_private(learned, owner, text):
    """Give ``owner`` a private playbook bullet."""
    playbook.write_playbook(
        playbook.owner_path(owner, learned),
        {"version": 1, "items": [
            {"id": "b-%s" % owner, "text": text, "status": "active",
             "scope": "owner:%s" % owner, "trigger": "", "provenance": []}]})


def _agent(learned):
    """A bare OnIt instance -- no model, no MCP servers.

    The pin is a pure function of the store and the owner, so the
    instance does not need a serving host to exercise it.  It is built
    the way pydantic builds it, because a hand-rolled instance would
    not have the private attributes the real one has, and the test
    would then be testing a different object than the one that ships.
    """
    from src.onit import OnIt
    # ``_pin_playbook`` resolves the owner from ``dirname(session_path)``,
    # so the session path must be a *file* inside the sessions directory.
    session_file = os.path.join(learned["_sessions_dir"], "sess.jsonl")
    agent = OnIt.model_construct(
        config_data=learned,
        session_path=session_file,
    )
    agent._playbook_pinned = False
    agent._playbook_context = ""
    agent._playbook_items = []
    agent._playbook_version = None
    agent._playbook_owner = None
    agent._playbook_by_owner = {}
    return agent


def _own_session(learned, session_id, owner):
    """Record an owner for a session, the way the web UI does."""
    from src.sessions import set_session_owner
    set_session_owner(session_id, owner,
                      sessions_dir=learned["_sessions_dir"])


class TestPlaybookPinIsPerOwner:
    """The pin is cached, and the cache must not cross the boundary."""

    def test_a_second_tenant_does_not_see_the_first_tenant_s_bullets(self, learned):
        """The leak this file exists for.

        Alice's session runs first on a shared instance.  Bob's session
        then runs on the same instance.  Bob must not be shown Alice's
        private bullets.
        """
        _write_private(learned, "alice@corp.com", "ALICE PRIVATE bullet")
        _own_session(learned, "sess-alice", "alice@corp.com")
        _own_session(learned, "sess-bob", "bob@corp.com")
        agent = _agent(learned)

        alice_text = agent._pin_playbook("sess-alice")
        bob_text = agent._pin_playbook("sess-bob")

        assert "ALICE PRIVATE bullet" in alice_text, \
            "alice must get her own bullet, or the test proves nothing"
        assert "ALICE PRIVATE bullet" not in bob_text, \
            "bob was shown alice's private bullet"

    def test_each_owner_gets_their_own_playbook(self, learned):
        """The pin is a per-owner cache, not a per-owner rebuild.

        Alice and Bob each have a private bullet.  Each must get
        theirs, and only theirs.
        """
        _write_private(learned, "alice@corp.com", "ALICE PRIVATE bullet")
        _write_private(learned, "bob@corp.com", "BOB PRIVATE bullet")
        _own_session(learned, "sess-alice", "alice@corp.com")
        _own_session(learned, "sess-bob", "bob@corp.com")
        agent = _agent(learned)

        alice_text = agent._pin_playbook("sess-alice")
        bob_text = agent._pin_playbook("sess-bob")

        assert "ALICE PRIVATE bullet" in alice_text
        assert "BOB PRIVATE bullet" not in alice_text
        assert "BOB PRIVATE bullet" in bob_text
        assert "ALICE PRIVATE bullet" not in bob_text

    def test_the_ownerless_default_is_not_a_tenant(self, learned):
        """A session with no owner gets the global playbook.

        The global playbook is shared by design.  The point is that a
        session *with* an owner does not fall back onto it.
        """
        _write_private(learned, "alice@corp.com", "ALICE PRIVATE bullet")
        _own_session(learned, "sess-alice", "alice@corp.com")
        agent = _agent(learned)

        anonymous = agent._pin_playbook("sess-anon")
        assert "ALICE PRIVATE bullet" not in anonymous

    def test_repeating_a_pin_for_the_same_owner_is_stable(self, learned):
        """The cache still works.

        The fix must not turn the pin into a rebuild on every task:
        that would trade a privacy bug for a latency bug.
        """
        _write_private(learned, "alice@corp.com", "ALICE PRIVATE bullet")
        _own_session(learned, "sess-alice", "alice@corp.com")
        _own_session(learned, "sess-alice-again", "alice@corp.com")
        agent = _agent(learned)

        first = agent._pin_playbook("sess-alice")
        second = agent._pin_playbook("sess-alice-again")
        assert first == second

    def test_the_pin_is_cached_per_owner(self, learned):
        """Two owners means two entries in the cache."""
        _write_private(learned, "alice@corp.com", "ALICE PRIVATE bullet")
        _write_private(learned, "bob@corp.com", "BOB PRIVATE bullet")
        _own_session(learned, "sess-alice", "alice@corp.com")
        _own_session(learned, "sess-bob", "bob@corp.com")
        _own_session(learned, "sess-alice-2", "alice@corp.com")
        agent = _agent(learned)

        agent._pin_playbook("sess-alice")
        agent._pin_playbook("sess-bob")
        agent._pin_playbook("sess-alice-2")

        assert len(getattr(agent, "_playbook_by_owner", {})) == 2, \
            "the pin was rebuilt instead of cached per owner"


class TestRecallIsAlreadyPerCall:
    """Recall is the control: it was already right.

    Loop A filters by owner on every call and caches nothing, so it
    never had this bug.  These tests are here so that a future change
    that makes recall cache the way the playbook pins would fail.
    """

    def test_recall_filters_by_owner_per_call(self, learned):
        from src.learn import recall
        from src.learn import trajectory

        def _record(session, owner):
            return {"schema": 1, "kind": "task", "session": session,
                    "task": "shared question", "response": "an answer",
                    "owner": owner, "turn": 1}

        for session, owner in (("s1", "alice"), ("s2", "bob")):
            path = os.path.join(learned["learn"]["path"], "trajectories")
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "%s.jsonl" % session), "w") as fh:
                fh.write(json.dumps(_record(session, owner)) + "\n")

        alice = recall.recall("shared question", config_data=learned,
                               owner="alice")
        bob = recall.recall("shared question", config_data=learned, owner="bob")
        assert alice and bob, "both owners must get something"
        assert all("s2" not in ep["id"] for ep in alice)
        assert all("s1" not in ep["id"] for ep in bob)
