'''
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Human-in-the-loop approvals for commands the policy will not decide alone.

When the policy returns ASK, the server does not run the command and does not
refuse it either: it mints a single-use ticket bound to that exact command and
that exact session, hands the ticket back to the agent harness, and waits to
be called again. The harness shows a human the command, and only if they say
yes does it re-issue the call carrying the ticket.

Three properties do the security work here, and none of them may be relaxed:

1.  **A ticket is bound to a command and a session.** Redeeming it for any
    other command, or from any other session directory, fails. An approval is
    never a general-purpose capability.

2.  **The model cannot mint or forge one.** The ticket is a 128-bit secret
    the server generates; the harness overwrites the ``approval_token``
    argument on every call (the same trust boundary that already makes
    ``data_path`` unspoofable), so a value the model invents never arrives.

3.  **Only the discretionary layer is askable.** The caller decides which
    refusals may become questions — an unlisted executable, an unpinned
    install. Catastrophic operations, an operator's explicit deny rules, and,
    on a multi-tenant deployment, the session path jail are refused outright
    and never reach this module. A human can only ever be asked to take a
    risk that is theirs to take: on a shared host the person answering the
    prompt is not the only person the answer affects.

Grants and denials live in this process only. Nothing is persisted: an
approval that outlived the session that asked for it would be a capability
nobody remembers granting.
'''

import hashlib
import os
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

# A ticket is useless after this long. Long enough for someone to read the
# command and think about it; short enough that an unanswered prompt from an
# abandoned session is not left standing.
#
# It must outlive the harness's own prompt window (chat.APPROVAL_TIMEOUT) with
# room to spare, and that is the whole reason it is not 300. Set equal to it,
# every answer given near the end of the window arrived at a ticket that had
# just expired: the person said yes, the server had already forgotten the
# question and asked it again, and the harness reported the command blocked
# right after they approved it. The margin is what keeps a slow answer — the
# common one, since reading a long command takes a minute — from landing on a
# ticket that is no longer there.
PENDING_TTL = 900.0

# Bounds on in-memory state, so a session that asks endlessly cannot grow the
# server. Both are far above any real interactive rate.
MAX_PENDING = 64
MAX_SESSIONS = 256
MAX_GRANTS_PER_SESSION = 128

# Scopes a human can choose.
ONCE = "once"
SESSION = "session"
DENY = "deny"


def command_digest(command: str) -> str:
    return hashlib.sha256((command or "").encode("utf-8", "replace")).hexdigest()


@dataclass
class _Pending:
    token: str
    digest: str
    base: str
    subjects: tuple
    reason: str
    created: float


@dataclass
class _Session:
    """What one session's human has already agreed to."""
    commands: set = field(default_factory=set)   # executables, "install:pip"
    roots: set = field(default_factory=set)      # extra filesystem roots
    denied: set = field(default_factory=set)     # digests of refused commands
    touched: float = field(default_factory=time.monotonic)


class ApprovalBroker:
    """Tickets, session grants, and the memory of what a human said no to."""

    def __init__(self) -> None:
        self._pending: dict[str, _Pending] = {}
        self._sessions: dict[str, _Session] = {}

    # ── requests ────────────────────────────────────────────────────────
    def request(self, command: str, base: str, subjects: tuple,
                reason: str) -> dict:
        """Mint a ticket for ``command`` and return the payload for the agent.

        The payload is what the model will read, so it names the command and
        what is being asked about it, and says plainly that the call did not
        run. It does not tell the model to re-issue the call — the harness
        does that, or nobody does.
        """
        self._expire()
        if len(self._pending) >= MAX_PENDING:
            # Drop the oldest rather than refuse: a backlog this size means
            # tickets are being minted and abandoned, not answered.
            oldest = min(self._pending.values(), key=lambda p: p.created)
            self._pending.pop(oldest.token, None)
        token = secrets.token_urlsafe(16)
        self._pending[token] = _Pending(
            token=token, digest=command_digest(command), base=base,
            subjects=tuple(subjects), reason=reason, created=time.monotonic(),
        )
        return {
            "status": "needs_approval",
            "approval_id": token,
            "command": command,
            "reason": reason,
            "subjects": list(subjects),
            "scopes": [ONCE, SESSION],
            "expires_in": int(PENDING_TTL),
            "note": ("The command was not run. It needs a person's approval. "
                     "Wait for the result of this call rather than retrying, "
                     "and do not attempt another route around the policy."),
        }

    # ── redemption ──────────────────────────────────────────────────────
    def redeem(self, token: str, command: str, base: str,
               scope: str = ONCE) -> bool:
        """Consume a ticket. True when it really was issued for this command.

        Single use in every outcome: a ticket that fails validation is still
        burned, so a wrong guess costs the caller the ticket rather than
        giving them another try at it.
        """
        self._expire()
        pending = self._pending.pop(token or "", None)
        if pending is None:
            return False
        if pending.digest != command_digest(command):
            return False
        if pending.base != base:
            return False
        if scope == SESSION:
            self._grant_session(base, pending.subjects)
        return True

    def refuse(self, token: str, command: str, base: str) -> None:
        """Record that a human said no, and drop the ticket.

        Remembered so that the same command coming back around is not a fresh
        question but a refusal — and one the server counts, because a model
        re-submitting a command a person has already rejected is the pattern
        auto-containment exists for.
        """
        self._pending.pop(token or "", None)
        session = self._session(base)
        if len(session.denied) < MAX_GRANTS_PER_SESSION:
            session.denied.add(command_digest(command))

    def was_refused(self, command: str, base: str) -> bool:
        session = self._sessions.get(base)
        return bool(session and command_digest(command) in session.denied)

    # ── standing grants ─────────────────────────────────────────────────
    def _grant_session(self, base: str, subjects: tuple) -> None:
        session = self._session(base)
        for subject in subjects:
            if len(session.commands) + len(session.roots) >= MAX_GRANTS_PER_SESSION:
                break
            if subject.startswith("path:"):
                session.roots.add(subject[5:])
            else:
                session.commands.add(subject)

    def commands_for(self, base: str) -> frozenset:
        """Executables this session's human has approved for the session.

        Only the bare ones. A subject carrying a prefix — ``install:pip``,
        ``pattern:ps command``, ``rule:make`` — is answered by its own lookup
        (``installs_allowed``, ``granted``); letting one through here would
        put a string no executable is named after into the allowlist, where it
        would sit harmlessly and confuse the next person to read it.
        """
        session = self._sessions.get(base)
        if not session:
            return frozenset()
        return frozenset(c for c in session.commands if ":" not in c)

    def installs_allowed(self, base: str) -> bool:
        session = self._sessions.get(base)
        return bool(session and any(c.startswith("install:")
                                    for c in session.commands))

    def roots_for(self, base: str) -> frozenset:
        session = self._sessions.get(base)
        return frozenset(session.roots) if session else frozenset()

    def granted(self, base: str, subject: str) -> bool:
        """Whether this exact subject is already approved for the session.

        Used by the checks whose subject is not an executable — a blocked
        pattern, a settings rule — so that re-evaluating an approved command
        does not ask the same question again and loop.
        """
        session = self._sessions.get(base)
        if not session:
            return False
        if subject.startswith("path:"):
            target = subject[5:]
            return any(target == root or target.startswith(root.rstrip("/") + "/")
                       for root in session.roots)
        return subject in session.commands

    @contextmanager
    def transient(self, base: str, subjects: tuple):
        """Hold ``subjects`` for the length of one re-evaluation.

        A "once" approval still has to survive being looked at a second time:
        the command is re-checked after the ticket is redeemed, so that a
        settings file edited between minting and redeeming is obeyed rather
        than bypassed. Without this the re-check would ask the same question
        again, forever. Only the subjects the ticket actually named are held,
        and only the ones not already granted are taken back afterwards.
        """
        session = self._session(base)
        added_commands, added_roots = set(), set()
        for subject in subjects:
            if subject.startswith("path:"):
                root = subject[5:]
                if root not in session.roots:
                    session.roots.add(root)
                    added_roots.add(root)
            elif subject not in session.commands:
                session.commands.add(subject)
                added_commands.add(subject)
        try:
            yield
        finally:
            session.commands -= added_commands
            session.roots -= added_roots

    # ── housekeeping ────────────────────────────────────────────────────
    def _session(self, base: str) -> _Session:
        session = self._sessions.get(base)
        if session is None:
            if len(self._sessions) >= MAX_SESSIONS:
                stalest = min(self._sessions.items(), key=lambda kv: kv[1].touched)
                self._sessions.pop(stalest[0], None)
            session = _Session()
            self._sessions[base] = session
        session.touched = time.monotonic()
        return session

    def _expire(self) -> None:
        now = time.monotonic()
        for token, pending in list(self._pending.items()):
            if now - pending.created > PENDING_TTL:
                del self._pending[token]

    def reset(self) -> None:
        """Forget everything — for tests, and for a session being torn down."""
        self._pending.clear()
        self._sessions.clear()


BROKER = ApprovalBroker()


def approval_channel_available() -> bool:
    """Whether this deployment has a human it can actually ask.

    Declared by the harness process before it spawns the tool servers, which
    inherit the variable. Absent — a cron run, an A2A server, a gateway bot,
    a directly-launched MCP server — every ask collapses to the refusal it
    would have been before approvals existed. That is the fail-closed default,
    and it is the reason nothing here needs a timeout policy of its own: with
    no channel, no ticket is ever minted.
    """
    return os.environ.get("ONIT_APPROVAL_CHANNEL") == "1"


def ask_scope() -> frozenset:
    """Which refusal classes may become questions in this deployment.

    ``ONIT_ASK_APPROVAL=0`` turns asking off for an operator who wants the
    old binary behaviour on a machine that does have a human attached.
    """
    if not approval_channel_available():
        return frozenset()
    if os.environ.get("ONIT_ASK_APPROVAL") == "0":
        return frozenset()
    scope = {"command", "install"}
    if not multi_tenant():
        # Reaching outside the session directory is askable only where the
        # person answering owns everything they could reach. On the web UI one
        # process serves many logged-in people under one OS account, so the
        # jail is the only thing keeping their sessions apart and no user of
        # it is entitled to waive it. On a terminal or a single-user host the
        # human at the prompt already owns those files, and this is where most
        # of the friction is: an absolute path outside the jail is the most
        # common refusal there and almost always a benign one.
        scope.add("path")
    return frozenset(scope)


def multi_tenant() -> bool:
    """True when one server process serves several distinct people.

    The web UI authenticates many users into one process running as one OS
    account; the terminal UI is one person at their own machine.
    """
    return os.environ.get("ONIT_WEB_UI") == "1"
