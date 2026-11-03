"""The read-only slice of the harness a UI is allowed to see.

A front end needs three things from the agent that are not part of a run:
the configuration (to decide whether ratings are recorded, where the
trajectory store lives, which model is configured), the table of fact-checks
still running behind an answer, and the ability to cancel one.  Before this
existed, ``onit/ui/api.py`` reached for all three with ``getattr`` on the
agent object \u2014 six call sites, each one a guess at an attribute name that
no test would catch if the harness renamed it.

``UISettings`` turns those guesses into a declared surface.  The harness
builds one and hands it over; a front end reads it and never touches the
agent.  Rename a field here and every front end fails loudly at the call
site instead of silently reading ``None``.

Deliberately narrow.  This is not a proxy for the agent: it exposes what the
web UI demonstrably uses today and nothing more.  A new front end that wants
more should add a field here, where the change is visible on both sides of
the boundary, rather than reach around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class UISettings:
    """What a front end may read from the harness, and nothing else.

    ``config_data`` is the parsed configuration mapping (``onit.py``'s
    ``config_data`` field).  It is handed over as a read-only view: the
    ``onit.learn`` helpers that consume it only read, and a UI that mutated
    it would be changing the harness's behaviour from outside.

    ``background_checks`` maps a session id to the ``asyncio.Task`` running
    that session's fact-check.  A UI reads it to answer "is a correction
    still coming?" so it can keep a stream open.  The mapping is the live
    one the harness owns \u2014 copying it would answer the question with stale
    data \u2014 so it is typed as a plain mapping and must not be mutated.

    ``cancel_background_check`` is the harness method that ends one, or
    ``None`` when the harness has no such capability.  A UI calls it through
    :meth:`cancel_check`, which tolerates the ``None`` case so callers do
    not each repeat the check.
    """

    config_data: Mapping[str, Any] = field(default_factory=dict)
    background_checks: Mapping[str, Any] = field(default_factory=dict)
    cancel_background_check: Callable[[str], None] | None = None

    def cancel_check(self, session_id: str) -> bool:
        """End this session's background fact-check.  True if one was asked to stop.

        Returns False when the harness has no cancel hook at all, so a caller
        can distinguish "nothing to cancel" from "cancelled".
        """
        if self.cancel_background_check is None:
            return False
        self.cancel_background_check(session_id)
        return True

    def check_running(self, session_id: str) -> bool:
        """Whether a fact-check is still running behind this session's answer."""
        task = self.background_checks.get(session_id)
        return bool(task and not task.done())
