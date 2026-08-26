"""
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

The state of one ``chat()`` run.

``OnIt`` is already a typed object with ~50 declared fields, so this is not a
second attempt at one — it is the other half of a distinction that object does
not make.  Its fields are *configuration*: ``web_port``, ``theme``,
``history_turns``, set at startup and stable for the life of the process.  What
was missing is *run* state: how many turns have passed, what has been called,
how much of each budget is spent, what text is carried between turns.  All of
it lived in locals inside ``chat()`` and died on return, which cost three
things — a stalled run could not be asked what it had tried, a decision twenty
turns in could not be tested without driving the preceding twenty, and a
resumed session replayed what was *said* with no record of what was *done*.

**Owned by the caller and mutated in place.**  That is the pattern
``TurnMetrics`` already uses with its ``metrics`` sink, and it is the reason
this is a plain dataclass rather than something that returns copies: the loop
returns from a dozen places, and state assembled on the way out would be
missing from most of them.  A caller that passes one in can read it after
``chat()`` returns — including after the returns that report a failure, which
are the ones worth reading.

**Two lifetimes share one class.**  A ``RunState`` handed to ``chat()`` covers
one task.  The copy persisted next to the session file accumulates across the
tasks of a session, by ``merge()``.  They are the same fields because the
question a resumed session asks — what has this session already tried — is the
same question asked of a longer window.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from ...lib.text import REFERENCE_ONLY
except ImportError:  # imported with src/ itself on sys.path (tests, scripts)
    from lib.text import REFERENCE_ONLY

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# How much tool history survives into the persisted file.  The live list is
# unbounded — the repeated-call detector counts against it and must not have
# its evidence trimmed mid-run — but the on-disk copy is read back to write a
# sentence into a prompt, and a session that has run four hundred tools does
# not need all four hundred to say what it has been doing.
MAX_PERSISTED_CALLS = 200

# Distinct tool names named in the resume note.  Past this it stops being a
# reminder and starts being a list.
MAX_RESUME_TOOLS = 12

# How a run ended.  Recorded because the difference between "answered" and
# "stopped at the turn limit" is invisible in the session file — both leave
# text behind — and it is the first thing a resumed session should know.
STOP_ANSWERED = "answered"
STOP_TURN_LIMIT = "turn_limit"
STOP_REPEATED_TOOL_CALL = "repeated_tool_call"
STOP_PLANNING_EXHAUSTED = "planning_exhausted"
STOP_SAFETY_ABORT = "safety_abort"

_STOP_PHRASES = {
    STOP_ANSWERED: "it finished and answered",
    STOP_TURN_LIMIT: "it hit the turn limit before finishing",
    STOP_REPEATED_TOOL_CALL: "it kept making the same tool call and was stopped",
    STOP_PLANNING_EXHAUSTED: "it described a plan but never called a tool",
    STOP_SAFETY_ABORT: "it was stopped by the user",
}


@dataclass
class RunState:
    """Everything about a run that changes while the run is happening."""

    # ── where the loop is ───────────────────────────────────────────────────
    iteration_count: int = 0
    # (tool_name, args_json) per call, in order.  Appended to in place by
    # ``_execute_tool``, which is also what reads it: identical entries are
    # counted to catch a model stuck on one call, so the pairing of name with
    # arguments is the point — the same tool on five different files is a run
    # making progress, and the same call five times is a run stuck.
    tool_call_history: list = field(default_factory=list)

    # ── budgets, each capped by a MAX_* in chat() ───────────────────────────
    planning_continuation_count: int = 0
    ack_continuation_count: int = 0
    final_continuation_count: int = 0

    # ── the shape of the next API call ──────────────────────────────────────
    force_tool_call: bool = False
    force_compact: bool = False
    # None until the loop starts, because its opening value is ``max_tokens``
    # and this object does not know it.  A caller constructing a mid-run state
    # may set it, and the loop leaves a set value alone.
    active_max_tokens: Optional[int] = None
    last_prompt_tokens: int = 0

    # ── answer text carried across turns ────────────────────────────────────
    # Text from earlier turns that were cut off by the output budget, and the
    # prose written just before a tool call.  Both are answer material that
    # exists nowhere else once the turn that produced it is over.
    final_answer_prefix: str = ""
    prose_before_tools: str = ""

    # ── how it ended, and the session-level totals ──────────────────────────
    stop_reason: str = ""
    # Only meaningful on the persisted copy; see merge().
    task_count: int = 0
    total_turns: int = 0

    # ── tool history ────────────────────────────────────────────────────────

    def tool_counts(self) -> dict:
        """``{tool_name: times called}``, in the order first called."""
        counts: dict = {}
        for entry in self.tool_call_history:
            name = entry[0] if isinstance(entry, (list, tuple)) and entry else str(entry)
            counts[name] = counts.get(name, 0) + 1
        return counts

    # ── what a resumed session is told ──────────────────────────────────────

    def resume_note(self) -> str:
        """One block for the instruction, or "" when there is nothing to say.

        Deliberately a summary and not a transcript.  The session file already
        replays what was said; this answers the question that file cannot —
        what was *done*, and whether the last run got there.  Naming the tools
        and their counts is enough for the model to avoid redoing work without
        paying for the arguments of every call.
        """
        counts = self.tool_counts()
        if not counts and not self.stop_reason:
            return ""
        lines = []
        if counts:
            shown = list(counts.items())[:MAX_RESUME_TOOLS]
            rendered = ", ".join(f"{n}×{c}" if c > 1 else n for n, c in shown)
            if len(counts) > MAX_RESUME_TOOLS:
                rendered += f", and {len(counts) - MAX_RESUME_TOOLS} more"
            lines.append(f"- Tools already run in this session: {rendered}")
        if self.total_turns:
            lines.append(f"- Turns spent so far: {self.total_turns}")
        phrase = _STOP_PHRASES.get(self.stop_reason)
        if phrase and self.stop_reason != STOP_ANSWERED:
            lines.append(f"- The last attempt ended early: {phrase}.")
        if not lines:
            return ""
        return ("\n## Earlier in this session\n"
                + REFERENCE_ONLY + "\n"
                + "\n".join(lines) + "\n"
                "Build on what already worked. Do not repeat it.\n")

    # ── accumulation across the tasks of a session ──────────────────────────

    def merge(self, other: "RunState") -> "RunState":
        """Fold a finished run into this (session-level) state, in place.

        Tool history accumulates and is trimmed from the front; the counters
        describe the most recent run, because a budget is per-run and a sum of
        budgets is not a budget.  The text fields are deliberately not carried:
        a half-written answer belongs to the task that was writing it.
        """
        self.tool_call_history.extend(other.tool_call_history)
        if len(self.tool_call_history) > MAX_PERSISTED_CALLS:
            del self.tool_call_history[:-MAX_PERSISTED_CALLS]
        self.iteration_count = other.iteration_count
        self.planning_continuation_count = other.planning_continuation_count
        self.ack_continuation_count = other.ack_continuation_count
        self.final_continuation_count = other.final_continuation_count
        self.stop_reason = other.stop_reason
        self.total_turns += other.iteration_count
        self.task_count += 1
        return self

    # ── serialization ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """A JSON-safe view.  The text fields stay out: they are answer drafts,
        not run state anyone resumes from, and they are the only fields big
        enough to make this file worth reading twice."""
        return {
            "schema": SCHEMA_VERSION,
            "iteration_count": self.iteration_count,
            "tool_call_history": [list(e) for e in self.tool_call_history],
            "planning_continuation_count": self.planning_continuation_count,
            "ack_continuation_count": self.ack_continuation_count,
            "final_continuation_count": self.final_continuation_count,
            "stop_reason": self.stop_reason,
            "task_count": self.task_count,
            "total_turns": self.total_turns,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RunState":
        """Rebuild from ``to_dict`` output.  Tolerant by construction — a state
        file written by an older build, or half-written by a killed process, is
        a reason to resume with less, never a reason to fail to resume."""
        state = cls()
        if not isinstance(data, dict):
            return state
        for name in ("iteration_count", "planning_continuation_count",
                     "ack_continuation_count", "final_continuation_count",
                     "task_count", "total_turns"):
            try:
                setattr(state, name, int(data.get(name, 0) or 0))
            except (TypeError, ValueError):
                pass
        stop = data.get("stop_reason")
        state.stop_reason = stop if isinstance(stop, str) else ""
        history = data.get("tool_call_history")
        if isinstance(history, list):
            for entry in history:
                # Back to tuples: the live list is counted against tuple keys,
                # and a list read out of JSON would never match one.
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    state.tool_call_history.append((str(entry[0]), str(entry[1])))
                elif isinstance(entry, str):
                    state.tool_call_history.append((entry, ""))
        return state

    # ── the file beside the session ─────────────────────────────────────────

    def save(self, path: str) -> bool:
        """Write to ``path``.  Never raises: a state file that could not be
        written is a session that resumes with less, not a task that failed."""
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # Written whole and moved into place, so a process killed mid-write
            # leaves the previous state rather than a truncated one.
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f)
            os.replace(tmp, path)
            return True
        except Exception as e:  # pragma: no cover - disk full, permissions, …
            logger.debug("run state not saved to %s: %s", path, e)
            return False

    @classmethod
    def load(cls, path: str) -> "RunState":
        """Read from ``path``, or a fresh state when there is nothing to read."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except FileNotFoundError:
            return cls()
        except Exception as e:
            logger.debug("run state not loaded from %s: %s", path, e)
            return cls()


def state_path_for(session_path: str) -> str:
    """The state file beside a session's ``.jsonl``.

    Same directory and same stem, so the two halves of a session — what was
    said and what was done — are found and deleted together.
    """
    if not session_path:
        return ""
    base = session_path[:-6] if session_path.endswith(".jsonl") else session_path
    return f"{base}.state.json"
