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

Tool lifecycle triggers, as policy instead of vibes.

The dynamic-tool loop needs three decisions made repeatedly and made the same
way every time: when a recurring workaround has earned a *proposal*, when a
live tool has misbehaved enough to earn an *update*, and when a tool has
stopped earning its schema tokens and should be *archived*.  This module is
those decisions, written down as pure functions over the event store and the
trajectory store — no I/O of their own, no model calls, nothing to mock but
the records.

Every threshold here is a default the operator can override, and every
function returns *a proposal*, never an action.  Loading, updating and
archiving stay behind the same gate as any other live-code event
(docs/SELF_IMPROVEMENT_GAPS.md gap #7): the store measures, this reads the
store, and the gate decides.  An agent that could grant itself these
transitions would be the agent grading its own homework.

Creation evidence comes from recurrence; retention evidence comes from the
holdout.  Recent usage — the signal that drifts narrow — is deliberately not
the ruler for keeping a tool.
"""

from collections import defaultdict

from .config import recording_enabled
from .events import iter_events
from .report import summarize
from .trajectory import iter_records

# --- creation: recurrence is the only honest evidence -----------------------
# A single episode proves a task existed once.  Three recurrences across two
# distinct sessions is the smallest pattern that is probably not a coincidence
# — and "across sessions" is what stops a tool being minted from one lucky run.
CREATION_MIN_OCCURRENCES = 3
CREATION_MIN_SESSIONS = 2

# --- update: measured misbehavior, not complaint ----------------------------
UPDATE_FAILURE_RATE = 0.30
UPDATE_MIN_CALLS = 10

# --- deletion: non-earnings, and always soft --------------------------------
ARCHIVE_UNUSED_EPISODES = 20
ARCHIVE_GRACE_EPISODES = 10
# A tool whose failures track task failure is worse than useless: it pollutes
# planning.  Negative correlation retires it early.
ARCHIVE_NEGATIVE_VALUE_CALLS = 5


def episodes_since(ts: str, config_data: dict | None = None) -> int:
    """How many tasks have completed since ``ts``.

    Episodes, not days: a weekend of no use should not archive a tool that
    worked every weekday, and a benchmark burst should not immortalize one
    that never earned its keep.  Counts every recorded task record after the
    timestamp, which is the same unit the creation thresholds are denominated
    in.
    """
    count = 0
    for record in iter_records(config_data):
        if str(record.get("ts", "")) > ts:
            count += 1
    return count


def _tool_calls(config_data: dict | None = None) -> dict:
    """Per-tool call/error counts across every recorded trajectory."""
    stats = defaultdict(lambda: {"calls": 0, "errors": 0})
    for record in iter_records(config_data):
        for turn in record.get("trajectory") or []:
            for call in turn.get("tools") or []:
                name = call.get("name")
                if not name:
                    continue
                stats[name]["calls"] += 1
                if call.get("ok") is False:
                    stats[name]["errors"] += 1
    return stats


def should_propose_creation(*, occurrences: int, sessions: int,
                            category_share: float | None = None,
                            category_cap: float = 0.40,
                            config_data: dict | None = None) -> dict:
    """Whether a recurring pattern has earned a tool *proposal*.

    ``occurrences``/``sessions`` come from clustering the bash/run_code
    fragments in the trajectory store (Loop C's mining step).  The category
    cap is the narrow-optimization defense: when one category already holds
    more than ``category_cap`` of the dynamic toolset, same-category
    proposals are blocked — a week of PDF tasks must not bequeath fifteen
    PDF tools that every other task pays for in schema tokens.

    Returns a decision dict, not a bool, so the caller can log *why*.
    """
    reasons = []
    if occurrences < CREATION_MIN_OCCURRENCES:
        reasons.append(f"only {occurrences} occurrence(s), "
                       f"need {CREATION_MIN_OCCURRENCES}")
    if sessions < CREATION_MIN_SESSIONS:
        reasons.append(f"seen in {sessions} session(s), "
                       f"need {CREATION_MIN_SESSIONS}")
    if category_share is not None and category_share > category_cap:
        reasons.append(f"category at {category_share:.0%} of dynamic toolset, "
                       f"cap {category_cap:.0%}")
    return {"propose": not reasons, "reasons": reasons}


def should_update(*, calls: int, errors: int,
                  config_data: dict | None = None) -> dict:
    """Whether a live tool's failure rate has earned an update pass.

    A tool that fails more than ``UPDATE_FAILURE_RATE`` of its last
    ``UPDATE_MIN_CALLS`` calls is not unlucky — its description, schema or
    code no longer match what the model asks of it.  Below the call floor the
    rate is noise, and a tool called twice and failed once must not trigger a
    rewrite loop.
    """
    if calls < UPDATE_MIN_CALLS:
        return {"update": False,
                "reasons": [f"{calls} call(s), need {UPDATE_MIN_CALLS}"]}
    rate = errors / max(calls, 1)
    return {"update": rate > UPDATE_FAILURE_RATE,
            "reasons": [] if rate > UPDATE_FAILURE_RATE
            else [f"failure rate {rate:.0%} within tolerance"]}


def should_archive(*, tool: str, last_used_ts: str | None,
                   calls: int, errors: int,
                   config_data: dict | None = None) -> dict:
    """Whether a tool has stopped earning its place — soft deletion only.

    Three independent grounds, checked in order:

    1. **Disuse** — zero calls across ``ARCHIVE_UNUSED_EPISODES`` episodes,
       with a grace window so a tool mid-experiment is not yanked.
    2. **Negative value** — enough calls to judge and a failure rate so high
       that selecting the tool correlates with the task going badly.
    3. **Supersession** — left to the caller, who knows a successor landed;
       the event store records it when they do.

    An archive here is an *archive*: the code stays on disk under the
    tool store's archive/ directory, restorable by a human or by grace.  The
    agent can never permanently destroy capability — that invariant is the
    whole reason deletion is safe to automate at all.
    """
    reasons = []
    if last_used_ts is None:
        if calls == 0:
            reasons.append("never called")
    else:
        idle = episodes_since(last_used_ts, config_data)
        if idle > ARCHIVE_UNUSED_EPISODES + ARCHIVE_GRACE_EPISODES:
            reasons.append(f"unused for {idle} episodes "
                           f"(threshold {ARCHIVE_UNUSED_EPISODES}, "
                           f"grace {ARCHIVE_GRACE_EPISODES})")
    if calls >= ARCHIVE_NEGATIVE_VALUE_CALLS:
        rate = errors / max(calls, 1)
        if rate > UPDATE_FAILURE_RATE * 2:
            reasons.append(f"failure rate {rate:.0%} tracks task failure")
    return {"archive": bool(reasons), "reasons": reasons}


def pending_proposals(config_data: dict | None = None) -> list[dict]:
    """What the event store says is due, across every live tool.

    The gardener pass's input: one batched read at session end, never in the
    hot path.  Counters update per call; decisions happen here.
    """
    if not recording_enabled(config_data):
        return []
    proposals: list[dict] = []
    stats = _tool_calls(config_data)
    live = {s for s in stats if s}
    for tool, stat in sorted(stats.items()):
        verdict = should_update(calls=stat["calls"], errors=stat["errors"])
        if verdict["update"]:
            proposals.append({"tool": tool, "action": "update",
                              "reasons": verdict["reasons"] or
                              [f"failure rate above "
                               f"{UPDATE_FAILURE_RATE:.0%}"]})
    # Tools that were once loaded but never called at all are the purest
    # schema-token waste; they surface here even though they have no entry
    # in the call stats.
    for record in iter_events(config_data, event="tool.loaded"):
        subject = record.get("subject")
        if subject and subject not in live:
            proposals.append({"tool": subject, "action": "review",
                              "reasons": ["loaded but never called"]})
    return proposals