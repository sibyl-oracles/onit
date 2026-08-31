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

The event log: what the learning loops did, kept beside what the runs did.

Trajectories record what a task did.  This records what the *loops* did about
it — a tool registry loading, a skill being proposed, a playbook bullet
promoted, a tool archived.  Both land in the same per-session JSONL file on
purpose: ``docs/SELF_IMPROVEMENT.md`` §4.4 wants one event log, two loops, not
two parallel logging systems that would inevitably disagree about what
happened.  A tool's lifecycle can then be read next to the trajectories that
justify it, and a trajectory can name the tool state that was live when it ran.

The first consumer is tooling (the lifecycle events below); the second is Loop
A's episodic recall, which reads the ``task`` records this same file already
carries.  Neither consumer changes what the other writes.

Events are best-effort by construction, exactly like trajectories: a failed
event write must never turn a working loop into a broken one.
"""

import json
import logging
import os

from .config import recording_enabled, trajectory_dir
from .trajectory import (_UNSAFE_NAME, KIND_EVENT, _now, session_file)

logger = logging.getLogger(__name__)

# Events this layer knows are written.  Open on purpose: ``record_event``
# accepts any string, because the loops that come later will name events this
# module never anticipated, and a store that refused them would be a store the
# future has to fork.  The known set exists for readers and for the summary.
TOOL_EVENTS = (
    "tool.registry_loaded",   # a session discovered its toolset
    "tool.proposed",          # a candidate tool reached the gate
    "tool.loaded",            # the gate passed; the tool went live
    "tool.rejected",          # the gate refused it
    "tool.updated",           # a live tool's code or manifest changed
    "tool.reloaded",          # a server re-read its toolset
    "tool.archived",          # soft deletion — restorable, never destroyed
    "tool.restored",          # an archive was brought back
)


def _safe_session(session_id: str | None) -> str:
    """A session id reduced to the same safe name trajectories use."""
    return _UNSAFE_NAME.sub("_", str(session_id or "unknown"))[:96] or "unknown"


def record_event(*, event: str, subject: str | None = None,
                 session_id: str | None = None,
                 source: str = "loop",
                 detail: dict | None = None,
                 config_data: dict | None = None) -> str | None:
    """Append one lifecycle event to the session's event log.

    Returns the path written to, or None when recording is off or the write
    failed.  Never raises: the loops calling this are improvements to a working
    agent, and a bookkeeping failure must not become a task failure.

    ``session_id`` may be empty for events that belong to no conversation — a
    nightly gardener pass, a CLI-driven promotion.  Those land in a file of
    their own (``loop.jsonl``) rather than being invented into a session that
    never had them.
    """
    if not recording_enabled(config_data):
        return None
    record = {
        "schema": 1,
        "kind": KIND_EVENT,
        "event": str(event),
        "subject": str(subject) if subject else None,
        "ts": _now(),
        "source": str(source),
        "detail": detail or {},
    }
    sid = _safe_session(session_id) if session_id else "loop"
    path = session_file(sid, config_data)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return path
    except Exception as e:  # pragma: no cover - disk full, permissions, …
        logger.debug("event write failed for %s: %s", path, e)
        return None


def record_tool_event(*, event: str, tool: str, session_id: str | None = None,
                      source: str = "loop", detail: dict | None = None,
                      config_data: dict | None = None) -> str | None:
    """The tool-lifecycle half of the contract.

    A thin, named wrapper so the future dynamic loader (Loop C's serving half)
    has one obvious call to make — and so a reader grepping for how tools
    change lands here rather than on the generic function.
    """
    return record_event(event=event, subject=tool, session_id=session_id,
                        source=source, detail=detail, config_data=config_data)


def iter_events(config_data: dict | None = None, *,
                event: str | None = None, subject: str | None = None,
                source: str | None = None):
    """Every event on disk, oldest first, optionally filtered.

    Yields the raw records.  ``event`` matches exactly; ``subject`` matches the
    tool, skill or bullet the event was about.
    """
    directory = trajectory_dir(config_data)
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue  # a torn line loses one event, not the file
                    if record.get("kind") != KIND_EVENT:
                        continue
                    if event is not None and record.get("event") != event:
                        continue
                    if subject is not None and record.get("subject") != subject:
                        continue
                    if source is not None and record.get("source") != source:
                        continue
                    yield record
        except OSError as e:  # pragma: no cover - permissions
            logger.debug("event read failed for %s: %s", path, e)


def tool_timeline(tool: str, config_data: dict | None = None) -> list[dict]:
    """One tool's full lifecycle, in order: proposed, loaded, updated, archived.

    The record a deletion decision is made from, and the audit trail a
    restoration reads.  Soft deletion only — an archive event is the end of
    serving, never the end of the record.
    """
    return list(iter_events(config_data, subject=tool))


def summarize_events(config_data: dict | None = None) -> dict:
    """Counts worth printing: which events happened, to what, how often."""
    counts: dict[str, int] = {}
    subjects: dict[str, set] = {}
    total = 0
    for record in iter_events(config_data):
        total += 1
        name = record.get("event", "?")
        counts[name] = counts.get(name, 0) + 1
        if record.get("subject"):
            subjects.setdefault(name, set()).add(record["subject"])
    return {
        "total": total,
        "by_event": dict(sorted(counts.items())),
        "subjects": {k: len(v) for k, v in sorted(subjects.items())},
    }