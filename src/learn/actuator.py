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

The actuator: the thing that makes ``lifecycle.py``'s policy real.

``lifecycle.py`` was 178 lines of decisions with zero production
callers.  It was correct as policy and inert as software, and inert
policy that reads as done is worse than no policy at all: a reader
skimming the tree concludes the tool-lifecycle loop exists.

This module is the other half of the pair.  The policy says *whether*;
this says *what to do about it*.

**What an actuator can and cannot do.**  The RFC's rule is that the
agent can never permanently destroy capability, and that a decision
about what the agent may do with itself is not the agent's to make
(§4.0's invariants).  So the actuator:

* **may** withhold a tool from a session's registry — the tool stays on
  disk, the code stays loadable, and the next session gets it back;
* **may** record the decision in the event log, so the log says *why*;
* **may not** delete anything, rewrite a manifest, or change what a
  *different* session can use.

Withholding is reversible by construction, which is what makes
automating it safe at all.  The archive file under
``~/.onit/learned/archive/`` is the durable record of what was withheld
and when, and ``onit learn --lifecycle`` shows it.

**Why the actuator is opt-in.**  Withholding a tool is a real change to
what the agent can do, and a deployment that has not asked for it should
not get it.  The actuator runs at autonomy 2 (``adapt``) and above, and
only when ``learn.lifecycle.enabled`` is on.  Below that, the policy is
computed and reported but nothing is withheld.
"""

import json
import os
import time

from .config import recording_enabled, trajectory_dir
from .lifecycle import (pending_proposals, should_archive, should_update)
from .events import record_tool_event

# The archive file: what was withheld, when, and why.  It is a record,
# not a mechanism — the withholding itself is a registry decision made
# at startup, and the archive is what makes that decision auditable.
ARCHIVE_NAME = "archive.json"

# How many times the same tool may be withheld before the actuator
# stops proposing it.  A tool that keeps coming back was needed; a
# threshold that lets the same tool be withheld forty times in a row
# is a policy that is not learning.
MAX_WITHHOLDINGS = 3

# The tools that are never withheld, whatever the policy says.  These
# are the ones whose absence would make the agent unable to recover:
# the ones that read and write files, and the ones that report what
# happened.  Withholding the ability to write a note would make the
# agent unable to record why it was wrong.
NEVER_WITHHOLD = frozenset({
    "read_file", "edit_file", "write_file", "list_directory",
    "grep", "bash", "result_read", "result_grep", "context_status",
    "note_read", "note_write",
})


def archive_path(config_data: dict | None = None) -> str:
    """Where the archive of withheld tools lives."""
    return os.path.join(os.path.dirname(trajectory_dir(config_data)),
                        "archive.json")


def read_archive(config_data: dict | None = None) -> dict:
    """The archive of withheld tools.

    A missing or broken file reads as empty: an unreadable archive
    means "nothing is withheld", which is the safe direction.
    """
    try:
        with open(archive_path(config_data), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def write_archive(archive: dict, config_data: dict | None = None) -> bool:
    """Write the archive.  Best-effort."""
    try:
        path = archive_path(config_data)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(archive, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        return False
    return True


def withheld_tools(config_data: dict | None = None) -> set:
    """The tools currently withheld from sessions."""
    archive = read_archive(config_data)
    return {name for name, entry in (archive.get("tools") or {}).items()
            if isinstance(entry, dict) and entry.get("withheld")}


def lifecycle_enabled(config_data: dict | None = None) -> bool:
    """Whether the actuator may withhold tools at all.

    Off unless asked for, and off below autonomy 2.  The policy is
    still *computed* when it is off — ``onit learn --lifecycle`` reports
    what it would do — but nothing is withheld.
    """
    from .config import ADAPT, autonomy
    learn = (config_data or {}).get("learn")
    if not isinstance(learn, dict):
        return False
    lifecycle = learn.get("lifecycle")
    if not isinstance(lifecycle, dict) or not lifecycle.get("enabled"):
        return False
    return autonomy(config_data) >= ADAPT


def tool_stats(config_data: dict | None = None) -> dict:
    """Per-tool call and error counts, from the recorded trajectories."""
    from .lifecycle import _tool_calls
    return dict(_tool_calls(config_data))


def evaluate(config_data: dict | None = None) -> list:
    """The decisions the policy reaches, and what it would do about them.

    Returns a list of ``{"tool", "action", "reasons"}`` dicts.  This is
    the report half: it is computed whether or not the actuator is
    enabled, so a deployment can see what the policy thinks before
    letting it act.
    """
    if not recording_enabled(config_data):
        return []
    proposals = pending_proposals(config_data)
    stats = tool_stats(config_data)
    decisions = []
    for proposal in proposals:
        tool = proposal.get("tool")
        if not tool:
            continue
        if tool in NEVER_WITHHOLD:
            continue
        stat = stats.get(tool) or {"calls": 0, "errors": 0}
        last_used = None
        archive = read_archive(config_data)
        entry = (archive.get("tools") or {}).get(tool) or {}
        last_used = entry.get("last_used")
        verdict = should_archive(tool=tool, last_used_ts=last_used,
                                 calls=stat["calls"], errors=stat["errors"],
                                 config_data=config_data)
        if verdict.get("archive"):
            decisions.append({
                "tool": tool,
                "action": "withhold",
                "reasons": verdict.get("reasons") or [],
            })
        else:
            update = should_update(calls=stat["calls"], errors=stat["errors"],
                                   config_data=config_data)
            if update.get("update"):
                decisions.append({
                    "tool": tool,
                    "action": "update",
                    "reasons": update.get("reasons") or [],
                })
    return decisions


def apply(config_data: dict | None = None, *,
          session_id: str | None = None) -> dict:
    """Run the actuator: withhold the tools the policy says to withhold.

    Returns ``{"withheld": [...], "decisions": [...], "applied": bool}``.

    The write is to the archive and the event log.  The *withholding
    itself* takes effect at the next session's registry build, which
    reads the archive — a session that is already running is not
    robbed of a tool mid-task, because the tool it is calling is
    already in its registry and the model has already been told about
    it in the prompt.
    """
    decisions = evaluate(config_data)
    if not decisions:
        return {"withheld": [], "decisions": [], "applied": False}
    if not lifecycle_enabled(config_data):
        return {"withheld": [], "decisions": decisions, "applied": False}

    archive = read_archive(config_data)
    tools = archive.get("tools") or {}
    withheld = []
    for decision in decisions:
        if decision.get("action") != "withhold":
            continue
        tool = decision["tool"]
        entry = tools.get(tool) or {}
        count = int(entry.get("count") or 0)
        if count >= MAX_WITHHOLDINGS:
            # The tool keeps coming back.  A policy that is not learning
            # is a policy that is wrong, and the fix is to stop acting on
            # it rather than to keep acting.
            continue
        tools[tool] = {
            "withheld": True,
            "count": count + 1,
            "since": entry.get("since") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reasons": decision.get("reasons") or [],
        }
        withheld.append(tool)
        try:
            record_tool_event(event="tool.archived", tool=tool,
                              session_id=session_id,
                              config_data=config_data)
        except Exception:
            # The event is a record of the decision; failing to write it
            # must not fail the decision.
            pass
    if withheld:
        archive["tools"] = tools
        write_archive(archive, config_data)
    return {"withheld": withheld, "decisions": decisions,
            "applied": bool(withheld)}


def restore(tool: str, config_data: dict | None = None,
            session_id: str | None = None) -> bool:
    """Bring a withheld tool back.

    The reverse operation, and the reason the archive is not a deletion:
    a tool that was withheld can always be restored, by a person or by
    the next policy pass.
    """
    archive = read_archive(config_data)
    tools = archive.get("tools") or {}
    entry = tools.get(tool)
    if not entry:
        return False
    entry["withheld"] = False
    entry["restored"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tools[tool] = entry
    archive["tools"] = tools
    written = write_archive(archive, config_data)
    if written:
        try:
            record_tool_event(event="tool.restored", tool=tool,
                              session_id=session_id,
                              config_data=config_data)
        except Exception:
            pass
    return written


def filter_tools(tools, config_data: dict | None = None) -> list:
    """Drop the withheld tools from a list of tool names.

    The one place the archive reaches the running agent.  It is a
    filter on a *list*, not a deletion: the tool is still registered,
    still on disk, and still reachable by name at the next session
    unless the policy says otherwise.
    """
    withheld = withheld_tools(config_data)
    if not withheld:
        return list(tools)
    return [name for name in tools if name not in withheld]


def format_report(decisions: list, archive: dict | None = None) -> str:
    """The lifecycle report, as one screen of text."""
    archive = archive if archive is not None else {}
    tools = archive.get("tools") or {}
    withheld = sorted(name for name, entry in tools.items()
                      if isinstance(entry, dict) and entry.get("withheld"))
    lines = ["Tool lifecycle"]
    if not decisions and not withheld:
        lines.append("  Nothing is due. No tool has failed often enough to")
        lines.append("  earn an update, and none has gone unused long enough")
        lines.append("  to earn an archive.")
        return "\n".join(lines)
    if decisions:
        lines.append("  %d decision(s) due:" % len(decisions))
        for decision in decisions:
            reasons = "; ".join(decision.get("reasons") or [])
            lines.append("    %-8s %-20s %s"
                         % (decision.get("action", ""),
                            decision.get("tool", ""), reasons))
    if withheld:
        lines.append("  Withheld: %s" % ", ".join(withheld))
        lines.append("  (reversible — 'onit learn --lifecycle --restore TOOL')")
    return "\n".join(lines)
