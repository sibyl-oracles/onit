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

Loop C, part one: mining.  Finding the fragments worth promoting.

The playbook mines *tactics* -- prose.  This mines *capabilities*: the
ad-hoc fragment the agent writes and runs again and again.  A tactic
you have to remember is a tactic you will forget; a fragment you have
to rewrite is a fragment you will rewrite slightly differently each time,
and the difference is where the bugs live.

**How a fragment is found, without storing the fragment.**  Trajectories
redact argument *values* by default -- the values are where file paths,
search queries and the occasional pasted credential live, and the store
is the agent's own.  But each call also carries an ``args_digest``: the
SHA-256 of the full argument mapping, order-independent.  Two calls with
the same digest made the *same call with the same arguments*, and two
calls with different digests did not.

So the miner groups by ``(tool, args_digest)`` and counts the distinct
*sessions* each group appears in.  It never needs the argument values
to find the repetition, and it never writes them down.  The digest is
the identity; the values stay where they were.

**Why sessions and not calls.**  One session that calls the same thing
twenty times is one agent working; twenty sessions that call the same
thing once each is a pattern the agent has not learned it does not know
it has.  The threshold counts distinct sessions for that reason.

**What a candidate is.**  A ``(tool, args_digest)`` group that appears
in at least ``MIN_SESSIONS`` distinct sessions, with the calls mostly
succeeding.  A group that keeps failing is not a skill -- it is a bug
that recurs, and promoting it would promote the bug.
"""

import collections
import os

from .trajectory import iter_records

# A candidate must appear in at least this many distinct sessions.  The
# RFC says ">= N sessions"; three is the floor at which a pattern is
# more likely to be a pattern than a coincidence of two.
MIN_SESSIONS = 3

# A candidate must have succeeded at least this fraction of the time.
# A fragment that reliably fails is a defect, and the fix for a defect
# is not to make it a tool.
MIN_SUCCESS_RATE = 0.5

# The tools whose calls are worth mining.  A skill is a *capability* the
# agent reaches for, so the candidates are the tools that carry work:
# the shell, the file editors, the network.  The bookkeeping tools
# (context_status, note_read) are the agent talking to itself, and
# promoting one of those to a tool would add schema tokens to save
# schema tokens.
MINABLE_TOOLS = frozenset({
    "bash", "edit_file", "read_file", "write_file", "grep",
    "search_document", "fetch_content", "search", "serve",
    "local_search", "index_documents",
})


def _session_of(record: dict) -> str:
    """The session a record belongs to."""
    return str(record.get("session_id") or "")


def collect_calls(records: list) -> dict:
    """Group the calls in ``records`` by ``(tool, args_digest)``.

    Returns a mapping from ``(tool, digest)`` to the sessions it appeared
    in, how many times it was called, and how many of those calls
    succeeded.

    A call with no digest is skipped: without one there is no way to
    know whether it is the same call as another, and guessing "the same
    tool" is how a miner produces a candidate that is really a family of
    different tasks.
    """
    groups = {}
    for record in records:
        session = _session_of(record)
        for turn in (record.get("trajectory") or []):
            for call in (turn.get("tools") or []):
                tool = call.get("name")
                if not isinstance(tool, str) or not tool:
                    continue
                digest = call.get("args_digest")
                if not digest:
                    continue
                key = (tool, str(digest))
                entry = groups.setdefault(key, {"sessions": set(), "calls": 0,
                                                "errors": 0})
                if session:
                    entry["sessions"].add(session)
                entry["calls"] += 1
                if call.get("ok") is False:
                    entry["errors"] += 1
    return groups


def mine(records: list, *, min_sessions: int = MIN_SESSIONS,
         min_success_rate: float = MIN_SUCCESS_RATE,
         tools: frozenset | None = MINABLE_TOOLS) -> list:
    """The skill candidates in ``records``.

    Each candidate is a dict:

        {"tool", "digest", "sessions", "calls", "errors",
         "success_rate", "arg_keys", "name"}

    Sorted by how many sessions the fragment spans, then by how often it
    succeeded, then by name, so the ordering is deterministic and a
    candidate that moves is a candidate that really moved.

    ``arg_keys`` are the argument *names* of the calls in the group --
    which is what a manifest's input schema is built from.  The values
    are not in the trajectory and are not invented here.
    """
    groups = collect_calls(records)
    candidates = []
    for (tool, digest), entry in groups.items():
        if tools is not None and tool not in tools:
            continue
        sessions = len(entry["sessions"])
        if sessions < min_sessions:
            continue
        calls = entry["calls"]
        errors = entry["errors"]
        success_rate = (calls - errors) / calls if calls else 0.0
        if success_rate < min_success_rate:
            continue
        candidates.append({
            "tool": tool,
            "digest": digest,
            "sessions": sessions,
            "calls": calls,
            "errors": errors,
            "success_rate": success_rate,
            "arg_keys": _arg_keys(records, tool, digest),
        })
    candidates.sort(key=lambda c: (-c["sessions"], -c["success_rate"],
                                   c["tool"], c["digest"]))
    for candidate in candidates:
        candidate["name"] = suggest_name(candidate)

    # Two candidates must never share a name.  They would fight over
    # one directory, and the loser would be the one that got written
    # second -- silently, and with no record of which fragment the
    # surviving skill actually came from.  The digest is in the name
    # for exactly this reason, but a name collision is still possible
    # after truncation, so it is checked rather than assumed.
    seen = {}
    for candidate in candidates:
        name = candidate["name"]
        if name in seen:
            seen[name]["collision"] = True
            candidate["collision"] = True
        else:
            seen[name] = candidate
    return candidates


def _arg_keys(records: list, tool: str, digest: str) -> list:
    """The argument names used by a group.

    The union across the group, sorted.  A parameter that only some of
    the calls used is still a parameter the tool takes -- the calls that
    omitted it relied on its default -- so the union is the right
    answer, not the intersection.
    """
    keys = set()
    for record in records:
        for turn in (record.get("trajectory") or []):
            for call in (turn.get("tools") or []):
                if call.get("name") != tool:
                    continue
                if str(call.get("args_digest")) != digest:
                    continue
                for key in (call.get("arg_keys") or []):
                    keys.add(str(key))
    return sorted(keys)


# The words that make a good tool name are the ones that say what the
# tool *does*, not what it was called during the mining.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at",
    "with", "from", "by", "is", "are", "was", "were", "be", "this", "that",
    "it", "as", "into", "via", "using", "use", "get", "run", "call", "tool",
})


def suggest_name(candidate: dict) -> str:
    """A name for a candidate, from what it is.

    The tool name is the base -- ``bash`` becomes ``bash_...`` -- and
    the distinguishing part comes from the argument names, which are the
    only thing that tells two different fragments of the same tool
    apart.

    The digest goes in the name, and it has to: two fragments of the
    same tool with the same argument *names* are different fragments --
    different URLs, different commands -- and a name that does not
    distinguish them would have two candidates fighting over one
    directory.  The digest is short, stable, and already a function of
    the call, so it is the honest discriminator.

    A name that cannot be derived is not invented.  The caller is told
    the name is provisional, because a skill whose name lies about what
    it does is a skill the model will call for the wrong thing.
    """
    tool = str(candidate.get("tool") or "skill")
    base = "".join(ch if ch.isalnum() else "_" for ch in tool).strip("_")
    base = base.lower() or "skill"

    parts = []
    for key in candidate.get("arg_keys") or []:
        word = "".join(ch if ch.isalnum() else "_" for ch in str(key)).strip("_")
        word = word.lower()
        if not word or word in _STOPWORDS:
            continue
        parts.append(word)

    name = base
    if parts:
        name = base + "_" + "_".join(parts[:2])

    # The discriminator, so two fragments of one tool never share a
    # name.  Eight hex characters is plenty: it only has to separate
    # the handful of candidates a corpus actually produces.
    digest = str(candidate.get("digest") or "")[:8]
    if digest:
        name = name + "_" + digest

    name = name[:48]
    # A tool name is a Python identifier and a filename, so it has to
    # start with a letter and hold nothing but word characters.
    name = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    name = name.strip("_")
    if not name or not name[0].isalpha():
        name = "skill_" + name
    return name[:48]


def mine_store(config_data: dict | None = None, **kwargs) -> list:
    """Mine the recorded corpus for skill candidates."""
    records = list(iter_records(config_data))
    return mine(records, **kwargs)


def format_candidates(candidates: list) -> str:
    """The candidate list, as text.

    Sorted the way :func:`mine` sorts it, so two runs of the command
    print the same thing.
    """
    if not candidates:
        return ("No skill candidates yet.  A candidate is a tool call that "
                "repeats across at least %d sessions and mostly succeeds."
                % MIN_SESSIONS)
    lines = ["Skill candidates", ""]
    lines.append("  %-30s %-12s %8s %8s %8s" %
                 ("tool", "digest", "sessions", "calls", "ok"))
    for candidate in candidates:
        lines.append("  %-30s %-12s %8d %8d %7.0f%%" % (
            candidate["tool"][:30], candidate["digest"][:12],
            candidate["sessions"], candidate["calls"],
            candidate["success_rate"] * 100))
        if candidate.get("arg_keys"):
            lines.append("      args: %s" % ", ".join(candidate["arg_keys"]))
    return "\n".join(lines)
