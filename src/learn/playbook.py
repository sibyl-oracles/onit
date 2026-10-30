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

Loop B: the playbook.  Durable tactics, distilled from what happened.

This is the main event of ``docs/SELF_IMPROVEMENT.md``.  Layer 0 records
what happened; Loop A recalls the last time this came up.  This is the
third thing: the *generalisation*.  Not "last time you asked this, the
search was slow" but "when a search returns a file whose name matches
and whose excerpts do not, open it anyway — the opening is a poor proxy
for the tail".

**What a bullet is.**  One itemized tactic, with a trigger, a
provenance, and counters.  The schema is the RFC's:

    {"id": "pb-0142", "scope": "global|owner:<hash>|topic:<slug>",
     "text": "When local_search returns a file whose *name* matches ...",
     "trigger": "local_search",
     "born": "2026-08-01", "last_used": "2026-08-03",
     "used": 41, "helped": 12, "hurt": 1,
     "status": "active|probation|deprecated",
     "provenance": ["traj:sess-a1#3", "traj:sess-b7#1"]}

**Delta-only (I2).**  The curator applies *operations* — insert, update,
deprecate — to identified items.  It never regenerates the file.  The
reason is context collapse (ACE): a wholesale rewrite erodes the
knowledge that was there, because the rewrite is written by a model that
saw a *sample* of the history and did not see the item it dropped.  A
delta names the item it touched, so a dropped item is a decision rather
than an oversight.

**Token-capped.**  The whole point of the playbook is to be *read*, and
everything in the prompt is paid for on every request.  The budget is
hard, and a bullet that does not fit is not injected — the same rule the
recall path follows, for the same reason: a truncated list is
indistinguishable from a complete one.

**Version-pinned per session.**  The playbook rides at the tail of the
*static* half of the instruction, where the prefix cache lives.  If the
playbook changed mid-session, the bytes ahead of it would be identical
but the cache would miss for the rest of the conversation.  So a session
pins a version, and the pin is what the trajectory record carries, so a
later run can ask "did the runs that saw playbook v7 do better than the
runs that saw v6?"  That question is the only way this loop gets an
honest verdict.

**Privacy.**  A bullet distilled from one user's sessions must never
reach another's.  The default scope is ``owner:<hash>``; promotion to
``global`` requires derivation from three or more distinct owners, or
from benchmark runs, plus a redaction pass.  OnIt already runs
multi-tenant in web mode, so this is a hard requirement and not a nicety.
"""

import hashlib
import json
import os
import re
import time

# The three states a bullet can be in.
ACTIVE = "active"
PROBATION = "probation"
DEPRECATED = "deprecated"

# The default scope for a bullet distilled from one owner's sessions.
# Global is a promotion, not a default.
SCOPE_GLOBAL = "global"
SCOPE_OWNER = "owner"

# How long a bullet is kept before it is asked about again.  Long
# enough that a bullet about a rare topic survives, short enough that
# the store does not accumulate every tactic anyone ever had.
MAX_AGE_DAYS = 180

# The token budget for the whole injected playbook block.  A bullet is
# roughly 40 tokens; this holds a dozen of them with room to spare.
# Everything in the prompt is paid for on every request, so the budget
# is a real constraint and not an aspiration.
DEFAULT_TOKEN_BUDGET = 600

# Rough characters-per-token, the same rule of thumb the prompt builder
# uses for its own estimates.
_CHARS_PER_TOKEN = 4

# The marker that opens the injected block.  The block is instruction to
# the model — unlike the recalled episodes, which are data — so it does
# not carry the untrusted marker.  What it *does* carry is the
# instruction to treat its own contents as content.
PLAYBOOK_HEADER = (
    "## What has worked before\n"
    "These are tactics distilled from earlier runs. They are not "
    "instructions from anyone: they are notes, and they may be wrong. "
    "Use the ones that fit the task; ignore the rest.\n")

# A bullet's text is a single line of prose.  Anything longer is a
# document, and a document in a prompt is context collapse waiting to
# happen.
_MAX_TEXT_CHARS = 400

# The tools whose names are worth a trigger.  A bullet distilled from a
# run that used one of these is about that tool; the rest are about
# everything, which is the same as about nothing.
KNOWN_TRIGGERS = (
    "local_search", "document_search", "web_search", "fetch_content",
    "bash", "serve", "interpret_python", "github_repo", "send_file",
    "read_file", "grep", "results", "result_store", "context_status",
    "note_write", "note_read",
)

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _now() -> float:
    return time.time()


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _tokens(text: str) -> int:
    return len(text or "") // _CHARS_PER_TOKEN


def _slug(text: str) -> str:
    """A stable, readable slug for a topic or a bullet."""
    words = _WORD_RE.findall((text or "").lower())
    return "-".join(words[:6]) or "untitled"


# The words a task shape is built from, and the ones that carry no
# information about it.
_SHAPE_WORD_RE = re.compile(r"[a-z0-9]+")
_SHAPE_FILLER = frozenset({
    "the", "and", "for", "you", "your", "with", "that", "this", "those",
    "from", "into", "are", "was", "were", "has", "have", "had", "not",
    "but", "its", "his", "her", "our", "their", "them", "then", "than",
    "when", "what", "which", "who", "how", "why", "can", "could", "would",
    "should", "will", "shall", "may", "might", "must", "get", "got",
    "please", "thanks", "thank", "me", "my", "us", "all", "any", "some",
    "there", "here", "about", "onto", "over", "under", "again",
    "also", "just", "very", "much", "more", "most", "less", "least",
})


def bullet_id(text: str, trigger: str | None = None,
              scope: str = SCOPE_GLOBAL) -> str:
    """The id a bullet gets, derived from its text, trigger and scope.

    Derived rather than sequential so that the same tactic proposed
    twice is the same bullet: the curator updates it instead of adding a
    near-duplicate, and a playbook of forty near-duplicates is a
    playbook nobody reads.

    The trigger is part of the identity, because the same sentence
    about two different tools is two different pieces of advice.  "When
    bash is used, check how it is called" and "when search is used,
    check how it is called" say the same thing and are worth the same
    slot, so collapsing them would lose one of the two triggers.

    The scope is part of it too, and it has to be: the same sentence
    derived from two owners' runs is two bullets, one per scope, and
    collapsing them would let a private tactic leak into the global
    file — which is the one thing the privacy rule forbids.
    """
    key = "%s\x00%s\x00%s" % ((text or "").strip().lower(),
                              (trigger or "").strip(),
                              (scope or "").strip())
    digest = hashlib.sha256(key.encode("utf-8"))
    return "pb-" + digest.hexdigest()[:8]


def playbook_dir(config_data: dict | None = None) -> str:
    """Directory the playbooks live in, inside the learned store."""
    from .config import trajectory_dir
    return os.path.join(os.path.dirname(trajectory_dir(config_data)),
                        "playbook")


def global_path(config_data: dict | None = None) -> str:
    return os.path.join(playbook_dir(config_data), "global.json")


def owner_path(owner: str, config_data: dict | None = None) -> str:
    """The per-owner playbook.  One file per owner, by owner hash."""
    return os.path.join(playbook_dir(config_data),
                        "user", "%s.json" % _safe_owner(owner))


def _safe_owner(owner: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(owner or "unknown"))[:96]


# ── the item ──────────────────────────────────────────────────────────────

def make_bullet(text: str, *, scope: str = SCOPE_GLOBAL,
                trigger: str | None = None,
                provenance: list | None = None,
                status: str = PROBATION) -> dict:
    """A new bullet, in the RFC's schema.

    New bullets start on ``probation``: they are sampled rather than
    always injected, so a tactic that turns out to be wrong costs a
    fraction of the runs instead of all of them.  A bullet that never
    hurts anything graduates to ``active`` on its own (see
    :func:`apply_deltas`).
    """
    text = " ".join((text or "").split())[:_MAX_TEXT_CHARS]
    trigger = str(trigger or "")
    scope = str(scope or SCOPE_GLOBAL)
    return {
        "id": bullet_id(text, trigger, scope),
        "scope": scope,
        "text": text,
        "trigger": trigger,
        "born": _today(),
        "last_used": "",
        "used": 0,
        "helped": 0,
        "hurt": 0,
        "status": status if status in (ACTIVE, PROBATION, DEPRECATED)
                   else PROBATION,
        "provenance": list(provenance or []),
    }


def _valid_bullet(item) -> bool:
    """Whether a dict is a bullet this module can work with."""
    return (isinstance(item, dict)
            and isinstance(item.get("text"), str)
            and item.get("text", "").strip() != ""
            and isinstance(item.get("id"), str))


def _normalize(item: dict) -> dict:
    """A bullet with every field present and of the right type.

    Read defensively: the file is hand-editable, and a bullet written by
    a person with a typo in it should be skipped, not crash the run.
    """
    out = dict(item)
    out["id"] = str(out.get("id") or bullet_id(out.get("text", ""),
                                               out.get("trigger"),
                                               out.get("scope")))
    out["text"] = " ".join(str(out.get("text", "")).split())[:_MAX_TEXT_CHARS]
    out["scope"] = str(out.get("scope") or SCOPE_GLOBAL)
    out["trigger"] = str(out.get("trigger") or "")
    out["status"] = (out.get("status")
                     if out.get("status") in (ACTIVE, PROBATION, DEPRECATED)
                     else PROBATION)
    for field in ("used", "helped", "hurt"):
        try:
            out[field] = int(out.get(field) or 0)
        except (TypeError, ValueError):
            out[field] = 0
    out["provenance"] = list(out.get("provenance") or [])
    return out


# ── reading and writing ──────────────────────────────────────────────────

def read_playbook(path: str) -> dict:
    """Read a playbook file.  A missing or broken file reads as empty.

    Empty rather than an error: a playbook that cannot be read must mean
    "no playbook", never "the task failed".
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {"version": 0, "items": []}
    if not isinstance(data, dict):
        return {"version": 0, "items": []}
    items = [ _normalize(i) for i in (data.get("items") or [])
              if _valid_bullet(i) ]
    try:
        version = int(data.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    return {"version": version, "items": items}


def write_playbook(path: str, playbook: dict) -> bool:
    """Write a playbook file.  Best-effort; returns whether it landed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(playbook, f, indent=2, sort_keys=True)
            f.write("\n")
        # Replace rather than append: a half-written playbook is worse
        # than the previous one.
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        return False
    return True


def load(config_data: dict | None = None, owner: str | None = None) -> dict:
    """The playbook that applies to this run.

    The owner's own bullets, plus the global ones.  The owner's come
    first: a private tactic beats a shared one, because it was distilled
    from the runs this person actually did.

    The scope of each bullet is stamped from the file it came from, not
    from what the file says.  A bullet written into the owner's file is
    *that owner's* bullet whatever its stored scope says, and a file
    that lies about it would leak a private tactic into the global set
    the next time the two are merged.  Stamping on read is the one place
    the truth is known.
    """
    playbook = read_playbook(global_path(config_data))
    items = [dict(i, scope=SCOPE_GLOBAL) for i in playbook["items"]]
    if owner:
        private = read_playbook(owner_path(owner, config_data))
        owner_scope = "%s:%s" % (SCOPE_OWNER, owner)
        items = [dict(i, scope=owner_scope) for i in private["items"]] + items
    return {"version": playbook["version"], "items": items}


# ── the deltas ───────────────────────────────────────────────────────────

def apply_deltas(playbook: dict, deltas: list) -> dict:
    """Apply a list of delta operations to a playbook (I2).

    Each delta is one of:

    * ``{"op": "insert", "text": ..., "trigger": ..., "provenance": [...]}``
    * ``{"op": "update", "id": ..., "text": ..., "trigger": ...}``
    * ``{"op": "deprecate", "id": ...}``
    * ``{"op": "reinstate", "id": ...}``
    * ``{"op": "attribute", "id": ..., "helped": +1, "hurt": +1, "used": +1}``

    Returns a *new* playbook.  Nothing is mutated in place: the caller
    decides whether to write it, and the old file stays on disk until
    then, which is what makes the change reversible (I3).

    The insert is idempotent on the bullet's id, which is derived from
    its text.  Proposing the same tactic twice updates the existing
    bullet's provenance rather than adding a near-duplicate.
    """
    result = {"version": int(playbook.get("version") or 0),
              "items": [dict(i) for i in (playbook.get("items") or [])
                        if _valid_bullet(i)]}
    by_id = {i["id"]: i for i in result["items"]}
    changed = False

    for delta in deltas or []:
        if not isinstance(delta, dict):
            continue
        op = str(delta.get("op") or "").lower()
        if op == "insert":
            text = str(delta.get("text") or "").strip()
            if not text:
                continue
            item = make_bullet(
                text,
                scope=str(delta.get("scope") or SCOPE_GLOBAL),
                trigger=delta.get("trigger"),
                provenance=delta.get("provenance"))
            existing = by_id.get(item["id"])
            if existing is None:
                result["items"].append(item)
                by_id[item["id"]] = item
                changed = True
            else:
                # The same tactic proposed again: it is confirmation, not
                # a new bullet.  Merge the provenance and leave the
                # counters alone -- the counters are measurements, and a
                # proposal is not a measurement.
                for source in item["provenance"]:
                    if source not in existing["provenance"]:
                        existing["provenance"].append(source)
                        changed = True
        elif op == "update":
            target = by_id.get(str(delta.get("id") or ""))
            if target is None:
                continue
            new_text = str(delta.get("text") or "").strip()
            if new_text and new_text != target["text"]:
                target["text"] = new_text[:_MAX_TEXT_CHARS]
                changed = True
            if delta.get("trigger") is not None:
                trigger = str(delta.get("trigger") or "")
                if trigger != target["trigger"]:
                    target["trigger"] = trigger
                    changed = True
            for source in (delta.get("provenance") or []):
                if source not in target["provenance"]:
                    target["provenance"].append(source)
                    changed = True
        elif op in ("deprecate", "reinstate"):
            target = by_id.get(str(delta.get("id") or ""))
            if target is None:
                continue
            wanted = DEPRECATED if op == "deprecate" else ACTIVE
            if target["status"] != wanted:
                target["status"] = wanted
                changed = True
        elif op == "attribute":
            target = by_id.get(str(delta.get("id") or ""))
            if target is None:
                continue
            for field in ("used", "helped", "hurt"):
                delta_value = delta.get(field)
                if delta_value is None:
                    continue
                try:
                    target[field] = max(0, int(target[field]) + int(delta_value))
                    changed = True
                except (TypeError, ValueError):
                    continue
            if delta.get("last_used"):
                target["last_used"] = str(delta["last_used"])
                changed = True

    if changed:
        result["version"] = int(result["version"]) + 1
    return result


# ── the budget ───────────────────────────────────────────────────────────

def select(items: list, *, token_budget: int = DEFAULT_TOKEN_BUDGET,
           trigger: str | None = None,
           sample: float | None = None) -> list:
    """The bullets to inject, within the budget.

    The order is: deprecated bullets are dropped, bullets matching the
    trigger come first, then the ones with the best track record, then
    the rest.  Within that order the budget is hard — a bullet that does
    not fit whole is not injected at all.

    ``sample`` injects a fraction of the probation bullets, which is how
    a new bullet earns its place without betting the run on it.  It is
    deterministic given the bullet's id, so a session that sees a
    probation bullet sees it for the whole session rather than
    flickering in and out between turns.
    """
    pool = [i for i in items if i.get("status") != DEPRECATED]
    if trigger:
        matching = [i for i in pool if i.get("trigger") == trigger]
        rest = [i for i in pool if i.get("trigger") != trigger]
        # A triggered bullet is not *only* a triggered bullet: the
        # general ones still apply.  The trigger moves a bullet up, it
        # does not hide the rest.
        pool = matching + rest
    if sample is not None and sample < 1.0:
        pool = [i for i in pool
                if i.get("status") != PROBATION or _sampled(i, sample)]
    # The trigger grouping is a *grouping*, not a total order: the
    # bullets inside each group are ordered by their own merit.  Sorting
    # the whole pool by merit alone would undo the grouping, which is
    # the one thing the trigger is for.
    if trigger:
        matching = [i for i in pool if i.get("trigger") == trigger]
        rest = [i for i in pool if i.get("trigger") != trigger]
        matching.sort(key=_selection_key)
        rest.sort(key=_selection_key)
        pool = matching + rest
    else:
        pool.sort(key=_selection_key)

    chosen = []
    used = _tokens(PLAYBOOK_HEADER)
    for item in pool:
        cost = _tokens(item.get("text", "")) + 2
        if used + cost > token_budget:
            # The budget is hard.  A partial list is indistinguishable
            # from a complete one, so the run gets fewer bullets rather
            # than a truncated one.
            break
        used += cost
        chosen.append(item)
    return chosen


def _selection_key(item: dict) -> tuple:
    """The order bullets are considered in.

    A private bullet comes before a global one: it was distilled from
    the runs *this* person actually did, so it is the more specific and
    more likely to apply.  Then active before probation (a probation
    bullet has not earned its place), then by how much good the bullet
    has done per use, then by recency, then by id, so the order is
    deterministic.
    """
    scope = str(item.get("scope") or SCOPE_GLOBAL)
    scope_rank = 0 if scope.startswith(SCOPE_OWNER) else 1
    status_rank = 0 if item.get("status") == ACTIVE else 1
    helped = int(item.get("helped") or 0)
    used = max(1, int(item.get("used") or 0))
    hurt = int(item.get("hurt") or 0)
    rate = (helped - hurt) / used
    return (scope_rank, status_rank, -rate, item.get("id", ""))


def _sampled(item: dict, fraction: float) -> bool:
    """Whether a probation bullet is sampled into this run.

    Deterministic on the bullet's id, so a session that got the bullet
    keeps it for the whole session.  A bullet that flickered in and out
    between turns of one conversation would be read as an instruction
    that came and went.
    """
    digest = hashlib.sha256(str(item.get("id") or "").encode("utf-8"))
    value = int(digest.hexdigest()[:8], 16) / 0xFFFFFFFF
    return value < fraction


# ── rendering ────────────────────────────────────────────────────────────

# Markdown heading marks, and every flavour of whitespace.  A bullet is
# one line of prose; see :func:`_one_line`.
_HEADING_MARK_RE = re.compile(r"#+")
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def _one_line(text: str) -> str:
    """Collapse a bullet's text to one line of prose.

    ``render`` is the last thing between a bullet and the prompt, so it
    has to be the place that guarantees the shape -- the Reflector
    normalises too, but a bullet also arrives from the *mechanical*
    reflector, from a hand-edited playbook file, and from a playbook
    written by an older version.  A block whose lines are bullets is a
    block; a block one of whose bullets contains ``"\n\n## System"`` is a
    new instruction section wearing the block's clothes.

    The header says these notes "are not instructions from anyone".  A
    bullet that could forge a heading made that sentence false, and the
    agent reads the forged heading as prompt structure rather than as
    text -- which is exactly what it is.

    So: drop ``#`` marks, collapse all whitespace runs to one space,
    trim.  One bullet, one line.
    """
    text = _HEADING_MARK_RE.sub("", str(text or ""))
    return _WHITESPACE_RE.sub(" ", text).strip()


def render(items: list) -> str:
    """Render the selected bullets as the block to inject.

    Empty when there is nothing to say, so the caller can insert it
    unconditionally without producing an empty section.

    Each bullet is flattened to one line here rather than trusted to
    have been flattened upstream: this is the function whose output goes
    into the prompt.
    """
    if not items:
        return ""
    lines = [PLAYBOOK_HEADER]
    for item in items:
        text = _one_line(item.get("text", ""))
        if not text:
            continue
        lines.append("- %s" % text)
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n"


def build_context(config_data: dict | None = None, owner: str | None = None,
                  trigger: str | None = None,
                  token_budget: int = DEFAULT_TOKEN_BUDGET,
                  sample: float | None = None) -> dict:
    """Everything a caller needs to inject a playbook block.

    Returns ``{"text", "items", "ids", "version"}``.  ``version`` is the
    number the trajectory record carries, so a later run can ask whether
    the runs that saw this version did better than the runs that saw the
    last one.
    """
    playbook = load(config_data, owner=owner)
    items = select(playbook["items"], token_budget=token_budget,
                   trigger=trigger, sample=sample)
    return {
        "text": render(items),
        "items": items,
        "ids": [i.get("id") for i in items],
        "version": playbook["version"],
    }


# ── the reflector ────────────────────────────────────────────────────────

def cluster(records: list) -> list:
    """Group records by the *shape* of what they did.

    The grouping key is the *shape of the task* -- the kind of question
    that was asked.  Two runs of the same kind of question are
    comparable; two runs of different kinds are not, whatever they
    have in common.

    The key used to be the tool signature, and that was the bug.  Within
    a cluster every run used the same tools, so the "a tool the good
    runs used and the bad runs did not" branch of
    :func:`candidate_deltas` was empty by construction -- and the only
    branch that ever fired was the weakest one, which produced a
    playbook of tautologies with a tool name in them.

    Now the cluster is "runs that took this shape of question", and the
    contrast inside it runs along the tools: what the good ones did
    that the bad ones did not.  That is the comparison the reflector
    was always trying to make.

    The topic is not part of the key beyond its shape.  The tactic
    "the search tool returns a file whose name matches but whose
    excerpts do not" is the same tactic whether the file was about
    parsers or about taxes, and grouping by topic would keep it from
    ever being learned twice.

    Returns a list of ``{"key", "records"}`` dicts, largest first.
    """
    buckets: dict = {}
    for record in records:
        key = task_shape(record)
        buckets.setdefault(key, []).append(record)
    return [{"key": key, "records": buckets[key]}
            for key in sorted(buckets, key=lambda k: (-len(buckets[k]), k))]


def tool_signature(record: dict) -> str:
    """The signature of a run: the tools it used and how they went.

    Purely structural.  The topic of the task is deliberately not part
    of it: the tactic "the search tool returns a file whose name
    matches but whose excerpts do not" is the same tactic whether the
    file was about parsers or about taxes, and grouping by topic would
    keep it from ever being learned twice.
    """
    parts = []
    for turn in (record.get("trajectory") or []):
        for call in (turn.get("tools") or []):
            name = call.get("name") or "?"
            parts.append("%s:%s" % (name, "ok" if call.get("ok", True)
                                    else "err"))
    return "|".join(parts) or "(no tools)"


def shape_signature(record: dict) -> str:
    """The signature of a run with the outcome removed.

    Which tools ran, in what order.  Not whether they worked: that is
    the thing the contrast is about, and putting it in the key meant
    the two sides of a contrast never met.
    """
    parts = []
    for turn in (record.get("trajectory") or []):
        for call in (turn.get("tools") or []):
            parts.append(call.get("name") or "?")
    return "|".join(parts) or "(no tools)"


def task_shape(record: dict) -> str:
    """The shape of the question a run answered.

    The leading content words of the task, in order, lowercased and
    stripped of the punctuation that varies between two people asking
    the same thing.  Deliberately a *shape* and not the whole task:
    "the weather in Manila today" and "weather in Manila today please"
    are the same shape, and two runs of the same question asked on
    different days are the same shape too.

    Filler words are dropped because they carry no information about
    the kind of task: "the", "a", "is", "for", "me", "please" appear in
    almost every task and would merge unrelated ones.
    """
    words = _SHAPE_WORD_RE.findall((record.get("task") or "").lower())
    content = [w for w in words if w not in _SHAPE_FILLER and len(w) > 2]
    # Three words, not four, and a prefix rather than a bag: a shape is
    # "the kind of question", and the words that trail it are the
    # difference between two instances of one kind, not two kinds.
    return "|".join(content[:3]) or "(no task)"


def contrast(cluster_records: list) -> dict:
    """Contrast the successful runs in a cluster against the failures.

    Returns ``{"good": [...], "bad": [...], "signature": ...}`` — the
    records the reflector should look at, split by outcome.

    A record counts as good when the verifier found nothing wrong, or
    when the user accepted it.  A record counts as bad when the
    verifier corrected something, or when the user came back and asked
    again.  A record with neither is unjudged and appears in neither
    list: the reflector's job is to contrast, and an unjudged record
    cannot be on either side of the contrast.
    """
    good, bad = [], []
    for record in cluster_records:
        signals = record.get("signals") or {}
        verifier = signals.get("verifier")
        implicit = signals.get("implicit")
        # A human click outranks both heuristics.  It is the one
        # verdict the agent could not influence, so a thumbs-down
        # belongs on the bad side even when the verifier called the
        # answer clean, and a thumbs-up belongs on the good side even
        # when the heuristic found nothing.
        rating = signals.get("user_rating")
        if rating is not None and rating < 0:
            bad.append(record)
            continue
        if rating is not None and rating > 0:
            good.append(record)
            continue
        if implicit == "bad" or verifier == "issues":
            bad.append(record)
        elif implicit == "good" or verifier == "clean":
            good.append(record)
    return {"good": good, "bad": bad}


def _record_ref(record: dict) -> str:
    """A precise pointer to one recorded run.

    ``session_id`` alone is not precise enough.  A session file holds
    one record per turn, and a session can contain a turn whose
    question is in the holdout *and* a turn whose question is not.
    Naming the session would point an audit at a session that contains
    a holdout run -- the holdout's answer would not be in the lesson,
    but its existence would be in the pointer that says where the
    lesson came from.

    ``session_id#turn`` is the same identity ``eval._record_id`` uses,
    and the one the holdout's exclusion set is keyed on, so a bullet's
    provenance can be checked against the exclusion set directly.
    """
    return "%s#%s" % (record.get("session_id", "?"), record.get("turn", "?"))


def candidate_deltas(cluster_records: list) -> list:
    """The deltas a cluster of runs suggests, without a model.

    This is the *mechanical* half of the Reflector: it finds the runs
    that went well and the runs that did not, and it names the
    difference between them in the only terms the store has — the tools
    that ran, and whether they worked.

    The contrast is between the *tools present*, not between the
    clusters.  A cluster is defined by its tool signature, so within a
    cluster the good runs and the bad runs ran the same tools; what
    differs is *which* of them, and whether the calls succeeded.  So
    the comparison is:

    * a tool the good runs used and the bad runs did not — a tactic
      worth proposing;
    * a tool the bad runs used and the good runs did not — a warning,
      and the more valuable of the two;
    * a tool that failed in the bad runs and succeeded in the good ones
      — the same tool, used badly.

    It is deliberately small.  A model-based reflector (the other half,
    and the one the RFC describes) reads these pairs and writes the
    prose; this function is what it reads, and it is what makes the
    reflection reproducible when there is no model to ask.
    """
    contrast_result = contrast(cluster_records)
    good, bad = contrast_result["good"], contrast_result["bad"]
    if not good or not bad:
        # A contrast needs both sides.  A cluster of only successes has
        # nothing to teach, and a cluster of only failures has nothing
        # to be taught against.
        return []
    good_tools = _tool_set(good)
    bad_tools = _tool_set(bad)
    good_ok = _succeeded_set(good)
    bad_ok = _succeeded_set(bad)
    deltas = []
    for tool in sorted(good_tools - bad_tools):
        deltas.append({
            "op": "insert",
            "trigger": tool,
            "text": ("When this kind of task comes up, %s worked in the runs "
                     "that went well and did not appear in the ones that "
                     "did not." % tool),
            "provenance": [_record_ref(r) for r in good[:3]],
        })
    for tool in sorted(bad_tools - good_tools):
        deltas.append({
            "op": "insert",
            "trigger": tool,
            "text": ("When this kind of task comes up, %s was present in "
                     "the runs that went badly and absent from the ones "
                     "that went well." % tool),
            "provenance": [_record_ref(r) for r in bad[:3]],
        })
    # The same tool, used badly: it failed in the runs that went badly
    # and worked in the ones that went well.  This is the most specific
    # thing the store can say, and the most actionable.
    #
    # The set is ``bad_tools - bad_ok``, not ``bad_tools``.  It used to be
    # ``good_ok & bad_tools`` -- the tool merely *appeared* in a bad run --
    # which had two consequences, both wrong: a tool that succeeded on
    # both sides was described as having "failed in the ones that did
    # not", a claim the store does not support; and a tool that failed on
    # *both* sides was dropped, losing the one case this rule exists to
    # report.
    for tool in sorted(good_ok & (bad_tools - bad_ok)):
        deltas.append({
            "op": "insert",
            "trigger": tool,
            "text": ("When %s is used on this kind of task, it succeeded in "
                     "the runs that went well and failed in the ones that "
                     "did not — check how it is called before relying on "
                     "it." % tool),
            "provenance": ([_record_ref(r) for r in bad[:3]]
                           + [_record_ref(r) for r in good[:3]]),
        })
    return deltas


def _tool_set(records: list) -> set:
    """Every tool that ran in these records."""
    tools = set()
    for record in records:
        for turn in (record.get("trajectory") or []):
            for call in (turn.get("tools") or []):
                if call.get("name"):
                    tools.add(call["name"])
    return tools


def _succeeded_set(records: list) -> set:
    """The tools that ran *and worked* in these records."""
    tools = set()
    for record in records:
        for turn in (record.get("trajectory") or []):
            for call in (turn.get("tools") or []):
                if call.get("name") and call.get("ok", True):
                    tools.add(call["name"])
    return tools


def reflect(records: list, *, owner: str | None = None,
            scope: str = SCOPE_GLOBAL,
            exclude_holdout: bool = True,
            config_data: dict | None = None) -> list:
    """The deltas the whole corpus suggests.

    The offline pass the RFC calls the Reflector, in its mechanical
    form: cluster the records, contrast the successes against the
    failures inside each cluster, and name the difference.

    ``scope`` is the scope the resulting bullets get.  The default is
    ``owner:<hash>`` for a private corpus, and the caller has to ask for
    ``global`` explicitly — promotion to global is a decision with
    conditions on it, not a default.
    """
    # The holdout is not reflected on.  A bullet distilled from a
    # holdout run is a bullet written from the exam, and it is then
    # injected into the agent that is being examined on that exam.  The
    # exclusion is by question, for the reason the recall path has it:
    # the same question asked twenty times is one question with twenty
    # answers, and excluding one of them leaks the other nineteen.
    if exclude_holdout:
        from .eval import filter_excluded
        records = filter_excluded(records, config_data)

    deltas = []
    for group in cluster(records):
        for delta in candidate_deltas(group["records"]):
            if scope == SCOPE_GLOBAL:
                delta["scope"] = SCOPE_GLOBAL
            elif owner:
                delta["scope"] = "%s:%s" % (SCOPE_OWNER, owner)
            deltas.append(delta)
    return deltas


def curate(playbook: dict, deltas: list, *, config_data: dict | None = None,
           owner: str | None = None) -> dict:
    """Apply deltas and write the result (the Curator).

    Writes to the owner's own file when a scope is given, and to the
    global file otherwise.  Returns the new playbook.  The write is
    best-effort: a playbook that cannot be saved means the run keeps
    the old one, never that the run fails.

    The previous version is kept beside the live one, so ``--rollback``
    is a file copy rather than a reconstruction (I3).  Nothing is ever
    mutated in place.
    """
    new_playbook = apply_deltas(playbook, deltas)
    if new_playbook == playbook:
        return playbook
    path = (owner_path(owner, config_data) if owner
            else global_path(config_data))
    _keep_previous(path)
    write_playbook(path, new_playbook)
    return new_playbook


def _keep_previous(path: str) -> None:
    """Copy the live playbook to its ``.prev`` sibling, if there is one.

    The copy is what makes the change reversible.  It is written
    *before* the live file is replaced, so the backup is always the
    version that was actually in use, never the one that was proposed.
    """
    try:
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        with open(path + ".prev", "w", encoding="utf-8") as f:
            f.write(data)
    except (OSError, ValueError):
        # A missing backup means the rollback is unavailable, not that
        # the change must not happen.
        pass


# ── the lifecycle ────────────────────────────────────────────────────────

def attribute(items: list, ids: list, *, helped: int = 0,
              hurt: int = 0) -> list:
    """Credit the bullets a run was shown, and return the new counters.

    This is the credit-assignment half of the loop.  A run that was
    shown a set of bullets and went well credits them; a run that went
    badly does not.  The attribution is *coarse* — every bullet shown
    in a good run gets the credit — and the RFC calls that out: it is a
    weak signal, and a bullet that helps in one run and hurts in another
    shows up as exactly that, which is the point.
    """
    by_id = {i.get("id"): i for i in items}
    for item_id in ids or []:
        item = by_id.get(item_id)
        if item is None:
            continue
        item["used"] = int(item.get("used") or 0) + 1
        item["helped"] = int(item.get("helped") or 0) + max(0, helped)
        item["hurt"] = int(item.get("hurt") or 0) + max(0, hurt)
        item["last_used"] = _today()
    return items


def decay(items: list, *, now: float | None = None,
          max_age_days: int = MAX_AGE_DAYS) -> list:
    """Move stale and counterproductive bullets out of the active set.

    Two rules, both from the RFC:

    * a bullet unused for ``max_age_days`` drops to ``probation`` — it
      may still be right, but nothing has said so lately;
    * a bullet that has hurt more than it has helped is deprecated.

    Deprecated bullets are *kept*.  The store is the audit trail: a
    tactic that was tried and did not work is evidence, and deleting it
    would leave no way to notice it being proposed again.
    """
    now = now if now is not None else _now()
    cutoff = now - (max_age_days * 86400)
    for item in items:
        if item.get("status") == DEPRECATED:
            continue
        helped = int(item.get("helped") or 0)
        hurt = int(item.get("hurt") or 0)
        used = int(item.get("used") or 0)
        if used and hurt > helped:
            item["status"] = DEPRECATED
            continue
        last_used = item.get("last_used") or ""
        if last_used:
            try:
                stamp = time.mktime(time.strptime(last_used, "%Y-%m-%d"))
            except (ValueError, OverflowError):
                continue
            if stamp < cutoff:
                item["status"] = PROBATION
    return items
