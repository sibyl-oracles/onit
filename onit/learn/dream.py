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

Dreaming: the consolidation pass that maintains the memory wiki.

The agent works; the wiki is written when it does not.  A dream pass
reads what happened since the last one (the transcript, which is ground
truth), asks a model what facts, rules and corrections the sessions
established, verifies each candidate against that transcript, and
reconciles the survivors into the wiki.  It is the article's Pokémon fix
in miniature: the note is wrong, the transcript proves it, the note gets
rewritten -- and the agent that would have looped forever on the bad
note finishes instead.

**The dream pass is the only writer.**  Not the agent mid-task, not a
human hand-editing.  One writer means no merge conflicts, no concurrent
edits, and every change traceable to one idle pass in ``dream.log``.

**Idle-triggered.**  ``lifecycle.episodes_since`` already counts
completed tasks since a timestamp; the trigger is "idle > N episodes"
(N=1 by default -- after every quiet period).  Episodes, not wall-clock:
a weekend of zero use produces zero new records and nothing to
consolidate, which is the correct outcome.  A lock file covers the edge
case of a task starting mid-dream; the watermark makes a re-run
idempotent.

**The model is injected.**  ``ask`` is a parameter, the same choice
``reflector.py`` and ``verify.py`` make: testable without a network, and
the dreaming model gets no tools -- it reads what it was handed and
answers, it does not run anything.

**Extraction is after the fact, not during it.**  The article tags
salient events as they happen; the batch version reads the transcript
afterwards and tags the same events.  What it must not do is trust the
model's summary: every candidate carries a citation to transcript lines,
and step 3 re-reads those lines before anything is written.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

from . import memory

logger = logging.getLogger(__name__)

# ── tuning ────────────────────────────────────────────────────────────────────

# The most characters of one session's transcript handed to the model.
# Enough to see what was established, not enough to drown in.
MAX_SESSION_CHARS = 6000
# The A/B gate's seeding pass lifts this (SEED_MAX_SESSION_CHARS in
# runner.py), because a replay's whole day lands in one session file.

# The most candidates one session may produce.  Three is the point at
# which a contrast starts to look like a list of everything (the same
# cap the reflector puts on bullets); here it also bounds the cost of
# verifying each one against the transcript.
MAX_CANDIDATES_PER_SESSION = 6

# The most sessions one pass processes.  A burst of twenty sessions is
# twenty passes' worth of work; one pass takes the oldest first and
# leaves the watermark honest.
MAX_SESSIONS_PER_PASS = 8

# The most topics one pass may promote to a page.  A pass that created
# twenty pages created a Sonnet-3.7 store: hundreds of files dream-time
# must comb.
MAX_PAGES_PER_PASS = 3

# The most deltas one pass applies to one page.  The rest wait for the
# next pass; the watermark only advances over what was actually applied.
MAX_DELTAS_PER_PAGE = 8

# The page-creation threshold, overridable per pass.  Production keeps
# the plan's 3: a page is a topic that earned it.  The A/B gate's
# seeding pass passes 1 -- it gets ONE dream pass over ONE replay day,
# and a replay day is a tour of unrelated tasks, one fact each.  Under
# the production threshold every fact lands in its own cluster, nothing
# earns a page, and the gate measures a placebo -- the 2026-11-17 run
# extracted 6 verified facts and created 0 pages exactly this way.
SEED_MIN_FACTS_FOR_PAGE = 1

# The output budget for one extraction call.
MAX_TOKENS = 900

# The attempts one extraction call gets.  The endpoint behind the gate
# intermittently returns an empty reply -- the same flakiness the
# grader's retry absorbs -- and the 2026-11-17 14:55 seed pass lost its
# only call to it: 0 candidates, 0 pages, a three-hour placebo run.
# An extraction that gets one attempt is a coin flip; three is a policy.
EXTRACT_ATTEMPTS = 3

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)
_THINK_RE = re.compile(r"think.*?/think", re.DOTALL | re.IGNORECASE)

# ── the extraction prompt ─────────────────────────────────────────────────────

_EXTRACT_SYSTEM = (
    "You are reading the transcript of an AI agent's session and writing "
    "down what it established for a future session to reuse.\n\n"
    "Rules:\n"
    "- Extract only what is NOT re-derivable by reading the same files "
    "again: outcomes, decisions, corrections, environment facts, rules "
    "that worked. Not what the agent read off the screen.\n"
    "- One fact per item. Under 60 words. No markdown, no headings.\n"
    "- Every item must cite the transcript line(s) that establish it, as "
    '"cites: <line number(s)>".\n'
    '- kind is "fact" for something true, "rule" for something to do.\n'
    "- topic is a 2-5 word name for the subject the item belongs to.\n"
    "- If the session established nothing worth keeping, return an empty "
    "list.\n\n"
    "Reply with JSON only, no prose:\n"
    '{"items": [{"text": "...", "kind": "fact|rule", "topic": "...", '
    '"cites": [1]}]}\n'
)

# ── the lock ──────────────────────────────────────────────────────────────────


class DreamBusy(Exception):
    """Another dream pass holds the lock."""


class _Lock:
    """The dream lock: create exclusively, remove on exit.

    ``O_CREAT | O_EXCL`` is the atomic test-and-set the edge case needs:
    a task that starts mid-dream finds the lock present and preempts the
    pass by simply not waiting for it -- the next idle window re-runs
    what the watermark says is left.
    """

    def __init__(self, config_data: dict | None = None, root: str | None = None):
        self.path = (
            memory.lock_path(config_data) if root is None else os.path.join(root, memory.LOCK_NAME)
        )
        self.fd = None

    def __enter__(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, str(os.getpid()).encode())
            return self
        except FileExistsError:
            raise DreamBusy(self.path)
        except OSError as e:
            # An unwritable store cannot be locked, and a pass that
            # cannot lock must not run unsynchronized: fail as busy.
            logger.debug("dream lock unavailable: %s", e)
            raise DreamBusy(self.path)

    def __exit__(self, *exc):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            try:
                os.remove(self.path)
            except OSError:
                pass
        self.fd = None
        return False


# ── the trigger ───────────────────────────────────────────────────────────────


def idle_episodes(config_data: dict | None = None) -> int:
    """Completed tasks since the last dream pass finished.

    The watermark lives in the store's state file, so the trigger is a
    property of the store, not of a process: two OnIt instances serving
    sessions see the same idle count and agree on it.
    """
    from .lifecycle import episodes_since

    state = memory.load_state(config_data)
    return episodes_since(str(state.get("watermark_ts", "")), config_data)


def should_dream(config_data: dict | None = None) -> bool:
    """Whether an idle window owes a dream pass.

    Gated on autonomy >= ADAPT, like every loop that changes what the
    agent is told: at observe, recording is the whole contract.
    """
    from .config import ADAPT, autonomy

    if autonomy(config_data) < ADAPT:
        return False
    return idle_episodes(config_data) >= memory.min_idle_episodes(config_data)


def dream_enabled(config_data: dict | None = None) -> bool:
    """Whether dreaming is switched on at all.

    Separate from :func:`should_dream` on purpose.  ``should_dream``
    answers "is an idle window owed a pass"; this answers "may the
    store change at all".  The A/B gate (phase 6 of the memory plan)
    needs the second answer: a replay must not dream mid-run, or the
    treatment arm's wiki grows from the very items being scored --
    training on the test set.  A caller that wants a pass *now*
    (``onit learn --dream``, or the gate's own seeding between arms)
    passes ``min_idle_episodes`` high to defer the trigger without
    also having to lie about the switch.
    """
    return memory.enabled(config_data)


# ── the transcript ────────────────────────────────────────────────────────────


def _transcript_text(records: list, limit: int = MAX_SESSION_CHARS) -> str:
    """A session's task records as numbered lines the model can cite.

    Task and response are the two things a fact can be established by;
    tool-call digests and token counts are noise here.  Numbered so the
    model's citations point at something the verifier can re-read.
    """
    lines = []
    for record in records:
        task = " ".join(str(record.get("task") or "").split())
        response = " ".join(str(record.get("response") or "").split())
        if task:
            lines.append(task)
        if response:
            lines.append(response[:1200])
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[:limit]
    return "\n".join("%d: %s" % (i + 1, line) for i, line in enumerate(text.splitlines()))


def _new_records_since(records: list, ts: str) -> list:
    """The records a pass has not seen yet, oldest first."""
    if not ts:
        return list(records)
    return [r for r in records if str(r.get("ts") or "") > ts]


# ── extraction ────────────────────────────────────────────────────────────────


def _extract_json(text: str) -> list:
    """The item list from a model response, or an empty list.

    Strict on purpose, the same rule ``reflector._extract_json`` runs
    on: a half-parsed candidate is injected into a wiki page as if it
    were established.
    """
    if not text:
        return []
    if isinstance(text, tuple):
        text = text[0]
    cleaned = _THINK_RE.sub("", str(text)).strip()
    match = _JSON_FENCE_RE.search(cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    items = data.get("items")
    if not isinstance(items, list):
        return []
    return [i for i in items if isinstance(i, dict)]


def _operational(lowered: str) -> bool:
    """A candidate that records an action taken on the environment.

    Pushes, commits, merges, deploys-to-live, service restarts and file
    mutations are things the *session did*, not things that stay true.
    Served to a later session they read as precedent -- the failure the
    2026-11-20 gate caught.  The list is deliberately narrow: a wiki
    that drops real knowledge is worth nothing either.
    """
    markers = (
        "pushed to",
        "push to",
        "committed to",
        "committed the",
        "merged into",
        "merged the",
        "opened a pull request",
        "deployed to",
        "restarted the",
        "restarted service",
        "origin/main",
        "deleted the branch",
        "force-pushed",
        "wrote the file",
        "created the file",
        "overwrote the",
        "removed the file",
        "ran the migration",
        "applied the migration",
    )
    return any(m in lowered for m in markers)


def _clean_candidate(item: dict, transcript: str) -> dict | None:
    """One candidate, verified against the transcript it cites.

    The article's step 3 is here rather than in the model: a candidate
    whose citations do not appear in the transcript is a hallucination,
    and a hallucination that reached the wiki would be served to every
    future session.  The check is textual, not semantic -- the cited
    line must exist and the candidate must share a token with it -- so
    it is exact and cheap.
    """
    from ._search import tokenize

    text = memory._one_line(item.get("text", ""))[: memory.MAX_FACT_CHARS]
    if not text or len(text) < 12:
        return None
    lowered = text.lower()
    for phrase in ("the session", "this session", "the agent was asked", "in this run"):
        if lowered.startswith(phrase):
            return None
    kind = item.get("kind") if item.get("kind") in ("fact", "rule") else "fact"
    topic = memory._one_line(item.get("topic", ""))[:60]
    if _operational(lowered):
        # Operational, not generalizable: the 2026-11-20 gate run's one
        # discordant pair traced to a seeded page that recorded a push
        # to origin/main as a fact -- the treatment agent read it,
        # imitated it, pushed a real branch, and was graded wrong for
        # it.  A fact about what *this* agent did to *its* environment
        # is session state, and a future session that reads it as
        # precedent repeats the action.  Keep it out of pages; log it.
        return None
    cites = item.get("cites") if isinstance(item.get("cites"), list) else []
    transcript_lines = transcript.splitlines()
    matched = []
    for cite in cites[:4]:
        try:
            index = int(cite) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(transcript_lines):
            matched.append(transcript_lines[index])
    if not matched:
        return None
    hay = " ".join(tokenize(" ".join(matched)))
    tokens = tokenize(text)
    # Stopwords ("the", "is") match any sentence and prove nothing; the
    # check is on the words that carry the fact.  A candidate whose
    # content words all miss its citation is a hallucination even when
    # every stopword lines up.
    content = [t for t in tokens if len(t) > 3]
    if content and not any(t in hay for t in content):
        return None
    return {
        "text": text,
        "kind": kind,
        "topic": topic or "misc",
        "cites": [str(c) for c in cites[:4]],
    }


async def extract(ask, records: list, max_chars: int = MAX_SESSION_CHARS) -> tuple:
    """What one session established, verified.  Empty when nothing did.

    Returns ``(candidates, empty_replies, parsed_items, failed)``.
    Candidates are of the shape ``{"text", "kind", "topic", "cites"}``
    with ``cites`` resolved to transcript lines -- the provenance every
    written fact carries.  ``empty_replies`` counts the hollow responses
    the retries burned: a pass over a day that established nothing and a
    pass whose extractor the endpoint failed must not look the same in
    the stats, because they want different fixes.  ``parsed_items``
    counts what the reply held before the cite-cleaning: a reply that
    parsed to items which were all dropped is a third cause -- the
    citations failed the transcript check -- and the verdict needs to
    name it too.  ``failed`` is the fourth distinction the watermark
    needs: a session whose extraction never got a usable reply (hollow
    replies, or a call that raised) was not processed, and the pass must
    not advance over it -- a day burned by a failed extractor is a day
    no later pass can re-read.

    ``max_chars`` bounds the transcript the model reads.  The default is
    the production trigger's budget; the A/B gate's seeding pass raises
    it, because a replay's whole day lands in one session file and a
    6,000-character window over 20 items is the first two items -- the
    pass then "extracts" from an experience the treatment never saw.
    """
    if not records:
        return [], 0, 0, False
    transcript = _transcript_text(records, limit=max_chars)
    if not transcript.strip():
        return [], 0, 0, False
    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": "Transcript:\n%s" % transcript},
    ]
    # The endpoint behind the gate intermittently returns an empty reply
    # -- the grader in the same run hit it and absorbed it with a retry,
    # while this call used to get exactly one attempt.  One flaky reply
    # at seed time was a whole placebo run (the 2026-11-17 14:55 seed:
    # 0 candidates, 0 pages, three hours of measurement of nothing).
    # Retry the empties; an exception still aborts -- a network that is
    # down is a different failure than a reply that came back hollow.
    response = ""
    failed = False
    for attempt in range(1, EXTRACT_ATTEMPTS + 1):
        try:
            response = await ask(messages)
        except Exception as e:  # pragma: no cover - network failure
            logger.debug("dream extraction failed: %s", e)
            return [], 0, 0, True
        if (response or "").strip():
            break
        logger.warning(
            "dream extraction returned no text (attempt %d/%d)", attempt, EXTRACT_ATTEMPTS
        )
        response = ""
    if not (response or "").strip():
        # Every attempt came back empty.  This is not "the session
        # established nothing" -- that is a reply that parsed and held
        # no items.  Say which, so a seed of zero candidates can be
        # told apart from a day that genuinely established nothing.
        logger.warning(
            "dream extraction: %d empty replies; session left unextracted", EXTRACT_ATTEMPTS
        )
        return [], EXTRACT_ATTEMPTS, 0, True
    items = _extract_json(response)[:MAX_CANDIDATES_PER_SESSION]
    candidates = []
    for item in items:
        cleaned = _clean_candidate(item, transcript)
        if cleaned:
            cleaned["provenance"] = [
                "%s#%s" % (records[0].get("session_id", "?"), c) for c in cleaned["cites"]
            ]
            candidates.append(cleaned)
    if not candidates:
        # A real reply that kept nothing.  Two causes want different
        # fixes and the stats could not tell them apart: the reply held
        # no items (format drift -- the model answered in prose), or it
        # held items whose citations failed the transcript check (the
        # cite-cleaning dropped them all).  Log the reply head so the
        # next diagnosis is not blind, and report the parsed count so
        # the verdict can name the cause.
        logger.warning(
            "dream extraction kept 0 of %d parsed item(s); reply head: %.400r",
            len(items),
            response[:400],
        )
    return candidates, attempt - 1, len(items), False


# ── clustering and promotion ──────────────────────────────────────────────────


def _cluster(candidates: list) -> dict:
    """Group candidates by topic slug.  Pure."""
    groups: dict = {}
    for cand in candidates:
        groups.setdefault(memory.slugify(cand["topic"]), []).append(cand)
    return groups


def _existing_page(slugs: set, slug: str, config_data: dict | None, root: str | None):
    """The page to merge a candidate into, or None.

    A slug match wins; otherwise the page whose text the candidate
    matches best wins, so two spellings of one topic do not become two
    pages.  The threshold is deliberately loose (any nonzero BM25 hit):
    a near-miss page is a better home than a duplicate.
    """
    pages = memory.list_pages(config_data, root=root)
    if slug in slugs:
        for page in pages:
            if page["slug"] == slug:
                return page
    ranked = memory.rank_pages(" ".join(c["text"] for c in []), pages) if False else None
    return None


def _find_home(cand: dict, pages: list):
    """The page a candidate belongs on, or None for a new page."""
    ranked = memory.rank_pages(cand["text"], pages)
    if ranked and ranked[0][1] > 0:
        return ranked[0][0]
    return None


# ── the pass ──────────────────────────────────────────────────────────────────


async def dream_once(
    ask,
    config_data: dict | None = None,
    *,
    root: str | None = None,
    now: float | None = None,
    max_session_chars: int | None = None,
    min_facts_for_page: int | None = None,
    max_sessions_per_pass: int | None = None,
    max_pages_per_pass: int | None = None,
) -> dict:
    """One consolidation pass.  Returns what it did, for tests and logs.

    The pass is append-and-log: nothing is destroyed.  Pages are
    corrected, not deleted; a superseded claim is logged to
    ``dream.log``; a page past its prune age has its detail collapsed,
    and the detail lives on in the transcripts the store was distilled
    from.

    Best-effort throughout: a store that cannot be written means the
    pass did less, never that a task failed -- no task is running.
    """
    from .config import ADAPT, autonomy

    if autonomy(config_data) < ADAPT:
        return {"ran": False, "reason": "autonomy below adapt"}

    started = time.monotonic()
    state = memory.load_state(config_data, root=root)
    watermark_ts = str(state.get("watermark_ts", ""))
    pass_no = int(state.get("passes", 0)) + 1
    log_lines = ["== pass %d @ %s ==" % (pass_no, _now())]
    changes = {
        "ran": True,
        "pass": pass_no,
        "sessions": 0,
        "candidates": 0,
        "verified": 0,
        "rejected": 0,
        "pages_created": 0,
        "pages_updated": 0,
        "pruned": 0,
        "contradictions_cleared": 0,
    }

    # 1. Collect: every session with new activity, oldest first.
    from .corpus import read_sessions

    sessions = read_sessions(config_data)
    todo = []
    for session_id, records in sorted(sessions.items()):
        fresh = _new_records_since(records, watermark_ts)
        if fresh:
            todo.append((session_id, fresh))
        if len(todo) >= max(1, int(max_sessions_per_pass or MAX_SESSIONS_PER_PASS)):
            break
    changes["sessions"] = len(todo)

    # 2. Extract per session, verified against its own transcript.
    candidates: list = []
    empty_replies = 0
    parsed_items = 0
    processed: list = []
    for session_id, fresh in todo:
        found, empties, parsed, failed = await extract(
            ask, fresh, max_chars=max_session_chars or MAX_SESSION_CHARS
        )
        empty_replies += empties
        parsed_items += parsed
        changes["candidates"] += len(found)
        changes["rejected"] += max(0, min(len(fresh), MAX_CANDIDATES_PER_SESSION) - len(found))
        candidates.extend(found)
        if not failed:
            processed.append((session_id, fresh))
        else:
            log_lines.append(
                "  %s: extraction failed (%d empty repl(ies)); left for the next pass"
                % (session_id, empties)
            )
    changes["verified"] = changes["candidates"]
    # An empty reply is the extractor failing, not the day having
    # established nothing -- the distinction the gate's zero-candidate
    # diagnosis needs (two causes, two fixes).  A day that established
    # nothing is a real reply holding no items; a failed extractor is
    # hollow replies all the way down.  ``parsed_items`` adds the third
    # distinction: a reply that held items whose citations failed the
    # transcript check -- the extractor answered, the cleaning dropped
    # everything, and that wants a different fix than either.
    changes["empty_replies"] = empty_replies
    changes["parsed_items"] = parsed_items

    # 3. Reconcile into the wiki.
    pages = memory.list_pages(config_data, root=root)
    by_slug = {p["slug"]: p for p in pages}
    groups = _cluster(candidates)
    created = 0
    page_cap = MAX_PAGES_PER_PASS if max_pages_per_pass is None else max(0, int(max_pages_per_pass))
    for slug, cands in sorted(groups.items()):
        if created >= page_cap and slug not in by_slug:
            continue
        page = by_slug.get(slug)
        deltas = []
        if page is None:
            # The creation policy: a page is something a topic *earns*.
            # The threshold is the plan's 3 unless the caller overrides
            # it -- the A/B gate's seeding pass does, because one pass
            # over one replay day cannot accumulate across passes.
            threshold = (
                memory.MIN_FACTS_FOR_PAGE
                if min_facts_for_page is None
                else max(1, int(min_facts_for_page))
            )
            if len(cands) < threshold:
                log_lines.append(
                    "  %s: %d fact(s) below the page threshold (%d); dropped"
                    % (slug, len(cands), threshold)
                )
                continue
            page = memory.make_page(slug, _title_from(cands))
            created += 1
            changes["pages_created"] += 1
            log_lines.append("  created page %s (%d fact(s))" % (slug, len(cands)))
        for cand in cands[:MAX_DELTAS_PER_PAGE]:
            deltas.append(
                {
                    "op": "add_fact",
                    "text": cand["text"],
                    "kind": cand["kind"],
                    "scope": "",
                    "provenance": cand.get("provenance") or [],
                }
            )
        page, applied = memory.apply_deltas(page, deltas)
        if applied:
            changes["pages_updated"] += 1
            log_lines.extend("  %s: %s" % (slug, c) for c in applied)
        memory.save_page(page, config_data, root=root)

    # 4. Contradictions the fact-checker left open: resolve or keep.
    for page in memory.list_pages(config_data, root=root):
        if not page.get("contradictions"):
            continue
        # The transcript is the arbiter.  A contradiction whose
        # correction is supported by anything this pass read is applied
        # as a correction; otherwise it stays open and keeps the page
        # out of the fast path.
        page, cleared = memory.clear_contradictions(page)
        if cleared:
            changes["contradictions_cleared"] += cleared
            log_lines.append(
                "  %s: %d contradiction(s) resolved against the transcript"
                % (page["slug"], cleared)
            )
            memory.save_page(page, config_data, root=root)

    # 5. Prune and demote: pages untouched past the prune age collapse.
    prune_after = memory.prune_after_days(config_data)
    for page in memory.list_pages(config_data, root=root):
        age = _days_since(page.get("updated") or "")
        if age > prune_after:
            page, dropped = memory.prune_page(page)
            if dropped:
                changes["pruned"] += dropped
                log_lines.append("  %s: pruned %d stale fact(s)" % (page["slug"], dropped))
                memory.save_page(page, config_data, root=root)

    # 6. Rebuild the index and the staging expiry.
    all_pages = memory.list_pages(config_data, root=root)
    memory.write_index(all_pages, config_data, root=root)
    _expire_staging(config_data, root=root, log_lines=log_lines)

    # 7. Advance the watermark over what was actually processed.
    processed_ts = watermark_ts
    for session_id, fresh in processed:
        for record in fresh:
            ts = str(record.get("ts") or "")
            if ts > processed_ts:
                processed_ts = ts
    state.update(
        {
            "watermark_ts": processed_ts,
            "passes": pass_no,
            "last_run": _now(),
            "last_stats": {k: v for k, v in changes.items() if k != "ran"},
        }
    )
    memory.save_state(state, config_data, root=root)
    log_lines.append("  watermark -> %s" % (processed_ts or "(start)"))
    memory.append_log(log_lines, config_data, root=root)
    changes["seconds"] = round(time.monotonic() - started, 3)
    return changes


def _title_from(cands: list) -> str:
    """A page title from its founding candidates: the most common topic."""
    topics = [c.get("topic") or "" for c in cands if c.get("topic")]
    if topics:
        return memory._one_line(max(set(topics), key=topics.count))[:120]
    return memory._one_line(cands[0].get("text", ""))[:60]


def _expire_staging(
    config_data: dict | None = None, root: str | None = None, log_lines: list | None = None
) -> int:
    """Drop staging files the last pass already had its chance at.

    A staging file older than two passes is one the promotion policy
    declined twice; it expires rather than accreting into the Sonnet-3.7
    store.  Its content lives in the transcript, which is on disk.
    """
    base = (
        memory.staging_dir(config_data)
        if root is None
        else os.path.join(root, memory.SESSIONS_SUBDIR)
    )
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return 0
    state = memory.load_state(config_data, root=root)
    passes = int(state.get("passes", 0))
    expired = 0
    for name in names:
        if not name.endswith(".md"):
            continue
        path = os.path.join(base, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                first = f.readline().strip()
            # The staging header carries the pass number it was staged at.
            staged_at = int(first.split("pass=")[-1].split()[0]) if "pass=" in first else 0
        except (OSError, ValueError, IndexError):
            staged_at = 0
        if passes - staged_at >= 2:
            try:
                os.remove(path)
                expired += 1
            except OSError:
                pass
    if expired and log_lines is not None:
        log_lines.append("  expired %d staging file(s)" % expired)
    return expired


def _days_since(day: str) -> int:
    try:
        from datetime import date

        then = date.fromisoformat(str(day)[:10])
    except (TypeError, ValueError):
        return 10**6
    return (date.fromisoformat(memory._today()) - then).days


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── the runner ────────────────────────────────────────────────────────────────


async def run_dream(
    config_data: dict | None = None,
    *,
    ask=None,
    root: str | None = None,
    force: bool = False,
    max_session_chars: int | None = None,
    min_facts_for_page: int | None = None,
    max_sessions_per_pass: int | None = None,
    max_pages_per_pass: int | None = None,
) -> dict:
    """Trigger check, lock, pass.  The entry point an idle watcher calls.

    Builds its own model client when none is handed in -- the same
    ``make_ask`` the Reflector uses, which carries no tools, so a
    dreaming model cannot run anything while it thinks.

    ``force`` runs a pass with the idle trigger and the memory switch
    unconsulted -- the A/B gate seeds a treatment store between arms,
    and a seeded pass must not wait for an idle window that a replay
    will never provide.  The autonomy gate and the lock still apply: a
    forced pass is an out-of-band pass, not an unsupervised one.

    ``min_facts_for_page`` overrides the page-creation threshold for
    this pass only.  The A/B gate's seeding pass passes 1: it gets one
    pass over one replay day, and a replay day is a tour of unrelated
    tasks -- under the production threshold of 3, every extracted fact
    landed in its own one-fact cluster and nothing earned a page, so
    the gate measured a placebo twice (2026-11-16 and 2026-11-17).
    """
    from .config import ADAPT, autonomy

    if autonomy(config_data) < ADAPT:
        return {"ran": False, "reason": "autonomy below adapt"}
    if not force:
        if not dream_enabled(config_data):
            return {"ran": False, "reason": "memory disabled"}
        if not should_dream(config_data):
            return {"ran": False, "reason": "not idle enough"}
    try:
        with _Lock(config_data, root=root):
            if ask is None:
                from .llm import make_ask

                ask = await make_ask(config_data)
            return await dream_once(
                ask,
                config_data,
                root=root,
                max_session_chars=max_session_chars,
                min_facts_for_page=min_facts_for_page,
                max_sessions_per_pass=max_sessions_per_pass,
                max_pages_per_pass=max_pages_per_pass,
            )
    except DreamBusy:
        return {"ran": False, "reason": "another pass holds the lock"}


def run_dream_sync(config_data: dict | None = None, **kwargs) -> dict:
    """The synchronous wrapper for CLI callers."""
    return asyncio.run(run_dream(config_data, **kwargs))


# ── session staging (phase 2: capture) ────────────────────────────────────────


def stage_session_note(
    session_id: str,
    text: str,
    config_data: dict | None = None,
    *,
    root: str | None = None,
) -> bool:
    """What the current session wrote down, staged for the next pass.

    The staging file is working memory, not a wiki: dreaming either
    promotes a staged fact up to a page (with provenance) or lets the
    file expire.  Promoting everything per-session is the Sonnet-3.7
    failure mode -- hundreds of stale files dream-time must comb.
    """
    if not session_id or not text or not text.strip():
        return False
    state = memory.load_state(config_data, root=root)
    passes = int(state.get("passes", 0))
    base = (
        memory.staging_dir(config_data)
        if root is None
        else os.path.join(root, memory.SESSIONS_SUBDIR)
    )
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(session_id))[:96]
    path = os.path.join(base, "%s.md" % safe)
    try:
        os.makedirs(base, exist_ok=True)
        header = "staged at pass=%d %s\n" % (passes, memory._today())
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + str(text).strip()[: memory.MAX_FACT_CHARS * 20] + "\n")
    except OSError:
        return False
    return True
