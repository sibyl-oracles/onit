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

The model-based half of the Reflector: reading two runs and naming the
difference between them.

The mechanical Reflector in ``playbook.py`` can only say what the store
already knows.  It clusters runs by their tool signature and contrasts
the good ones against the bad ones, and because a cluster *is* a tool
signature, the two sides of every contrast ran the same tools -- so the
"a tool the good runs used and the bad ones did not" branch is empty by
construction, and the only branch that ever fires is the weakest one:
"the same tool, used badly".  The result is a playbook of tautologies
with a tool name in them, which is a template, not a tactic.

This module is the other half.  It hands a model two contrasting runs --
one that went well, one that went badly -- and asks for the difference
that a person reading both transcripts would name: not *which* tool
ran, but *what the run did differently*, in one sentence, in the form
of advice someone could follow.

**Why a model is required.**  The contrast is not in the tool names; it
is in the arguments, the order, the reasoning and the answer.  A tool
that appears in both runs can have been called with the wrong path in
the failing one.  Only something that can read the two transcripts and
compare them can say that.

**Why the model is injected.**  ``ask`` is a parameter, not an import.
The reflector is then testable without a network and without a model
server, which is the same choice ``src/model/serving/verify.py`` makes
for the fact-checker.  It also means the reflector never borrows the
*agent's* tool registry: the reflecting model gets no tools at all, so
it cannot run anything, write anything, or read anything it was not
handed in the prompt.

**What the output is allowed to be.**  A bullet is one piece of advice
with a trigger.  The model is asked for that and nothing else, and the
parse is strict: a response that does not carry the requested shape
yields no bullet rather than a broken one.  A playbook with a broken
bullet in it is worse than a smaller playbook, because the broken one
is injected verbatim into the agent's prompt.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# The output budget for one reflection.  Small on purpose: the answer is
# one sentence of advice, and a large budget buys prose nobody reads.
MAX_TOKENS = 400

# The most bullets one contrast may produce.  Three is the point at which
# a contrast starts to look like a list of everything that differed.
MAX_BULLETS_PER_CONTRAST = 3

# The most characters a bullet may be.  The playbook caps the whole
# block at a token budget, and a bullet that fills it alone is a bullet
# that crowds out every other one.
MAX_BULLET_CHARS = 240

# The most characters of a single run's transcript handed to the model.
# Enough to see the mistake, not enough to drown in it.
MAX_RUN_CHARS = 2000

# The output shape.  JSON in, JSON out, because the parse has to be
# unambiguous: a model that returns three bullets as prose and a model
# that returns three bullets as a list are the same answer, and only one
# of them is safe to inject.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ── the prompt ───────────────────────────────────────────────────────────────

_SYSTEM = """\
You are reading two runs of the same kind of task by an AI agent: one \
that went well and one that went badly. Your job is to name the \
difference that would have helped the bad run, as advice.

Rules:
- One piece of advice per bullet. Never two.
- Say what to DO, not what happened. "Open the file before reading its \
excerpts", not "the bad run read excerpts first".
- Be specific to this task shape. If the advice would be true of every \
task, it is worth nothing.
- No general advice about being careful, thorough, or verifying work.
- No mention of the runs themselves, no "in the failing run", no \
meta-commentary about the contrast.
- Under 30 words.
- The trigger is the short phrase a future agent would recognise: a \
tool name, a file type, a situation. Not a whole sentence.

Reply with JSON only, no prose:
{"bullets": [{"text": "...", "trigger": "..."}]}
Return at most %d bullets. Return an empty list if the two runs differ \
in nothing that would help a future run.
""" % MAX_BULLETS_PER_CONTRAST


def _run_excerpt(record: dict, limit: int = MAX_RUN_CHARS) -> str:
    """A run's transcript, as text the model can read.

    The tool calls and the answer, and nothing else.  Token counts and
    timestamps are noise here: the question is what the run *did*, and a
    transcript that opens with a metrics dump is a transcript the model
    spends its attention on the wrong part of.
    """
    if not record:
        return ""
    lines = []
    for turn in (record.get("trajectory") or []):
        for call in (turn.get("tools") or []):
            name = call.get("name") or "?"
            ok = call.get("ok", True)
            lines.append("tool %s%s" % (name, "" if ok else "  (FAILED)"))
    response = (record.get("response") or "").strip()
    if response:
        lines.append("answer: %s" % response)
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[:limit] + "\n[truncated]"
    return text


def _user_prompt(good: dict, bad: dict) -> str:
    """The two runs, side by side, labelled by outcome."""
    return (
        "TASK\n%s\n\n"
        "RUN THAT WENT WELL\n%s\n\n"
        "RUN THAT WENT BADLY\n%s\n\n"
        "What would have helped the bad run?"
        % ((good.get("task") or "").strip(),
           _run_excerpt(good) or "(no tools ran)",
           _run_excerpt(bad) or "(no tools ran)")
    )


# ── the parse ───────────────────────────────────────────────────────────────

def _extract_json(text: str) -> list:
    """The bullet list from a model response, or an empty list.

    Strict on purpose.  A response that is not the shape that was asked
    for yields nothing, because a half-parsed bullet is injected into
    the agent's prompt as if it were advice.
    """
    if not text:
        return []
    cleaned = _THINK_RE.sub("", text).strip()
    match = _JSON_FENCE_RE.search(cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        # A model that wrapped the JSON in prose still has a JSON object
        # in there; find its braces and try that.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start:end + 1]
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    bullets = data.get("bullets")
    if not isinstance(bullets, list):
        return []
    return [b for b in bullets if isinstance(b, dict)]


def _clean_bullet(bullet: dict) -> dict | None:
    """One bullet, normalised, or None if it is not usable.

    A bullet without text is nothing.  A bullet without a trigger is
    still worth keeping -- the trigger is a convenience, and the text
    can stand on its own -- but a bullet whose text is a meta-comment
    ("the bad run failed") is worth nothing, and it is the most common
    thing a model produces when asked to compare two runs.
    """
    text = str(bullet.get("text") or "").strip()
    if not text:
        return None
    if len(text) > MAX_BULLET_CHARS:
        text = text[:MAX_BULLET_CHARS].rstrip()
    lowered = text.lower()
    for phrase in ("in the bad run", "in the failing run", "the bad run",
                   "the failing run", "this run", "the run that"):
        if lowered.startswith(phrase):
            return None
    trigger = str(bullet.get("trigger") or "").strip()
    if len(trigger) > 60:
        trigger = trigger[:60].rstrip()
    return {"op": "insert", "text": text, "trigger": trigger}


# ── the reflector ───────────────────────────────────────────────────────────

async def reflect_contrast(ask, good: dict, bad: dict) -> list:
    """Ask a model what the difference between two runs was.

    ``ask(messages)`` returns the model's text.  It is the same shape
    the fact-checker takes, so the caller can hand over a model client
    that already knows how to talk to the endpoint.

    Returns a list of deltas ready for ``playbook.apply_deltas``.  Empty
    when the model said nothing usable, and empty without raising when
    the call failed: a reflection that cannot run is no reflection, and
    the playbook it would have written is better absent than guessed.
    """
    if not good or not bad:
        return []
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _user_prompt(good, bad)},
    ]
    try:
        response = await ask(messages)
    except Exception as e:  # pragma: no cover - network failure
        logger.debug("model reflection failed: %s", e)
        return []
    if isinstance(response, tuple):
        response = response[0]
    bullets = _extract_json(str(response or ""))
    deltas = []
    for bullet in bullets[:MAX_BULLETS_PER_CONTRAST]:
        cleaned = _clean_bullet(bullet)
        if cleaned:
            cleaned["provenance"] = [
                good.get("session_id", "?"), bad.get("session_id", "?")
            ]
            deltas.append(cleaned)
    return deltas


async def reflect_corpus(ask, records: list, *, max_contrasts: int = 8,
                         config_data: dict | None = None) -> list:
    """Reflect over a corpus, one contrasting pair at a time.

    The pairs come from the same clustering the mechanical reflector
    uses -- two runs of the same task shape, one of which went well and
    one of which did not -- because a contrast between runs of *different*
    shapes says nothing transferable.

    The holdout is excluded here for the same reason it is excluded
    everywhere else: a bullet written from a holdout run is a bullet
    written from the exam.

    Returns the deltas, in the order the contrasts were read.
    """
    from .playbook import cluster, contrast

    if config_data is not None:
        from .eval import filter_excluded
        records = filter_excluded(records, config_data)

    deltas = []
    seen = set()
    for group in cluster(records):
        sides = contrast(group["records"])
        good, bad = sides["good"], sides["bad"]
        if not good or not bad:
            continue
        # One pair per cluster, and a pair that has already been
        # reflected on is not reflected on again: the same two runs
        # produce the same bullet, and a playbook of near-duplicates is a
        # playbook nobody reads.
        pair = (good[0].get("session_id"), good[0].get("turn"),
                bad[0].get("session_id"), bad[0].get("turn"))
        if pair in seen:
            continue
        seen.add(pair)
        deltas.extend(await reflect_contrast(ask, good[0], bad[0]))
        if len(deltas) >= max_contrasts * MAX_BULLETS_PER_CONTRAST:
            break
    return deltas
