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

Fact-checking the answer after it has been written.

The final answer is composed from whatever survives in the model's context by
the time it writes: tool results from ten turns ago, already decayed to a
summary, next to whatever the weights remember.  That is where a figure drifts
by a digit and a date lands in the wrong year — the tool result was right and
the sentence quoting it is not.

So the answer is checked against the evidence that produced it *after* it is
written, not instead of writing it.  The draft still streams to the user at
full speed — nothing here delays a token — and the check runs on the finished
text.  Claims the evidence covers are settled from the transcript for free;
claims it does not cover are looked up with read-only tools, but only when the
checker asks for them.  A clean verdict returns the draft untouched, which is
the common case and costs one small call.

Everything here fails open: an unparseable verdict, a failed call, a revision
that comes back empty or gutted, and the draft is what the user gets.  A
fact-checker that can turn a good answer into a worse one is not worth having.
"""

import json
import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# How much of the transcript is shown to the checker, and how much of any one
# tool result.  Evidence is what the answer was built from, so it is worth real
# context — but the checker is a small call by design, and a single 16k tool
# result would otherwise crowd out the five that matter.
MAX_EVIDENCE_CHARS = 8000
MAX_EVIDENCE_PER_ITEM = 1200

# The draft is quoted in full: checking half an answer finds the errors in half
# an answer.  Past this it is a document, and the claims worth checking are in
# the part a reader actually reads.
MAX_DRAFT_CHARS = 12000

# Output budget for the two calls.  A verdict is a short JSON list — 79 tokens
# for a two-claim finding on the model this was measured against — and the
# caller asks for it with thinking switched off.  The ceiling is nonetheless
# set for a model that thinks anyway, because some will: a server whose chat
# template has no such switch, or a provider that ignores it.  Too small a
# budget there does not buy a terser verdict, it buys finish_reason=length and
# no verdict at all.  The revision is the answer again, so it gets room for one.
VERDICT_MAX_TOKENS = 4096
REVISION_MAX_TOKENS = 8192

# Below this, an answer is a sentence — a greeting, a confirmation, a question
# back to the user — and there is nothing in it to check.
MIN_ANSWER_CHARS = 80

# A revision that comes back this much shorter than the draft did not correct
# the answer, it replaced it with a summary or an apology.  Keep the draft.
MIN_REVISION_RATIO = 0.5

# Claims worth the call.  Verification is only useful where an answer asserts
# something that could be wrong: a quantity, a date, a version, a named entity,
# an address, a quote, or an absolute.  Prose that does none of that ("I've
# saved the file, let me know if you'd like it formatted differently") has no
# truth to check, and running a model over it buys nothing but latency.
_CLAIM_PATTERNS = (
    r"\d",                                   # any number: counts, dates, versions, money
    r"https?://|www\.",                      # links
    r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",          # addresses
    r"[\"“][^\"”]{12,}[\"”]",                # quoted material
    r"\b(?:according to|reported|published|founded|released|announced|states?"
    r"|discovered|invented|located in|based in|named after|authored)\b",
    r"\b(?:always|never|only|first|last|largest|smallest|fastest|slowest"
    r"|best|worst|most|least|no other)\b",
)
_CLAIM_RE = re.compile("|".join(_CLAIM_PATTERNS), re.IGNORECASE)

# Answers that report on the run itself rather than about the world.  "I could
# not find anything" and "I stopped after 50 steps" are true by construction —
# they describe what just happened — and fact-checking them means asking a
# model whether the agent's own account of its own failure is accurate.
_SELF_REPORT_PREFIXES = (
    "i stopped after", "i couldn't", "i could not", "i wasn't able",
    "i was not able", "i don't have", "i do not have", "i'm sorry",
    "i am sorry", "this model", "sorry,",
)


def needs_verification(answer: str, min_chars: int = MIN_ANSWER_CHARS) -> bool:
    """Whether this answer asserts anything a fact-check could settle."""
    text = (answer or "").strip()
    if len(text) < min_chars:
        return False
    if text.lower().startswith(_SELF_REPORT_PREFIXES):
        return False
    return bool(_CLAIM_RE.search(text))


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} chars omitted]"


def _text_of(content: Any) -> str:
    """Message content as text, whatever shape the provider used for it."""
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content
                        if isinstance(part, dict))
    return str(content or "")


def evidence_digest(messages: list, max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    """The sourced material from a run, newest first, as plain text.

    Only tool results and what the user supplied count as evidence.  The
    assistant's own turns are excluded on purpose: an answer that cites an
    earlier sentence of its own is exactly the failure being looked for, and
    feeding those back would let a claim corroborate itself.

    Newest first because the tail of a long run is what the answer was written
    from, and it is the tail that survives the character budget.
    """
    items: list[str] = []
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "tool":
            name = msg.get("name") or "tool"
            body = _clip(_text_of(msg.get("content")).strip(), MAX_EVIDENCE_PER_ITEM)
            if body:
                items.append(f"[{name}]\n{body}")
        elif role == "user":
            body = _clip(_text_of(msg.get("content")).strip(), MAX_EVIDENCE_PER_ITEM)
            if body:
                items.append(f"[user said]\n{body}")

    out: list[str] = []
    used = 0
    for item in items:
        if used + len(item) > max_chars:
            break
        out.append(item)
        used += len(item) + 2
    return "\n\n".join(out)


_VERDICT_SYSTEM = """\
You are a fact-checker. You are given a user's request, the evidence gathered \
while answering it, and a draft answer. Find the claims in the draft that the \
evidence contradicts or fails to support.

What counts as a claim: quantities, dates, versions, names, locations, quotes, \
URLs, file or code contents, and statements of cause or definition.
What does not: tone, formatting, structure, opinions, recommendations, offers \
of further help, and anything the draft explicitly marks as uncertain.

Rules:
- The evidence is the ground truth. Where it contradicts the draft, the draft \
is wrong.
- A claim the evidence supports is fine. So is uncontroversial common \
knowledge. Do not manufacture doubt.
- If a claim matters, the evidence does not cover it, and a tool can settle \
it, call the tool. Do not guess, and do not report a claim as unsupported \
without trying the tool first.
- Do not rewrite the answer and do not comment on its style. Report only.

When you are done checking, reply with JSON and nothing else:
{"issues": [{"claim": "<quoted from the draft>", "problem": "contradicted|\
unsupported", "correction": "<what is actually true, in one short phrase>"}]}

Reply with {"issues": []} when every claim holds up. That is the expected \
result for most answers."""


def build_verdict_messages(task: str, answer: str, evidence: str) -> list[dict]:
    """The fact-checker's opening request."""
    parts = [f"## The user asked\n{_clip(task or '', 2000)}"]
    if evidence:
        parts.append(f"## Evidence gathered while answering\n{evidence}")
    else:
        parts.append("## Evidence gathered while answering\n"
                     "(nothing was gathered — the draft was written from the "
                     "model's own knowledge)")
    parts.append(f"## Draft answer to check\n{_clip(answer, MAX_DRAFT_CHARS)}")
    return [
        {"role": "system", "content": _VERDICT_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)

# Reasoning a model emitted inline rather than in a separate field.  It is
# prose about the claims, and it is full of braces and quoted fragments of the
# draft — everything the object scanner below is looking for.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> Optional[dict]:
    """The first JSON object in a reply that may be wrapped in prose or fences."""
    if not text:
        return None
    text = _THINK_RE.sub("", text)
    # An unterminated think block means the reply is reasoning all the way
    # down; there is no verdict after it to find.
    if "<think>" in text.lower():
        text = text[:text.lower().index("<think>")]
    candidates: list[str] = []
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start = text.find("{")
    if start != -1:
        # Scan to the matching brace rather than the last one in the string:
        # a reply that follows the JSON with prose containing a brace would
        # otherwise swallow it and fail to parse.
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_verdict(text: str) -> Optional[list[dict]]:
    """The issues a verdict reported, or None when it could not be read.

    None and [] are different answers and the caller acts on them differently:
    [] is a clean bill of health, None is a checker that did not report one.
    Neither changes the draft, but only the first should be logged as a pass.
    """
    parsed = _extract_json_object(text)
    if parsed is None:
        return None
    issues = parsed.get("issues")
    if issues is None:
        # A checker that replied {"verdict": "ok"} said the same thing the
        # empty list says.  Anything else is a shape we don't understand.
        verdict = str(parsed.get("verdict", "")).lower()
        return [] if verdict in ("ok", "pass", "verified", "correct") else None
    if not isinstance(issues, list):
        return None
    cleaned: list[dict] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        claim = str(issue.get("claim", "")).strip()
        correction = str(issue.get("correction", "")).strip()
        if not claim and not correction:
            continue
        cleaned.append({
            "claim": claim,
            "problem": str(issue.get("problem", "unsupported")).strip().lower(),
            "correction": correction,
        })
    return cleaned


def build_revision_messages(task: str, answer: str, issues: list[dict],
                            evidence: str) -> list[dict]:
    """The request that turns a checked draft into the answer the user keeps."""
    findings = "\n".join(
        f"- {issue['claim'] or '(claim)'} — {issue['problem']}: "
        f"{issue['correction'] or 'not supported by the evidence'}"
        for issue in issues
    )
    system = (
        "You revise a draft answer so that it is factually correct. Apply "
        "exactly the corrections listed and change nothing else: keep the same "
        "structure, formatting, level of detail, and language. Where a claim "
        "cannot be supported, drop it or attribute it honestly rather than "
        "restating it. Do not mention that a revision happened, do not add a "
        "preamble or a summary of your changes, and do not address the "
        "fact-checker. Output the corrected answer alone."
    )
    parts = [f"## The user asked\n{_clip(task or '', 2000)}"]
    if evidence:
        parts.append(f"## Evidence\n{evidence}")
    parts.append(f"## Corrections to apply\n{findings}")
    parts.append(f"## Draft answer\n{_clip(answer, MAX_DRAFT_CHARS)}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# How much of the note one finding may take before the rest are counted rather
# than quoted.  The note is a line under an answer, not a report.
_NOTE_ITEM_CHARS = 90
_NOTE_MAX_CHARS = 240


def revision_note(issues: list[dict]) -> str:
    """One line saying what the fact-check changed.

    Built here rather than asked of the model: a note is only worth showing if
    it reliably describes the corrections that were actually applied, and a
    model asked to summarize its own edit writes a sentence about diligence.
    """
    items: list[str] = []
    for issue in issues:
        text = issue.get("correction") or issue.get("claim") or ""
        text = " ".join(text.split())
        if not text:
            continue
        if len(text) > _NOTE_ITEM_CHARS:
            text = text[:_NOTE_ITEM_CHARS - 1] + "…"
        items.append(text)
    if not items:
        return ""
    note = "; ".join(items)
    if len(note) > _NOTE_MAX_CHARS:
        kept = 1
        note = items[0]
        while kept < len(items) and len(note) + len(items[kept]) + 2 <= _NOTE_MAX_CHARS:
            note += "; " + items[kept]
            kept += 1
        remaining = len(items) - kept
        if remaining > 0:
            note += f" (+{remaining} more)"
    return note


NOTE_PREFIX = "Revised after fact-check: "

# Used when the check found something but the answer could not be rewritten —
# it was too long to reproduce safely, or the rewrite came back unusable.  The
# finding is still the user's to know: an answer carrying a flagged error beats
# the same answer carrying a silent one.
CORRECTION_PREFIX = "Correction after fact-check: "


def append_note(answer: str, note: str, prefix: str = NOTE_PREFIX) -> str:
    """Attach the fact-check note to the answer the user sees.

    In the answer text rather than in a UI event, because every surface — web,
    terminal, Telegram, A2A, the saved session — shows the answer, and only one
    of them could be taught to render an event.  A user who watched a figure
    change on screen is owed the reason wherever they read it back.
    """
    if not note:
        return answer
    return f"{answer.rstrip()}\n\n*{prefix}{note}*"


async def verify_answer(
    *,
    task: str,
    answer: str,
    messages: list,
    ask: Callable,
    run_tools: Optional[Callable] = None,
    tools: Optional[list] = None,
    max_tool_turns: int = 2,
    verdict_max_tokens: int = VERDICT_MAX_TOKENS,
    revision_max_tokens: int = REVISION_MAX_TOKENS,
    log: Optional[Callable] = None,
) -> tuple[str, str, list[dict]]:
    """Check a finished answer against its evidence and correct it if needed.

    ``ask(messages, tools=None, max_tokens=N)`` runs one non-streaming
    completion and returns ``(content, tool_calls)``.  ``run_tools(tool_calls,
    messages)`` executes a batch and appends the assistant turn and its results
    to ``messages``.  Both are supplied by the caller so that provider
    handling, tool dispatch, and session policy stay where they already live.

    Returns ``(answer, note, issues)``.  The answer is ready to hand back —
    the note, when there is one, is already appended to it.  A clean check
    returns the draft byte for byte and an empty note.
    """
    def _log(message: str, level: str = "info") -> None:
        if log:
            log(message, level)

    evidence = evidence_digest(messages)
    verify_messages = build_verdict_messages(task, answer, evidence)

    issues: Optional[list[dict]] = None
    checked_with_tools = False
    for turn in range(max_tool_turns + 1):
        # Tools are offered only while there is budget left to run them; on the
        # last turn the checker is asked for its verdict and nothing else, so a
        # model that would keep looking things up forever still returns one.
        offer_tools = tools if (run_tools and turn < max_tool_turns) else None
        try:
            content, tool_calls = await ask(
                verify_messages, tools=offer_tools, max_tokens=verdict_max_tokens)
        except Exception as e:  # a failed check must not fail the answer
            logger.warning("Fact-check call failed: %s", e)
            _log(f"Fact-check skipped: {e}", "warning")
            return answer, "", []

        if tool_calls and run_tools:
            checked_with_tools = True
            try:
                await run_tools(tool_calls, verify_messages)
            except Exception as e:
                logger.warning("Fact-check tool call failed: %s", e)
                _log(f"Fact-check tool call failed: {e}", "warning")
                return answer, "", []
            continue

        issues = parse_verdict(content or "")
        break

    if issues is None:
        _log("Fact-check returned no readable verdict; keeping the answer as written.",
             "warning")
        return answer, "", []
    if not issues:
        _log("Fact-check found nothing to correct"
             + (" (after checking with tools)." if checked_with_tools else "."))
        return answer, "", []

    note = revision_note(issues)
    _log(f"Fact-check found {len(issues)} claim(s) to correct; revising the answer.",
         "warning")

    if len(answer) > MAX_DRAFT_CHARS:
        # Rewriting means reproducing, and what the revision call never sees it
        # cannot reproduce.  Past this length the tail would be quietly dropped
        # in exchange for a corrected opening, so the answer is kept whole and
        # the finding is flagged under it instead.
        _log("Answer is too long to rewrite safely; flagging the correction "
             "under it instead.", "warning")
        return append_note(answer, note, CORRECTION_PREFIX), note, issues

    try:
        revised, _ = await ask(
            build_revision_messages(task, answer, issues, evidence),
            tools=None, max_tokens=revision_max_tokens)
    except Exception as e:
        logger.warning("Answer revision failed: %s", e)
        _log(f"Revision failed; flagging the correction instead: {e}", "warning")
        return append_note(answer, note, CORRECTION_PREFIX), note, issues

    revised = (revised or "").strip()
    if not revised or len(revised) < len(answer.strip()) * MIN_REVISION_RATIO:
        # Not a correction — the model summarized, apologized, or gave up.
        # The draft with the finding flagged beats a paragraph that lost the
        # work, and beats the draft with the error left silent.
        logger.warning("Revision discarded: %d chars against a %d char draft",
                       len(revised), len(answer.strip()))
        _log("Revision came back too short to be the answer; flagging the "
             "correction under the original instead.", "warning")
        return append_note(answer, note, CORRECTION_PREFIX), note, issues
    return append_note(revised, note), note, issues
