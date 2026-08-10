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
text.  A clean verdict returns the draft untouched, which is the common case
and costs one small call.

By default the check is settled from the transcript alone: the evidence the run
already gathered against the sentences that quote it.  That is where the drift
this catches actually happens, and it is one call with no lookups behind it.
Chasing claims the evidence never covered is a different and far more expensive
job — a round trip per lookup, a tool run, and another verdict call after each
— so it is off unless ``max_tool_turns`` is raised, and without it a claim the
evidence simply does not speak to is left alone rather than doubted.

The check runs in two stages, because correctness and latency want opposite
things.  The first stage is what the user waits for and is kept to roughly a
second: a free local pass that clears every figure already copied verbatim out
of a document or a reputable source, and at most one small verdict call on
what is left, under a hard deadline, correcting by note rather than by
rewrite.  The second stage runs behind the answer once it has been handed over
— lookups, uncovered claims, a real revision — and reports back only if it
finds something and the user has not moved on.  ``verify_answer`` serves both;
the caller decides which one it is asking for by what it allows.

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

# Output budget for the two calls.  A verdict is a short JSON list: measured at
# 6 tokens for a clean bill of health and 108 for a two-claim finding, with
# thinking switched off as the caller asks for it.  512 is that with room to
# spare, and a model that overruns it was reasoning rather than reporting.
# THINKING_VERDICT_MAX_TOKENS is for the hosts that cannot switch thinking off
# at all: there the same verdict arrives behind 1,100–1,500 tokens of
# reasoning, and too small a budget does not buy a terser verdict, it buys
# finish_reason=length and no verdict.  The revision is the answer again, so it
# gets room for one.
VERDICT_MAX_TOKENS = 512
THINKING_VERDICT_MAX_TOKENS = 4096
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


# ── the free pass ───────────────────────────────────────────────────────────
#
# A figure that appears verbatim in a document the run read, or on a source
# that is reputable about the thing being claimed, has already been checked:
# the answer copied it correctly, and that is the whole question.  Deciding
# that takes string comparison, not a model, so it happens first and costs
# nothing — and when it clears the answer there is no call at all, in front of
# the user or behind them.
#
# Tool results that are a source in their own right.  These read something the
# user pointed at: their files, their documents, their directories.  There is
# no more authoritative version of the user's own file to check it against.
TRUSTED_TOOLS = frozenset({
    "read_file", "search_document", "get_document_context", "local_search",
    "extract_tables", "grep", "find_files", "search_directory",
})

# Domains taken at their word.  Deliberately short and boring: reference works,
# primary sources, and the places that publish the record rather than report on
# it.  Entries beginning with a dot match any host ending in them, so ".gov"
# covers a country's ministries without listing them.  Extend per deployment
# with `verify_trusted_domains` rather than editing this.
DEFAULT_TRUSTED_DOMAINS = (
    ".gov", ".edu", ".int", ".mil",
    "wikipedia.org", "wikidata.org", "britannica.com",
    "arxiv.org", "doi.org", "pubmed.ncbi.nlm.nih.gov", "nature.com",
    "science.org", "ieee.org", "acm.org",
    "who.int", "un.org", "worldbank.org", "imf.org", "oecd.org",
    "europa.eu", "iso.org", "ietf.org", "rfc-editor.org", "w3.org",
    "python.org", "docs.python.org", "developer.mozilla.org",
    "github.com", "gitlab.com", "pypi.org", "npmjs.com",
)

_URL_RE = re.compile(r"https?://([^\s/\"'>)\]]+)", re.IGNORECASE)

# What the free pass actually compares.  Numbers are the point — a figure
# drifting by a digit is the failure this whole file exists for — alongside the
# things that are wrong in the same literal way: a version, a URL, an address,
# a quoted line.  Prose is not compared: two sentences can say the same true
# thing in different words, and demanding a byte match would clear nothing.
_TOKEN_PATTERNS = (
    r"\d[\d,]*(?:\.\d+)?%?",                 # 3.1 · 4,200 · 61% · 2019
    r"https?://[^\s\"'>)\]]+",               # links
    r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",          # addresses
    r"[\"“]([^\"”]{12,})[\"”]",              # quoted material
)
_TOKEN_RE = re.compile("|".join(_TOKEN_PATTERNS))

# One-digit numbers are skipped: "2 files", "3 steps", a list that reached 4.
# They match any haystack that mentions any number, so requiring them proves
# nothing, and demanding them proves nothing either.
_MIN_TOKEN_LEN = 2


def _normalize(text: str) -> str:
    """Text as the free pass compares it: lowercase, no thousands separators."""
    return re.sub(r"(?<=\d),(?=\d)", "", (text or "").lower())


def claim_tokens(text: str) -> list[str]:
    """The literals in an answer that a source can confirm byte for byte."""
    out: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(text or ""):
        token = _normalize(match.group(1) or match.group(0)).strip().rstrip(".,;:")
        if len(token) < _MIN_TOKEN_LEN or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def is_trusted_url(url: str, domains: tuple = DEFAULT_TRUSTED_DOMAINS) -> bool:
    """Whether a host is one of the sources taken at its word."""
    # Takes a bare host or a whole URL: the caller may have either, and a
    # scheme left on the front would be read as the first path segment.
    host = re.sub(r"^[a-z][a-z0-9+.-]*://", "", (url or "").strip().lower())
    host = host.split("/")[0].split("?")[0].split("@")[-1].split(":")[0].strip()
    host = host[4:] if host.startswith("www.") else host
    if not host:
        return False
    for domain in domains:
        domain = str(domain).lower().lstrip("*").strip()
        if not domain:
            continue
        if domain.startswith("."):
            if host.endswith(domain):
                return True
        elif host == domain or host.endswith("." + domain):
            return True
    return False


def trusted_evidence(messages: list,
                     domains: tuple = DEFAULT_TRUSTED_DOMAINS) -> str:
    """The part of the transcript that counts as a source in its own right.

    Document tools qualify outright — they read the user's own material.  A web
    result qualifies only when every link in it is to a trusted host: a page of
    mixed search hits is not a reputable source just because one reputable
    source is among them, and there is no way to tell from here which hit a
    given sentence came from.

    The user's own turns are in here too.  A figure they typed is not something
    the model invented, and an answer quoting it back has nothing to get wrong.
    """
    parts: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            parts.append(_text_of(msg.get("content")))
            continue
        if role != "tool":
            continue
        body = _text_of(msg.get("content"))
        if not body.strip():
            continue
        name = str(msg.get("name") or "")
        if name in TRUSTED_TOOLS:
            parts.append(body)
            continue
        hosts = _URL_RE.findall(body)
        if hosts and all(is_trusted_url(h, domains) for h in hosts):
            parts.append(body)
    return "\n".join(parts)


def covered_by_trusted_sources(answer: str, messages: list,
                               domains: tuple = DEFAULT_TRUSTED_DOMAINS) -> bool:
    """Whether every checkable literal in the answer is already in a source.

    True means the answer copied its figures correctly out of material that is
    itself the authority, and no call of any kind is warranted.  False is not a
    finding — it only means something in the answer was not settled here.
    """
    tokens = claim_tokens(answer)
    if not tokens:
        # Nothing this pass can compare.  The answer may still assert plenty
        # ("the only implementation that never allocates"), so it is handed on
        # rather than cleared.
        return False
    haystack = _normalize(trusted_evidence(messages, domains))
    if not haystack:
        return False
    return all(token in haystack for token in tokens)


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


def evidence_digest(messages: list, max_chars: int = MAX_EVIDENCE_CHARS,
                    focus: Optional[list] = None) -> str:
    """The sourced material from a run, newest first, as plain text.

    Only tool results and what the user supplied count as evidence.  The
    assistant's own turns are excluded on purpose: an answer that cites an
    earlier sentence of its own is exactly the failure being looked for, and
    feeding those back would let a claim corroborate itself.

    Newest first because the tail of a long run is what the answer was written
    from, and it is the tail that survives the character budget.

    ``focus`` is the answer's own literals (see ``claim_tokens``).  Given them,
    the results that mention one are moved to the front, so what survives the
    budget is the evidence bearing on the figures actually being checked rather
    than whatever happened to run last.  Prompt size is latency here: every
    thousand characters that does not settle a claim is time the user waits.
    """
    items: list[tuple[bool, str]] = []
    focus = [f for f in (focus or []) if len(f) >= _MIN_TOKEN_LEN]
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "tool":
            label = f"[{msg.get('name') or 'tool'}]"
        elif role == "user":
            label = "[user said]"
        else:
            continue
        body = _clip(_text_of(msg.get("content")).strip(), MAX_EVIDENCE_PER_ITEM)
        if not body:
            continue
        hay = _normalize(body)
        items.append((any(token in hay for token in focus), f"{label}\n{body}"))

    if focus:
        # Stable: within the relevant group and within the rest, newest still
        # comes first.
        items.sort(key=lambda pair: not pair[0])

    out: list[str] = []
    used = 0
    for _, item in items:
        if used + len(item) > max_chars:
            break
        out.append(item)
        used += len(item) + 2
    return "\n\n".join(out)


_CLAIM_SCOPE = """\
What counts as a claim: quantities, dates, versions, names, locations, quotes, \
URLs, file or code contents, and statements of cause or definition.
What does not: tone, formatting, structure, opinions, recommendations, offers \
of further help, and anything the draft explicitly marks as uncertain."""

_VERDICT_FORMAT = """\
When you are done checking, reply with JSON and nothing else:
{"issues": [{"claim": "<quoted from the draft>", "problem": "contradicted|\
unsupported", "correction": "<what is actually true, in one short phrase>"}]}

Reply with {"issues": []} when every claim holds up. That is the expected \
result for most answers."""

# The default check.  It compares the draft against what the run already has
# and stops there: no lookups, so nothing to say about a claim the evidence
# never touched, and no reason to invite the model to speculate about one.
_VERDICT_SYSTEM_LIGHT = f"""\
You are a fact-checker. You are given a user's request, the evidence gathered \
while answering it, and a draft answer. Find the claims in the draft that the \
evidence directly contradicts.

{_CLAIM_SCOPE}

Rules:
- The evidence is the ground truth. Where it contradicts the draft, the draft \
is wrong.
- Report only contradictions you can point to in the evidence. A claim the \
evidence does not cover is not your concern — skip it. You have no way to look \
anything up, so do not guess and do not manufacture doubt.
- Do not rewrite the answer and do not comment on its style. Report only.

{_VERDICT_FORMAT}"""

# Used only when the caller grants tool turns.  Then an uncovered claim is
# worth raising, because there is a way to settle it.
_VERDICT_SYSTEM_TOOLS = f"""\
You are a fact-checker. You are given a user's request, the evidence gathered \
while answering it, and a draft answer. Find the claims in the draft that the \
evidence contradicts or fails to support.

{_CLAIM_SCOPE}

Rules:
- The evidence is the ground truth. Where it contradicts the draft, the draft \
is wrong.
- A claim the evidence supports is fine. So is uncontroversial common \
knowledge. Do not manufacture doubt.
- If a claim matters, the evidence does not cover it, and a tool can settle \
it, call the tool. Do not guess, and do not report a claim as unsupported \
without trying the tool first.
- Do not rewrite the answer and do not comment on its style. Report only.

{_VERDICT_FORMAT}"""


def build_verdict_messages(task: str, answer: str, evidence: str,
                           with_tools: bool = False) -> list[dict]:
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
        {"role": "system",
         "content": _VERDICT_SYSTEM_TOOLS if with_tools else _VERDICT_SYSTEM_LIGHT},
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
            # An issue reported without a label is taken as a contradiction:
            # both prompts ask for those, only one of them asks for anything
            # else, and a caller running without lookups drops the rest.  A
            # finding a model bothered to write down should not be discarded
            # over a missing field.
            "problem": str(issue.get("problem", "contradicted")).strip().lower(),
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
    max_tool_turns: int = 0,
    verdict_max_tokens: int = VERDICT_MAX_TOKENS,
    revision_max_tokens: int = REVISION_MAX_TOKENS,
    allow_revision: bool = True,
    trusted_domains: tuple = DEFAULT_TRUSTED_DOMAINS,
    log: Optional[Callable] = None,
) -> tuple[str, str, list[dict]]:
    """Check a finished answer against its evidence and correct it if needed.

    ``ask(messages, tools=None, max_tokens=N)`` runs one non-streaming
    completion and returns ``(content, tool_calls)``.  ``run_tools(tool_calls,
    messages)`` executes a batch and appends the assistant turn and its results
    to ``messages``.  Both are supplied by the caller so that provider
    handling, tool dispatch, and session policy stay where they already live.

    ``allow_revision`` is what separates the stage the user waits for from the
    one that runs behind them.  Rewriting an answer means generating it a
    second time, at the same tokens per second it was written the first time —
    the single most expensive thing in this file, and the one thing that cannot
    fit in a second.  With it off, a finding is reported as a note under the
    answer instead and the call returns immediately.

    Returns ``(answer, note, issues)``.  The answer is ready to hand back —
    the note, when there is one, is already appended to it.  A clean check
    returns the draft byte for byte and an empty note.
    """
    def _log(message: str, level: str = "info") -> None:
        if log:
            log(message, level)

    with_tools = bool(run_tools and tools and max_tool_turns > 0)
    if covered_by_trusted_sources(answer, messages, trusted_domains):
        # Every figure in the answer is already sitting in a document the run
        # read or on a source that is authoritative about it.  Nothing a model
        # could add here, so nothing is spent asking one.
        _log("Fact-check skipped: every figure in the answer matches a "
             "document, a trusted source, or what you asked.")
        return answer, "", []

    evidence = evidence_digest(messages, focus=claim_tokens(answer))
    if not evidence and not with_tools:
        # Nothing gathered and no way to gather anything: the only thing left to
        # check the draft against is the same weights that wrote it, which is a
        # call that cannot find what it is looking for.  Skip it.
        _log("Fact-check skipped: the run gathered no evidence to check against.")
        return answer, "", []

    verify_messages = build_verdict_messages(task, answer, evidence, with_tools)

    issues: Optional[list[dict]] = None
    checked_with_tools = False
    for turn in range(max_tool_turns + 1):
        # Tools are offered only while there is budget left to run them; on the
        # last turn the checker is asked for its verdict and nothing else, so a
        # model that would keep looking things up forever still returns one.
        offer_tools = tools if (with_tools and turn < max_tool_turns) else None
        try:
            content, tool_calls = await ask(
                verify_messages, tools=offer_tools, max_tokens=verdict_max_tokens)
        except Exception as e:  # a failed check must not fail the answer
            logger.warning("Fact-check call failed: %s", e)
            _log(f"Fact-check skipped: {e}", "warning")
            return answer, "", []

        # Only when tools were on the table.  A model that emits a call it was
        # never offered is not owed the round trip.
        if tool_calls and with_tools:
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

    if issues and not with_tools:
        # Without lookups there is no evidence behind "unsupported" — only the
        # absence of evidence, which is not a finding.  Rewriting an answer to
        # drop or hedge a claim on that basis makes it worse, so these are
        # counted and dropped rather than acted on.
        kept = [i for i in issues if i.get("problem") != "unsupported"]
        if len(kept) != len(issues):
            _log(f"Fact-check dropped {len(issues) - len(kept)} uncovered "
                 f"claim(s); the check ran against gathered evidence only.")
        issues = kept

    if issues is None:
        _log("Fact-check returned no readable verdict; keeping the answer as written.",
             "warning")
        return answer, "", []
    if not issues:
        _log("Fact-check found nothing to correct"
             + (" (after checking with tools)." if checked_with_tools else "."))
        return answer, "", []

    note = revision_note(issues)

    if not allow_revision:
        # The stage the user is waiting on.  A rewrite here would cost a whole
        # second generation of the answer, so the finding goes under it as a
        # line instead and the turn ends now.
        _log(f"Fact-check found {len(issues)} claim(s); flagging them under "
             f"the answer.", "warning")
        return append_note(answer, note, CORRECTION_PREFIX), note, issues

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
