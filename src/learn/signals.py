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

Outcome signals that cost nothing: what the *next* message implies.

The verifier's verdict is free but weak, and the explicit rating is strong but
arrives zero times in a thousand tasks because it needs a human to click
(``docs/SELF_IMPROVEMENT.md`` §4.1).  This module is the third kind: a signal
read off the conversation that follows the answer.

The reasoning is old and sound.  People say "that's not what I meant" and
"thanks, that worked" more often than they press a button, and both say
something about the answer that preceded them.  Reflexion and the Memp
lineage both lean on exactly this: the follow-up is the feedback.

**What is read.**

* *Rephrasing* — the next task in the same session restates the previous
  one.  A restatement is a failure the user chose to work around rather
  than to report, which makes it a signal about the *answer*, not about the
  task.
* *Accepting* — the next task continues the same thread without restating
  it, and the session then ends.  An unanswered thread is an accepted one.
* *Escalating* — the next task switches to a different tool or a different
  subject.  Weak, and recorded as such.

**What is deliberately not read.**  Nothing here looks at the *content* of
the answer to guess whether it was right.  There is no model in this
module, no heuristic on the answer text, and no judge: a signal that
guesses at correctness by reading prose is exactly the reward-hacking
failure §2.7 of the RFC is about.  These signals are about what the user
*did next*, which is the one thing the agent cannot fake.

**Strength.**  Every signal is a three-way value — ``good`` / ``bad`` /
``unknown`` — and ``unknown`` is the common one.  A signal that is wrong
half the time and silent the other half is worse than no signal, because
the loops downstream are built to trust it.  The counts are reported
honestly, including the unknowns, so the strength of the corpus is
visible rather than assumed.

**What this measures on the real corpus.**  On the 1,010 records that
existed when this was written, the signals label **173 of them (17.1%)**
— 69 good, 104 bad, 837 unlabelled — against **zero** explicit human
ratings.  That is the whole point of the exercise: 173 labels instead of
none, at no cost, retroactively.

The 17% is low and that is the honest number, not a bug.  Most tasks in
that corpus are one-offs in a session of one or two, and there is no
follow-up to read a signal from.  A corpus of longer sessions would
label more of it; a corpus of terser ones would label less.

**Where the false positives are.**  The heuristic cannot tell a *retry*
("run the tests" twice) from a *re-ask* ("run the tests" then "run the
tests, but with -v"), and it treats the first as a failure.  It is
wrong in that direction more often than the other: the corpus's most
common pattern is the user repeating a task because the first attempt
failed, which is exactly what a "bad" label should say — so the bias is
toward the label the corpus deserves.  But "do the same thing for X"
(the same shape, a different subject) is *not* a rephrase, and the
substitution check exists to keep those out.

**Retroactive.**  Every function here works on records already on disk.
The 1,010-record corpus gains a second label without waiting for a single
new session, which is the whole point: the data gate is open, and the
labelling was the part still missing.
"""

import re
from difflib import SequenceMatcher

# The phrases that mean "I am saying this again because you did not do it
# the first time".  Matched against the whole next task, at word
# boundaries, because a correction is often preceded by a courtesy:
# "thanks, no, do it again" is a rephrase and the "thanks" would
# otherwise win.
_REPHRASE_ANYWHERE = (
    "not quite", "that's not", "thats not", "that is not",
    "i said", "i already said", "i asked", "as i said", "like i said",
    "you didn't", "you did not", "you ignored", "you misunderstood",
    "that's wrong", "thats wrong", "that's not right", "thats not right",
    "incorrect", "wrong answer",
    "still not", "still doesn't", "still does not", "still isn't",
    "not working", "doesn't work", "does not work",
    "didn't work", "did not work",
    "nope", "try again", "not what i", "not what you",
)

# The same signal, but only trusted at the start of the task.  Each of
# these is an ordinary English word that appears in innocent tasks —
# "no" in "there is no way", "again" in "the same again" — so they count
# only where a correction would put them.
_REPHRASE_PREFIX = (
    "no", "not quite", "nope", "again", "wrong", "incorrect",
    "that's not", "thats not", "that is not", "i said", "i asked",
)

# The openers that mean "good, next thing" — a continuation, not a
# correction.  Kept small and explicit on purpose: a long list of
# sentiment words is a sentiment classifier, and this is not one.
#
# Every one of them is an *explicit* positive.  Continuation openers like
# "now" or "also" are deliberately absent: "now do it properly" is a
# correction that says nothing positive, and reading it as approval is
# the failure mode this module exists to avoid.
_ACCEPT_OPENERS = (
    "thanks", "thank you", "perfect", "great", "nice work",
    "that works", "that worked", "good", "ok", "okay", "cool", "nice",
    "awesome", "excellent", "got it", "yep", "yes", "done",
)

# A rephrase is judged on how much of the previous task it repeats.  The
# threshold is deliberately high: a follow-up that shares a few words with
# the task it follows is a *continuation* of it, and calling that a failure
# would label every multi-step task as a failure.
_SIMILARITY_THRESHOLD = 0.62

# Below this, two tasks are not comparable at all — "hi" vs "run the
# tests" share nothing, and a similarity score between them is noise.
_MIN_LENGTH_FOR_SIMILARITY = 8

# How much of a task's *meaning* has to be shared for it to be the same
# request.  Similarity above says "the shape is the same"; this says "the
# content is the same".  Both are needed, because they fail in opposite
# directions:
#
#   "run the unit tests"     -> "run the parser unit tests"
#   similarity 0.86, overlap 0.75 — the same request, said differently.
#
#   "benchmark mbpp ..."     -> "benchmark bigcodebench ..."
#   similarity 0.93, overlap 0.93 — the same *shape* with a different
#   subject.  This is a new task, and calling it a rephrase would label
#   every "now do the same for X" as a failure of the previous answer.
#
# The threshold sits high on purpose: the corpus is full of "now do the
# same thing for <other subject>", and those are continuations.
_MIN_CONTENT_OVERLAP = 0.80

# A *substitution* is the other half of the same problem.  "mbpp" and
# "bigcodebench" are not two spellings of one word; one replaced the
# other, and the replacement is the point of the new task.  So the test
# is not "do the token sets differ" but "did a distinctive word get
# swapped out for a different distinctive one".
#
# The marker is that the *only* differences are content words.  If the
# two tasks differ by one content word and agree on everything else,
# they are the same sentence with the subject swapped — which is a new
# task, not a repeat of the old one.
_SUBSTITUTION_MIN_DIFFERENCES = 1

# Words that carry no subject: changing them does not change the task.
# "the", "file" and "now" are interchangeable between two requests that
# are otherwise identical; "mbpp" and "bigcodebench" are not.
_SUBJECT_MARKERS = frozenset("""
file files now then also add run make use get set new old update updated
write read show list check test tests code script data value values
result results output outputs step steps line lines number numbers
""".split())


def _numbers_differ(only_a: set, only_b: set) -> bool:
    """Whether the two tasks disagree on a number.

    "3-layer" and "2-layer" are different architectures.  "epoch 5" and
    "epoch 10" are different runs.  Either way the request changed, and
    a request that changed is not a request being retried.

    A number that appears in both is not a difference and is not
    counted, so "run test 1" and "run test 2" disagree even though
    "test" is common to both.
    """
    def numbers(words):
        return {w for w in words if w.isdigit()}

    a_numbers, b_numbers = numbers(only_a), numbers(only_b)
    return bool(a_numbers) and bool(b_numbers)

# The comparison runs on this, not on the raw text: punctuation, casing and
# the filler words that make two spellings of the same request look
# different.  Removed here rather than by a stopword list so that the
# comparison is one function, not a stack of them.
_FILLER = frozenset("""
a an the and or but if then than so as of to in on at for with from by into
about over under again between within without across after before during
through out up down off onto upon per via is are was were be been being am
do does did doing done can could would should will shall may might must
have has had i me my mine we us our ours you your yours it its they them
their he she his her this that these those there here what which who whom
whose when where why how please pls kindly hi hey hello just some any all
very really actually maybe let lets like want wants need needs help sure
also now new please again just really very much more most other another
""".split())

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> list:
    """Comparable tokens of a task: lowercased, filler removed.

    Numbers survive, and they have to: "a 3-layer CNN" and "a 2-layer
    CNN" are different requests, and a tokenizer that drops one-digit
    numbers cannot tell them apart.
    """
    return [w for w in _WORD_RE.findall((text or "").lower())
            if (w not in _FILLER and len(w) > 1) or w.isdigit()]


def similarity(a: str, b: str) -> float:
    """How much of task ``a`` task ``b`` repeats, in ``[0, 1]``.

    Token-level, not character-level: "add a test for the parser" and
    "please add a test for the parser" are the same request, and a
    character comparison rates them as barely similar.  Returns 0.0 when
    either side is too short to compare — two one-word tasks share nothing
    reliably, and pretending otherwise manufactures rephrases.
    """
    ta, tb = _tokens(a), _tokens(b)
    if len(ta) < 2 or len(tb) < 2:
        return 0.0
    if len("".join(ta)) < _MIN_LENGTH_FOR_SIMILARITY:
        return 0.0
    return SequenceMatcher(None, ta, tb).ratio()


def content_overlap(a: str, b: str) -> float:
    """How much of task ``a`` survives in task ``b``, by content word.

    The share of ``a``'s content words that also appear in ``b``,
    measured against ``a``'s length — so a longer ``b`` is not penalised
    for saying more.

    This is the half of the judgement that similarity cannot make.
    "Benchmark mbpp" and "benchmark bigcodebench" are 93% similar and
    share almost none of their content: the second is a *different task
    in the same shape*, and calling it a rephrase of the first would
    label every "now the same for the next dataset" as a failure.
    """
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


def is_substitution(a: str, b: str) -> bool:
    """Whether ``b`` is ``a`` with its *subject* swapped out.

    The corpus's most common false positive: "now do the same for
    <other subject>".  Those two tasks agree on everything except the
    thing that matters, and the new subject is the whole point of the
    second one.

    The test is deliberately narrow, because getting it wrong in the
    other direction loses real signal:

    * the two tasks must differ by a small number of words, and
    * at least one of the differing words must be a *subject* — a word
      that names what the task is about, not a word that describes how
      it is being done.

    "Run the unit tests" -> "run the parser unit tests" adds a word and
    changes no subject, so it is a rephrase.  "Benchmark mbpp" ->
    "benchmark bigcodebench" swaps a subject, so it is not.

    A number counts as a subject when it qualifies one.  "A 3-layer CNN"
    and "a 2-layer CNN" are different architectures, and asking for the
    second after the first is not the first one being retried — it is
    the first one being corrected *in a new task*.  The distinction
    that matters is whether the *whole request* is the same, and a
    changed parameter is a changed request.
    """
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return False
    only_a = ta - tb
    only_b = tb - ta
    if not only_a or not only_b:
        return False
    # Too many differences to be a one-word swap: these are different
    # tasks that happen to share a shape.
    if len(only_a | only_b) > _SUBSTITUTION_MIN_DIFFERENCES * 4:
        return False
    subject_a = {w for w in only_a if w not in _SUBJECT_MARKERS}
    subject_b = {w for w in only_b if w not in _SUBJECT_MARKERS}
    # A subject was taken out and a different one put in.
    if subject_a and subject_b:
        return True
    # Or a qualifying number changed: "3-layer" is not "2-layer".
    return _numbers_differ(only_a, only_b)


def _starts_with(text: str, openers) -> bool:
    """Whether the task opens with one of the phrases."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    for opener in openers:
        if lowered == opener.strip():
            return True
        if lowered.startswith(opener) and (
                len(lowered) == len(opener)
                or not lowered[len(opener)].isalnum()):
            return True
    return False


def _contains_phrase(text: str, phrases) -> bool:
    """Whether the task contains one of the phrases, at word boundaries.

    Word boundaries matter: "i said" must not match inside "i saidso" and
    "no" must not match inside "notebook".  A phrase that is a prefix of
    a longer word is not a phrase.
    """
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    padded = " %s " % lowered
    for phrase in phrases:
        if (" %s " % phrase) in padded:
            return True
    return False


def _clauses(text: str) -> list:
    """The task split into clauses, on punctuation.

    A correction often arrives behind a courtesy: "thanks, no, do it
    again" is a rephrase with a "thanks" in front of it, and checking
    only the first clause would read it as an acceptance.  Splitting on
    the punctuation that separates a courtesy from the correction it
    contradicts is what makes the two distinguishable.
    """
    parts = re.split(r"[,;:.!?]|\s-\s", (text or "").strip().lower())
    return [p.strip() for p in parts if p.strip()]


def is_rephrase(previous_task: str, next_task: str) -> bool:
    """Whether ``next_task`` is the user saying ``previous_task`` again.

    Three independent tests, and any one of them is enough:

    1. the next task opens with a rephrase cue ("no", "I said");
    2. a *later* clause opens with one ("thanks, no, do it again");
    3. the next task repeats most of the previous one.

    The third catches the paraphrased restatement — "run the tests" after
    "please run the unit tests" — and the first two catch the terse
    correction — "no" — that no similarity score would.
    """
    if not (previous_task or "").strip() or not (next_task or "").strip():
        return False
    if _starts_with(next_task, _REPHRASE_PREFIX):
        return True
    if _contains_phrase(next_task, _REPHRASE_ANYWHERE):
        return True
    clauses = _clauses(next_task)
    if len(clauses) > 1:
        for clause in clauses[1:]:
            if _starts_with(clause, _REPHRASE_PREFIX):
                return True
    if similarity(previous_task, next_task) < _SIMILARITY_THRESHOLD:
        return False
    # The same shape with a different subject is a new task, not a repeat.
    # "Benchmark mbpp" and "benchmark bigcodebench" are 93% similar and
    # share none of their content; the corpus is full of that pattern.
    if is_substitution(previous_task, next_task):
        return False
    return content_overlap(previous_task, next_task) >= _MIN_CONTENT_OVERLAP


def is_acceptance(previous_task: str, next_task: str) -> bool:
    """Whether ``next_task`` reads as "that worked, next thing".

    Only the explicit openers count.  A continuation that merely *looks*
    different from the previous task is not evidence of anything: the user
    may have moved on because the answer was wrong and did not bother
    saying so, and reading that as success would poison the label.
    """
    if not (next_task or "").strip():
        return False
    if is_rephrase(previous_task, next_task):
        return False
    return _starts_with(next_task, _ACCEPT_OPENERS)


def label_pair(previous_task: str, next_task: str) -> str:
    """The label one next-task implies about the answer before it.

    ``"bad"`` for a rephrase, ``"good"`` for an acceptance, ``"unknown"``
    for everything else.  The order matters: a task that opens with
    "thanks, now run it again" is a rephrase, not an acceptance, and
    checking acceptance first would label it wrong.
    """
    if is_rephrase(previous_task, next_task):
        return "bad"
    if is_acceptance(previous_task, next_task):
        return "good"
    return "unknown"


def label_session(records: list) -> list:
    """Label every task in one session, from what came after it.

    ``records`` is a session's task records in order.  The *last* task of
    a session is labelled from the session's end: a thread that was never
    picked up again is an accepted one, and it is the only acceptance
    signal available for a one-task session, which is the most common kind.

    Returns a list of labels, one per record, in the same order.

    The end-of-session acceptance is the one inference here that is not
    read off a *message*, and it is the weakest of the three.  A user who
    closed the tab because the answer was wrong looks exactly like one who
    closed the tab satisfied.  It is kept because a corpus with no
    positive labels at all cannot be ranked, and it is reported as
    ``"good"`` only for the last turn of a session — never for a turn in
    the middle of one, where a following task exists and is used instead.
    """
    labels = ["unknown"] * len(records)
    if not records:
        return labels
    for index in range(len(records) - 1):
        previous = records[index].get("task") or ""
        following = records[index + 1].get("task") or ""
        labels[index] = label_pair(previous, following)
    labels[-1] = "good"
    return labels


def rating_verdict(record: dict) -> str:
    """The verdict a human rating implies: "good", "bad" or "".

    A click is the strongest evidence in the corpus: it is the one
    thing the agent cannot influence and cannot argue with.  It was
    nevertheless invisible to every loop, because ``label_records``
    deliberately leaves a rated record's ``signals.implicit`` alone
    (a click beats a heuristic) and ``contrast`` read only the
    heuristic field.  A thumbs-down was written, shown in the status
    report, and consumed by nothing.

    This reads the rating directly, so a record that carries one can
    be placed on a side of a contrast without a heuristic standing in
    for it.  It returns "" when there is no rating, which means "the
    human said nothing" - not "the human was satisfied".
    """
    signals = record.get("signals") or {}
    rating = signals.get("user_rating")
    if rating is None:
        return ""
    try:
        value = int(rating)
    except (TypeError, ValueError):
        return ""
    if value > 0:
        return "good"
    if value < 0:
        return "bad"
    return ""


def label_records(records: list) -> list:
    """Annotate records with an implicit label, in place.

    Each record gains ``signals.implicit`` — ``"good"``, ``"bad"`` or
    ``"unknown"`` — and ``signals.implicit_source``.  A record that
    already carries an explicit human rating is left alone: a click is
    stronger evidence than a heuristic, and overwriting it would throw
    away the only high-quality label in the corpus.

    The source says *where* the label came from, and the difference
    matters: ``follow-up`` is read off a message the user wrote, and
    ``session-end`` is an inference from the absence of one.  A loop that
    trains on both cannot tell them apart unless the field is there.
    """
    for index, (record, label) in enumerate(zip(records, label_session(records))):
        signals = record.setdefault("signals", {})
        if signals.get("user_rating") is not None:
            signals.setdefault("implicit_source", "explicit-rating")
            continue
        # A label that is already on the record is authoritative.  The
        # heuristic reads the *next* task, and a record whose verdict
        # was written down - by the verifier, by a backfill, by an
        # operator - knows more than the heuristic does.  Overwriting
        # it would silently replace a measurement with a guess.
        if signals.get("implicit") is not None:
            continue
        signals["implicit"] = label
        if label == "unknown":
            signals["implicit_source"] = "none"
        elif index == len(records) - 1:
            signals["implicit_source"] = "session-end"
        else:
            signals["implicit_source"] = "follow-up"
    return records


def summarize(records: list) -> dict:
    """Count the implicit labels in a list of records.

    The counts include the unknowns on purpose.  A report that says "820
    good, 190 bad" without saying how many were unlabelled reads as a
    much stronger corpus than it is.
    """
    counts = {"good": 0, "bad": 0, "unknown": 0}
    explicit = 0
    for record in records:
        signals = record.get("signals") or {}
        if signals.get("user_rating") is not None:
            explicit += 1
        label = signals.get("implicit")
        if label in counts:
            counts[label] += 1
        else:
            counts["unknown"] += 1
    total = sum(counts.values())
    labeled = counts["good"] + counts["bad"]
    return {
        "total": total,
        "good": counts["good"],
        "bad": counts["bad"],
        "unknown": counts["unknown"],
        "labeled": labeled,
        "explicit_ratings": explicit,
        # Share of the corpus carrying a usable label, and share of the
        # *labeled* part that is positive.  Both are needed to read the
        # first one: a corpus that is 90% labeled and 5% positive is a
        # different thing from one that is 90% labeled and 95% positive.
        "coverage": round(labeled / total, 4) if total else 0.0,
        "positive_rate": round(counts["good"] / labeled, 4) if labeled else 0.0,
    }
