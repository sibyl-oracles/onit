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

The instrument: replay the split through the agent and grade the answers.

``eval.py`` holds the ruler as *data* -- a frozen holdout, a dev split, a
pinned baseline.  This module is the thing that *reads* the ruler and
produces a number, which is what was missing: without it the holdout is
a file nobody replays, and "the agent is better" is a claim with no
measurement behind it.

**What a run does.**  For each item in a split, the real agent is asked
the question, the answer is graded against the recorded reference answer,
and the result is written down.  Two arms -- one at ``observe`` and one
at ``adapt`` -- are then compared, and the comparison is a delta with an
error bar, not a vibe.

**Why the agent is driven directly.**  Through ``OnIt.process_task``,
the same way the benchmark provider drives it, so that a number produced
here means what it would mean in a real run: the same prompt
engineering, the same tools, the same loops, the same fact-check.  A
runner that called the model directly would be measuring the model.

**Why each arm gets its own store.**  The two arms must not contaminate
each other.  The control arm's trajectories are not the treatment arm's
recall pool, and neither arm's trajectories are the operator's own
corpus.  Each arm gets a directory of its own, and the runner reads
*its* store for recall and *its* store for nothing else.

**Why the grade is a separate model.**  The agent that answers is not
the agent that grades.  A self-graded answer is not a measurement, and
one that scored itself would report a number the agent controls.
"""

import inspect
import json
import logging
import os
import shutil
import tempfile
import time

logger = logging.getLogger(__name__)

# The result file's name inside an arm's directory.
RESULTS_NAME = "results.json"

# The grader's rubric.  Three grades, and the third is the one that
# matters: a run that hedges without committing is not wrong, it is
# unmeasured, and counting it as wrong would make "say less" look like
# a strategy.
GRADE_CORRECT = "C"
GRADE_INCORRECT = "I"
GRADE_NOT_ATTEMPTED = "N"

_GRADE_VALUES = (GRADE_CORRECT, GRADE_INCORRECT, GRADE_NOT_ATTEMPTED)

_RUBRIC = (
    "Grade the submitted answer against the reference answer.\n"
    "Reply with exactly one line: GRADE: C (correct), GRADE: I (incorrect), "
    "or GRADE: N (not attempted / refused / hedged without committing).\n"
    "An answer is correct only if it fully contains the reference answer's "
    "factual content without contradicting it."
)

# The arms a run compares.  ``off`` is the control: the scaffold as
# shipped.  ``adapt`` is the treatment: the same scaffold with recall
# and the playbook injected.
ARM_CONTROL = "off"
ARM_TREATMENT = "adapt"


# ── the grader ──────────────────────────────────────────────────────────────

def _grade_prompt(task: str, reference: str, answer: str) -> str:
    """The grading prompt, in the shape the rubric expects."""
    return (
        "%s\n\n"
        "Question:\n%s\n\n"
        "Reference answer:\n%s\n\n"
        "Submitted answer:\n%s"
        % (_RUBRIC, task, reference, answer)
    )


def parse_grade(text: str) -> str:
    """The grade in a grader's reply, or an empty string.

    Strict about the letter and forgiving about the prose around it: a
    grader that explains itself and then says "GRADE: C" is a grader
    that said C.
    """
    if not text:
        return ""
    upper = str(text).upper()
    marker = upper.find("GRADE:")
    if marker < 0:
        return ""
    tail = upper[marker + len("GRADE:"):].strip()
    for value in _GRADE_VALUES:
        if tail.startswith(value):
            return value
    return ""


async def grade(ask, task: str, reference: str, answer: str) -> dict:
    """Grade one answer.  Returns ``{"grade", "raw"}``.

    A failed call grades as empty rather than as wrong: an unreachable
    grader is a missing measurement, and a missing measurement that
    counted as a failure would make the arms look different for
    reasons that have nothing to do with the agent.
    """
    if not answer or not answer.strip():
        return {"grade": GRADE_NOT_ATTEMPTED, "raw": ""}
    messages = [
        {"role": "system", "content": "You grade answers. Reply with one line."},
        {"role": "user", "content": _grade_prompt(task, reference, answer)},
    ]
    try:
        raw = await ask(messages)
    except Exception as e:  # pragma: no cover - network failure
        logger.debug("grading call failed: %s", e)
        return {"grade": "", "raw": ""}
    if isinstance(raw, tuple):
        raw = raw[0]
    raw = str(raw or "")
    return {"grade": parse_grade(raw), "raw": raw[:500]}


# ── the run ─────────────────────────────────────────────────────────────────

def _arm_store(root: str, arm: str) -> str:
    """A store directory for one arm."""
    path = os.path.join(root, "store", arm)
    os.makedirs(path, exist_ok=True)
    return path


def _items_for(split: str, config_data: dict | None = None) -> list:
    """The items of a split, by name."""
    from .eval import read_dev, read_holdout
    if split == "holdout":
        return read_holdout(config_data)
    if split == "dev":
        return read_dev(config_data)
    raise ValueError("unknown split: %r" % (split,))


def _config_for_arm(base_config: dict, arm: str, store: str) -> dict:
    """The agent config for one arm.

    The arm differs in two ways and they are both load-bearing:
    ``autonomy`` decides whether anything is injected at all, and
    ``learn.path`` decides whose corpus the injection reads.  Setting
    only the first would have the treatment arm recall the *operator's*
    corpus, which is not the treatment arm's experience.
    """
    config = dict(base_config)
    learn = dict(config.get("learn") or {})
    learn["autonomy"] = arm
    learn["path"] = store
    config["learn"] = learn
    return config


def _record_arm_result(path: str, results: list) -> None:
    """Write an arm's results atomically."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp, path)


async def run_arm(items: list, *, arm: str, root: str,
                  base_config: dict | None = None,
                  make_agent=None, ask=None,
                  limit: int | None = None) -> list:
    """Run one arm over a list of items and grade the answers.

    ``make_agent(config)`` returns an object with ``process_task``; it
    is injected so that the runner is testable without an endpoint and
    without a tool registry.  ``ask`` is the grader's client.

    Returns a list of result dicts, one per item.
    """
    from .eval import eval_dir
    from .trajectory import trajectory_dir

    base_config = dict(base_config or {})
    store = _arm_store(root, arm)
    config = _config_for_arm(base_config, arm, store)
    # The ruler must be visible to the arm but not writable by it, and
    # the arm's own store must not be the operator's corpus.
    config.setdefault("learn", {})
    if isinstance(config["learn"], dict):
        config["learn"].setdefault("eval", {"path": eval_dir(base_config)})

    agent = make_agent(config) if make_agent else None
    if agent is None:
        raise ValueError("run_arm needs a make_agent callable")
    if inspect.isawaitable(agent):
        # The factory may be async -- the real one is, because the
        # agent's constructor cannot run inside a running event loop.
        # Awaiting it here is the difference between an arm that runs
        # the agent and an arm that quietly records an error per item
        # and scores zero.
        agent = await agent

    results = []
    subset = items[:limit] if limit else items
    for item in subset:
        task = item.get("task") or ""
        reference = item.get("response") or ""
        started = time.time()
        answer = ""
        error = ""
        try:
            answer = await agent.process_task(
                task,
                session_path=os.path.join(store, "sessions",
                                         "%s.jsonl" % (item.get("id") or "item")),
                data_path=os.path.join(store, "data"),
                safety_queue=None,
            )
        except Exception as e:
            error = str(e)[:300]
        verdict = await grade(ask, task, reference, answer) if ask else {
            "grade": "", "raw": ""}
        results.append({
            "id": item.get("id"),
            "task": task,
            "answer": answer,
            "grade": verdict["grade"],
            "error": error,
            "wall_s": round(time.time() - started, 3),
            "arm": arm,
        })
    _record_arm_result(os.path.join(root, "arms", arm, RESULTS_NAME), results)
    return results


# ── the comparison ──────────────────────────────────────────────────────────

def summarise(results: list) -> dict:
    """The headline numbers for one arm.

    The attempted rate is reported next to the accuracy, and it is not
    decoration.  An arm that answers less is graded more often as
    "not attempted", and an arm that hedges is not more correct -- it
    is less measured.  A comparison that reports accuracy alone would
    let that pass as a win.
    """
    total = len(results)
    graded = [r for r in results if r.get("grade")]
    correct = sum(1 for r in graded if r["grade"] == GRADE_CORRECT)
    incorrect = sum(1 for r in graded if r["grade"] == GRADE_INCORRECT)
    abstained = sum(1 for r in graded if r["grade"] == GRADE_NOT_ATTEMPTED)
    errors = sum(1 for r in results if r.get("error"))
    accuracy = (correct / total) if total else 0.0
    return {
        "n": total,
        "graded": len(graded),
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": abstained,
        "ungraded": total - len(graded),
        "errors": errors,
        "accuracy": accuracy,
    }


def _stderr(p: float, n: int) -> float:
    """The standard error of a proportion at ``n`` trials."""
    if n <= 0:
        return 0.0
    return (p * (1.0 - p) / n) ** 0.5


def compare(control: list, treatment: list) -> dict:
    """Compare two arms, with an error bar on the difference.

    The error bar is the *conservative* one: the sum of the two arms'
    standard errors, which is the right comparison for two independent
    proportions and is larger than the pooled estimate.  A delta that
    does not clear it is not a result, and the caller should say so.
    """
    c = summarise(control)
    t = summarise(treatment)
    delta = t["accuracy"] - c["accuracy"]
    err = (_stderr(c["accuracy"], c["n"]) + _stderr(t["accuracy"], t["n"]))
    # Significant when the delta is bigger than the error bar.  A zero
    # error bar is the case of perfect separation -- one arm got
    # everything right and the other got everything wrong -- and a
    # non-zero delta against it is the strongest result there is, so
    # the test must not require the error bar to be positive.  A zero
    # delta against a zero error bar means the arms are identical,
    # which is not a result in either direction.
    significant = abs(delta) > err and delta != 0.0
    return {
        "control": c,
        "treatment": t,
        "delta": delta,
        "stderr": err,
        "significant": significant,
        "n_items": min(c["n"], t["n"]),
    }


def format_comparison(result: dict, *, label: str = "") -> str:
    """A comparison, as text."""
    c, t = result["control"], result["treatment"]
    lines = []
    if label:
        lines.append(label)
    lines.append("  control  (%s): %.3f  n=%d  correct=%d  wrong=%d  "
                 "not-attempted=%d  ungraded=%d  errors=%d"
                 % (ARM_CONTROL, c["accuracy"], c["n"], c["correct"],
                    c["incorrect"], c["not_attempted"], c["ungraded"],
                    c["errors"]))
    lines.append("  treatment(%s): %.3f  n=%d  correct=%d  wrong=%d  "
                 "not-attempted=%d  ungraded=%d  errors=%d"
                 % (ARM_TREATMENT, t["accuracy"], t["n"], t["correct"],
                    t["incorrect"], t["not_attempted"], t["ungraded"],
                    t["errors"]))
    verdict = "SIGNIFICANT" if result["significant"] else "not significant"
    lines.append("  delta: %+.3f  (stderr %.3f)  -> %s"
                 % (result["delta"], result["stderr"], verdict))
    if not result["significant"]:
        lines.append("  A delta smaller than its error bar is not a result. "
                     "Ship nothing on this evidence.")
    return "\n".join(lines)
