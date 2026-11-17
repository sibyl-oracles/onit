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

import asyncio
import inspect
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


# The result file's name inside an arm's directory.
RESULTS_NAME = "results.json"

# The transcript budget for the gate's seeding pass.  A replay's whole
# day lands in one session file -- 20 items is one session id with 20
# turns -- and the production dream budget (6,000 characters) would
# show the pass only the first two items.  The treatment's wiki is
# distilled from the control's *entire* recorded experience or the gate
# measures a placebo.
SEED_MAX_SESSION_CHARS = 200_000

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
# The arm names that are autonomy levels.  The wiki gate's arms are
# labels, not levels, and must not be coerced through ``autonomy()``.
_LEVEL_ARMS = frozenset({ARM_CONTROL, ARM_TREATMENT})


def _wiki_agent_factory(config: dict):
    """Build the real agent for a wiki-gate arm, off the event loop.

    A thin re-export of ``ab._make_agent`` kept at module level so a
    test can patch it: the CLI passes no ``make_agent``, and the
    fallback must be reachable without importing ``ab`` at module load
    (``ab`` imports ``onit.onit`` lazily, inside its factory).
    """
    from .ab import _make_agent

    return _make_agent(config)


# ── the grader ──────────────────────────────────────────────────────────────


def _grade_prompt(task: str, reference: str, answer: str) -> str:
    """The grading prompt, in the shape the rubric expects."""
    return "%s\n\nQuestion:\n%s\n\nReference answer:\n%s\n\nSubmitted answer:\n%s" % (
        _RUBRIC,
        task,
        reference,
        answer,
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
    tail = upper[marker + len("GRADE:") :].strip()
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
        # Counted, not just logged: a grader that cannot be reached used
        # to leave only a debug line nobody read, and a run whose every
        # grade came back empty still printed a tidy 0.000 vs 0.000.  The
        # count surfaces in the summary as ``grading_errors`` and in the
        # verdict, which refuses to call an all-empty run a measurement.
        logger.warning("grading call failed: %s", e)
        return {
            "grade": "",
            "raw": "",
            "grading_error": "%s: %s" % (type(e).__name__, str(e)[:200]),
        }
    if isinstance(raw, tuple):
        raw = raw[0]
    raw = str(raw or "")
    if not raw.strip():
        # A call that succeeded but returned nothing is the failure mode
        # the 2026-11-17 gate run exposed: a reasoning model spends its
        # output budget thinking and never writes the verdict, and the
        # item silently fell out of the comparison -- 12 of 40 items,
        # with ``grading_errors`` reporting 0.  One retry, then the
        # item is recorded as a grading failure: an empty grade is not
        # data, and it must be visible in the summary, not only here.
        logger.warning("grading call returned no text; retrying once")
        try:
            raw = await ask(messages)
        except Exception as e:
            logger.warning("grading retry failed: %s", e)
            return {
                "grade": "",
                "raw": "",
                "grading_error": "retry after empty reply: %s: %s"
                % (type(e).__name__, str(e)[:200]),
            }
        if isinstance(raw, tuple):
            raw = raw[0]
        raw = str(raw or "")
        if not raw.strip():
            logger.warning("grading call returned no text twice; item left ungraded")
            return {
                "grade": "",
                "raw": "",
                "grading_error": "grading call returned no text (retried once)",
            }
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

    The autonomy assignment applies only when the arm name *is* a level
    (the classic off/adapt pair).  The wiki gate's arms are labels, not
    levels -- their autonomy is pinned by ``_config_for_wiki_arm``, and
    overwriting it here with "wiki-on" would coerce to nothing and
    silence both arms at once.
    """
    config = dict(base_config)
    learn = dict(config.get("learn") or {})
    if arm in _LEVEL_ARMS:
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


async def run_arm(
    items: list,
    *,
    arm: str,
    root: str,
    base_config: dict | None = None,
    make_agent=None,
    ask=None,
    limit: int | None = None,
    data_path: str | None = None,
) -> list:
    """Run one arm over a list of items and grade the answers.

    ``make_agent(config)`` returns an object with ``process_task``; it
    is injected so that the runner is testable without an endpoint and
    without a tool registry.  ``ask`` is the grader's client.

    ``data_path`` is the working directory every item runs in, and it
    must be the directory the tool servers were registered with -- they
    take it from the config's ``data_path`` at launch and refuse any
    path outside it.  It defaults to the config's own ``data_path`` so
    that a caller who does not think about it still gets a run whose
    tools work, rather than one that answers every task with a
    description of why its tools were refused.

    Returns a list of result dicts, one per item.
    """
    from .eval import eval_dir

    base_config = dict(base_config or {})
    store = _arm_store(root, arm)
    if data_path is None:
        data_path = base_config.get("data_path") or os.path.join(root, "data")
    data_path = os.path.realpath(os.path.abspath(os.path.expanduser(str(data_path))))
    os.makedirs(data_path, exist_ok=True)
    config = _config_for_arm(base_config, arm, store)
    # The data_path the factory resolves must be the data_path the items
    # run in.  ``_ensure_mcp_servers`` sets ``ONIT_DATA_PATH`` from the
    # config at factory time, and the MCP servers' jail accepts only
    # paths under that root -- so the resolved directory is stamped into
    # the config *before* ``make_agent`` runs.  A config without its own
    # ``data_path`` used to send the servers to the default sandbox while
    # every item ran under ``<root>/data``: every filesystem tool call
    # refused, both arms answering from the refusals, and a comparison
    # between two piles of refusals.  (``realpath`` because the jail
    # compares against the server's own resolved root; /private/var vs
    # /var on macOS must not decide this.)
    config["data_path"] = data_path
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
    label = getattr(agent, "label", arm)
    for index, item in enumerate(subset, 1):
        task = item.get("task") or ""
        reference = item.get("response") or ""
        # Progress, not decoration: an item takes minutes, and a run
        # that prints nothing between items is indistinguishable from
        # a hung one -- which is exactly what happened on the first
        # real wiki-gate run.  ``flush`` because stdout buffers when it
        # is not a terminal, and the point is to be visible.
        print(
            "  [%s] item %d/%d  %s" % (label, index, len(subset), (item.get("id") or "?")[:40]),
            flush=True,
        )
        started = time.time()
        answer = ""
        error = ""
        try:
            answer = await agent.process_task(
                task,
                session_path=os.path.join(
                    store, "sessions", "%s.jsonl" % (item.get("id") or "item")
                ),
                # The working directory must be the one the MCP tool
                # servers were registered with.  They take it from the
                # config's ``data_path`` at launch (``cli._ensure_mcp_servers``
                # sets ``ONIT_DATA_PATH``), and the servers' jail accepts
                # only paths under that root.  Two bugs lived here.  The
                # first passed ``<store>/data`` -- a directory the servers
                # never heard of -- so every tool call was refused.  The
                # second left the config's ``data_path`` unset, so the
                # servers went to the default sandbox while the items ran
                # under ``<root>/data``: the same refusal with a subtler
                # cause, and a run that looked healthy while both arms
                # scored on how well they described the refusals.  The
                # stamp above pins the config to this directory, so the
                # two sides cannot disagree again.
                data_path=data_path,
            )
        except Exception as e:
            error = str(e)[:300]
        verdict = await grade(ask, task, reference, answer) if ask else {"grade": "", "raw": ""}
        results.append(
            {
                "id": item.get("id"),
                "grade": verdict["grade"],
                "grading_error": verdict.get("grading_error", ""),
                "raw": verdict.get("raw", ""),
                "task": task,
                "answer": answer,
                "grade": verdict["grade"],
                "error": error,
                "wall_s": round(time.time() - started, 3),
                "arm": arm,
            }
        )
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
    # A grader that could not be reached is not the same as an answer
    # that was not given.  Counted separately so an unreachable grader
    # can be told apart from a hedging agent in the summary.
    grading_errors = sum(1 for r in results if r.get("grading_error"))
    accuracy = (correct / total) if total else 0.0
    return {
        "n": total,
        "graded": len(graded),
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": abstained,
        "ungraded": total - len(graded),
        "errors": errors,
        "grading_errors": grading_errors,
        "accuracy": accuracy,
    }


def _stderr(p: float, n: int) -> float:
    """The standard error of a proportion at ``n`` trials."""
    if n <= 0:
        return 0.0
    return (p * (1.0 - p) / n) ** 0.5


# How many standard errors a difference has to clear to be called a
# result.  1.96 is the two-sided 95% normal quantile: a null difference
# clears it 5% of the time, which is the false-positive rate this test is
# supposed to have.
#
# The test used to compare against ``se_c + se_t`` -- the *sum* of the two
# arms' standard errors -- and the docstring called that "conservative".
# It is the opposite.  With equal n the sum is ``2 * sqrt(p(1-p)/n)`` and
# the standard error of the *difference* is ``sqrt(2) * sqrt(p(1-p)/n)``,
# so the bar was ``sqrt(2)`` = 1.414 sigma rather than 1.96: a null
# difference cleared it 15.7% of the time, not 5%.  Measured on 3000
# simulated null pairs, 15.9%.  A placebo arm therefore had roughly a
# one-in-six chance of being reported as a win, which is the failure mode
# that matters -- not the missed real effect, but the shipped nothing
# that looked like something.
Z_95 = 1.96


def compare(control: list, treatment: list) -> dict:
    """Compare two arms, with an error bar on the difference.

    The error bar is the standard error of the *difference* of two
    independent proportions, times :data:`Z_95`:

        se_diff = sqrt(se_c**2 + se_t**2)
        bar     = Z_95 * se_diff

    Adding the variances rather than the standard errors is what makes
    the bar the right width: the two arms are independent samples, and
    the variance of their difference is the sum of their variances.

    A delta that does not clear the bar is not a result, and the caller
    should say so.

    ``stderr`` in the returned dict is the *bar* -- the quantity the
    delta is compared against -- so a caller that prints it is printing
    the thing the verdict was made on.  ``se_diff`` and ``z`` are
    returned alongside it so a reader can reconstruct the test.
    """
    c = summarise(control)
    t = summarise(treatment)
    delta = t["accuracy"] - c["accuracy"]
    se_c = _stderr(c["accuracy"], c["n"])
    se_t = _stderr(t["accuracy"], t["n"])
    se_diff = (se_c**2 + se_t**2) ** 0.5
    err = Z_95 * se_diff
    # Significant when the delta clears the bar.  A zero error bar is the
    # case of perfect separation -- one arm got everything right and the
    # other got everything wrong -- and a non-zero delta against it is the
    # strongest result there is, so the test must not require the error
    # bar to be positive.  A zero delta against a zero error bar means the
    # arms are identical, which is not a result in either direction.
    significant = abs(delta) > err and delta != 0.0
    return {
        "control": c,
        "treatment": t,
        "delta": delta,
        "stderr": err,
        "se_diff": se_diff,
        "z": Z_95,
        "significant": significant,
        "n_items": min(c["n"], t["n"]),
    }


def format_comparison(
    result: dict, *, label: str = "", arm_names: tuple[str, str] = (ARM_CONTROL, ARM_TREATMENT)
) -> str:
    """A comparison, as text.

    ``arm_names`` labels the rows.  The wiki gate reuses this formatter
    for arms whose names are not autonomy levels, and a gate that
    printed "off"/"adapt" for arms at the same level would mislabel the
    very thing it measures.
    """
    c, t = result["control"], result["treatment"]
    control_name, treatment_name = arm_names
    lines = []
    if label:
        lines.append(label)
    # An ungraded run is not a tie.  When nothing was graded -- a grader
    # that could not be reached, a rubric the model would not follow --
    # the honest verdict is "not measured", not "no effect": a 0.000 vs
    # 0.000 with every item ungraded once read as a clean null result,
    # and the reader walked away believing the wiki had been tested.
    ungraded_both = c["ungraded"] == c["n"] and t["ungraded"] == t["n"]
    if ungraded_both and c["n"] > 0:
        lines.append("  *** NOTHING WAS GRADED -- this is not a measurement. ***")
        lines.append("  Every item in both arms came back ungraded. The most common cause")
        lines.append("  is an unreachable grader endpoint (grading_errors in results.json);")
        lines.append("  the second is a grader model that never emits a GRADE: line.")
        lines.append("  Fix the grader and re-run: no conclusion about the arms can be")
        lines.append("  drawn from this run.")
    lines.append(
        "  control  (%s): %.3f  n=%d  correct=%d  wrong=%d  "
        "not-attempted=%d  ungraded=%d  errors=%d  grade-err=%d"
        % (
            control_name,
            c["accuracy"],
            c["n"],
            c["correct"],
            c["incorrect"],
            c["not_attempted"],
            c["ungraded"],
            c["errors"],
            c.get("grading_errors", 0),
        )
    )
    lines.append(
        "  treatment(%s): %.3f  n=%d  correct=%d  wrong=%d  "
        "not-attempted=%d  ungraded=%d  errors=%d  grade-err=%d"
        % (
            treatment_name,
            t["accuracy"],
            t["n"],
            t["correct"],
            t["incorrect"],
            t["not_attempted"],
            t["ungraded"],
            t["errors"],
            t.get("grading_errors", 0),
        )
    )
    verdict = "SIGNIFICANT" if result["significant"] else "not significant"
    lines.append(
        "  delta: %+.3f  (stderr %.3f)  -> %s" % (result["delta"], result["stderr"], verdict)
    )
    if not result["significant"]:
        lines.append(
            "  A delta smaller than its error bar is not a result. Ship nothing on this evidence."
        )
    return "\n".join(lines)


# ── the wiki gate (phase 6 of the memory plan) ───────────────────────────
#
# The A/B above measures the whole learn stack: its arms differ in
# ``autonomy``, which moves recall, playbooks, the wiki and the dream
# trigger at once.  That number cannot say what the *wiki* is worth,
# which is the only number that can raise the wiki's default, because
# the wiki is the one loop with a poisoning risk and a token cost all
# its own.  The gate below holds autonomy fixed and flips the wiki's
# master switch instead: two arms at the same level, one with memory
# and one without, so the delta is the wiki's own effect.

WIKI_CONTROL = "wiki-off"
WIKI_TREATMENT = "wiki-on"

# The idle threshold a replay runs under, in completed tasks.  Far above
# any real idle window: a replay's items are consecutive tasks with no
# quiet period between them, so the dream trigger never fires mid-run
# and the wiki cannot grow from the items being scored.
REPLAY_MIN_IDLE_EPISODES = 1_000_000


def _config_for_wiki_arm(base_config: dict, enabled: bool, store: str | None = None) -> dict:
    """The agent config for one wiki arm.

    One difference only: ``learn.memory.enabled``.  The autonomy level
    and everything else are copied through untouched, so anything the
    two arms do differently is the wiki and nothing else.

    The autonomy is pinned to ``adapt`` explicitly: the wiki's gates
    read the level, and a config that left it unset would follow the
    operator's own default rather than the gate's design.

    ``min_idle_episodes`` is raised high for the replay itself: a
    mid-run dream pass would extract facts from the very items being
    scored, which is training on the test set.  The trigger is deferred,
    not disabled -- the gate's own seeding between arms calls
    ``run_dream(force=True)``, which consults neither the switch nor the
    idle count, so a high threshold blocks the trigger without blocking
    the seed.  The value is far above any real idle window, and a run
    that somehow hit it would only owe a pass, which the lock and the
    watermark would still make safe.

    ``store`` pins both stores to the arm's own directory.  Without it,
    an operator config that names ``learn.memory.path`` would point both
    arms at the *same* wiki -- the treatment would answer from pages the
    control wrote, and the delta would measure nothing.  The pin wins
    over the operator's override on purpose: isolation is the gate's
    contract, and ``memory_dir`` reads the config before the environment.
    """
    config = dict(base_config)
    learn = dict(config.get("learn") or {})
    learn["autonomy"] = "adapt"
    memory = dict(learn.get("memory") or {})
    memory["enabled"] = bool(enabled)
    memory["min_idle_episodes"] = REPLAY_MIN_IDLE_EPISODES
    if store:
        learn["path"] = store
        memory["path"] = os.path.join(store, "memory")
    learn["memory"] = memory
    config["learn"] = learn
    return config


async def seed_wiki(ask, config_data: dict | None = None, **kwargs) -> dict:
    """Distil an arm's recorded runs into its wiki, once, before it runs.

    The wiki answers from what the agent has *already done*; a fresh
    store has done nothing, and a treatment arm with an empty wiki is
    the control arm under another name.  One forced dream pass over the
    store's recorded runs fills it -- forced because a replay never
    idles, and the pass must not wait for a window that will not come.

    The pass reads and writes the arm's own store (``dream_once``
    resolves both through the config), so seeding cannot touch the
    operator's wiki or the other arm's.

    Keyword arguments pass through to :func:`dream.run_dream`; the gate
    uses ``max_session_chars`` to lift the transcript cap, because a
    replay's whole day lands in one session file and the production
    budget would show the pass only the first two items.

    Returns the dream pass's own stats (``candidates``,
    ``pages_created``, ...), so the gate can tell an extraction failure
    from a policy starvation -- the two ways a seed yields zero pages,
    which want different fixes.
    """
    from . import dream

    return await dream.run_dream(
        config_data,
        ask=ask,
        force=True,
        min_facts_for_page=dream.SEED_MIN_FACTS_FOR_PAGE,
        **kwargs,
    )


def _inherit_trajectories(src_store: str, dst_store: str) -> int:
    """Copy one arm's recorded runs into another arm's store.

    This is what makes the gate self-contained and the comparison fair
    at the same time.  A fresh store has no episodes, so a treatment
    seeded only from itself would have nothing to distil; copying the
    control's trajectories in first gives the seeding pass something to
    read, and because the treatment's *recall* pool would have been that
    same corpus either way (both arms sit at ``adapt``, both recall from
    their own store), the copy adds no information the control did not
    have -- it renders the same experience in a second form.

    Best-effort: a store with no trajectories inherits nothing, and the
    treatment then runs with an empty wiki, which is the honest result
    of having nothing to remember.
    """
    import shutil

    src = os.path.join(src_store, "trajectories")
    dst = os.path.join(dst_store, "trajectories")
    if not os.path.isdir(src):
        return 0
    os.makedirs(dst, exist_ok=True)
    from .trajectory import _is_session_name

    copied = 0
    for name in sorted(os.listdir(src)):
        if not name.endswith(".jsonl") or not _is_session_name(name[: -len(".jsonl")]):
            continue
        target = os.path.join(dst, name)
        if os.path.exists(target):
            continue
        try:
            shutil.copy2(os.path.join(src, name), target)
            copied += 1
        except OSError as e:
            logger.debug("trajectory inherit failed for %s: %s", name, e)
    return copied


async def run_wiki_arm(
    items: list,
    *,
    enabled: bool,
    root: str,
    base_config: dict | None = None,
    make_agent=None,
    ask=None,
    limit: int | None = None,
    data_path: str | None = None,
    seed_ask=None,
    seed_from: str | None = None,
) -> list:
    """Run one wiki arm: seed its store when it is on, then replay.

    The arm name encodes the switch, and ``_config_for_wiki_arm`` sets
    the switch to match it -- a name that lied about its own config
    would put two numbers under one label.  The store is pinned to the
    arm's own directory *before* seeding, so the dream pass reads and
    writes this arm's store and nothing else.

    Returns ``(results, seed_stats)``: the per-item rows, and the
    seeding pass's own stats for the treatment arm (``{}`` for the
    control, which is never seeded).  The gate carries the seed stats
    into the comparison so the verdict can tell an extraction failure
    from a policy starvation -- the two zero-page seeds want different
    fixes.
    """
    arm = WIKI_TREATMENT if enabled else WIKI_CONTROL
    store = _arm_store(root, arm)
    config = _config_for_wiki_arm(base_config or {}, enabled, store=store)
    seed_stats: dict = {}
    if enabled and seed_ask is not None:
        if seed_from:
            inherited = _inherit_trajectories(seed_from, store)
            if inherited:
                logger.info("wiki arm inherited %d session file(s) from %s", inherited, seed_from)
                print(
                    "  seeding: inherited %d episode file(s) from the control arm" % inherited,
                    flush=True,
                )
        # A dream pass is minutes of model calls, and this used to be
        # silent -- the run looked stuck exactly here.
        print(
            "  seeding: distilling the control's episodes into the wiki (one dream pass)...",
            flush=True,
        )
        seed_stats = await seed_wiki(seed_ask, config, max_session_chars=SEED_MAX_SESSION_CHARS)
        print("  seeding: done", flush=True)
        if seed_stats.get("ran") is False:
            # The pass never ran at all -- autonomy below adapt, or the
            # lock was held.  The wiki is empty for a third reason, and
            # the verdict needs to know this one too.
            message = "seeding pass did not run: %s" % seed_stats.get("reason", "?")
            logger.warning(message)
            print("  *** %s ***" % message, flush=True)
        from .memory import list_pages

        pages = list_pages(config)
        if not pages:
            # The 2026-11-17 gate run shipped exactly this: a treatment
            # whose wiki was empty because the seeding pass extracted
            # nothing, and a delta that looked like a finding.  A
            # placebo must never be silent again.
            message = (
                "seeding produced NO wiki pages -- the treatment will run with an empty wiki"
            )
            logger.warning(message)
            print("  *** %s ***" % message, flush=True)
        else:
            print("  seeding: %d page(s) in the treatment wiki" % len(pages), flush=True)
    elif enabled:
        # A treatment seeded with nothing is the control under another
        # name.  That must never happen silently: say why the wiki is
        # empty before reporting a number that depends on it.
        message = (
            "seeding skipped: no client to distil with; the treatment will run with an empty wiki"
        )
        logger.warning(message)
        print("  " + message, flush=True)
    return await run_arm(
        items,
        arm=arm,
        root=root,
        base_config=config,
        make_agent=make_agent,
        ask=ask,
        limit=limit,
        data_path=data_path,
    ), seed_stats


async def run_wiki_ab(
    items: list,
    *,
    root: str,
    base_config: dict | None = None,
    limit: int | None = None,
    with_grader: bool = True,
    make_agent=None,
    seed_ask=None,
    data_path: str | None = None,
    ask=None,
) -> dict:
    """The wiki gate: same autonomy, same corpus, wiki off versus wiki on.

    The control runs first, as in :func:`ab.run_ab` -- and here the
    ordering is load-bearing rather than merely generous.  The
    treatment's wiki is distilled from the control's recorded runs, so
    the treatment is measured with memory of the very experience the
    control just had: both arms recall the same episodes, and the
    treatment additionally holds them in distilled form.  A win is the
    wiki beating raw recall on identical experience, which is exactly
    the claim the wiki's default has to survive.

    The distillation is the same dream pass production runs, with the
    same verification: a fact the control never established does not
    enter the wiki, and a wrong answer the control gave can become a
    wrong fact -- which the treatment's fact-check is there to catch,
    and which is part of what is being measured.

    ``seed_ask`` is the extraction client the seeding pass thinks with,
    and ``ask`` the grader's.  Both default to ``make_ask`` when a
    grader is available, which keeps a real run to one model; a test
    passes a stub for either or both.  ``make_agent`` defaults the same
    way ``run_ab``'s does: absent, the real factory builds an ``OnIt``
    per arm in a worker thread.  A caller who passes none -- the CLI
    does -- still gets an arm that runs the agent, rather than one that
    raises before the first item.
    """
    from .llm import make_ask

    def factory(config: dict):
        if make_agent is not None:
            return make_agent(config)
        return _wiki_agent_factory(config)

    base_config = dict(base_config or {})
    if ask is None and with_grader:
        try:
            ask = await make_ask(base_config)
        except Exception as e:
            logger.warning("no grader available (%s); answers will be recorded but not graded", e)
            # Printed as well as logged -- see run_ab.
            print(
                "  no grader available (%s); answers will be recorded but not graded" % e,
                flush=True,
            )
    if seed_ask is None:
        seed_ask = ask

    per_arm = {}
    seed_stats = {}
    for enabled in (False, True):
        results, seed_stats = await run_wiki_arm(
            items,
            enabled=enabled,
            root=root,
            base_config=base_config,
            make_agent=factory,
            ask=ask,
            limit=limit,
            seed_ask=seed_ask,
            data_path=data_path,
            seed_from=_arm_store(root, WIKI_CONTROL) if enabled else None,
        )
        per_arm[WIKI_TREATMENT if enabled else WIKI_CONTROL] = results

    control = per_arm.get(WIKI_CONTROL, [])
    treatment = per_arm.get(WIKI_TREATMENT, [])
    result = compare(control, treatment)
    result["results"] = per_arm
    result["summaries"] = {arm: summarise(r) for arm, r in per_arm.items()}
    result["wiki_pages"] = _wiki_page_count(root)
    result["wiki_seed"] = seed_stats
    result["paired"] = _paired_stats(control, treatment)
    return result


def _wiki_page_count(root: str) -> int:
    """Pages in the treatment arm's wiki, or 0 when there is none.

    A gate run whose treatment wiki is empty measured a placebo -- the
    2026-11-17 run's delta was exactly that.  The verdict needs the
    count to be able to say so instead of dressing the placebo up as a
    finding.
    """
    from .memory import list_pages

    config = _config_for_wiki_arm({}, True, store=_arm_store(root, WIKI_TREATMENT))
    return len(list_pages(config))


def _paired_stats(control: list, treatment: list) -> dict:
    """The comparison that survives ungraded items: paired, by item id.

    The headline delta divides by every item, so an item the grader
    could not grade in one arm counts as a miss there and not in the
    other.  The 2026-11-17 run's -0.15 was manufactured exactly that
    way: the control's two correct answers had no graded treatment
    counterpart.  Over the pairs where *both* arms were graded, each
    item is compared with itself, and the only asymmetry left is the
    treatment's wiki.
    """
    t_by_id = {r.get("id"): r for r in treatment}
    pairs = []
    for c in control:
        t = t_by_id.get(c.get("id"))
        if t is None:
            continue
        cg, tg = c.get("grade") or "", t.get("grade") or ""
        if not cg or not tg:
            continue
        pairs.append((cg, tg))
    return {
        "pairs": len(pairs),
        "control_correct": sum(1 for cg, _ in pairs if cg == GRADE_CORRECT),
        "treatment_correct": sum(1 for _, tg in pairs if tg == GRADE_CORRECT),
        "discordant": sum(1 for cg, tg in pairs if cg != tg),
    }


def format_wiki_verdict(result: dict) -> str:
    """What the gate says about the wiki's default, as text.

    The number is the A/B delta; the verdict is the decision the plan
    ties it to.  A significant non-negative delta raises the default;
    a significant negative one is the poisoning signal, and the plan
    says what to do with it: keep the wiki off and find the leak.
    """
    lines = [
        format_comparison(
            result,
            label="Wiki gate (same autonomy, wiki off vs on):",
            arm_names=(WIKI_CONTROL, WIKI_TREATMENT),
        )
    ]
    c, t = result.get("control", {}), result.get("treatment", {})
    if c.get("ungraded") == c.get("n") and t.get("ungraded") == t.get("n") and c.get("n"):
        # format_comparison has already said nothing was graded.  The
        # verdict must agree: "no measurable effect" is a finding about
        # the wiki, and this run contains no finding.
        lines.append(
            "  verdict: NOT MEASURED -- the grader produced no grades."
            " Fix the grader and re-run; this run says nothing about the wiki."
        )
        return "\n".join(lines)
    pages = result.get("wiki_pages")
    if pages == 0:
        # The treatment ran with an empty wiki, which is the control
        # under another name; the delta then measures noise and the
        # overhead of carrying a useless tool.  Said, not implied: the
        # 2026-11-17 run printed a tidy -0.150 over exactly this.
        lines.append(
            "  *** PLACEBO: the treatment wiki has 0 pages -- the delta is not about the wiki. ***"
        )
        seed = result.get("wiki_seed") or {}
        if seed.get("ran") is False:
            # Third way to an empty wiki: the pass never ran at all.
            lines.append(
                "  seed diagnosis: the seeding pass did not run (%s)."
                % seed.get("reason", "unknown")
            )
        elif seed.get("candidates", 0) == 0 and seed.get("empty_replies", 0) > 0:
            # The extractor never got a usable reply: the endpoint
            # returned hollow responses through every retry -- the
            # 2026-11-17 14:55 seed lost its only call this way and
            # cost a three-hour placebo run.  Fix the extractor's
            # client, or re-run; nothing about the wiki was measured.
            lines.append(
                "  seed diagnosis: extraction got %d empty reply(ies) and 0 candidates --"
                " the endpoint failed the extractor, not the page policy."
                % seed.get("empty_replies", 0)
            )
        elif seed.get("candidates", 0) == 0:
            # Real replies came back and held nothing: the day the
            # control recorded established nothing worth keeping.  Not
            # a failure -- but the treatment still runs without a wiki,
            # so the run is still a placebo.
            lines.append(
                "  seed diagnosis: extraction read the day and kept nothing --"
                " the control's day established no reusable facts,"
                " so the treatment still ran with an empty wiki."
            )
        elif seed.get("pages_created", 0) == 0:
            # Facts existed and every cluster fell below the threshold.
            # Production keeps the 3-fact policy; the gate's seed pass
            # now runs at 1, so this line should not survive the fix --
            # if it does, the threshold override did not reach the pass.
            lines.append(
                "  seed diagnosis: %d candidate(s) extracted but 0 pages created --"
                " the page policy starved them."
                % seed.get("candidates", 0)
            )
    paired = result.get("paired") or {}
    if paired.get("pairs"):
        lines.append(
            "  paired (both graded): %d item(s), control %d correct vs treatment %d,"
            " %d discordant"
            % (
                paired["pairs"],
                paired["control_correct"],
                paired["treatment_correct"],
                paired["discordant"],
            )
        )
    graded_total = c.get("graded", 0) + t.get("graded", 0)
    items_total = (c.get("n", 0) + t.get("n", 0)) or 1
    if graded_total < items_total * 0.8:
        lines.append(
            "  note: %d of %d items ungraded -- the delta leans on an incomplete ruler."
            % (items_total - graded_total, items_total)
        )
    graded_gap = abs(c.get("graded", 0) - t.get("graded", 0))
    n = c.get("n", 0) or t.get("n", 0) or 1
    if graded_gap and abs(result["delta"]) <= graded_gap / n and not result["significant"]:
        # The headline delta divides by every item, so an item graded
        # in one arm and not the other is a miss in one and nothing in
        # the other.  One such item swings the delta by 1/n -- and the
        # 2026-11-17 run's +0.05 was exactly 1/20, manufactured by the
        # control's single grading error on an item the treatment
        # graded correct.  When the asymmetry could account for the
        # whole delta, the paired line is the honest comparison.
        lines.append(
            "  note: %d more item(s) graded in one arm than the other --"
            " enough to account for the whole delta;"
            " read the paired line, not the headline."
            % graded_gap
        )
    if result["significant"] and result["delta"] > 0:
        lines.append("  verdict: the wiki earns its default. Raise learn.memory.enabled.")
    elif result["significant"]:
        lines.append(
            "  verdict: the wiki hurts. Keep it off and look for poisoning --"
            " start with the pages the treatment arm served."
        )
    else:
        lines.append(
            "  verdict: no measurable effect. The default stays off until"
            " a run with more items says otherwise."
        )
    return "\n".join(lines)
