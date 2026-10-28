#!/usr/bin/env python3
"""Measure one arm over a fixed task file.  The ruler, for a holdout you
authored yourself.

``onit learn --eval-run`` replays a split that was *drawn from the corpus*,
which is the right instrument for a corpus that already exists.  It is the
wrong instrument for the first measurement of a new agent: there is no
corpus yet, so there is nothing to draw a holdout from, and a holdout drawn
from the very runs the agent is about to make measures nothing.

This script closes that gap.  It takes a task file you wrote -- a holdout
that was never in the store -- and replays it through the real agent under
one arm, and scores the answers by string match against a reference you
wrote.  No judge model: the grader is ``str``, so the measurement has one
model in it (the agent) and not two.

Usage:

    python3 scripts/measure.py --config cfg.yaml --tasks holdout.jsonl \\
        --autonomy off --out baseline.json

    python3 scripts/measure.py --config cfg.yaml --tasks holdout.jsonl \\
        --autonomy adapt --store ~/demo/store.frozen --out treatment.json

``--autonomy off`` is a hermetic arm: nothing is injected and nothing is
recorded, so it cannot contaminate the store it is measuring.

The task file is JSONL, one item per line::

    {"task": "What weekday is 2027-02-14?", "response": "Sunday"}

``response`` is the reference.  It may hold several alternatives separated
by ``|`` (``Sunday|Sundays``); an item is correct when the agent's answer
contains one of them.  A task with no reference is recorded as ``ungraded``
rather than counted as wrong -- a missing measurement that counted as a
failure would make the arms look different for reasons that have nothing
to do with the agent.

**The leak check, and why it is not optional.**  An authored holdout is
*not* protected by the built-in exclusion set, which is built from
``benchmarks/eval/holdout.jsonl``.  If the loop worked on the same
question the holdout asks -- in a different wording -- the treatment arm
is handed the answer before it is asked the question, and the "win" is the
answer key.  ``eval.task_key()`` normalises case, punctuation and
whitespace, but not *wording*: "What weekday is 2027-02-14?" and
"Compute the weekday of the following date: 2027-02-14" are different
keys.

So this script measures the leak itself.  After the run it asks, for
every item, whether the store contains a run of a question that shares a
*distinctive token* with it -- a date, a number, a quoted string -- and
reports the item as ``contaminated`` if it does.  A contaminated item is
excluded from the accuracy, and the report says so.

**The controls.**  A single arm cannot tell a win from a prompt-shape
effect or from the model's own run-to-run wobble, so the arms that
decide the question are here too:

``--shuffle-recall``
    Keep the store, keep the injection, but return a *random* episode
    instead of the top-scored one.  If the shuffled arm scores like the
    treatment arm, then ``recall()`` is not doing the work -- the agent
    is just being reminded that it is being watched, and any "win" is
    the reminder.

``--isolate``
    Give every item its own fresh copy of the store, so item *k* cannot
    be shown item *k-1*'s run.  Without it the arms are not independent,
    and a holdout whose questions resemble the loop's own questions will
    leak into itself.

``--seed N``
    The shuffle's seed, so the shuffled arm is reproducible.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# The repository root goes on the path, NOT ``src/``: ``src/mcp/`` would
# then shadow the installed ``mcp`` SDK and break FastMCP's import.
sys.path.insert(0, ROOT)

import yaml  # noqa: E402


# ── scoring ──────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def score(answer: str, reference: str) -> str:
    """``correct`` / ``incorrect`` / ``ungraded`` for one answer."""
    if not reference or not str(reference).strip():
        return "ungraded"
    norm_answer = normalise(answer)
    if not norm_answer:
        return "not_attempted"
    for alternative in str(reference).split("|"):
        needle = normalise(alternative)
        if needle and needle in norm_answer:
            return "correct"
    return "incorrect"


def read_tasks(path: str) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


# ── the leak check ──────────────────────────────────────────────────────────

# A token that is distinctive enough to mean "the same question, asked
# again".  Ordinary words are excluded; a date, a number with a unit, a
# quoted string and a path are not.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "am", "it", "its", "this",
    "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "their", "what", "which", "who", "whom",
    "when", "where", "why", "how", "can", "could", "would", "should",
    "will", "shall", "may", "might", "must", "do", "does", "did", "done",
    "please", "thanks", "thank", "give", "get", "make", "tell", "show",
    "answer", "compute", "calculate", "following", "date", "name", "only",
    "day", "week", "weekday", "next", "also", "just", "very", "more",
    "most", "less", "least", "some", "any", "all", "there", "here",
    "about", "over", "under", "again", "not", "but", "if", "then",
    "into", "from", "out", "up", "down", "off", "one", "two", "use",
    "using", "used", "work", "out", "following", "does", "fall", "falls",
})


def distinctive_tokens(text: str) -> set:
    """Tokens specific enough that sharing one means the same question."""
    tokens = set()
    # dates, in the shapes people write them
    for match in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text or ""):
        tokens.add(match)
    for match in re.findall(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text or ""):
        tokens.add(match)
    # quoted strings
    for match in re.findall(r"[\"'“]([^\"'”]{6,})[\"'”]", text or ""):
        tokens.add(match.lower().strip())
    # paths and URLs
    for match in re.findall(r"https?://\S+|\b[\w.-]+/[\w./-]+", text or ""):
        tokens.add(match.rstrip(".,;:"))
    # bare numbers with a unit
    for match in re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|kb|mb|gb|tb|ms|hz|px)\b",
                             text or ""):
        tokens.add(re.sub(r"\s+", "", match.lower()))
    # long-ish content words
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{4,}", text or ""):
        low = word.lower()
        if low not in _STOPWORDS:
            tokens.add(low)
    return tokens


def find_contamination(items: list, store: str) -> list:
    """Items whose question is also in the store, by distinctive token.

    Returns one dict per contaminated item.  A contaminated item is one
    the agent could have been shown the answer to.

    ``store`` is a *snapshot* taken before the arm ran.  The arm's own
    recordings are not a leak -- it answered the holdout question and wrote
    down its own answer, which tells nobody anything.  What matters is
    whether a run of that question existed *before* the arm was asked it.
    """
    if not store or not os.path.isdir(store):
        return []
    try:
        from src.learn.trajectory import iter_records
        records = list(iter_records({"learn": {"path": store}}))
    except Exception:
        return []

    # The store's own questions, by token.
    stored = []
    for record in records:
        if record.get("kind") not in (None, "task"):
            continue
        task = record.get("task") or ""
        if not task:
            continue
        stored.append((task, distinctive_tokens(task)))

    contaminated = []
    for index, item in enumerate(items):
        task = item.get("task") or ""
        tokens = distinctive_tokens(task)
        if not tokens:
            continue
        for other, other_tokens in stored:
            shared = tokens & other_tokens
            if not shared:
                continue
            # A shared generic word is not a leak; a shared date or path is.
            distinctive = {t for t in shared
                           if re.search(r"\d", t) or "/" in t or len(t) > 8}
            if not distinctive:
                continue
            contaminated.append({
                "index": index,
                "task": task,
                "contaminated_by": other,
                "shared": sorted(distinctive),
            })
            break
    return contaminated


def snapshot_store(store: str, destination: str) -> str:
    """Copy the store, so the leak check can ask what existed *before*.

    The store is read before the arm runs, and the copy is what the
    contamination check compares against.  Without this, the arm's own
    recordings of the holdout questions look like leaks.
    """
    if not store or not os.path.isdir(store):
        return destination
    if os.path.isdir(destination):
        shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(store, destination, dirs_exist_ok=True)
    return destination


def patch_shuffled_recall(rng: random.Random) -> None:
    """Replace ``recall``'s ranking with a shuffle.

    The patch is at the *scorer*, not the store, so the control keeps
    everything the treatment arm has: the same episodes, the same
    ``episodes_used`` bookkeeping, the same block in the prompt.  The
    only thing destroyed is the ranking -- which is the only thing the
    control is allowed to destroy, because the control's question is
    "does the *best* episode matter, or does any episode matter?".

    ``recall`` returns a list of episodes sorted by score, and
    ``build_context`` renders the top ``top_k`` of them.  Shuffling the
    *returned* list keeps the recall machinery intact and only changes
    which episodes win the cut.
    """
    from src.learn import recall as recall_mod

    original = recall_mod.recall

    def shuffled(*a, **kw):
        episodes = list(original(*a, **kw))
        rng.shuffle(episodes)
        return episodes

    recall_mod.recall = shuffled

    # ``build_context`` lives in the same module and calls ``recall``
    # by module-global name, so patching the module attribute is enough.
    # But ``src.onit`` imports ``build_context`` from this module, and
    # ``build_context`` calls the module-global ``recall`` -- so the
    # patch above is picked up without touching the import site.


# ── the run ──────────────────────────────────────────────────────────────────

async def run(args, cfg: dict) -> dict:
    from src.onit import OnIt

    items = read_tasks(args.tasks)
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise SystemExit("No items in %s" % args.tasks)

    # The arm: autonomy decides whether anything is injected, and
    # ``learn.path`` decides whose corpus the injection reads.  Setting only
    # the first would have the arm recall the operator's own corpus, which
    # is not the arm's experience.
    cfg = dict(cfg)
    learn = dict(cfg.get("learn") or {})
    learn["autonomy"] = args.autonomy
    if args.store:
        learn["path"] = os.path.abspath(os.path.expanduser(args.store))
    cfg["learn"] = learn

    store = learn.get("path") or os.path.join(ROOT, "tmp", "measure-store")
    store = os.path.abspath(os.path.expanduser(str(store)))
    sessions = os.path.join(store, "sessions")
    os.makedirs(sessions, exist_ok=True)

    # The leak check needs to know what the store held BEFORE the arm ran.
    # The arm's own recordings of the holdout questions are not a leak --
    # it answered them and wrote down its own answer -- so the check
    # compares against this snapshot, not against the store as it ends up.
    snapshot = os.path.join(ROOT, "tmp", "measure-snapshot-%s" % args.autonomy)
    snapshot_store(store, snapshot)

    # The session's working directory.  It has to live inside the bash
    # server's DATA_PATH, which the CLI sets from the config's data_path;
    # a data_path outside that jail makes every tool call refuse, and the
    # agent then measures refusals instead of answers.
    data_path = args.data_path or cfg.get("data_path")
    if not data_path:
        data_path = os.path.join(ROOT, "tmp", "onit", "data")
    data_path = os.path.abspath(os.path.expanduser(str(data_path)))
    os.makedirs(data_path, exist_ok=True)

    # The agent is built in a worker thread: OnIt's constructor discovers
    # its tools with asyncio.run, which cannot run inside a running loop.
    def factory():
        from src.cli import _ensure_mcp_servers
        _ensure_mcp_servers(cfg)
        return OnIt(cfg)

    agent = await asyncio.get_running_loop().run_in_executor(None, factory)

    # The shuffled-recall control patches the scorer, not the store: the
    # store still holds the real runs, the agent is still shown *something*,
    # and only the *ranking* is destroyed.  That is exactly the question
    # the control asks -- is it the top-scored episode doing the work, or
    # is it the mere presence of an episode in the prompt?
    if args.shuffle_recall:
        patch_shuffled_recall(random.Random(args.seed))

    results = []
    for index, item in enumerate(items):
        task = item.get("task") or ""
        reference = item.get("response") or ""
        started = time.time()
        answer = ""
        error = ""
        # The isolated arm: every item gets its own copy of the store, so
        # item k cannot be shown item k-1's run.  The config's learn.path
        # is swapped per item, and the *playbook* is pinned on the OnIt
        # instance -- so the instance is rebuilt per item too, or the
        # second item would keep the first item's pinned playbook.
        item_store = store
        if args.isolate:
            item_store = os.path.join(store + ".iso", "item-%04d" % index)
            shutil.rmtree(item_store, ignore_errors=True)
            if os.path.isdir(store):
                shutil.copytree(store, item_store)
            else:
                os.makedirs(item_store, exist_ok=True)
            learn["path"] = item_store
            agent.config_data["learn"] = dict(learn)
            agent._playbook_pinned = False
            agent._playbook_context = ""
        try:
            # The safety queue is not optional.  process_task drains it
            # before the run, and a caller that leaves it unset is the
            # difference between a run that answers and a run that raises.
            queue = asyncio.Queue(maxsize=10)
            answer = await agent.process_task(
                task,
                session_path=os.path.join(sessions, "m%04d.jsonl" % index),
                data_path=data_path,
                safety_queue=queue,
            )
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        results.append({
            "id": item.get("id") or "m%04d" % index,
            "task": task,
            "answer": answer,
            "reference": reference,
            "grade": score(answer, reference),
            "error": error,
            "wall_s": round(time.time() - started, 3),
        })
        print("  [%3d/%3d] %-11s %s"
              % (index + 1, len(items), results[-1]["grade"], task[:60]))

    graded = [r for r in results if r["grade"] in ("correct", "incorrect")]

    # The leak check, against the pre-run snapshot of the store.
    contamination = find_contamination(items, snapshot)
    contaminated_indexes = {c["index"] for c in contamination}
    for index, result in enumerate(results):
        result["contaminated"] = index in contaminated_indexes
    clean = [r for r in graded if not r["contaminated"]]
    correct = sum(1 for r in clean if r["grade"] == "correct")

    summary = {
        "arm": args.autonomy,
        "store": store,
        "tasks": args.tasks,
        "controls": {
            "shuffle_recall": bool(args.shuffle_recall),
            "isolate": bool(args.isolate),
            "seed": args.seed,
        },
        "n": len(results),
        "graded": len(graded),
        "contaminated": len(contaminated_indexes),
        "clean_graded": len(clean),
        "correct": correct,
        "incorrect": len(clean) - correct,
        "not_attempted": sum(1 for r in results if r["grade"] == "not_attempted"),
        "ungraded": sum(1 for r in results if r["grade"] == "ungraded"),
        "errors": sum(1 for r in results if r["error"]),
        "accuracy": (correct / len(clean)) if clean else 0.0,
        "accuracy_including_contaminated": (
            (sum(1 for r in graded if r["grade"] == "correct") / len(graded))
            if graded else 0.0),
        "contamination": contamination,
        "results": results,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay a task file through the real agent and score it.")
    parser.add_argument("--config", required=True,
                        help="The agent's YAML config.")
    parser.add_argument("--tasks", required=True,
                        help="JSONL of {\"task\": ..., \"response\": ...} items.")
    parser.add_argument("--autonomy", default="off",
                        choices=["off", "observe", "adapt"],
                        help="The arm. 'off' injects nothing and records "
                             "nothing; 'adapt' injects recall + playbook.")
    parser.add_argument("--store", default=None,
                        help="The learned store this arm reads and writes.")
    parser.add_argument("--data-path", default=None,
                        help="The session working directory. Must be inside "
                             "the bash server's DATA_PATH.")
    parser.add_argument("--out", default=None,
                        help="Where to write the JSON report.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Replay only the first N items.")
    parser.add_argument("--shuffle-recall", action="store_true",
                        help="The control: return a RANDOM episode instead "
                             "of the top-scored one. If this arm scores like "
                             "the treatment arm, recall() is not doing the "
                             "work.")
    parser.add_argument("--isolate", action="store_true",
                        help="Give every item its own fresh copy of the "
                             "store, so item k cannot be shown item k-1's "
                             "run and the arms are independent.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for --shuffle-recall (default 0).")
    args = parser.parse_args()

    config_path = os.path.abspath(os.path.expanduser(args.config))
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    print("Replaying %s  arm=%s  store=%s"
          % (args.tasks, args.autonomy, args.store or "(config default)"))

    summary = asyncio.run(run(args, cfg))

    print()
    print("  n=%d  graded=%d  correct=%d  wrong=%d  not-attempted=%d  "
          "ungraded=%d  errors=%d  accuracy=%.3f"
          % (summary["n"], summary["graded"], summary["correct"],
             summary["incorrect"], summary["not_attempted"],
             summary["ungraded"], summary["errors"], summary["accuracy"]))

    if summary["contaminated"]:
        print()
        print("  LEAK CHECK: %d of %d item(s) ask a question the store "
              "already contains a run of." % (summary["contaminated"],
                                               summary["n"]))
        for entry in summary["contamination"]:
            print("    %s" % entry["task"][:64])
            print("      the store already has: %s"
                  % str(entry["contaminated_by"])[:64])
            print("      shared: %s" % ", ".join(entry["shared"][:5]))
        if args.autonomy == "off":
            print("  This arm injects nothing, so it was NOT shown any of "
                  "it.  The hit means the HOLDOUT overlaps the store -- "
                  "which is still a leak, because the treatment arm reads "
                  "the same store.")
        else:
            print("  Those items are EXCLUDED from the accuracy above.  The "
                  "treatment arm was shown the answer before it was asked "
                  "the question, so a difference on them is the answer "
                  "key, not learning.")
        print("  Accuracy including them: %.3f"
              % summary["accuracy_including_contaminated"])
        print("  Fix: use holdout dates the loop never worked on, and give "
              "the loop and the holdout different wording for the same "
              "underlying task.")

    out = args.out or os.path.join(ROOT, "tmp", "measure-%s.json" % args.autonomy)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("  Results: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
