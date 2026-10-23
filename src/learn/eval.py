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

The ruler: a frozen holdout, a dev split, and the boundary that keeps them honest.

This is the missing instrument.  Everything else in ``docs/SELF_IMPROVEMENT.md``
is a way of changing the agent; without this, none of it can be *measured*, and
a change that cannot be measured is a change that ships on the strength of a
coincidence.

Two rules make the measurement mean something, and they are separate failures:

**I1 — no loop reads the holdout.**  Iteration happens on the dev split.  The
holdout is read exactly once per candidate, at promotion time, by the gate.

**I7 — the ruler is outside the agent's writable surface.**  The holdout, the
dev split, the pinned baseline and the judge configuration live in a tree the
agent cannot write to.  Not "the agent is not supposed to" — cannot.  A
promotion gate that reads an exam the agent may edit is not a gate.

The boundary is enforced by *layout*, not by policy: the eval tree is resolved
relative to the repository (or an explicit override), and the check
:func:`assert_outside_agent_surface` refuses any path that lies under the
agent's writable roots.  A deployment that wants the holdout elsewhere can
point ``eval.path`` at a read-only mount; the default is inside the repo,
where the agent's sandbox has no write access.

The split itself is drawn from real recorded runs, because that is the only
source of tasks that resemble what the agent is actually asked.  Two rules
keep the split honest:

- **Per model, not globally.**  The corpus is multi-model (twelve models as of
  October 2026).  The same task on a different model is a different
  measurement, so a holdout is pinned per model and a candidate is only ever
  compared against a baseline taken on the same model.
- **Per owner, or owner-less.**  A task recorded by one tenant must never be
  replayed into another tenant's evaluation.  Records with no owner are
  shared: they come from single-user CLI runs, where there is no tenant to
  leak across.

Nothing in this module writes to the agent's store, and nothing in the agent's
store is read to *build* the split without an explicit operator action
(``onit learn --holdout``).  Drawing the split is an offline, deliberate act.
"""

import json
import logging
import os
import re
import tempfile

logger = logging.getLogger(__name__)

# Split file names inside the eval tree.
HOLDOUT_NAME = "holdout.jsonl"
DEV_NAME = "dev.jsonl"
BASELINE_NAME = "baseline.json"
SPLIT_NAME = "split.json"

# Schema version for the split/baseline files.  Bumped when the shape of a
# record changes in a way a reader would trip over.
SCHEMA_VERSION = 1

# The kinds of split a task can land in.  Kept small and explicit: a task that
# is neither held out nor in dev is not in the evaluation at all, and the
# count of those is reported rather than hidden.
SPLIT_HOLDOUT = "holdout"
SPLIT_DEV = "dev"

# A task has to be worth replaying to be worth holding out.  A run that
# errored out, or one so short that its answer is a guess, measures the
# harness rather than the agent.
MIN_TASK_CHARS = 24
MIN_RESPONSE_CHARS = 8

# A holdout nobody can read is not a holdout, and a holdout of forty items is
# not a holdout either -- the stderr on a 40-item set is so wide that almost
# nothing clears it.  These are the sizes the plan assumed.
MIN_HOLDOUT_ITEMS = 20
MIN_DEV_ITEMS = 20


# ─── paths ──────────────────────────────────────────────────────────────────

def _repo_root() -> str:
    """The repository root, derived from this file's location.

    ``src/learn/eval.py`` -> ``<root>``.  Deliberately not an environment
    variable: the whole point of I7 is that the ruler's location is fixed by
    the checkout, not by something the agent can set.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def eval_dir(config_data: dict | None = None) -> str:
    """Directory holding the holdout, the dev split and the pinned baseline.

    Defaults to ``<repo>/benchmarks/eval``.  ``learn.eval.path`` overrides it,
    and ``ONIT_EVAL_PATH`` overrides that -- the override exists so a
    deployment can point the ruler at a read-only mount, which is the one
    supported way to satisfy I7 outside a checkout.
    """
    learn = (config_data or {}).get("learn")
    if isinstance(learn, dict):
        nested = learn.get("eval")
        if isinstance(nested, dict) and nested.get("path"):
            return os.path.abspath(os.path.expanduser(str(nested["path"])))
    env = os.environ.get("ONIT_EVAL_PATH")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(_repo_root(), "benchmarks", "eval")


def holdout_path(config_data: dict | None = None) -> str:
    return os.path.join(eval_dir(config_data), HOLDOUT_NAME)


def dev_path(config_data: dict | None = None) -> str:
    return os.path.join(eval_dir(config_data), DEV_NAME)


def baseline_path(config_data: dict | None = None) -> str:
    return os.path.join(eval_dir(config_data), BASELINE_NAME)


def split_path(config_data: dict | None = None) -> str:
    return os.path.join(eval_dir(config_data), SPLIT_NAME)


# ─── the I7 boundary ────────────────────────────────────────────────────────

def agent_write_roots(config_data: dict | None = None) -> list:
    """Roots the agent can write to.

    Two of them, and both are load-bearing:

    - ``learn.path`` -- where trajectories, the playbook and skills live.  The
      agent's own learned artifacts.
    - ``data_path`` -- the session's working directory, and the root of the
      harness's per-session stores.  Whatever the sandbox lets the agent touch
      lives under here.
    """
    roots = []
    learn = (config_data or {}).get("learn")
    if isinstance(learn, dict) and learn.get("path"):
        roots.append(os.path.abspath(os.path.expanduser(str(learn["path"]))))
    else:
        roots.append(os.path.abspath(os.path.expanduser(
            os.environ.get("ONIT_LEARN_PATH", "~/.onit/learned"))))

    data_path = (config_data or {}).get("data_path")
    if data_path:
        roots.append(os.path.abspath(os.path.expanduser(str(data_path))))
    return roots


def _is_within(path: str, root: str) -> bool:
    """True when ``path`` is ``root`` or lies under it."""
    try:
        path = os.path.abspath(path)
        root = os.path.abspath(root)
    except (TypeError, ValueError):
        return False
    if path == root:
        return True
    return path.startswith(root.rstrip(os.sep) + os.sep)


def assert_outside_agent_surface(path: str,
                                  config_data: dict | None = None) -> None:
    """Raise when ``path`` lies inside a root the agent can write to.

    This is the I7 check, and it is the reason the ruler cannot be edited by
    the thing being measured.  It is called by every writer in this module
    and by the promotion gate.

    The check is on the *resolved* path, so a symlink or a ``..`` segment
    cannot slip a file under an agent-writable root.
    """
    resolved = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    for root in agent_write_roots(config_data):
        if _is_within(resolved, os.path.realpath(root)):
            raise PermissionError(
                "I7: %s lies under the agent's writable surface (%s). The "
                "holdout, the dev split and the baseline must live outside "
                "it, or the agent can edit its own exam."
                % (path, root))


def is_outside_agent_surface(path: str,
                             config_data: dict | None = None) -> bool:
    """Boolean form of :func:`assert_outside_agent_surface`."""
    try:
        assert_outside_agent_surface(path, config_data)
    except PermissionError:
        return False
    return True


# ─── task records in the split ──────────────────────────────────────────────

def _record_id(record: dict) -> str:
    """A stable identity for a recorded task.

    ``session_id`` plus ``turn`` is unique within a store and does not change
    when the record's contents are re-derived, which matters because a
    backfill rewrites signals in place.
    """
    return "%s#%s" % (record.get("session_id", "?"), record.get("turn", "?"))


def _is_usable(record: dict) -> bool:
    """Whether a recorded task is worth replaying as an evaluation item.

    Filters on substance, not on metadata: a task with no text, or a run that
    ended without an answer, measures the harness rather than the agent.
    """
    if record.get("kind") not in (None, "task"):
        return False
    task = (record.get("task") or "").strip()
    response = (record.get("response") or "").strip()
    if len(task) < MIN_TASK_CHARS or len(response) < MIN_RESPONSE_CHARS:
        return False
    return True


def _split_entry(record: dict, split: str, model: str | None,
                 owner: str | None) -> dict:
    """One line of the split file."""
    return {
        "id": _record_id(record),
        "split": split,
        "task": record.get("task"),
        "response": record.get("response"),
        "model": model,
        "owner": owner,
        "session_id": record.get("session_id"),
        "turn": record.get("turn"),
        "ts": record.get("ts"),
        # Kept so the evaluation can be reproduced: a replay needs the same
        # tool set and the same model, and neither is in the task text.
        "tools_available": record.get("tools_available") or [],
        "verifier": (record.get("signals") or {}).get("verifier"),
    }


def _load_records(config_data: dict | None = None):
    """Every recorded task, in one place, with the owner filter applied.

    The owner filter is the cross-tenant boundary, and it lives here rather
    than in ``iter_records`` because it is an *evaluation* concern: the
    trajectory store is a record of what happened and stays complete, but
    what may be *replayed* is a narrower question.
    """
    from .trajectory import iter_records
    return iter_records(config_data)


def build_split(*, config_data: dict | None = None,
                model: str | None = None,
                owner: str | None = None,
                holdout_ratio: float = 0.4,
                max_items: int = 400,
                seed: int = 20261022) -> dict:
    """Draw a holdout and a dev split from the recorded corpus.

    Pure: writes nothing, raises nothing.  Returns the split as data so the
    caller can inspect it before anything is frozen.

    ``model`` restricts the draw to one model.  A split drawn across models
    mixes two different agents' work into one measurement, and a candidate
    is only ever compared against a baseline taken on the same model -- so
    the default is to refuse a mixed split rather than silently build one.

    ``owner`` restricts the draw to one tenant.  ``None`` means *owner-less
    only* -- the tasks recorded by single-user CLI runs.  That is the safe
    default: a split containing another tenant's tasks would replay them
    into a measurement, and an evaluation is a thing that gets shared.
    """
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be between 0 and 1")

    candidates = []
    for record in _load_records(config_data):
        if not _is_usable(record):
            continue
        record_model = record.get("model")
        if model and record_model != model:
            continue
        record_owner = record.get("owner")
        if owner is not None:
            if record_owner != owner:
                continue
        else:
            # Owner-less only: a task with an owner belongs to a tenant and
            # is not eligible for a shared split.
            if record_owner is not None:
                continue
        candidates.append(record)

    # Deterministic order, so two runs of this function on the same corpus
    # produce the same split.  A split that changed on every run would make
    # every comparison against it meaningless.
    candidates.sort(key=lambda r: (_record_id(r), r.get("ts") or ""))

    # Deterministic shuffle: sort by a hash of the id, seeded.  Stable across
    # runs and independent of insertion order.
    import random
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    if max_items and len(shuffled) > max_items:
        shuffled = shuffled[:max_items]
    candidates = shuffled

    n_holdout = int(round(len(shuffled) * holdout_ratio))
    holdout_items = shuffled[:n_holdout]
    dev_items = shuffled[n_holdout:]

    holdout = [_split_entry(r, SPLIT_HOLDOUT, r.get("model"), r.get("owner"))
               for r in holdout_items]
    dev = [_split_entry(r, SPLIT_DEV, r.get("model"), r.get("owner"))
           for r in dev_items]

    return {
        "schema": SCHEMA_VERSION,
        "created": _now(),
        "model": model,
        "owner": owner,
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "counts": {
            "holdout": len(holdout),
            "dev": len(dev),
            "candidates": len(candidates),
        },
        "holdout": holdout,
        "dev": dev,
    }


# ─── reading and writing the split ──────────────────────────────────────────

def _now() -> str:
    from .trajectory import _now as _t
    return _t()


def read_split(config_data: dict | None = None) -> dict:
    """The frozen split, or an empty one when there is none.

    Never raises: "is there a holdout" has to be answerable when the rest of
    the stack is down, the same way ``onit learn`` is.
    """
    path = split_path(config_data)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"schema": SCHEMA_VERSION, "holdout": [], "dev": [],
                "counts": {"holdout": 0, "dev": 0, "candidates": 0}}
    if not isinstance(data, dict):
        return {"schema": SCHEMA_VERSION, "holdout": [], "dev": [],
                "counts": {"holdout": 0, "dev": 0, "candidates": 0}}
    data.setdefault("holdout", [])
    data.setdefault("dev", [])
    data.setdefault("counts", {"holdout": len(data["holdout"]),
                                "dev": len(data["dev"]), "candidates": 0})
    return data


def read_holdout(config_data: dict | None = None) -> list:
    """The frozen holdout items.

    Read through the split file, which is the single source of truth.  The
    separate ``holdout.jsonl`` exists so the holdout can be *handed* to a
    process that should not see the dev split -- the promotion gate reads
    this one; a loop iterating on a candidate reads ``dev.jsonl``.
    """
    path = holdout_path(config_data)
    items = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn line loses one item, not the file
    except (FileNotFoundError, OSError):
        return []
    return items


def read_dev(config_data: dict | None = None) -> list:
    """The dev-split items -- the ones a loop is allowed to iterate against."""
    path = dev_path(config_data)
    items = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (FileNotFoundError, OSError):
        return []
    return items


def _atomic_write(path: str, lines: list) -> None:
    """Write lines to path atomically.

    A holdout that is half-written is worse than no holdout: it looks frozen
    and is not.  Write to a temp file in the same directory, then replace.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".eval-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, default=str) + "\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_json(path: str, data: dict) -> None:
    """Write one JSON object to path atomically.

    ``_atomic_write`` writes JSON *lines* -- one value per line -- which
    is the right shape for the holdout and the dev split.  The split
    *file* is a single object, and it is read back with ``json.load``,
    so it needs its own writer.  Running it through the line writer
    double-encoded the object, and every split ever drawn came back
    empty.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".eval-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_split(split: dict, config_data: dict | None = None) -> dict:
    """Freeze a split to disk.

    Refuses to write inside the agent's writable surface (I7).  This is the
    moment the check matters most: a holdout written into the agent's own
    tree is a holdout the agent can edit, and every later measurement is
    then a measurement of the agent's edits to its exam.

    Returns the paths written.  Raises ``PermissionError`` when the target
    violates I7 -- loudly, because the alternative is a quietly compromised
    ruler.
    """
    directory = eval_dir(config_data)
    assert_outside_agent_surface(directory, config_data)

    holdout = split.get("holdout") or []
    dev = split.get("dev") or []

    holdout_file = os.path.join(directory, HOLDOUT_NAME)
    dev_file = os.path.join(directory, DEV_NAME)
    split_file = os.path.join(directory, SPLIT_NAME)

    for path in (holdout_file, dev_file, split_file):
        assert_outside_agent_surface(path, config_data)

    _atomic_write(holdout_file, holdout)
    _atomic_write(dev_file, dev)

    meta = {k: v for k, v in split.items() if k not in ("holdout", "dev")}
    # The split file carries the metadata *and* the two lists, so that
    # read_split() can answer "which split is this record in?" from one
    # file.  Writing only the metadata made read_split() return empty
    # lists for every split ever drawn -- and it is why the split
    # filter had nothing to filter on.
    meta["holdout"] = holdout
    meta["dev"] = dev
    _atomic_write_json(split_file, meta)

    return {
        "holdout": holdout_file,
        "dev": dev_file,
        "split": split_file,
        "holdout_items": len(holdout),
        "dev_items": len(dev),
    }


# ── the I1 exclusion set ───────────────────────────────────────────────

def task_key(text: str) -> str:
    """The identity of a *question*, as opposed to a recorded run.

    Two runs of the same question are two records but one question.  The
    corpus contains both: "weather today" was asked 21 times, "marikina
    river water level latest" 6.  A holdout drawn from it therefore has to
    exclude the whole family, not one member of it — otherwise the agent
    is handed the answer to the question it is about to be asked, twenty
    times over, by a rule that only excluded the record it happened to
    pick.

    The normalisation is aggressive on purpose: case, punctuation and
    whitespace all go, because "Weather today?" and "weather today" are
    the same question and a holdout that separated them would leak the
    answer to the second.
    """
    if not text:
        return ""
    lowered = str(text).strip().lower()
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def holdout_exclusions(config_data: dict | None = None) -> dict:
    """Everything the loops must not learn from: the holdout's ids and tasks.

    Two sets, and both are needed:

    ``ids``    the record ids the holdout was drawn from.  Excluding them
                stops the exact recorded run from being replayed.
    ``tasks``  the normalised question texts.  Excluding them stops a
                *sibling* run of the same question from being replayed —
                which is the leak that matters, because the sibling has the
                same answer.

    A loop that only excludes ids has excluded the symptom.  A loop that
    only excludes task texts has excluded the question, which is the thing
    that was actually being measured.

    Empty when no holdout has been drawn: an absent ruler must not be
    mistaken for a ruler that excludes everything, and must not be
    mistaken for one that excludes nothing either.  The caller decides
    what to do with an empty set; this function only reports it.
    """
    items = read_holdout(config_data)
    ids = set()
    tasks = set()
    for item in items:
        item_id = item.get("id")
        if item_id:
            ids.add(str(item_id))
        key = task_key(item.get("task") or "")
        if key:
            tasks.add(key)
    return {"ids": ids, "tasks": tasks}


def excluded_record_ids(config_data: dict | None = None) -> set:
    """The record ids a learning loop must skip."""
    return holdout_exclusions(config_data)["ids"]


def excluded_task_keys(config_data: dict | None = None) -> set:
    """The normalised question texts a learning loop must skip."""
    return holdout_exclusions(config_data)["tasks"]


def is_excluded(record: dict, config_data: dict | None = None) -> bool:
    """Whether a recorded run may be learned from.

    The single predicate every loop that reads the corpus goes through.
    It exists as one function so that the I1 boundary is greppable: if a
    new loop wants to read recorded runs, there is one obvious question
    to ask, and one place where the answer lives.
    """
    exclusions = holdout_exclusions(config_data)
    if not exclusions["ids"] and not exclusions["tasks"]:
        return False
    record_id = "%s#%s" % (record.get("session_id", "?"),
                           record.get("turn", "?"))
    if record_id in exclusions["ids"]:
        return True
    key = task_key(record.get("task") or "")
    return bool(key) and key in exclusions["tasks"]


def filter_excluded(records: list, config_data: dict | None = None) -> list:
    """Drop the records a learning loop must not see."""
    exclusions = holdout_exclusions(config_data)
    if not exclusions["ids"] and not exclusions["tasks"]:
        return list(records)
    kept = []
    for record in records:
        record_id = "%s#%s" % (record.get("session_id", "?"),
                               record.get("turn", "?"))
        if record_id in exclusions["ids"]:
            continue
        key = task_key(record.get("task") or "")
        if key and key in exclusions["tasks"]:
            continue
        kept.append(record)
    return kept


def read_baseline(config_data: dict | None = None) -> dict:
    """The pinned baseline for the current model, or an empty dict."""
    path = baseline_path(config_data)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_baseline(baseline: dict, config_data: dict | None = None) -> str:
    """Pin a baseline, refusing to write it inside the agent's tree (I7)."""
    path = baseline_path(config_data)
    assert_outside_agent_surface(path, config_data)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    _atomic_write(path, [baseline])
    return path


# ─── the promotion gate ─────────────────────────────────────────────────────

def holdout_is_sealed(config_data: dict | None = None) -> bool:
    """Whether a usable holdout exists.

    "Usable" means big enough to have a stderr worth comparing against.
    A three-item holdout has a stderr of roughly 0.29 and will promote
    nothing; the plan's own Gate 3 says a loop that cannot lose on the
    holdout cannot win on it either.
    """
    return len(read_holdout(config_data)) >= MIN_HOLDOUT_ITEMS


def dev_is_sealed(config_data: dict | None = None) -> bool:
    """Whether the dev split is big enough to iterate against."""
    return len(read_dev(config_data)) >= MIN_DEV_ITEMS


def assert_readable_by_gate(config_data: dict | None = None) -> list:
    """The holdout, for the promotion gate only.

    The single place that reads the holdout.  A loop iterating on a candidate
    must call :func:`read_dev` instead.  Keeping the read in one function
    makes the I1 boundary greppable: there is exactly one reader of the
    holdout in the codebase, and it is the gate.
    """
    items = read_holdout(config_data)
    if not items:
        raise FileNotFoundError(
            "No frozen holdout at %s. Draw one with 'onit learn --holdout' "
            "before promoting anything." % holdout_path(config_data))
    return items


def summarize(config_data: dict | None = None) -> dict:
    """A summary of the ruler, for `onit learn` and for the gate."""
    split = read_split(config_data)
    holdout = read_holdout(config_data)
    dev = read_dev(config_data)
    baseline = read_baseline(config_data)
    return {
        "eval_dir": eval_dir(config_data),
        "outside_agent_surface": is_outside_agent_surface(
            eval_dir(config_data), config_data),
        "holdout": len(holdout),
        "dev": len(dev),
        "holdout_sealed": len(holdout) >= MIN_HOLDOUT_ITEMS,
        "dev_sealed": len(dev) >= MIN_DEV_ITEMS,
        "model": split.get("model"),
        "owner": split.get("owner"),
        "created": split.get("created"),
        "baseline": baseline.get("model") or baseline.get("learn") or None,
        "baseline_created": baseline.get("created"),
    }
