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

What a task actually did, kept.

The session file next to this one records ``{task, response, timestamp}`` —
enough to replay a conversation, and nothing a later loop can learn from.  The
tool calls, the failures, the retries and the turn-by-turn token counts are all
computed while a task runs and then dropped on the floor.  This module writes
them down.

Two record kinds share one file per session:

  ``task``    one per completed task: the trajectory, the metrics, the signals
  ``rating``  a later verdict on an earlier task, appended rather than merged

Appending is the whole storage strategy.  A rating arrives minutes after the
task it judges, from a different process (the web UI), while the agent may
still be writing; rewriting a line in place would need a lock, and losing a
trajectory to a half-finished rewrite costs more than a reader that folds
ratings in at read time.  Use :func:`read_session` to get the folded view.
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone

from .config import recording_enabled, redact_tool_args, trajectory_dir

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Written into every task record so a later reader can tell how a trajectory
# was produced without guessing from which keys happen to be present.
KIND_TASK = "task"
KIND_RATING = "rating"
# Lifecycle events from the loops themselves — a tool registry loading, a skill
# being proposed, a playbook bullet promoted.  Written into the same file as the
# runs they belong to (docs/SELF_IMPROVEMENT.md §4.4: one event log, two loops)
# and ignored by :func:`read_session`, which folds only ratings into tasks.
KIND_EVENT = "event"

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

# Long enough that two distinct argument sets colliding is not a practical
# concern, short enough to read in a terminal.
_DIGEST_CHARS = 16


def _safe_name(value: str) -> str:
    """A session id reduced to something safe to use as a file name."""
    return _UNSAFE_NAME.sub("_", str(value or "unknown"))[:96] or "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def owner_hash(owner: str | None) -> str | None:
    """Stable pseudonym for a session owner.

    The owner is an email address in web mode.  Scoping later loops per owner
    needs a key that groups the same person's sessions together; it does not
    need the address itself, and a trajectory store is read back by machinery
    rather than by the person who typed into it.
    """
    if not owner:
        return None
    return "sha256:" + hashlib.sha256(str(owner).encode("utf-8")).hexdigest()[:32]


def args_digest(arguments: dict | None) -> str:
    """Digest of a tool call's arguments, order-independent."""
    try:
        payload = json.dumps(arguments or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(arguments)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


def describe_tool_call(name: str, arguments: dict | None, *,
                       ok: bool, ms: int, result_chars: int,
                       redact: bool = True) -> dict:
    """One tool call, as it is stored.

    Argument *names* are kept either way: which parameters a model reaches for
    is most of what makes a trajectory diagnosable, and a parameter name is not
    the thing that carries a file path or a pasted credential.  The values are
    kept only when a deployment has asked for them.
    """
    record = {
        "name": name,
        "ok": bool(ok),
        "ms": int(ms),
        "result_chars": int(result_chars),
        "arg_keys": sorted(str(k) for k in (arguments or {})),
        "args_digest": args_digest(arguments),
    }
    if not redact:
        record["args"] = arguments or {}
    return record


def _turn_view(turn: dict) -> dict:
    """One turn of a ``TurnMetrics`` sink, trimmed to what is worth storing."""
    return {
        "n": turn.get("n"),
        "prompt_tokens": turn.get("prompt_tokens", 0),
        "completion_tokens": turn.get("completion_tokens", 0),
        "finish_reason": turn.get("finish_reason"),
        "ttft_s": turn.get("ttft_s"),
        "model_s": turn.get("model_s"),
        "tool_s": turn.get("tool_s", 0.0),
        # Rich per-call records when the loop supplied them, names otherwise:
        # a turn whose tools ran through a path that predates this field still
        # says which tools ran.
        "tools": turn.get("tool_runs") or [
            {"name": n} for n in (turn.get("tools") or [])
        ],
    }


def derive_signals(metrics: dict | None, stop_reason: str | None = None) -> dict:
    """Outcome signals that cost nothing, read off a finished run.

    None of these is a verdict on whether the answer was right.  They are the
    cheap half of the reward: a run that hit three tool errors, retried the API
    twice and truncated its answer went badly whatever the answer said.

    ``stop_reason`` comes from the run's ``RunState`` (model/serving/state.py)
    and is the one signal the metrics cannot supply: whether the loop finished
    or gave up.  It is read from that object rather than re-derived here, so
    the two records of how a run ended cannot drift apart.
    """
    metrics = metrics or {}
    turns = metrics.get("turns") or []
    tool_errors = 0
    for turn in turns:
        for call in turn.get("tool_runs") or []:
            if not call.get("ok", True):
                tool_errors += 1
    return {
        "tool_errors": tool_errors,
        "stop_reason": stop_reason or "",
        "retries": int(metrics.get("api_retries", 0) or 0),
        "truncations": sum(1 for t in turns if t.get("finish_reason") == "length"),
        "compactions": int(metrics.get("compactions", 0) or 0),
        # Filled in later, by a person or a verifier, via append_rating().
        "user_rating": None,
        "verifier": None,
    }


def build_record(*, session_id: str, turn: int, task: str, response: str,
                 metrics: dict | None = None,
                 tools_available: list | None = None,
                 owner: str | None = None,
                 topic_hint: str | None = None,
                 model: str | None = None,
                 stop_reason: str | None = None,
                 playbook_version=None,
                 episodes_used: list | None = None) -> dict:
    """Assemble one task record.  Pure: writes nothing, raises nothing."""
    metrics = metrics or {}
    return {
        "schema": SCHEMA_VERSION,
        "kind": KIND_TASK,
        "session_id": session_id,
        "turn": int(turn),
        "ts": _now(),
        "owner": owner_hash(owner),
        "model": model,
        "topic_hint": topic_hint,
        "task": task,
        "response": response,
        "tools_available": sorted(tools_available or []),
        "trajectory": [_turn_view(t) for t in (metrics.get("turns") or [])],
        "metrics": {k: v for k, v in metrics.items() if k != "turns"},
        "signals": derive_signals(metrics, stop_reason=stop_reason),
        # What the agent was told beyond the task itself.  Empty until the
        # loops that fill it exist; present from the start because a run
        # recorded without it cannot be compared against one recorded with it,
        # and comparing those two is the entire point of recording either.
        "learned_context": {
            "playbook_version": playbook_version,
            "episodes_used": list(episodes_used or []),
        },
    }


def session_file(session_id: str, config_data: dict | None = None) -> str:
    """Path of the trajectory file for a session."""
    return os.path.join(trajectory_dir(config_data),
                        f"{_safe_name(session_id)}.jsonl")


def _append(path: str, record: dict) -> bool:
    """Append one JSON line.  Never raises: this is not the user's task."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return True
    except Exception as e:  # pragma: no cover - disk full, permissions, …
        logger.debug("trajectory write failed for %s: %s", path, e)
        return False


def write_record(record: dict, config_data: dict | None = None) -> str | None:
    """Append a record to its session's file.  Returns the path, or None."""
    if not recording_enabled(config_data):
        return None
    path = session_file(record.get("session_id", ""), config_data)
    return path if _append(path, record) else None


def record_task(*, session_id: str, turn: int, task: str, response: str,
                config_data: dict | None = None, **kwargs) -> str | None:
    """Build and append one task record.

    The single call the agent loop makes.  Best-effort by construction: a
    trajectory that fails to write must not turn a completed task into a failed
    one, so nothing here propagates.
    """
    if not recording_enabled(config_data):
        return None
    try:
        record = build_record(session_id=session_id, turn=turn, task=task,
                              response=response, **kwargs)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("trajectory record build failed: %s", e)
        return None
    return write_record(record, config_data)


def append_rating(*, session_id: str, turn: int, rating,
                  comment: str | None = None,
                  source: str = "user",
                  config_data: dict | None = None) -> str | None:
    """Record a verdict on an earlier task.

    ``rating`` is normalized to +1 / -1 / None so that a thumbs-up from the web
    UI, a "good" from a gateway and a 1 from a script all land as one value.
    """
    if not recording_enabled(config_data):
        return None
    record = {
        "schema": SCHEMA_VERSION,
        "kind": KIND_RATING,
        "session_id": session_id,
        "turn": int(turn),
        "ts": _now(),
        "rating": normalize_rating(rating),
        "comment": (comment or "")[:2000] or None,
        "source": source,
    }
    return write_record(record, config_data)


_POSITIVE = {"1", "+1", "up", "good", "yes", "true", "👍"}
_NEGATIVE = {"-1", "down", "bad", "no", "false", "👎"}


def normalize_rating(rating) -> int | None:
    """+1, -1 or None, from whatever spelling arrived."""
    if rating is None:
        return None
    if isinstance(rating, bool):
        return 1 if rating else -1
    if isinstance(rating, (int, float)):
        if rating > 0:
            return 1
        if rating < 0:
            return -1
        return None
    text = str(rating).strip().lower()
    if text in _POSITIVE:
        return 1
    if text in _NEGATIVE:
        return -1
    return None


def read_session(session_id: str, config_data: dict | None = None) -> list[dict]:
    """Task records for a session, with later ratings folded in.

    Ratings are stored as their own lines (see the module docstring); a reader
    that ignored them would see every ``user_rating`` as None.
    """
    path = session_file(session_id, config_data)
    tasks: list[dict] = []
    ratings: dict[int, dict] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn line loses one record, not the file
                if record.get("kind") == KIND_RATING:
                    ratings[record.get("turn")] = record
                elif record.get("kind", KIND_TASK) == KIND_TASK:
                    tasks.append(record)
    except FileNotFoundError:
        return []
    except OSError as e:  # pragma: no cover - permissions
        logger.debug("trajectory read failed for %s: %s", path, e)
        return []

    for record in tasks:
        rating = ratings.get(record.get("turn"))
        if rating:
            signals = record.setdefault("signals", {})
            signals["user_rating"] = rating.get("rating")
            if rating.get("comment"):
                signals["user_comment"] = rating["comment"]
    return tasks


def iter_records(config_data: dict | None = None):
    """Every task record on disk, oldest session file first.

    The entry point for the loops that come after this one — reflection,
    skill mining, holdout construction — none of which exist yet.
    """
    directory = trajectory_dir(config_data)
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        for record in read_session(name[:-len(".jsonl")], config_data):
            yield record
