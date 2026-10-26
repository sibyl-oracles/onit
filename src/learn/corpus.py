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

Retroactive labelling of the corpus, and the report that says how good it is.

The corpus on disk was recorded before the implicit-signal heuristics
existed, so it carries no ``signals.implicit`` field.  This module puts
one there — over the records already written — so the 1,010 tasks that
exist today gain a second label without waiting for a single new session.

**Nothing is rewritten.**  The trajectory files are append-only and are
the audit trail; a backfill that rewrote them would destroy the very
evidence a later audit needs.  The labels are *derived* on read, and
``onit learn --signals`` reports them.  A deployment that wants them
materialised writes them to a separate derived file, which is disposable
by construction.

That is the same rule the retrieval index follows: derived things live
outside the record and can be thrown away and rebuilt.
"""

import os

from .config import learn_root, trajectory_dir
from .signals import label_records, summarize as summarize_labels
from .trajectory import _is_session_name, read_session

# Where the derived labels live.  Beside the trajectories, because it is
# derived from the trajectories - but it is a *file*, not a trajectory, and
# the trajectory walk must never see it.
#
# It used to live *inside* the trajectory directory, and that was a real
# bug: the walk lists every "*.jsonl" in that directory, so the label
# rows were read back as if they were recorded runs.  A store of 1,022
# tasks reported 2,044.  The label rows carry no response and no turn, so
# they were recalled as episodes of an agent that had never answered
# anything.  The file is derived, and derived things live outside the
# record they are derived from.
LABELS_NAME = "labels.jsonl"


def labels_path(config_data: dict | None = None) -> str:
    """Path of the derived label file."""
    return os.path.join(learn_root(config_data), LABELS_NAME)


def _session_ids(config_data: dict | None = None) -> list:
    """Every session file on disk, oldest first.

    The file *name* is not enough to decide what a ".jsonl" file in the
    trajectory directory is.  The store keeps its own derived files
    there - the event log, and older installs a labels.jsonl - and a
    walk that treats every one of them as a session inflates the
    corpus with rows that are not runs.  A name that is not a session
    id is skipped here, and the record shape is checked when the file
    is read.
    """
    directory = trajectory_dir(config_data)
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    return [n[:-len(".jsonl")] for n in names
            if n.endswith(".jsonl") and _is_session_name(n[:-len(".jsonl")])]


def read_sessions(config_data: dict | None = None) -> dict:
    """Every session's task records, keyed by session id, in turn order.

    The per-session grouping is the whole point: the implicit signal is
    read off the task that *follows* one, so the records have to be seen
    together and in order.  A flat list of every record on disk would
    label a task against a stranger's.
    """
    sessions: dict = {}
    for session_id in _session_ids(config_data):
        records = read_session(session_id, config_data)
        if records:
            sessions[session_id] = records
    return sessions


def label_corpus(config_data: dict | None = None) -> dict:
    """Label every session's records and return the summary.

    Pure with respect to the store: nothing is written.  The summary says
    how many records carry a usable label and how many do not, because a
    corpus of 1,010 records with 900 labels and one with 90 are very
    different instruments and the difference has to be visible.
    """
    sessions = read_sessions(config_data)
    all_records: list = []
    for records in sessions.values():
        label_records(records)
        all_records.extend(records)
    summary = summarize_labels(all_records)
    summary["sessions"] = len(sessions)
    return summary


def write_labels(config_data: dict | None = None) -> str | None:
    """Materialise the derived labels next to the store.

    Returns the path written, or None when recording is off or the
    write failed.  Best-effort: a corpus that cannot be written must not
    stop the agent from answering.
    """
    from .config import recording_enabled
    if not recording_enabled(config_data):
        return None
    sessions = read_sessions(config_data)
    path = labels_path(config_data)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            import json
            for session_id, records in sessions.items():
                label_records(records)
                for record in records:
                    f.write(json.dumps({
                        "session_id": session_id,
                        "turn": record.get("turn"),
                        "task": record.get("task"),
                        "signals": record.get("signals") or {},
                    }, default=str) + "\n")
    except OSError:
        return None
    return path


def read_labels(config_data: dict | None = None) -> list:
    """The materialised labels, if they were written."""
    import json
    path = labels_path(config_data)
    rows: list = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


def format_summary(summary: dict) -> str:
    """The signal report as one screen of text."""
    total = summary.get("total", 0)
    if not total:
        return ("No trajectories on disk, so there is nothing to label.\n"
                "Run a task first — one file per session appears under the "
                "learned path.")
    labeled = summary.get("labeled", 0)
    lines = [
        "Implicit signals",
        f"  records      {total} across {summary.get('sessions', 0)} session(s)",
        f"  labeled      {labeled} ({summary.get('coverage', 0.0) * 100:.1f}% of the corpus)",
        f"  good / bad   {summary.get('good', 0)} / {summary.get('bad', 0)}",
        f"  unlabeled    {summary.get('unknown', 0)}",
    ]
    if summary.get("explicit_ratings"):
        lines.append(f"  explicit     {summary['explicit_ratings']} human rating(s)")
    if labeled:
        lines.append(f"  positive     {summary.get('positive_rate', 0.0) * 100:.1f}% "
                     f"of the labelled records")
    lines += [
        "",
        "The label is read off what the user did next: a rephrase of the",
        "same task means the answer missed, an explicit 'thanks' means it",
        "landed, and a session that ended means nobody objected. The last",
        "of those is the weakest and is counted separately.",
    ]
    return "\n".join(lines)
