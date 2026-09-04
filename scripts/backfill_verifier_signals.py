#!/usr/bin/env python3
"""Backfill ``signals.verifier`` in recorded trajectories.

One-off for docs/SELF_IMPROVEMENT.md §8 item 1.  ``derive_signals()`` wrote
``"verifier": None`` into every task record while the fact-check's issue
count sat unread in the same record's metrics blob.  This script reads the
field with the same function the live writer now uses —
``learn.trajectory.verifier_signal()`` — so the labels it writes cannot
drift from the ones future runs get, and rewrites each session file in
place.

Per file the rewrite is atomic (write to a temp file in the same directory,
then ``os.replace``), and the file is skipped if it changed size while being
copied — a session being appended to by a live agent must not be truncated.

Usage:
    python3 scripts/backfill_verifier_signals.py            # dry run
    python3 scripts/backfill_verifier_signals.py --apply    # write
    python3 scripts/backfill_verifier_signals.py --path DIR # other store
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from learn.trajectory import verifier_signal  # noqa: E402

DEFAULT_PATH = os.path.expanduser("~/.onit/learned/trajectories")


def backfill_file(path: str, apply: bool) -> tuple[int, int]:
    """Return (records_fixed, records_scanned) for one session file."""
    lines_out: list[str] = []
    changed = scanned = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                lines_out.append(line)
                continue
            if record.get("kind") != "task":
                lines_out.append(line)
                continue
            scanned += 1
            signals = record.get("signals")
            if not isinstance(signals, dict) or signals.get("verifier") is not None:
                lines_out.append(line)
                continue
            label = verifier_signal(record.get("metrics"))
            if label is None:
                lines_out.append(line)
                continue
            signals["verifier"] = label
            changed += 1
            lines_out.append(json.dumps(record, default=str) + "\n")
    if apply and changed:
        size_before = os.path.getsize(path)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.writelines(lines_out)
            if os.path.getsize(path) != size_before:
                os.unlink(tmp)
                print(f"  SKIP {path}: file changed while reading")
                return 0, scanned
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    return changed, scanned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", default=DEFAULT_PATH,
                        help=f"Trajectory directory (default: {DEFAULT_PATH})")
    parser.add_argument("--apply", action="store_true",
                        help="Write the backfill. Without this, only report.")
    args = parser.parse_args()

    total_changed = total_scanned = 0
    for name in sorted(os.listdir(args.path)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(args.path, name)
        changed, scanned = backfill_file(path, args.apply)
        total_changed += changed
        total_scanned += scanned
        if changed:
            print(f"  {name}: {changed} of {scanned} task record(s) "
                  f"{'backfilled' if args.apply else 'backfillable'}")

    mode = "backfilled" if args.apply else "backfillable (dry run; use --apply)"
    print(f"\n{total_changed} record(s) {mode} out of "
          f"{total_scanned} scanned in {args.path}")


if __name__ == "__main__":
    main()