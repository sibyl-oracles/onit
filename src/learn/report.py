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

What the trajectory store has to say so far.

Recording that nobody can read is indistinguishable from not recording, so this
answers the two questions worth asking of it: is anything being written, and
what does it say.  The second one is the Phase 0 exit criterion in
``docs/SELF_IMPROVEMENT.md`` — *which tool fails most, and on what?* — and it is
worth having even if none of the later loops are ever built.
"""

from collections import Counter, defaultdict

from .config import autonomy, level_name, recording_enabled, trajectory_dir
from .trajectory import iter_records


def summarize(config_data: dict | None = None) -> dict:
    """Aggregate every recorded task into counts worth printing."""
    tools: dict[str, Counter] = defaultdict(Counter)
    tool_ms: Counter = Counter()
    sessions: set[str] = set()
    models: Counter = Counter()
    ratings: Counter = Counter()
    tasks = 0
    totals = Counter()
    turn_counts: list[int] = []

    for record in iter_records(config_data):
        # Event records share these files (one event log, two loops —
        # SELF_IMPROVEMENT.md §4.4); they are loop bookkeeping, not tasks, and
        # counting them would inflate every number below.
        if record.get("kind") not in (None, "task"):
            continue
        tasks += 1
        sessions.add(record.get("session_id"))
        if record.get("model"):
            models[record["model"]] += 1

        signals = record.get("signals") or {}
        for key in ("tool_errors", "retries", "truncations", "compactions"):
            totals[key] += int(signals.get(key) or 0)
        verdict = signals.get("user_rating")
        if verdict:
            ratings["up" if verdict > 0 else "down"] += 1

        metrics = record.get("metrics") or {}
        if metrics.get("turn_count"):
            turn_counts.append(int(metrics["turn_count"]))
        totals["completion_tokens"] += int(metrics.get("completion_tokens") or 0)

        for turn in record.get("trajectory") or []:
            for call in turn.get("tools") or []:
                name = call.get("name", "?")
                tools[name]["calls"] += 1
                # A record predating per-call detail has no "ok"; counting it
                # as a failure would invent errors that never happened.
                if call.get("ok") is False:
                    tools[name]["errors"] += 1
                tool_ms[name] += int(call.get("ms") or 0)

    return {
        "tasks": tasks,
        "sessions": len(sessions),
        "models": models,
        "ratings": ratings,
        "totals": totals,
        "avg_turns": (sum(turn_counts) / len(turn_counts)) if turn_counts else 0.0,
        "tools": {name: {"calls": c["calls"], "errors": c["errors"],
                         "avg_ms": tool_ms[name] // max(c["calls"], 1)}
                  for name, c in tools.items()},
    }


def format_status(config_data: dict | None = None) -> str:
    """The whole store as one screen of text."""
    level = autonomy(config_data)
    directory = trajectory_dir(config_data)
    lines = [
        "Learning",
        f"  autonomy   {level_name(level)} ({level})",
        f"  recording  {'on' if recording_enabled(config_data) else 'off'}",
        f"  path       {directory}",
    ]

    if not recording_enabled(config_data):
        lines += ["", "Recording is off, so nothing is being written.",
                  "Set learn.autonomy: observe in your config, or ONIT_LEARN=observe."]
        return "\n".join(lines)

    s = summarize(config_data)
    if not s["tasks"]:
        lines += ["", "No trajectories recorded yet.",
                  "Run a task — one file per session appears under the path above."]
        return "\n".join(lines)

    totals = s["totals"]
    lines += [
        "",
        f"Recorded     {s['tasks']} task(s) across {s['sessions']} session(s)",
        f"Averages     {s['avg_turns']:.1f} turn(s) per task, "
        f"{totals['completion_tokens']:,} token(s) generated in total",
        f"Trouble      {totals['tool_errors']} tool error(s), "
        f"{totals['retries']} API retry/retries, "
        f"{totals['truncations']} truncation(s), "
        f"{totals['compactions']} compaction(s)",
    ]
    if s["ratings"]:
        lines.append(f"Ratings      {s['ratings'].get('up', 0)} up, "
                     f"{s['ratings'].get('down', 0)} down")
    if s["models"]:
        served = ", ".join(f"{name} ({n})" for name, n in s["models"].most_common(3))
        lines.append(f"Models       {served}")

    # The loops' own bookkeeping, one line: lifecycle events written so far.
    # Empty until the dynamic-tool loop starts emitting — the registry-load
    # event from the harness is usually the first.
    try:
        from .events import summarize_events
        ev = summarize_events(config_data)
        if ev["total"]:
            top = ", ".join(f"{name}×{n}" for name, n in ev["by_event"].items())
            lines += ["", f"Events       {ev['total']} — {top}"]
    except Exception:  # pragma: no cover - status is best-effort
        pass

    if s["tools"]:
        lines += ["", "Tools (worst failure rate first)",
                  f"  {'tool':<24} {'calls':>6} {'errors':>7} {'fail':>6} {'avg ms':>8}"]
        # The question this whole layer exists to answer, so it leads: sort by
        # failure rate, then by how often the tool is reached for at all.
        ranked = sorted(s["tools"].items(),
                        key=lambda kv: (-(kv[1]["errors"] / max(kv[1]["calls"], 1)),
                                        -kv[1]["calls"]))
        for name, stat in ranked:
            rate = stat["errors"] / max(stat["calls"], 1)
            lines.append(f"  {name[:24]:<24} {stat['calls']:>6} {stat['errors']:>7} "
                         f"{rate:>5.0%} {stat['avg_ms']:>8,}")

    return "\n".join(lines)
