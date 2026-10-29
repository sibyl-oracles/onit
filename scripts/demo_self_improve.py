#!/usr/bin/env python3
"""Demonstrate OnIt's self-improvement loops, from an empty store.

This is the *mechanism* demo: it shows the agent record a run, recall an
earlier run, distil a lesson from a contrast, and then be shown that
lesson.  Every stage is driven through the real code paths -- the real
agent through the real ``process_task``, the real trajectory store on
disk, the real Reflector and Curator through the real CLI.  Nothing is
mocked except the *inputs*, and the inputs are labelled as such.

What this does NOT do is show that the loop makes the agent *better*.
That is a separate claim, with a separate instrument:

    onit --config <cfg> learn --holdout
    onit --config <cfg> learn --eval-run --ab --eval-split dev

Two arms, one split, a delta with an error bar.  The demo proves the
machinery; the A/B decides the question.

Usage:
    python3 scripts/demo_self_improve.py
    python3 scripts/demo_self_improve.py --config my-config.yaml
    python3 scripts/demo_self_improve.py --host http://localhost:8001/v1 \\
        --model Qwen/Qwen3.8-27B

The store is created under ``tmp/`` and deleted on every run, so the demo
always starts from *no logs*.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# The repository root goes on the path, NOT ``src/``: ``src/mcp/`` would
# then shadow the installed ``mcp`` SDK and break FastMCP's import.  The
# package is imported as ``src.<module>`` throughout instead.
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

# The question the demo runs on.  It is short on purpose: the point is
# the machinery, not the task.
TASK = "What is the capital of Japan? Answer in one sentence."

# Where the demo's own store lives.  Under tmp/, which is gitignored, so a
# demo run never touches an operator's real corpus.
WORK = os.path.join(ROOT, "tmp", "self-improve-demo")
STORE = os.path.join(WORK, "store")
DATA = os.path.join(WORK, "data")
SESSIONS = os.path.join(WORK, "sessions")
CONFIG = os.path.join(WORK, "config.yaml")


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print("  " + text)
    print("=" * 70)


def fresh() -> None:
    """Delete the store.  This is the 'no logs' the demo starts from."""
    shutil.rmtree(WORK, ignore_errors=True)
    for path in (STORE, DATA, SESSIONS):
        os.makedirs(path, exist_ok=True)


def write_config(host: str | None, model: str | None) -> dict:
    """The demo's own config, so the operator's is never touched."""
    config = {
        "serving": {"host": host or "http://202.92.159.240:8001/v1",
                    "model": model or "Qwen/Qwen3.8-27B"},
        "learn": {"autonomy": "adapt", "path": STORE},
        "data_path": DATA,
        "session_path": SESSIONS,
    }
    with open(CONFIG, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    return config


async def ask(agent, task: str, session_name: str) -> str:
    """One task, through the real agent, with a real safety queue.

    The queue is not optional: ``process_task`` drains it before the run,
    and a caller that leaves it unset is the difference between a run that
    answers and a run that raises.
    """
    queue = asyncio.Queue(maxsize=10)
    path = os.path.join(SESSIONS, session_name + ".jsonl")
    return await agent.process_task(task, session_path=path, data_path=DATA,
                                    safety_queue=queue)


def show_records(cfg: dict) -> list:
    from src.learn.trajectory import iter_records
    records = list(iter_records(cfg))
    print("  trajectories on disk:")
    for r in records:
        lc = r.get("learned_context") or {}
        print("    session=%-8s turn=%s  episodes=%-30s playbook_v=%s" % (
            (r.get("session_id") or "")[:8], r.get("turn"),
            str(lc.get("episodes_used")), lc.get("playbook_version")))
    return records


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                        help="An existing config to read the endpoint from.")
    parser.add_argument("--host", default=None, help="Serving host URL.")
    parser.add_argument("--model", default=None, help="Model name.")
    args = parser.parse_args()

    fresh()
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
        host = args.host or (base.get("serving") or {}).get("host")
        model = args.model or (base.get("serving") or {}).get("model")
    else:
        host, model = args.host, args.model
    cfg = write_config(host, model)

    from src.cli import _ensure_mcp_servers
    from src.onit import OnIt
    from src.learn.trajectory import iter_records, record_task
    from src.learn import playbook as pb

    # ── 1. the cold agent ───────────────────────────────────────────
    banner("1. A cold agent: empty store, nothing to recall")
    _ensure_mcp_servers(cfg)
    agent = await asyncio.get_running_loop().run_in_executor(
        None, lambda: OnIt(cfg))
    answer = await ask(agent, TASK, "s1")
    print("  answer:", answer[:110])
    records = show_records(cfg)
    print("  episodes recalled: %s" %
          (records[0].get("learned_context") or {}).get("episodes_used"))
    print("  -> nothing to recall from.  The store was empty.")

    # ── 2. the same question again ──────────────────────────────────
    banner("2. The same question again -- Loop A (recall) fires")
    answer = await ask(agent, TASK, "s2")
    print("  answer:", answer[:110])
    records = show_records(cfg)
    used = (records[-1].get("learned_context") or {}).get("episodes_used")
    print("  episodes recalled by the second run: %s" % used)
    print("  -> the second run was SHOWN the first run's answer, marked")
    print("     as untrusted history, before it was asked the question.")

    # ── 3. a contrast, then the Reflector writes a bullet ───────────
    banner("3. A contrast, then the Reflector writes a bullet")
    print("  A lesson needs two runs of the same task shape: one that")
    print("  went well and one that did not.  Real use produces that on")
    print("  its own -- the fact-check's verdict is the reward.  It is")
    print("  seeded here through the same record_task() the agent itself")
    print("  calls, so every later stage is exercised for real.")
    # The two seeded runs carry a *tool call* as well as a verdict, and
    # that is not decoration.  The mechanical reflector contrasts the
    # tools the good runs used against the tools the bad runs used; two
    # runs that called nothing give it an empty difference, so it writes
    # no bullet -- and this demo then claimed, on the screen below, that
    # a bullet it had never written was in the prompt.  The tool call is
    # what makes the contrast something the reflector can name.
    record_task(session_id="s_bad", turn=1, task=TASK,
                response="Kyoto is the capital of Japan.",
                metrics={"verify_s": 1.2, "verify_issues": 2,
                         "turns": [{"n": 1,
                                    "tool_runs": [{"name": "search",
                                                   "ok": False}]}]},
                config_data=cfg)
    record_task(session_id="s_good", turn=1, task=TASK,
                response="Tokyo is the capital of Japan.",
                metrics={"verify_s": 1.1, "verify_issues": 0,
                         "turns": [{"n": 1,
                                    "tool_runs": [{"name": "local_search",
                                                   "ok": True}]}]},
                config_data=cfg)
    print("  seeded: s_bad (verifier found 2 issues, its search failed),")
    print("          s_good (clean, its local_search worked).")

    print("\n  running the real CLI:  onit learn --reflect")
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--config", CONFIG,
         "learn", "--reflect"],
        capture_output=True, text=True, cwd=ROOT)
    for line in result.stdout.splitlines():
        print("   ", line)
    if result.returncode != 0:
        print("   stderr:", result.stderr[-800:])

    # ── 4. the next run is shown the bullet ──────────────────────────
    banner("4. The next run is shown the bullet")
    playbook_path = pb.global_path(cfg)
    print("  playbook: %s" % playbook_path)
    if os.path.exists(playbook_path):
        book = json.load(open(playbook_path, encoding="utf-8"))
        print("  version: %s   bullets: %s" % (
            book.get("version"), len(book.get("items") or [])))
        for item in book.get("items") or []:
            print("    - [%s] %s" % (item.get("status"), item.get("text")))
    context = pb.build_context(cfg)
    print("\n  the block that goes into the prompt:")
    for line in (context["text"] or "(empty)").splitlines():
        print("   ", line)

    # The playbook is PINNED per OnIt instance, on purpose: the bytes must
    # not move between the turns of one session or the prefix cache misses.
    # So the deployment shape is "reflect offline, then start the agent" --
    # and a long-running process keeps the version it started with until
    # it is restarted.  Hence a new agent here.
    before = {(r.get("session_id"), r.get("turn"))
              for r in iter_records(cfg)}
    agent2 = await asyncio.get_running_loop().run_in_executor(
        None, lambda: OnIt(cfg))
    answer = await ask(agent2, TASK, "s3")
    print("\n  third run answer:", answer[:110])
    records = show_records(cfg)
    third = [r for r in records
             if (r.get("session_id"), r.get("turn")) not in before]
    lc = (third[0].get("learned_context") if third else {}) or {}
    print("  the third run's own record:")
    print("    playbook version seen: %s" % lc.get("playbook_version"))
    print("    bullets seen        : %s" % lc.get("playbook_items"))
    print("    episodes recalled   : %s" % lc.get("episodes_used"))

    # The claim is checked, not asserted.  This demo used to print "the
    # bullet was in the prompt, and the record says so" whatever the
    # record said -- on a store where the reflector had written no
    # bullet at all, it printed that line directly above
    # "bullets seen : []".  A demo that reports success it did not have
    # is worse than a demo that fails, because the failure is the thing
    # it exists to rule out.
    shown = lc.get("playbook_items") or []
    recalled = lc.get("episodes_used") or []
    claims = [
        ("a bullet reached the prompt", bool(shown)),
        ("an episode reached the prompt", bool(recalled)),
        ("the run was recorded", bool(third)),
    ]
    failed = [name for name, held in claims if not held]
    print()
    for name, held in claims:
        print("    [%s] %s" % ("ok" if held else "MISSING", name))
    if failed:
        print("  -> the demo did NOT show what it set out to show: %s."
              % ", ".join(failed))
        print("     Nothing above should be read as a working loop.")
        print("  Demo artifacts kept for inspection: %s" % WORK)
        return 1
    print("  -> the bullet was in the prompt, and the record says so.")

    # ── 5. the number ───────────────────────────────────────────────
    banner("5. The number: did the loop make the agent better?")
    print("  The mechanism above is real.  Whether it makes the agent")
    print("  BETTER is a separate claim, and it needs the instrument:")
    print()
    print("      onit --config %s learn --holdout" % CONFIG)
    print("      onit --config %s learn --eval-run --ab --eval-split dev" % CONFIG)
    print()
    print("  Two arms, the same split, a delta with an error bar.  A delta")
    print("  that does not clear the bar is reported as not significant,")
    print("  and the correct response to that is to ship nothing.")
    print()
    print("  Demo artifacts: %s" % WORK)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
