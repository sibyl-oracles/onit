"""SWE-bench runner for the OnIt agent.

SWE-bench is a *repo-editing agent* benchmark, not a code-generation one, so it
does not fit the final-answer provider path used by the other coding tasks. This
runner integrates it the way OnIt actually works:

    1. **Edit (OnIt).** For each instance, the repo is checked out at its base
       commit into a per-instance workspace. OnIt's ``data_path`` is set to that
       workspace, so OnIt's ``read_file`` / ``edit_file`` / ``bash`` / ``grep``
       tools operate *inside the repo* (OnIt roots all file ops at ``data_path``).
       OnIt reads the issue, edits the source, and — when ``--onit-sandbox`` is
       set — runs code via its MCP sandbox provider instead of the host.
    2. **Capture.** The model patch is ``git diff`` of the workspace (test files
       excluded; the harness applies the gold test patch itself).
    3. **Grade (official harness).** Patches are written to ``predictions.jsonl``
       and scored by the official ``swebench`` Docker harness, which applies each
       patch to the canonical per-instance image and runs the tests. This is the
       same grader used by the public leaderboard.

Prerequisites: a Docker daemon, ``pip install swebench``, network access to
clone repos and pull SWE-bench images, and the usual ``ONIT_BENCH_*`` eval
target. See ``benchmarks/README.md`` → "SWE-bench" for full instructions.

Usage:
    python -m benchmarks.swe_bench_runner --dataset lite --tier smoke
    python -m benchmarks.swe_bench_runner --dataset verified --limit 50 \
        --onit-sandbox --run-id onit-v1 --max-workers 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from . import config as bench_config

# Module logger. By default it has no handlers (library behaviour); ``main`` /
# ``setup_logging`` attach a console + file handler so the final score and any
# per-instance errors land in a tail-able log file for monitoring.
log = logging.getLogger("benchmarks.swe_bench")


def setup_logging(log_file: str | Path | None) -> Path | None:
    """Send this runner's output to the console and (optionally) a log file.

    Returns the resolved log-file path (or ``None`` if no file was requested) so
    callers can report where to tail. Idempotent: re-attaching to the same file
    does not duplicate handlers.
    """
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")

    have_stream = any(isinstance(h, logging.StreamHandler) and
                      not isinstance(h, logging.FileHandler) for h in log.handlers)
    if not have_stream:
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        log.addHandler(stream)

    if not log_file:
        return None
    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    already = any(
        isinstance(h, logging.FileHandler) and
        os.path.realpath(getattr(h, "baseFilename", "")) == resolved
        for h in log.handlers
    )
    if not already:
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return path

# Public SWE-bench dataset ids on Hugging Face.
DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}

# Test paths whose edits are dropped from the model patch (the harness supplies
# the gold test patch; the model must not "fix" the tests).
_TEST_PATH_HINTS = ("test", "tests/")

_PROMPT = """\
You are fixing a bug in a software repository. The repository is your working \
directory ({workspace}); all your file tools operate inside it.

Resolve the following GitHub issue by editing the repository's source code. Make \
the smallest change that fixes the issue. Do NOT edit test files — the grader \
supplies its own tests. When done, ensure the change is saved to disk.

<issue>
{issue}
</issue>
"""


def _run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def _prepare_workspace(repo: str, base_commit: str, dest: Path) -> None:
    """Clone ``repo`` (``owner/name``) at ``base_commit`` into ``dest``.

    If ``dest`` already holds a valid checkout (e.g. a resumed run re-attempting
    an instance that errored last time), reset it to ``base_commit`` and discard
    any leftover edits so the captured diff starts from a clean base. A leftover
    directory that is *not* a valid repo (a half-finished clone) is removed and
    cloned fresh.
    """
    if (dest / ".git").is_dir():
        _run(["git", "reset", "--hard", "--quiet", base_commit], cwd=str(dest))
        _run(["git", "clean", "-fdq"], cwd=str(dest))
        return
    if dest.exists():
        _run(["rm", "-rf", str(dest)])
    url = f"https://github.com/{repo}.git"
    _run(["git", "clone", "--quiet", url, str(dest)])
    _run(["git", "checkout", "--quiet", base_commit], cwd=str(dest))


def _strip_test_hunks(diff: str) -> str:
    """Drop per-file hunks that touch test files from a unified ``git diff``.

    The harness applies the gold test patch itself, so model edits to tests must
    not leak into the prediction. Splits on ``diff --git`` headers and keeps only
    files whose path has no test marker.
    """
    keep, skip = [], False
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git"):
            path = line.split(" b/")[-1].strip()
            skip = any(h in path.lower() for h in _TEST_PATH_HINTS)
        if not skip:
            keep.append(line)
    return "".join(keep)


def _model_patch(workspace: Path) -> str:
    """Return the agent's patch (working-tree diff, excluding test files)."""
    diff = _run(["git", "diff"], cwd=str(workspace), check=False).stdout
    return _strip_test_hunks(diff)


def _load_existing(preds_path: Path) -> dict[str, dict]:
    """Load prior predictions keyed by ``instance_id`` (for resume).

    Tolerates a truncated final line from a hard crash (e.g. the process was
    killed mid-write) by skipping records that don't parse.
    """
    results: dict[str, dict] = {}
    if not preds_path.exists():
        return results
    for line in preds_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        iid = rec.get("instance_id")
        if iid:
            results[iid] = rec
    return results


def _write_predictions(preds_path: Path, results: dict[str, dict]) -> None:
    """Atomically rewrite the predictions file from the in-memory results map.

    Written after every instance so a run that dies (out of credits, OOM, Ctrl-C)
    leaves a consistent file we can resume from. The rewrite is cheap relative to
    a single agent call, and ``os.replace`` is atomic so a crash mid-write can
    never corrupt the existing predictions.
    """
    tmp = preds_path.with_suffix(preds_path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for rec in results.values():
            fh.write(json.dumps(rec) + "\n")
    tmp.replace(preds_path)


async def _solve_instance(agent, inst: dict, workspace: Path, timeout: int) -> str:
    prompt = _PROMPT.format(workspace=str(workspace), issue=inst["problem_statement"])
    run_id = uuid.uuid4().hex[:12]
    sessions = Path(tempfile.gettempdir()) / "onit-bench-sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    await agent.process_task(
        prompt,
        data_path=str(workspace),
        session_path=str(sessions / f"swe-{run_id}.jsonl"),
        safety_queue=asyncio.Queue(),
    )
    return _model_patch(workspace)


def generate_predictions(args) -> tuple[Path, str]:
    """Run OnIt over the instances and write ``predictions.jsonl``."""
    from datasets import load_dataset

    from .onit_provider import _build_agent_blocking

    dataset_id = DATASETS[args.dataset]
    rows = list(load_dataset(dataset_id, split=args.split))
    if args.limit:
        rows = rows[: args.limit]

    model_name = bench_config.model_label()
    overrides = {"sandbox": True} if args.onit_sandbox else None
    timeout = bench_config.bench_timeout()

    data_root = Path(args.data_root).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    preds_path = data_root / f"predictions_{args.run_id}.jsonl"

    # Resume: keep instances that already succeeded and re-attempt the rest. A
    # run halted mid-way (e.g. out of cloud credits) leaves the completed slice
    # on disk, so we only spend on instances that never finished. Only "ok"
    # records count as done — an instance that errored last time (empty patch)
    # is retried, so a credit outage doesn't permanently zero out the tail.
    results = {} if args.fresh else _load_existing(preds_path)
    done = {iid for iid, rec in results.items() if rec.get("_onit_status") == "ok"}
    todo = [r for r in rows if r["instance_id"] not in done]
    if done:
        log.info("[swe-bench] resuming: %d/%d already complete, %d to run",
                 len(done), len(rows), len(todo))
    if not todo:
        log.info("[swe-bench] all %d instances already complete: %s", len(rows), preds_path)
        return preds_path, model_name

    # Build the agent only once there is real work to do.
    agent = _build_agent_blocking(overrides)  # sync: no running loop here

    for i, inst in enumerate(rows, 1):
        iid = inst["instance_id"]
        if iid in done:
            continue
        ws = data_root / iid
        log.info("[swe-bench] (%d/%d) %s", i, len(rows), iid)
        try:
            _prepare_workspace(inst["repo"], inst["base_commit"], ws)
            patch = asyncio.run(_solve_instance(agent, inst, ws, timeout))
            status = "ok"
            if not patch.strip():
                log.warning("  ! %s produced an empty patch", iid)
        except Exception as exc:  # noqa: BLE001 - record error; resume retries it
            log.error("  ! %s failed: %s", iid, exc, exc_info=True)
            patch, status = "", "error"
        results[iid] = {
            "instance_id": iid,
            "model_name_or_path": model_name,
            "model_patch": patch,
            "_onit_status": status,  # internal; the swebench harness ignores it
        }
        _write_predictions(preds_path, results)

    errored = [iid for iid, rec in results.items() if rec.get("_onit_status") == "error"]
    if errored:
        log.warning("[swe-bench] %d instance(s) errored: %s", len(errored), ", ".join(errored))
    log.info("[swe-bench] wrote %s", preds_path)
    return preds_path, model_name


def grade(preds_path: Path, args, model_name: str) -> None:
    """Invoke the official SWE-bench Docker harness on the predictions."""
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", DATASETS[args.dataset],
        "--split", args.split,
        "--predictions_path", str(preds_path),
        "--run_id", args.run_id,
        "--max_workers", str(args.max_workers),
    ]
    log.info("[swe-bench] grading: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        log.error("[swe-bench] grading harness failed (exit %s): %s", exc.returncode, exc)
        raise
    # The harness writes <model>.<run_id>.json in the CWD.
    report = Path(f"{model_name}.{args.run_id}.json".replace("/", "__"))
    if report.exists():
        data = json.loads(report.read_text())
        total = data.get("total_instances") or data.get("submitted_instances")
        resolved = len(data.get("resolved_ids", []))
        rate = (resolved / total) if total else 0.0
        log.info("\n# SWE-bench summary\n")
        log.info("| Dataset | Model | Instances | Resolved | Resolve rate |")
        log.info("|---|---|---|---|---|")
        log.info("| %s | %s | %s | %s | %.3f |",
                 args.dataset, model_name, total, resolved, rate)
        log.info("[swe-bench] FINAL SCORE: resolved %s/%s (%.1f%%)",
                 resolved, total, rate * 100)
    else:
        log.error("[swe-bench] report %s not found; see harness logs.", report)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="benchmarks.swe_bench_runner", description=__doc__)
    p.add_argument("--dataset", choices=list(DATASETS), default="lite")
    p.add_argument("--split", default="test")
    p.add_argument("--tier", choices=list(bench_config.TIERS), default=None,
                   help="Convenience: sets --limit from the tier preset.")
    p.add_argument("--limit", type=int, default=None,
                   help="Max instances (overrides --tier limit).")
    p.add_argument("--onit-sandbox", action="store_true",
                   help="Run OnIt with sandbox mode (delegates code execution to "
                        "the MCP sandbox provider; requires one configured).")
    p.add_argument("--run-id", default="onit")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--data-root", default=str(Path(tempfile.gettempdir()) / "onit-swebench"))
    p.add_argument("--no-grade", action="store_true",
                   help="Only generate predictions; skip the Docker harness.")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing predictions file and start over "
                        "(default is to resume, skipping completed instances).")
    p.add_argument("--log-file", default=None,
                   help="Tee the run's progress, errors, and final score to this "
                        "file (in addition to the console) for easy monitoring.")
    args = p.parse_args(argv)

    log_path = setup_logging(args.log_file)
    if log_path:
        log.info("[swe-bench] logging to %s", log_path)

    if args.limit is None and args.tier:
        args.limit = bench_config.TIERS[args.tier].limit

    try:
        preds_path, model_name = generate_predictions(args)
        if not args.no_grade:
            grade(preds_path, args, model_name)
    except Exception:
        log.exception("[swe-bench] run failed")
        raise


if __name__ == "__main__":
    main()
