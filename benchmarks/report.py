"""Aggregate Inspect ``.eval`` logs into a markdown + JSON summary.

Reads every eval log under a directory, extracts the headline metric per
benchmark, writes ``summary.json`` and ``summary.md`` alongside them, and
(optionally) diffs against a committed baseline to flag regressions. The CI
smoke gate uses ``--baseline`` + ``--fail-on-regression`` to fail on drops.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from . import config as bench_config

# A metric drop larger than this (absolute) fails the regression gate.
DEFAULT_TOLERANCE = 0.05


def collect(log_dir: str) -> dict[str, dict]:
    """Map ``task -> {metric: value, ...}`` from all logs under ``log_dir``."""
    results: dict[str, dict] = {}
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info, header_only=True)
        if log.status != "success" or not log.results:
            continue
        task = log.eval.task
        metrics: dict[str, float] = {}
        for score in log.results.scores:
            for name, metric in score.metrics.items():
                metrics[f"{score.name}/{name}"] = metric.value
        results[task] = {
            "model": log.eval.model,
            "samples": log.results.total_samples,
            "metrics": metrics,
        }
    return results


def to_markdown(results: dict[str, dict]) -> str:
    lines = ["# OnIt benchmark summary", "",
             "| Benchmark | Alias | Model | Samples | Metrics |",
             "|---|---|---|---|---|"]
    for task in sorted(results):
        r = results[task]
        # ``task`` is the Inspect task name == the CLI alias.
        name = bench_config.display_name(task)
        metrics = ", ".join(f"{k}={v:.3f}" for k, v in sorted(r["metrics"].items()))
        lines.append(f"| {name} | {task} | {r['model']} | {r['samples']} | {metrics} |")
    return "\n".join(lines) + "\n"


def diff_baseline(results: dict, baseline: dict, tolerance: float) -> list[str]:
    """Return human-readable regression messages (empty if none)."""
    regressions: list[str] = []
    for task, base in baseline.items():
        cur = results.get(task)
        if not cur:
            regressions.append(f"{task}: missing from current run")
            continue
        for metric, base_val in base.get("metrics", {}).items():
            cur_val = cur["metrics"].get(metric)
            if cur_val is None:
                regressions.append(f"{task}/{metric}: missing")
            elif cur_val < base_val - tolerance:
                regressions.append(
                    f"{task}/{metric}: {cur_val:.3f} < baseline {base_val:.3f} "
                    f"(tol {tolerance})")
    return regressions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="benchmarks.report", description=__doc__)
    parser.add_argument("--log-dir", default="benchmarks/logs")
    parser.add_argument("--baseline", help="Baseline summary.json to diff against.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args(argv)

    results = collect(args.log_dir)
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    (out_dir / "summary.md").write_text(to_markdown(results))
    print(to_markdown(results))

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        regressions = diff_baseline(results, baseline, args.tolerance)
        if regressions:
            print("\nREGRESSIONS:")
            for r in regressions:
                print(f"  - {r}")
            if args.fail_on_regression:
                sys.exit(1)
        else:
            print("\nNo regressions vs baseline.")


if __name__ == "__main__":
    main()
