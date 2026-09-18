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
    """Map ``task -> {metric: value, ...}`` from all logs under ``log_dir``.

    Alongside the headline scores, aggregates the per-sample wall/token
    metadata the OnIt provider publishes (S1): wall seconds per sample,
    prompt/output tokens, cache hits, compactions, and TTFT percentiles.
    These are the columns every speed change in the plan is judged on.
    """
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
    _collect_speed(results, log_dir)
    return results


# Sample metadata report.py aggregates for the speed columns.  Each entry is
# (metadata key, aggregation): mean for rates and times, max for peaks,
# p50/p95 for the latency percentiles the plan asks TTFT to be read as.
_SPEED_FIELDS = (
    ("wall_s", "mean"),
    ("model_s", "mean"),
    ("prefill_s", "mean"),
    ("decode_s", "mean"),
    ("tool_s", "mean"),
    ("ttft_s", "p50"),
    ("ttft_s", "p95"),
    ("prompt_tokens_max", "mean"),
    ("completion_tokens", "mean"),
    ("cached_tokens", "mean"),
    ("compactions", "mean"),
    ("retries", "mean"),
)


def _collect_speed(results: dict[str, dict], log_dir: str) -> None:
    """Fold per-sample speed metadata into ``results`` as ``speed/<field>``.

    Read header-only first; samples are only re-read (the expensive path)
    when a task has none of the columns, which is exactly the pre-S1 logs.
    """
    for task, r in results.items():
        vals: dict[tuple, list] = {}
        for info in list_eval_logs(log_dir, task=task):
            try:
                log = read_eval_log(info, header_only=False)
            except Exception:
                continue
            if log.status != "success":
                continue
            for sample in (log.samples or []):
                md = getattr(sample, "metadata", None) or {}
                if not isinstance(md, dict) or "wall_s" not in md:
                    continue
                for key, agg in _SPEED_FIELDS:
                    v = md.get(key)
                    if v is None:
                        continue
                    vals.setdefault((key, agg), []).append(float(v))
        if not vals:
            continue
        speed: dict[str, float] = {}
        for (key, agg), xs in vals.items():
            if agg == "mean":
                speed[key] = sum(xs) / len(xs)
            elif agg == "p50":
                s = sorted(xs)
                speed["ttft_p50_s"] = s[len(s) // 2]
            elif agg == "p95":
                s = sorted(xs)
                speed["ttft_p95_s"] = s[min(len(s) - 1, int(len(s) * 0.95))]
        # Token-weighted cache hit rate across the task's samples.
        _pt = sum(sum(xs) for (k, _), xs in vals.items() if k == "prompt_tokens_max")
        _ct = sum(sum(xs) for (k, _), xs in vals.items() if k == "cached_tokens")
        if _pt > 0:
            speed["cache_hit_pct"] = _ct / _pt
        r["speed"] = speed


def read_run_meta(log_dir: str) -> dict:
    """Agent configuration the logs were produced under, if it was stamped.

    Written by ``benchmarks.run``; absent for logs produced before it was, and
    for ``eval`` invocations that bypassed the runner.
    """
    try:
        return json.loads((Path(log_dir) / "run_meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def to_markdown(results: dict[str, dict], meta: dict | None = None) -> str:
    lines = ["# OnIt benchmark summary", ""]
    # Two runs of the same benchmark are only comparable if the agent was
    # allowed to change itself by the same amount in both.
    learn = (meta or {}).get("learn")
    if learn:
        lines += [f"Autonomy level (`learn`): **{learn}**", ""]
    # The speed columns exist when the logs carry per-sample metadata
    # (post-S1 provider).  Older logs render the accuracy-only table.
    _has_speed = any("speed" in r for r in results.values())
    if _has_speed:
        lines += ["| Benchmark | Alias | Samples | Accuracy | Wall s/sample | Model s (prefill/decode) | TTFT p50/p95 s | Prompt tok (peak) | Out tok | Cache hit | Compactions |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for task in sorted(results):
            r = results[task]
            name = bench_config.display_name(task)
            acc = ", ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in sorted(r["metrics"].items()))
            sp = r.get("speed") or {}
            lines.append(
                f"| {name} | {task} | {r['samples']} | {acc} "
                f"| {sp.get('wall_s', 0):.1f} "
                f"| {sp.get('model_s', 0):.1f} ({sp.get('prefill_s', 0):.1f}/{sp.get('decode_s', 0):.1f}) "
                f"| {sp.get('ttft_p50_s', 0):.1f}/{sp.get('ttft_p95_s', 0):.1f} "
                f"| {sp.get('prompt_tokens_max', 0):,.0f} "
                f"| {sp.get('completion_tokens', 0):,.0f} "
                f"| {sp.get('cache_hit_pct', 0):.0%} "
                f"| {sp.get('compactions', 0):.1f} |")
        return "\n".join(lines) + "\n"
    lines += ["| Benchmark | Alias | Model | Samples | Metrics |",
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
    meta = read_run_meta(args.log_dir)
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"run": meta, "results": results} if meta else results
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(to_markdown(results, meta))
    print(to_markdown(results, meta))

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        # Summaries stamped with a run block nest the per-task rows; ones
        # written before that (including the committed baselines) do not.
        baseline = baseline.get("results", baseline)
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
