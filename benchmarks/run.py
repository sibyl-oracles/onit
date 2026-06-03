"""CLI entry point for the OnIt benchmark suite.

Thin wrapper over ``inspect_ai.eval`` that:
    * registers the ``onit`` model provider,
    * resolves the eval target (host/model) from the environment,
    * applies a tier preset (sample limit + concurrency),
    * and runs one or more benchmark tasks.

Examples:
    python -m benchmarks.run --tier smoke --tasks gsm8k
    python -m benchmarks.run --tier sampled --tasks gsm8k humaneval
    ONIT_BENCH_MODEL=glm-5.1:cloud python -m benchmarks.run --tier smoke --tasks gsm8k
"""

from __future__ import annotations

import argparse
import sys

from inspect_ai import eval as inspect_eval
from inspect_ai import eval_retry as inspect_eval_retry

# Importing the provider module registers the "onit" model provider.
from . import config as bench_config
from . import onit_provider  # noqa: F401 - import for side effect (registration)
from .tasks import coding, reasoning

# Registry of runnable benchmark tasks by name and category.
TASKS = {
    "gsm8k": reasoning.gsm8k,
    "humaneval": coding.humaneval,
    "mbpp": coding.mbpp,
    "bigcodebench": coding.bigcodebench,
    "livecodebench": coding.livecodebench,
    # Deferred: needs the native-tools bridge (see tasks/coding.py). Excluded
    # from default categories; invoke explicitly with `--tasks swe_bench`.
    "swe_bench": coding.swe_bench,
}
CATEGORIES = {
    "reasoning": ["gsm8k"],
    # Provider-compatible coding tasks (all require a Docker daemon).
    "coding": ["humaneval", "mbpp", "bigcodebench", "livecodebench"],
    # Everything wired and provider-compatible (excludes deferred swe_bench).
    "all": ["gsm8k", "humaneval", "mbpp", "bigcodebench", "livecodebench"],
}


def _resolve_task_names(requested: list[str]) -> list[str]:
    names: list[str] = []
    for item in requested:
        if item in CATEGORIES:
            names.extend(CATEGORIES[item])
        elif item in TASKS:
            names.append(item)
        else:
            sys.exit(f"Unknown task or category: {item!r}. "
                     f"Tasks: {sorted(TASKS)}; categories: {sorted(CATEGORIES)}")
    # De-duplicate while preserving order.
    seen: set[str] = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def _print_benchmark_list() -> None:
    """Print the alias -> benchmark-name table (what --tasks accepts)."""
    print(f"{'Alias':<16} {'Benchmark':<20} Category")
    print(f"{'-' * 16} {'-' * 20} {'-' * 10}")
    for alias in TASKS:
        name, category = bench_config.BENCHMARKS.get(alias, (alias, "-"))
        print(f"{alias:<16} {name:<20} {category}")
    print(f"\nCategories: {', '.join(sorted(CATEGORIES))}")


def _find_resumable(log_dir: str, task_name: str, model: str):
    """Return the most recent incomplete log for ``task_name`` to resume, or None.

    Inspect writes each sample to the ``.eval`` log as it completes, so a run
    that died part-way (out of credits, killed, crashed) leaves the finished
    samples on disk with the log marked ``cancelled``/``error``/``started``. We
    resume that log (via ``eval_retry``) instead of starting over.

    Only the *most recent* log for the task is considered, and only if it was the
    same model — a finished (``success``) run, a different model, or no log at all
    means "run fresh". This makes re-running the same command idempotent: it
    continues where it left off rather than re-paying for completed samples.
    """
    from inspect_ai.log import list_eval_logs, read_eval_log

    try:
        logs = list_eval_logs(log_dir)  # newest first
    except Exception:  # noqa: BLE001 - missing/unreadable dir -> nothing to resume
        return None

    for info in logs:
        if info.task != task_name:
            continue
        # Most recent log for this task. Decide based on its header alone.
        header = read_eval_log(info, header_only=True)
        if header.eval.model != model:
            return None  # last run for this task used a different model
        if header.status == "success":
            return None  # already complete
        return info
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="benchmarks.run", description=__doc__)
    parser.add_argument("--tier", default=bench_config.DEFAULT_TIER,
                        choices=list(bench_config.TIERS),
                        help="Run preset (sample limit + concurrency).")
    parser.add_argument("--tasks", nargs="+", default=["all"],
                        help="Task aliases or categories to run "
                             f"({sorted(TASKS)} / {sorted(CATEGORIES)}).")
    parser.add_argument("--list", action="store_true",
                        help="List benchmark aliases and their names, then exit.")
    parser.add_argument("--log-dir", default="benchmarks/logs",
                        help="Where Inspect writes .eval logs.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any prior logs and start every task over. "
                             "By default a re-run resumes the most recent "
                             "incomplete log per task (same model + tier), "
                             "re-running only the unfinished samples.")
    args = parser.parse_args(argv)

    if args.list:
        _print_benchmark_list()
        return

    tier = bench_config.TIERS[args.tier]
    model = f"onit/{bench_config.model_label()}"
    log_dir = f"{args.log_dir}/{tier.name}"
    task_names = _resolve_task_names(args.tasks)

    # Sample-level backstop: a single task may make several bounded agent
    # requests (tool loop), so allow a multiple of the per-request timeout
    # before Inspect aborts the sample. Disabled when the timeout is disabled.
    req_timeout = bench_config.bench_timeout()
    time_limit = req_timeout * 4 if req_timeout > 0 else None

    # Resume by default: split tasks into those with a resumable incomplete log
    # and those to run fresh. --fresh forces everything to start over.
    resume_targets = []  # EvalLogInfo objects to hand to eval_retry
    fresh_names = []
    for name in task_names:
        info = None if args.fresh else _find_resumable(log_dir, name, model)
        if info is not None:
            resume_targets.append(info)
        else:
            fresh_names.append(name)

    task_labels = [f"{n} ({bench_config.display_name(n)})" for n in task_names]
    print(f"[bench] tier={tier.name} model={model} limit={tier.limit} "
          f"time_limit={time_limit}s")
    print(f"[bench] benchmarks: {', '.join(task_labels)}")
    if resume_targets:
        print(f"[bench] resuming {len(resume_targets)} incomplete log(s): "
              f"{', '.join(i.name for i in resume_targets)}")

    if resume_targets:
        inspect_eval_retry(
            resume_targets,
            max_connections=tier.max_connections,
            log_dir=log_dir,
        )

    if fresh_names:
        tasks = [TASKS[name]() for name in fresh_names]
        inspect_eval(
            tasks,
            model=model,
            limit=tier.limit,
            max_connections=tier.max_connections,
            time_limit=time_limit,
            log_dir=log_dir,
        )


if __name__ == "__main__":
    main()
