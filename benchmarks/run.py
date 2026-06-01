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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="benchmarks.run", description=__doc__)
    parser.add_argument("--tier", default=bench_config.DEFAULT_TIER,
                        choices=list(bench_config.TIERS),
                        help="Run preset (sample limit + concurrency).")
    parser.add_argument("--tasks", nargs="+", default=["all"],
                        help="Task names or categories to run "
                             f"({sorted(TASKS)} / {sorted(CATEGORIES)}).")
    parser.add_argument("--log-dir", default="benchmarks/logs",
                        help="Where Inspect writes .eval logs.")
    args = parser.parse_args(argv)

    tier = bench_config.TIERS[args.tier]
    model = f"onit/{bench_config.model_label()}"
    task_names = _resolve_task_names(args.tasks)
    tasks = [TASKS[name]() for name in task_names]

    # Sample-level backstop: a single task may make several bounded agent
    # requests (tool loop), so allow a multiple of the per-request timeout
    # before Inspect aborts the sample. Disabled when the timeout is disabled.
    req_timeout = bench_config.bench_timeout()
    time_limit = req_timeout * 4 if req_timeout > 0 else None

    print(f"[bench] tier={tier.name} model={model} "
          f"tasks={task_names} limit={tier.limit} time_limit={time_limit}s")

    inspect_eval(
        tasks,
        model=model,
        limit=tier.limit,
        max_connections=tier.max_connections,
        time_limit=time_limit,
        log_dir=f"{args.log_dir}/{tier.name}",
    )


if __name__ == "__main__":
    main()
