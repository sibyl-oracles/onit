"""Benchmark task definitions, grouped by capability.

Each module exposes Inspect ``@task`` functions. Phase 1 ships the two smoke
tasks (``gsm8k`` reasoning, ``humaneval`` coding); later phases fill in the
remaining benchmarks listed in the suite plan.
"""

from .coding import bigcodebench, humaneval, livecodebench, mbpp, swe_bench
from .reasoning import gsm8k

__all__ = [
    "gsm8k",
    "humaneval",
    "mbpp",
    "bigcodebench",
    "livecodebench",
    "swe_bench",
]
