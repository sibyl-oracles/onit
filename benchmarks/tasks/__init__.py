"""Benchmark task definitions, grouped by capability.

Each module exposes Inspect ``@task`` functions. Phase 1 shipped the two smoke
tasks (``gsm8k`` reasoning, ``humaneval`` coding); the registry in
``benchmarks/run.py`` decides which are runnable, and a module being present
here does not mean its datasets are reachable (GAIA is gated on HF_TOKEN).
"""

from .agentic import gaia
from .coding import bigcodebench, humaneval, mbpp
from .factuality import simpleqa
from .reasoning import gsm8k

__all__ = [
    "gsm8k",
    "humaneval",
    "mbpp",
    "bigcodebench",
    "simpleqa",
    "gaia",
]
