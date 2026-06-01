"""Offline tests for the native coding scorers — no Docker, no network.

Validates that the program builders compose source that genuinely passes for a
correct solution and fails for a wrong one, by executing locally with the same
semantics the sandbox uses (``python -c <program>``; non-zero exit == wrong).
"""

from __future__ import annotations

import subprocess
import sys

from benchmarks.tasks.coding import (
    _extract_code,
    _humaneval_program,
    _mbpp_program,
)


def _exits_zero(program: str) -> bool:
    proc = subprocess.run([sys.executable, "-c", program], capture_output=True)
    return proc.returncode == 0


def test_extract_code_prefers_longest_block():
    answer = "```python\nx=1\n```\nand\n```python\ndef f():\n    return 2\n```"
    assert _extract_code(answer) == "def f():\n    return 2"


def test_mbpp_program_passes_for_correct_solution():
    meta = {"tests": ["assert add(2, 3) == 5", "assert add(0, 0) == 0"],
            "test_imports": []}
    program = _mbpp_program("def add(a, b):\n    return a + b", meta)
    assert _exits_zero(program)


def test_mbpp_program_fails_for_wrong_solution():
    meta = {"tests": ["assert add(2, 3) == 5"], "test_imports": []}
    program = _mbpp_program("def add(a, b):\n    return a - b", meta)
    assert not _exits_zero(program)


def test_humaneval_program_passes_and_fails():
    meta = {
        "test": "def check(candidate):\n    assert candidate(3) == 6",
        "entry_point": "double",
    }
    assert _exits_zero(_humaneval_program("def double(x):\n    return x * 2", meta))
    assert not _exits_zero(_humaneval_program("def double(x):\n    return x + 2", meta))
