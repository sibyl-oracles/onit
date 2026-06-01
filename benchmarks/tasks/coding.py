"""Coding benchmarks.

Phase 1 ships HumanEval as the smoke coding task, scored by executing the
agent's solution against the held-out unit tests inside a sandbox. Later phases
add MBPP(+), LiveCodeBench, and SWE-bench (see the suite plan).

Generated code is executed inside an Inspect sandbox (Docker by default,
mirroring OnIt's own ``--container``/``--sandbox`` posture). Run with a Docker
daemon available; ``run.py`` wires the sandbox for coding tasks.
"""

from __future__ import annotations

import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message
from inspect_ai.util import ExecResult, sandbox

_HUMANEVAL_INSTRUCTION = (
    "Complete the Python function. Return the full, self-contained function "
    "(including the signature) in a single ```python code block. Do not include "
    "explanation, tests, or example usage."
)

_VERIFY_TIMEOUT = 30


def _humaneval_record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["prompt"],
        target=record["canonical_solution"],
        id=record["task_id"],
        metadata={
            "prompt": record["prompt"],
            "test": record["test"],
            "entry_point": record["entry_point"],
        },
    )


def _extract_code(answer: str) -> str:
    """Pull the Python source from the agent's free-form answer."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL)
    if blocks:
        # Prefer the longest fenced block (usually the full solution).
        return max(blocks, key=len).strip()
    return answer.strip()


@scorer(metrics=[accuracy(), stderr()])
def humaneval_scorer():
    """Run the model's solution against the HumanEval unit tests in a sandbox."""

    async def score(state: TaskState, target: Target) -> Score:
        code = _extract_code(state.output.completion)
        meta = state.metadata
        # HumanEval's `test` defines check(candidate); call it on the entry point.
        program = f"{code}\n\n{meta['test']}\n\ncheck({meta['entry_point']})\n"
        try:
            result: ExecResult = await sandbox().exec(
                cmd=["python", "-c", program],
                timeout=_VERIFY_TIMEOUT,
            )
            passed = result.success
            explanation = result.stderr if not passed else "all tests passed"
        except (TimeoutError, Exception) as exc:  # noqa: BLE001 - report any failure
            passed = False
            explanation = f"execution error: {exc}"
        return Score(
            value=CORRECT if passed else INCORRECT,
            answer=code,
            explanation=explanation,
        )

    return score


@task
def humaneval() -> Task:
    """HumanEval Python function synthesis (unit-test pass@1 in a sandbox)."""
    dataset = hf_dataset(
        path="openai_humaneval",
        split="test",
        sample_fields=_humaneval_record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=[system_message(_HUMANEVAL_INSTRUCTION), generate()],
        scorer=humaneval_scorer(),
        sandbox="docker",
    )
