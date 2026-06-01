"""Coding benchmarks.

Two tiers of implementation:

* **Native** (``humaneval``, ``mbpp``) — self-contained tasks scored by running
  the agent's solution against held-out tests in an Inspect Docker sandbox. They
  use the shared :func:`code_exec_scorer` factory.
* **Wrapped** (``bigcodebench``, ``livecodebench``) — thin wrappers over the
  validated ``inspect_evals`` implementations. These use plain ``generate()``
  solvers, so they work with OnIt's final-answer provider (the agent runs its
  own tool loop internally and returns code, which the task then executes).

``swe_bench`` is wired but **deferred**: its ``inspect_evals`` task drives a
tool-calling agent inside the grading sandbox, which is incompatible with the
final-text provider. Benchmarking OnIt on SWE-bench needs the native-tools /
shared-sandbox bridge (see ``benchmarks/onit_agent.py``). It is excluded from
the default ``coding``/``all`` categories; invoke it explicitly once the bridge
lands.

All coding tasks execute model-generated code, so a Docker daemon must be
available (mirrors OnIt's own ``--container``/``--sandbox`` posture).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message
from inspect_ai.util import ExecResult, sandbox

_VERIFY_TIMEOUT = 30

_HUMANEVAL_INSTRUCTION = (
    "Complete the Python function. Return the full, self-contained function "
    "(including the signature) in a single ```python code block. Do not include "
    "explanation, tests, or example usage."
)

_MBPP_INSTRUCTION = (
    "Write a self-contained Python solution for the task, including any imports "
    "and using the exact function name implied by the tests. Return only a single "
    "```python code block — no explanation, tests, or example usage."
)


def _extract_code(answer: str) -> str:
    """Pull the Python source from the agent's free-form answer."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL)
    if blocks:
        # Prefer the longest fenced block (usually the full solution).
        return max(blocks, key=len).strip()
    return answer.strip()


def code_exec_scorer(build_program: Callable[[str, dict], str]):
    """Scorer factory: run a model solution against tests in a sandbox.

    Args:
        build_program: ``(extracted_code, sample_metadata) -> python source``.
            The returned program must exit non-zero (e.g. via an assertion or
            uncaught exception) when the solution is wrong.
    """

    @scorer(metrics=[accuracy(), stderr()])
    def _scorer():
        async def score(state: TaskState, target: Target) -> Score:
            code = _extract_code(state.output.completion)
            program = build_program(code, state.metadata)
            try:
                result: ExecResult = await sandbox().exec(
                    cmd=["python", "-c", program],
                    timeout=_VERIFY_TIMEOUT,
                )
                passed = result.success
                explanation = "all tests passed" if passed else result.stderr
            except Exception as exc:  # noqa: BLE001 - any failure => incorrect
                passed = False
                explanation = f"execution error: {exc}"
            return Score(
                value=CORRECT if passed else INCORRECT,
                answer=code,
                explanation=explanation,
            )

        return score

    return _scorer()


# --------------------------------------------------------------------------- #
# Native tasks
# --------------------------------------------------------------------------- #

def _humaneval_record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["prompt"],
        target=record["canonical_solution"],
        id=record["task_id"],
        metadata={"test": record["test"], "entry_point": record["entry_point"]},
    )


def _humaneval_program(code: str, meta: dict) -> str:
    # HumanEval's `test` defines check(candidate); call it on the entry point.
    return f"{code}\n\n{meta['test']}\n\ncheck({meta['entry_point']})\n"


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
        scorer=code_exec_scorer(_humaneval_program),
        sandbox="docker",
    )


def _mbpp_record_to_sample(record: dict) -> Sample:
    tests = record["test_list"]
    prompt = record["prompt"]
    # Surface the asserts so the model infers the exact function signature.
    body = f"{prompt}\n\nYour solution must pass these tests:\n" + "\n".join(tests)
    return Sample(
        input=body,
        target="\n".join(tests),
        id=str(record["task_id"]),
        metadata={"tests": tests, "test_imports": record.get("test_imports", [])},
    )


def _mbpp_program(code: str, meta: dict) -> str:
    imports = "\n".join(meta.get("test_imports", []))
    asserts = "\n".join(meta["tests"])
    return f"{imports}\n{code}\n{asserts}\n"


@task
def mbpp() -> Task:
    """MBPP basic Python programming (unit-test pass@1 in a sandbox)."""
    dataset = hf_dataset(
        path="google-research-datasets/mbpp",
        name="sanitized",
        split="test",
        sample_fields=_mbpp_record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=[system_message(_MBPP_INSTRUCTION), generate()],
        scorer=code_exec_scorer(_mbpp_program),
        sandbox="docker",
    )


# --------------------------------------------------------------------------- #
# Wrapped inspect_evals tasks (provider-compatible: generate() based)
# --------------------------------------------------------------------------- #

@task
def bigcodebench(subset: list[int] | None = None) -> Task:
    """BigCodeBench: realistic library/tool use in code (inspect_evals)."""
    from inspect_evals.bigcodebench import bigcodebench as _bigcodebench

    return _bigcodebench(subset=subset)


@task
def livecodebench(difficulty=None) -> Task:
    """LiveCodeBench Pro: contamination-resistant competitive coding (inspect_evals)."""
    from inspect_evals.livecodebench_pro import livecodebench_pro

    return livecodebench_pro(difficulty=difficulty)


@task
def swe_bench(dataset: str = "princeton-nlp/SWE-bench_Verified") -> Task:
    """SWE-bench via the stock ``inspect_evals`` tool-calling agent.

    This does NOT use the OnIt agent — its solver drives Inspect's own
    tool-calling agent in the grading sandbox, which the final-text OnIt provider
    cannot satisfy. To benchmark **OnIt** on SWE-bench, use the dedicated runner
    instead (it lets OnIt edit the repo in its ``data_path`` workspace and grades
    with the official harness)::

        python -m benchmarks.swe_bench_runner --dataset lite --tier smoke

    See ``benchmarks/README.md`` → "SWE-bench". This wrapper is kept only for
    comparison against the stock Inspect agent.
    """
    from inspect_evals.swe_bench import swe_bench as _swe_bench

    return _swe_bench(dataset=dataset)
