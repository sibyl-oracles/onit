"""Reasoning / problem-solving benchmarks.

Phase 1 ships GSM8K as the smoke reasoning task. Later phases add GPQA-Diamond,
MMLU-Pro, MATH/AIME, BBH, and DROP (see the suite plan).
"""

from __future__ import annotations

import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message

_GSM8K_INSTRUCTION = (
    "Solve the math word problem. Reason step by step, then on the final line "
    "write the answer as a single number in the form: ANSWER: <number>"
)


def _gsm8k_record_to_sample(record: dict) -> Sample:
    # GSM8K gold answers end with "#### <final number>".
    answer = record["answer"]
    final = answer.split("####")[-1].strip()
    final = re.sub(r"[^\d\-\.]", "", final)
    return Sample(input=record["question"], target=final, metadata={"solution": answer})


@task
def gsm8k(shuffle: bool = True, seed: int = 42) -> Task:
    """GSM8K grade-school math word problems (numeric exact match)."""
    dataset = hf_dataset(
        path="gsm8k",
        name="main",
        split="test",
        sample_fields=_gsm8k_record_to_sample,
        shuffle=shuffle,
        seed=seed,
    )
    return Task(
        dataset=dataset,
        solver=[system_message(_GSM8K_INSTRUCTION), generate()],
        # numeric match anywhere tolerates the agent's free-form trailing text.
        scorer=match(location="any", numeric=True),
    )
