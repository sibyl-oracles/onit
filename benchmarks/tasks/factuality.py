"""Factuality benchmarks (Phase 3 — scaffold).

Measures short-fact accuracy and hallucination rate. Web-search/fetch tools are
left enabled so the score reflects the *agent*, not just the base model.

Wired in Phase 3: SimpleQA, TruthfulQA, FRAMES. Not yet registered in
``run.py``; add to ``benchmarks/run.py`` TASKS when the datasets/scorers are
validated.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset

from ..scorers import onit_judge


def _simpleqa_record_to_sample(record: dict) -> Sample:
    return Sample(input=record["problem"], target=record["answer"])


@task
def simpleqa(judge_model: str | None = None) -> Task:
    """SimpleQA short-fact QA, graded correct/incorrect/not-attempted.

    Note: the not-attempted bucket needs a custom metric to report the full
    SimpleQA split; the default scorer captures correct vs incorrect. Refine in
    Phase 3.
    """
    dataset = hf_dataset(
        path="basicv8vc/SimpleQA",
        split="test",
        sample_fields=_simpleqa_record_to_sample,
    )
    return Task(dataset=dataset, scorer=onit_judge(model=judge_model))
