"""LLM-as-judge scorer for open-ended factuality benchmarks.

Thin wrapper over Inspect's ``model_graded_qa`` that defaults to the SimpleQA
grading rubric (correct / incorrect / not-attempted) used to measure
hallucination rate. By default the judge is the same model under test; pass a
stronger ``model`` for ``full`` runs to reduce judge bias (see suite plan).
"""

from __future__ import annotations

from inspect_ai.model import Model
from inspect_ai.scorer import Scorer, model_graded_qa

# SimpleQA-style rubric: reward correct answers, penalise confidently wrong ones,
# and treat hedged/abstained answers as "not attempted" rather than wrong.
_SIMPLEQA_INSTRUCTIONS = (
    "Grade the submitted answer against the reference answer.\n"
    "Reply with exactly one line: GRADE: C (correct), GRADE: I (incorrect), "
    "or GRADE: N (not attempted / refused / hedged without committing).\n"
    "An answer is correct only if it fully contains the reference answer's "
    "factual content without contradicting it."
)


def onit_judge(model: str | Model | None = None) -> Scorer:
    """Return a SimpleQA-style model-graded scorer.

    Args:
        model: Judge model. ``None`` uses the model under test.
    """
    return model_graded_qa(
        instructions=_SIMPLEQA_INSTRUCTIONS,
        grade_pattern=r"GRADE:\s*([CIN])",
        model=model,
    )
