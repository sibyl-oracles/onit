"""Agentic / tool-use benchmarks (Phase 4 — scaffold).

These are the benchmarks where OnIt's tool loop matters most:

* GAIA — canonical general-assistant, multi-step tool tasks. Run in provider
  mode (OnIt drives its own tools); scored by exact match. Uses the gated
  ``gaia-benchmark/GAIA`` dataset (needs ``HF_TOKEN``).
* BFCL / tau-bench — tool-calling fidelity. Run in *native-tools* mode
  (``benchmarks.onit_agent``) so Inspect can score tool selection and argument
  accuracy directly against OnIt's real tool schemas.

Not yet registered in ``run.py``; wire in Phase 4 once dataset access and the
native-tools agent are validated.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match


def _gaia_record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["Question"],
        target=record.get("Final answer", ""),
        metadata={"level": record.get("Level"), "file_name": record.get("file_name")},
    )


@task
def gaia(subset: str = "2023_all", split: str = "validation") -> Task:
    """GAIA general-assistant tasks (exact match). Requires HF_TOKEN."""
    dataset = hf_dataset(
        path="gaia-benchmark/GAIA",
        name=subset,
        split=split,
        sample_fields=_gaia_record_to_sample,
        trust=True,
    )
    return Task(dataset=dataset, scorer=match(location="any"))
