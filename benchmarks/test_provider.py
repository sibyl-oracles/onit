"""Unit tests for the OnIt Inspect provider — no model, network, or Docker.

A stub agent stands in for the real ``OnIt`` so we can verify the provider's
message flattening, the generate() path, and the full Inspect eval+scorer wiring
deterministically.
"""

from __future__ import annotations

import pytest
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import match

from benchmarks import onit_provider
from benchmarks.onit_provider import _messages_to_task
from benchmarks.tasks.coding import _extract_code


class _StubAgent:
    """Fake OnIt whose process_task returns a canned answer."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[str] = []

    async def process_task(self, task: str, **kwargs) -> str:
        self.calls.append(task)
        return self.answer


@pytest.fixture(autouse=True)
def _reset_agent():
    """Ensure each test controls the shared agent singleton."""
    onit_provider._agent = None
    yield
    onit_provider._agent = None


def test_messages_to_task_flattens_system_and_user():
    msgs = [ChatMessageSystem(content="Be precise."), ChatMessageUser(content="2+2?")]
    assert _messages_to_task(msgs) == "Be precise.\n\n2+2?"


def test_extract_code_prefers_fenced_block():
    answer = "Here you go:\n```python\ndef f():\n    return 1\n```\nDone."
    assert _extract_code(answer) == "def f():\n    return 1"


def test_extract_code_falls_back_to_raw():
    assert _extract_code("def f():\n    return 1") == "def f():\n    return 1"


@pytest.mark.asyncio
async def test_generate_returns_agent_answer():
    onit_provider._agent = _StubAgent("ANSWER: 42")
    model = get_model("onit/stub")
    out = await model.generate("ignored")
    assert "42" in out.completion


def test_eval_pipeline_scores_correct_answer():
    """End-to-end: provider + numeric match scorer + Inspect runner."""
    onit_provider._agent = _StubAgent("Working... ANSWER: 42")
    task = Task(
        dataset=MemoryDataset([Sample(input="What is 6 times 7?", target="42")]),
        scorer=match(location="any", numeric=True),
    )
    logs = inspect_eval(task, model="onit/stub", display="none")
    assert logs[0].status == "success"
    accuracy = logs[0].results.scores[0].metrics["accuracy"].value
    assert accuracy == 1.0
