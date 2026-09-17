"""Reproduction of the reported repetition loop, on the streaming path."""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.chat import chat  # noqa: E402

REPEATED = (
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\nI'll write.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\n"
    "Go.\n\nNow.\n\nLet me write.\n\nOK.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\nI'll write.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\n"
    "Go.\n\nNow.\n\nLet me write.\n\nOK.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\nI'll write.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\n"
    "Go.\n\nNow.\n\nLet me write.\n\nOK.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\nI'll write.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\n"
    "Go.\n\nNow.\n\nLet me write.\n\nOK.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\nI'll write.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\n"
    "Go.\n\nNow.\n\nLet me write.\n\nOK.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\nI'll write.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\n"
    "Go.\n\nNow.\n\nLet me write.\n\nOK.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n\nI'll write.\n\n"
    "Go.\n\nNow.\n\nLet me output.\n\nOK.\n"
)


class _Delta(SimpleNamespace):
    pass


def _chunk(content=None, finish_reason=None, usage=None):
    delta = _Delta(content=content, tool_calls=None, reasoning_content=None,
                   reasoning=None, thinking=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


class _Stream:
    """Async iterator of chunks, repeating the same looped turn forever."""

    def __init__(self, text, finish_reason="length"):
        self._text = text
        self._finish = finish_reason

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        # stream the loop in small pieces, as a real host would
        for i in range(0, len(self._text), 7):
            yield _chunk(content=self._text[i:i + 7])
        yield _chunk(finish_reason=self._finish)


@pytest.mark.asyncio
async def test_streaming_repetition_loop_is_stopped():
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=lambda **kw: _Stream(REPEATED))

    with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
         patch("model.serving.chat._resolve_model_id",
               new_callable=AsyncMock, return_value="test-model"):
        result = await asyncio.wait_for(chat(
            host="http://localhost:8000/v1",
            instruction="Write the file.",
            tool_registry=None,
            safety_queue=asyncio.Queue(),
            stream=True,
            max_ack_continuations=0,
            verify_answers=False,
        ), timeout=30)

    print("CALLS:", mock_client.chat.completions.create.call_count)
    print("COPIES:", result.count("Let me output"))
    print("RESULT:", repr(result[:200]))
    assert result.count("Let me output") <= 2, (
        f"loop returned to user: {result.count('Let me output')} copies")
    assert mock_client.chat.completions.create.call_count <= 5
