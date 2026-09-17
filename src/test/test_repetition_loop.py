"""Reproduction: a verbatim repetition loop inside a single turn.

The model re-emits a short span ("Go. Now. Let me output. OK. ...") until the
output budget runs out.  The repetition guard in chat() exists to catch this;
this test pins the behaviour on the exact text a user reported.
"""

import asyncio
import itertools
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.chat import chat, _repetition_span, _trim_repetition  # noqa: E402

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


def _mock_completion(content, finish_reason="stop"):
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage.prompt_tokens = 0
    return completion


def test_detector_sees_the_reported_loop():
    period, copies, start = _repetition_span(REPEATED)
    assert period, "the detector did not see the reported loop"
    assert copies >= 3


@pytest.mark.asyncio
async def test_repetition_loop_terminates_and_is_trimmed():
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=itertools.repeat(_mock_completion(REPEATED)))

    with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
         patch("model.serving.chat._resolve_model_id",
               new_callable=AsyncMock, return_value="test-model"):
        result = await asyncio.wait_for(chat(
            host="http://localhost:8000/v1",
            instruction="Write the file.",
            tool_registry=None,
            safety_queue=asyncio.Queue(),
            max_ack_continuations=0,
            verify_answers=False,
        ), timeout=30)

    # The user should not be handed the loop back.
    assert result.count("Let me output") <= 2, (
        f"the loop was returned to the user ({result.count('Let me output')} copies)")
    # And the run must not spin: a handful of calls at most.
    assert mock_client.chat.completions.create.call_count <= 5
