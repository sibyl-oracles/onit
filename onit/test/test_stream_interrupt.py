"""Regression tests for streams that die mid-answer.

An 84-minute turn on ollama-cloud ended mid-sentence and the run reported
"answered": the stream processors swallowed the disconnect and returned a
partial that read as a clean finish.  These tests pin the contract:

- a dead stream reports finish_reason="error" (or done_reason="error"),
  never None;
- the chat loop treats "error" like "length" — one resume ask, stitched
  onto what is already on screen;
- a run that ends with the answer still incomplete records
  STOP_TRUNCATED, not STOP_ANSWERED.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from onit.model.serving.chat import chat


def _mock_completion_with_finish(content="Hello!", tool_calls=None, finish_reason="stop"):
    """Build a mock chat completion with an explicit finish_reason."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage.prompt_tokens = 0
    return completion


def _ollama_chunk(content, **extra):
    c = MagicMock()
    c.message.content = content
    c.message.tool_calls = None
    c.message.thinking = None
    for k in ("done_reason", "prompt_eval_count", "eval_count"):
        setattr(c, k, extra.get(k))
    return c


def _responses_event(etype, **kw):
    e = MagicMock()
    e.type = etype
    for k, v in kw.items():
        setattr(e, k, v)
    return e


class TestOllamaStreamInterrupt:
    @pytest.mark.asyncio
    async def test_interrupt_reports_error_not_clean_finish(self):
        """A stream that dies mid-answer must not read as a clean finish.

        With no final chunk there is no done_reason, and None downstream
        means "no evidence" — every truncated-answer guard then falls
        through and the partial is handed back as a complete reply.
        """
        from onit.model.serving.chat import _ollama_process_streaming_response

        async def stream():
            yield _ollama_chunk("The answer is ")
            raise ConnectionResetError("peer closed connection")

        result = await _ollama_process_streaming_response(stream(), asyncio.Queue(), None, False)
        content, _, _, _, _, done_reason, _ = result
        assert content == "The answer is "
        assert done_reason == "error"

    @pytest.mark.asyncio
    async def test_clean_finish_still_reports_stop(self):
        """The interrupt path must not change the clean-finish contract."""
        from onit.model.serving.chat import _ollama_process_streaming_response

        async def stream():
            yield _ollama_chunk("Two scholarships.")
            yield _ollama_chunk("", done_reason="stop", prompt_eval_count=5441, eval_count=128)

        result = await _ollama_process_streaming_response(stream(), asyncio.Queue(), None, False)
        content, _, _, _, _, done_reason, _ = result
        assert content == "Two scholarships."
        assert done_reason == "stop"


class TestResponsesStreamInterrupt:
    @pytest.mark.asyncio
    async def test_interrupt_reports_error_not_clean_finish(self):
        """A Responses event stream that dies mid-answer must not read as a
        clean finish either — the consumer maps the dead-stream sentinel to
        finish_reason="error"."""
        from onit.model.serving.chat import _process_responses_streaming_response

        async def events():
            yield _responses_event("response.output_text.delta", delta="The answer is ")
            raise ConnectionResetError("peer closed connection")

        result = await _process_responses_streaming_response(events(), asyncio.Queue(), None, False)
        content, _, _, _, _, finish_reason = result
        assert content == "The answer is "
        assert finish_reason == "error"


class TestChatLoopResumesDeadStream:
    @pytest.mark.asyncio
    async def test_error_finish_reason_resumes_and_stitches(self):
        """finish_reason="error" (a stream that died mid-answer) is resumed
        like finish_reason=length, and the pieces are stitched together."""
        part1 = "Here is the first half of the answer "
        part2 = "and here is the rest of it."

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                _mock_completion_with_finish(content=part1, finish_reason="error"),
                _mock_completion_with_finish(content=part2, finish_reason="stop"),
            ]
        )

        with (
            patch("onit.model.serving.chat.AsyncOpenAI", return_value=mock_client),
            patch(
                "onit.model.serving.chat._resolve_model_id",
                new_callable=AsyncMock,
                return_value="test-model",
            ),
        ):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="Explain at length.",
                tool_registry=None,
                safety_queue=asyncio.Queue(),
            )

        assert result == part1 + part2
        assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_unresumable_error_records_stop_truncated(self):
        """When every resume attempt also dies, the run stops with
        STOP_TRUNCATED — a session resumed after this must not read the
        partial as a finished reply."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                _mock_completion_with_finish(content="part one ", finish_reason="error"),
                _mock_completion_with_finish(content="part two ", finish_reason="error"),
                _mock_completion_with_finish(content="part three ", finish_reason="error"),
                _mock_completion_with_finish(content="part four ", finish_reason="error"),
            ]
        )

        with (
            patch("onit.model.serving.chat.AsyncOpenAI", return_value=mock_client),
            patch(
                "onit.model.serving.chat._resolve_model_id",
                new_callable=AsyncMock,
                return_value="test-model",
            ),
        ):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="Explain at length.",
                tool_registry=None,
                safety_queue=asyncio.Queue(),
            )

        assert result is not None
        assert "part one" in result
        assert mock_client.chat.completions.create.call_count == 4
