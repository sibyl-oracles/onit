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


# The literal think tags, spelled through chr() so this file never carries
# the markup it is testing for.
OPEN = chr(60) + "think" + chr(62)
CLOSE = chr(60) + "/think" + chr(62)


class TestQuotedThinkOpener:
    """A model quoting a chat template writes a literal think opener; a host
    that treats it as a real one swallows the rest of the turn.

    glm-5.3-flash:cloud explaining the Qwen3.5 template lost its tail this way
    three turns in a row on 13 Nov: done_reason "stop", usage matching, and a
    partial that ended mid-quote with no evidence anything went wrong.
    """

    def _turn(self, opener=True):
        body = (
            "Now I understand the packer. The prefix property check:\n\n"
            "```python\npre = tok.apply_chat_template(msgs[:cut],\n"
            "                              add_generation_prompt=True,\n"
            "                              enable_thinking=False)\n```\n\n"
            "The prefix render includes the empty think block: "
            '"<|im_start|>assistant\\n'
        )
        return body + (OPEN if opener else "")

    @pytest.mark.asyncio
    async def test_a_quoted_opener_that_ate_the_tail_is_resumed(self):
        """done_reason "stop" with an unpaired opener is a cut, not a finish."""
        part2 = " and the template continues here."
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                _mock_completion_with_finish(content=self._turn(), finish_reason="stop"),
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
            result = await asyncio.wait_for(
                chat(
                    host="http://localhost:8000/v1",
                    instruction="Explain the template.",
                    tool_registry=None,
                    safety_queue=asyncio.Queue(),
                ),
                timeout=30,
            )

        assert mock_client.chat.completions.create.call_count == 2
        assert OPEN not in result
        assert result.endswith(part2)

    @pytest.mark.asyncio
    async def test_a_paired_quote_is_left_alone(self):
        """The model quoting a complete empty think block is not a cut."""
        body = (
            "The empty think block: "
            + chr(34)
            + "<|im_start|>assistant"
            + chr(92)
            + "n"
            + OPEN
            + "\\n\\n"
            + CLOSE
            + "\\n\\n"
            + chr(34)
            + " and that is the whole prompt."
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion_with_finish(content=body, finish_reason="stop")
        )

        with (
            patch("onit.model.serving.chat.AsyncOpenAI", return_value=mock_client),
            patch(
                "onit.model.serving.chat._resolve_model_id",
                new_callable=AsyncMock,
                return_value="test-model",
            ),
        ):
            result = await asyncio.wait_for(
                chat(
                    host="http://localhost:8000/v1",
                    instruction="Explain the template.",
                    tool_registry=None,
                    safety_queue=asyncio.Queue(),
                ),
                timeout=30,
            )

        assert mock_client.chat.completions.create.call_count == 1
        assert OPEN not in result
        assert CLOSE not in result
        assert result.endswith("that is the whole prompt.")

    @pytest.mark.asyncio
    async def test_an_unresumable_quote_records_stop_truncated(self):
        """Every resume repeats the same quote: the run must stop with
        STOP_TRUNCATED, not hand back a silent partial."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                _mock_completion_with_finish(content=self._turn(), finish_reason="stop")
                for _ in range(6)
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
            result = await asyncio.wait_for(
                chat(
                    host="http://localhost:8000/v1",
                    instruction="Explain the template.",
                    tool_registry=None,
                    safety_queue=asyncio.Queue(),
                ),
                timeout=30,
            )

        assert result is not None
        assert OPEN not in result
        assert mock_client.chat.completions.create.call_count == 3


class TestUnclosedThinkHelper:
    def test_an_unpaired_opener_is_evidence(self):
        from onit.model.serving.chat import _ends_with_unclosed_think

        assert _ends_with_unclosed_think("prose " + OPEN) is True

    def test_a_paired_block_is_not(self):
        from onit.model.serving.chat import _ends_with_unclosed_think

        assert _ends_with_unclosed_think("prose " + OPEN + "x" + CLOSE) is False

    def test_empty_text_is_not(self):
        from onit.model.serving.chat import _ends_with_unclosed_think

        assert _ends_with_unclosed_think("") is False

    def test_an_opener_after_a_paired_block_is(self):
        from onit.model.serving.chat import _ends_with_unclosed_think

        text = "a " + OPEN + "x" + CLOSE + " b " + OPEN
        assert _ends_with_unclosed_think(text) is True
