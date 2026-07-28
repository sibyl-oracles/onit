"""Tests for src/model/serving/chat.py — _resolve_api_key, _parse_tool_call_from_content, chat."""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.chat import (_resolve_api_key, _parse_tool_call_from_content,
                                _is_planning_response, _build_messages,
                                _trim_history, chat)


# ── _resolve_api_key ────────────────────────────────────────────────────────

class TestResolveApiKey:
    @pytest.fixture(autouse=True)
    def _isolate_credentials(self, monkeypatch):
        """Keep tests hermetic: ignore real env vars and the OS keychain."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        import src.setup
        monkeypatch.setattr(src.setup, "get_secret", lambda key: None)

    def test_vllm_returns_host_key(self):
        assert _resolve_api_key("http://localhost:8000/v1", "EMPTY") == "EMPTY"

    def test_vllm_explicit_host_key_wins(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "env-key")
        assert _resolve_api_key("http://localhost:8000/v1", "yaml-key") == "yaml-key"

    def test_vllm_from_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "env-key")
        assert _resolve_api_key("http://localhost:8000/v1") == "env-key"

    def test_openrouter_with_host_key(self):
        assert _resolve_api_key("https://openrouter.ai/api/v1", "sk-or-abc") == "sk-or-abc"

    def test_openrouter_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-key")
        assert _resolve_api_key("https://openrouter.ai/api/v1") == "sk-env-key"

    def test_openrouter_missing_key_raises(self):
        with pytest.raises(ValueError, match="OpenRouter requires"):
            _resolve_api_key("https://openrouter.ai/api/v1")


# ── _parse_tool_call_from_content ───────────────────────────────────────────

class TestParseToolCallFromContent:
    def _registry(self, names):
        reg = MagicMock()
        reg.tools = set(names)
        return reg

    def test_valid_tool_call(self):
        content = '{"name": "search", "arguments": {"query": "test"}}'
        result = _parse_tool_call_from_content(content, self._registry(["search"]))
        assert result["name"] == "search"
        assert result["arguments"]["query"] == "test"

    def test_with_think_tags(self):
        content = '<think>thinking...</think>{"name": "search", "arguments": {"q": "x"}}'
        result = _parse_tool_call_from_content(content, self._registry(["search"]))
        assert result is not None
        assert result["name"] == "search"

    def test_unknown_tool_returns_none(self):
        content = '{"name": "unknown_tool", "arguments": {}}'
        result = _parse_tool_call_from_content(content, self._registry(["search"]))
        assert result is None

    def test_no_json_returns_none(self):
        result = _parse_tool_call_from_content("just plain text", self._registry(["search"]))
        assert result is None

    def test_malformed_json_returns_none(self):
        result = _parse_tool_call_from_content('{"name": broken}', self._registry(["search"]))
        assert result is None

    def test_empty_content_returns_none(self):
        assert _parse_tool_call_from_content("", self._registry(["x"])) is None

    def test_none_content_returns_none(self):
        assert _parse_tool_call_from_content(None, self._registry(["x"])) is None

    def test_none_registry_returns_none(self):
        assert _parse_tool_call_from_content('{"name":"x","arguments":{}}', None) is None


# ── chat (async) ────────────────────────────────────────────────────────────

def _mock_completion(content="Hello!", tool_calls=None):
    """Build a mock chat completion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage.prompt_tokens = 0
    return completion


def _mock_tool_call(name="search", arguments='{"query": "test"}', call_id="call_123"):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    tc.id = call_id
    return tc


class TestChat:
    @pytest.mark.asyncio
    async def test_simple_response(self):
        """No tools, model returns plain text."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("The answer is 42.")
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="What is 6*7?",
                safety_queue=asyncio.Queue(),
            )

        assert result == "The answer is 42."

    @pytest.mark.asyncio
    async def test_strips_think_tags(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("<think>pondering</think>Result here")
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="test",
                safety_queue=asyncio.Queue(),
            )

        assert result == "Result here"

    @pytest.mark.asyncio
    async def test_safety_queue_aborts(self):
        sq = asyncio.Queue()
        sq.put_nowait("stop")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("should not reach")
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="test",
                safety_queue=sq,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_api_timeout_returns_none(self):
        from openai import APITimeoutError

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError(request=MagicMock())
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="test",
                safety_queue=asyncio.Queue(),
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_openai_error_returns_none(self):
        from openai import OpenAIError

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=OpenAIError("fail")
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="test",
                safety_queue=asyncio.Queue(),
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_tool_calling_loop(self):
        """Model requests a tool, gets result, then gives final answer."""
        tc = _mock_tool_call("search", '{"query": "weather"}')

        # First call: model returns tool_calls
        first_completion = _mock_completion(content=None, tool_calls=[tc])
        # Second call: model returns final answer
        second_completion = _mock_completion("It's sunny!")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[first_completion, second_completion]
        )

        # Mock tool registry
        mock_handler = AsyncMock(return_value="Weather: sunny, 25C")
        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [{"type": "function", "function": {"name": "search"}}]
        mock_registry.tools = {"search"}
        mock_registry.__getitem__ = MagicMock(return_value=mock_handler)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="What is the weather?",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
            )

        assert result == "It's sunny!"
        mock_handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_session_history_injected(self):
        """Session history entries are added as user/assistant message pairs."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("ok")
        )

        history = [{"task": "prior question", "response": "prior answer"}]

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            await chat(
                host="http://localhost:8000/v1",
                instruction="follow up",
                safety_queue=asyncio.Queue(),
                session_history=history,
            )

        # Verify the messages list included session history before the current instruction
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        # Should have: system, history user, history assistant, current user instruction
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert "prior question" in contents
        assert "prior answer" in contents
        # Current instruction must be the last user message
        assert messages[-1]["content"] == "follow up"
        assert messages[-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_custom_prompt_intro(self):
        """Custom prompt_intro overrides the default system message."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("ok")
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            await chat(
                host="http://localhost:8000/v1",
                instruction="hello",
                safety_queue=asyncio.Queue(),
                prompt_intro="I am a custom bot.",
            )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_content = messages[0]["content"]
        assert "custom bot" in system_content
        assert "OnIt" not in system_content


# ── _is_planning_response ───────────────────────────────────────────────────

class TestIsPlanningResponse:
    def test_let_me_prefix(self):
        assert _is_planning_response("Let me create the files and push them.")

    def test_i_will_prefix(self):
        assert _is_planning_response("I will now implement the solution.")

    def test_ill_prefix(self):
        assert _is_planning_response("I'll start by reading the repository.")

    def test_mid_sentence_planning(self):
        assert _is_planning_response("Analysis done. Let me now write the output.")

    def test_non_planning_returns_false(self):
        assert not _is_planning_response("Here is the result you asked for.")

    def test_final_answer_returns_false(self):
        assert not _is_planning_response("The answer is 42.")

    def test_empty_returns_false(self):
        assert not _is_planning_response("")

    def test_think_tags_stripped(self):
        content = "<think>reasoning</think>Let me proceed with the plan."
        assert _is_planning_response(content)

    def test_answer_with_trailing_plan_is_not_planning(self):
        """A long answer closing with one forward-looking sentence is still an answer."""
        answer = (
            "The MEng AI program at UP Diliman offers the DOST-SEI ERDT scholarship, "
            "which covers full tuition and a monthly stipend. " * 4
        )
        assert not _is_planning_response(f"{answer}\nLet me verify the current deadline.")

    def test_early_mid_sentence_plan_still_detected(self):
        """The phrase still counts while it is near the start — a short lead is a preamble."""
        assert _is_planning_response(
            "I checked the repository and it is out of date. Let me update it now."
        )


# ── planning-continuation integration ──────────────────────────────────────

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


class TestPlanningContinuation:
    @pytest.mark.asyncio
    async def test_planning_response_triggers_continuation(self):
        """When tools are available and the model returns planning text, a continuation
        prompt is injected and the model is called again."""
        planning_text = "Let me create the files and push them to the repo."
        final_answer = "All done!"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion_with_finish(content=planning_text),
            _mock_completion_with_finish(content=final_answer),
        ])

        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [{"type": "function", "function": {"name": "write_file"}}]
        mock_registry.tools = {"write_file"}

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="glm-5.1"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="Create a README and push it.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
            )

        assert result == final_answer
        # Must have called the API twice: once for planning, once for the continuation
        assert mock_client.chat.completions.create.call_count == 2
        second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
        # Continuation message should include explicit JSON tool-call format
        second_call_messages = second_call_kwargs["messages"]
        roles_and_contents = [(m["role"], m.get("content", "")) for m in second_call_messages
                              if isinstance(m, dict)]
        assert any("json" in c.lower() or "{" in c for _, c in roles_and_contents)
        # tool_choice should be "required" to force tool use (OpenAI/vLLM path)
        assert second_call_kwargs.get("tool_choice") == "required"
        # max_tokens should be capped to limit time waste on a stuck model
        assert second_call_kwargs.get("max_tokens") == 512

    @pytest.mark.asyncio
    async def test_planning_without_tools_no_continuation(self):
        """Without tools, planning text is returned as-is (no continuation)."""
        planning_text = "Let me think about the answer."

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion_with_finish(content=planning_text)
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="Think about it.",
                tool_registry=None,
                safety_queue=asyncio.Queue(),
            )

        assert result == planning_text
        assert mock_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_truncated_final_answer_resumes_and_stitches(self):
        """A final answer cut off by finish_reason=length is resumed and the
        pieces are stitched into one complete response."""
        part1 = "Here is the first half of the answer "
        part2 = "and here is the rest of it."

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion_with_finish(content=part1, finish_reason="length"),
            _mock_completion_with_finish(content=part2, finish_reason="stop"),
        ])

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="Explain at length.",
                tool_registry=None,
                safety_queue=asyncio.Queue(),
            )

        assert result == part1 + part2
        assert mock_client.chat.completions.create.call_count == 2
        # The resume call must carry the partial answer + a continuation prompt.
        second_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
        contents = [m.get("content", "") for m in second_messages if isinstance(m, dict)]
        assert part1 in contents
        assert any("cut off" in c.lower() for c in contents)

    @pytest.mark.asyncio
    async def test_truncated_final_answer_resume_is_bounded(self):
        """If the model keeps getting truncated, resumes are capped and whatever
        text accumulated is still returned (never less than before)."""
        mock_client = AsyncMock()
        # Always truncated — model never finishes.
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion_with_finish(content="chunk ", finish_reason="length")
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="Explain forever.",
                tool_registry=None,
                safety_queue=asyncio.Queue(),
            )

        # 1 initial + MAX_FINAL_CONTINUATIONS (3) resumes = 4 calls.
        assert mock_client.chat.completions.create.call_count == 4
        # All four chunks are stitched into the returned answer.
        assert result == "chunk " * 4

    @pytest.mark.asyncio
    async def test_planning_exhausted_returns_error(self):
        """When MAX_PLANNING_CONTINUATIONS are exhausted, a clear error is returned."""
        planning_text = "Let me write the comprehensive analysis."

        mock_client = AsyncMock()
        # Every call returns planning text — model never calls a tool
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion_with_finish(content=planning_text)
        )

        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [{"type": "function", "function": {"name": "bash"}}]
        mock_registry.tools = {"bash"}

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="glm-5.1"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="Write a review.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
            )

        # Should return an error message, not the planning text
        assert result is not None
        assert "unable to complete" in result.lower() or "tool" in result.lower()
        # Should have tried the initial call + MAX_PLANNING_CONTINUATIONS continuations
        assert mock_client.chat.completions.create.call_count == 3  # 1 initial + 2 continuations


class TestHangPrevention:
    """Regression tests for the forever-hang: no client timeout + blocked API call."""

    def test_positive_timeout_passed_through(self):
        from model.serving.chat import _build_client_timeout
        assert _build_client_timeout(120, stream=True) == 120
        assert _build_client_timeout(0, stream=False) == 0

    def test_no_timeout_still_bounds_connect_and_stream_stall(self):
        import httpx
        from model.serving.chat import _build_client_timeout, CONNECT_TIMEOUT, STREAM_STALL_TIMEOUT
        for t in (None, -1):
            cfg = _build_client_timeout(t, stream=True)
            assert isinstance(cfg, httpx.Timeout)
            assert cfg.connect == CONNECT_TIMEOUT
            assert cfg.read == STREAM_STALL_TIMEOUT

    def test_no_timeout_non_streaming_has_unbounded_read(self):
        """Non-streaming long generations must not be cut off by the stall timeout."""
        import httpx
        from model.serving.chat import _build_client_timeout, CONNECT_TIMEOUT
        cfg = _build_client_timeout(-1, stream=False)
        assert isinstance(cfg, httpx.Timeout)
        assert cfg.connect == CONNECT_TIMEOUT
        assert cfg.read is None

    @pytest.mark.asyncio
    async def test_safety_queue_aborts_blocked_api_call(self):
        """Pressing stop must interrupt an API call that never returns."""
        from model.serving.chat import _await_with_safety, _SAFETY_ABORT

        async def never_returns():
            await asyncio.sleep(3600)

        safety_queue = asyncio.Queue()
        safety_queue.put_nowait("stop")
        result = await asyncio.wait_for(
            _await_with_safety(never_returns(), safety_queue, poll=0.05), timeout=5
        )
        assert result is _SAFETY_ABORT

    @pytest.mark.asyncio
    async def test_safety_race_returns_result_when_call_completes(self):
        from model.serving.chat import _await_with_safety

        async def quick():
            return "ok"

        result = await _await_with_safety(quick(), asyncio.Queue(), poll=0.05)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_safety_race_propagates_exceptions(self):
        from model.serving.chat import _await_with_safety
        from openai import APITimeoutError
        import httpx

        async def boom():
            raise APITimeoutError(request=httpx.Request("POST", "http://localhost:8000/v1"))

        with pytest.raises(APITimeoutError):
            await _await_with_safety(boom(), asyncio.Queue(), poll=0.05)


# ── answer text written alongside tool calls ────────────────────────────────

class TestProseAlongsideToolCalls:
    """A model that answers and then calls one last tool must not lose the answer."""

    def test_streaming_history_keeps_prose(self):
        from model.serving.chat import _unify_streaming_result

        tool_calls = {0: {"id": "call_1", "name": "save", "arguments": "{}"}}
        _, _, message = _unify_streaming_result("Here is the report.", tool_calls)
        assert message["content"] == "Here is the report."
        assert message["tool_calls"][0]["function"]["name"] == "save"

    def test_streaming_history_drops_think_block(self):
        from model.serving.chat import _unify_streaming_result

        tool_calls = {0: {"id": "call_1", "name": "save", "arguments": "{}"}}
        _, _, message = _unify_streaming_result(
            "<think>plan it</think>The answer is 42.", tool_calls
        )
        assert message["content"] == "The answer is 42."

    def test_streaming_history_content_none_when_silent(self):
        from model.serving.chat import _unify_streaming_result

        tool_calls = {0: {"id": "call_1", "name": "save", "arguments": "{}"}}
        _, _, message = _unify_streaming_result("", tool_calls)
        assert message["content"] is None

    def test_bare_acknowledgment_restores_answer(self):
        from model.serving.chat import _recover_dropped_answer

        prose = "The quarterly figures are up 12%. " * 10
        assert _recover_dropped_answer("Done.", prose) == f"{prose.strip()}\n\nDone."

    def test_empty_final_restores_answer(self):
        from model.serving.chat import _recover_dropped_answer

        prose = "The quarterly figures are up 12%. " * 10
        assert _recover_dropped_answer("", prose) == prose.strip()

    def test_substantial_final_is_kept_alone(self):
        from model.serving.chat import _recover_dropped_answer

        prose = "Let me look that up. " * 10
        final = "Here is the full answer you asked for. " * 20
        assert _recover_dropped_answer(final, prose) == final

    def test_summarizing_acknowledgment_restores_answer(self):
        """A closing that describes the answer instead of being it loses to the prose."""
        from model.serving.chat import _recover_dropped_answer

        prose = "MEng AI applicants may apply for the DOST-SEI scholarship. " * 12
        # 217 characters — long enough to clear any plausible fixed ack cap.
        final = (
            "Research on UPD AI program scholarships is complete. The comprehensive "
            "answer was provided above, covering all available scholarship and "
            "financial assistance options for the UP Diliman AI Program "
            "(MEng AI and PhD AI)."
        )
        assert _recover_dropped_answer(final, prose) == f"{prose.strip()}\n\n{final}"

    def test_summarizing_acknowledgment_restores_short_answer(self):
        """The reference phrase wins even when the prose barely outruns the closing."""
        from model.serving.chat import _recover_dropped_answer

        prose = "The deadline is 15 March and the stipend is 25,000 PHP a month. " * 4
        final = "Done — the details are described above."
        assert _recover_dropped_answer(final, prose) == f"{prose.strip()}\n\n{final}"

    def test_short_prose_is_not_restored(self):
        from model.serving.chat import _recover_dropped_answer

        assert _recover_dropped_answer("Done.", "I'll delete report.pdf.") == "Done."

    def test_no_prose_is_a_no_op(self):
        from model.serving.chat import _recover_dropped_answer

        assert _recover_dropped_answer("Done.", "") == "Done."

    @pytest.mark.asyncio
    async def test_chat_returns_answer_not_acknowledgment(self):
        """End to end: prose, then a tool call, then "Done." — the prose wins."""
        answer = "The quarterly figures are up 12% year over year. " * 8
        tc = _mock_tool_call("save", '{"path": "report.md"}')
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion(content=answer, tool_calls=[tc]),
            _mock_completion("Done."),
        ])

        mock_handler = AsyncMock(return_value="saved")
        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [
            {"type": "function", "function": {"name": "save"}}
        ]
        mock_registry.tools = {"save"}
        mock_registry.__getitem__ = MagicMock(return_value=mock_handler)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="Summarize the quarter and save it.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
            )

        assert answer.strip() in result
        assert result.endswith("Done.")

    @pytest.mark.asyncio
    async def test_chat_keeps_answer_that_ends_with_a_next_step(self):
        """End to end: the answer closes with "Let me verify ..." before the last tool call."""
        answer = (
            "The MEng AI program at UP Diliman offers the DOST-SEI ERDT scholarship, "
            "which covers full tuition and a monthly stipend. " * 4
        )
        prose = f"{answer}\nLet me verify the current application deadline."
        tc = _mock_tool_call("fetch", '{"url": "https://upd.edu.ph"}')
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion(content=prose, tool_calls=[tc]),
            _mock_completion(
                "Research on UPD AI program scholarships is complete. The "
                "comprehensive answer was provided above."
            ),
        ])

        mock_handler = AsyncMock(return_value="deadline: 15 March")
        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [
            {"type": "function", "function": {"name": "fetch"}}
        ]
        mock_registry.tools = {"fetch"}
        mock_registry.__getitem__ = MagicMock(return_value=mock_handler)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="UPD AI Program scholarships",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
            )

        assert answer.strip() in result


# ── prompt assembly ─────────────────────────────────────────────────────────

class TestBuildMessages:
    """Where each half of the instruction lands. The system message is the only
    one ahead of the session history, so it is the only place the standing rules
    keep the same offset from one request to the next."""

    def test_standing_rules_go_to_the_system_message(self):
        messages = _build_messages(
            "## Task\nwhat scholarships exist", [], "I am OnIt.",
            session_history=None, memories=None,
            system_rules="## Instructions\nCite your sources.",
        )
        assert messages[0]["role"] == "system"
        assert "I am OnIt." in messages[0]["content"]
        assert "Cite your sources." in messages[0]["content"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "## Task\nwhat scholarships exist"
        assert "Cite your sources." not in messages[-1]["content"]

    def test_rules_keep_their_offset_as_history_grows(self):
        """The point of the split: the opening bytes of request N+1 match those
        of request N, so a prefix cache can skip prefilling them."""
        rules = "## Instructions\nCite your sources."
        turn_one = _build_messages("## Task\nfirst", [], "I am OnIt.",
                                   session_history=[], memories=None,
                                   system_rules=rules)
        turn_two = _build_messages("## Task\nsecond", [], "I am OnIt.",
                                   session_history=[{"task": "first",
                                                     "response": "an answer"}],
                                   memories=None, system_rules=rules)
        assert turn_one[0] == turn_two[0]

    def test_no_rules_leaves_the_system_message_alone(self):
        """A custom template yields no static half; nothing should change."""
        messages = _build_messages("do the thing", [], "I am OnIt.",
                                   session_history=None, memories=None)
        assert messages[0]["content"] == "I am OnIt."


class TestTrimHistory:
    def _history(self, n, answer_chars=5000):
        return [{"task": f"question {i}", "response": "x" * answer_chars}
                for i in range(n)]

    def test_recent_exchanges_are_untouched(self):
        history = self._history(6)
        trimmed = _trim_history(history, keep_full=3, head_chars=100)
        assert trimmed[-3:] == history[-3:]

    def test_older_answers_are_cut_to_their_opening(self):
        history = self._history(6)
        trimmed = _trim_history(history, keep_full=3, head_chars=100)
        for entry in trimmed[:3]:
            assert len(entry["response"]) < 200
            assert "trimmed" in entry["response"]

    def test_questions_are_never_cut(self):
        """A follow-up is unintelligible without the question it follows."""
        history = self._history(6)
        trimmed = _trim_history(history, keep_full=3, head_chars=10)
        assert [e["task"] for e in trimmed] == [e["task"] for e in history]

    def test_short_history_is_returned_as_is(self):
        history = self._history(2)
        assert _trim_history(history, keep_full=3) is history

    def test_a_short_answer_is_left_whole(self):
        history = self._history(6, answer_chars=20)
        trimmed = _trim_history(history, keep_full=3, head_chars=100)
        assert [e["response"] for e in trimmed] == [e["response"] for e in history]

    def test_trimming_does_not_mutate_the_caller_s_history(self):
        history = self._history(6)
        _trim_history(history, keep_full=3, head_chars=100)
        assert all(len(e["response"]) == 5000 for e in history)
