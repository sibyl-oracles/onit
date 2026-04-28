"""Tests for src/model/serving/chat.py — _resolve_api_key, _parse_tool_call_from_content, chat."""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.chat import _resolve_api_key, _parse_tool_call_from_content, _is_planning_response, chat


# ── _resolve_api_key ────────────────────────────────────────────────────────

class TestResolveApiKey:
    def test_vllm_returns_host_key(self):
        assert _resolve_api_key("http://localhost:8000/v1", "EMPTY") == "EMPTY"

    def test_openrouter_with_host_key(self):
        assert _resolve_api_key("https://openrouter.ai/api/v1", "sk-or-abc") == "sk-or-abc"

    def test_openrouter_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-key")
        assert _resolve_api_key("https://openrouter.ai/api/v1") == "sk-env-key"

    def test_openrouter_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
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
