"""Tests for src/model/serving/chat.py — _resolve_api_key, _parse_tool_call_from_content, chat."""

import asyncio
import itertools
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.chat import (_resolve_api_key, _parse_tool_call_from_content,
                                _is_planning_response, _build_messages,
                                _trim_history, _is_acknowledgment_response,
                                _is_meta_commentary_response, _build_tool_example,
                                _build_planning_continuation_prompt,
                                _is_noop_tool_call, _is_content_free_response,
                                _content_residue, _is_answering_a_nudge,
                                _ACK_CONTINUATION_PROMPT,
                                _extract_final_response, _looks_incomplete,
                                _reasoning_text,
                                _execute_tool, _compact_context, chat)


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

    @pytest.mark.parametrize("content", [
        "Now let me update `captions.py` to handle the fused results:",
        "Now let me understand the full scope of changes needed. I need to:",
        "Next, let me update the manifest builder.",
        "Then I'll wire the loader into the manifest builder.",
        "Alright, let's run the tests now.",
        "Now I need to check whether `trajectory_clauses` already exists.",
        "Good — it exists and takes an `AcousticContext`. Now let me update the smoke test.",
        "Ok, so now let me fix the smoke test.",
    ])
    def test_leadin_before_planning_phrase(self, content):
        """A connective in front of the plan does not make it an answer.

        These ended sessions early: the response fell through to the final-answer
        return, so the user saw a sentence announcing work that never ran and had
        to type "continue".
        """
        assert _is_planning_response(content)

    def test_trailing_colon_is_a_plan(self):
        """A reply that ends on a colon is introducing work it never did."""
        assert _is_planning_response("Here is the corrected implementation:")

    def test_long_answer_ending_in_colon_is_not_a_plan(self):
        """The colon test is bounded — a long answer stays an answer."""
        answer = ("The scholarship covers full tuition and a monthly stipend. " * 10)
        assert not _is_planning_response(answer + "\nThe deadline is 30 June:")

    @pytest.mark.parametrize("content", [
        "Done — 3 files updated and the tests pass.",
        "The bug was a missing await in loader.py:88; the suite is green now.",
        "Now the tests pass and the manifest builder emits all four captions.",
        "I updated the smoke test and it exercises every model.",
    ])
    def test_completed_work_is_not_a_plan(self, content):
        """Past-tense reports of finished work must reach the user."""
        assert not _is_planning_response(content)


# ── _is_acknowledgment_response ────────────────────────────────────────────

class TestIsAcknowledgmentResponse:
    def test_ready_for_next_message(self):
        assert _is_acknowledgment_response(
            "Context window management acknowledged. Ready for the next message."
        )

    def test_understood_continuing(self):
        assert _is_acknowledgment_response("Understood. Continuing based on the context summary.")

    def test_awaiting_instructions(self):
        assert _is_acknowledgment_response("Acknowledged — awaiting further instructions.")

    def test_tool_mechanics_are_not_an_answer(self):
        """Signing off on the plumbing after a long tool run says the call
        returned, not what it found."""
        assert _is_acknowledgment_response("Done. Tool called successfully.")
        assert _is_acknowledgment_response("The tool executed successfully.")
        assert _is_acknowledgment_response("All tools called successfully.")

    def test_action_task_completion_is_an_answer(self):
        """"Task completed" reports the work; only "tool" reports the plumbing."""
        assert not _is_acknowledgment_response("Task completed successfully.")

    def test_real_answer_returns_false(self):
        assert not _is_acknowledgment_response("The build fails because pytest is run from src/.")

    def test_short_legitimate_answer_returns_false(self):
        assert not _is_acknowledgment_response("Done — 3 files updated.")

    def test_empty_returns_false(self):
        assert not _is_acknowledgment_response("")

    def test_long_answer_opening_with_filler_is_kept(self):
        """Filler only counts when it is the whole reply, not a lead-in to real work."""
        answer = "Understood. Continuing based on the context summary. " + (
            "The remaining failures are in test_web_api.py, caused by the sandbox path jail. " * 4
        )
        assert not _is_acknowledgment_response(answer)

    def test_think_tags_stripped(self):
        assert _is_acknowledgment_response("<think>ok</think>Ready for the next message.")


# ── _is_meta_commentary_response ───────────────────────────────────────────

class TestIsMetaCommentaryResponse:
    def test_narrating_the_continuation_prompt(self):
        """The reply a stuck model gives to "do not write any text"."""
        assert _is_meta_commentary_response(
            "I understand you want me to call a tool without any text. I've been "
            "calling bash with simple echo commands. However, I notice you may be "
            "testing my ability to follow instructions precisely.\n\n"
            "Let me know what specific command or task you'd like me to execute "
            "with the bash tool, and I'll call it immediately without any "
            "additional text."
        )

    def test_short_meta_reply(self):
        assert _is_meta_commentary_response("I understand you want me to use a tool.")

    def test_real_answer_returns_false(self):
        assert not _is_meta_commentary_response(
            "The sandbox path jail rejects absolute paths outside data_path."
        )

    def test_empty_returns_false(self):
        assert not _is_meta_commentary_response("")

    def test_long_answer_mentioning_instructions_is_kept(self):
        """Real work that touches on the instructions still runs past the cap."""
        answer = "I understand you want me to review the tool schemas. " + (
            "Each server declares data_path so the session jail is applied. " * 12
        )
        assert not _is_meta_commentary_response(answer)

    def test_think_tags_stripped(self):
        assert _is_meta_commentary_response(
            "<think>reasoning</think>I understand you want me to call a tool."
        )


# ── _is_content_free_response ──────────────────────────────────────────────

_DATA_PATH = "/Users/rowel/sandbox/9f1fd947-3b60-400f-9c76-d593372c01ed"


class TestIsContentFreeResponse:
    """Replies pulled from real session transcripts, none of which the phrase
    lists matched."""

    def test_working_directory_read_back(self):
        assert _is_content_free_response(
            "Working directory confirmed: 9f1fd947-3b60-400f-9c76-d593372c01ed",
            _DATA_PATH,
        )

    def test_working_directory_read_back_without_data_path(self):
        """The uuid pattern carries it even when the caller passes no path."""
        assert _is_content_free_response(
            "Working directory confirmed: 9f1fd947-3b60-400f-9c76-d593372c01ed"
        )

    def test_bare_status_word(self):
        assert _is_content_free_response("Ready.")

    def test_status_word_plus_handback(self):
        assert _is_content_free_response("Ready. What would you like me to do?")

    def test_tool_call_mechanics(self):
        assert _is_content_free_response("Tool call completed successfully.")

    def test_mechanics_claim_is_stripped_whole(self):
        """The success clause and its "with no errors" tail both come out, so
        neither is left behind looking like a finding."""
        assert _content_residue(
            "The command ran successfully — `ready` was printed with no errors."
        ) == []

    def test_sign_off_after_real_work_survives(self):
        """"Done" reports the work, not the exchange — the same line
        ``_is_acknowledgment_response`` draws at "Task completed successfully."
        A run that finished may say only this."""
        assert not _is_content_free_response("All done!")
        assert not _is_content_free_response("Done.")

    def test_short_real_answer_survives(self):
        assert not _is_content_free_response("Done — 3 files updated.")

    def test_success_with_a_finding_survives(self):
        """The mechanics clause is stripped, not treated as proof of filler."""
        assert not _is_content_free_response(
            "The command ran successfully and printed 42."
        )

    def test_real_answer_survives(self):
        assert not _is_content_free_response(
            "The build fails because pytest is run from src/."
        )

    def test_action_task_completion_survives(self):
        assert not _is_content_free_response("Task completed successfully.")

    def test_answer_naming_a_path_survives(self):
        assert not _is_content_free_response(
            "The jail rejects it in /Users/rowel/sandbox/x because realpath escapes."
        )

    def test_long_reply_is_never_filler(self):
        answer = "Ready. " + ("The scholarship deadline is March 1 each year. " * 12)
        assert not _is_content_free_response(answer, _DATA_PATH)

    def test_empty_returns_false(self):
        assert not _is_content_free_response("")

    def test_think_tags_stripped(self):
        assert _is_content_free_response("<think>planning</think>Ready.")


class TestIsAnsweringANudge:
    """The structural test is scoped to replies to harness-written prompts —
    "All done!" means different things after "resume the task" and after a real
    user turn."""

    def test_ack_continuation_prompt(self):
        assert _is_answering_a_nudge(
            [{"role": "user", "content": "summarize the docs"},
             {"role": "assistant", "content": "Ready."},
             {"role": "user", "content": _ACK_CONTINUATION_PROMPT}]
        )

    def test_compacted_prompt(self):
        assert _is_answering_a_nudge(
            [{"role": "user", "content": "[CONTEXT COMPACTED]\n..."}]
        )

    def test_planning_continuation_prompt(self):
        assert _is_answering_a_nudge(
            [{"role": "user",
              "content": "Task: x\nCall a tool.\nUse this exact JSON format:\n{}"}]
        )

    def test_real_user_turn(self):
        assert not _is_answering_a_nudge(
            [{"role": "user", "content": "## Task\nsummarize the docs"},
             {"role": "assistant", "content": None},
             {"role": "tool", "content": "results"}]
        )

    def test_multimodal_user_turn_is_not_a_nudge(self):
        assert not _is_answering_a_nudge(
            [{"role": "user", "content": [{"type": "text", "text": "describe"}]}]
        )

    def test_no_user_turn(self):
        assert not _is_answering_a_nudge([{"role": "system", "content": "rules"}])


# ── blank required tool arguments ──────────────────────────────────────────

class TestBlankRequiredArgs:
    class _Registry:
        """Duck-typed like ToolRegistry, recording whether dispatch happened."""

        def __init__(self):
            self.tools = {"bash"}
            self.called_with = None

        def tool_accepts_param(self, tool_name, param_name):
            return False

        def blank_required_args(self, tool_name, arguments):
            return [n for n in ("command",)
                    if not str(arguments.get(n, "")).strip()]

        def __getitem__(self, name):
            async def handler(log_handler=None, **kwargs):
                self.called_with = dict(kwargs)
                return "ready"
            return handler

    def _run(self, args):
        registry = self._Registry()
        messages = []
        asyncio.run(_execute_tool(
            "bash", args, "call_1", registry, timeout=5, data_path=None,
            chat_ui=None, verbose=False, messages=messages,
            tool_call_history=[], max_repeated=30,
        ))
        return registry, messages

    def test_blank_command_is_not_dispatched(self):
        registry, messages = self._run({"command": ""})
        assert registry.called_with is None
        assert "no value for: command" in messages[-1]["content"]

    def test_error_tells_the_model_not_to_repeat_it(self):
        _, messages = self._run({"command": "   "})
        content = messages[-1]["content"]
        assert "do not repeat this call unchanged" in content
        assert "do not report this error as the answer" in content

    def test_real_command_still_runs(self):
        registry, messages = self._run({"command": "ls"})
        assert registry.called_with == {"command": "ls"}
        assert messages[-1]["content"] == "ready"

    def test_registry_without_the_method_still_dispatches(self):
        """The check is advisory — a duck-typed registry that lacks it works."""

        class _Bare:
            def __init__(self):
                self.tools = {"bash"}
                self.called = False

            def tool_accepts_param(self, tool_name, param_name):
                return False

            def __getitem__(self, name):
                async def handler(log_handler=None, **kwargs):
                    self.called = True
                    return "ok"
                return handler

        registry = _Bare()
        asyncio.run(_execute_tool(
            "bash", {"command": ""}, "call_1", registry, timeout=5,
            data_path=None, chat_ui=None, verbose=False, messages=[],
            tool_call_history=[], max_repeated=30,
        ))
        assert registry.called is True


# ── argument validation against the declared schema ────────────────────────

class TestSchemaValidationOnDispatch:
    class _Registry:
        """Duck-typed like ToolRegistry, recording whether dispatch happened."""

        def __init__(self, params):
            self.tools = {"search"}
            self.params = params
            self.called_with = None

        def tool_accepts_param(self, tool_name, param_name):
            return False

        def blank_required_args(self, tool_name, arguments):
            return []

        def parameters_schema(self, tool_name):
            return self.params

        def __getitem__(self, name):
            async def handler(log_handler=None, **kwargs):
                self.called_with = dict(kwargs)
                return "results"
            return handler

    _SCHEMA = {"type": "object",
               "properties": {"query": {"type": "string"},
                              "depth": {"type": "integer", "maximum": 10},
                              "mode": {"type": "string", "enum": ["fast", "deep"]}},
               "required": ["query"]}

    def _run(self, args, params=None):
        registry = self._Registry(self._SCHEMA if params is None else params)
        messages = []
        asyncio.run(_execute_tool(
            "search", args, "call_1", registry, timeout=5, data_path=None,
            chat_ui=None, verbose=False, messages=messages,
            tool_call_history=[], max_repeated=30,
        ))
        return registry, messages

    def test_wrong_type_is_refused_before_dispatch(self):
        registry, messages = self._run({"query": "cats", "depth": "three"})
        assert registry.called_with is None, "the MCP server must never see this call"
        assert "depth" in messages[-1]["content"]
        assert "integer" in messages[-1]["content"]

    def test_bad_enum_is_refused(self):
        registry, messages = self._run({"query": "cats", "mode": "medium"})
        assert registry.called_with is None
        assert "'fast'" in messages[-1]["content"]

    def test_out_of_range_is_refused(self):
        registry, messages = self._run({"query": "cats", "depth": 99})
        assert registry.called_with is None
        assert "<= 10" in messages[-1]["content"]

    def test_error_tells_the_model_not_to_repeat_it(self):
        _, messages = self._run({"query": "cats", "depth": "three"})
        content = messages[-1]["content"]
        assert "do not repeat this call unchanged" in content
        assert "do not report this error as the answer" in content

    def test_valid_call_still_dispatches(self):
        registry, messages = self._run({"query": "cats", "depth": 3, "mode": "deep"})
        assert registry.called_with == {"query": "cats", "depth": 3, "mode": "deep"}
        assert messages[-1]["content"] == "results"

    def test_numeric_string_is_coerced_rather_than_refused(self):
        """Refusing "3" costs a round trip to fix something unambiguous."""
        registry, messages = self._run({"query": "cats", "depth": "3"})
        assert registry.called_with == {"query": "cats", "depth": 3}
        assert messages[-1]["content"] == "results"

    def test_coerced_value_is_what_the_history_records(self):
        _, messages = self._run({"query": "cats", "depth": "3"})
        assert messages[-1]["parameters"]["depth"] == 3

    def test_multiple_problems_are_reported_together(self):
        _, messages = self._run({"query": 5, "mode": "medium"})
        content = messages[-1]["content"]
        assert "query" in content and "mode" in content

    def test_harness_injected_params_are_not_rejected(self):
        """session_id/data_path are injected by the harness and are not in the
        declared schema; refusing them would break every sandboxed tool."""
        registry, _ = self._run({"query": "cats", "session_id": "abc",
                                 "data_path": "/tmp/x"})
        assert registry.called_with is not None

    def test_empty_schema_dispatches_unchanged(self):
        """A server that declared no schema keeps dispatching as it always did."""
        registry, messages = self._run({"whatever": "unchecked"}, params={})
        assert registry.called_with == {"whatever": "unchecked"}
        assert messages[-1]["content"] == "results"

    def test_registry_without_the_method_still_dispatches(self):
        """Advisory, like blank_required_args — a bare duck type still works."""

        class _Bare:
            def __init__(self):
                self.tools = {"search"}
                self.called = False

            def tool_accepts_param(self, tool_name, param_name):
                return False

            def __getitem__(self, name):
                async def handler(log_handler=None, **kwargs):
                    self.called = True
                    return "ok"
                return handler

        registry = _Bare()
        asyncio.run(_execute_tool(
            "search", {"depth": "not a number"}, "call_1", registry, timeout=5,
            data_path=None, chat_ui=None, verbose=False, messages=[],
            tool_call_history=[], max_repeated=30,
        ))
        assert registry.called is True

    def test_malformed_schema_does_not_block_dispatch(self):
        registry, _ = self._run({"query": "cats"},
                                params={"properties": "not a dict"})
        assert registry.called_with is not None

    def test_blank_required_arg_still_wins(self):
        """One refusal path, one message: the blank-argument error is more
        specific, so it must not be replaced by a generic schema complaint."""

        class _Reg(self.__class__._Registry):
            def blank_required_args(self, tool_name, arguments):
                return ["query"]

        registry = _Reg(self._SCHEMA)
        messages = []
        asyncio.run(_execute_tool(
            "search", {"query": "", "depth": "three"}, "call_1", registry,
            timeout=5, data_path=None, chat_ui=None, verbose=False,
            messages=messages, tool_call_history=[], max_repeated=30,
        ))
        assert registry.called_with is None
        assert "no value for: query" in messages[-1]["content"]


# ── what actually goes on the wire ─────────────────────────────────────────

class TestApiToolPayload:
    def test_returns_is_stripped(self):
        """`returns` is ours alone: no provider reads it, and a strict one
        rejects the request for carrying it."""
        from model.serving.chat import _api_tool_payload
        items = [{"type": "function", "function": {
            "name": "search", "description": "d",
            "parameters": {"type": "object", "properties": {}},
            "returns": {"hits": {"type": "array"}}}}]
        payload = _api_tool_payload(items)
        assert payload[0]["function"] == {
            "name": "search", "description": "d",
            "parameters": {"type": "object", "properties": {}}}
        assert set(payload[0]) == {"type", "function"}

    def test_source_items_are_not_mutated(self):
        from model.serving.chat import _api_tool_payload
        items = [{"type": "function",
                  "function": {"name": "a", "returns": {"x": 1}}}]
        _api_tool_payload(items)
        assert "returns" in items[0]["function"]

    def test_unknown_future_field_does_not_reach_the_wire(self):
        """A projection, not a deletion — a field added later stays internal
        until someone decides otherwise."""
        from model.serving.chat import _api_tool_payload
        items = [{"type": "function",
                  "function": {"name": "a", "cost_hint": 3}}]
        assert "cost_hint" not in _api_tool_payload(items)[0]["function"]

    def test_missing_optional_keys_are_omitted_not_nulled(self):
        from model.serving.chat import _api_tool_payload
        payload = _api_tool_payload([{"type": "function",
                                      "function": {"name": "a"}}])
        assert payload[0]["function"] == {"name": "a"}

    def test_non_tool_records_pass_through(self):
        """Test doubles hand this whatever they please; a payload builder is
        the wrong place to start rejecting things."""
        from model.serving.chat import _api_tool_payload
        assert _api_tool_payload(["nonsense", {"no": "function"}]) == \
            ["nonsense", {"no": "function"}]

    @pytest.mark.asyncio
    async def test_chat_sends_the_projected_payload(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion(content="Hi"))
        mock_registry = MagicMock()
        mock_registry.tools = {"search"}
        mock_registry.get_tool_items.return_value = [{
            "type": "function",
            "function": {"name": "search", "description": "d",
                         "parameters": {"type": "object", "properties": {}},
                         "returns": {"hits": {}}}}]

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            await chat(host="http://localhost:8000/v1", instruction="hi",
                       tool_registry=mock_registry, safety_queue=asyncio.Queue())

        sent = mock_client.chat.completions.create.call_args.kwargs["tools"]
        assert "returns" not in sent[0]["function"]


# ── the harness's own tools ────────────────────────────────────────────────

class TestHarnessToolsInChat:
    """Phase 2: context_status and the note tools, offered by chat() itself."""

    @staticmethod
    def _registry(names=("search",)):
        registry = MagicMock()
        registry.tools = set(names)
        registry.get_tool_items.return_value = [
            {"type": "function",
             "function": {"name": n, "description": "d",
                          "parameters": {"type": "object", "properties": {}}}}
            for n in names]
        registry.tool_accepts_param.return_value = False
        registry.blank_required_args.return_value = []
        registry.parameters_schema.return_value = {}
        return registry

    async def _run(self, tmp_path, registry, **kwargs):
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=_mock_completion(content="Hi"))
        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            await chat(host="http://localhost:8000/v1", instruction="hi",
                       tool_registry=registry, safety_queue=asyncio.Queue(),
                       data_path=str(tmp_path), **kwargs)
        return [t["function"]["name"]
                for t in client.chat.completions.create.call_args.kwargs.get("tools", [])]

    @pytest.mark.asyncio
    async def test_offered_alongside_the_mcp_tools(self, tmp_path):
        sent = await self._run(tmp_path, self._registry())
        assert set(sent) == {"search", "context_status", "note_write", "note_read"}

    @pytest.mark.asyncio
    async def test_withheld_from_a_run_with_no_tools_of_its_own(self, tmp_path):
        """One turn, no compaction: the schemas would buy nothing."""
        assert await self._run(tmp_path, None) == []

    @pytest.mark.asyncio
    async def test_note_tools_need_a_data_path(self):
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=_mock_completion(content="Hi"))
        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            await chat(host="http://localhost:8000/v1", instruction="hi",
                       tool_registry=self._registry(), safety_queue=asyncio.Queue())
        sent = [t["function"]["name"]
                for t in client.chat.completions.create.call_args.kwargs["tools"]]
        assert "context_status" in sent
        assert "note_write" not in sent and "note_read" not in sent

    @pytest.mark.asyncio
    async def test_config_can_withdraw_them(self, tmp_path):
        sent = await self._run(tmp_path, self._registry(), harness_tools=False)
        assert sent == ["search"]

    @pytest.mark.asyncio
    async def test_a_call_is_answered_in_process(self, tmp_path):
        """The model asks; the loop's own numbers come back, no server involved."""
        registry = self._registry()
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion(content=None,
                             tool_calls=[_mock_tool_call("context_status", "{}", "c1")]),
            _mock_completion(content="Plenty of room left."),
        ])
        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await chat(host="http://localhost:8000/v1", instruction="hi",
                                tool_registry=registry, safety_queue=asyncio.Queue(),
                                data_path=str(tmp_path))
        assert result == "Plenty of room left."
        registry.__getitem__.assert_not_called()
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        status = json.loads(next(m["content"] for m in sent if m.get("role") == "tool"))
        assert status["turns_taken"] == 1 and status["tool_calls_made"] == 0

    @pytest.mark.asyncio
    async def test_compaction_tells_the_model_it_happened(self, tmp_path):
        """The UI already got show_context_compaction; this is the model's copy."""
        from model.serving.harness import COMPACTION_NOTICE
        filled = _mock_completion(content=None,
                                  tool_calls=[_mock_tool_call("search", "{}", "c1")])
        filled.usage.prompt_tokens = 9_500  # 95% of the window below
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            filled,                                   # turn 1: calls a tool
            _mock_completion("Summary of prior work"),  # the compaction call
            _mock_completion("Done."),                # turn 2: answers
        ])
        registry = self._registry()
        registry.__getitem__.return_value = AsyncMock(return_value="result")
        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await chat(host="http://localhost:8000/v1", instruction="hi",
                                tool_registry=registry, safety_queue=asyncio.Queue(),
                                data_path=str(tmp_path), max_context_tokens=10_000,
                                max_tokens=1024)
        assert result == "Done."
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        notices = sum(str(m.get("content", "")).count(COMPACTION_NOTICE) for m in sent)
        assert notices == 1

    @pytest.mark.asyncio
    async def test_a_note_written_by_the_model_is_on_disk(self, tmp_path):
        args = json.dumps({"key": "findings", "text": "the port is 8001"})
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion(content=None,
                             tool_calls=[_mock_tool_call("note_write", args, "c1")]),
            _mock_completion(content="Saved."),
        ])
        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            await chat(host="http://localhost:8000/v1", instruction="hi",
                       tool_registry=self._registry(), safety_queue=asyncio.Queue(),
                       data_path=str(tmp_path))
        assert (tmp_path / ".onit" / "notes" / "findings.md").read_text() == "the port is 8001"


# ── continuation prompt for stuck models ───────────────────────────────────

class TestPlanningContinuationPrompt:
    @staticmethod
    def _registry(*names):
        registry = MagicMock()
        registry.tools = set(names)
        registry.get_tool_items.return_value = [
            {"function": {"name": "write_file",
                          "parameters": {"properties": {"path": {"type": "string"},
                                                        "content": {"type": "string"}}}}},
            {"function": {"name": "bash",
                          "parameters": {"properties": {"command": {"type": "string"}}}}},
        ]
        return registry

    def test_non_shell_task_avoids_bash_example(self):
        """A bash example invites a no-op echo that satisfies the format only."""
        example = _build_tool_example(self._registry("bash", "write_file"),
                                      "Summarize the scholarship deadlines.")
        assert json.loads(example)["name"] == "write_file"

    def test_shell_task_prefers_bash_example(self):
        example = _build_tool_example(self._registry("bash", "write_file"),
                                      "Run the test suite and report failures.")
        assert json.loads(example)["name"] == "bash"

    def test_bash_still_used_when_it_is_the_only_tool(self):
        example = _build_tool_example(self._registry("bash"), "Summarize the docs.")
        assert json.loads(example)["name"] == "bash"

    def test_prompt_restates_the_task(self):
        prompt = _build_planning_continuation_prompt(
            self._registry("bash", "write_file"), 1, "Summarize the scholarship deadlines."
        )
        assert "Summarize the scholarship deadlines." in prompt
        assert "Do not write any text" not in prompt

    def test_long_task_is_trimmed(self):
        task = "Audit every MCP server for data_path declarations. " * 20
        prompt = _build_planning_continuation_prompt(
            self._registry("write_file"), 1, task
        )
        assert "..." in prompt.splitlines()[0]
        assert len(prompt.splitlines()[0]) < 400

    def test_second_continuation_escalates_to_json_only(self):
        prompt = _build_planning_continuation_prompt(
            self._registry("write_file"), 2, "Write the report."
        )
        assert "one JSON object" in prompt

    def test_missing_task_omits_the_task_line(self):
        prompt = _build_planning_continuation_prompt(self._registry("write_file"), 1)
        assert not prompt.startswith("Task:")


# ── _is_noop_tool_call ─────────────────────────────────────────────────────

class TestIsNoopToolCall:
    def test_bare_echo_is_a_noop(self):
        assert _is_noop_tool_call("bash", {"command": "echo ok"})

    def test_true_and_colon_are_noops(self):
        assert _is_noop_tool_call("bash", {"command": "true"})
        assert _is_noop_tool_call("bash", {"command": ": "})

    def test_missing_command_is_a_noop(self):
        assert _is_noop_tool_call("bash", {})

    def test_echo_with_redirect_writes_a_file(self):
        assert not _is_noop_tool_call("bash", {"command": "echo 'x = 1' > setup.py"})

    def test_echo_with_substitution_reads_the_world(self):
        assert not _is_noop_tool_call("bash", {"command": "echo $(git rev-parse HEAD)"})

    def test_echo_piped_is_real_work(self):
        assert not _is_noop_tool_call("bash", {"command": "echo hi | wc -c"})

    def test_real_command_is_not_a_noop(self):
        assert not _is_noop_tool_call("bash", {"command": "pytest src/test -q"})

    def test_echoing_tool_is_not_shell(self):
        """Only shell tools have a no-op shape; a write always lands a file."""
        assert not _is_noop_tool_call("write_file", {"path": "a.txt", "content": "echo"})

    def test_alternate_argument_keys(self):
        assert _is_noop_tool_call("run_command", {"cmd": "echo hello"})
        assert not _is_noop_tool_call("shell", {"script": "make build"})


# ── _compact_context ───────────────────────────────────────────────────────

class TestCompactContext:
    @staticmethod
    def _client(summary="Prior work: read three files, fixed one bug."):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_mock_completion(summary))
        return client

    @staticmethod
    def _messages():
        return [
            {"role": "system", "content": "You are OnIt."},
            {"role": "user", "content": "Fix the failing tests."},
            {"role": "assistant", "content": "Working on it."},
        ]

    @pytest.mark.asyncio
    async def test_compacted_history_ends_on_the_user_turn(self):
        """A trailing assistant ack would make the model reply with filler and stop."""
        out = await _compact_context(
            self._messages(), self._client(), "test-model",
            max_tokens=1024, chat_ui=None, verbose=False,
        )
        assert out[-1]["role"] == "user"
        assert not any(m["role"] == "assistant" for m in out)

    @pytest.mark.asyncio
    async def test_system_message_is_preserved(self):
        out = await _compact_context(
            self._messages(), self._client(), "test-model",
            max_tokens=1024, chat_ui=None, verbose=False,
        )
        assert out[0] == {"role": "system", "content": "You are OnIt."}

    @pytest.mark.asyncio
    async def test_instruction_is_restated_verbatim(self):
        out = await _compact_context(
            self._messages(), self._client(), "test-model",
            max_tokens=1024, chat_ui=None, verbose=False,
            instruction="Fix the failing tests.",
        )
        assert "Fix the failing tests." in out[-1]["content"]

    @pytest.mark.asyncio
    async def test_harness_notice_is_shown_to_the_model_exactly_once(self):
        """Compaction was an event only the UI heard about."""
        from model.serving.harness import COMPACTION_NOTICE
        out = await _compact_context(
            self._messages(), self._client(), "test-model",
            max_tokens=1024, chat_ui=None, verbose=False,
            harness_note=COMPACTION_NOTICE,
        )
        joined = "\n".join(m["content"] for m in out)
        assert joined.count(COMPACTION_NOTICE) == 1
        assert out[-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_the_resume_line_stays_last(self):
        """The notice must not land where "get back to work" belongs."""
        from model.serving.harness import COMPACTION_NOTICE
        out = await _compact_context(
            self._messages(), self._client(), "test-model",
            max_tokens=1024, chat_ui=None, verbose=False,
            harness_note=COMPACTION_NOTICE,
        )
        assert out[-1]["content"].rstrip().endswith("already done.]")

    @pytest.mark.asyncio
    async def test_no_notice_when_the_harness_tools_are_off(self):
        from model.serving.harness import COMPACTION_NOTICE
        out = await _compact_context(
            self._messages(), self._client(), "test-model",
            max_tokens=1024, chat_ui=None, verbose=False,
        )
        assert COMPACTION_NOTICE not in "\n".join(m["content"] for m in out)

    @pytest.mark.asyncio
    async def test_failed_summarization_returns_original_messages(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
        messages = self._messages()
        out = await _compact_context(
            messages, client, "test-model",
            max_tokens=1024, chat_ui=None, verbose=False,
        )
        assert out is messages


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


    @pytest.mark.asyncio
    async def test_noop_tool_call_does_not_refill_the_planning_budget(self):
        """A model that answers every nudge with `echo` is stuck, not working.

        Counting the echo as progress resets the budget, so the plan → nudge →
        echo cycle never reaches the exhaustion check and the loop runs forever.
        """
        planning_text = "Let me start working on the analysis."
        echo_call = _mock_completion_with_finish(
            content=None, tool_calls=[_mock_tool_call("bash", '{"command": "echo ok"}')]
        )

        mock_client = AsyncMock()
        # Alternate forever: plan, echo, plan, echo, ...  The test hangs if the
        # planning budget is refilled by the echo.
        mock_client.chat.completions.create = AsyncMock(side_effect=itertools.chain.from_iterable(
            itertools.repeat([_mock_completion_with_finish(content=planning_text), echo_call])
        ))

        mock_handler = AsyncMock(return_value="ok")
        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [{"type": "function", "function": {"name": "bash"}}]
        mock_registry.tools = {"bash"}
        mock_registry.__getitem__ = MagicMock(return_value=mock_handler)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="glm-5.1"):
            result = await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="Analyze the repository.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
            ), timeout=30)

        assert "unable to complete" in result.lower()

    @pytest.mark.asyncio
    async def test_real_tool_call_still_clears_the_planning_budget(self):
        """Progress resets the counter — a plan earlier in a working run is free."""
        real_call = _mock_completion_with_finish(
            content=None, tool_calls=[_mock_tool_call("bash", '{"command": "pytest -q"}')]
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion_with_finish(content="Let me run the tests."),  # count -> 1
            real_call,                                                      # count -> 0
            _mock_completion_with_finish(content="Let me check the failures."),  # count -> 1
            real_call,                                                      # count -> 0
            _mock_completion_with_finish(content="All 3 failures were path-jail bugs."),
        ])

        mock_handler = AsyncMock(return_value="3 failed")
        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [{"type": "function", "function": {"name": "bash"}}]
        mock_registry.tools = {"bash"}
        mock_registry.__getitem__ = MagicMock(return_value=mock_handler)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="glm-5.1"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="Run the tests and explain the failures.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
            )

        assert result == "All 3 failures were path-jail bugs."


class TestLoopPolicy:
    """The ceilings on the ways the agent loop can fail to terminate."""

    # ── _as_positive_or_disabled ────────────────────────────────────────────

    def test_positive_int_passes_through(self):
        from model.serving.chat import _as_positive_or_disabled
        assert _as_positive_or_disabled(12) == 12

    def test_yaml_string_is_coerced(self):
        """YAML hands back whatever was typed; "25" is a ceiling, not garbage."""
        from model.serving.chat import _as_positive_or_disabled
        assert _as_positive_or_disabled("25") == 25

    def test_negative_and_zero_are_the_opt_out(self):
        from model.serving.chat import _as_positive_or_disabled
        assert _as_positive_or_disabled(-1) == -1
        assert _as_positive_or_disabled(-99) == -1
        assert _as_positive_or_disabled(0) == -1

    def test_garbage_falls_back_to_default_not_to_disabled(self):
        """A typo must not silently turn a bound into an unbounded loop."""
        from model.serving.chat import _as_positive_or_disabled
        assert _as_positive_or_disabled(None, default=50) == 50
        assert _as_positive_or_disabled("fifty", default=50) == 50
        assert _as_positive_or_disabled([], default=7) == 7

    # ── the iteration cap ───────────────────────────────────────────────────

    def _alternating_registry_and_client(self, contents):
        """A model that alternates between tool calls that are never identical.

        This is the case MAX_REPEATED_TOOL_CALLS cannot catch: it keys on the
        tool name *and* byte-identical arguments, so a counter in the argument
        means the count never reaches its threshold.
        """
        calls = itertools.chain.from_iterable(
            itertools.repeat([
                _mock_completion_with_finish(
                    content=None,
                    tool_calls=[_mock_tool_call("bash", '{"command": "ls page-%d"}' % n)],
                )
                for n in range(2)
            ])
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=calls)
        mock_handler = AsyncMock(return_value="ok")
        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [
            {"type": "function", "function": {"name": "bash"}}]
        mock_registry.tools = {"bash"}
        mock_registry.__getitem__ = MagicMock(return_value=mock_handler)
        return mock_client, mock_registry

    @pytest.mark.asyncio
    async def test_never_repeating_tool_calls_still_terminate(self):
        """An explicit cap bounds the run MAX_REPEATED_TOOL_CALLS cannot.

        The arguments differ every turn, so the repeated-call check never fires
        and nothing else stops this.  The cap is off by default — it caps
        healthy long runs too — so this is the guarantee you opt into by
        setting max_chat_iterations, and without it this run does not end.
        """
        mock_client, mock_registry = self._alternating_registry_and_client(None)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            result = await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="Page through everything.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
                max_chat_iterations=5,
            ), timeout=30)

        assert result is not None
        assert mock_client.chat.completions.create.call_count == 5

    @pytest.mark.asyncio
    async def test_turn_limit_message_is_distinguishable(self):
        """The turn limit and the repeated-call bail-out are different failures.

        They returned the same sentence, so a log could not tell "stuck on one
        call" from "making progress too slowly" — and neither can whoever is
        trying to pick a better ceiling.
        """
        mock_client, mock_registry = self._alternating_registry_and_client(None)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            result = await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="Page through everything.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
                max_chat_iterations=3,
            ), timeout=30)

        assert "3 steps" in result
        # The repeated-tool-call bail-out's wording, which this must not reuse.
        assert "rephrase or provide additional details" not in result

    @pytest.mark.asyncio
    async def test_turn_limit_returns_partial_answer_when_one_exists(self):
        """Work the user already watched stream past must not be thrown away.

        But it must not come back bare either: the partial is the prose written
        just before a tool call that never ran, and with nothing appended it
        reads as a finished answer to a task that was actually abandoned.
        """
        answer = "The three regressions are in the path-jail validator."
        tool_call = _mock_completion_with_finish(
            content=answer,
            tool_calls=[_mock_tool_call("bash", '{"command": "pytest -q"}')],
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=itertools.repeat(tool_call))
        mock_handler = AsyncMock(return_value="ok")
        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [
            {"type": "function", "function": {"name": "bash"}}]
        mock_registry.tools = {"bash"}
        mock_registry.__getitem__ = MagicMock(return_value=mock_handler)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            result = await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="Run the tests.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
                max_chat_iterations=2,
            ), timeout=30)

        assert result.startswith(answer)
        assert "2 steps" in result
        assert "not a completed answer" in result

    @pytest.mark.asyncio
    async def test_turn_limit_tells_the_ui_not_just_the_log(self):
        """add_log fills a panel most runs never show.

        Reported through that alone, hitting the ceiling looked identical on
        screen to finishing: the run just stopped after a step.
        """
        ui = MagicMock()
        mock_client, mock_registry = self._alternating_registry_and_client(None)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="Page through everything.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
                chat_ui=ui,
                max_chat_iterations=3,
            ), timeout=30)

        ui.show_turn_limit.assert_called_once_with(3)

    @pytest.mark.asyncio
    async def test_default_lets_a_long_run_finish(self):
        """No turn cap unless one is asked for.

        A task needing more turns than the old ceiling of 50 used to be cut off
        mid-step, returning the prose written before a tool call that never
        ran.  Nothing here sets max_chat_iterations, so nothing bounds it.
        """
        answer = "Rewrote the caption builder across all five layers."
        turns = 80
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion_with_finish(
                content=None,
                tool_calls=[_mock_tool_call("bash", '{"command": "sed -i s/x/y/ f%d"}' % n)])
            for n in range(turns)
        ] + [_mock_completion_with_finish(content=answer)])
        mock_handler = AsyncMock(return_value="ok")
        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [
            {"type": "function", "function": {"name": "bash"}}]
        mock_registry.tools = {"bash"}
        mock_registry.__getitem__ = MagicMock(return_value=mock_handler)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            result = await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="Rewrite the caption builder.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
            ), timeout=60)

        assert result == answer
        assert mock_client.chat.completions.create.call_count == turns + 1

    @pytest.mark.asyncio
    async def test_cap_disabled_does_not_bound_a_short_run(self):
        """-1 restores the old unbounded behavior; a normal run is unaffected."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion_with_finish(
                content=None,
                tool_calls=[_mock_tool_call("bash", '{"command": "ls"}')]),
            _mock_completion_with_finish(content="Two files."),
        ])
        mock_handler = AsyncMock(return_value="a\nb")
        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [
            {"type": "function", "function": {"name": "bash"}}]
        mock_registry.tools = {"bash"}
        mock_registry.__getitem__ = MagicMock(return_value=mock_handler)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="List the files.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
                max_chat_iterations=-1,
            )

        assert result == "Two files."

    # ── the other ceilings ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_planning_continuations_are_configurable(self):
        """Raising the budget buys more continuations before the loop gives up."""
        planning_text = "Let me start working on the analysis."
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=itertools.repeat(
                _mock_completion_with_finish(content=planning_text)))

        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [
            {"type": "function", "function": {"name": "bash"}}]
        mock_registry.tools = {"bash"}

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            result = await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="Analyze the repository.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
                max_planning_continuations=4,
            ), timeout=30)

        assert "unable to complete" in result.lower()
        # 1 initial + 4 continuations, rather than the default 1 + 2.
        assert mock_client.chat.completions.create.call_count == 5

    @pytest.mark.asyncio
    async def test_final_continuations_are_configurable(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion_with_finish(
                content="chunk ", finish_reason="length"))

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="Explain forever.",
                tool_registry=None,
                safety_queue=asyncio.Queue(),
                max_final_continuations=1,
            ), timeout=30)

        # 1 initial + 1 resume, rather than the default 1 + 3.
        assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_defaults_are_unchanged_when_config_is_silent(self):
        """A caller that sets nothing gets exactly the previous behavior —
        except for the iteration cap, which was the bug."""
        planning_text = "Let me start working on the analysis."
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=itertools.repeat(
                _mock_completion_with_finish(content=planning_text)))

        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [
            {"type": "function", "function": {"name": "bash"}}]
        mock_registry.tools = {"bash"}

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="Analyze the repository.",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
            ), timeout=30)

        assert mock_client.chat.completions.create.call_count == 3  # 1 + 2


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

    @pytest.mark.asyncio
    async def test_cancelled_gather_does_not_log_unretrieved_exception(self):
        """Tearing down a parallel tool batch must not dump a stack at ERROR.

        A gather() resolves a requested cancellation by *setting* CancelledError
        once its children unwind, so nothing is left to retrieve it and asyncio
        logs the whole tool stack as "exception was never retrieved".
        """
        import gc
        from model.serving.chat import _await_with_safety

        logged: list[str] = []
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, ctx: logged.append(ctx.get("message", "")))
        try:
            async def slow_tool():
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    await asyncio.sleep(0.05)  # child unwinds after the cancel
                    raise

            outer = asyncio.ensure_future(_await_with_safety(
                asyncio.gather(*(slow_tool() for _ in range(3)), return_exceptions=True),
                asyncio.Queue(), poll=0.05,
            ))
            await asyncio.sleep(0.2)
            outer.cancel()  # request torn down, e.g. the client disconnected
            with pytest.raises(asyncio.CancelledError):
                await outer
            await asyncio.sleep(0.3)  # children unwind, gather resolves
            gc.collect()
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous)

        assert not [m for m in logged if "never retrieved" in m], logged


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


# ── explicit run state ──────────────────────────────────────────────────────

def _bash_registry():
    """A registry with one tool that always succeeds."""
    handler = AsyncMock(return_value="ok")
    registry = MagicMock()
    registry.get_tool_items.return_value = [
        {"type": "function", "function": {"name": "bash"}}]
    registry.tools = {"bash"}
    registry.__getitem__ = MagicMock(return_value=handler)
    return registry


class TestRunState:
    """What the run did, readable by whoever asked for it.

    Everything here used to live in locals inside chat() and die on return: a
    stalled run could not be asked what it had tried, and a decision twenty
    turns in could not be tested without driving the preceding twenty.
    """

    @pytest.mark.asyncio
    async def test_the_caller_reads_the_run_after_it_returns(self):
        from model.serving.state import RunState, STOP_ANSWERED

        state = RunState()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion_with_finish(
                content=None,
                tool_calls=[_mock_tool_call("bash", '{"command": "ls"}')]),
            _mock_completion_with_finish(content="Done."),
        ])

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="List the files.",
                tool_registry=_bash_registry(),
                safety_queue=asyncio.Queue(),
                run_state=state,
                verify_answers=False,
            )

        assert result == "Done."
        assert state.iteration_count == 2
        assert state.tool_call_history == [("bash", '{"command": "ls"}')]
        assert state.stop_reason == STOP_ANSWERED

    @pytest.mark.asyncio
    async def test_a_run_that_stopped_short_says_so(self):
        """The returns that report a failure are the ones worth reading, so
        they record why before they return."""
        from model.serving.state import RunState, STOP_TURN_LIMIT

        state = RunState()
        calls = itertools.chain.from_iterable(itertools.repeat([
            _mock_completion_with_finish(
                content=None,
                tool_calls=[_mock_tool_call("bash", '{"command": "ls %d"}' % n)])
            for n in range(2)]))
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=calls)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="Page through everything.",
                tool_registry=_bash_registry(),
                safety_queue=asyncio.Queue(),
                max_chat_iterations=3,
                run_state=state,
            ), timeout=30)

        assert state.stop_reason == STOP_TURN_LIMIT
        assert len(state.tool_call_history) == 3

    @pytest.mark.asyncio
    async def test_no_state_supplied_behaves_exactly_as_before(self):
        """The kwarg is optional: absent, the loop makes its own."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion_with_finish(content="Hello!"))

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="Hi.",
                tool_registry=None,
                safety_queue=asyncio.Queue(),
            )

        assert result == "Hello!"

    # ── a state constructed mid-run, without running the preceding turns ────

    @pytest.mark.asyncio
    async def test_a_spent_planning_budget_ends_the_run_on_turn_one(self):
        """Twenty turns in, budget exhausted — asserted in one API call.

        Reaching this state by driving the loop takes MAX_PLANNING_CONTINUATIONS
        planning responses first.  Handed the state directly, the very next
        planning response is the one that gives up.
        """
        from model.serving.state import RunState, STOP_PLANNING_EXHAUSTED

        state = RunState(planning_continuation_count=2)
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion_with_finish(
                content="Let me create the files and push them."))

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="Create a README.",
                tool_registry=_bash_registry(),
                safety_queue=asyncio.Queue(),
                max_planning_continuations=2,
                run_state=state,
            )

        assert "unable to complete the task" in result
        assert mock_client.chat.completions.create.call_count == 1
        assert state.stop_reason == STOP_PLANNING_EXHAUSTED

    @pytest.mark.asyncio
    async def test_a_fresh_budget_continues_where_the_spent_one_gave_up(self):
        """The control for the test above: same response, unspent budget."""
        from model.serving.state import RunState

        state = RunState(planning_continuation_count=0)
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion_with_finish(content="Let me create the files."),
            _mock_completion_with_finish(content="All done!"),
        ])

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="Create a README.",
                tool_registry=_bash_registry(),
                safety_queue=asyncio.Queue(),
                max_planning_continuations=2,
                run_state=state,
                verify_answers=False,
            )

        assert result == "All done!"
        assert mock_client.chat.completions.create.call_count == 2
        assert state.planning_continuation_count == 1

    @pytest.mark.asyncio
    async def test_a_supplied_force_tool_call_shapes_the_first_request(self):
        """State is what the next API call is built from, not just a record of
        the last one."""
        from model.serving.state import RunState

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion_with_finish(content="All done!"))

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="glm-5.1"):
            await chat(
                host="http://localhost:8000/v1",
                instruction="Create a README.",
                tool_registry=_bash_registry(),
                safety_queue=asyncio.Queue(),
                run_state=RunState(force_tool_call=True),
                verify_answers=False,
            )

        first = mock_client.chat.completions.create.call_args_list[0].kwargs
        assert first.get("tool_choice") == "required"

    @pytest.mark.asyncio
    async def test_a_supplied_token_budget_is_not_overwritten(self):
        """active_max_tokens opens at max_tokens only when nobody set it."""
        from model.serving.state import RunState

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion_with_finish(content="Hi."))

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            await chat(
                host="http://localhost:8000/v1",
                instruction="Hi.",
                tool_registry=None,
                safety_queue=asyncio.Queue(),
                max_tokens=4096,
                run_state=RunState(active_max_tokens=256),
                verify_answers=False,
            )

        first = mock_client.chat.completions.create.call_args_list[0].kwargs
        assert first.get("max_tokens") == 256


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


# ── an answer that stops mid-sentence ───────────────────────────────────────

class TestLooksIncomplete:
    """The signal that a "stop" finish hides a swallowed tail.

    Both truncated cases below are verbatim tails from real sessions against a
    vLLM host whose reasoning parser ate the rest of the answer.
    """

    @pytest.mark.parametrize("text", [
        "Three takeaways:\n\n- **The naturalness floor at",
        "6. **Add the missing test layer:** (a) parity test per graph (max abs diff",
        "Run it with `onit --host",
        "Here is the script:\n\n```python\nx = 1",
        "See the table in [the appendix",
    ])
    def test_unclosed_pairs_are_incomplete(self, text):
        assert _looks_incomplete(text) is True

    @pytest.mark.parametrize("text", [
        "All done.",
        "Live here: https://github.com/roatienza/onit",
        "| **Kokoro** | 82M | 4.302 |",
        "Install it with `pip install onit`.",
        "Here is the script:\n\n```python\nx = 1\n```",
        "- **Bold** and (parens) and [a link](https://example.com)",
        "",
    ])
    def test_finished_answers_are_left_alone(self, text):
        assert _looks_incomplete(text) is False


class TestResumeIncompleteAnswer:
    @pytest.mark.asyncio
    async def test_stop_with_a_cut_off_answer_is_resumed_and_stitched(self):
        """finish_reason="stop" is not proof the answer finished: a host whose
        reasoning parser swallows the tail reports exactly that."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion_with_finish(
                content="Takeaways:\n\n- **The naturalness floor at",
                finish_reason="stop"),
            _mock_completion_with_finish(
                content=" 4.4 UTMOS** is the new baseline.", finish_reason="stop"),
        ])

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="What is the SOTA in TTS?",
                tool_registry=None,
                safety_queue=asyncio.Queue(),
            ), timeout=30)

        assert mock_client.chat.completions.create.call_count == 2
        assert result == ("Takeaways:\n\n- **The naturalness floor at"
                          " 4.4 UTMOS** is the new baseline.")

    @pytest.mark.asyncio
    async def test_a_finished_answer_is_not_resumed(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion_with_finish(
                content="Kokoro 82M is still the reference.", finish_reason="stop"))

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="What is the SOTA in TTS?",
                tool_registry=None,
                safety_queue=asyncio.Queue(),
            ), timeout=30)

        assert mock_client.chat.completions.create.call_count == 1
        assert result == "Kokoro 82M is still the reference."

    @pytest.mark.asyncio
    async def test_resumes_are_bounded(self):
        """A model that never finishes the sentence still ends the run."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=itertools.repeat(_mock_completion_with_finish(
                content="and then (", finish_reason="stop")))

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="Explain forever.",
                tool_registry=None,
                safety_queue=asyncio.Queue(),
                max_final_continuations=2,
                max_ack_continuations=0,
            ), timeout=30)

        assert mock_client.chat.completions.create.call_count == 3  # 1 + 2


# ── thinking, whatever the host calls it ────────────────────────────────────

class TestReasoningText:
    def test_reads_every_field_a_host_may_use(self):
        for field in ("reasoning_content", "reasoning", "thinking"):
            message = SimpleNamespace(**{field: "weighing the options"})
            assert _reasoning_text(message) == "weighing the options"

    def test_reasoning_content_wins_when_a_host_sends_both(self):
        message = SimpleNamespace(reasoning_content="canonical", reasoning="alias")
        assert _reasoning_text(message) == "canonical"

    def test_no_thinking_is_the_empty_string(self):
        assert _reasoning_text(SimpleNamespace(content="hello")) == ""
        assert _reasoning_text(SimpleNamespace(reasoning=None)) == ""

    def test_vllm_reasoning_alias_reaches_the_final_answer(self):
        """vLLM 0.27 streams thinking as ``reasoning``.  When it puts the whole
        reply there and leaves content empty, that is still the answer."""
        assert _extract_final_response(
            "", _reasoning_text(SimpleNamespace(reasoning="42 is the answer.")), ""
        ) == "42 is the answer."


class TestStripThinkBlocks:
    def test_a_mid_answer_think_block_does_not_eat_the_tail(self):
        content = ("The floor is <think>which number?</think> 4.4 UTMOS, "
                   "<think>and the second one?</think> up from 4.1.")
        assert _extract_final_response(content, "", "") == (
            "The floor is  4.4 UTMOS,  up from 4.1.")

    def test_an_orphan_closing_tag_still_splits_reasoning_from_answer(self):
        assert _extract_final_response(
            "thinking out loud</think>The answer is 4.4.", "", ""
        ) == "The answer is 4.4."
