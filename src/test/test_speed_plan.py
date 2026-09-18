"""Tests for the 2026-09-18 speed/token plan (token_saving_proposal.md).

Covers the acceptance checks that are testable without a live endpoint:

- S2: the shipped default config no longer overrides the code's history_turns.
- S7: load_session_history reads a long session file correctly (tail-read).
- S6: the verify fast pass is skipped when the run gathered no tool evidence.
- S3: cached_tokens reaches the metrics sink; the request prefix is
  byte-stable across turns (the prefix-cache contract, Tier 5.2).
- A1: the first-read timeout scales with the prompt estimate.
- A2: a timeout retries at most once, with a trimmed prompt.
- B1/B3: a repeated tool call steers instead of ending the turn; only a
  proven loop bails.
- S4: incremental compaction summarizes only the delta and keeps the cursor.
- 5.1: per-tool response budgets.
"""

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.chat import (
    MAX_TOOL_RESPONSE,
    TOOL_RESPONSE_BUDGETS,
    _REPEAT_STEER_AFTER,
    _REPEAT_STREAK_BAIL,
    _build_client_timeout,
    _build_messages,
    _cached_tokens_of,
    _compact_context,
    _decay_old_tool_results,
    _truncate_tool_response,
    chat,
)
from model.serving.results import ResultStore
from model.serving.state import RunState
from model.serving.verify import has_tool_evidence, needs_verification


def _mock_completion(content="Hello!", tool_calls=None, prompt_tokens=0,
                     cached_tokens=None):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage.prompt_tokens = prompt_tokens
    if cached_tokens is not None:
        completion.usage.prompt_tokens_details = SimpleNamespace(
            cached_tokens=cached_tokens)
    else:
        # No prompt_tokens_details attribute at all: the shape Ollama and the
        # Responses path produce, which must read as "unknown", not 0.
        del completion.usage.prompt_tokens_details
    return completion


def _mock_tool_call(name="search", arguments='{"query": "test"}',
                    call_id="call_123"):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    tc.id = call_id
    return tc


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


# ── S2: the shipped config matches the code default ─────────────────────────

class TestHistoryTurnsDefault:
    def test_shipped_config_does_not_override_the_code_default(self):
        import yaml
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "configs", "default.yaml")) as f:
            cfg = yaml.safe_load(f)
        # 6 is the code default (onit.py Field); the yaml must not raise it.
        assert cfg.get("history_turns", 6) == 6


# ── S7: tail-read the session file ──────────────────────────────────────────

class TestTailReadSessionHistory:
    def _agent(self, tmp_path, history_turns=6):
        from src.onit import OnIt
        from type.tools import ToolRegistry
        cfg = {
            "serving": {"host": "http://localhost:8000/v1", "think": False,
                        "max_tokens": 1024},
            "mcp": {"servers": []},
            "session_path": str(tmp_path / "sessions"),
            "theme": "white", "verbose": False,
        }
        with patch("src.onit.discover_tools", return_value=ToolRegistry()):
            agent = OnIt(config=cfg)
        agent.history_turns = history_turns
        agent.session_path = str(tmp_path / "session.jsonl")
        return agent

    def test_reads_recent_pairs_from_a_long_file(self, tmp_path):
        agent = self._agent(tmp_path)
        with open(agent.session_path, "w") as f:
            for i in range(500):
                f.write(json.dumps({"task": f"task {i}",
                                    "response": f"resp {i}"}) + "\n")
        history = agent.load_session_history()
        assert len(history) == 6
        assert history[0]["task"] == "task 494"
        assert history[-1]["task"] == "task 499"

    def test_handles_long_lines_crossing_chunk_boundaries(self, tmp_path):
        agent = self._agent(tmp_path, history_turns=2)
        big = "x" * 200_000  # one record far larger than the 64KB chunk
        with open(agent.session_path, "w") as f:
            f.write(json.dumps({"task": "big", "response": big}) + "\n")
            f.write(json.dumps({"task": "last", "response": "ok"}) + "\n")
        history = agent.load_session_history()
        assert [h["task"] for h in history] == ["big", "last"]
        assert history[0]["response"] == big

    def test_missing_file_returns_empty(self, tmp_path):
        agent = self._agent(tmp_path)
        assert agent.load_session_history() == []

    def test_malformed_lines_are_skipped(self, tmp_path):
        agent = self._agent(tmp_path)
        with open(agent.session_path, "w") as f:
            f.write("not json\n")
            f.write(json.dumps({"task": "a", "response": "b"}) + "\n")
        history = agent.load_session_history()
        assert [h["task"] for h in history] == ["a"]


# ── S6: skip the verify fast pass without evidence ─────────────────────────

class TestHasToolEvidence:
    def test_tool_message_is_evidence(self):
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "looking"},
                {"role": "tool", "name": "search", "content": "42%"}]
        assert has_tool_evidence(msgs) is True

    def test_no_tools_is_no_evidence(self):
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "long answer " * 20}]
        assert has_tool_evidence(msgs) is False

    def test_empty_is_no_evidence(self):
        assert has_tool_evidence([]) is False

    def test_fast_pass_skipped_without_tools(self):
        """needs_verification alone would fire; the gate must hold it back."""
        answer = ("The Eiffel Tower is 330 metres tall and was completed in "
                  "1889. It remains the tallest structure in Paris today, "
                  "and it is visited by millions every year.")
        assert needs_verification(answer) is True
        assert has_tool_evidence([{"role": "user", "content": "hi"},
                                  {"role": "assistant", "content": answer}]) is False


# ── S3: cached_tokens reach the sink; request prefix is byte-stable ─────────

class TestCachedTokens:
    def test_end_api_records_cached_tokens(self):
        from model.serving.chat import TurnMetrics
        sink = {}
        m = TurnMetrics(sink)
        m.start_api()
        m.end_api(prompt_tokens=1000, completion_tokens=10,
                  cached_tokens=800)
        assert sink["cached_tokens"] == 800
        assert sink["prompt_tokens_sum"] == 1000
        assert m.turns[0]["cached_tokens"] == 800

    def test_end_api_without_details_is_none_not_zero(self):
        from model.serving.chat import TurnMetrics
        sink = {}
        m = TurnMetrics(sink)
        m.start_api()
        m.end_api(prompt_tokens=1000, completion_tokens=5)
        assert m.turns[0]["cached_tokens"] is None
        assert "cached_tokens" not in sink

    def test_summarize_metrics_reports_hit_rate(self):
        from model.serving.chat import TurnMetrics, summarize_metrics
        sink = {}
        m = TurnMetrics(sink)
        m.start_api()
        m.end_api(prompt_tokens=1000, completion_tokens=5, cached_tokens=900)
        text = summarize_metrics(sink)
        assert "cache 90% hit" in text

    def test_cached_tokens_of_reads_nested_details(self):
        from model.serving.chat import _cached_tokens_of
        usage = SimpleNamespace(prompt_tokens=10,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=7))
        assert _cached_tokens_of(usage) == 7
        assert _cached_tokens_of(SimpleNamespace(prompt_tokens=10)) is None
        assert _cached_tokens_of(None) is None

    @pytest.mark.asyncio
    async def test_nonstream_path_passes_cached_tokens_to_metrics(self):
        """The non-streaming end_api call must carry the nested field."""
        tc = _mock_tool_call("search", '{"query": "x"}', "c1")
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion(content=None, tool_calls=[tc],
                             prompt_tokens=500, cached_tokens=400),
            _mock_completion("Done.", prompt_tokens=600, cached_tokens=500),
        ])
        registry = _registry()
        registry.__getitem__.return_value = AsyncMock(return_value="result")
        sink_holder = {}

        class _UI:
            def set_metrics(self, sink):
                sink_holder["sink"] = sink

            def add_log(self, message, level="info", **kwargs):
                pass

            def __getattr__(self, name):
                # Any other UI hook the loop calls is a no-op; this test only
                # reads the metrics sink.
                return lambda *a, **k: None

        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await chat(host="http://localhost:8000/v1", instruction="hi",
                                tool_registry=registry, safety_queue=asyncio.Queue(),
                                chat_ui=_UI())
        assert result == "Done."
        sink = sink_holder["sink"]
        assert sink["cached_tokens"] == 900  # 400 + 500
        assert sink["prompt_tokens_sum"] == 1100


class TestRequestPrefixByteStable:
    """Tier 5.2: the cacheable prefix must not move between turns."""

    def test_system_message_identical_across_two_builds(self):
        m1 = _build_messages("task one", [], "intro", [], None,
                             system_rules="standing rules")
        m2 = _build_messages("task two", [], "intro", [], None,
                             system_rules="standing rules")
        # The system message is the prefix; the task lives after it.
        assert m1[0]["content"] == m2[0]["content"]
        assert json.dumps(m1[0], sort_keys=True) == json.dumps(m2[0], sort_keys=True)

    def test_system_message_leads_and_is_stable_with_images(self):
        m1 = _build_messages("t", [], "intro", [], None, system_rules="rules")
        m2 = _build_messages("t", ["b64"], "intro", [], None, system_rules="rules")
        assert m1[0]["role"] == "system" and m2[0]["role"] == "system"
        # Both start with the same intro bytes; the vision variant appends
        # its fixed block, so the common prefix is still stable.
        assert m2[0]["content"].startswith(m1[0]["content"].split("\n")[0])


# ── A1: first-read timeout scales with the prompt ───────────────────────────

class TestFirstReadTimeout:
    def test_flat_stall_budget_for_small_prompts(self):
        t = _build_client_timeout(None, stream=True, prompt_tokens=1000)
        assert t.read == 300.0

    def test_scaled_budget_for_large_prompts(self):
        # 200k tokens / 300 ≈ 667s, clamped to the 600s ceiling.
        t = _build_client_timeout(None, stream=True, prompt_tokens=200_000)
        assert t.read == 600.0
        t = _build_client_timeout(None, stream=True, prompt_tokens=120_000)
        assert t.read == 400.0

    def test_non_streaming_keeps_no_read_limit(self):
        t = _build_client_timeout(None, stream=False, prompt_tokens=200_000)
        assert t.read is None

    def test_explicit_timeout_wins(self):
        t = _build_client_timeout(120, stream=True, prompt_tokens=200_000)
        assert t == 120


# ── A2: timeout retries are capped and shrink the prompt ────────────────────

class TestTimeoutRetryPolicy:
    @pytest.mark.asyncio
    async def test_timeout_retries_once_with_trim_then_fails(self):
        from openai import APITimeoutError
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError(request=MagicMock()))
        registry = _registry()
        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await chat(host="http://localhost:8000/v1", instruction="hi",
                                tool_registry=registry,
                                safety_queue=asyncio.Queue(),
                                max_api_retries=3)
        assert result is None
        # 1 initial attempt + 1 timeout retry = 2 calls, not MAX_API_RETRIES.
        assert client.chat.completions.create.await_count == 2


# ── B1/B3: repeated calls steer before they bail ────────────────────────────

class TestRepeatedCallSteering:
    @pytest.mark.asyncio
    async def test_repeated_call_steers_and_the_loop_continues(self, tmp_path):
        """The same call 4 times in a row: the turn must NOT end with an
        apology — the model gets its steering notice and answers."""
        tc = _mock_tool_call("search", '{"query": "same"}', "c1")
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            *[_mock_completion(content=None, tool_calls=[tc]) for _ in range(_REPEAT_STEER_AFTER)],
            _mock_completion("Finally done."),
        ])
        registry = _registry()
        registry.__getitem__.return_value = AsyncMock(return_value="same result")
        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await chat(host="http://localhost:8000/v1", instruction="hi",
                                tool_registry=registry, safety_queue=asyncio.Queue(),
                                data_path=str(tmp_path))
        assert result == "Finally done."
        # The steering notice reached the conversation.
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert any("repeated call" in str(m.get("content", "")) for m in sent)

    @pytest.mark.asyncio
    async def test_proven_loop_still_bails_with_an_actionable_message(self, tmp_path):
        tc = _mock_tool_call("search", '{"query": "same"}', "c1")
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=_mock_completion(content=None, tool_calls=[tc]))
        registry = _registry()
        registry.__getitem__.return_value = AsyncMock(return_value="same")
        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await chat(host="http://localhost:8000/v1", instruction="hi",
                                tool_registry=registry, safety_queue=asyncio.Queue(),
                                data_path=str(tmp_path))
        assert result is not None
        assert "rephrase" not in result.lower()
        assert "change approach" in result


# ── S4: incremental compaction ──────────────────────────────────────────────

class TestIncrementalCompaction:
    @pytest.mark.asyncio
    async def test_delta_only_summarization(self):
        """With a prior summary and cursor, the summarizer sees only new
        messages, and the marker carries the advanced cursor."""
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("merged summary"))
        old = [{"role": "user", "content": "old task"},
               {"role": "tool", "name": "t", "content": "old result"},
               {"role": "assistant", "content": "old answer"}]
        new = [{"role": "tool", "name": "t", "content": "new result"}]
        msgs = [{"role": "system", "content": "sys"}] + old + new
        out = await _compact_context(
            msgs, client, "m", 1024, None, False,
            prior_summary="prior summary text", summarized_upto=len(old))
        # The summarizer prompt contains the prior summary and the new message
        # but not the already-summarized old transcript.
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "merged summary" not in prompt  # that's the reply, not the prompt
        assert "Existing summary" in prompt
        assert "old result" not in prompt
        assert "new result" in prompt
        marker = out[-1].pop("_compaction")
        assert marker["summary"] == "merged summary"
        assert marker["summarized_upto"] == len(old) + len(new)

    @pytest.mark.asyncio
    async def test_stale_cursor_falls_back_to_full(self):
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("full summary"))
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "task"}]
        out = await _compact_context(
            msgs, client, "m", 1024, None, False,
            prior_summary="prior", summarized_upto=99)  # out of range
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "Existing summary" not in prompt
        assert "task" in prompt

    @pytest.mark.asyncio
    async def test_summarizer_failure_returns_original(self):
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("down"))
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "task"}]
        out = await _compact_context(msgs, client, "m", 1024, None, False)
        assert out == msgs

    @pytest.mark.asyncio
    async def test_full_run_compacts_once_and_strips_marker(self, tmp_path):
        """End-to-end: a compaction fires mid-run, the compacted prompt carries
        the summary, and no _compaction key ever reaches the API."""
        filled = _mock_completion(content=None,
                                  tool_calls=[_mock_tool_call("search", "{}", "c1")])
        filled.usage.prompt_tokens = 9_500
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            filled,                                    # turn 1: tool call at 95%
            _mock_completion("summary one"),           # compaction 1
            _mock_completion("Done."),                 # turn 2: final answer
        ])
        registry = _registry()
        registry.__getitem__.return_value = AsyncMock(return_value="result")
        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await chat(host="http://localhost:8000/v1", instruction="hi",
                                tool_registry=registry, safety_queue=asyncio.Queue(),
                                data_path=str(tmp_path), max_context_tokens=10_000,
                                max_tokens=1024)
        assert result == "Done."
        # Every payload the API saw was marker-free.
        for call in client.chat.completions.create.await_args_list:
            for m in call.kwargs["messages"]:
                assert "_compaction" not in m
        # The compacted prompt carries the summary.
        sent = client.chat.completions.create.await_args_list[-1].kwargs["messages"]
        assert any("summary one" in str(m.get("content", "")) for m in sent)

    @pytest.mark.asyncio
    async def test_second_compaction_merges_into_the_first_summary(self, tmp_path):
        """Two compactions in one run: the second summarizer call sees the
        first summary as 'Existing summary' and only the delta as new."""
        # Turn 1 at 95% → compact; turn 2 calls a tool; turn 3 also at 95%
        # (usage reported per call) → compact again; then final.
        t1 = _mock_completion(content=None,
                              tool_calls=[_mock_tool_call("search", "{}", "c1")])
        t1.usage.prompt_tokens = 9_500
        t2 = _mock_completion(content=None,
                              tool_calls=[_mock_tool_call("search", "{}", "c2")])
        t2.usage.prompt_tokens = 9_600
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            t1, _mock_completion("summary one"),
            t2, _mock_completion("summary two"),
            _mock_completion("Done."),
        ])
        registry = _registry()
        registry.__getitem__.return_value = AsyncMock(return_value="result")
        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await chat(host="http://localhost:8000/v1", instruction="hi",
                                tool_registry=registry, safety_queue=asyncio.Queue(),
                                data_path=str(tmp_path), max_context_tokens=10_000,
                                max_tokens=1024)
        assert result == "Done."
        prompts = [c.kwargs["messages"][0]["content"]
                   for c in client.chat.completions.create.await_args_list]
        # Compaction 1: full summarization.
        assert any("Summarize the following agent conversation" in p for p in prompts)
        # Compaction 2: incremental — prior summary present, old transcript gone.
        assert any("Existing summary" in p and "summary one" in p for p in prompts)


# ── 5.1: per-tool budgets ───────────────────────────────────────────────────

class TestPerToolBudgets:
    def test_unlisted_tool_keeps_global_budget(self):
        big = "x" * (MAX_TOOL_RESPONSE * 3)
        out = _truncate_tool_response(big, "bash")
        assert len(out) < MAX_TOOL_RESPONSE + 100
        assert "truncated" in out

    def test_listed_tool_uses_its_own_budget(self):
        big = "x" * 5000
        out = _truncate_tool_response(big, "get_weather")
        assert len(out) < 2500  # budget 2000 → head+tail around it

    def test_small_results_pass_through(self):
        assert _truncate_tool_response("short", "get_weather") == "short"

    def test_weather_budget_is_the_tightest(self):
        assert min(TOOL_RESPONSE_BUDGETS.values()) == TOOL_RESPONSE_BUDGETS["get_weather"]


# ── decay returns a count (A2 helper) ───────────────────────────────────────

class TestDecayReturnsCount:
    def test_counts_trims_and_is_idempotent(self):
        msgs = [{"role": "system", "content": "s"}]
        for i in range(5):
            msgs.append({"role": "tool", "name": "t",
                         "content": "y" * 9000 + f" #{i}"})
        n1 = _decay_old_tool_results(msgs, keep_full=2)
        assert n1 == 3
        n2 = _decay_old_tool_results(msgs, keep_full=2)
        assert n2 == 0  # already decayed — second pass is a no-op