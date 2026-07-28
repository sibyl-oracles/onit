"""Tests for chat.py turn telemetry and endpoint-metadata caching,
plus the vLLM prefix-cache probe in model/serving/diagnostics.py."""

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.chat import (
    _DECAY_MARKER,
    TOOL_RESULT_DECAY_CHARS,
    TurnMetrics,
    _autodetect_fallback_model,
    _decay_old_tool_results,
    _get_model_max_context,
    _handle_structured_tool_calls,
    _resolve_model_id,
    chat,
    reset_endpoint_caches,
    summarize_metrics,
)
from model.serving.diagnostics import (
    _metrics_url,
    format_report,
    parse_prometheus,
    prefix_cache_report,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    """Endpoint answers are cached for the life of the process, so a test that
    resolved a model would otherwise decide the next test's answer."""
    reset_endpoint_caches()
    yield
    reset_endpoint_caches()


def _mock_completion(content="done", tool_calls=None, prompt_tokens=100,
                     completion_tokens=20):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls" if tool_calls else "stop"
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage.prompt_tokens = prompt_tokens
    completion.usage.completion_tokens = completion_tokens
    return completion


def _mock_tool_call(name="local_search", arguments='{"query": "x"}', call_id="c1"):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    tc.id = call_id
    return tc


# ── TurnMetrics ─────────────────────────────────────────────────────────────

class TestTurnMetrics:
    def test_records_turn_and_totals(self):
        sink = {}
        m = TurnMetrics(sink)
        m.start_api()
        m.first_token()
        m.end_api(prompt_tokens=1000, completion_tokens=50, finish_reason="stop")

        assert sink["turn_count"] == 1
        assert sink["prompt_tokens_max"] == 1000
        assert sink["completion_tokens"] == 50
        assert sink["turns"][0]["finish_reason"] == "stop"
        assert sink["turns"][0]["ttft_s"] is not None

    def test_prompt_tokens_max_tracks_the_peak(self):
        sink = {}
        m = TurnMetrics(sink)
        for tokens in (500, 9000, 3000):
            m.start_api()
            m.end_api(prompt_tokens=tokens)
        assert sink["prompt_tokens_max"] == 9000
        assert sink["turn_count"] == 3

    def test_non_streaming_turn_has_no_prefill_split(self):
        """Without a first-token timestamp there is no line between waiting
        and generating, so neither bucket may be guessed at."""
        sink = {}
        m = TurnMetrics(sink)
        m.start_api()
        m.end_api(prompt_tokens=10)
        assert sink["turns"][0]["ttft_s"] is None
        assert sink["prefill_s"] == 0.0
        assert sink["decode_s"] == 0.0

    def test_tool_time_attaches_to_the_turn_that_called(self):
        sink = {}
        m = TurnMetrics(sink)
        m.start_api()
        m.end_api(prompt_tokens=10)
        m.add_tools(["local_search", "read_file"], 1.5)

        assert sink["tool_calls"] == 2
        assert sink["tool_s"] == 1.5
        assert sink["turns"][0]["tools"] == ["local_search", "read_file"]
        assert sink["turns"][0]["tool_s"] == 1.5

    def test_compaction_counted_apart_from_model_time(self):
        sink = {}
        m = TurnMetrics(sink)
        m.add_compaction(2.0)
        assert sink["compactions"] == 1
        assert sink["compaction_s"] == 2.0
        assert sink["model_s"] == 0.0

    def test_non_int_usage_does_not_poison_totals(self):
        """Providers report usage as None, or not at all; mocks report a mock."""
        sink = {}
        m = TurnMetrics(sink)
        m.start_api()
        m.end_api(prompt_tokens=None, completion_tokens=MagicMock())
        assert sink["prompt_tokens_max"] == 0
        assert sink["completion_tokens"] == 0

    def test_summarize_empty(self):
        assert summarize_metrics({}) == "no turns recorded"

    def test_summarize_mentions_turns_and_buckets(self):
        sink = {}
        m = TurnMetrics(sink)
        m.start_api()
        m.end_api(prompt_tokens=1234, completion_tokens=10)
        m.add_tools(["local_search"], 0.5)
        text = summarize_metrics(sink)
        assert "1 turn(s)" in text
        assert "1 tool call(s)" in text
        assert "1,234 tok" in text


# ── chat() wiring ───────────────────────────────────────────────────────────

class TestChatMetrics:
    @pytest.mark.asyncio
    async def test_counts_turns_and_tool_calls(self):
        """One tool call then a final answer: two model turns, one tool."""
        tc = _mock_tool_call()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion(content=None, tool_calls=[tc], prompt_tokens=1000),
            _mock_completion("The answer.", prompt_tokens=4000),
        ])
        registry = MagicMock()
        registry.tools = {"local_search"}
        registry.get_tool_items.return_value = [{"type": "function"}]
        registry.call_tool = AsyncMock(return_value="search results")

        metrics: dict = {}
        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"), \
             patch("model.serving.chat._handle_structured_tool_calls",
                   new_callable=AsyncMock, return_value=None):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="what do the docs say?",
                tool_registry=registry,
                safety_queue=asyncio.Queue(),
                metrics=metrics,
            )

        assert result == "The answer."
        assert metrics["turn_count"] == 2
        assert metrics["tool_calls"] == 1
        assert metrics["turns"][0]["tools"] == ["local_search"]
        # Prompt growth across turns is the cost the streaming rate can't see.
        assert metrics["prompt_tokens_max"] == 4000

    @pytest.mark.asyncio
    async def test_metrics_optional(self):
        """No sink, no accounting — chat() behaves exactly as before."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("plain answer"))

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="hi",
                safety_queue=asyncio.Queue(),
            )
        assert result == "plain answer"

    @pytest.mark.asyncio
    async def test_metrics_survive_a_failed_run(self):
        """A run that ends in an error is the one worth reading: the turn that
        was attempted must still be recorded."""
        from openai import OpenAIError

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=OpenAIError("boom"))

        metrics: dict = {}
        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"), \
             patch("model.serving.chat.asyncio.sleep", new_callable=AsyncMock):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="hi",
                safety_queue=asyncio.Queue(),
                metrics=metrics,
            )

        assert result is None
        assert metrics["turn_count"] == 0  # no API call completed
        assert metrics["turns"] == []


# ── first-token hook (the prefill/decode split) ─────────────────────────────

def _chunk(content=None, reasoning=None, tool_calls=None, finish_reason=None,
           usage=None, no_choices=False):
    c = MagicMock()
    c.usage = usage
    if no_choices:
        c.choices = []
        return c
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = reasoning
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    c.choices = [choice]
    return c


async def _aiter(chunks):
    for c in chunks:
        yield c


class TestFirstTokenHook:
    @pytest.mark.asyncio
    async def test_fires_once_on_the_first_generated_token(self):
        from model.serving.chat import _process_streaming_response

        calls = []
        chunks = [
            _chunk(no_choices=True),        # usage-only preamble
            _chunk(),                       # role-only delta: nothing generated yet
            _chunk(content="Hel"),
            _chunk(content="lo", finish_reason="stop"),
        ]
        result = await _process_streaming_response(
            _aiter(chunks), asyncio.Queue(), None, think=False,
            on_first_token=lambda: calls.append(1),
        )

        assert result[0] == "Hello"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_a_tool_call_turn_counts_as_generation(self):
        """A turn that only emits a tool call still ends its prefill."""
        from model.serving.chat import _process_streaming_response

        tc = MagicMock()
        tc.index = 0
        tc.id = "c1"
        tc.function.name = "local_search"
        tc.function.arguments = '{"query": "x"}'

        calls = []
        await _process_streaming_response(
            _aiter([_chunk(), _chunk(tool_calls=[tc], finish_reason="tool_calls")]),
            asyncio.Queue(), None, think=False,
            on_first_token=lambda: calls.append(1),
        )
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_never_fires_when_nothing_is_generated(self):
        from model.serving.chat import _process_streaming_response

        calls = []
        await _process_streaming_response(
            _aiter([_chunk(no_choices=True), _chunk(finish_reason="stop")]),
            asyncio.Queue(), None, think=False,
            on_first_token=lambda: calls.append(1),
        )
        assert calls == []


# ── reasoning budget across turns ───────────────────────────────────────────

class TestThinkToolTurns:
    async def _run(self, think_tool_turns):
        """Two turns: a tool call, then the answer.  Returns the per-request
        chat_template_kwargs the model was sent."""
        tc = _mock_tool_call()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _mock_completion(content=None, tool_calls=[tc]),
            _mock_completion("The answer."),
        ])
        registry = MagicMock()
        registry.tools = {"local_search"}
        registry.get_tool_items.return_value = [{"type": "function"}]

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="qwen3-30b"), \
             patch("model.serving.chat._handle_structured_tool_calls",
                   new_callable=AsyncMock, return_value=None):
            await chat(
                host="http://localhost:8000/v1", instruction="q",
                tool_registry=registry, safety_queue=asyncio.Queue(),
                think=True, think_tool_turns=think_tool_turns,
            )
        return [call.kwargs["extra_body"].get("chat_template_kwargs")
                for call in mock_client.chat.completions.create.call_args_list]

    @pytest.mark.asyncio
    async def test_thinking_on_every_turn_by_default(self):
        sent = await self._run(think_tool_turns=True)
        assert all(kw and kw["enable_thinking"] for kw in sent)

    @pytest.mark.asyncio
    async def test_thinking_kept_for_the_opening_turn_only(self):
        """The reasoning that decides the approach is worth paying for once;
        repeating it before every tool-call JSON is what costs the loop."""
        sent = await self._run(think_tool_turns=False)
        assert sent[0]["enable_thinking"] is True
        assert sent[1] is None


# ── endpoint metadata caching ───────────────────────────────────────────────

class TestEndpointCaches:
    @pytest.mark.asyncio
    async def test_model_id_resolved_once_per_host(self):
        client = AsyncMock()
        models = MagicMock()
        models.data = [MagicMock(id="qwen3")]
        client.models.list = AsyncMock(return_value=models)

        first = await _resolve_model_id(client, "http://h:8000/v1")
        second = await _resolve_model_id(client, "http://h:8000/v1")

        assert first == second == "qwen3"
        assert client.models.list.await_count == 1

    @pytest.mark.asyncio
    async def test_model_id_cached_per_host_not_globally(self):
        def _client(model_id):
            c = AsyncMock()
            models = MagicMock()
            models.data = [MagicMock(id=model_id)]
            c.models.list = AsyncMock(return_value=models)
            return c

        assert await _resolve_model_id(_client("a"), "http://one/v1") == "a"
        assert await _resolve_model_id(_client("b"), "http://two/v1") == "b"

    @pytest.mark.asyncio
    async def test_fallback_detection_bypasses_the_cache(self):
        """After a 404 the cached id is the dead one; re-detecting it would
        report 'no fallback available' forever."""
        client = AsyncMock()
        first = MagicMock()
        first.data = [MagicMock(id="old-model")]
        second = MagicMock()
        second.data = [MagicMock(id="new-model")]
        client.models.list = AsyncMock(side_effect=[first, second])

        assert await _resolve_model_id(client, "http://h/v1") == "old-model"
        detected = await _autodetect_fallback_model(
            client, None, False, "http://h/v1", "old-model")

        assert detected == "new-model"
        assert client.models.list.await_count == 2

    @pytest.mark.asyncio
    async def test_max_context_queried_once(self):
        payload = {"data": [{"id": "m", "max_model_len": 262144}]}
        resp = MagicMock(status_code=200)
        resp.json.return_value = payload
        http = AsyncMock()
        http.get = AsyncMock(return_value=resp)
        http.__aenter__ = AsyncMock(return_value=http)
        http.__aexit__ = AsyncMock(return_value=False)

        with patch("model.serving.chat.httpx.AsyncClient", return_value=http):
            a = await _get_model_max_context("http://h/v1", "EMPTY", "m")
            b = await _get_model_max_context("http://h/v1", "EMPTY", "m")

        assert a == b == 262144
        assert http.get.await_count == 1

    @pytest.mark.asyncio
    async def test_unknown_context_is_retried_after_the_ttl(self):
        """A host that was down for one call must be asked again, not written
        off until the process restarts."""
        resp = MagicMock(status_code=500)
        http = AsyncMock()
        http.get = AsyncMock(return_value=resp)
        http.__aenter__ = AsyncMock(return_value=http)
        http.__aexit__ = AsyncMock(return_value=False)

        with patch("model.serving.chat.httpx.AsyncClient", return_value=http):
            assert await _get_model_max_context("http://h/v1", "EMPTY", "m") is None
            assert await _get_model_max_context("http://h/v1", "EMPTY", "m") is None
            assert http.get.await_count == 1  # cached for now

            import model.serving.chat as chat_mod
            key = ("http://h/v1", "m")
            value, _expiry = chat_mod._MAX_CONTEXT_CACHE[key]
            chat_mod._MAX_CONTEXT_CACHE[key] = (value, 0.0)  # expire it

            assert await _get_model_max_context("http://h/v1", "EMPTY", "m") is None
            assert http.get.await_count == 2


# ── parallel tool batches ───────────────────────────────────────────────────

class TestParallelToolCalls:
    """A batch of reads is only as slow as its slowest call, but the results
    must still line up with the ids the model asked for."""

    def _registry(self, names, delay=0.0, order=None):
        def _handler_for(name):
            async def _handler(log_handler=None, **kwargs):
                if delay:
                    await asyncio.sleep(delay)
                if order is not None:
                    order.append(name)
                return f"result of {name}"
            return _handler

        reg = MagicMock()
        reg.tools = set(names)
        reg.tool_accepts_param.return_value = False
        reg.__getitem__ = lambda _self, key: _handler_for(key)
        return reg

    @pytest.mark.asyncio
    async def test_read_only_batch_runs_concurrently(self):
        calls = [_mock_tool_call("read_file", '{"path": "/a"}', "c1"),
                 _mock_tool_call("read_file", '{"path": "/b"}', "c2"),
                 _mock_tool_call("search_document", '{"query": "x"}', "c3")]
        registry = self._registry({"read_file", "search_document"}, delay=0.05)
        messages: list = []

        started = time.monotonic()
        bail = await _handle_structured_tool_calls(
            calls, {"role": "assistant"}, registry, None, "", None, False,
            messages, [], 30, asyncio.Queue(), session_id="s",
        )
        elapsed = time.monotonic() - started

        assert bail is None
        # Serial would be ~0.15s; concurrent is ~0.05s.
        assert elapsed < 0.12
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2", "c3"]

    @pytest.mark.asyncio
    async def test_results_keep_call_order_not_completion_order(self):
        """The API pairs tool messages to tool_call ids by position, so a fast
        second call must not overtake a slow first one."""
        slow_first = [_mock_tool_call("read_file", '{"path": "/slow"}', "c1"),
                      _mock_tool_call("read_file", '{"path": "/fast"}', "c2")]

        async def _handler(log_handler=None, path=None, **kwargs):
            await asyncio.sleep(0.05 if path == "/slow" else 0.0)
            return f"content of {path}"

        registry = MagicMock()
        registry.tools = {"read_file"}
        registry.tool_accepts_param.return_value = False
        registry.__getitem__ = lambda _self, key: _handler

        messages: list = []
        await _handle_structured_tool_calls(
            slow_first, {"role": "assistant"}, registry, None, "", None, False,
            messages, [], 30, asyncio.Queue(), session_id="s",
        )
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
        assert "slow" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_writes_stay_sequential(self):
        """A batch that writes is a script, and a script has an order."""
        order: list = []

        async def _handler(log_handler=None, **kwargs):
            name = kwargs.get("path", "?")
            await asyncio.sleep(0.02 if name == "/first" else 0.0)
            order.append(name)
            return "written"

        registry = MagicMock()
        registry.tools = {"write_file"}
        registry.tool_accepts_param.return_value = False
        registry.__getitem__ = lambda _self, key: _handler

        calls = [_mock_tool_call("write_file", '{"path": "/first"}', "c1"),
                 _mock_tool_call("write_file", '{"path": "/second"}', "c2")]
        await _handle_structured_tool_calls(
            calls, {"role": "assistant"}, registry, None, "", None, False,
            [], [], 30, asyncio.Queue(), session_id="s",
        )
        assert order == ["/first", "/second"]

    @pytest.mark.asyncio
    async def test_a_single_call_is_unaffected(self):
        registry = MagicMock()
        registry.tools = {"read_file"}
        registry.tool_accepts_param.return_value = False

        async def _handler(log_handler=None, **kwargs):
            return "one result"
        registry.__getitem__ = lambda _self, key: _handler

        messages: list = []
        await _handle_structured_tool_calls(
            [_mock_tool_call("read_file", '{"path": "/a"}', "c1")], {"role": "assistant"},
            registry, None, "", None, False, messages, [], 30,
            asyncio.Queue(), session_id="s",
        )
        assert [m["tool_call_id"] for m in messages if m.get("role") == "tool"] == ["c1"]

    @pytest.mark.asyncio
    async def test_every_call_is_answered_even_when_one_fails(self):
        """A tool_call id with no tool message gets the next request rejected."""
        async def _handler(log_handler=None, path=None, **kwargs):
            if path == "/bad":
                raise RuntimeError("exploded")
            return "fine"

        registry = MagicMock()
        registry.tools = {"read_file"}
        registry.tool_accepts_param.return_value = False
        registry.__getitem__ = lambda _self, key: _handler

        messages: list = []
        await _handle_structured_tool_calls(
            [_mock_tool_call("read_file", '{"path": "/bad"}', "c1"),
             _mock_tool_call("read_file", '{"path": "/ok"}', "c2")],
            {"role": "assistant"}, registry, None, "", None, False,
            messages, [], 30, asyncio.Queue(), session_id="s",
        )
        ids = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
        assert ids == ["c1", "c2"]


# ── tool-result decay ───────────────────────────────────────────────────────

class TestToolResultDecay:
    def _tool_msg(self, text):
        return {"role": "tool", "content": text, "name": "read_file",
                "tool_call_id": "c"}

    def test_recent_results_are_untouched(self):
        big = "x" * (TOOL_RESULT_DECAY_CHARS * 2)
        messages = [{"role": "user", "content": "hi"}] + \
                   [self._tool_msg(big) for _ in range(3)]
        _decay_old_tool_results(messages, keep_full=3)
        assert all(m["content"] == big for m in messages[1:])

    def test_older_results_are_cut_to_their_opening(self):
        big = "x" * (TOOL_RESULT_DECAY_CHARS * 2)
        messages = [self._tool_msg(big) for _ in range(5)]
        _decay_old_tool_results(messages, keep_full=2)
        assert all(m["content"].endswith(_DECAY_MARKER) for m in messages[:3])
        assert all(m["content"] == big for m in messages[3:])

    def test_short_results_are_left_alone(self):
        messages = [self._tool_msg("brief"), self._tool_msg("also brief"),
                    self._tool_msg("third")]
        _decay_old_tool_results(messages, keep_full=1)
        assert [m["content"] for m in messages] == ["brief", "also brief", "third"]

    def test_decay_is_idempotent(self):
        big = "x" * (TOOL_RESULT_DECAY_CHARS * 2)
        messages = [self._tool_msg(big), self._tool_msg(big), self._tool_msg(big)]
        _decay_old_tool_results(messages, keep_full=1)
        once = messages[0]["content"]
        _decay_old_tool_results(messages, keep_full=1)
        assert messages[0]["content"] == once

    def test_image_results_are_left_to_the_image_stripper(self):
        """Their content is a list of parts, not text to slice."""
        parts = [{"type": "text", "text": "y" * 9000},
                 {"type": "image_url", "image_url": {"url": "data:..."}}]
        messages = [{"role": "tool", "content": parts, "tool_call_id": "c"},
                    self._tool_msg("z" * 9000), self._tool_msg("recent")]
        _decay_old_tool_results(messages, keep_full=1)
        assert messages[0]["content"] is parts


# ── prefix-cache probe ──────────────────────────────────────────────────────

class TestPrefixCacheProbe:
    def test_metrics_url_drops_the_api_prefix(self):
        assert _metrics_url("http://h:8000/v1") == "http://h:8000/metrics"
        assert _metrics_url("http://h:8000/v1/") == "http://h:8000/metrics"
        assert _metrics_url("http://h:8000") == "http://h:8000/metrics"

    def test_parse_sums_samples_across_labels(self):
        text = (
            "# HELP vllm:prefix_cache_hits_total Hits\n"
            'vllm:prefix_cache_hits_total{model_name="a"} 30.0\n'
            'vllm:prefix_cache_hits_total{model_name="b"} 12.0\n'
        )
        assert parse_prometheus(text, ("vllm:prefix_cache_hits_total",)) == 42.0

    def test_parse_returns_none_when_absent(self):
        assert parse_prometheus("vllm:num_requests_running 1.0\n",
                                ("vllm:prefix_cache_hits_total",)) is None

    def test_parse_ignores_a_longer_metric_with_the_same_head(self):
        text = "vllm:prefix_cache_hits_total_extra 5.0\n"
        assert parse_prometheus(text, ("vllm:prefix_cache_hits_total",)) is None

    @pytest.mark.asyncio
    async def test_report_computes_hit_rate(self):
        text = ("vllm:prefix_cache_queries_total 1000.0\n"
                "vllm:prefix_cache_hits_total 900.0\n")
        resp = MagicMock(status_code=200, text=text)
        http = AsyncMock()
        http.get = AsyncMock(return_value=resp)
        http.__aenter__ = AsyncMock(return_value=http)
        http.__aexit__ = AsyncMock(return_value=False)

        with patch("model.serving.diagnostics.httpx.AsyncClient", return_value=http):
            report = await prefix_cache_report("http://h:8000/v1")

        assert report["enabled"] is True
        assert report["hit_rate"] == 0.9
        assert "hit rate: 90.0%" in format_report("http://h:8000/v1", report)

    @pytest.mark.asyncio
    async def test_no_metrics_reads_as_unknown_not_disabled(self):
        resp = MagicMock(status_code=200, text="vllm:num_requests_running 1.0\n")
        http = AsyncMock()
        http.get = AsyncMock(return_value=resp)
        http.__aenter__ = AsyncMock(return_value=http)
        http.__aexit__ = AsyncMock(return_value=False)

        with patch("model.serving.diagnostics.httpx.AsyncClient", return_value=http):
            report = await prefix_cache_report("http://h:8000/v1")

        assert report["reachable"] is True
        assert report["enabled"] is None
        assert "unknown" in format_report("http://h:8000/v1", report)

    @pytest.mark.asyncio
    async def test_unreachable_host_reports_cleanly(self):
        with patch("model.serving.diagnostics.httpx.AsyncClient",
                   side_effect=OSError("connection refused")):
            report = await prefix_cache_report("http://nope:8000/v1")

        assert report["reachable"] is False
        assert "unreachable" in format_report("http://nope:8000/v1", report)

    @pytest.mark.asyncio
    async def test_low_hit_rate_is_called_out(self):
        text = ("vllm:prefix_cache_queries_total 1000.0\n"
                "vllm:prefix_cache_hits_total 100.0\n")
        resp = MagicMock(status_code=200, text=text)
        http = AsyncMock()
        http.get = AsyncMock(return_value=resp)
        http.__aenter__ = AsyncMock(return_value=http)
        http.__aexit__ = AsyncMock(return_value=False)

        with patch("model.serving.diagnostics.httpx.AsyncClient", return_value=http):
            report = await prefix_cache_report("http://h:8000/v1")

        assert "prefix is probably changing" in format_report("http://h", report)
