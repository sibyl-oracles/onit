"""Reasoning on by default, and a 1M context window by default.

Two defaults changed together, and they are tested together because they fail
together in the same way: neither is visible in a diff of the code a user runs.
``think`` used to be ``False`` in four places (the ``chat()`` signature, two
``model_serving.get("think", False)`` fallbacks in ``onit.py``, the three
doctor probes, the web-only defaults block and the shipped YAML), so turning
it on in one place left the other five answering ``False``.  The context
window had the same shape: a constant, a hard-coded ceiling in the Ollama
``num_ctx`` branch, the shipped YAML and four docs.

So these tests are about the *effective* value — what reaches the server —
rather than about any one of those places.  A test that asserts
``DEFAULT_THINK is True`` would have passed the whole time the CLI was still
sending ``enable_thinking: False``.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.serving.chat import DEFAULT_MAX_CONTEXT_TOKENS, DEFAULT_MAX_TOKENS, DEFAULT_THINK


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
REPO = os.path.dirname(SRC)


# ── what the model server is actually told ───────────────────────────────────


def _completion(text="Done."):
    message = MagicMock()
    message.content = text
    message.tool_calls = None
    message.reasoning_content = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage.prompt_tokens = 0
    return completion


async def _vllm_request(**chat_kwargs):
    """The body a vLLM request carries, with nothing said about thinking."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=_completion())
    with (
        patch("model.serving.chat.AsyncOpenAI", return_value=client),
        patch(
            "model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="qwen3-30b"
        ),
        patch(
            "model.serving.chat._get_model_max_context", new_callable=AsyncMock, return_value=None
        ),
    ):
        from model.serving.chat import chat

        await chat(
            host="http://localhost:8000/v1",
            instruction="hi",
            safety_queue=asyncio.Queue(),
            **chat_kwargs,
        )
    return client.chat.completions.create.call_args.kwargs


async def _ollama_request(**chat_kwargs):
    """The kwargs an Ollama chat() carries, with nothing said about thinking."""
    client = AsyncMock()
    client.chat = AsyncMock(
        return_value=MagicMock(
            message=MagicMock(content="Done.", tool_calls=None, thinking=None),
            done=True,
            done_reason="stop",
        )
    )
    with (
        patch("model.serving.chat._create_ollama_client", return_value=client),
        patch(
            "model.serving.chat._ollama_resolve_model_id",
            new_callable=AsyncMock,
            return_value="glm-5.3:cloud",
        ),
        patch(
            "model.serving.chat._get_ollama_max_context", new_callable=AsyncMock, return_value=None
        ),
    ):
        from model.serving.chat import chat

        await chat(
            host="https://ollama.com",
            host_key="k",
            instruction="hi",
            model="glm-5.3:cloud",
            safety_queue=asyncio.Queue(),
            **chat_kwargs,
        )
    return client.chat.call_args.kwargs


class TestTheThinkConstant:
    """One number, in one place.  ``think`` used to be written ``False`` into
    six of them; a constant is what stops that recurring."""

    def test_it_is_true(self):
        assert DEFAULT_THINK is True

    def test_it_is_a_bool(self):
        """A truthy non-bool would survive ``if think`` and then go out on the
        wire as a string on the Ollama path, which wants a real boolean."""
        assert isinstance(DEFAULT_THINK, bool)

    def test_chat_defaults_to_it(self):
        import inspect
        from model.serving.chat import chat

        default = inspect.signature(chat).parameters["think"].default
        assert default is DEFAULT_THINK


class TestReasoningReachesTheServer:
    """The default has to be *said* on the wire, not just held in a variable.

    A vLLM request that omits ``enable_thinking`` is not a request with
    thinking off — the Qwen3 template reads a missing switch as thinking on.
    So "on by default" is only true if the parameter is sent.
    """

    @pytest.mark.asyncio
    async def test_vllm_is_told_to_think(self):
        body = await _vllm_request()
        assert body["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}

    @pytest.mark.asyncio
    async def test_ollama_is_told_to_think(self):
        body = await _ollama_request()
        assert body.get("think") is True

    @pytest.mark.asyncio
    async def test_vllm_can_still_be_told_not_to(self):
        """The off switch is what makes the on default usable."""
        body = await _vllm_request(think=False)
        assert body["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}

    @pytest.mark.asyncio
    async def test_ollama_can_still_be_told_not_to(self):
        body = await _ollama_request(think=False)
        assert not body.get("think")


class TestTheContextWindowDefault:
    """1M when nothing can say what the real window is."""

    def test_the_constant_is_one_million(self):
        assert DEFAULT_MAX_CONTEXT_TOKENS == 1_000_000

    def test_it_is_the_same_spelling_the_flag_accepts(self):
        """``--max-context-tokens 1M`` parses to 1000000, so the default and
        the flag have to name the same number or a user who sets it explicitly
        gets a different run from one who does not."""
        from src.cli import _token_count

        assert _token_count("1M") == DEFAULT_MAX_CONTEXT_TOKENS

    def test_it_leaves_the_compaction_threshold_off_the_floor(self):
        """The output default is reserved against the window, so a window this
        size must not push the threshold to its 0.50 floor — that would force
        compaction twice as often as a 1M window should need."""
        window = DEFAULT_MAX_CONTEXT_TOKENS
        reserved = DEFAULT_MAX_TOKENS + int(window * 0.05)
        threshold = min(0.90, max(0.50, 1.0 - reserved / window))
        assert threshold == pytest.approx(0.90, abs=1e-4)

    @pytest.mark.asyncio
    async def test_ollama_is_asked_for_the_same_window(self):
        """The window a run is accounted against and the window Ollama is
        asked to allocate have to match, or the agent measures prompts against
        a window the server never allocated."""
        body = await _ollama_request()
        assert body["options"]["num_ctx"] == DEFAULT_MAX_CONTEXT_TOKENS

    @pytest.mark.asyncio
    async def test_a_reported_window_still_wins(self):
        """The default is a fallback, not an override."""
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=_completion())
        with (
            patch("model.serving.chat.AsyncOpenAI", return_value=client),
            patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="m"),
            patch(
                "model.serving.chat._get_model_max_context",
                new_callable=AsyncMock,
                return_value=131072,
            ),
        ):
            from model.serving.chat import chat

            await chat(
                host="http://localhost:8000/v1", instruction="hi", safety_queue=asyncio.Queue()
            )
        body = client.chat.completions.create.call_args.kwargs
        # A 128k window cannot take a 1M output budget; the clamp proves the
        # reported window is what the run was measured against.
        assert body["max_tokens"] <= 131072

    @pytest.mark.asyncio
    async def test_an_explicit_window_still_wins(self):
        body = await _ollama_request(max_context_tokens=65536)
        assert body["options"]["num_ctx"] == 65536


# ── the CLI and the shipped config ───────────────────────────────────────────


def _shipped_config() -> dict:
    with open(os.path.join(SRC, "configs", "default.yaml")) as f:
        return yaml.safe_load(f) or {}


def _flag(flag: str):
    from src.cli import _build_parser

    action = _build_parser()._option_string_actions.get(flag)
    assert action is not None, f"{flag} is not an option of the parser"
    return action


class TestTheShippedConfig:
    def test_it_turns_reasoning_on(self):
        """The file ships with ``think: true``.  A user config replaces this
        file rather than merging with it, so the code default has to agree —
        which is what the tests above cover."""
        assert _shipped_config()["serving"]["think"] is True

    def test_it_does_not_pin_a_context_window(self):
        """Auto-detection is better than a number, so the shipped file leaves
        the key commented out and the code default answers."""
        assert "max_context_tokens" not in _shipped_config()["serving"]


class TestTheCli:
    def test_think_is_an_on_off_pair(self):
        """``--think`` alone cannot express "off" once the default is on."""
        action = _flag("--think")
        assert "--no-think" in action.option_strings

    def test_an_absent_flag_is_not_a_value(self):
        """``None`` is what lets the config decide; a ``False`` default here
        would make every run without the flag a --no-think run."""
        assert _flag("--think").default is None

    def test_the_help_says_on_by_default(self):
        assert "On by default" in (_flag("--think").help or "")

    def test_the_context_help_names_1m(self):
        help_text = _flag("--max-context-tokens").help or ""
        assert "1M" in help_text
        assert "262144" not in help_text


# ── the whole chain, from the command line to chat()'s kwargs ────────────────


class TestTheChainFromTheFlagToChat:
    """Each link is tested above; this is the chain.

    A default that is right in ``chat()`` and wrong in the CLI is a default
    nobody gets — which is exactly how ``think`` was ``False`` in six places.
    The path is: parse the flags -> resolve the config -> build the agent ->
    ``_chat_with_failover`` -> ``chat(think=...)``.  The last hop matters:
    ``think`` is *not* in the kwargs dict ``_chat_kwargs`` builds (it is
    passed as its own argument, so putting it in the dict too would be a
    duplicate keyword), so a test that reads that dict sees nothing either
    way and proves nothing.  These read the real call.
    """

    def _think_passed_to_chat(self, tmp_path, extra_flags, config_extra=None):
        import yaml
        from unittest.mock import AsyncMock, patch
        from src import setup as setup_mod
        from src.cli import _build_parser, _parse_and_resolve_config
        from src.onit import OnIt, RunState, _chat_with_failover
        from type.tools import ToolRegistry

        serving = {"host": "http://x/v1"}
        serving.update(config_extra or {})
        cfg = {
            "serving": serving,
            "data_path": str(tmp_path),
            "session_path": str(tmp_path / "sessions"),
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg))
        with (
            patch.object(setup_mod, "get_secret", lambda k: None),
            patch.object(setup_mod, "CONFIG_PATH", str(tmp_path / "no-setup.yaml")),
        ):
            args = _build_parser().parse_args(["--config", str(path)] + extra_flags)
            resolved = _parse_and_resolve_config(args)
        with patch("src.onit.discover_tools", return_value=ToolRegistry()):
            agent = OnIt(config=resolved)

        seen = {}

        async def _fake_chat(**kwargs):
            seen.update(kwargs)
            return "OK."

        kwargs = agent._chat_kwargs(
            metrics={},
            run_state=RunState(),
            chat_ui=None,
            verbose=False,
            data_path=str(tmp_path),
            session_id="s",
            session_history=[],
        )
        import asyncio as _asyncio

        with patch("src.onit.chat", _fake_chat), patch("src.onit._call_sandbox_stop", AsyncMock()):
            _asyncio.run(
                _chat_with_failover(
                    agent,
                    instruction="hi",
                    kwargs=kwargs,
                    safety_queue=_asyncio.Queue(),
                    session_key="s",
                )
            )
        return resolved, seen

    def test_a_run_with_no_flag_reasons(self, tmp_path):
        """The default a user who types nothing gets."""
        resolved, seen = self._think_passed_to_chat(tmp_path, [])
        assert resolved["serving"].get("think", DEFAULT_THINK) is True
        assert seen.get("think", DEFAULT_THINK) is True

    def test_no_think_turns_it_off(self, tmp_path):
        resolved, seen = self._think_passed_to_chat(tmp_path, ["--no-think"])
        assert resolved["serving"]["think"] is False
        assert seen["think"] is False

    def test_think_turns_it_back_on(self, tmp_path):
        resolved, seen = self._think_passed_to_chat(tmp_path, ["--think"])
        assert resolved["serving"]["think"] is True
        assert seen.get("think", DEFAULT_THINK) is True

    def test_a_config_that_says_false_is_obeyed(self, tmp_path):
        """An explicit ``serving.think: false`` is a preference, not a
        default, so nothing on the way through may override it."""
        resolved, seen = self._think_passed_to_chat(tmp_path, [], config_extra={"think": False})
        assert resolved["serving"]["think"] is False
        assert seen["think"] is False

    def test_the_flag_beats_the_config(self, tmp_path):
        """``--think`` over a config that says false: the flag is the more
        recent statement of intent."""
        resolved, seen = self._think_passed_to_chat(
            tmp_path, ["--think"], config_extra={"think": False}
        )
        assert resolved["serving"]["think"] is True
        assert seen["think"] is True


# ── the docs ─────────────────────────────────────────────────────────────────


class TestDocs:
    """The docs are what a user reads instead of the code, and 262144 was
    written into four of them."""

    def _read(self, *parts: str) -> str:
        with open(os.path.join(REPO, *parts)) as f:
            return f.read()

    @pytest.mark.parametrize(
        "name",
        [
            "CLI.md",
            "CONFIGURATION.md",
            "MODEL_SERVING.md",
            "RUN_A_MODEL_SERVER.md",
        ],
    )
    def test_no_doc_claims_the_old_fallback(self, name):
        """262144 is still a fine number to *illustrate* with — it is a real
        server flag (``--max-model-len 262144``) and a real window for a real
        model.  What must be gone is the claim that it is what applies when
        nothing else says, so the search is for the claim, not the digits.
        """
        text = self._read("docs", name)
        lowered = text.lower()
        for old in ("262144", "262,144"):
            for phrase in (
                f"falls back to an assumed {old}",
                f"falls back to an assumed {old[:3]},{old[3:]}",
                f"fallback of {old}",
                f"fallback of {old[:3]},{old[3:]}",
                f"assumed {old}",
                f"else `{old}`",
            ):
                assert phrase not in lowered, f"docs/{name} still claims {old} is the fallback"
        # And no line may offer it as the "else" arm of the resolution.
        for line in text.splitlines():
            if "262144" in line or "262,144" in line:
                assert "fallback" not in line.lower(), (
                    f"docs/{name} still calls 262144 a fallback: {line}"
                )
                assert "unreported" not in line.lower(), (
                    f"docs/{name} still calls 262144 the unreported default: {line}"
                )

    def test_the_cli_doc_states_the_new_fallback(self):
        text = self._read("docs", "CLI.md")
        row = [ln for ln in text.splitlines() if "--max-context-tokens" in ln]
        assert row, "docs/CLI.md no longer documents --max-context-tokens"
        assert "1M" in " ".join(row) or "1000000" in " ".join(row)

    def test_the_cli_doc_states_think_is_on(self):
        text = self._read("docs", "CLI.md")
        row = [ln for ln in text.splitlines() if "`--think`" in ln]
        assert row, "docs/CLI.md no longer documents --think"
        assert "`true`" in " ".join(row)


# ── the Ollama fallbacks: thinking, and the window it really allocated ──────
#
# Both defects have the same shape: a request the server quietly answers with
# something other than what was asked.  A model without the thinking
# capability refuses the switch with a 400 that the generic retry re-sent
# verbatim, and a model whose trained window is smaller than ``num_ctx``
# allocates what it can and reports success.  Neither is visible in the
# request, so both are tested through what reaches the server.


def _ollama_client(allocated=None, capabilities=None, native_ctx=None, on_think=None):
    """An AsyncClient stub that behaves like the real server on these paths."""
    calls: list = []

    def _chat(**kwargs):
        calls.append(kwargs)
        if kwargs.get("think") and on_think is not None:
            return on_think()
        response = MagicMock(
            message=MagicMock(content="Done.", tool_calls=None, thinking=None),
            done=True,
            done_reason="stop",
            prompt_eval_count=100,
            eval_count=5,
        )

        async def _ok():
            return response

        return _ok()

    client = AsyncMock()
    client.chat = _chat
    show = MagicMock()
    show.capabilities = capabilities
    show.modelinfo = {"qwen3_5.context_length": native_ctx} if native_ctx else {}
    client.show = AsyncMock(return_value=show)
    running = MagicMock()
    running.models = [MagicMock(model="m:1b", context_length=allocated)] if allocated else []
    client.ps = AsyncMock(return_value=running)
    client._calls = calls
    return client


async def _run_ollama(client, reset: bool = True, **chat_kwargs):
    """One chat() against the stub.

    ``reset`` clears the module caches first, which is what a fresh process
    looks like.  A test that asserts on what one run remembers for the next
    passes ``reset=False`` — clearing between them would erase the very
    memory under test.
    """
    from model.serving.chat import chat, reset_endpoint_caches

    if reset:
        reset_endpoint_caches()
    with (
        patch("model.serving.chat._create_ollama_client", return_value=client),
        patch(
            "model.serving.chat._ollama_resolve_model_id",
            new_callable=AsyncMock,
            return_value="m:1b",
        ),
    ):
        return await chat(
            host="https://ollama.com",
            host_key="k",
            model="m:1b",
            instruction="hi",
            safety_queue=asyncio.Queue(),
            **chat_kwargs,
        )


def _rejection():
    """The 400 Ollama returns for a model built without the capability."""
    import ollama

    async def _raise():
        raise ollama.ResponseError('{"error": "m:1b does not support thinking"}', 400)

    return _raise()


class TestANonThinkingOllamaModelStillAnswers:
    """Thinking is on by default, so a model that cannot think must not turn
    every turn into three identical rejections and no answer."""

    @pytest.mark.asyncio
    async def test_the_capability_list_is_honoured_before_the_request(self):
        """``/api/show`` says what the model takes; ask it rather than the 400."""
        client = _ollama_client(capabilities=["completion", "tools"])
        answer = await _run_ollama(client)
        assert answer and "Done." in answer
        assert len(client._calls) == 1, "no rejection should have been spent"
        assert "think" not in client._calls[0]

    @pytest.mark.asyncio
    async def test_a_rejection_is_retried_without_the_switch(self):
        """An old server publishes no capabilities, so the 400 is the only
        signal — and the retry has to differ from the request that failed."""
        client = _ollama_client(capabilities=None, on_think=_rejection)
        answer = await _run_ollama(client)
        assert answer and "Done." in answer
        assert len(client._calls) == 2, "one refusal, one recovery"
        assert client._calls[0].get("think") is True
        assert "think" not in client._calls[1]

    @pytest.mark.asyncio
    async def test_the_rejection_is_remembered_for_later_turns(self):
        """Paying for the same refusal on every turn would make a multi-turn
        run on a non-thinking model quadratic in wasted requests."""
        client = _ollama_client(capabilities=None, on_think=_rejection)
        await _run_ollama(client)
        first = len(client._calls)
        await _run_ollama(client, reset=False)
        # Second run: the refusal is remembered, so one request and no probe.
        assert len(client._calls) - first == 1
        assert "think" not in client._calls[-1]

    @pytest.mark.asyncio
    async def test_a_thinking_model_still_gets_the_switch(self):
        """The fallback must not turn reasoning off for models that have it."""
        client = _ollama_client(capabilities=["completion", "thinking"])
        await _run_ollama(client)
        assert client._calls[0].get("think") is True

    def test_an_unrecognised_capability_shape_is_silence_not_a_no(self):
        """Reading silence as "cannot think" would disable reasoning for every
        model on a server that words its answer differently."""
        from model.serving.chat import _record_ollama_thinking_capability

        assert _record_ollama_thinking_capability("m", MagicMock(capabilities=MagicMock())) is None
        assert _record_ollama_thinking_capability("m", MagicMock(capabilities=None)) is None

    def test_an_empty_thinking_block_is_an_explicit_no(self):
        from model.serving.chat import _record_ollama_thinking_capability

        resp = MagicMock(capabilities=None, thinking={"values": []})
        assert _record_ollama_thinking_capability("m", resp) is False


class TestTheWindowOllamaActuallyAllocated:
    """A ``num_ctx`` larger than the model's trained window is clamped in
    silence, so the window a run is accounted against has to come from the
    server after the first turn, not from the request."""

    @pytest.mark.asyncio
    async def test_the_allocated_window_replaces_an_optimistic_one(self):
        """No native window known: 1M is asked for, 262144 is allocated."""
        seen: list = []

        class _UI:
            def set_token_budgets(self, max_output_tokens=0, max_context_tokens=0):
                seen.append(max_context_tokens)

            def set_context_usage(self, pct, max_tokens=0):
                pass

            def __getattr__(self, name):
                return MagicMock()

        client = _ollama_client(allocated=262144, capabilities=["completion", "thinking"])
        await _run_ollama(client, chat_ui=_UI())
        assert client._calls[0]["options"]["num_ctx"] == DEFAULT_MAX_CONTEXT_TOKENS
        # The UI is told twice: the request, then the fact.
        assert seen[-1] == 262144
        assert seen[0] == DEFAULT_MAX_CONTEXT_TOKENS

    @pytest.mark.asyncio
    async def test_the_clamped_window_drives_compaction(self):
        """The point of the number: compaction has to fire at 90% of what the
        server holds, not 90% of what was asked for."""
        import model.serving.chat as chat_mod

        captured: dict = {}
        real_threshold = chat_mod._ModelCaller

        def _spy(**kwargs):
            caller = real_threshold(**kwargs)
            original = caller.call_turn

            async def _wrapped(messages, state, turn_think):
                captured["window"] = caller.max_context_tokens
                return await original(messages, state, turn_think)

            caller.call_turn = _wrapped
            return caller

        client = _ollama_client(allocated=262144, capabilities=["completion", "thinking"])
        with patch("model.serving.chat._ModelCaller", side_effect=_spy):
            await _run_ollama(client)
        # First call still carries the request; the reconcile runs after it.
        assert captured["window"] == DEFAULT_MAX_CONTEXT_TOKENS

    @pytest.mark.asyncio
    async def test_a_server_that_reports_nothing_keeps_the_request(self):
        """``/api/ps`` without the field is not evidence of a smaller window."""
        client = _ollama_client(allocated=None, capabilities=["completion", "thinking"])
        client.ps = AsyncMock(return_value=MagicMock(models=[MagicMock(model="m:1b")]))
        answer = await _run_ollama(client)
        assert answer and "Done." in answer

    @pytest.mark.asyncio
    async def test_a_failing_ps_does_not_fail_the_run(self):
        client = _ollama_client(capabilities=["completion", "thinking"])
        client.ps = AsyncMock(side_effect=RuntimeError("no such endpoint"))
        answer = await _run_ollama(client)
        assert answer and "Done." in answer

    def test_a_bare_name_matches_the_latest_tag(self):
        """Ollama normalises ``qwen3`` to ``qwen3:latest``; a strict compare
        would miss the model it was just told about."""
        from model.serving.chat import _ollama_model_matches

        assert _ollama_model_matches("qwen3", "qwen3:latest")
        assert _ollama_model_matches("qwen3:latest", "qwen3")
        assert _ollama_model_matches("qwen3:8b", "qwen3:8b")
        assert not _ollama_model_matches("qwen3:8b", "qwen3:70b")


class TestAWindowFromTheModelIsNotSecondGuessed:
    """``/api/ps`` is not always the better witness.

    Measured on Ollama 0.34.4: a server whose ``OLLAMA_CONTEXT_LENGTH`` is
    262144 reports ``context_length: 4096`` from ``/api/ps`` for every request,
    whatever ``num_ctx`` asked for.  The window derived from the model's own
    metadata is already clamped to what the model was trained on, so trusting
    ``/api/ps`` over it would account the run against 4096 and compact at 3,686
    tokens -- a fifth of the window the server really holds, on a prompt it
    answered happily.

    So the reconcile runs only when the request was a guess.
    """

    @pytest.mark.asyncio
    async def test_a_native_window_is_not_replaced_by_a_smaller_ps_figure(self):
        """The model says 262144; ``/api/ps`` says 4096.  The model wins."""
        seen: list = []

        class _UI:
            def set_token_budgets(self, max_output_tokens=0, max_context_tokens=0):
                seen.append(max_context_tokens)

            def set_context_usage(self, pct, max_tokens=0):
                pass

            def __getattr__(self, name):
                return MagicMock()

        client = _ollama_client(
            allocated=4096, capabilities=["completion", "thinking"], native_ctx=262144
        )
        await _run_ollama(client, chat_ui=_UI())
        # Asked for the model's window, not the 1M ceiling.
        assert client._calls[0]["options"]["num_ctx"] == 262144
        # And never told the run is smaller than that.
        assert 4096 not in seen
        assert seen[-1] == 262144

    @pytest.mark.asyncio
    async def test_an_unknown_native_window_still_reconciles(self):
        """The guard must not disable the check it was added beside: with no
        native window the request is a guess, so ``/api/ps`` still corrects it.
        """
        seen: list = []

        class _UI:
            def set_token_budgets(self, max_output_tokens=0, max_context_tokens=0):
                seen.append(max_context_tokens)

            def set_context_usage(self, pct, max_tokens=0):
                pass

            def __getattr__(self, name):
                return MagicMock()

        client = _ollama_client(allocated=262144, capabilities=["completion", "thinking"])
        await _run_ollama(client, chat_ui=_UI())
        assert client._calls[0]["options"]["num_ctx"] == DEFAULT_MAX_CONTEXT_TOKENS
        assert seen[-1] == 262144

    @pytest.mark.asyncio
    async def test_an_explicit_window_is_not_second_guessed(self):
        """A caller that names ``num_ctx`` has said what it wants; there is no
        guess to check, so ``/api/ps`` is not consulted."""
        client = _ollama_client(
            allocated=4096, capabilities=["completion", "thinking"], native_ctx=262144
        )
        client.ps = AsyncMock(side_effect=AssertionError("ps should not be called"))
        answer = await _run_ollama(client, num_ctx=65536)
        assert answer and "Done." in answer
        assert client._calls[0]["options"]["num_ctx"] == 65536
