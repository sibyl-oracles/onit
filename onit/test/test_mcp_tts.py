"""Tests for the local TTS tool.

The Kokoro engine is never loaded here: weights are hundreds of megabytes
and a test run must not touch the network. ``synthesize_to_wav`` is
monkeypatched at the module boundary, which is also the seam a different
engine would plug into.
"""

import json
import os
import wave

import pytest

from onit.mcp.servers.tasks.tools import tts


@pytest.fixture()
def fake_engine(monkeypatch, tmp_path):
    """Replace the engine with one that emits a half-second of silence."""

    class FakeEngine:
        def create(self, text, voice, speed, lang):
            assert voice, "voice must reach the engine"
            # 12000 samples at 24 kHz = 0.5 s; the sample rate is what the
            # WAV header ends up carrying, so a non-default value proves
            # the plumbing rather than a coincidence.
            import numpy as np

            return np.zeros(12000, dtype="float32"), 24000

    monkeypatch.setattr(tts, "_ENGINE", FakeEngine())
    return None


@pytest.fixture(autouse=True)
def sandbox(monkeypatch, tmp_path):
    """Every write lands in a throwaway directory, never the real sandbox."""
    monkeypatch.setattr(tts, "DATA_PATH", str(tmp_path))
    return tmp_path


def test_speak_writes_wav_and_reports_metadata(fake_engine, sandbox):
    out = json.loads(tts.speak_impl(text="Hello there.", data_path=str(sandbox)))
    assert out["status"] == "ok"
    assert out["filename"].startswith("tts_af_heart_")
    assert out["duration_s"] == 0.5
    assert out["sample_rate"] == 24000
    path = os.path.join(str(sandbox), out["filename"])
    assert os.path.isfile(path)
    with wave.open(path, "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.getnframes() == 12000


def test_speak_requires_text(fake_engine):
    out = json.loads(tts.speak_impl(text=None))
    assert out["error"].startswith("Missing required argument")
    assert out["status"] == "error"


def test_speak_rejects_unknown_voice(fake_engine, sandbox):
    out = json.loads(tts.speak_impl(text="Hi", voice="nope", data_path=str(sandbox)))
    assert out["status"] == "error"
    assert "af_heart" in out["error"]


def test_speak_clamps_speed(fake_engine, sandbox, monkeypatch):
    seen = {}

    def fake_create(text, voice, speed, lang):
        seen["speed"] = speed
        import numpy as np

        return np.zeros(100, dtype="float32"), 24000

    monkeypatch.setattr(tts, "_ENGINE", type("E", (), {"create": staticmethod(fake_create)})())
    out = json.loads(tts.speak_impl(text="Hi", speed=9.0, data_path=str(sandbox)))
    assert out["status"] == "ok"
    assert seen["speed"] == 2.0


def test_speak_strips_markdown(fake_engine, sandbox, monkeypatch):
    seen = {}

    def fake_create(text, voice, speed, lang):
        seen["text"] = text
        import numpy as np

        return np.zeros(100, dtype="float32"), 24000

    monkeypatch.setattr(tts, "_ENGINE", type("E", (), {"create": staticmethod(fake_create)})())
    out = json.loads(
        tts.speak_impl(text="**Bold** and `code`\n\nnext line", data_path=str(sandbox))
    )
    assert out["status"] == "ok"
    assert "*" not in seen["text"]
    assert "`" not in seen["text"]
    assert "next line" in seen["text"]


def test_speak_truncates_long_text(fake_engine, sandbox, monkeypatch):
    seen = {}

    def fake_create(text, voice, speed, lang):
        seen["text"] = text
        import numpy as np

        return np.zeros(100, dtype="float32"), 24000

    monkeypatch.setattr(tts, "_ENGINE", type("E", (), {"create": staticmethod(fake_create)})())
    out = json.loads(tts.speak_impl(text="x" * 9000, data_path=str(sandbox)))
    assert out["status"] == "ok"
    assert len(seen["text"]) == tts.MAX_CHARS


def test_missing_dependency_is_a_json_error(monkeypatch, sandbox):
    monkeypatch.setattr(tts, "_ENGINE", None)
    import sys

    saved = sys.modules.get("kokoro_onnx")
    monkeypatch.setitem(sys.modules, "kokoro_onnx", None)  # forces ImportError
    try:
        out = json.loads(tts.speak_impl(text="Hi", data_path=str(sandbox)))
    finally:
        if saved is not None:
            sys.modules["kokoro_onnx"] = saved
    assert out["status"] == "error"
    assert "kokoro-onnx" in out["error"]


def test_engine_failure_is_a_json_error(monkeypatch, sandbox):
    class Broken:
        def create(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(tts, "_ENGINE", Broken())
    out = json.loads(tts.speak_impl(text="Hi", data_path=str(sandbox)))
    assert out["status"] == "error"
    assert "boom" in out["error"]


def test_registered_on_tools_server():
    import importlib

    import onit.mcp.servers.tasks.tools.mcp_server as tools

    importlib.reload(tools)
    # _registered_tool_names runs its own loop via asyncio.run, which cannot
    # nest inside pytest-asyncio's; the sync fastmcp fallback path is what a
    # plain call resolves to here, so take it directly.
    names = tools._registered_tool_names()
    assert "speak" in names


def test_disabled_by_env(monkeypatch):
    import importlib

    import onit.mcp.servers.tasks.tools.mcp_server as tools

    monkeypatch.setenv("ONIT_DISABLE_TTS", "1")
    importlib.reload(tools)
    names = tools._registered_tool_names()
    assert "speak" not in names
    monkeypatch.delenv("ONIT_DISABLE_TTS")
    importlib.reload(tools)


# ─── Read-aloud summarization ────────────────────────────────────────────────


class FakeSummarizer:
    """Stand-in for the model behind ``make_ask``.

    Records what it was asked and replies with a canned summary, so tests
    can assert on the exchange without an endpoint.
    """

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def __call__(self, config_data=None, timeout=None, model=None, endpoint=None):
        self.config_data = config_data
        self.timeout = timeout
        self.model = model
        self.endpoint = endpoint

        async def ask(messages):
            self.calls.append(messages)
            return self.reply

        return ask


@pytest.fixture()
def patch_make_ask(monkeypatch):
    """Point ``make_ask`` at a FakeSummarizer; returns the factory."""

    def install(reply):
        fake = FakeSummarizer(reply)
        import onit.learn.llm as llm

        monkeypatch.setattr(llm, "make_ask", fake)
        return fake

    return install


LONG_TEXT = (
    "The weather report for Tokyo on December 10 2026 shows a detailed breakdown. " * 12
).strip()


@pytest.mark.asyncio
async def test_short_text_skips_the_model():
    short = "Today is sunny with a max of 30 degrees."
    assert await tts.summarize_for_speech(short) == short


@pytest.mark.asyncio
async def test_long_text_is_summarized_and_cleaned(patch_make_ask):
    fake = patch_make_ask("Summary: Today is sunny with a maximum of 30 degrees.")
    out = await tts.summarize_for_speech(LONG_TEXT, {"serving": {"model": "qwen"}})
    assert out == "Today is sunny with a maximum of 30 degrees."
    # The exchange the model saw: system prompt first, full text as the user.
    assert fake.calls[0][0]["role"] == "system"
    assert fake.calls[0][1]["content"] == LONG_TEXT
    # The agent's own config and model reached the client builder.
    assert fake.config_data == {"serving": {"model": "qwen"}}
    assert fake.timeout == tts.SUMMARY_TIMEOUT_S
    assert fake.model is None


@pytest.mark.asyncio
async def test_agent_model_is_passed_through(patch_make_ask):
    fake = patch_make_ask("Today is sunny with a maximum of 30 degrees.")
    await tts.summarize_for_speech(LONG_TEXT, {}, model="z-ai/glm-5.3-flash")
    assert fake.model == "z-ai/glm-5.3-flash"


@pytest.mark.asyncio
async def test_endpoint_is_passed_through(patch_make_ask):
    """A caller-known endpoint reaches the model call untouched.

    Without it a ``serving.endpoints`` list config re-resolves to the
    localhost default, where nothing listens - the bug that made every
    summary silently fall back to the full text.
    """
    fake = patch_make_ask("Today is sunny with a maximum of 30 degrees.")
    endpoint = {"host": "https://openrouter.ai/api/v1", "model": "m", "host_key": "k"}
    await tts.summarize_for_speech(LONG_TEXT, {}, endpoint=endpoint)
    assert fake.endpoint == endpoint


def test_resolve_endpoint_prefers_the_caller_known_one():
    from onit.learn.llm import resolve_endpoint

    # A serving.endpoints list config has no serving.host key; the explicit
    # endpoint must win over the localhost default it would otherwise fall
    # back to.
    resolved = resolve_endpoint(
        {"serving": {"endpoints": [{"host": "https://x", "model": "a"}]}},
        endpoint={"host": "https://openrouter.ai/api/v1", "model": "m", "host_key": "k"},
    )
    assert resolved == {
        "host": "https://openrouter.ai/api/v1",
        "model": "m",
        "host_key": "k",
    }
    # No endpoint given: the old behavior stands.
    assert resolve_endpoint({})["host"] == "http://127.0.0.1:8001/v1"


@pytest.mark.asyncio
async def test_fenced_summary_is_unwrapped(patch_make_ask):
    patch_make_ask("```text\nToday is sunny with a maximum of 30 degrees.\n```")
    out = await tts.summarize_for_speech(LONG_TEXT)
    assert out == "Today is sunny with a maximum of 30 degrees."


@pytest.mark.asyncio
async def test_model_failure_speaks_the_full_text(monkeypatch):
    import onit.learn.llm as llm

    async def boom(config_data=None, timeout=None):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(llm, "make_ask", boom)
    assert await tts.summarize_for_speech(LONG_TEXT) == LONG_TEXT


@pytest.mark.asyncio
async def test_junk_summary_speaks_the_full_text(patch_make_ask):
    patch_make_ask("ok")
    assert await tts.summarize_for_speech(LONG_TEXT) == LONG_TEXT


@pytest.mark.asyncio
async def test_summary_as_long_as_the_text_speaks_the_full_text(patch_make_ask):
    patch_make_ask(LONG_TEXT + " and a little more on top of that.")
    assert await tts.summarize_for_speech(LONG_TEXT) == LONG_TEXT


@pytest.mark.asyncio
async def test_overlong_summary_is_capped_on_a_sentence(patch_make_ask):
    patch_make_ask("Tokyo will be sunny with a high of 30 degrees and light winds all week. " * 10)
    out = await tts.summarize_for_speech(LONG_TEXT)
    assert len(out) <= tts.SUMMARY_MAX_CHARS
    assert out.endswith(".")


@pytest.mark.asyncio
async def test_structured_summary_speaks_the_full_text(patch_make_ask):
    """A model that answers with a table has reformatted, not summarized."""
    patch_make_ask("| Property | Price |\n|---|---|\n| Unplan | 3,200 yen |\n| Oak | 3,600 yen |")
    assert await tts.summarize_for_speech(LONG_TEXT) == LONG_TEXT
