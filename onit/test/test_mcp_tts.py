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
