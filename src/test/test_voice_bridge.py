"""Tests for src/ui/voice.py — the full-duplex speech-to-speech bridge.

Everything here runs against fake_voicechat.FakeVoiceChat, so no GPU and no
container are involved. What is real is the bridge, the FastAPI websocket route
and the auth in front of it.
"""

import array
import asyncio
import base64
import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starlette.testclient import TestClient

from test.fake_voicechat import FakeVoiceChat
from ui.api import WebApiUI
from ui.voice import (
    CANCELLED_MESSAGE,
    EV,
    FRAME_BYTES,
    VoiceConfig,
    rms_pcm16,
    speakable,
    tool_specs,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

class FakeVoiceOnit:
    """Stub agent whose process_task can be made slow and cancellable."""

    def __init__(self, response="The answer is 42.", delay=0.0, config_data=None):
        self.response = response
        self.delay = delay
        self.calls = []
        self.cancelled = False
        self.config_data = config_data if config_data is not None else {}

    async def process_task(self, task, session_path=None, data_path=None,
                           safety_queue=None, tool_status_callback=None,
                           tool_result_callback=None, session_id=None, **kwargs):
        self.calls.append(task)
        if tool_status_callback:
            tool_status_callback("web_search(query)")
        # Poll the safety queue the way the real agent loop does, so a stop
        # request lands as a short-circuit rather than a hard cancellation.
        waited = 0.0
        while waited < self.delay:
            if safety_queue is not None and not safety_queue.empty():
                self.cancelled = True
                return ""
            await asyncio.sleep(0.02)
            waited += 0.02
        if tool_status_callback:
            tool_status_callback("")
        return self.response


def pcm_frame(amplitude: int, samples: int = FRAME_BYTES // 2) -> str:
    """A base64 PCM16 frame at a fixed amplitude (alternating sign)."""
    data = array.array("h", [amplitude if i % 2 == 0 else -amplitude
                             for i in range(samples)])
    return base64.b64encode(data.tobytes()).decode("ascii")


SILENCE = pcm_frame(0)
SPEECH = pcm_frame(6000)


@pytest.fixture
def bg_loop():
    """A running event loop on a background thread (stands in for OnIt's)."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)


@pytest.fixture
def make_ui(tmp_path, bg_loop):
    """Build a voice-enabled UI wired to a fake container."""
    def _make(on_send=None, onit=None, **voice_overrides):
        voice = {"enabled": True, "url": "ws://fake:9100/v1/realtime"}
        voice.update(voice_overrides)
        ui = WebApiUI(
            data_path=str(tmp_path / "data"),
            session_path=str(tmp_path / "sessions" / "current.jsonl"),
            title="Test Chat",
            require_auth=False,
            voice=voice,
        )
        ui._onit = onit or FakeVoiceOnit(config_data={
            "learn": {"autonomy": "observe", "path": str(tmp_path / "learned")}})
        ui._loop = bg_loop
        server = FakeVoiceChat(on_send=on_send)
        ui._voice_ws_connect = server.connect
        ui.build_app()
        return ui, server
    return _make


def read_until(ws, wanted, limit=60):
    """Collect client messages until one of *wanted* types arrives."""
    seen = []
    for _ in range(limit):
        msg = ws.receive_json()
        seen.append(msg)
        if msg.get("type") in wanted:
            return seen
    raise AssertionError(f"never saw {wanted}; got {[m.get('type') for m in seen]}")


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ── speakable() ─────────────────────────────────────────────────────────────

class TestSpeakable:
    def test_output_is_ascii(self):
        # The model card requires ASCII-only tool responses; OnIt's own
        # empty-answer fallback carries an emoji.
        out = speakable("I am sorry \U0001f614 — café naïve “quoted”")
        assert out.isascii()
        assert "cafe" in out and "naive" in out

    def test_code_fences_become_a_stand_in(self):
        out = speakable("Here:\n```python\nprint('x')\n```\nDone.")
        assert "print" not in out
        assert "transcript" in out
        assert out.startswith("Here:")

    def test_urls_are_not_read_out(self):
        out = speakable("See https://example.com/a/very/long/path for details.")
        assert "example.com" not in out
        assert "link in the transcript" in out

    def test_markdown_link_keeps_its_label(self):
        assert speakable("Read [the paper](https://arxiv.org/abs/1) now.") == \
            "Read the paper now."

    def test_markdown_furniture_is_stripped(self):
        out = speakable("# Title\n\n- **one**\n- _two_\n\n| a | b |\n")
        assert "#" not in out and "*" not in out and "|" not in out
        assert "one" in out and "two" in out

    def test_truncates_on_a_sentence_boundary(self):
        text = "First sentence here. " * 60
        out = speakable(text, limit=100)
        assert len(out) <= 100
        assert out.endswith(".")

    def test_short_text_is_untouched(self):
        assert speakable("Sure, one moment.") == "Sure, one moment."

    def test_empty_input(self):
        assert speakable("") == ""
        assert speakable(None) == ""


class TestRms:
    def test_silence_is_zero(self):
        assert rms_pcm16(array.array("h", [0] * 100).tobytes()) == 0.0

    def test_speech_registers(self):
        assert rms_pcm16(array.array("h", [4000, -4000] * 50).tobytes()) == 4000.0

    def test_odd_length_buffer_does_not_raise(self):
        assert rms_pcm16(b"\x01\x02\x03") >= 0.0

    def test_empty_buffer(self):
        assert rms_pcm16(b"") == 0.0


class TestToolFacade:
    def test_stays_under_the_documented_ceiling(self):
        # NVIDIA documents degradation past five tools per session.
        assert len(tool_specs()) <= 5

    def test_ask_onit_is_the_delegate(self):
        spec = {t["name"]: t for t in tool_specs()}["ask_onit"]
        assert "request" in spec["parameters"]["properties"]
        assert spec["parameters"]["required"] == ["request"]

    def test_descriptions_are_ascii(self):
        for tool in tool_specs():
            assert json.dumps(tool).isascii()


class TestVoiceConfig:
    def test_health_url_derives_from_ws_url(self):
        cfg = VoiceConfig(url="ws://gpu-box:9100/v1/realtime")
        assert cfg.health_url() == "http://gpu-box:9100/v1/realtime/health"

    def test_secure_ws_maps_to_https(self):
        cfg = VoiceConfig(url="wss://voice.example.com/v1/realtime")
        assert cfg.health_url() == "https://voice.example.com/v1/realtime/health"

    def test_unknown_keys_are_ignored(self):
        cfg = VoiceConfig.from_config({"voice": {"enabled": True, "bogus": 1}})
        assert cfg.enabled is True

    def test_absent_block_leaves_voice_off(self):
        assert VoiceConfig.from_config({}).enabled is False


# ── The route ───────────────────────────────────────────────────────────────

class TestVoiceRoute:
    def test_disabled_by_default(self, tmp_path, bg_loop):
        ui = WebApiUI(data_path=str(tmp_path / "d"),
                      session_path=str(tmp_path / "s" / "c.jsonl"),
                      require_auth=False)
        ui._loop = bg_loop
        ui.build_app()
        client = TestClient(ui.app)
        assert client.get("/api/config").json()["voice_enabled"] is False
        with pytest.raises(Exception):
            with client.websocket_connect("/api/voice"):
                pass

    def test_config_advertises_voice(self, make_ui):
        ui, _ = make_ui()
        config = TestClient(ui.app).get("/api/config").json()
        assert config["voice_enabled"] is True
        assert config["voice_sample_rate"] == 24000

    def test_unauthenticated_call_is_refused(self, tmp_path, bg_loop):
        # A websocket never passes through @app.middleware("http"), so the
        # route has to check auth itself or voice becomes the way in.
        ui = WebApiUI(
            data_path=str(tmp_path / "d"),
            session_path=str(tmp_path / "s" / "c.jsonl"),
            google_client_id="cid", google_client_secret="secret",
            voice={"enabled": True},
        )
        ui._loop = bg_loop
        ui.build_app()
        client = TestClient(ui.app)
        with pytest.raises(Exception):
            with client.websocket_connect("/api/voice"):
                pass

    def test_session_update_opens_the_call(self, make_ui):
        ui, server = make_ui()
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["sample_rate"] == 24000

        update = server.first(EV["session_update"])
        assert update is not None
        assert update["audio"]["input"]["format"]["rate"] == 24000
        assert [t["name"] for t in update["tools"]] == [
            "ask_onit", "get_current_datetime", "stop_current_task"]
        assert update["instructions"].isascii()

    def test_audio_relays_both_ways(self, make_ui):
        ui, server = make_ui()
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "audio", "data": SILENCE})
            assert wait_for(lambda: server.of_type(EV["audio_append"]))
            assert server.first(EV["audio_append"])["audio"] == SILENCE

            server.emit(EV["audio_delta"], delta=SILENCE)
            msg = ws.receive_json()
            assert msg["type"] == "audio"
            assert msg["data"] == SILENCE

    def test_transcripts_reach_the_browser(self, make_ui):
        ui, server = make_ui()
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            server.emit(EV["user_text_done"], transcript="what is the time")
            server.emit(EV["agent_text_delta"], delta="It is ")
            user, agent = ws.receive_json(), ws.receive_json()
        assert user == {"type": "transcript", "role": "user",
                        "text": "what is the time", "final": True}
        assert agent["role"] == "assistant" and agent["delta"] == "It is "

    def test_speech_started_triggers_a_flush(self, make_ui):
        # The browser may be holding a second of queued agent audio; draining
        # it would talk over the user who just interrupted.
        ui, server = make_ui()
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            server.emit(EV["speech_started"])
            msg = ws.receive_json()
        assert msg["type"] == "barge_in"
        assert msg["reason"] == "speech_started"

    def test_malformed_frame_does_not_kill_the_call(self, make_ui):
        ui, server = make_ui()
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            server.emit_raw("{not json")
            server.emit(EV["speech_stopped"])
            assert ws.receive_json()["type"] == "speech_stopped"

    def test_runaway_continuation_ends_the_call(self, make_ui):
        # Documented failure mode: the model starts turns with no user input
        # and talks to itself.
        ui, server = make_ui(max_unprompted_turns=2)
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            for _ in range(4):
                server.emit(EV["response_created"])
            seen = read_until(ws, {"error"})
        assert "without input" in seen[-1]["message"]


# ── Tool dispatch ───────────────────────────────────────────────────────────

def call_tool(server, name, arguments, call_id="call-1"):
    server.emit(EV["tool_call"], call_id=call_id, name=name,
                arguments=json.dumps(arguments))


class TestToolDispatch:
    def test_ask_onit_runs_the_agent_loop(self, make_ui):
        ui, server = make_ui()
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "ask_onit", {"request": "what is the airspeed of a swallow"})
            read_until(ws, {"answer"})
            assert wait_for(lambda: server.of_type(EV["item_create"]))

        assert ui._onit.calls == ["what is the airspeed of a swallow"]
        item = server.first(EV["item_create"])["item"]
        assert item["type"] == "function_call_output"
        assert item["call_id"] == "call-1"
        assert item["output"] == "The answer is 42."

    def test_tool_output_is_ascii_and_short(self, make_ui):
        onit = FakeVoiceOnit(response="Done \U0001f389\n\n```py\nx=1\n```\n" + "word " * 400)
        ui, server = make_ui(onit=onit)
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "ask_onit", {"request": "do it"})
            assert wait_for(lambda: server.of_type(EV["item_create"]))
        output = server.first(EV["item_create"])["item"]["output"]
        assert output.isascii()
        assert len(output) <= 600
        assert "x=1" not in output

    def test_full_answer_reaches_the_transcript(self, make_ui):
        # The spoken form is a summary by necessity; the real answer, with its
        # markdown, still has to arrive somewhere.
        onit = FakeVoiceOnit(response="Full **markdown** answer with [a link](https://x.dev).")
        ui, server = make_ui(onit=onit)
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "ask_onit", {"request": "explain"})
            seen = read_until(ws, {"answer"})
        answer = seen[-1]
        assert "**markdown**" in answer["content"]
        assert "https://x.dev" in answer["content"]
        assert answer["files"] == []

    def test_tool_status_streams_while_work_happens(self, make_ui):
        ui, server = make_ui(onit=FakeVoiceOnit(delay=0.3))
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "ask_onit", {"request": "search"})
            seen = read_until(ws, {"answer"})
        statuses = [m["text"] for m in seen if m["type"] == "status"]
        assert "web_search(query)" in statuses

    def test_datetime_answers_without_the_agent(self, make_ui):
        ui, server = make_ui()
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "get_current_datetime", {})
            assert wait_for(lambda: server.of_type(EV["item_create"]))
        assert ui._onit.calls == []
        assert "It is" in server.first(EV["item_create"])["item"]["output"]

    def test_unknown_tool_gets_a_spoken_answer_not_a_crash(self, make_ui):
        ui, server = make_ui()
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "launch_missiles", {})
            assert wait_for(lambda: server.of_type(EV["item_create"]))
        assert "no tool called" in server.first(EV["item_create"])["item"]["output"]

    def test_unparseable_arguments_are_still_usable(self, make_ui):
        # The model emits arguments as a string and does not always close the
        # JSON; a bare string is still a request worth running.
        ui, server = make_ui()
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            server.emit(EV["tool_call"], call_id="c2", name="ask_onit",
                        arguments='{"request": "find the paper')
            assert wait_for(lambda: server.of_type(EV["item_create"]))
        assert ui._onit.calls == ['{"request": "find the paper']

    def test_empty_request_asks_for_a_repeat(self, make_ui):
        ui, server = make_ui()
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "ask_onit", {"request": "  "})
            assert wait_for(lambda: server.of_type(EV["item_create"]))
        assert ui._onit.calls == []
        assert "say it again" in server.first(EV["item_create"])["item"]["output"]

    def test_timeout_releases_the_call(self, make_ui):
        ui, server = make_ui(onit=FakeVoiceOnit(delay=5.0), tool_timeout=0.3)
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "ask_onit", {"request": "hang forever"})
            assert wait_for(lambda: server.of_type(EV["item_create"]), timeout=5)
        assert "too long" in server.first(EV["item_create"])["item"]["output"]


# ── Barge-in ────────────────────────────────────────────────────────────────

class TestBargeIn:
    def test_speech_during_a_tool_call_cancels_the_task(self, make_ui):
        # The model cannot be interrupted while a tool call is outstanding,
        # and an agent task occupies exactly that window — so OnIt measures
        # the microphone itself and cancels through the safety queue.
        onit = FakeVoiceOnit(delay=5.0)
        ui, server = make_ui(onit=onit, tool_timeout=10)
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "ask_onit", {"request": "a long search"})
            for _ in range(10):
                ws.send_json({"type": "audio", "data": SPEECH})
            assert wait_for(lambda: server.of_type(EV["item_create"]), timeout=5)

        assert onit.cancelled is True
        assert server.first(EV["item_create"])["item"]["output"].startswith(
            "The user interrupted")
        assert CANCELLED_MESSAGE.startswith("The user interrupted")

    def test_silence_during_a_tool_call_does_not_cancel(self, make_ui):
        onit = FakeVoiceOnit(delay=0.4)
        ui, server = make_ui(onit=onit, tool_timeout=10)
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "ask_onit", {"request": "a short search"})
            for _ in range(10):
                ws.send_json({"type": "audio", "data": SILENCE})
            assert wait_for(lambda: server.of_type(EV["item_create"]), timeout=5)
        assert onit.cancelled is False
        assert server.first(EV["item_create"])["item"]["output"] == "The answer is 42."

    def test_barge_in_can_be_turned_off(self, make_ui):
        onit = FakeVoiceOnit(delay=0.4)
        ui, server = make_ui(onit=onit, barge_in=False, tool_timeout=10)
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "ask_onit", {"request": "search"})
            for _ in range(10):
                ws.send_json({"type": "audio", "data": SPEECH})
            assert wait_for(lambda: server.of_type(EV["item_create"]), timeout=5)
        assert onit.cancelled is False

    def test_stop_message_cancels_and_flushes(self, make_ui):
        onit = FakeVoiceOnit(delay=5.0)
        ui, server = make_ui(onit=onit, tool_timeout=10)
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "ask_onit", {"request": "a long search"})
            time.sleep(0.1)
            ws.send_json({"type": "stop"})
            seen = read_until(ws, {"barge_in"})
            assert wait_for(lambda: server.of_type(EV["item_create"]), timeout=5)
        assert onit.cancelled is True
        assert seen[-1]["reason"] == "stop"

    def test_stop_current_task_tool_cancels(self, make_ui):
        onit = FakeVoiceOnit(delay=5.0)
        ui, server = make_ui(onit=onit, tool_timeout=10)
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            call_tool(server, "ask_onit", {"request": "a long search"}, call_id="a")
            time.sleep(0.1)
            call_tool(server, "stop_current_task", {}, call_id="b")
            assert wait_for(lambda: len(server.of_type(EV["item_create"])) == 2,
                            timeout=5)
        assert onit.cancelled is True


# ── History ─────────────────────────────────────────────────────────────────

class TestHistory:
    def test_spoken_turn_is_recorded(self, make_ui):
        # A call and a typed chat should leave one continuous history behind.
        ui, server = make_ui()
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            server.emit(EV["user_text_done"], transcript="hello there")
            server.emit(EV["agent_text_delta"], delta="Hi! How can I help?")
            server.emit(EV["audio_done"])
            read_until(ws, {"transcript"} , limit=10)
            time.sleep(0.2)

        sid = next(iter(ui._web_sessions))
        with open(ui._web_sessions[sid].session_path, encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
        assert lines[-1]["task"] == "hello there"
        assert lines[-1]["response"] == "Hi! How can I help?"

    def test_tool_turn_is_not_double_recorded(self, make_ui):
        # process_task already wrote its own line for this turn.
        ui, server = make_ui()
        with TestClient(ui.app).websocket_connect("/api/voice") as ws:
            ws.receive_json()
            server.emit(EV["response_created"])
            server.emit(EV["user_text_done"], transcript="find the paper")
            call_tool(server, "ask_onit", {"request": "find the paper"})
            read_until(ws, {"answer"})
            server.emit(EV["agent_text_delta"], delta="It is 42.")
            server.emit(EV["audio_done"])
            time.sleep(0.2)

        sid = next(iter(ui._web_sessions))
        with open(ui._web_sessions[sid].session_path, encoding="utf-8") as f:
            lines = [x for x in f if x.strip()]
        assert lines == []  # FakeVoiceOnit does not write; the bridge must not either
