"""
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Full-duplex speech-to-speech for OnIt, bridged to NVIDIA NemotronLabs VoiceChat.

The VoiceChat NIM container serves an 11B end-to-end speech model over an
OpenAI-Realtime-compatible WebSocket.  It hears, thinks and speaks in one
model — OnIt supplies no ASR, no TTS, no VAD and no turn-taking policy.  What
it supplies is *work*: the MCP tool registry and the agent loop behind
``OnIt.process_task``.

So this module is a bridge, and it sits in the middle on purpose:

    browser  <--ws-->  OnIt /api/voice  <--ws-->  VoiceChat container
                            |
                            +-- ask_onit --> process_task --> MCP tools

The browser never reaches the GPU server directly.  Everything crosses OnIt so
that Google auth, session ownership and the per-session ``data_path`` jail keep
holding for a voice call exactly as they do for a typed one.

Two documented constraints of the model shape the rest of the design:

1. At most five tools per session, and no parallel calls.  OnIt's registry is
   far larger than that, and its value is the *multi-step* loop, not any single
   tool.  So the voice model is handed a three-tool facade whose centrepiece is
   ``ask_onit`` — one door through which the whole agent is reachable.

2. "User cannot interrupt the agent during tool calling execution."  An OnIt
   task can run for a minute.  Barge-in across that window is therefore ours to
   implement, not the model's: the client pump keeps measuring inbound audio
   while a tool runs and cancels the task through the same safety queue the
   web UI's stop button uses.
"""

import array
import asyncio
import base64
import json
import logging
import math
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from src.lib.text import remove_tags

logger = logging.getLogger(__name__)


# ── Wire constants ──────────────────────────────────────────────────────────

# Event names the container speaks.  Collected here rather than sprinkled
# through the pumps because the realtime API reports itself as version 0.2.0 —
# a rename upstream should be a one-line change down here.
EV = {
    # client -> server
    "audio_append":   "input_audio_buffer.append",
    "session_update": "session.update",
    "item_create":    "conversation.item.create",
    "session_close":  "session.close",
    # server -> client
    "session_created":   "session.created",
    "session_updated":   "session.updated",
    "session_end":       "session.end",
    "speech_started":    "input_audio_buffer.speech_started",
    "speech_stopped":    "input_audio_buffer.speech_stopped",
    "response_created":  "response.created",
    "audio_delta":       "response.output_audio.delta",
    "audio_done":        "response.output_audio.done",
    "agent_text_delta":  "response.output_audio_transcript.delta",
    "user_text_delta":   "conversation.item.input_audio_transcription.delta",
    "user_text_done":    "conversation.item.input_audio_transcription.completed",
    "tool_call":         "response.function_call_arguments.done",
    "error":             "error",
}

DEFAULT_URL = "ws://localhost:9100/v1/realtime"

# PCM 16-bit mono little-endian, 24 kHz on the wire both ways (the server
# resamples to 16 kHz internally).  3840 bytes is the ~80 ms frame the
# reference client sends; anything much larger costs latency, anything much
# smaller costs syscalls.
SAMPLE_RATE = 24000
FRAME_BYTES = 3840

# The model requires ASCII-only instructions and tool responses.  Everything
# spoken to it passes through speakable() first.
MAX_SPOKEN_CHARS = 600

# Deliberately not OnIt's full agent instruction (onit.py:_assistant_instruction).
# That one is long, and it describes a tool registry this model cannot see —
# handing it over would only teach the voice model to hallucinate tools.  This
# prompt has one job: keep the conversation moving and route real work to the
# one door that leads to it.  ASCII only, per the model card.
DEFAULT_SYSTEM_PROMPT = (
    "You are OnIt, a spoken assistant. You are talking with the user out loud, "
    "so keep every reply short and natural - one or two sentences, no lists, no "
    "markdown, no code. "
    "You cannot search, read files, run commands or look anything up yourself. "
    "For any question about facts, files, code, the web, or anything that needs "
    "work done, call the ask_onit tool and pass the user's request in your own "
    "words. Answer directly only when the reply is pure conversation. "
    "When a tool result comes back, say it plainly in your own words. "
    "If the user starts speaking while you are talking, stop and listen."
)

# Spoken the moment the model commits to a tool call, so the user is not left
# in silence while the agent works.  Folded into the instructions rather than
# sent as a protocol field: the on-hold mechanism is documented by behaviour,
# not by a stable key name, and a wrong key risks a session.update rejection.
DEFAULT_ON_HOLD = "Give me a moment while I look that up."

CANCELLED_MESSAGE = "The user interrupted, so the task was cancelled."
TIMEOUT_MESSAGE = "That took too long, so I stopped it. Ask me to try again."


# ── Text for the ear ────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_BARE_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_RULE_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$", re.MULTILINE)
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_WS_RE = re.compile(r"\s+")

# Characters NFKD leaves alone but that still fall outside ASCII.
_PUNCT_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": " - ", "…": "...", "•": " ",
    " ": " ", "·": " ", "→": " to ", "×": " x ",
    "™": "", "®": "", "©": "",
}

_CODE_STANDIN = " I put the code in the transcript. "
_LINK_STANDIN = " a link in the transcript "


def speakable(text: str, limit: int = MAX_SPOKEN_CHARS) -> str:
    """Reduce an OnIt answer to something worth reading aloud, in ASCII.

    Two separate requirements meet here.  The model card requires tool
    responses to be pure ASCII, and OnIt's answers are markdown carrying
    emoji, code fences, tables and URLs — the empty-response fallback in
    onit.py is literally "I am sorry \U0001f614".  Sent raw, that is either a
    protocol violation or a minute of the agent spelling out a URL.

    The full answer is not lost: the caller ships it to the browser transcript
    unmodified, which is where code and links belong anyway.
    """
    if not text:
        return ""

    # Fences first — they routinely contain angle brackets that remove_tags
    # would otherwise chew through, taking real code with them.
    out = _FENCE_RE.sub(_CODE_STANDIN, text)
    out = remove_tags(out)
    out = _TABLE_RE.sub(" ", out)
    out = _RULE_RE.sub(" ", out)
    out = _INLINE_CODE_RE.sub(r"\1", out)
    # Keep the label, drop the target: nobody wants a URL read out.
    out = _MD_LINK_RE.sub(r"\1", out)
    out = _BARE_URL_RE.sub(_LINK_STANDIN, out)
    out = _HEADING_RE.sub("", out)
    out = _BULLET_RE.sub("", out)
    out = _EMPHASIS_RE.sub(r"\2", out)

    for src, dst in _PUNCT_MAP.items():
        out = out.replace(src, dst)
    # NFKD splits accents from their letters so the letters survive the
    # ASCII encode below; emoji have no decomposition and simply drop out.
    out = unicodedata.normalize("NFKD", out)
    out = out.encode("ascii", "ignore").decode("ascii")
    out = _WS_RE.sub(" ", out).strip()

    return _truncate_spoken(out, limit)


def _truncate_spoken(text: str, limit: int) -> str:
    """Cut to *limit* on a sentence boundary where one is close enough.

    Cutting mid-clause is worse than cutting early: the TTS decoder carries
    the dangling intonation and the reply sounds like a dropped call.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    window = text[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut > limit * 0.5:
        return window[:cut + 1]
    # No usable sentence end: break on a word and mark the cut. The ellipsis
    # has to fit inside the limit too — the caller's budget is the budget.
    window = text[:max(limit - 3, 0)]
    cut = window.rfind(" ")
    return (window[:cut] if cut > 0 else window).rstrip() + "..."


def rms_pcm16(raw: bytes) -> float:
    """Root-mean-square level of a PCM16 frame, on the int16 scale.

    Hand-rolled on stdlib ``array`` rather than numpy: this runs on every
    80 ms frame of every call, and the bridge should not drag a numeric stack
    into the web UI's dependency set for one dot product.
    """
    if len(raw) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(raw[:len(raw) - (len(raw) % 2)])
    if not samples:
        return 0.0
    total = 0
    for s in samples:
        total += s * s
    return math.sqrt(total / len(samples))


# ── Configuration ───────────────────────────────────────────────────────────

@dataclass
class VoiceConfig:
    """Everything the `voice:` block in the config controls."""
    enabled: bool = False
    url: str = DEFAULT_URL
    sample_rate: int = SAMPLE_RATE
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    on_hold_message: str = DEFAULT_ON_HOLD
    max_spoken_chars: int = MAX_SPOKEN_CHARS
    tool_timeout: float = 90.0
    barge_in: bool = True
    # Sustained level and frame count that count as "the user started talking"
    # while a tool is running.  Five frames is ~400 ms — long enough to ignore
    # a chair creak, short enough that cancelling still feels immediate.
    barge_in_rms: float = 900.0
    barge_in_frames: int = 5
    # The model is prone to continuing on its own; cap how many turns it may
    # take without the user having said anything in between.
    max_unprompted_turns: int = 3
    connect_timeout: float = 20.0

    @classmethod
    def from_config(cls, config_data: dict | None) -> "VoiceConfig":
        block = (config_data or {}).get("voice") or {}
        if not isinstance(block, dict):
            block = {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in block.items() if k in known and v is not None})

    def health_url(self) -> str:
        """The container's health endpoint, derived from the websocket URL."""
        base = self.url.replace("wss://", "https://").replace("ws://", "http://")
        base = base.split("/v1/realtime")[0].rstrip("/")
        return f"{base}/v1/realtime/health"


def tool_specs(on_hold: str = DEFAULT_ON_HOLD) -> list[dict]:
    """The facade handed to the voice model.

    Three tools, against a documented ceiling of five.  The headroom is
    deliberate — quality degrades as the list grows, and ``ask_onit`` already
    covers everything the other two do not.
    """
    return [
        {
            "name": "ask_onit",
            "description": (
                "Ask the OnIt agent to do something that needs real work: search "
                "the web, read or write files, run code, look through local "
                "documents, or answer any factual question. Pass the user's "
                "request as a clear sentence. This takes several seconds, so say "
                f'"{speakable(on_hold)}" before calling it.'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "What the user wants, as a full sentence.",
                    },
                },
                "required": ["request"],
            },
        },
        {
            "name": "get_current_datetime",
            "description": (
                "Get the current local date and time. Use this instead of "
                "ask_onit when the user only wants the time or date."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "stop_current_task",
            "description": (
                "Cancel the work started by ask_onit. Call this when the user "
                "says stop, cancel, never mind, or forget it."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    ]


# ── The bridge ──────────────────────────────────────────────────────────────

class VoiceBridge:
    """One live call: a browser socket, a container socket, and OnIt between.

    Instantiated per connection by the ``/api/voice`` route.  The caller owns
    authentication and session resolution; by the time a bridge exists the
    session is already the same ``ApiSession`` the typed UI uses, so voice and
    text share history, files and the stop button.
    """

    def __init__(self, ui, session, client_ws, config: VoiceConfig,
                 ws_connect: Optional[Callable] = None) -> None:
        self.ui = ui
        self.session = session
        self.client = client_ws
        self.cfg = config
        # Injectable so the tests can point a bridge at a fake container
        # without monkeypatching aiohttp module-wide.
        self._ws_connect = ws_connect

        self.upstream = None
        self._http = None

        # Every write to the browser socket funnels through this queue and a
        # single writer task.  Two reasons: a Starlette WebSocket is not safe
        # for concurrent sends, and process_task's callbacks fire on OnIt's
        # loop, not this one, so they need a thread-safe way in.
        self._out: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._client_loop: Optional[asyncio.AbstractEventLoop] = None

        self._closing = False
        self._task_running = False
        self._task_future = None
        self._loud_frames = 0
        self._cancelled = False
        self._tool_tasks: set = set()

        # Transcript of the turn in flight, flushed to the session JSONL when
        # the agent stops speaking.
        self._user_text: list[str] = []
        self._agent_text: list[str] = []
        self._turn_used_tool = False
        self._unprompted_turns = 0

    # ── lifecycle ──────────────────────────────────────────────────────

    async def run(self) -> None:
        """Open the call and pump it until either side hangs up."""
        self._client_loop = asyncio.get_running_loop()
        writer = asyncio.create_task(self._pump_out())
        try:
            await self._connect_upstream()
        except Exception as exc:
            logger.error("voice: cannot reach %s: %s", self.cfg.url, exc)
            await self._send_client("error", {
                "message": "The voice service is not reachable. "
                           "Check that the VoiceChat container is running."
            })
            self._closing = True
            await self._drain(writer)
            return

        try:
            await self._send_session_update()
            await self._send_client("ready", {"sample_rate": self.cfg.sample_rate,
                                              "frame_bytes": FRAME_BYTES})
            self.ui.add_log("Voice call started")
            done, pending = await asyncio.wait(
                {asyncio.create_task(self._pump_client()),
                 asyncio.create_task(self._pump_server())},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                # Surface a pump that died on an exception rather than an EOF.
                exc = task.exception()
                if exc and not isinstance(exc, asyncio.CancelledError):
                    logger.error("voice: pump failed: %s", exc)
        finally:
            self._closing = True
            await self._cancel_running_task()
            await self._drain(writer)
            await self._close_upstream()
            self.ui.add_log("Voice call ended")

    async def _connect_upstream(self) -> None:
        if self._ws_connect is not None:
            self.upstream = await self._ws_connect(self.cfg.url)
            return
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "The voice bridge needs aiohttp. Install it with: "
                "pip install 'onit[voice]'"
            ) from exc
        self._http = aiohttp.ClientSession()
        # max_msg_size=0 lifts the 4 MB cap: audio frames are small, but a
        # long transcript event on a slow drain can coalesce.
        self.upstream = await asyncio.wait_for(
            self._http.ws_connect(self.cfg.url, max_msg_size=0, heartbeat=20.0),
            timeout=self.cfg.connect_timeout,
        )

    async def _close_upstream(self) -> None:
        try:
            if self.upstream is not None:
                await self._send_upstream({"type": EV["session_close"]})
                await self.upstream.close()
        except Exception:
            pass
        try:
            if self._http is not None:
                await self._http.close()
        except Exception:
            pass

    async def _drain(self, writer: asyncio.Task) -> None:
        """Let queued frames reach the browser, then stop the writer."""
        try:
            await asyncio.wait_for(self._out.join(), timeout=2.0)
        except Exception:
            pass
        writer.cancel()
        try:
            await writer
        except (asyncio.CancelledError, Exception):
            pass

    # ── browser -> container ───────────────────────────────────────────

    async def _pump_client(self) -> None:
        """Relay mic frames upstream, and watch them for barge-in."""
        while not self._closing:
            try:
                msg = await self.client.receive_json()
            except Exception:
                return  # browser closed the tab or the socket dropped
            mtype = msg.get("type")

            if mtype == "audio":
                b64 = msg.get("data") or ""
                if not b64:
                    continue
                self._watch_for_barge_in(b64)
                await self._send_upstream({
                    "type": EV["audio_append"], "audio": b64,
                })
            elif mtype == "stop":
                # The composer's stop button, pressed mid-call.
                await self._cancel_running_task()
                await self._send_client("barge_in", {"reason": "stop"})
            elif mtype == "bye":
                return
            elif mtype == "ping":
                await self._send_client("pong", {})

    def _watch_for_barge_in(self, b64: str) -> None:
        """Cancel a running agent task once the user talks over it.

        This is the documented hole in the model: while a tool call is
        outstanding it will not accept an interruption, and an OnIt task can
        easily outlast the user's patience.  So the level is measured here and
        the cancellation goes through ``safety_queue`` — the very mechanism
        behind /api/chat/stop — which unwinds the agent loop and stops the
        sandbox container on its way out.
        """
        if not (self.cfg.barge_in and self._task_running) or self._cancelled:
            return
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception:
            return
        if rms_pcm16(raw) >= self.cfg.barge_in_rms:
            self._loud_frames += 1
        else:
            self._loud_frames = 0
        if self._loud_frames >= self.cfg.barge_in_frames:
            logger.info("voice: barge-in during tool call, cancelling task")
            self._request_cancel()
            self._send_client_nowait("barge_in", {"reason": "speech"})
            self._send_client_nowait("status", {"text": ""})

    # ── container -> browser ───────────────────────────────────────────

    async def _pump_server(self) -> None:
        """Translate container events onto OnIt's wire."""
        async for payload in self._upstream_messages():
            etype = payload.get("type")

            if etype == EV["audio_delta"]:
                await self._send_client("audio", {"data": payload.get("delta") or ""})

            elif etype == EV["agent_text_delta"]:
                delta = payload.get("delta") or ""
                self._agent_text.append(delta)
                await self._send_client("transcript", {
                    "role": "assistant", "delta": delta, "final": False,
                })

            elif etype == EV["user_text_delta"]:
                await self._send_client("transcript", {
                    "role": "user", "delta": payload.get("delta") or "", "final": False,
                })

            elif etype == EV["user_text_done"]:
                text = payload.get("transcript") or "".join(self._user_text)
                self._user_text = [text]
                self._unprompted_turns = 0
                await self._send_client("transcript", {
                    "role": "user", "text": text, "final": True,
                })

            elif etype == EV["speech_started"]:
                # The browser is holding up to a second of queued playback.
                # Draining it would have the agent talking over the user, so
                # the client flushes the ring buffer outright.
                self._loud_frames = 0
                await self._send_client("barge_in", {"reason": "speech_started"})

            elif etype == EV["speech_stopped"]:
                await self._send_client("speech_stopped", {})

            elif etype == EV["response_created"]:
                self._agent_text = []
                self._turn_used_tool = False
                if not self._user_text:
                    self._unprompted_turns += 1
                    if self._unprompted_turns > self.cfg.max_unprompted_turns:
                        # Documented failure mode: the model can keep starting
                        # turns with no user input and talk to itself.
                        logger.warning("voice: runaway continuation, ending call")
                        await self._send_client("error", {
                            "message": "Ending the call — the agent kept talking "
                                       "without input."})
                        return

            elif etype == EV["audio_done"]:
                await self._send_client("transcript", {
                    "role": "assistant", "text": "".join(self._agent_text), "final": True,
                })
                self._record_turn()

            elif etype == EV["tool_call"]:
                # Dispatched off-pump: a tool can run for a minute and the pump
                # must stay free to carry audio in the meantime. Held in a set
                # because asyncio keeps only a weak reference to a bare task —
                # an unreferenced one can be collected mid-tool-call.
                task = asyncio.create_task(self._dispatch_tool(payload))
                self._tool_tasks.add(task)
                task.add_done_callback(self._tool_tasks.discard)

            elif etype == EV["error"]:
                message = payload.get("error", {}).get("message") or str(payload)
                logger.error("voice: server error: %s", message)
                self.ui.add_log(f"Voice error: {message}", level="error")
                await self._send_client("error", {"message": message})

            elif etype == EV["session_end"]:
                return

    async def _upstream_messages(self):
        """Yield decoded JSON events, whichever websocket client is in use."""
        async for msg in self.upstream:
            if self._closing:
                return
            data = getattr(msg, "data", msg)
            if not isinstance(data, (str, bytes)):
                return  # close/error frame
            try:
                yield json.loads(data)
            except (ValueError, TypeError):
                logger.debug("voice: dropping non-JSON frame")

    # ── tools ──────────────────────────────────────────────────────────

    async def _dispatch_tool(self, payload: dict) -> None:
        call_id = payload.get("call_id") or payload.get("id") or ""
        name = payload.get("name") or ""
        raw_args = payload.get("arguments")
        args: dict = {}
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
                args = parsed if isinstance(parsed, dict) else {"request": str(parsed)}
            except ValueError:
                # The model emits arguments as a string and does not always
                # close the JSON.  A bare string is still a usable request.
                args = {"request": raw_args}

        logger.info("voice: tool %s(%s)", name, ", ".join(args))
        try:
            if name == "ask_onit":
                output = await self._ask_onit(str(args.get("request") or "").strip())
            elif name == "get_current_datetime":
                output = datetime.now().strftime("It is %A, %B %d, %Y at %I:%M %p.")
            elif name == "stop_current_task":
                await self._cancel_running_task()
                output = "Stopped."
            else:
                output = f"There is no tool called {name}."
        except Exception as exc:
            logger.error("voice: tool %s failed: %s", name, exc)
            output = "That did not work. Please try asking a different way."

        await self._send_upstream({
            "type": EV["item_create"],
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": speakable(output, self.cfg.max_spoken_chars),
            },
        })

    async def _ask_onit(self, request: str) -> str:
        """Run the full agent loop for one spoken request.

        ``process_task`` lives on OnIt's event loop, which is not this one —
        uvicorn runs the web server on its own loop in a background thread.
        So the call is handed across and awaited as a future, and the status
        callbacks come back the other way through the thread-safe send.
        """
        if not request:
            return "I did not catch that. Could you say it again?"
        if self._onit is None or self.ui._loop is None:
            return "The agent is not ready yet."

        self._turn_used_tool = True
        self._task_running = True
        self._cancelled = False
        self._loud_frames = 0
        self.session.processing = True
        # A stop pressed before this task must not cancel it.
        while not self.session.safety_queue.empty():
            try:
                self.session.safety_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        started = time.monotonic()
        try:
            coro = self._onit.process_task(
                request,
                session_path=self.session.session_path,
                data_path=self.session.data_path,
                safety_queue=self.session.safety_queue,
                tool_status_callback=self._on_tool_status,
                tool_result_callback=self._on_tool_result,
                session_id=self.session.session_id,
            )
            self._task_future = asyncio.run_coroutine_threadsafe(coro, self.ui._loop)
            response = await asyncio.wait_for(
                asyncio.wrap_future(self._task_future),
                timeout=self.cfg.tool_timeout,
            )
        except asyncio.TimeoutError:
            self._request_cancel()
            return TIMEOUT_MESSAGE
        except Exception as exc:
            logger.error("voice: ask_onit failed: %s", exc)
            return "Something went wrong while I was working on that."
        finally:
            self._task_running = False
            self._task_future = None
            self.session.processing = False
            await self._send_client("status", {"text": ""})

        if self._cancelled:
            return CANCELLED_MESSAGE

        elapsed = time.monotonic() - started
        logger.info("voice: ask_onit answered in %.1fs", elapsed)

        # The spoken form is a summary; the transcript gets the real answer,
        # links and generated files included, exactly as typed chat would.
        display, file_paths = self.ui._extract_file_paths(
            response, data_path=self.session.data_path,
            session_id=self.session.session_id,
        )
        await self._send_client("answer", {
            "content": display,
            "files": self.ui._file_infos(file_paths, self.session.session_id),
            "elapsed": round(elapsed, 1),
        })
        return response

    def _on_tool_status(self, text: str) -> None:
        """Called from OnIt's loop while tools run — hop threads to send."""
        self._send_client_threadsafe("status", {"text": text or ""})

    def _on_tool_result(self, _name: str, result: str) -> None:
        """Tool output is grounded source material for this session."""
        try:
            self.ui._record_email_sources(self.session, result or "")
        except Exception:
            pass

    async def _cancel_running_task(self) -> None:
        if self._task_running:
            self._request_cancel()

    def _request_cancel(self) -> None:
        """Fill the session's safety queue, the way the stop button does."""
        if self._cancelled or not self._task_running:
            return
        self._cancelled = True
        loop = getattr(self.ui, "_loop", None)
        if loop is not None:
            loop.call_soon_threadsafe(self.session.safety_queue.put_nowait, True)

    # ── history ────────────────────────────────────────────────────────

    def _record_turn(self) -> None:
        """Append a spoken turn to the session JSONL.

        Turns that went through ask_onit are already there — process_task
        writes its own line — so only the purely conversational ones need
        recording here.  The point is that a call and a typed chat leave one
        continuous history behind them.
        """
        user = "".join(self._user_text).strip()
        agent = "".join(self._agent_text).strip()
        self._user_text = []
        self._agent_text = []
        if self._turn_used_tool or not (user and agent):
            return
        try:
            with open(self.session.session_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "task": user, "response": agent, "timestamp": time.time(),
                }) + "\n")
            from src.sessions import update_session
            update_session(self.session.session_id, task=user,
                           sessions_dir=self.ui._sessions_dir())
        except Exception as exc:
            logger.debug("voice: could not record turn: %s", exc)

    # ── plumbing ───────────────────────────────────────────────────────

    @property
    def _onit(self):
        return getattr(self.ui, "_onit", None)

    async def _send_session_update(self) -> None:
        prompt = speakable(self.cfg.system_prompt, limit=0)
        on_hold = speakable(self.cfg.on_hold_message, limit=0)
        if on_hold:
            prompt = f'{prompt} When you call ask_onit, first say: "{on_hold}"'
        await self._send_upstream({
            "type": EV["session_update"],
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": self.cfg.sample_rate}},
                "output": {"format": {"type": "audio/pcm", "rate": self.cfg.sample_rate}},
            },
            "instructions": prompt,
            "tools": tool_specs(self.cfg.on_hold_message),
        })

    async def _send_upstream(self, message: dict) -> None:
        if self.upstream is None:
            return
        message.setdefault("event_id", str(uuid.uuid4()))
        try:
            await self.upstream.send_str(json.dumps(message))
        except Exception as exc:
            logger.debug("voice: upstream send failed: %s", exc)

    async def _send_client(self, etype: str, payload: dict) -> None:
        try:
            self._out.put_nowait({"type": etype, **payload})
        except asyncio.QueueFull:
            # Audio is the only high-rate event, and a browser too far behind
            # to keep up is better served by a gap than by a growing backlog.
            logger.debug("voice: client queue full, dropping %s", etype)

    def _send_client_threadsafe(self, etype: str, payload: dict) -> None:
        """Enqueue from OnIt's loop (or the barge-in path) onto ours."""
        loop = self._client_loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(
                lambda: self._send_client_nowait(etype, payload))
        except RuntimeError:
            pass

    def _send_client_nowait(self, etype: str, payload: dict) -> None:
        try:
            self._out.put_nowait({"type": etype, **payload})
        except asyncio.QueueFull:
            pass

    async def _pump_out(self) -> None:
        """The single writer to the browser socket."""
        while True:
            message = await self._out.get()
            try:
                await self.client.send_json(message)
            except Exception:
                self._closing = True
                return
            finally:
                self._out.task_done()


async def check_health(cfg: VoiceConfig, timeout: float = 5.0) -> dict:
    """Ask the container whether it can take a call.

    Loading the 11B model takes minutes, and a websocket opened against a
    warming container fails in a way the browser cannot explain.  The UI gates
    the microphone button on this instead.
    """
    if not cfg.enabled:
        return {"ok": False, "status": "disabled"}
    try:
        import aiohttp
    except ImportError:
        return {"ok": False, "status": "aiohttp is not installed"}
    try:
        async with aiohttp.ClientSession() as http:
            async with http.get(cfg.health_url(),
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                body: Any = {}
                try:
                    body = await resp.json(content_type=None)
                except Exception:
                    body = {}
                ok = resp.status == 200
                return {
                    "ok": ok,
                    "status": (body or {}).get("triton_status")
                              or ("ready" if ok else f"http {resp.status}"),
                    "url": cfg.url,
                }
    except Exception as exc:
        return {"ok": False, "status": f"unreachable: {exc}", "url": cfg.url}
