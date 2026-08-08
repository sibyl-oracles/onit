"""A stand-in for the NemotronLabs VoiceChat container.

Speaks the same realtime event vocabulary as the real NIM container (see
src/ui/voice.py:EV) so the bridge can be exercised end to end without an 80 GB
GPU: session setup, audio relay, tool dispatch, barge-in and transcripts.

It is *not* a websocket server — it is the object ``VoiceBridge`` treats as one,
injected through the ``ws_connect`` seam. That keeps the tests free of sockets
and timing while still driving the real bridge code.
"""

import asyncio
import json
import uuid


class Frame:
    """Mimics aiohttp's WSMessage, which carries its payload in .data."""

    def __init__(self, data: str):
        self.data = data


class FakeVoiceChat:
    """One fake container session.

    Pass ``on_send`` to react to what the bridge sends upstream — that is how a
    test makes the model "decide" to call a tool.
    """

    def __init__(self, on_send=None):
        self.sent: list[dict] = []
        self.url: str | None = None
        self.closed = False
        self._on_send = on_send
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── the ws_connect seam ────────────────────────────────────────

    async def connect(self, url: str) -> "FakeVoiceChat":
        self.url = url
        # Tests drive emit() from their own thread while the bridge awaits the
        # inbox on the server's loop. asyncio.Queue is not thread-safe, so the
        # loop is captured here (connect runs on it) and every enqueue is
        # handed over with call_soon_threadsafe. Without this a test that emits
        # and then only polls — never touching the websocket again — can wait
        # forever for a loop nothing woke up.
        self._loop = asyncio.get_running_loop()
        return self

    # ── what the bridge calls ──────────────────────────────────────

    async def send_str(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        if self._on_send is not None:
            await self._on_send(message, self)

    async def close(self) -> None:
        self.closed = True
        self._inbox.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self) -> Frame:
        item = await self._inbox.get()
        if item is None:
            raise StopAsyncIteration
        return Frame(item)

    # ── what a test calls ──────────────────────────────────────────

    def emit(self, event_type: str, **fields) -> None:
        """Queue a server event for the bridge to read."""
        payload = {"type": event_type, "event_id": str(uuid.uuid4()), **fields}
        self.emit_raw(json.dumps(payload))

    def emit_raw(self, raw: str | None) -> None:
        """Queue a frame verbatim — for malformed-input tests."""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._inbox.put_nowait, raw)
        else:
            self._inbox.put_nowait(raw)

    def hang_up(self) -> None:
        self.emit_raw(None)

    def of_type(self, event_type: str) -> list[dict]:
        return [m for m in self.sent if m.get("type") == event_type]

    def first(self, event_type: str) -> dict | None:
        matches = self.of_type(event_type)
        return matches[0] if matches else None
