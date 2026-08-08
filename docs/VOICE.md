# Voice — full-duplex speech to speech

Talk to OnIt out loud, interrupt it mid-sentence, and have it run its real MCP
tools while you listen.

Speech is handled end to end by NVIDIA's
[NemotronLabs VoiceChat 11B](https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat)
— a single model that hears, reasons and speaks (fast-conformer encoder →
Nemotron-Nano-9B-v2 backbone → TTS decoder), served by a NIM container over an
OpenAI-Realtime-compatible WebSocket. OnIt supplies no ASR, no TTS, no VAD and
no turn-taking policy. It supplies the *work*.

```
browser  <--ws-->  OnIt /api/voice  <--ws-->  VoiceChat container
  mic/speaker           bridge                  (GPU, port 9100)
                          |
                          +-- ask_onit --> process_task --> MCP tools
```

The browser never reaches the GPU server. Everything crosses OnIt, so Google
login, session ownership and the per-session `data_path` jail hold for a voice
call exactly as they do for a typed one.

---

## Prerequisites

- An NVIDIA GPU with **~66 GB free**, exclusively for VoiceChat. It cannot share
  a card with the vLLM serving the text model behind `ask_onit` — plan for a
  second GPU or a second host.
- Docker with the NVIDIA runtime (see [DOCKER.md](DOCKER.md)).
- The NGC CLI (no API key needed for this model).
- `pip install 'onit[voice]'` — one package, `aiohttp`, for the websocket client.
- A browser on **HTTPS or localhost**. `getUserMedia` is unavailable on plain
  HTTP origins; for a remote deployment terminate TLS at Caddy
  (see [HTTPS_DEPLOYMENT.md](HTTPS_DEPLOYMENT.md)).

## 1. Download the model

```bash
ngc registry model download-version nim/nvidia/nemotron-labs-voicechat:1.0.0
chmod -R 777 nemotron-labs-voicechat_v1.0.0
```

## 2. Run the container

Published on **9100**, not the container's own default of 9000 — that collides
with OnIt's web UI.

```bash
docker run -it --rm --name=nemotron-labs-voicechat \
  --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
  -e NIM_HTTP_API_PORT=9000 -p 127.0.0.1:9100:9000 \
  -v $(pwd)/nemotron-labs-voicechat_v1.0.0:/data/models \
  --entrypoint /s2s/run_s2s_server.sh \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
```

Or through compose, where the service is behind an opt-in profile:

```bash
export ONIT_VOICECHAT_MODELS=/abs/path/to/nemotron-labs-voicechat_v1.0.0
docker compose --profile voice up
```

Loading an 11B model takes **minutes**. Wait for:

```bash
curl http://localhost:9100/v1/realtime/health   # {"triton_status": "ready", ...}
```

Bind it to loopback (or a private network). The realtime socket has no auth of
its own — OnIt is what stands in front of it.

## 3. Confirm the container on its own

Before involving OnIt, prove the container works with NVIDIA's own client:

```bash
docker cp nemotron-labs-voicechat:/s2s/nemotron-voicechat-client.py .
pip install websockets soundfile numpy pyaudio
python3 nemotron-voicechat-client.py --server ws://localhost:9100
```

## 4. Start OnIt

```bash
onit serve web --voice
# or against another host:
onit serve web --voice --voice-url ws://gpu-box:9100/v1/realtime
```

A microphone button appears in the composer — but only once
`/api/voice/health` reports ready, so a warming container shows no button
rather than a socket that fails for reasons the browser cannot explain.

Click it, allow the microphone, and talk. The bar above the composer shows
**Listening / Speaking**, plus **Interrupt**, **Mute** and **End**.

---

## Configuration

All keys live under `voice:` in your config (defaults in
[`src/configs/default.yaml`](../src/configs/default.yaml)):

| key | default | what it does |
|---|---|---|
| `enabled` | `false` | Draw the mic button and accept `/api/voice`. |
| `url` | `ws://localhost:9100/v1/realtime` | The container's realtime socket. |
| `sample_rate` | `24000` | PCM16 mono, both directions. |
| `system_prompt` | see below | ASCII-only voice persona. |
| `on_hold_message` | "Give me a moment…" | Spoken when a tool call starts. |
| `max_spoken_chars` | `600` | Where a spoken answer is cut. |
| `tool_timeout` | `90` | Seconds before a runaway `ask_onit` is cancelled. |
| `barge_in` | `true` | Cancel the agent when you talk over it. |
| `barge_in_rms` / `barge_in_frames` | `900` / `5` | What counts as speech (~400 ms). |
| `max_unprompted_turns` | `3` | End the call if the model talks to itself. |

## How the agent is reachable

The model documents a ceiling of **five tools per session** and cannot call
tools in parallel. OnIt's registry is far larger than that, and its value is the
*multi-step* loop rather than any one tool. So the voice model is handed a
three-tool facade:

| tool | what it does |
|---|---|
| `ask_onit(request)` | Runs the full agent loop — the whole registry behind one door. |
| `get_current_datetime()` | The commonest question, answered without a round trip. |
| `stop_current_task()` | "Stop", "never mind", "forget it". |

`ask_onit` runs `OnIt.process_task` against the same session as the typed UI, so
a voice call and a chat share one history, one working directory and one set of
generated files.

**An answer read aloud has to be short.** So the spoken form is sanitised to
ASCII (the model card requires it) and cut at a sentence boundary near
`max_spoken_chars`; code fences, tables and URLs become brief stand-ins. The
full markdown answer — links, file chips, previews — still lands in the
transcript, which is where it belongs.

## Interruption

Handled at three levels, because barge-in is the part most likely to feel
broken:

1. **While the agent speaks.** The model is full duplex and the server decides.
   The client is the failure point: the browser holds up to a second of queued
   audio, so on `barge_in` the playback ring buffer is *discarded*, not drained.
2. **While a tool runs.** NVIDIA documents that the model cannot be interrupted
   during tool execution — which is exactly the window an agent task occupies.
   OnIt therefore measures the microphone itself and, on sustained speech, fills
   the session's safety queue: the same mechanism as the stop button, which
   unwinds the agent loop and stops the sandbox container on its way out.
3. **Explicitly.** The **Interrupt** button does both at once. **Mute** gates
   the microphone at the worklet for noisy rooms.

## Known limitations

From NVIDIA's model card, and worth knowing before you judge the integration:

- Not suitable for noisy or reverberant rooms, or where other people are
  talking nearby.
- The model may keep taking turns without user input (self-talk). OnIt ends the
  call after `max_unprompted_turns`; browser echo cancellation is on and is the
  other half of the defence.
- User transcription can drop leading or mid-phrase words even on clear audio.
- Spoken output can end early, or loop on a word or sentence.
- Backchannelling ("mm-hm", "right") is not handled systematically.
- The audio context window is 2 minutes.

## Troubleshooting

**No mic button.** `voice.enabled` is off, or health is not green. Check
`curl http://localhost:9100/v1/realtime/health` and the browser console.

**Button appears, call fails immediately.** Check OnIt's log for
`voice: cannot reach`. From inside a container, `localhost` is the container —
use the compose service name (`ws://voicechat:9000/v1/realtime`).

**"Microphone access was denied."** Non-localhost origins need HTTPS.

**The agent answers itself.** Echo cancellation is off, or output is on speakers
loud enough to defeat it. Use headphones.

**The agent talks over you.** Barge-in flushes on `speech_started`; if audio
continues, the worklet failed to load — check for a
`/static/pcm-worklet.js` 404.

## Testing without a GPU

`src/test/fake_voicechat.py` is a protocol-accurate stand-in that speaks the
same event vocabulary as the container, so the bridge, tool dispatch, barge-in,
sanitiser and auth are all covered without hardware:

```bash
python -m pytest src/test/test_voice_bridge.py -q   # from the repo root
```

Run from the repo root, not `src/` — `src/mcp` shadows the installed `mcp`
package and the failure looks like an unrelated import error.
