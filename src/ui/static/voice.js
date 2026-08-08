/* OnIt voice call — browser half of the full-duplex bridge.
 *
 * Talks to /api/voice (WebSocket) in src/ui/voice.py:
 *   out: {type:"audio",data:b64} {type:"stop"} {type:"bye"}
 *   in:  ready | audio | transcript | barge_in | speech_stopped
 *        status | answer | error | end
 *
 * Audio is PCM16 mono 24 kHz both ways, ~80 ms per frame, base64 on the wire.
 * Capture and playback live in pcm-worklet.js on the audio thread; this file
 * is only plumbing and state.
 *
 * Exposed as window.OnItVoice for app.js.
 */
(function () {
  "use strict";

  const SAMPLE_RATE = 24000;

  const state = {
    ws: null,
    ctx: null,
    stream: null,
    capture: null,
    playback: null,
    active: false,
    connecting: false,
    agentSpeaking: false,
    muted: false,
    handlers: {},
  };

  function emit(event, payload) {
    const fn = state.handlers[event];
    if (fn) { try { fn(payload); } catch (e) { console.error(e); } }
  }

  // ── Encoding ───────────────────────────────────────────────────
  // Chunked so a frame never blows the argument limit of String.fromCharCode.

  function toBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let bin = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(bin);
  }

  function fromBase64(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
  }

  // ── Audio graph ────────────────────────────────────────────────

  async function openAudio() {
    // Echo cancellation is not optional here. Without it the agent hears its
    // own voice through the speakers, treats it as user speech, and answers
    // itself — which compounds the model's documented tendency to keep taking
    // turns unprompted.
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });

    // Match the wire rate so neither end resamples.
    state.ctx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: SAMPLE_RATE,
      latencyHint: "interactive",
    });
    await state.ctx.audioWorklet.addModule("/static/pcm-worklet.js?v=1");

    const src = state.ctx.createMediaStreamSource(state.stream);
    state.capture = new AudioWorkletNode(state.ctx, "onit-capture");
    state.capture.port.onmessage = (e) => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ type: "audio", data: toBase64(e.data) }));
      }
    };
    src.connect(state.capture);
    // The capture node produces no output, but an unconnected worklet is not
    // guaranteed to be pulled — route it to a muted gain so it keeps running.
    const sink = state.ctx.createGain();
    sink.gain.value = 0;
    state.capture.connect(sink).connect(state.ctx.destination);

    state.playback = new AudioWorkletNode(state.ctx, "onit-playback", {
      outputChannelCount: [1],
    });
    state.playback.port.onmessage = (e) => {
      if (e.data && e.data.type === "speaking") {
        state.agentSpeaking = e.data.value;
        emit("state", status());
      }
    };
    state.playback.connect(state.ctx.destination);

    if (state.ctx.state === "suspended") await state.ctx.resume();
  }

  function closeAudio() {
    try { if (state.capture) state.capture.disconnect(); } catch (e) {}
    try { if (state.playback) state.playback.disconnect(); } catch (e) {}
    try { if (state.stream) state.stream.getTracks().forEach((t) => t.stop()); } catch (e) {}
    try { if (state.ctx) state.ctx.close(); } catch (e) {}
    state.capture = state.playback = state.stream = state.ctx = null;
    state.agentSpeaking = false;
  }

  function flushPlayback() {
    if (state.playback) state.playback.port.postMessage({ type: "flush" });
    state.agentSpeaking = false;
    emit("state", status());
  }

  // ── Session ────────────────────────────────────────────────────

  async function start(sessionId) {
    if (state.active || state.connecting) return;
    state.connecting = true;
    emit("state", status());
    try {
      await openAudio();
    } catch (err) {
      state.connecting = false;
      closeAudio();
      emit("error", err && err.name === "NotAllowedError"
        ? "Microphone access was denied."
        : "Could not open the microphone.");
      emit("state", status());
      return;
    }

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const qs = sessionId ? "?session=" + encodeURIComponent(sessionId) : "";
    const ws = new WebSocket(proto + "//" + location.host + "/api/voice" + qs);
    state.ws = ws;

    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      handle(msg);
    };
    ws.onerror = () => emit("error", "The voice connection failed.");
    ws.onclose = (ev) => {
      const wasActive = state.active;
      state.active = false;
      state.connecting = false;
      state.ws = null;
      closeAudio();
      emit("state", status());
      if (!wasActive && ev.code === 1008) {
        emit("error", "Voice is not available on this session.");
      }
      emit("ended", {});
    };
  }

  function handle(msg) {
    switch (msg.type) {
      case "ready":
        state.active = true;
        state.connecting = false;
        emit("state", status());
        break;
      case "audio":
        if (state.playback && msg.data) {
          const buf = fromBase64(msg.data);
          state.playback.port.postMessage({ type: "audio", buffer: buf }, [buf]);
        }
        break;
      case "transcript":
        emit("transcript", msg);
        break;
      case "barge_in":
        // Drop queued agent audio outright rather than letting it drain.
        flushPlayback();
        break;
      case "status":
        emit("status", msg.text || "");
        break;
      case "answer":
        emit("answer", msg);
        break;
      case "error":
        emit("error", msg.message || "Voice error.");
        break;
      case "end":
        stop();
        break;
      default:
        break;
    }
  }

  function send(obj) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(obj));
    }
  }

  /* Stop whatever the agent is doing without ending the call: flushes the
   * speaker queue here and cancels any running agent task server-side. */
  function interrupt() {
    flushPlayback();
    send({ type: "stop" });
  }

  function stop() {
    if (!state.ws) { closeAudio(); return; }
    send({ type: "bye" });
    try { state.ws.close(); } catch (e) {}
    state.ws = null;
    state.active = false;
    state.connecting = false;
    closeAudio();
    emit("state", status());
  }

  function setMuted(value) {
    state.muted = !!value;
    if (state.capture) {
      state.capture.port.postMessage({ type: "mute", value: state.muted });
    }
    emit("state", status());
  }

  function status() {
    return {
      active: state.active,
      connecting: state.connecting,
      speaking: state.agentSpeaking,
      muted: state.muted,
    };
  }

  window.OnItVoice = {
    start, stop, interrupt, setMuted, status,
    on(event, fn) { state.handlers[event] = fn; },
    get active() { return state.active || state.connecting; },
  };
})();
