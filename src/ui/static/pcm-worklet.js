/* Audio worklets for the OnIt voice call.
 *
 * Both ends of the call are PCM16 mono at 24 kHz — the format the VoiceChat
 * container speaks. The AudioContext is created at that rate so no resampling
 * is needed on either side; the browser handles the device rate conversion.
 *
 * These run on the audio thread. Nothing here may allocate unpredictably or
 * block: a missed 128-sample render quantum is an audible click.
 */

const FRAME_SAMPLES = 1920;   // 3840 bytes of PCM16 ≈ 80 ms at 24 kHz

/* Mic → PCM16 frames posted to the main thread. */
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Int16Array(FRAME_SAMPLES);
    this._n = 0;
    this._muted = false;
    this.port.onmessage = (e) => {
      if (e.data && e.data.type === "mute") this._muted = !!e.data.value;
    };
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      // Clamp before scaling: a float slightly over 1.0 would wrap to a loud
      // negative sample, which the server's VAD reads as a transient.
      const s = this._muted ? 0 : Math.max(-1, Math.min(1, ch[i]));
      this._buf[this._n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this._n === FRAME_SAMPLES) {
        // Copy, then transfer that copy: _buf is refilled on the next quantum,
        // so handing its own memory across would corrupt the frame in flight.
        const frame = this._buf.slice().buffer;
        this.port.postMessage(frame, [frame]);
        this._n = 0;
      }
    }
    return true;
  }
}

/* Agent audio → speakers, through a ring buffer that can be dropped instantly.
 *
 * The flush is the whole point. When the user starts talking, the server stops
 * generating — but this buffer may still be holding a second of speech. Playing
 * it out would have the agent talking over the person who just interrupted it,
 * which is exactly what barge-in is supposed to prevent. So flush discards.
 */
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._queue = [];      // Float32Array chunks, oldest first
    this._offset = 0;      // read position within _queue[0]
    this._queued = 0;      // samples still to play, for the "speaking" state
    this.port.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "audio") {
        const pcm = new Int16Array(msg.buffer);
        const f = new Float32Array(pcm.length);
        for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 0x8000;
        this._queue.push(f);
        this._queued += f.length;
      } else if (msg.type === "flush") {
        this._queue.length = 0;
        this._offset = 0;
        this._queued = 0;
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;
    let i = 0;
    while (i < out.length && this._queue.length) {
      const chunk = this._queue[0];
      const n = Math.min(out.length - i, chunk.length - this._offset);
      out.set(chunk.subarray(this._offset, this._offset + n), i);
      this._offset += n;
      this._queued -= n;
      i += n;
      if (this._offset >= chunk.length) {
        this._queue.shift();
        this._offset = 0;
      }
    }
    // Silence for whatever the queue could not fill.
    if (i < out.length) out.fill(0, i);

    // Let the UI know when speech starts and stops without polling from here
    // every quantum — only the transitions are posted.
    const speaking = this._queued > 0;
    if (speaking !== this._wasSpeaking) {
      this._wasSpeaking = speaking;
      this.port.postMessage({ type: "speaking", value: speaking });
    }
    return true;
  }
}

registerProcessor("onit-capture", CaptureProcessor);
registerProcessor("onit-playback", PlaybackProcessor);
