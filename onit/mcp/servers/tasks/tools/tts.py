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

Kokoro TTS engine - text to speech as local ONNX inference.

The model is Kokoro-82M (Apache 2.0), run through ``kokoro-onnx`` on
ONNX Runtime: no GPU, no cloud call, no key. Weights are downloaded once
to ``~/.onit/models/kokoro`` on first use and reused from disk after
that, so the tool works offline from the second call on.

The engine is loaded lazily and kept as a module singleton: an 82M model
takes a few seconds to map into memory and the MCP server process is
long-lived, so paying that once per process beats paying it per call.
Everything here is importable without ``kokoro_onnx`` installed - the
import happens inside :func:`_get_engine`, and a missing dependency is
reported as a JSON error the model can act on rather than a traceback
that kills the tool server.

Phonemization goes through espeak-ng (bundled by ``espeakng-loader``,
no system install needed). espeak-ng is GPLv3; it is used here as a
separate pip-installed component at runtime, not linked into anything
distributed, which keeps the Apache-2.0 model weights and this code
clean.
"""

import array
import json
import logging
import os
import re
import time
import wave
from typing import Optional

from onit.mcp.servers.tasks.shared import validate_required as _validate_required

logger = logging.getLogger(__name__)

# Where weights live. Overridable for tests and for pointing several
# runs at one shared download.
KOKORO_MODEL_DIR = os.environ.get(
    "ONIT_KOKORO_DIR", os.path.join(os.path.expanduser("~"), ".onit", "models", "kokoro")
)
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)
MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"

DEFAULT_VOICE = "af_heart"
DEFAULT_LANG = "en-us"
DEFAULT_SPEED = 1.0
MAX_CHARS = 5000  # one call's budget; longer text should be split by the caller

# Voices shipped in voices.bin that read as female/male American English,
# the set a user is most likely to mean by "a normal voice". The full
# catalog is larger; anything outside this list is accepted as-is and
# validated by the engine itself.
KNOWN_VOICES = frozenset(
    {
        "af_heart",
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
    }
)

_ENGINE = None  # Kokoro instance once loaded; None until first use


class TTSError(Exception):
    """Raised for conditions the tool should report as a JSON error."""


def _download(url: str, dest: str) -> None:
    """Fetch one file to *dest* with a resume, so an interrupted first
    run continues rather than restarting a 300 MB download.

    GitHub's release CDN honours ``Range``, but a proxy that does not
    would answer a resume with the whole body; blindly appending that
    yields a corrupt file that loads as nothing. The response status
    decides: 206 appends, anything else starts over."""
    import urllib.request

    part = dest + ".part"
    resume_from = os.path.getsize(part) if os.path.isfile(part) else 0
    req = urllib.request.Request(url)
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")
    try:
        resp_ctx = urllib.request.urlopen(req, timeout=60)
    except Exception as e:
        # A first-use download that cannot reach the CDN must read as a
        # clean tool error, not a raw URLError traceback: speak_impl
        # catches only TTSError, and callers build user-visible JSON
        # from it.
        raise TTSError(f"Could not download {url}: {e}") from e
    with resp_ctx as resp:
        append = resume_from and resp.status == 206
        mode = "ab" if append else "wb"
        offset = resume_from if append else 0
        with open(part, mode) as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                offset += len(chunk)
    # A truncated file is worse than a missing one: the next call would
    # skip the download and fail to load. Only a complete body promotes.
    expected = resp.headers.get("Content-Length")
    if expected is not None and offset != int(expected) + (resume_from if append else 0):
        os.unlink(part)
        raise TTSError(f"Download of {url} was truncated ({offset} of {expected} bytes)")
    os.replace(part, dest)


def ensure_weights() -> tuple[str, str]:
    """Return (model_path, voices_path), downloading either when missing."""
    model_path = os.path.join(KOKORO_MODEL_DIR, MODEL_FILE)
    voices_path = os.path.join(KOKORO_MODEL_DIR, VOICES_FILE)
    os.makedirs(KOKORO_MODEL_DIR, exist_ok=True)
    for url, path in ((MODEL_URL, model_path), (VOICES_URL, voices_path)):
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        logger.info("Downloading %s -> %s", url, path)
        _download(url, path)
    return model_path, voices_path


def _get_engine():
    """Load the Kokoro engine once per process; raise TTSError when it cannot."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    try:
        from kokoro_onnx import Kokoro
    except ImportError as e:
        raise TTSError(
            "kokoro-onnx is not installed. Install it with: pip install -e '.[tts]' "
            "(from the repo) or pip install 'onit[tts]'"
        ) from e
    model_path, voices_path = ensure_weights()
    try:
        _ENGINE = Kokoro(model_path=model_path, voices_path=voices_path)
    except Exception as e:
        # A corrupt cache (truncated download, disk full) must not be a
        # permanent failure: drop the weights and fetch them once more.
        # One retry, not a loop — a genuinely broken download should
        # surface as an error, not spin.
        for path in (model_path, voices_path):
            try:
                os.unlink(path)
            except OSError:
                pass
        logger.warning("Kokoro model failed to load (%s); re-downloading weights", e)
        model_path, voices_path = ensure_weights()
        try:
            _ENGINE = Kokoro(model_path=model_path, voices_path=voices_path)
        except Exception as e2:
            raise TTSError(f"Failed to load the Kokoro model: {e2}") from e2
    return _ENGINE


def synthesize_to_wav(
    text: str,
    out_path: str,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    lang: str = DEFAULT_LANG,
) -> dict:
    """Synthesize *text* to a WAV file at *out_path*. Returns metadata."""
    start = time.time()
    engine = _get_engine()
    try:
        audio, sample_rate = engine.create(text=text, voice=voice, speed=speed, lang=lang)
    except Exception as e:
        raise TTSError(f"Speech synthesis failed: {e}") from e

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # soundfile writes WAVs too, but the stdlib wave module has no
    # dependency to fail on. Kokoro returns a numpy array, which scales
    # and casts in one shot; any other engine's plain sequence of floats
    # goes through the stdlib array module instead, so this writer never
    # requires numpy itself (the test fakes rely on that — CI has no
    # numpy, and a test-only dependency would be the wrong fix).
    try:
        frames = (audio * 32767.0).astype("int16").tobytes()
    except (TypeError, AttributeError):
        frames = array.array(
            "h",
            (int(max(-1.0, min(1.0, float(sample))) * 32767.0) for sample in audio),
        ).tobytes()
    try:
        with wave.open(out_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(frames)
    except Exception as e:
        # A write failure (bad path, disk full) is the tool's error to
        # report, not a traceback that kills the server.
        raise TTSError(f"Failed to write the WAV file: {e}") from e

    duration = len(audio) / sample_rate
    return {
        "path": out_path,
        "duration_s": round(duration, 2),
        "sample_rate": sample_rate,
        "voice": voice,
        "synthesize_s": round(time.time() - start, 2),
    }


def sanitize_text(text: str) -> str:
    """Reduce a chat reply to what reads well aloud.

    Markdown symbols are noise to a listener: emphasis markers, headings,
    link targets and fenced code say nothing when spoken. Link text stays
    (the words are the point), fenced code blocks go entirely — reading
    source aloud is never what "read this to me" means.
    """
    out = re.sub(r"```[\s\S]*?```", " ", text)
    out = re.sub(r"`([^`]*)`", r"\1", out)
    out = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", out)
    out = re.sub(r"(^|\n)\s*#{1,6}\s*", r"\1", out)
    out = re.sub(r"[*_~]{1,3}", " ", out)
    out = re.sub(r"[ \t]*\n+[ \t]*", ". ", out)
    return re.sub(r"  +", " ", out).strip()


# Below this length a reply is already short enough to speak in full —
# summarizing "yes, done." would only add a model call. The bar is what a
# listener hears, not what a reader sees: 160 chars is roughly ten seconds
# of speech, about two short sentences. A 400-char answer reads in a glance
# but plays for half a minute, which is exactly the "long version" the
# read-aloud exists to avoid.
SUMMARY_SKIP_CHARS = 160
# The spoken summary is cut here. A listener cannot skim audio: a summary
# that runs past a minute is not a summary, it is the problem again.
SUMMARY_MAX_CHARS = 600
# Anything shorter than this from the model is not a summary, it is a
# refusal or an empty string — speak the original text instead.
SUMMARY_MIN_CHARS = 20
# The summarizer must not hold the spoken reply hostage: a read-aloud that
# waits half a minute for a condensation is worse than no condensation.
SUMMARY_TIMEOUT_S = 30.0

# When the summarizer cannot run — no model, a timeout, a refusal — the
# spoken form is the *lead* of the answer, not the whole answer. A listener
# cannot skim audio: the full text is what read-aloud exists to avoid
# speaking. The lead is the first sentences of the sanitized text, cut on a
# sentence boundary near this cap.
SPOKEN_LEAD_MAX_CHARS = 280


def spoken_lead(text: str, max_chars: int = SPOKEN_LEAD_MAX_CHARS) -> str:
    """The opening sentences of *text*, cut on a sentence boundary.

    The fallback when condensation is impossible: what the listener hears is
    the start of the answer — usually its conclusion — and never the whole
    wall of text. ``text`` must already be sanitized; this only trims.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    # ". " with a following capital is the safest sentence end in prose;
    # a bare period inside "e.g." or "3.5" does not get one.
    cut = text.rfind(". ", 0, max_chars)
    if cut > 0:
        return text[: cut + 1]
    # No sentence end in reach — cut at the last space so a word is not
    # severed, and close the fragment so it does not read as a mistake.
    space = text.rfind(" ", 0, max_chars)
    return text[:space].rstrip(" ,;:") + "…" if space > 0 else text[:max_chars] + "…"


_SUMMARY_SYSTEM_PROMPT = (
    "You condense text for a listener. Reply with ONLY a spoken summary of the "
    "text, at most 40 words and always clearly shorter than the text itself: "
    "the result or answer first, then the key numbers, names and decisions. "
    'No preamble, no "Summary" heading, no markdown, no lists. Plain sentences '
    "a person can hear once and follow. When the text is already short, one "
    "sentence carrying its core fact is enough."
)


def _clean_summary(raw: str) -> str:
    """Strip the wrappers a model puts around a summary.

    Fenced code, a "Summary:" heading, "Here is the summary:" — all
    wrappers a listener should never hear. What survives is plain prose.
    """
    fences = re.findall(r"```[\s\S]*?```", raw)
    out = re.sub(r"```[\s\S]*?```", " ", raw)
    out = re.sub(r"```", " ", out)
    if not out.strip() and fences:
        # The whole reply was fenced. The fence was the wrapper, not the
        # content — speak what was inside it.
        out = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", fences[0].strip())
        out = re.sub(r"```$", "", out)
    out = re.sub(
        r"^\s*(?:here(?:'s| is) (?:the |a )?(?:spoken |brief |short )?summary"
        r"(?: for listening| of the text)?|summary)\s*[:\-—]\s*",
        "",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _is_structured(summary: str) -> bool:
    """True when the summary is still markdown structure, not prose.

    A table separator anywhere, or several list-marker lines, means the
    model reformatted the answer instead of condensing it.
    """
    if re.search(r"\|\s*:?-{2,}", summary):
        return True
    lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
    markers = sum(1 for ln in lines if re.match(r"^(?:[-*+]|\d+[.)])\s+", ln))
    return markers >= 3


async def summarize_for_speech(
    text: str,
    config_data: dict | None = None,
    model: str | None = None,
    endpoint: dict | None = None,
) -> str:
    """Return the short spoken form of *text*, or *text* itself.

    Read-aloud should carry the parts a listener cannot get by glancing at
    the screen: what the answer concluded, the numbers that matter, what
    was decided. The full text stays on screen for everything else, so the
    spoken form is a condensation, not a reading.

    The model is the agent's own configured endpoint, reached through the
    same plain, tool-less client the offline learn loops use — the
    summarizer is a judgment about text that already exists, so it runs
    non-streaming at temperature 0 with no tools and no conversation.
    ``model`` is the agent's own model name when the caller knows it; left
    unset, the endpoint's default is auto-detected, which on a multi-model
    host can pick something the account cannot actually serve.

    ``endpoint`` pins where the summarization call goes - the caller's
    known host/model/key, normally the load balancer's preferred endpoint.
    Without it the config is re-resolved, and a ``serving.endpoints`` list
    config (no ``serving.host`` key) resolves to the localhost default,
    where nothing listens: the summary silently fails and the full text
    gets spoken. Passing the endpoint the agent itself is on removes that
    whole failure class.

    Every failure path returns a *short* spoken form — the lead of the text,
    via ``spoken_lead`` — never the full text: a summarizer that cannot run
    must not turn read-aloud back into a full reading. The caller speaks
    something either way.
    """
    if not text or len(text) <= SUMMARY_SKIP_CHARS:
        return text
    try:
        from onit.learn.llm import make_ask

        ask = await make_ask(config_data, model=model, timeout=SUMMARY_TIMEOUT_S, endpoint=endpoint)
        raw = await ask(
            [
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
        )
    except Exception as e:
        logger.warning("tts summarization failed, speaking the lead: %s", e)
        return spoken_lead(text)
    summary = _clean_summary(str(raw or ""))
    if len(summary) < SUMMARY_MIN_CHARS or len(summary) >= len(text):
        # Not a usable condensation — a refusal, an empty reply, or
        # something as long as the original. The lead carries the start.
        return spoken_lead(text)
    if _is_structured(summary):
        # A model that ignores "no markdown" and hands back a table or a
        # list has not summarized — it has reformatted, and a listener
        # cannot skim a table read aloud. The lead is no worse.
        return spoken_lead(text)
    if len(summary) > SUMMARY_MAX_CHARS:
        # Cut on a sentence boundary when one is in reach, so the clip
        # does not end mid-word.
        cut = summary.rfind(". ", 0, SUMMARY_MAX_CHARS)
        summary = summary[: cut + 1] if cut > 0 else summary[:SUMMARY_MAX_CHARS]
    return summary.strip()


# Shared with the unified ToolsMCPServer re-registration (tools/mcp_server.py)
TTS_TOOL_DESCRIPTION = """Convert text to spoken audio as a WAV file, using the local
Kokoro TTS model (offline, no API key). Use this when the user asks to
hear something read aloud, or to produce a voice clip.

Args:
- text: What to say (plain prose; markdown symbols are stripped)
- voice: Voice name (default: af_heart; try af_bella, am_michael, bf_emma, bm_george, ...)
- speed: Speaking rate, 0.5 to 2.0 (default 1.0)
- lang: BCP-47 language code (default en-us)
- data_path: Session working directory - set automatically by the harness; leave unset.

Returns JSON: {path, url, filename, duration_s, sample_rate, voice, synthesize_s}"""


def speak_impl(
    text: Optional[str] = None,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    lang: str = DEFAULT_LANG,
    data_path: str = "",
) -> str:
    """Core speak implementation (always defined for import by other modules)."""
    if err := _validate_required(text=text):
        return err

    text = sanitize_text(str(text))
    if not text:
        return json.dumps({"error": "Nothing to speak after stripping markup.", "status": "error"})
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    if voice not in KNOWN_VOICES:
        return json.dumps(
            {
                "error": f"Unknown voice {voice!r}. Known voices include: "
                + ", ".join(sorted(KNOWN_VOICES)[:8])
                + " ...",
                "status": "error",
            }
        )
    speed = min(max(float(speed), 0.5), 2.0)

    base = os.path.realpath(os.path.expanduser(data_path or DATA_PATH))
    os.makedirs(base, exist_ok=True)
    # Filenames carry the voice and a timestamp so two clips never collide
    # and a listener can tell them apart in the file list.
    filename = "tts_%s_%s.wav" % (voice, time.strftime("%Y%m%d_%H%M%S"))
    out_path = os.path.join(base, filename)

    try:
        meta = synthesize_to_wav(text, out_path, voice=voice, speed=speed, lang=lang)
    except TTSError as e:
        return json.dumps({"error": str(e), "status": "error"})

    meta.update(
        {
            "filename": filename,
            # The harness rewrites bare filenames it finds in the reply into
            # /uploads/<session>/ links (ui/api.py _extract_file_paths), so
            # the plain name is what the model should surface.
            "url": filename,
            "bytes": os.path.getsize(out_path),
            "status": "ok",
        }
    )
    return json.dumps(meta)


# Set via run() options in tools/mcp_server.py, like the other sub-modules.
DATA_PATH = os.path.join(os.path.expanduser("~"), "sandbox")
