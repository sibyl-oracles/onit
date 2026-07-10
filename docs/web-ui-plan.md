# OnIt Web UI Rebuild Plan

**Status:** approved 2026-07-08 · replaces the Gradio UI in `src/ui/web.py`

## Motivation

The Gradio UI does not use native streaming. A timer-driven polling loop
(`poll_response` in `src/ui/web.py`) re-sends the entire chat history to the
browser on every tick and simulates streaming by replacing the last message
with the full accumulated text. Every token update is a full-history payload
plus a re-render — the source of the perceived lag and scroll fighting.

The backend is already in good shape: `Onit.process_task()` exposes token-level
`stream_callback`s, per-session `safety_queue` stop signals, and per-session
paths. Only the presentation layer needs replacing.

## Framework decision

- **Streamlit — rejected.** Full script rerun per interaction; streaming from a
  background agent requires the same polling anti-pattern; rudimentary chat
  components; cannot match a Claude/Codex/Gemini look.
- **Chainlit — rejected.** Fast (~2–4 days) and ChatGPT-like out of the box, but
  layout is locked to their design and the project has a breaking-change
  history. Caps out below the "looks like claude.ai" bar.
- **FastAPI + React — chosen.** FastAPI is already a dependency and the OAuth,
  upload, and A2A code is already written against it. A custom React frontend
  owns every pixel and consumes true server-push streaming.

## Architecture

```
Browser (SPA, static assets)
   │  fetch + SSE (POST /api/chat → text/event-stream)
   ▼
FastAPI app (src/ui/api.py)  ── mounts ──  static frontend build
   │  process_task(stream_callback, safety_queue, …)
   ▼
Onit core (src/onit.py)
```

SSE event schema (one `event:`/`data:` pair per message, JSON payloads):

| event           | payload                                   |
|-----------------|-------------------------------------------|
| `token`         | `{ "content": full_text_so_far }`         |
| `tool_start`    | `{ "name", "arguments" }`                 |
| `tool_log`      | `{ "name", "data", "level" }`             |
| `tool_progress` | `{ "name", "elapsed_seconds" }`           |
| `tool_end`      | `{ "name", "result", "success" }`         |
| `file`          | `{ "url", "name", "size" }`               |
| `stats`         | `{ "elapsed", "tokens_per_second" }`      |
| `done`          | `{ "content": final_text }`               |
| `error`         | `{ "message" }`                           |

## Phases

### Phase 1 — UI-agnostic API server (`src/ui/api.py`)

- `POST /api/chat` → SSE stream of the events above, driven by
  `process_task()` callbacks bridged through a per-request asyncio queue.
- `POST /api/chat/stop` → push to the session's `safety_queue`.
- `GET /api/history` → replay messages from the session JSONL.
- `POST /api/clear` → reset session data dir + JSONL.
- `POST /api/upload`, `GET /uploads/{session}/{file}` — ported unchanged.
- `/auth/login`, `/auth/callback`, `/auth/logout`, `/auth/check` — Google
  OAuth PKCE ported unchanged (`SessionManager`, `OAuthFlowManager`,
  `GoogleAuthenticator` reused from `web.py`).
- Per-browser session isolation via the existing `onit_session` cookie.

### Phase 2 — SPA frontend (`src/ui/static/`)

> **Adaptation (2026-07-08):** the original plan called for Vite + React, but
> no Node toolchain is available in the dev environment. Implemented instead
> as a **no-build vanilla-JS SPA** — hand-written `index.html` / `style.css` /
> `app.js` with vendored `marked`, `DOMPurify`, and `highlight.js` (fetched
> once from jsDelivr into `src/ui/static/vendor/`). Same UX target, zero
> toolchain: the assets ship directly in the wheel and no Node is needed for
> development or production. A React port remains possible later; the API
> contract is unchanged.

- Claude-style layout: session sidebar (new/switch/rename/delete), centered
  chat column, sticky composer with file attach and stop button,
  thinking/status indicator, streaming markdown with syntax-highlighted code
  blocks (copy button per block), light/dark theme, auto-scroll with
  scroll-lock when the user scrolls up, drag-and-drop upload, execution-logs
  drawer, login view for Google OAuth.

### Phase 3 — Parity and cutover

- Port: upload indicator, generated-file + code-zip downloads
  (`zip_code_files`), elapsed / tok-per-second footer, session restore on
  reload, multi-tab isolation.
- `onit serve web` launches the new server; `--ui gradio` remains as a
  temporary fallback until the bake-in period ends, then the Gradio path and
  dependency are deleted.
- `docker-compose.yml` unchanged apart from the image rebuild (same port 9000).
- API tests replace the Gradio UI tests: SSE event sequence, auth gating,
  stop, upload round-trip.

## Risks

- Frontend scope creep — the component set above is the whole scope; anything
  else waits for a follow-up.
- Event schema drift — the schema in this document is the contract; change it
  here first.
