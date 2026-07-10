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

FastAPI + SSE web chat UI for OnIt.

Replaces the Gradio UI (web.py) with a plain FastAPI backend that streams
agent output over Server-Sent Events and serves a static single-page app
from src/ui/static/.  Implements the same interface as ChatUI (text.py)
so chat.py and onit.py work without changes.

SSE event schema (see docs/web-ui-plan.md):
    token         {"delta": str}
    phase_end     {"content": str}
    status        {"text": str}
    done          {"content": str, "files": [...], "zip": {...}|null,
                   "elapsed": float, "tok_s": float}
    error         {"message": str}
"""

import asyncio
import json
import logging
import mimetypes
import os
import queue
import re
import secrets
import shutil
import tempfile
import threading
import time
import urllib.parse
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

from src.lib.files import zip_code_files
from src.sessions import delete_session as _delete_session_index
from src.sessions import list_sessions as _list_sessions
from src.sessions import tag_session as _tag_session
from .auth import (
    GOOGLE_AUTH_AVAILABLE,
    GoogleAuthenticator,
    NullConsole,
    OAuthFlowManager,
    SessionManager,
)

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, Request, Response, UploadFile
    from fastapi.responses import (
        FileResponse,
        HTMLResponse,
        JSONResponse,
        RedirectResponse,
        StreamingResponse,
    )
    from fastapi.staticfiles import StaticFiles
except ImportError:
    raise ImportError(
        "FastAPI is required for the web UI. Install it with: pip install fastapi uvicorn"
    )

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Pattern for validating session IDs (UUIDs only)
_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

# Sentinel pushed into the event queue when a chat turn is finished
_END = ("__end__", None)


def _sse(event: str, payload: dict) -> str:
    """Format one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@dataclass
class ApiSession:
    """Per-browser session state for the web UI."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_path: str = ""
    data_path: str = ""
    processing: bool = False
    safety_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=10))
    created: datetime = field(default_factory=datetime.now)


class WebApiUI:
    """FastAPI web UI server implementing the ChatUI interface subset used
    by onit.py in web mode (add_message, add_log, tool_log, tool_progress,
    render, stop_status, console, launch)."""

    def __init__(
        self,
        theme: str = "white",
        agent_cursor: str = "OnIt",
        data_path: str = "~/.onit/data",
        show_logs: bool = False,
        server_port: int = 9000,
        google_client_id: Optional[str] = None,
        google_client_secret: Optional[str] = None,
        allowed_emails: Optional[list[str]] = None,
        session_path: Optional[str] = None,
        title: str = "OnIt Chat",
        verbose: bool = False,
    ) -> None:
        self.title = title
        self.theme = theme
        self.agent_cursor = agent_cursor
        self.data_path = os.path.expanduser(data_path)
        self.session_path = session_path
        self.show_logs = show_logs
        self.verbose = verbose
        self.server_port = server_port
        self.console = NullConsole()

        # authentication
        self.google_client_id = google_client_id
        self.google_client_secret = google_client_secret

        # Nullify placeholder credentials so auth is cleanly disabled
        for attr in ('google_client_id', 'google_client_secret'):
            val = getattr(self, attr, None)
            if val and "YOUR_" in str(val).upper():
                setattr(self, attr, None)

        self.auth_enabled = bool(self.google_client_id and self.google_client_secret)
        if self.auth_enabled and not GOOGLE_AUTH_AVAILABLE:
            print("Google OAuth credentials provided but google-auth package not installed. Authentication disabled.")
            self.auth_enabled = False

        self.session_manager = SessionManager() if self.auth_enabled else None
        self.oauth_flow_manager = OAuthFlowManager() if self.auth_enabled else None
        self.authenticator = (
            GoogleAuthenticator(self.google_client_id, self.google_client_secret, allowed_emails)
            if self.auth_enabled
            else None
        )

        if self.auth_enabled:
            print(f"OAuth2 authentication enabled (client: {self.google_client_id[:20]}...)")
            if allowed_emails:
                print(f"   Allowed emails: {', '.join(allowed_emails)}")
        else:
            print("Authentication disabled — no Google OAuth credentials configured.")

        self._web_sessions: dict[str, ApiSession] = {}
        self._onit = None  # set by OnIt after creation
        self.execution_logs: deque[dict] = deque(maxlen=100)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # {cookie_value: {email, session_id, expires}}
        self._authenticated_cookies: dict[str, dict] = {}

        self.app: Optional[FastAPI] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _sessions_dir(self) -> str:
        if self.session_path:
            return os.path.dirname(self.session_path)
        return os.path.expanduser("~/.onit/sessions")

    def _get_or_create_session(self, session_id: str | None = None) -> tuple[str, ApiSession]:
        """Get an existing session or create a new one.

        Restores sessions from disk (JSONL survives restarts); otherwise
        creates a new one, reusing *session_id* when supplied so the browser
        cookie and server-side record stay in sync.
        """
        # Reject invalid session IDs to prevent path traversal
        if session_id and not _UUID_RE.match(session_id):
            session_id = None

        if session_id and session_id in self._web_sessions:
            return session_id, self._web_sessions[session_id]

        sessions_dir = self._sessions_dir()
        os.makedirs(sessions_dir, exist_ok=True)

        session = ApiSession(session_id=session_id) if session_id else ApiSession()
        session.session_path = os.path.join(sessions_dir, f"{session.session_id}.jsonl")
        if not os.path.exists(session.session_path):
            with open(session.session_path, "w", encoding="utf-8") as f:
                f.write("")
        session.data_path = str(Path(tempfile.gettempdir()) / "onit" / "data" / session.session_id)
        os.makedirs(session.data_path, exist_ok=True)

        self._web_sessions[session.session_id] = session
        logger.info("Web session %s ready", session.session_id)

        # Drop idle in-memory sessions older than 24h (files stay on disk)
        now = datetime.now()
        expired = [
            sid for sid, s in self._web_sessions.items()
            if (now - s.created) > timedelta(hours=24) and not s.processing
        ]
        for sid in expired:
            del self._web_sessions[sid]

        return session.session_id, session

    def _resolve_session(self, request: Request) -> tuple[str, ApiSession]:
        """Resolve the session from the X-Session-Id header or cookie."""
        sid = request.headers.get("x-session-id") or request.cookies.get("onit_session")
        return self._get_or_create_session(sid)

    # ------------------------------------------------------------------
    # Interface methods (matching ChatUI contract)
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        response: str,
        elapsed: str = "",
    ) -> None:
        """No-op for web UI. Responses are routed through per-session SSE."""
        pass

    def add_log(
        self,
        message: str,
        level: Literal["info", "warning", "error", "debug"] = "info",
    ) -> None:
        self.execution_logs.append({
            "message": message,
            "level": level,
            "timestamp": datetime.now().strftime("%I:%M %p %d %b"),
        })

    def tool_log(self, name: str, data: str, level: str = "info") -> None:
        """Forward real-time log messages from MCP tools (e.g. sandbox output)."""
        self.add_log(f"[{name}] {data}", level=level)

    def tool_progress(self, name: str, elapsed_seconds: int) -> None:
        """No-op; per-request status events cover progress display."""
        pass

    def render(self, thinking: bool = False):
        """No-op for web UI; the browser renders."""
        pass

    def stop_status(self) -> None:
        """No-op for web UI."""
        pass

    # ------------------------------------------------------------------
    # File-path extraction (shared with history loading)
    # ------------------------------------------------------------------

    def _extract_file_paths(self, text: str, data_path: str | None = None, session_id: str | None = None) -> tuple[str, list[str]]:
        """Extract file paths from text; replace with safe download links.

        Handles three cases:
        1. Absolute paths under data_path (local MCP tools)
        2. HTTP URLs containing /uploads/ (remote MCP callback uploads)
        3. Bare filenames that exist in data_path (model just mentions the name)
        """
        effective_data_path = data_path or self.data_path
        found_files: list[str] = []  # absolute paths in data_path
        cleaned = text

        # 1. Match absolute paths under data_path
        local_pattern = re.escape(effective_data_path) + r'/[\w\-\.]+(?:\.\w+)'
        for match in re.findall(local_pattern, cleaned):
            fname = os.path.basename(match)
            fpath = os.path.join(effective_data_path, fname)
            if fpath not in found_files:
                found_files.append(fpath)
        # Replace full paths with just basenames
        cleaned = re.sub(local_pattern, lambda m: os.path.basename(m.group(0)), cleaned)

        # 2. Match http(s) URLs pointing to /uploads/<filename>
        url_pattern = r'https?://[^/\s]+/uploads/([\w\-\.]+(?:\.\w+))'
        for match in re.finditer(url_pattern, cleaned):
            fname = match.group(1)
            fpath = os.path.join(effective_data_path, fname)
            if fpath not in found_files:
                found_files.append(fpath)
        # Replace full URLs with just the filename
        cleaned = re.sub(url_pattern, r'\1', cleaned)

        # Strip any other absolute file paths, keeping only the basename
        cleaned = re.sub(r'(?<![:\w])/(?:[\w\.\-]+/)+(?=[\w\.\-]+)', '', cleaned)

        # 3. Check for bare filenames that exist in data_path
        if os.path.isdir(effective_data_path):
            existing = set(os.listdir(effective_data_path))
            for fname in existing:
                if fname in cleaned:
                    fpath = os.path.join(effective_data_path, fname)
                    if fpath not in found_files:
                        found_files.append(fpath)

        # Turn found basenames into markdown download links
        upload_prefix = f"/uploads/{session_id}" if session_id else "/uploads"
        for fpath in found_files:
            fname = os.path.basename(fpath)
            cleaned = cleaned.replace(fname, f'[{fname}]({upload_prefix}/{fname})', 1)

        return cleaned, found_files

    def _file_infos(self, file_paths: list[str], session_id: str) -> list[dict]:
        """Build download descriptors for files that exist on disk."""
        infos = []
        for fpath in file_paths:
            if not os.path.isfile(fpath):
                continue
            fname = os.path.basename(fpath)
            infos.append({
                "name": fname,
                "url": f"/uploads/{session_id}/{urllib.parse.quote(fname)}",
                "size": os.path.getsize(fpath),
            })
        return infos

    def _load_history(self, session: ApiSession) -> list[dict]:
        """Load chat history from the JSONL session file as plain dicts."""
        messages: list[dict] = []
        path = session.session_path
        if not path or not os.path.exists(path):
            return messages
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "task" not in entry or "response" not in entry:
                        continue
                    # Replace raw absolute file paths with friendly basenames
                    task_display = re.sub(
                        r'Relevant files:\s*/[^\s]+/([^\s/]+)',
                        lambda m: f'📎 {m.group(1)}',
                        entry["task"],
                    )
                    messages.append({"role": "user", "content": task_display})
                    display, file_paths = self._extract_file_paths(
                        entry["response"],
                        data_path=session.data_path,
                        session_id=session.session_id,
                    )
                    messages.append({
                        "role": "assistant",
                        "content": display,
                        "files": self._file_infos(file_paths, session.session_id),
                    })
        except Exception:
            pass
        return messages

    # ------------------------------------------------------------------
    # Chat task execution (runs on the OnIt event loop)
    # ------------------------------------------------------------------

    async def _run_task(self, session: ApiSession, task: str, events: queue.Queue) -> None:
        """Run process_task on the OnIt loop, forwarding events to the SSE queue."""
        try:
            def _on_token(token, _full_content):
                if token:
                    events.put_nowait(("token", {"delta": token}))

            def _on_phase_end(content, _tok_s):
                events.put_nowait(("phase_end", {"content": content or ""}))

            def _on_tool_status(text):
                events.put_nowait(("status", {"text": text or ""}))

            stats: dict = {}
            start = time.monotonic()
            response = await self._onit.process_task(
                task,
                session_path=session.session_path,
                data_path=session.data_path,
                safety_queue=session.safety_queue,
                stream_callback=_on_token,
                stream_complete_callback=_on_phase_end,
                stats=stats,
                tool_status_callback=_on_tool_status,
                session_id=session.session_id,
            )
            elapsed = time.monotonic() - start
            tok_s = stats.get("tokens_per_second", 0)

            if not response:
                events.put_nowait(("error", {
                    "message": "I'm sorry, I couldn't process your request. Please try again."
                }))
                return

            display, file_paths = self._extract_file_paths(
                response, data_path=session.data_path, session_id=session.session_id
            )
            zip_info = None
            zip_path = zip_code_files(session.data_path)
            if zip_path:
                zip_info = {
                    "name": os.path.basename(zip_path),
                    "url": f"/api/zip/{session.session_id}",
                    "size": os.path.getsize(zip_path),
                }
            events.put_nowait(("done", {
                "content": display,
                "files": self._file_infos(file_paths, session.session_id),
                "zip": zip_info,
                "elapsed": round(elapsed, 2),
                "tok_s": round(tok_s, 1),
            }))
        except Exception as e:
            logger.error("Error processing task: %s", e)
            events.put_nowait(("error", {"message": f"Error: {e}"}))
        finally:
            session.processing = False
            events.put_nowait(_END)

    async def _event_stream(self, events: queue.Queue):
        """Async generator bridging the thread-safe event queue to SSE."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, events.get, True, 15.0)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if item == _END:
                break
            etype, payload = item
            yield _sse(etype, payload)

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _auth_email(self, request: Request) -> Optional[str]:
        """Return the authenticated email for this request, or None."""
        auth_cookie = request.cookies.get("onit_auth")
        if not auth_cookie or auth_cookie not in self._authenticated_cookies:
            return None
        cookie_data = self._authenticated_cookies[auth_cookie]
        if datetime.now() > cookie_data['expires']:
            del self._authenticated_cookies[auth_cookie]
            return None
        return cookie_data['email']

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------

    def build_app(self) -> FastAPI:
        app = FastAPI(title=self.title, docs_url=None, redoc_url=None)

        @app.middleware("http")
        async def session_cookie_middleware(request: Request, call_next):
            response = await call_next(request)
            if not request.cookies.get("onit_session"):
                response.set_cookie(
                    key="onit_session",
                    value=str(uuid.uuid4()),
                    max_age=86400 * 7,   # 7 days
                    httponly=True,
                    samesite="lax",
                )
            return response

        if self.auth_enabled:
            @app.middleware("http")
            async def auth_middleware(request: Request, call_next):
                path = request.url.path
                protected = path.startswith("/api/") or (
                    path.startswith("/uploads/") and request.method == "GET"
                )
                # /api/config is public: the SPA needs it to show the login view
                if protected and path != "/api/config" and not self._auth_email(request):
                    return JSONResponse({"detail": "Not authenticated"}, status_code=401)
                return await call_next(request)

        self._setup_oauth_routes(app)
        self._setup_api_routes(app)
        self._setup_file_routes(app)
        self._setup_static_routes(app)

        self.app = app
        return app

    def _setup_api_routes(self, app: FastAPI):

        @app.get("/api/config")
        async def api_config(request: Request):
            email = self._auth_email(request) if self.auth_enabled else None
            return {
                "title": self.title,
                "agent": self.agent_cursor,
                "auth_enabled": self.auth_enabled,
                "authenticated": bool(email) or not self.auth_enabled,
                "email": email,
                "show_logs": self.show_logs,
            }

        @app.get("/api/history")
        async def api_history(request: Request):
            sid, session = self._resolve_session(request)
            return {
                "session_id": sid,
                "processing": session.processing,
                "messages": self._load_history(session),
            }

        @app.post("/api/chat")
        async def api_chat(request: Request):
            if self._onit is None or self._loop is None:
                return JSONResponse({"detail": "Agent not ready"}, status_code=503)
            body = await request.json()
            message = (body.get("message") or "").strip()
            files = body.get("files") or []
            sid, session = self._resolve_session(request)

            if not message and not files:
                return JSONResponse({"detail": "Empty message"}, status_code=400)
            if session.processing:
                return JSONResponse({"detail": "A task is already running for this session"}, status_code=409)

            # Only accept uploaded files that live in this session's data dir
            safe_files = []
            for fpath in files:
                fname = os.path.basename(str(fpath))
                candidate = os.path.join(session.data_path, fname)
                if os.path.isfile(candidate):
                    safe_files.append(candidate)

            task = message
            if safe_files:
                task += "\nRelevant files: " + " ".join(safe_files)

            session.processing = True
            events: queue.Queue = queue.Queue()
            asyncio.run_coroutine_threadsafe(
                self._run_task(session, task, events), self._loop
            )
            return StreamingResponse(
                self._event_stream(events),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Session-Id": sid,
                },
            )

        @app.post("/api/chat/stop")
        async def api_stop(request: Request):
            sid, session = self._resolve_session(request)
            if session.processing and self._loop:
                self._loop.call_soon_threadsafe(session.safety_queue.put_nowait, True)
            return {"stopped": True, "session_id": sid}

        @app.post("/api/clear")
        async def api_clear(request: Request):
            sid, session = self._resolve_session(request)
            if session.data_path and os.path.isdir(session.data_path):
                shutil.rmtree(session.data_path, ignore_errors=True)
                os.makedirs(session.data_path, exist_ok=True)
            if session.session_path and os.path.exists(session.session_path):
                with open(session.session_path, "w", encoding="utf-8") as f:
                    f.write("")
            return {"cleared": True, "session_id": sid}

        @app.post("/api/upload")
        async def api_upload(request: Request, file: UploadFile = File(...)):
            sid, session = self._resolve_session(request)
            os.makedirs(session.data_path, exist_ok=True)
            safe_name = os.path.basename(file.filename or "upload")
            dest = os.path.join(session.data_path, safe_name)
            content = await file.read()
            with open(dest, "wb") as f:
                f.write(content)
            return {
                "name": safe_name,
                "url": f"/uploads/{sid}/{urllib.parse.quote(safe_name)}",
                "session_id": sid,
            }

        @app.get("/api/logs")
        async def api_logs():
            return {"logs": list(self.execution_logs)[-50:]}

        @app.get("/api/sessions")
        async def api_sessions():
            sessions = _list_sessions(sessions_dir=self._sessions_dir(), limit=50)
            for s in sessions:
                mem = self._web_sessions.get(s["session_id"])
                s["processing"] = bool(mem and mem.processing)
            return {"sessions": sessions}

        @app.post("/api/sessions/new")
        async def api_new_session():
            sid, _session = self._get_or_create_session(str(uuid.uuid4()))
            return {"session_id": sid}

        @app.delete("/api/sessions/{session_id}")
        async def api_delete_session(session_id: str):
            if not _UUID_RE.match(session_id):
                return JSONResponse({"detail": "Invalid session id"}, status_code=400)
            session = self._web_sessions.pop(session_id, None)
            if session and session.data_path and os.path.isdir(session.data_path):
                shutil.rmtree(session.data_path, ignore_errors=True)
            _delete_session_index(session_id, sessions_dir=self._sessions_dir())
            return {"deleted": True}

        @app.patch("/api/sessions/{session_id}")
        async def api_rename_session(session_id: str, request: Request):
            if not _UUID_RE.match(session_id):
                return JSONResponse({"detail": "Invalid session id"}, status_code=400)
            body = await request.json()
            tag = (body.get("tag") or "").strip()
            if not tag:
                return JSONResponse({"detail": "Missing tag"}, status_code=400)
            result = _tag_session(session_id, tag, sessions_dir=self._sessions_dir())
            if result is True:
                return {"renamed": True}
            detail = result if isinstance(result, str) else "Session not found"
            return JSONResponse({"detail": detail}, status_code=400)

        @app.get("/api/zip/{session_id}")
        async def api_zip(session_id: str):
            if session_id not in self._web_sessions:
                return Response(content="Session not found", status_code=404)
            data_path = self._web_sessions[session_id].data_path
            zip_path = data_path.rstrip(os.sep) + ".zip"
            if not os.path.isfile(zip_path):
                zip_path = zip_code_files(data_path)
            if zip_path and os.path.isfile(zip_path):
                return FileResponse(
                    zip_path,
                    media_type="application/zip",
                    filename=os.path.basename(zip_path),
                )
            return Response(content="No code files to download", status_code=404)

    def _setup_file_routes(self, app: FastAPI):
        """Routes to serve and receive files scoped per session."""

        @app.get("/uploads/{session_id}/{filename}")
        async def serve_upload(session_id: str, filename: str):
            # Use basename to prevent path traversal
            safe_name = os.path.basename(urllib.parse.unquote(filename))
            if session_id not in self._web_sessions:
                return Response(content="Session not found", status_code=404)
            data_path = self._web_sessions[session_id].data_path
            filepath = os.path.join(data_path, safe_name)
            if os.path.isfile(filepath):
                media_type = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
                return FileResponse(filepath, media_type=media_type, filename=safe_name)
            return Response(content="File not found", status_code=404)

        @app.post("/uploads/{session_id}/")
        async def receive_upload(session_id: str, file: UploadFile = File(...)):
            """Accept file uploads from remote MCP tools, scoped per session."""
            if session_id not in self._web_sessions:
                return Response(content="Session not found", status_code=404)
            data_path = self._web_sessions[session_id].data_path
            os.makedirs(data_path, exist_ok=True)
            safe_name = os.path.basename(file.filename)
            filepath = os.path.join(data_path, safe_name)
            content = await file.read()
            with open(filepath, "wb") as f:
                f.write(content)
            return {"filename": safe_name, "status": "ok"}

    def _setup_static_routes(self, app: FastAPI):
        """Serve the single-page app from src/ui/static/."""
        index_path = os.path.join(STATIC_DIR, "index.html")

        @app.get("/")
        async def index():
            if os.path.isfile(index_path):
                return FileResponse(index_path, headers={"Cache-Control": "no-cache"})
            return HTMLResponse("<h1>OnIt web UI assets missing</h1>", status_code=500)

        if os.path.isdir(STATIC_DIR):
            app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def _setup_oauth_routes(self, app: FastAPI):
        """Add OAuth2 login/callback/logout routes."""
        if not self.auth_enabled:
            return

        @app.get("/auth/login")
        async def oauth_login(request: Request):
            state, _code_verifier, code_challenge = self.oauth_flow_manager.create_flow()
            redirect_uri = f"http://{request.url.hostname}:{self.server_port}/auth/callback"
            auth_url = (
                "https://accounts.google.com/o/oauth2/v2/auth?"
                + urllib.parse.urlencode({
                    'client_id': self.google_client_id,
                    'redirect_uri': redirect_uri,
                    'response_type': 'code',
                    'scope': 'openid email profile',
                    'state': state,
                    'code_challenge': code_challenge,
                    'code_challenge_method': 'S256',
                    'access_type': 'online',
                    'prompt': 'select_account',
                })
            )
            return RedirectResponse(auth_url)

        def _error_page(title: str, message: str, status_code: int) -> HTMLResponse:
            return HTMLResponse(
                f"""
                <html>
                <head><title>{title}</title></head>
                <body style="font-family: -apple-system, Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #d32f2f;">❌ {title}</h1>
                <p>{message}</p>
                <a href="/" style="color: #4285f4;">Return to login</a>
                </body>
                </html>
                """,
                status_code=status_code,
            )

        @app.get("/auth/callback")
        async def oauth_callback(request: Request):
            code = request.query_params.get('code')
            state = request.query_params.get('state')
            error = request.query_params.get('error')

            if error:
                return _error_page("Authentication Failed", f"Error: {error}", 400)
            if not code or not state:
                return _error_page("Invalid Request", "Missing authorization code or state parameter.", 400)

            code_verifier = self.oauth_flow_manager.verify_and_get_verifier(state)
            if not code_verifier:
                return _error_page("Invalid or Expired Session", "The authentication session has expired or is invalid.", 400)

            redirect_uri = f"http://{request.url.hostname}:{self.server_port}/auth/callback"
            email = self.authenticator.exchange_code_for_token(code, code_verifier, redirect_uri)
            if not email:
                return _error_page("Authentication Failed", "Could not verify your Google account or you are not authorized.", 403)

            session_id = self.session_manager.create_session(email)
            auth_cookie = secrets.token_urlsafe(32)
            self._authenticated_cookies[auth_cookie] = {
                'email': email,
                'session_id': session_id,
                'expires': datetime.now() + timedelta(hours=24),
            }

            response = RedirectResponse("/", status_code=302)
            response.set_cookie(
                key="onit_auth",
                value=auth_cookie,
                max_age=86400,  # 24 hours
                httponly=True,
                samesite="lax",
            )
            return response

        @app.get("/auth/logout")
        async def oauth_logout(request: Request):
            auth_cookie = request.cookies.get("onit_auth")
            if auth_cookie and auth_cookie in self._authenticated_cookies:
                cookie_data = self._authenticated_cookies[auth_cookie]
                session_id = cookie_data.get('session_id')
                if session_id:
                    self.session_manager.revoke_session(session_id)
                del self._authenticated_cookies[auth_cookie]

            response = RedirectResponse("/", status_code=302)
            response.delete_cookie("onit_auth")
            return response

        @app.get("/auth/check")
        async def check_auth(request: Request):
            email = self._auth_email(request)
            if not email:
                return {"authenticated": False}
            return {"authenticated": True, "email": email}

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(self, loop: asyncio.AbstractEventLoop) -> None:
        """Launch the web server on a background daemon thread."""
        self._loop = loop
        if self.app is None:
            self.build_app()

        print(f"\n{'='*60}")
        print(f"🚀 Launching OnIt Web UI on http://0.0.0.0:{self.server_port}")
        if self.auth_enabled:
            print(f"   OAuth2 Authentication: ENABLED")
            print(f"   Login URL: http://localhost:{self.server_port}/auth/login")
        else:
            print(f"   Authentication: off (no Google OAuth credentials configured)")
        print(f"{'='*60}\n")

        def _run():
            import uvicorn
            uvicorn.run(
                self.app,
                host="0.0.0.0",
                port=self.server_port,
                log_level="info" if self.verbose else "warning",
                access_log=self.verbose,
            )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
