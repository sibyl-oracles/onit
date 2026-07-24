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

A plain FastAPI backend that streams agent output over Server-Sent Events
and serves a static single-page app from src/ui/static/.  Implements the
same interface as ChatUI (text.py) so chat.py and onit.py work without
changes.

SSE event schema (see docs/web-ui-plan.md):
    token         {"delta": str}
    phase_end     {"content": str}
    status        {"text": str}
    done          {"content": str, "files": [...],
                   "elapsed": float, "tok_s": float}
    error         {"message": str}
"""

import asyncio
import ipaddress
import json
import logging
import mimetypes
import os
import queue
import re
import secrets
import shutil
import socket
import threading
import time
import urllib.parse
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Optional

from src.sessions import delete_session as _delete_session_index
from src.sessions import get_session_owner as _get_session_owner
from src.sessions import list_sessions as _list_sessions
from src.sessions import set_session_owner as _set_session_owner
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


class _NoCacheStaticFiles(StaticFiles):
    """Static assets served with no-cache so browsers revalidate on every
    load (cheap 304s via ETag) — otherwise a heuristically cached app.js
    survives deploys and keeps rendering stale branding."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response

# Default web UI branding. When the UI is served through a real domain name
# the domain replaces the brand; a custom web_title still wins for the title.
DEFAULT_TITLE = "OnIt Chat"
DEFAULT_BRAND = "OnIt"
def _is_local_host(host: str) -> bool:
    """True for hostnames that should keep the default OnIt branding:
    localhost and bare IP addresses (an IP is not a brand)."""
    if host.lower() == "localhost":
        return True
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False

# GA4 measurement IDs look like G-XXXXXXXXXX. The ID is echoed to the browser
# and interpolated into a script URL, so anything else is dropped.
_GA_ID_RE = re.compile(r'^G-[A-Z0-9]{4,16}$', re.IGNORECASE)

# Pattern for validating session IDs (UUIDs only)
_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

# Sentinel pushed into the event queue when a chat turn is finished
_END = ("__end__", None)

# Cap on a single uploaded file. Uploads are read fully into memory before
# being written, so an unbounded body is a trivial memory-exhaustion DoS;
# reject anything larger. Override with ONIT_MAX_UPLOAD_MB.
try:
    _MAX_UPLOAD_BYTES = int(float(os.environ.get("ONIT_MAX_UPLOAD_MB", "25")) * 1024 * 1024)
except ValueError:
    _MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Response headers applied to every response — defence in depth alongside the
# reverse proxy. These don't depend on the request, so they're a module const;
# HSTS and CSP (which do vary) are added per-request in the middleware.
_STATIC_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), interest-cohort=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    # Replace uvicorn's "server: uvicorn" so the ASGI stack/version isn't
    # advertised. uvicorn is launched with server_header=False (see launch()).
    "Server": "OnIt",
}

# ── Link verification ───────────────────────────────────────────────────────
# Agents sometimes hallucinate URLs (e.g. https://ge.php, https://manual).
# The UI keeps external links non-clickable until POST /api/verify_links
# confirms they resolve; these limits bound that endpoint.
_VERIFY_MAX_URLS = 20          # per request
_VERIFY_TIMEOUT = 5            # seconds per probe
_VERIFY_TTL = 600              # seconds a verdict stays cached
_VERIFY_CACHE_MAX = 512        # cached verdicts kept in memory

# Hosts the verifier must never probe (loopback, link-local, RFC-1918) —
# the model's output is untrusted, so don't let it steer requests inward.
_PRIVATE_HOST_RE = re.compile(
    r'^(localhost$|127\.|0\.|10\.|192\.168\.|169\.254\.'
    r'|172\.(1[6-9]|2\d|3[01])\.|\[?::1\]?$|\[?f[cde])',
    re.IGNORECASE,
)


def _link_shape_ok(url: str) -> bool:
    """Cheap structural screen before any network probe."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    host = parts.hostname or ""
    # A bare word ("manual") can't be a public site; dotless hosts are
    # either typos or internal names we refuse to probe anyway.
    if "." not in host:
        return False
    return not _PRIVATE_HOST_RE.match(host)


def _sse(event: str, payload: dict) -> str:
    """Format one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _read_capped(file, limit: Optional[int] = None) -> Optional[bytes]:
    """Read an UploadFile into memory, aborting once *limit* bytes are
    exceeded. Returns None when the file is too large so callers can reject it
    without having buffered the whole (possibly huge) body."""
    if limit is None:
        limit = _MAX_UPLOAD_BYTES
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _host_resolves_public(host: str) -> bool:
    """True only if *host* resolves exclusively to public IP addresses.

    The structural screen (_link_shape_ok) rejects literal private hosts, but a
    public-looking name can still resolve to a loopback, RFC-1918, link-local
    or otherwise reserved address (e.g. cloud metadata at 169.254.169.254).
    Resolving here and rejecting any non-global result closes that SSRF gap."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, ValueError):
        return False
    if not infos:
        return False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError:
            return False
        if not ip.is_global or ip.is_reserved:
            return False
    return True


def _remove_session_zip(data_path: str) -> None:
    """Remove the sibling <data_path>.zip that older versions generated via
    zip_code_files, so a cleared/deleted session leaves no stale archive."""
    zip_path = data_path.rstrip(os.sep) + ".zip"
    if os.path.isfile(zip_path):
        try:
            os.remove(zip_path)
        except OSError:
            pass


@dataclass
class ApiSession:
    """Per-browser session state for the web UI."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_path: str = ""
    data_path: str = ""
    owner: Optional[str] = None  # authenticated email; None when auth is off
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
        title: str = DEFAULT_TITLE,
        ga_measurement_id: Optional[str] = None,
        verbose: bool = False,
        require_auth: bool = True,
    ) -> None:
        self.title = title
        self.ga_measurement_id = self._resolve_ga_id(ga_measurement_id)
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

        # --no-login / web_require_auth: false — run an open UI with no login
        # step, even when OAuth credentials are configured.
        if not require_auth:
            if self.auth_enabled:
                print("Login disabled (--no-login): the web UI is open to anyone who can reach it.")
            self.auth_enabled = False

        if self.auth_enabled and not GOOGLE_AUTH_AVAILABLE:
            raise RuntimeError(
                "Web UI login requires the google-auth package. "
                "Install it with: pip install google-auth requests"
            )

        # Sessions must start with a Google login: refuse to serve an open
        # web UI unless the operator explicitly opted out (--no-login).
        if require_auth and not self.auth_enabled:
            raise RuntimeError(
                "The web UI requires Google login, but no OAuth2 credentials are "
                "configured. Set web_google_client_id and web_google_client_secret "
                "(onit --setup), or run with --no-login / web_require_auth: false."
            )

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

        # {url: (ok, expires_at)} — verdicts from /api/verify_links
        self._link_cache: dict[str, tuple[bool, float]] = {}

        self.app: Optional[FastAPI] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _sessions_dir(self) -> str:
        if self.session_path:
            return os.path.dirname(self.session_path)
        return os.path.expanduser("~/.onit/sessions")

    def _session_owner_of(self, session_id: str) -> Optional[str]:
        """Owner recorded for a session, preferring in-memory state."""
        mem = self._web_sessions.get(session_id)
        if mem and mem.owner:
            return mem.owner
        return _get_session_owner(session_id, sessions_dir=self._sessions_dir())

    def _can_access(self, session_id: str, email: Optional[str]) -> bool:
        """True when *email* may touch *session_id*.

        Unowned sessions (created before auth was enabled) are accessible —
        the first authenticated user to use one claims it in
        ``_get_or_create_session``.
        """
        if not self.auth_enabled:
            return True
        recorded = self._session_owner_of(session_id)
        return recorded is None or recorded == email

    def _get_or_create_session(
        self, session_id: str | None = None, owner: Optional[str] = None
    ) -> tuple[str, ApiSession]:
        """Get an existing session or create a new one.

        Restores sessions from disk (JSONL survives restarts); otherwise
        creates a new one, reusing *session_id* when supplied so the browser
        cookie and server-side record stay in sync.

        When auth is enabled, sessions are gated per user: a session id that
        belongs to a different *owner* is treated as nonexistent and the
        caller gets a fresh session; an unowned session is claimed by (and
        thereafter locked to) the first owner who uses it.
        """
        # Reject invalid session IDs to prevent path traversal
        if session_id and not _UUID_RE.match(session_id):
            session_id = None

        # Never hand one user another user's session
        if session_id and not self._can_access(session_id, owner):
            session_id = None

        sessions_dir = self._sessions_dir()

        if session_id and session_id in self._web_sessions:
            session = self._web_sessions[session_id]
            if self.auth_enabled and owner and not session.owner:
                session.owner = owner
                _set_session_owner(session_id, owner, sessions_dir=sessions_dir)
            return session_id, session

        os.makedirs(sessions_dir, exist_ok=True)

        session = ApiSession(session_id=session_id) if session_id else ApiSession()
        if self.auth_enabled and owner:
            session.owner = owner
            _set_session_owner(session.session_id, owner, sessions_dir=sessions_dir)
        session.session_path = os.path.join(sessions_dir, f"{session.session_id}.jsonl")
        if not os.path.exists(session.session_path):
            with open(session.session_path, "w", encoding="utf-8") as f:
                f.write("")
        # Root the per-session dir under the shared data path — the same root
        # the MCP tools server jails writes to — so files generated by tools
        # land where /uploads/{sid}/ serves them (matches the messaging UIs).
        session.data_path = os.path.join(self.data_path, session.session_id)
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
        """Resolve the session from the X-Session-Id header or cookie,
        scoped to the authenticated user when auth is enabled."""
        sid = request.headers.get("x-session-id") or request.cookies.get("onit_session")
        owner = self._auth_email(request) if self.auth_enabled else None
        return self._get_or_create_session(sid, owner=owner)

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

    def tool_log(self, name: str, data, level: str = "info") -> None:
        """Forward real-time log messages from MCP tools (e.g. sandbox output)
        to the developer logs drawer, unwrapping structured MCP payloads."""
        if isinstance(data, dict):
            data = data.get("msg") or data.get("message") or data
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
        cleaned = re.sub(r'(?<![:\w/])/(?:[\w\.\-]+/)+(?=[\w\.\-]+)', '', cleaned)

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
    # Link verification
    # ------------------------------------------------------------------

    def _probe_url(self, url: str) -> bool:
        """Network reachability check: HEAD first, GET fallback (some
        servers reject HEAD). Any 2xx/3xx final status counts as alive."""
        import requests
        # Re-check after DNS resolution: a public-looking host that resolves to
        # a private/reserved address (cloud metadata, intranet) must not be
        # probed. The structural screen only catches literal private hosts.
        host = urllib.parse.urlsplit(url).hostname or ""
        if not _host_resolves_public(host):
            return False
        headers = {"User-Agent": "Mozilla/5.0 (compatible; OnIt-LinkCheck)"}
        try:
            resp = requests.head(url, timeout=_VERIFY_TIMEOUT,
                                 allow_redirects=True, headers=headers)
            if resp.status_code >= 400:
                resp = requests.get(url, timeout=_VERIFY_TIMEOUT,
                                    allow_redirects=True, stream=True,
                                    headers=headers)
                resp.close()
            return resp.status_code < 400
        except requests.RequestException:
            return False

    def _verify_link(self, url: str) -> bool:
        """Verdict for one URL: structural screen, then cached network probe."""
        if not _link_shape_ok(url):
            return False
        now = time.time()
        cached = self._link_cache.get(url)
        if cached and cached[1] > now:
            return cached[0]
        ok = self._probe_url(url)
        if len(self._link_cache) >= _VERIFY_CACHE_MAX:
            # Drop expired verdicts; if none expired, drop the oldest half
            live = {u: v for u, v in self._link_cache.items() if v[1] > now}
            if len(live) >= _VERIFY_CACHE_MAX:
                live = dict(sorted(live.items(), key=lambda kv: kv[1][1])
                            [len(live) // 2:])
            self._link_cache = live
        self._link_cache[url] = (ok, now + _VERIFY_TTL)
        return ok

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
            events.put_nowait(("done", {
                "content": display,
                "files": self._file_infos(file_paths, session.session_id),
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
        # openapi_url=None removes /openapi.json (and the interactive docs,
        # already disabled) so the full API surface — every route, schema and
        # parameter — isn't published to anonymous callers in production.
        app = FastAPI(
            title=self.title, docs_url=None, redoc_url=None, openapi_url=None
        )

        @app.middleware("http")
        async def security_headers_middleware(request: Request, call_next):
            response = await call_next(request)
            for name, value in _STATIC_SECURITY_HEADERS.items():
                response.headers[name] = value
            response.headers["Content-Security-Policy"] = self._csp_header()
            # Only advertise HSTS over a secure connection (behind the TLS
            # proxy request.url.scheme is https via proxy_headers); sending it
            # on plain-http dev/localhost would wrongly pin those hosts.
            proto = request.headers.get("x-forwarded-proto", request.url.scheme)
            if proto == "https":
                response.headers["Strict-Transport-Security"] = (
                    "max-age=63072000; includeSubDomains"
                )
            return response

        @app.middleware("http")
        async def session_cookie_middleware(request: Request, call_next):
            response = await call_next(request)
            # A chat session only starts after login: never hand a session
            # cookie to an unauthenticated visitor when auth is on.
            if self.auth_enabled and not self._auth_email(request):
                return response
            if not request.cookies.get("onit_session"):
                response.set_cookie(
                    key="onit_session",
                    value=str(uuid.uuid4()),
                    max_age=86400 * 7,   # 7 days
                    httponly=True,
                    samesite="lax",
                    secure=request.url.scheme == "https",
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
            authenticated = bool(email) or not self.auth_enabled
            title, brand = self._branding(request)
            return {
                "title": title,
                "brand": brand,
                # The analytics ID is only handed to authenticated clients (or
                # an open, no-login UI). /api/config is reachable pre-login so
                # the SPA can render the login view; withholding ga_id there
                # keeps it out of anonymous responses. The SPA re-fetches config
                # after the OAuth redirect, so analytics still loads once in.
                "ga_id": self.ga_measurement_id if authenticated else None,
                "agent": self.agent_cursor,
                "auth_enabled": self.auth_enabled,
                "authenticated": authenticated,
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
            _remove_session_zip(session.data_path)
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
            content = await _read_capped(file)
            if content is None:
                return JSONResponse(
                    {"detail": f"File exceeds the {_MAX_UPLOAD_BYTES // (1024*1024)} MB limit"},
                    status_code=413,
                )
            with open(dest, "wb") as f:
                f.write(content)
            return {
                "name": safe_name,
                "url": f"/uploads/{sid}/{urllib.parse.quote(safe_name)}",
                "session_id": sid,
            }

        @app.post("/api/verify_links")
        async def api_verify_links(request: Request):
            body = await request.json()
            urls = body.get("urls")
            if not isinstance(urls, list):
                return JSONResponse({"detail": "urls must be a list"}, status_code=400)
            urls = [str(u) for u in urls[:_VERIFY_MAX_URLS]]
            verdicts = await asyncio.gather(
                *(asyncio.to_thread(self._verify_link, u) for u in urls)
            )
            return {"results": dict(zip(urls, verdicts))}

        @app.get("/api/logs")
        async def api_logs():
            return {"logs": list(self.execution_logs)[-50:]}

        @app.get("/api/sessions")
        async def api_sessions(request: Request):
            owner = self._auth_email(request) if self.auth_enabled else None
            sessions = _list_sessions(
                sessions_dir=self._sessions_dir(), limit=50, owner=owner
            )
            for s in sessions:
                mem = self._web_sessions.get(s["session_id"])
                s["processing"] = bool(mem and mem.processing)
            return {"sessions": sessions}

        @app.post("/api/sessions/new")
        async def api_new_session(request: Request):
            owner = self._auth_email(request) if self.auth_enabled else None
            sid, _session = self._get_or_create_session(str(uuid.uuid4()), owner=owner)
            return {"session_id": sid}

        @app.delete("/api/sessions")
        async def api_delete_all_sessions(request: Request):
            owner = self._auth_email(request) if self.auth_enabled else None
            if any(
                s.processing for s in self._web_sessions.values()
                if self._can_access(s.session_id, owner)
            ):
                return JSONResponse(
                    {"detail": "A task is still running; stop it first"},
                    status_code=409,
                )
            sessions_dir = self._sessions_dir()
            indexed = _list_sessions(sessions_dir=sessions_dir, limit=1_000_000,
                                     owner=owner)
            sids = {s["session_id"] for s in indexed} | {
                sid for sid, s in self._web_sessions.items()
                if not self.auth_enabled or s.owner == owner
            }
            for sid in sids:
                session = self._web_sessions.pop(sid, None)
                data_path = (session.data_path if session and session.data_path
                             else os.path.join(self.data_path, sid))
                shutil.rmtree(data_path, ignore_errors=True)
                _remove_session_zip(data_path)
                _delete_session_index(sid, sessions_dir=sessions_dir)
            return {"deleted": len(sids)}

        @app.delete("/api/sessions/{session_id}")
        async def api_delete_session(session_id: str, request: Request):
            if not _UUID_RE.match(session_id):
                return JSONResponse({"detail": "Invalid session id"}, status_code=400)
            owner = self._auth_email(request) if self.auth_enabled else None
            if not self._can_access(session_id, owner):
                # 404, not 403 — don't confirm someone else's session exists
                return JSONResponse({"detail": "Session not found"}, status_code=404)
            session = self._web_sessions.pop(session_id, None)
            data_path = (session.data_path if session and session.data_path
                         else os.path.join(self.data_path, session_id))
            shutil.rmtree(data_path, ignore_errors=True)
            _remove_session_zip(data_path)
            _delete_session_index(session_id, sessions_dir=self._sessions_dir())
            return {"deleted": True}

        @app.patch("/api/sessions/{session_id}")
        async def api_rename_session(session_id: str, request: Request):
            if not _UUID_RE.match(session_id):
                return JSONResponse({"detail": "Invalid session id"}, status_code=400)
            owner = self._auth_email(request) if self.auth_enabled else None
            if not self._can_access(session_id, owner):
                return JSONResponse({"detail": "Session not found"}, status_code=404)
            body = await request.json()
            tag = (body.get("tag") or "").strip()
            if not tag:
                return JSONResponse({"detail": "Missing tag"}, status_code=400)
            result = _tag_session(session_id, tag, sessions_dir=self._sessions_dir())
            if result is True:
                return {"renamed": True}
            detail = result if isinstance(result, str) else "Session not found"
            return JSONResponse({"detail": detail}, status_code=400)

    def _setup_file_routes(self, app: FastAPI):
        """Routes to serve and receive files scoped per session."""

        @app.get("/uploads/{session_id}/{filename}")
        async def serve_upload(session_id: str, filename: str, request: Request):
            # Use basename to prevent path traversal
            safe_name = os.path.basename(urllib.parse.unquote(filename))
            if session_id not in self._web_sessions:
                return Response(content="Session not found", status_code=404)
            owner = self._auth_email(request) if self.auth_enabled else None
            if not self._can_access(session_id, owner):
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
            content = await _read_capped(file)
            if content is None:
                return Response(content="File too large", status_code=413)
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

        @app.get("/robots.txt")
        async def robots_txt():
            # Keep the API, auth and per-session upload paths out of crawler
            # indexes. This is guidance for well-behaved bots, not access
            # control (those paths are already auth-gated).
            body = (
                "User-agent: *\n"
                "Disallow: /api/\n"
                "Disallow: /auth/\n"
                "Disallow: /uploads/\n"
            )
            return Response(content=body, media_type="text/plain")

        @app.get("/.well-known/security.txt")
        async def security_txt():
            # Publish a disclosure channel only when the operator has set a
            # contact (ONIT_SECURITY_CONTACT, e.g. "mailto:security@example.com"
            # or an https URL); otherwise stay 404 rather than invent one.
            contact = os.environ.get("ONIT_SECURITY_CONTACT", "").strip()
            if not contact:
                return Response(content='{"detail":"Not Found"}',
                                media_type="application/json", status_code=404)
            expires = (datetime.now() + timedelta(days=365)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            body = f"Contact: {contact}\nExpires: {expires}\n"
            return Response(content=body, media_type="text/plain")

        if os.path.isdir(STATIC_DIR):
            app.mount("/static", _NoCacheStaticFiles(directory=STATIC_DIR), name="static")

    def _csp_header(self) -> str:
        """Content-Security-Policy for the SPA.

        Scripts and styles are same-origin (served from /static); markdown is
        rendered client-side and sanitised with DOMPurify. Google Analytics,
        when configured, injects gtag from googletagmanager.com and beacons to
        google-analytics.com — those hosts are whitelisted only when analytics
        is on. 'unsafe-inline' is allowed for styles (highlight.js themes and
        rendered-markdown styling) but never for scripts."""
        script_src = ["'self'"]
        connect_src = ["'self'"]
        img_src = ["'self'", "data:", "blob:"]
        if self.ga_measurement_id:
            script_src.append("https://www.googletagmanager.com")
            connect_src += [
                "https://www.google-analytics.com",
                "https://*.google-analytics.com",
                "https://*.googletagmanager.com",
            ]
            img_src.append("https://www.google-analytics.com")
        directives = [
            "default-src 'self'",
            "base-uri 'self'",
            "object-src 'none'",
            "frame-ancestors 'none'",
            "form-action 'self'",
            "script-src " + " ".join(script_src),
            "style-src 'self' 'unsafe-inline'",
            "img-src " + " ".join(img_src),
            "connect-src " + " ".join(connect_src),
            "font-src 'self' data:",
        ]
        return "; ".join(directives)

    @staticmethod
    def _resolve_ga_id(configured: Optional[str]) -> Optional[str]:
        """Google Analytics 4 measurement ID, or None when analytics is off.
        web_ga_measurement_id in the config wins; the ONIT_GA_MEASUREMENT_ID
        env var covers the docker deployment (set it in .env next to
        ONIT_DOMAIN). Malformed IDs are dropped with a warning rather than
        echoed into the page."""
        ga_id = (configured or os.environ.get("ONIT_GA_MEASUREMENT_ID", "")).strip()
        if not ga_id:
            return None
        if not _GA_ID_RE.match(ga_id):
            print(f"Ignoring malformed GA measurement ID: {ga_id!r} (expected G-XXXXXXXXXX)")
            return None
        return ga_id.upper()

    def _branding(self, request: Request) -> tuple[str, str]:
        """(title, brand) shown by the SPA — page/login title, and the name
        used in the sidebar and the composer hint ("<brand> can make
        mistakes"). Whenever the UI is served through a real domain name
        (e.g. mychat.ai) the domain becomes the brand; localhost and bare-IP
        access keep the default OnIt. A custom web_title always wins for the
        title but does not stop the brand swap. The domain comes from
        ONIT_DOMAIN (docker-compose .env), then ONIT_PUBLIC_URL, then the
        Host header the browser actually used."""
        domain = os.environ.get("ONIT_DOMAIN", "").strip()
        if not domain:
            public = os.environ.get("ONIT_PUBLIC_URL", "")
            domain = urllib.parse.urlparse(public).hostname or "" if public else ""
        if not domain:
            domain = request.url.hostname or ""
        if not domain or _is_local_host(domain):
            return self.title, DEFAULT_BRAND
        title = self.title if self.title != DEFAULT_TITLE else domain
        return title, domain

    def _external_base_url(self, request: Request) -> str:
        """Base URL as seen by the browser, for OAuth redirect URIs.

        ONIT_PUBLIC_URL (e.g. https://mychat.ai) wins when set. Otherwise the
        URL comes from the request itself, which reflects X-Forwarded-Proto and
        the original Host header when uvicorn runs with proxy_headers behind a
        TLS-terminating reverse proxy — so redirects stay https:// instead of
        leaking the internal http://host:port address.
        """
        configured = os.environ.get("ONIT_PUBLIC_URL", "").rstrip("/")
        if configured:
            return configured
        return str(request.base_url).rstrip("/")

    def _setup_oauth_routes(self, app: FastAPI):
        """Add OAuth2 login/callback/logout routes."""
        if not self.auth_enabled:
            return

        @app.get("/auth/login")
        async def oauth_login(request: Request):
            state, _code_verifier, code_challenge = self.oauth_flow_manager.create_flow()
            redirect_uri = self._external_base_url(request) + "/auth/callback"
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

            # Must match the redirect_uri sent at /auth/login exactly, or
            # Google rejects the token exchange.
            redirect_uri = self._external_base_url(request) + "/auth/callback"
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
                secure=request.url.scheme == "https",
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

        public_url = os.environ.get("ONIT_PUBLIC_URL", "").rstrip("/")
        print(f"\n{'='*60}")
        print(f"🚀 Launching OnIt Web UI on http://0.0.0.0:{self.server_port}")
        if public_url:
            print(f"   Public URL: {public_url}")
        if self.auth_enabled:
            print(f"   OAuth2 Authentication: ENABLED")
            print(f"   Login URL: {public_url or f'http://localhost:{self.server_port}'}/auth/login")
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
                # Don't emit uvicorn's "server: uvicorn" header; the security
                # middleware sets a generic Server value instead so the ASGI
                # stack isn't advertised.
                server_header=False,
                # Honor X-Forwarded-Proto/For from a TLS-terminating reverse
                # proxy so request.base_url is https://. The port must not be
                # exposed publicly, or clients could spoof these headers.
                proxy_headers=True,
                forwarded_allow_ips="*",
            )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
