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

Gradio-based web chat UI for OnIt.
Implements the same interface as ChatUI (text.py) so chat.py and onit.py
work without changes.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import time
import threading
import urllib.parse
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

try:
    import gradio as gr
    from fastapi import FastAPI, Request, Response, UploadFile, File
    from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
except ImportError:
    raise ImportError(
        "Gradio is required for the web UI. Install it with: pip install gradio"
    )

try:
    import requests as http_requests
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False


class NullConsole:
    """Stub console so chat.py kwargs work when chat_ui.console is passed."""
    def print(self, *args, **kwargs):
        pass

    def clear(self, *args, **kwargs):
        pass


class SessionManager:
    """Manages authenticated user sessions."""
    def __init__(self, session_duration_hours: int = 24):
        self.sessions: dict[str, dict] = {}
        self.session_duration = timedelta(hours=session_duration_hours)

    def create_session(self, email: str) -> str:
        """Create a new session for an authenticated user."""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            "email": email,
            "created": datetime.now(),
            "expires": datetime.now() + self.session_duration,
        }
        return session_id

    def verify_session(self, session_id: str) -> Optional[str]:
        """Verify a session and return the email if valid."""
        if not session_id or session_id not in self.sessions:
            return None
        session = self.sessions[session_id]
        if datetime.now() > session["expires"]:
            del self.sessions[session_id]
            return None
        return session["email"]

    def revoke_session(self, session_id: str) -> None:
        """Revoke a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]


class OAuthFlowManager:
    """Manages OAuth2 PKCE flows."""
    def __init__(self):
        # Store active OAuth flows: state -> {code_verifier, created_at}
        self.active_flows: dict[str, dict] = {}

    def create_flow(self) -> tuple[str, str, str]:
        """Create a new OAuth flow with PKCE.

        Returns:
            tuple: (state, code_verifier, code_challenge)
        """
        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)

        # Generate PKCE code verifier and challenge
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip('=')

        # Store the flow
        self.active_flows[state] = {
            'code_verifier': code_verifier,
            'created_at': datetime.now()
        }

        # Clean up old flows (older than 10 minutes)
        self._cleanup_old_flows()

        return state, code_verifier, code_challenge

    def verify_and_get_verifier(self, state: str) -> Optional[str]:
        """Verify state and return code_verifier if valid."""
        flow = self.active_flows.get(state)
        if not flow:
            return None

        # Check if flow is too old (10 minutes)
        if datetime.now() - flow['created_at'] > timedelta(minutes=10):
            del self.active_flows[state]
            return None

        code_verifier = flow['code_verifier']
        del self.active_flows[state]  # One-time use
        return code_verifier

    def _cleanup_old_flows(self):
        """Remove flows older than 10 minutes."""
        cutoff = datetime.now() - timedelta(minutes=10)
        expired = [s for s, f in self.active_flows.items() if f['created_at'] < cutoff]
        for state in expired:
            del self.active_flows[state]


class GoogleAuthenticator:
    """Handles Google OAuth2 authentication."""
    def __init__(self, client_id: str, client_secret: str, allowed_emails: Optional[list[str]] = None):
        if not GOOGLE_AUTH_AVAILABLE:
            raise ImportError(
                "Google authentication requires google-auth and requests. "
                "Install them with: pip install google-auth requests"
            )
        self.client_id = client_id
        self.client_secret = client_secret
        self.allowed_emails = set(allowed_emails) if allowed_emails else None

    def verify_token(self, token: str) -> Optional[str]:
        """Verify Google ID token and return email if valid."""
        try:
            idinfo = id_token.verify_oauth2_token(
                token, google_requests.Request(), self.client_id
            )
            email = idinfo.get("email")

            # Check if email is verified
            if not idinfo.get("email_verified"):
                return None

            # Check if email is in allowed list (if specified)
            if self.allowed_emails and not self._is_email_allowed(email):
                return None

            return email
        except Exception:
            return None

    def _is_email_allowed(self, email: str) -> bool:
        """Check if email matches allowed patterns.

        Supports:
        - Exact matches: "user@example.com"
        - Domain wildcards: "*@example.com" (allows all users from example.com)
        """
        if not self.allowed_emails:
            return True

        for pattern in self.allowed_emails:
            # Exact match
            if email == pattern:
                return True

            # Domain wildcard match (e.g., *@sibyl.ai)
            if pattern.startswith("*@"):
                domain = pattern[2:]  # Remove "*@" prefix
                if email.endswith("@" + domain):
                    return True

        return False

    def exchange_code_for_token(self, code: str, code_verifier: str, redirect_uri: str) -> Optional[str]:
        """Exchange authorization code for ID token using PKCE.

        Args:
            code: Authorization code from Google
            code_verifier: PKCE code verifier
            redirect_uri: Redirect URI used in the auth request

        Returns:
            Email if successful, None otherwise
        """
        token_url = "https://oauth2.googleapis.com/token"

        data = {
            'code': code,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code',
            'code_verifier': code_verifier,
        }

        try:
            response = http_requests.post(token_url, data=data)
            response.raise_for_status()

            tokens = response.json()
            id_token_str = tokens.get('id_token')

            if not id_token_str:
                return None

            # Verify and extract email from ID token
            return self.verify_token(id_token_str)

        except Exception as e:
            print(f"Error exchanging code for token: {e}")
            return None


@dataclass
class WebSession:
    """Per-browser-tab session state for the web UI."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_path: str = ""
    data_path: str = ""
    pending_responses: list = field(default_factory=list)
    processing: bool = False
    spinner_shown: bool = False
    spinner_step: int = 0
    spinner_tick: int = 0
    safety_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=10))
    created: datetime = field(default_factory=datetime.now)
    streaming_content: str = ""
    streaming_active: bool = False
    streaming_committed: bool = False  # True once streamed partial has been saved as a permanent message


class WebChatUI:
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

        # Guard against missing google-auth package
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

        # Per-browser-session state (replaces shared state)
        self._web_sessions: dict[str, WebSession] = {}
        self._onit = None  # set by OnIt after creation
        self.execution_logs: deque[dict] = deque(maxlen=100)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._spinner_ticks_per_message: int = 6  # change message every ~3s (6 × 0.5s poll)
        self._spinner_messages: list[str] = [
            "Analyzing your request...",
            "Thinking...",
            "Planning...",
            "Reasoning...",
            "Searching for answers...",
            "Connecting the dots...",
            "Analyzing the details...",
            "Crunching the numbers...",
            "Almost there...",
            "Piecing it together...",
            "Diving deeper...",
            "Finalizing...",
        ]

        # Store authenticated sessions with cookies
        # Format: {cookie_value: {email, session_id, expires}}
        self._authenticated_cookies: dict[str, dict] = {}

        # gradio app
        self.app: Optional[gr.Blocks] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    # Pattern for validating session IDs (UUIDs only)
    _UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

    def _get_or_create_session(self, session_id: str | None = None) -> tuple[str, WebSession]:
        """Get an existing session or create a new one.

        If *session_id* is provided and the session exists in memory, return it.
        If the session is not in memory but a JSONL file exists on disk (e.g.
        after a server restart or browser refresh), restore it.  Otherwise
        create a brand-new session, reusing *session_id* when supplied so the
        browser cookie and server-side record stay in sync.

        Returns (session_id, WebSession) tuple.
        """
        # Reject invalid session IDs to prevent path traversal
        if session_id and not self._UUID_RE.match(session_id):
            session_id = None

        if session_id and session_id in self._web_sessions:
            return session_id, self._web_sessions[session_id]

        # Derive sessions dir from the configured session_path
        if self.session_path:
            sessions_dir = os.path.dirname(self.session_path)
        else:
            sessions_dir = os.path.expanduser("~/.onit/sessions")
        os.makedirs(sessions_dir, exist_ok=True)

        # Try to restore session from disk (survives server restarts)
        if session_id:
            session_file = os.path.join(sessions_dir, f"{session_id}.jsonl")
            if os.path.exists(session_file):
                session = WebSession(session_id=session_id)
                session.session_path = session_file
                session.data_path = str(Path(tempfile.gettempdir()) / "onit" / "data" / session_id)
                os.makedirs(session.data_path, exist_ok=True)
                self._web_sessions[session_id] = session
                logger.info("Restored web session %s from disk", session_id)
                return session_id, session

        # Create new session, reusing the cookie session_id when available
        session = WebSession(session_id=session_id) if session_id else WebSession()
        session.session_path = os.path.join(sessions_dir, f"{session.session_id}.jsonl")
        if not os.path.exists(session.session_path):
            with open(session.session_path, "w", encoding="utf-8") as f:
                f.write("")
        session.data_path = str(Path(tempfile.gettempdir()) / "onit" / "data" / session.session_id)
        os.makedirs(session.data_path, exist_ok=True)

        self._web_sessions[session.session_id] = session
        logger.info("Created new web session %s", session.session_id)

        # Cleanup old sessions (older than 24h)
        now = datetime.now()
        expired = [
            sid for sid, s in self._web_sessions.items()
            if (now - s.created) > timedelta(hours=24)
        ]
        for sid in expired:
            del self._web_sessions[sid]

        return session.session_id, session

    # ------------------------------------------------------------------
    # Interface methods (matching ChatUI contract)
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

    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        response: str,
        elapsed: str = "",
    ) -> None:
        """No-op for web UI. Responses are routed through per-session state."""
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

    def render(self, thinking: bool = False):
        """No-op for web UI; Gradio handles rendering via polling."""
        pass

    def stop_status(self) -> None:
        """No-op for web UI."""
        pass

    def _load_chat_from_session(self, session_path: str | None = None, data_path: str | None = None, session_id: str | None = None) -> list:
        """Load chat history from the JSONL session file for display in the chatbot."""
        effective_path = session_path or self.session_path
        messages = []
        if not effective_path or not os.path.exists(effective_path):
            return messages
        try:
            with open(effective_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if "task" in entry and "response" in entry:
                            # Replace raw absolute file paths with friendly basenames
                            task_display = entry["task"]
                            task_display = re.sub(
                                r'Relevant files:\s*/[^\s]+/([^\s/]+)',
                                lambda m: f'📎 {m.group(1)}',
                                task_display,
                            )
                            messages.append(gr.ChatMessage(role="user", content=task_display))
                            display, file_paths = self._extract_file_paths(entry["response"], data_path=data_path, session_id=session_id)
                            messages.append(gr.ChatMessage(role="assistant", content=display))
                            for fpath in file_paths:
                                if os.path.isfile(fpath):
                                    messages.append(
                                        gr.ChatMessage(
                                            role="assistant",
                                            content=gr.FileData(path=fpath, mime_type=None),
                                        )
                                    )
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return messages

    # ------------------------------------------------------------------
    # Gradio app
    # ------------------------------------------------------------------

    def _format_logs(self) -> str:
        level_icons = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "debug": "🔍"}
        lines = []
        for log in list(self.execution_logs)[-20:]:
            icon = level_icons.get(log["level"], "•")
            lines.append(f"{icon} [{log['timestamp']}] {log['message']}")
        return "\n".join(lines)

    def build_app(self) -> gr.Blocks:
        """Construct the Gradio Blocks interface."""
        custom_css = """
        .gradio-container {
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 0.875rem;
            height: 100vh !important;
            max-height: 100vh !important;
            display: flex !important;
            flex-direction: column !important;
            padding: 8px 16px !important;
        }
        .chatbot { flex: 1 1 auto !important; min-height: 0 !important; }
        .chatbot .message { font-size: 0.875rem !important; line-height: 1.3 !important; padding: 4px 8px !important; }
        .chatbot .message p { margin: 1px 0 !important; font-size: 0.875rem !important; }
        .chatbot .message pre, .chatbot .message code { font-size: 0.825rem !important; }
        .input-row { align-items: stretch !important; }
        .input-row textarea { font-size: 1rem !important; }
        .action-btn { min-width: 0 !important; max-width: 80px !important; font-size: 0.875rem !important; padding: 0 !important; height: auto !important; }
        .action-btn button { font-size: 0.875rem !important; padding: 0 10px !important; height: 100% !important; width: 100% !important; }
        .login-container { max-width: 500px !important; margin: 100px auto !important; text-align: center !important; }
        .google-signin { margin: 20px 0 !important; }
        button[title="Share"], .share-button, [class*="share"] { display: none !important; }
        button[aria-label="Share"] { display: none !important; }
        .message-buttons-bot button:not([title="Copy"]):not([aria-label="Copy"]) { display: none !important; }
        .bot .actions button:not([title="Copy"]):not([aria-label="Copy"]) { display: none !important; }
        """
        self._custom_css = custom_css
        # JavaScript to prevent auto-scroll when user has scrolled up.
        # Gradio re-renders the chatbot on every update, which can reset
        # scroll position. This script observes DOM mutations on the chat
        # scroll container and restores the previous scroll position when
        # the user has scrolled away from the bottom.
        scroll_js = """
        () => {
            function attach() {
                // The scrollable chat container has class bubble-wrap or
                // panel-wrap inside the .chatbot element.
                const scroller =
                    document.querySelector('.chatbot .bubble-wrap') ||
                    document.querySelector('.chatbot .panel-wrap') ||
                    document.querySelector('.chatbot .wrapper') ||
                    document.querySelector('.chatbot');
                if (!scroller) { setTimeout(attach, 500); return; }

                let userScrolledUp = false;
                let savedScrollTop = scroller.scrollTop;
                let ignoreScroll = false;

                scroller.addEventListener('scroll', () => {
                    if (ignoreScroll) return;
                    const gap = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
                    // Very small threshold: even 2px of scroll-up is respected
                    userScrolledUp = gap > 2;
                    savedScrollTop = scroller.scrollTop;
                }, { passive: true });

                let rafPending = false;
                const observer = new MutationObserver(() => {
                    if (rafPending) return;
                    rafPending = true;
                    requestAnimationFrame(() => {
                        rafPending = false;
                        if (userScrolledUp) {
                            // Restore position so DOM changes don't yank user down
                            ignoreScroll = true;
                            scroller.scrollTop = savedScrollTop;
                            ignoreScroll = false;
                        } else {
                            // User is at the bottom – follow new content
                            ignoreScroll = true;
                            scroller.scrollTop = scroller.scrollHeight;
                            ignoreScroll = false;
                            savedScrollTop = scroller.scrollTop;
                        }
                    });
                });

                observer.observe(scroller, { childList: true, subtree: true });
            }
            setTimeout(attach, 1500);
        }
        """
        with gr.Blocks(title=self.title, analytics_enabled=False, js=scroll_js) as app:
            # Session state
            session_state = gr.State(value=None)
            authenticated_email = gr.State(value=None)

            # Authentication UI (only shown if auth is enabled)
            if self.auth_enabled:
                with gr.Column(visible=True, elem_classes=["login-container"]) as login_view:
                    gr.Markdown(f"# 🔐 {self.title}")
                    gr.Markdown("### Sign in to continue")
                    gr.Markdown(
                        """
                        <div style="margin: 40px 0;">
                        <a href="/auth/login" style="display: inline-flex; align-items: center; gap: 16px; padding: 18px 36px; background: #4285f4; color: white; text-decoration: none; border-radius: 4px; font-weight: 500; font-size: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                        <svg style="width: 24px; height: 24px; flex-shrink: 0; overflow: visible;" viewBox="0 0 24 24"><path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
                        <span>Sign in with Google</span>
                        </a>
                        </div>

                        <p style="color: #666; font-size: 14px; margin-top: 20px;">
                        Click the button above to authenticate with your Google account.<br/>
                        Your session will remain active for 24 hours.
                        </p>
                        """
                    )
                    login_status = gr.Markdown("")

            # Chat UI
            with gr.Column(visible=not self.auth_enabled) as chat_view:
                with gr.Row():
                    with gr.Column(scale=8):
                        gr.Markdown(f"## {self.title}")
                    if self.auth_enabled:
                        with gr.Column(scale=2):
                            user_info = gr.Markdown("")

                chatbot = gr.Chatbot(
                    label="Chat",
                    height="calc(100vh - 220px)",
                    buttons=["copy", "clear"],
                    autoscroll=True,
                )

                with gr.Row(elem_classes=["input-row"]):
                    msg_input = gr.Textbox(
                        placeholder="Type a message...",
                        show_label=False,
                        scale=8,
                        container=False,
                    )
                    upload_btn = gr.UploadButton(
                        label="Upload",
                        file_types=None,
                        file_count="single",
                        scale=1,
                        min_width=60,
                        elem_classes=["action-btn"],
                    )
                    stop_btn = gr.Button(
                        "Stop",
                        visible=False,
                        scale=1,
                        min_width=60,
                        elem_classes=["action-btn"],
                    )

                upload_indicator = gr.Markdown("", visible=False)

                with gr.Accordion(
                    "Execution Logs", open=False, visible=self.verbose
                ) as logs_accordion:
                    logs_display = gr.Markdown(
                        value="*No execution logs yet.*",
                        elem_id="execution-logs",
                    )

                gr.Markdown(
                    "<p style='text-align: center; color: #999; font-size: 0.95em; margin-top: 4px;'>"
                    "<a href='https://github.com/sibyl-oracles/onit' target='_blank' style='color: #999; text-decoration: none;'>OnIt</a>"
                    " may produce inaccurate information. Verify important details independently.</p>"
                )

            # state for uploaded file path
            uploaded_file_state = gr.State(value=None)

            # -- callbacks --
            # Note: Authentication is handled via OAuth redirect flow in FastAPI routes
            # No Gradio callbacks needed for login/logout
            def handle_upload(file, sess_id):
                if file is None:
                    return None, gr.update(value="", visible=False), sess_id
                # Gradio 6+ returns a filepath string; older versions return a file object
                file_path = file if isinstance(file, (str, os.PathLike)) else file.name
                file_path = str(file_path)
                # Use per-session data path
                _, session = self._get_or_create_session(sess_id)
                os.makedirs(session.data_path, exist_ok=True)
                fname = os.path.basename(file_path)
                dest = os.path.join(session.data_path, fname)
                shutil.copy2(file_path, dest)
                indicator = f"<p style='color: #4caf50; font-size: 0.8em; margin: 2px 0;'>📎 {fname}</p>"
                return dest, gr.update(value=indicator, visible=True), sess_id

            def handle_send(user_msg, uploaded_path, history, sess_id, request: gr.Request = None):
                hide_indicator = gr.update(value="", visible=False)
                # Verify authentication if enabled
                if self.auth_enabled:
                    auth_cookie = request.cookies.get("onit_auth") if request else None
                    if not auth_cookie or auth_cookie not in self._authenticated_cookies:
                        error_msg = gr.ChatMessage(
                            role="assistant",
                            content="❌ Session expired or invalid. Please login again."
                        )
                        return history + [error_msg], user_msg, uploaded_path, hide_indicator, sess_id

                    cookie_data = self._authenticated_cookies[auth_cookie]
                    if datetime.now() > cookie_data['expires']:
                        del self._authenticated_cookies[auth_cookie]
                        error_msg = gr.ChatMessage(
                            role="assistant",
                            content="❌ Session expired or invalid. Please login again."
                        )
                        return history + [error_msg], user_msg, uploaded_path, hide_indicator, sess_id

                if not user_msg and not uploaded_path:
                    return history, "", None, hide_indicator, sess_id

                sess_id, session = self._get_or_create_session(sess_id)

                display_msg = user_msg or ""
                queue_msg = user_msg or ""
                if uploaded_path:
                    fname = os.path.basename(uploaded_path)
                    file_url = f"/uploads/{sess_id}/{fname}"
                    display_msg += f"\n📎 [{fname}]({file_url})"
                    queue_msg += f"\nRelevant files: {uploaded_path}"

                # Route user message through per-session pending_responses
                session.pending_responses.append(
                    gr.ChatMessage(role="user", content=display_msg)
                )

                # Fire-and-forget: call process_task() directly (like Telegram gateway)
                session.processing = True
                if self._loop and self._onit:
                    async def _run_task(s=session, task=queue_msg):
                        try:
                            def _on_stream_token(_token, full_content):
                                s.streaming_content = full_content
                                s.streaming_active = True

                            def _on_stream_complete(_content, _tok_s):
                                s.streaming_active = False

                            _stats = {}
                            _task_start = time.monotonic()
                            response = await self._onit.process_task(
                                task,
                                session_path=s.session_path,
                                data_path=s.data_path,
                                safety_queue=s.safety_queue,
                                stream_callback=_on_stream_token,
                                stream_complete_callback=_on_stream_complete,
                                stats=_stats,
                            )
                            _task_elapsed = time.monotonic() - _task_start
                            s.streaming_active = False
                            s.streaming_content = ""
                            tok_s = _stats.get("tokens_per_second", 0)
                            if response:
                                display, file_paths = self._extract_file_paths(
                                    response, data_path=s.data_path, session_id=s.session_id
                                )
                                _footer_parts = []
                                if _task_elapsed > 0:
                                    _footer_parts.append(f"{_task_elapsed:.2f}s")
                                if tok_s > 0:
                                    _footer_parts.append(f"{tok_s:.1f} tok/s")
                                if _footer_parts:
                                    display += f"\n\n---\n*{' · '.join(_footer_parts)}*"
                                s.pending_responses.append(
                                    gr.ChatMessage(role="assistant", content=display)
                                )
                                for fpath in file_paths:
                                    if os.path.isfile(fpath):
                                        s.pending_responses.append(
                                            gr.ChatMessage(
                                                role="assistant",
                                                content=gr.FileData(path=fpath, mime_type=None),
                                            )
                                        )
                            else:
                                s.pending_responses.append(
                                    gr.ChatMessage(role="assistant", content="I'm sorry, I couldn't process your request. Please try again.")
                                )
                        except Exception as e:
                            logger.error("Error processing task: %s", e)
                            s.pending_responses.append(
                                gr.ChatMessage(role="assistant", content=f"Error: {e}")
                            )
                        finally:
                            s.streaming_active = False
                            s.streaming_content = ""
                            s.processing = False

                    asyncio.run_coroutine_threadsafe(_run_task(), self._loop)

                return history, "", None, hide_indicator, sess_id

            def handle_stop(sess_id):
                if sess_id and sess_id in self._web_sessions:
                    session = self._web_sessions[sess_id]
                    if self._loop:
                        self._loop.call_soon_threadsafe(
                            session.safety_queue.put_nowait, True
                        )
                return gr.update(visible=False)

            def poll_response(history, sess_id):
                if not sess_id or sess_id not in self._web_sessions:
                    logs_md = self._format_logs() if self.execution_logs else "*No execution logs yet.*"
                    return history, gr.update(visible=False), logs_md, sess_id

                session = self._web_sessions[sess_id]

                has_new_responses = bool(session.pending_responses)
                was_spinner_shown = session.spinner_shown
                chatbot_changed = False

                # Remove spinner placeholder before appending real responses or when done
                if session.spinner_shown and (session.pending_responses or not session.processing):
                    if history:
                        history = history[:-1]  # remove spinner
                    session.spinner_shown = False
                    chatbot_changed = True
                    # Also remove the committed partial — the final response
                    # contains the complete text so the partial is redundant.
                    if session.streaming_committed and history:
                        history = history[:-1]  # remove committed partial
                        session.streaming_committed = False
                    if not session.processing:
                        session.spinner_step = 0
                        session.spinner_tick = 0
                        session.streaming_committed = False

                while session.pending_responses:
                    resp = session.pending_responses.pop(0)
                    history = history + [resp]
                    chatbot_changed = True

                # Show streaming content or spinner while processing
                if session.processing:
                    if session.streaming_active and session.streaming_content:
                        # Show live streaming content (replaces spinner)
                        session.streaming_committed = False
                        stream_msg = gr.ChatMessage(
                            role="assistant",
                            content=session.streaming_content,
                        )
                        if session.spinner_shown:
                            history = history[:-1] + [stream_msg]
                        else:
                            history = history + [stream_msg]
                            session.spinner_shown = True
                        chatbot_changed = True
                    else:
                        # Streaming stopped but still processing (tool call phase).
                        # Commit the streamed partial as a permanent message so it
                        # stays visible, then show the spinner *below* it.
                        if not session.streaming_committed and session.streaming_content:
                            # Replace the live-stream placeholder with a permanent message
                            if session.spinner_shown and history:
                                history = history[:-1]
                                session.spinner_shown = False
                            committed_msg = gr.ChatMessage(
                                role="assistant",
                                content=session.streaming_content,
                            )
                            history = history + [committed_msg]
                            session.streaming_committed = True
                            session.streaming_content = ""
                            chatbot_changed = True

                        # Show spinner (below the committed message if any)
                        session.spinner_tick += 1
                        spinner_text_changed = session.spinner_tick >= self._spinner_ticks_per_message
                        if spinner_text_changed:
                            session.spinner_tick = 0
                            session.spinner_step += 1
                        status_msg = self._spinner_messages[
                            session.spinner_step % len(self._spinner_messages)
                        ]
                        if not session.spinner_shown or spinner_text_changed:
                            spinner_msg = gr.ChatMessage(
                                role="assistant",
                                content=f"### {status_msg}",
                                metadata={"title": "_On it_"},
                            )
                            if session.spinner_shown:
                                history = history[:-1] + [spinner_msg]
                            else:
                                history = history + [spinner_msg]
                                session.spinner_shown = True
                            chatbot_changed = True

                logs_md = self._format_logs() if self.execution_logs else "*No execution logs yet.*"

                # Only update the chatbot when there are actual changes
                # to avoid unnecessary re-renders that fight user scrolling.
                if not chatbot_changed:
                    return gr.skip(), gr.skip(), logs_md, sess_id
                return history, gr.update(visible=session.processing), logs_md, sess_id

            # wire events
            upload_btn.upload(handle_upload, [upload_btn, session_state], [uploaded_file_state, upload_indicator, session_state])

            # Note: gr.Request is automatically injected by Gradio, no need to pass it in inputs
            send_inputs = [msg_input, uploaded_file_state, chatbot, session_state]
            send_outputs = [chatbot, msg_input, uploaded_file_state, upload_indicator, session_state]

            msg_input.submit(
                handle_send,
                send_inputs,
                send_outputs,
            )

            stop_btn.click(handle_stop, [session_state], [stop_btn])

            def handle_clear(sess_id):
                """Delete all files in the session working directory when chat is cleared."""
                if not sess_id or sess_id not in self._web_sessions:
                    return sess_id
                session = self._web_sessions[sess_id]
                if session.data_path and os.path.isdir(session.data_path):
                    shutil.rmtree(session.data_path, ignore_errors=True)
                    os.makedirs(session.data_path, exist_ok=True)
                # Clear the session JSONL file
                if session.session_path and os.path.exists(session.session_path):
                    with open(session.session_path, "w", encoding="utf-8") as f:
                        f.write("")
                return sess_id

            chatbot.clear(handle_clear, [session_state], [session_state])

            # poll for assistant responses every 0.5s
            timer = gr.Timer(value=0.5)
            timer.tick(poll_response, [chatbot, session_state], [chatbot, stop_btn, logs_display, session_state])

            # Initialize session and restore chat history on page load
            def init_session(request: gr.Request):
                """Restore an existing session (via cookie) or create a new one."""
                cookie_sess_id = None
                if request:
                    cookie_sess_id = request.cookies.get("onit_session")
                sess_id, session = self._get_or_create_session(cookie_sess_id)
                history = self._load_chat_from_session(
                    session_path=session.session_path,
                    data_path=session.data_path,
                    session_id=sess_id,
                )
                return history, sess_id

            if self.auth_enabled:
                def check_auth_and_restore(request: gr.Request):
                    """Check auth via cookie and restore chat history."""
                    cookie_sess_id = None
                    if request:
                        cookie_sess_id = request.cookies.get("onit_session")
                    sess_id, session = self._get_or_create_session(cookie_sess_id)
                    history = self._load_chat_from_session(
                        session_path=session.session_path,
                        data_path=session.data_path,
                        session_id=sess_id,
                    )
                    not_auth = (gr.update(visible=True), gr.update(visible=False), history, "", sess_id)
                    if not request:
                        return not_auth

                    auth_cookie = request.cookies.get("onit_auth")
                    if not auth_cookie or auth_cookie not in self._authenticated_cookies:
                        return not_auth

                    cookie_data = self._authenticated_cookies[auth_cookie]
                    if datetime.now() > cookie_data['expires']:
                        del self._authenticated_cookies[auth_cookie]
                        return not_auth

                    # For authenticated users, use a stable session keyed by email
                    email = cookie_data.get('email', '')
                    auth_sess_id = cookie_data.get('session_id')
                    if auth_sess_id and auth_sess_id in self._web_sessions:
                        sess_id = auth_sess_id
                        session = self._web_sessions[sess_id]
                    else:
                        # Store session_id in cookie data for future lookups
                        cookie_data['session_id'] = sess_id

                    history = self._load_chat_from_session(
                        session_path=session.session_path,
                        data_path=session.data_path,
                        session_id=sess_id,
                    )
                    email_safe = email.replace("@", "&#64;")
                    email_display = f'<span style="unicode-bidi: embed; direction: ltr; pointer-events: none;">{email_safe}</span> | <a href="/auth/logout" style="color: inherit; text-decoration: none;">Logout</a>'
                    return gr.update(visible=False), gr.update(visible=True), history, email_display, sess_id

                app.load(check_auth_and_restore, None, [login_view, chat_view, chatbot, user_info, session_state])
            else:
                app.load(init_session, None, [chatbot, session_state])

        self.app = app
        return app

    def _setup_oauth_routes(self, fastapi_app: FastAPI):
        """Add OAuth2 callback routes to the FastAPI instance."""
        if not self.auth_enabled:
            return

        print("🔧 Setting up OAuth2 routes...")

        # Middleware to check authentication
        @fastapi_app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            """Check authentication for protected routes."""
            # Allow auth routes and static files
            if request.url.path.startswith("/auth/") or \
               request.url.path.startswith("/assets/") or \
               request.url.path.startswith("/file="):
                return await call_next(request)

            # Check if user is authenticated
            auth_cookie = request.cookies.get("onit_auth")
            is_authenticated = False

            if auth_cookie and auth_cookie in self._authenticated_cookies:
                cookie_data = self._authenticated_cookies[auth_cookie]
                if datetime.now() <= cookie_data['expires']:
                    is_authenticated = True

            # Redirect unauthenticated users to login page
            if not is_authenticated and request.url.path == "/":
                # Instead of redirecting, we'll let Gradio render and show login UI
                pass

            response = await call_next(request)
            return response

        @fastapi_app.get("/auth/login")
        async def oauth_login(request: Request):
            """Initiate OAuth2 login flow."""
            print("🔐 /auth/login endpoint called")
            # Generate PKCE parameters
            state, code_verifier, code_challenge = self.oauth_flow_manager.create_flow()

            # Build redirect URI
            redirect_uri = f"http://{request.url.hostname}:{self.server_port}/auth/callback"

            # Build Google OAuth URL
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

        @fastapi_app.get("/auth/callback")
        async def oauth_callback(request: Request, response: Response):
            """Handle OAuth2 callback."""
            code = request.query_params.get('code')
            state = request.query_params.get('state')
            error = request.query_params.get('error')

            if error:
                return HTMLResponse(
                    f"""
                    <html>
                    <head><title>Authentication Error</title></head>
                    <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: #d32f2f;">❌ Authentication Failed</h1>
                    <p>Error: {error}</p>
                    <a href="/" style="color: #4285f4;">Return to login</a>
                    </body>
                    </html>
                    """,
                    status_code=400
                )

            if not code or not state:
                return HTMLResponse(
                    """
                    <html>
                    <head><title>Authentication Error</title></head>
                    <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: #d32f2f;">❌ Invalid Request</h1>
                    <p>Missing authorization code or state parameter.</p>
                    <a href="/" style="color: #4285f4;">Return to login</a>
                    </body>
                    </html>
                    """,
                    status_code=400
                )

            # Verify state and get code_verifier
            code_verifier = self.oauth_flow_manager.verify_and_get_verifier(state)
            if not code_verifier:
                return HTMLResponse(
                    """
                    <html>
                    <head><title>Authentication Error</title></head>
                    <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: #d32f2f;">❌ Invalid or Expired Session</h1>
                    <p>The authentication session has expired or is invalid.</p>
                    <a href="/" style="color: #4285f4;">Try again</a>
                    </body>
                    </html>
                    """,
                    status_code=400
                )

            # Exchange code for token
            redirect_uri = f"http://{request.url.hostname}:{self.server_port}/auth/callback"
            email = self.authenticator.exchange_code_for_token(code, code_verifier, redirect_uri)

            if not email:
                return HTMLResponse(
                    """
                    <html>
                    <head><title>Authentication Error</title></head>
                    <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: #d32f2f;">❌ Authentication Failed</h1>
                    <p>Could not verify your Google account or you are not authorized.</p>
                    <a href="/" style="color: #4285f4;">Try again</a>
                    </body>
                    </html>
                    """,
                    status_code=403
                )

            # Create session
            session_id = self.session_manager.create_session(email)

            # Create authentication cookie
            auth_cookie = secrets.token_urlsafe(32)
            self._authenticated_cookies[auth_cookie] = {
                'email': email,
                'session_id': session_id,
                'expires': datetime.now() + timedelta(hours=24)
            }

            # Set cookie and redirect to app
            response = RedirectResponse("/", status_code=302)
            response.set_cookie(
                key="onit_auth",
                value=auth_cookie,
                max_age=86400,  # 24 hours
                httponly=True,
                samesite="lax"
            )

            return response

        @fastapi_app.get("/auth/logout")
        async def oauth_logout(request: Request):
            """Logout and clear session."""
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

        @fastapi_app.get("/auth/check")
        async def check_auth(request: Request):
            """Check if user is authenticated."""
            auth_cookie = request.cookies.get("onit_auth")
            if not auth_cookie or auth_cookie not in self._authenticated_cookies:
                return {"authenticated": False}

            cookie_data = self._authenticated_cookies[auth_cookie]
            if datetime.now() > cookie_data['expires']:
                del self._authenticated_cookies[auth_cookie]
                return {"authenticated": False}

            return {
                "authenticated": True,
                "email": cookie_data['email']
            }

        print("✓ OAuth2 routes registered: /auth/login, /auth/callback, /auth/logout, /auth/check")

    def _setup_file_routes(self, fastapi_app: FastAPI):
        """Add routes to serve and receive files scoped per session."""
        @fastapi_app.get("/uploads/{session_id}/{filename}")
        async def serve_upload(session_id: str, filename: str):
            # Use basename to prevent path traversal
            safe_name = os.path.basename(filename)
            # Look up session's data_path; reject unknown sessions
            if session_id not in self._web_sessions:
                return Response(content="Session not found", status_code=404)
            data_path = self._web_sessions[session_id].data_path
            filepath = os.path.join(data_path, safe_name)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, "rb") as f:
                        content = f.read()
                    import mimetypes
                    media_type = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
                    return Response(content=content, media_type=media_type)
                except OSError:
                    return Response(content="File read error", status_code=500)
            return Response(content="File not found", status_code=404)

        @fastapi_app.post("/uploads/{session_id}/")
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

    def launch(self, loop: asyncio.AbstractEventLoop) -> None:
        """Launch the Gradio server on a background daemon thread."""
        self._loop = loop
        if self.app is None:
            self.build_app()

        # Create our own FastAPI app
        fastapi_app = FastAPI()

        # Middleware: set a session cookie so the browser remembers its
        # session across page refreshes / reconnects.
        @fastapi_app.middleware("http")
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

        # Setup routes
        self._setup_oauth_routes(fastapi_app)
        self._setup_file_routes(fastapi_app)

        # Mount Gradio app onto our FastAPI app
        # Allow Gradio to serve files from session data directories so
        # gr.FileData download links render correctly in the chatbot.
        data_root = str(Path(tempfile.gettempdir()) / "onit" / "data")
        allowed = [data_root, self.data_path]
        fastapi_app = gr.mount_gradio_app(
            fastapi_app, self.app, path="/", allowed_paths=allowed,
            css=self._custom_css, footer_links=[],
        )

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
                fastapi_app,
                host="0.0.0.0",
                port=self.server_port,
                log_level="info" if self.verbose else "warning",
                access_log=self.verbose,
            )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
