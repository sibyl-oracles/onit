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
import os
import re
import secrets
import shutil
import threading
import urllib.parse
from collections import deque
from datetime import datetime, timedelta
from typing import Literal, Optional

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


class WebChatUI:
    def __init__(
        self,
        theme: str = "white",
        agent_cursor: str = "OnIt",
        data_path: str = "~/.onit/data",
        show_logs: bool = False,
        server_port: int = 9000,
        share: bool = False,
        google_client_id: Optional[str] = None,
        google_client_secret: Optional[str] = None,
        allowed_emails: Optional[list[str]] = None,
        session_path: Optional[str] = None,
        title: str = "OnIt Chat",
    ) -> None:
        self.title = title
        self.agent_cursor = agent_cursor
        self.data_path = os.path.expanduser(data_path)
        self.session_path = session_path
        self.show_logs = show_logs
        self.server_port = server_port
        self.share = share
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

        # internal state — chat history uses gr.ChatMessage objects
        self._chat_history: list[gr.ChatMessage] = []
        self.execution_logs: deque[dict] = deque(maxlen=100)
        self._pending_responses: list[gr.ChatMessage] = []
        self._user_input_queue: asyncio.Queue = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._processing: bool = False
        self._spinner_shown: bool = False
        self.safety_queue = None  # set by OnIt.run() when available

        # Store authenticated sessions with cookies
        # Format: {cookie_value: {email, session_id, expires}}
        self._authenticated_cookies: dict[str, dict] = {}

        # gradio app
        self.app: Optional[gr.Blocks] = None

    # ------------------------------------------------------------------
    # Interface methods (matching ChatUI contract)
    # ------------------------------------------------------------------

    def _extract_file_paths(self, text: str) -> tuple[str, list[str]]:
        """Extract data_path file paths from text; replace with download links."""
        pattern = re.escape(self.data_path) + r'/[\w\-\.]+(?:\.\w+)'
        found = re.findall(pattern, text)
        # First strip data_path file paths to just basenames
        cleaned = re.sub(pattern, lambda m: os.path.basename(m.group(0)), text)
        # Strip any other absolute file paths, keeping only the basename
        cleaned = re.sub(r'(?<![:\w])/(?:[\w\.\-]+/)+(?=[\w\.\-]+)', '', cleaned)
        # Now turn found basenames into download links
        for fpath in found:
            fname = os.path.basename(fpath)
            cleaned = cleaned.replace(fname, f'[{fname}](/uploads/{fname})', 1)
        return cleaned, found

    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        response: str,
        elapsed: str = "",
    ) -> None:
        """Called by chat.py and onit.py to record messages."""
        file_paths = []
        if role == "assistant":
            display, file_paths = self._extract_file_paths(response)
        else:
            display = response
        if elapsed:
            display = f"{display}\n\n_({elapsed})_"
        self._pending_responses.append(gr.ChatMessage(role=role, content=display))
        for fpath in file_paths:
            if os.path.isfile(fpath):
                self._pending_responses.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=gr.FileData(path=fpath, mime_type=None),
                    )
                )
        if role in ("assistant", "system"):
            self._processing = False

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

    def _load_chat_from_session(self) -> list:
        """Load chat history from the JSONL session file for display in the chatbot."""
        messages = []
        if not self.session_path or not os.path.exists(self.session_path):
            return messages
        try:
            with open(self.session_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if "task" in entry and "response" in entry:
                            messages.append(gr.ChatMessage(role="user", content=entry["task"]))
                            display, file_paths = self._extract_file_paths(entry["response"])
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

    async def get_user_input_async(self) -> str:
        """Await user input from Gradio's submit callback."""
        return await self._user_input_queue.get()

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
        footer { display: none !important; }
        .built-with { display: none !important; }
        a[href*="gradio.app"] { display: none !important; }
        button[title="Share"], .share-button, [class*="share"] { display: none !important; }
        button[aria-label="Share"] { display: none !important; }
        .message-buttons-bot button:not([title="Copy"]):not([aria-label="Copy"]) { display: none !important; }
        .bot .actions button:not([title="Copy"]):not([aria-label="Copy"]) { display: none !important; }
        """
        with gr.Blocks(title=self.title, css=custom_css) as app:
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
                    buttons=["copy"],
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

                gr.Markdown(
                    "<p style='text-align: center; color: #999; font-size: 0.8em; margin-top: 4px;'>"
                    "OnIt may produce inaccurate information. Verify important details independently.<br/>"
                    "<a href='https://github.com/sibyl-oracles/onit' target='_blank' style='color: #999;'>github.com/sibyl-oracles/onit</a></p>"
                )

            # state for uploaded file path
            uploaded_file_state = gr.State(value=None)

            # -- callbacks --
            # Note: Authentication is handled via OAuth redirect flow in FastAPI routes
            # No Gradio callbacks needed for login/logout
            def handle_upload(file):
                if file is None:
                    return None, gr.update(value="", visible=False)
                # Gradio 6+ returns a filepath string; older versions return a file object
                file_path = file if isinstance(file, (str, os.PathLike)) else file.name
                file_path = str(file_path)
                os.makedirs(self.data_path, exist_ok=True)
                fname = os.path.basename(file_path)
                dest = os.path.join(self.data_path, fname)
                shutil.copy2(file_path, dest)
                indicator = f"<p style='color: #4caf50; font-size: 0.8em; margin: 2px 0;'>📎 {fname}</p>"
                return dest, gr.update(value=indicator, visible=True)

            def handle_send(user_msg, uploaded_path, history, request: gr.Request = None):
                hide_indicator = gr.update(value="", visible=False)
                # Verify authentication if enabled
                if self.auth_enabled:
                    # Check authentication via cookie (not state, since OAuth sets cookies)
                    auth_cookie = request.cookies.get("onit_auth") if request else None
                    if not auth_cookie or auth_cookie not in self._authenticated_cookies:
                        error_msg = gr.ChatMessage(
                            role="assistant",
                            content="❌ Session expired or invalid. Please login again."
                        )
                        return history + [error_msg], user_msg, uploaded_path, hide_indicator

                    # Verify cookie hasn't expired
                    cookie_data = self._authenticated_cookies[auth_cookie]
                    if datetime.now() > cookie_data['expires']:
                        del self._authenticated_cookies[auth_cookie]
                        error_msg = gr.ChatMessage(
                            role="assistant",
                            content="❌ Session expired or invalid. Please login again."
                        )
                        return history + [error_msg], user_msg, uploaded_path, hide_indicator

                if not user_msg and not uploaded_path:
                    return history, "", None, hide_indicator

                display_msg = user_msg or ""
                queue_msg = user_msg or ""
                if uploaded_path:
                    fname = os.path.basename(uploaded_path)
                    file_url = f"/uploads/{fname}"
                    display_msg += f"\n📎 [{fname}]({file_url})"
                    queue_msg += f"\nRelevant files: {uploaded_path}"

                history = history + [gr.ChatMessage(role="user", content=display_msg)]

                # enqueue for OnIt event loop
                self._processing = True
                if self._loop:
                    asyncio.run_coroutine_threadsafe(
                        self._user_input_queue.put(queue_msg),
                        self._loop,
                    )

                return history, "", None, hide_indicator

            def handle_stop():
                if self._loop and self.safety_queue:
                    self._loop.call_soon_threadsafe(
                        self.safety_queue.put_nowait, True
                    )
                return gr.update(visible=False)

            def poll_response(history):
                # Remove spinner before appending real responses or when done
                if self._spinner_shown and (self._pending_responses or not self._processing):
                    if history:
                        history = history[:-1]
                    self._spinner_shown = False

                while self._pending_responses:
                    resp = self._pending_responses.pop(0)
                    history = history + [resp]

                # Add spinner once if processing and not already shown
                if self._processing and not self._spinner_shown:
                    history = history + [
                        gr.ChatMessage(
                            role="assistant",
                            content="...",
                            metadata={"title": "_thinking_"},
                        )
                    ]
                    self._spinner_shown = True

                return history, gr.update(visible=self._processing)

            # wire events
            upload_btn.upload(handle_upload, [upload_btn], [uploaded_file_state, upload_indicator])

            # Note: gr.Request is automatically injected by Gradio, no need to pass it in inputs
            send_inputs = [msg_input, uploaded_file_state, chatbot]
            send_outputs = [chatbot, msg_input, uploaded_file_state, upload_indicator]

            msg_input.submit(
                handle_send,
                send_inputs,
                send_outputs,
            )

            stop_btn.click(handle_stop, None, [stop_btn])

            # poll for assistant responses every 0.5s
            timer = gr.Timer(value=0.5)
            timer.tick(poll_response, [chatbot], [chatbot, stop_btn])

            # Restore chat history and check auth on page load
            def restore_chat():
                return self._load_chat_from_session()

            if self.auth_enabled:
                def check_auth_and_restore(request: gr.Request):
                    """Check auth via cookie and restore chat history."""
                    history = self._load_chat_from_session()
                    not_auth = (gr.update(visible=True), gr.update(visible=False), history, "")
                    if not request:
                        return not_auth

                    auth_cookie = request.cookies.get("onit_auth")
                    if not auth_cookie or auth_cookie not in self._authenticated_cookies:
                        return not_auth

                    cookie_data = self._authenticated_cookies[auth_cookie]
                    if datetime.now() > cookie_data['expires']:
                        del self._authenticated_cookies[auth_cookie]
                        return not_auth

                    email = cookie_data.get('email', '')
                    email_safe = email.replace("@", "&#64;")
                    email_display = f'<span style="unicode-bidi: embed; direction: ltr; pointer-events: none;">{email_safe}</span> | <a href="/auth/logout" style="color: inherit; text-decoration: none;">Logout</a>'
                    return gr.update(visible=False), gr.update(visible=True), history, email_display

                app.load(check_auth_and_restore, None, [login_view, chat_view, chatbot, user_info])
            else:
                app.load(restore_chat, None, [chatbot])

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
        """Add routes to serve and receive files from data_path."""
        @fastapi_app.get("/uploads/{filename}")
        async def serve_upload(filename: str):
            # Use basename to prevent path traversal
            safe_name = os.path.basename(filename)
            filepath = os.path.join(self.data_path, safe_name)
            if os.path.isfile(filepath):
                return FileResponse(filepath)
            return Response(content="File not found", status_code=404)

        @fastapi_app.post("/uploads/")
        async def receive_upload(file: UploadFile = File(...)):
            """Accept file uploads from remote MCP tools."""
            os.makedirs(self.data_path, exist_ok=True)
            safe_name = os.path.basename(file.filename)
            filepath = os.path.join(self.data_path, safe_name)
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

        # Setup routes
        self._setup_oauth_routes(fastapi_app)
        self._setup_file_routes(fastapi_app)

        # Mount Gradio app onto our FastAPI app
        fastapi_app = gr.mount_gradio_app(fastapi_app, self.app, path="/")

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
                log_level="warning"  # Changed from "info" to reduce log verbosity
            )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
