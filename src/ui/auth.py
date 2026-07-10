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

Shared authentication helpers for the OnIt web UIs.

Google OAuth2 (PKCE) session and flow management used by both the
FastAPI SSE web UI (api.py) and the legacy Gradio UI (web.py).
"""

import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

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
