"""Tests for src/ui/web.py — SessionManager, OAuthFlowManager, GoogleAuthenticator."""

import os
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

gradio = pytest.importorskip("gradio", reason="gradio not installed")

from ui.web import SessionManager, OAuthFlowManager, GoogleAuthenticator, NullConsole


# ── NullConsole ─────────────────────────────────────────────────────────────

class TestNullConsole:
    def test_print_noop(self):
        c = NullConsole()
        c.print("anything")  # no error

    def test_clear_noop(self):
        c = NullConsole()
        c.clear()  # no error


# ── SessionManager ──────────────────────────────────────────────────────────

class TestSessionManager:
    def test_create_session(self):
        sm = SessionManager()
        sid = sm.create_session("user@test.com")
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_verify_valid_session(self):
        sm = SessionManager()
        sid = sm.create_session("user@test.com")
        email = sm.verify_session(sid)
        assert email == "user@test.com"

    def test_verify_invalid_session(self):
        sm = SessionManager()
        assert sm.verify_session("nonexistent") is None

    def test_verify_empty_session_id(self):
        sm = SessionManager()
        assert sm.verify_session("") is None
        assert sm.verify_session(None) is None

    def test_verify_expired_session(self):
        sm = SessionManager(session_duration_hours=0)
        sid = sm.create_session("user@test.com")
        # Manually expire the session
        sm.sessions[sid]["expires"] = datetime.now() - timedelta(seconds=1)
        assert sm.verify_session(sid) is None
        # Session should be cleaned up
        assert sid not in sm.sessions

    def test_revoke_session(self):
        sm = SessionManager()
        sid = sm.create_session("user@test.com")
        sm.revoke_session(sid)
        assert sm.verify_session(sid) is None

    def test_revoke_nonexistent_session(self):
        sm = SessionManager()
        sm.revoke_session("nonexistent")  # no error


# ── OAuthFlowManager ───────────────────────────────────────────────────────

class TestOAuthFlowManager:
    def test_create_flow(self):
        fm = OAuthFlowManager()
        state, verifier, challenge = fm.create_flow()
        assert isinstance(state, str)
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)
        assert len(state) > 0
        assert len(verifier) > 0

    def test_verify_valid_state(self):
        fm = OAuthFlowManager()
        state, verifier, _ = fm.create_flow()
        result = fm.verify_and_get_verifier(state)
        assert result == verifier

    def test_verify_consumes_state(self):
        """State is one-time use."""
        fm = OAuthFlowManager()
        state, _, _ = fm.create_flow()
        fm.verify_and_get_verifier(state)
        assert fm.verify_and_get_verifier(state) is None

    def test_verify_invalid_state(self):
        fm = OAuthFlowManager()
        assert fm.verify_and_get_verifier("bogus") is None

    def test_verify_expired_state(self):
        fm = OAuthFlowManager()
        state, _, _ = fm.create_flow()
        # Manually expire the flow
        fm.active_flows[state]["created_at"] = datetime.now() - timedelta(minutes=11)
        assert fm.verify_and_get_verifier(state) is None

    def test_cleanup_old_flows(self):
        fm = OAuthFlowManager()
        state1, _, _ = fm.create_flow()
        fm.active_flows[state1]["created_at"] = datetime.now() - timedelta(minutes=15)
        # Creating a new flow should clean up the old one
        fm.create_flow()
        assert state1 not in fm.active_flows


# ── GoogleAuthenticator ─────────────────────────────────────────────────────

class TestGoogleAuthenticator:
    @pytest.fixture
    def authenticator(self):
        return GoogleAuthenticator(
            client_id="test-client-id",
            client_secret="test-client-secret",
            allowed_emails=["user@example.com", "*@company.com"],
        )

    def test_is_email_allowed_exact_match(self, authenticator):
        assert authenticator._is_email_allowed("user@example.com") is True

    def test_is_email_allowed_domain_wildcard(self, authenticator):
        assert authenticator._is_email_allowed("anyone@company.com") is True

    def test_is_email_not_allowed(self, authenticator):
        assert authenticator._is_email_allowed("hacker@evil.com") is False

    def test_is_email_allowed_no_restrictions(self):
        auth = GoogleAuthenticator("id", "secret", allowed_emails=None)
        assert auth._is_email_allowed("anyone@anywhere.com") is True

    def test_is_email_allowed_empty_list(self):
        auth = GoogleAuthenticator("id", "secret", allowed_emails=[])
        # Empty set is falsy, so _is_email_allowed returns True
        assert auth._is_email_allowed("anyone@anywhere.com") is True

    def test_verify_token_valid(self, authenticator):
        mock_idinfo = {
            "email": "user@example.com",
            "email_verified": True,
        }
        with patch("ui.web.id_token.verify_oauth2_token", return_value=mock_idinfo):
            result = authenticator.verify_token("fake-token")
        assert result == "user@example.com"

    def test_verify_token_unverified_email(self, authenticator):
        mock_idinfo = {
            "email": "user@example.com",
            "email_verified": False,
        }
        with patch("ui.web.id_token.verify_oauth2_token", return_value=mock_idinfo):
            result = authenticator.verify_token("fake-token")
        assert result is None

    def test_verify_token_not_allowed_email(self, authenticator):
        mock_idinfo = {
            "email": "hacker@evil.com",
            "email_verified": True,
        }
        with patch("ui.web.id_token.verify_oauth2_token", return_value=mock_idinfo):
            result = authenticator.verify_token("fake-token")
        assert result is None

    def test_verify_token_invalid_raises(self, authenticator):
        with patch("ui.web.id_token.verify_oauth2_token", side_effect=ValueError("bad token")):
            result = authenticator.verify_token("bad-token")
        assert result is None

    def test_exchange_code_success(self, authenticator):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id_token": "valid-id-token"}
        mock_resp.raise_for_status = MagicMock()

        mock_idinfo = {"email": "user@example.com", "email_verified": True}

        with patch("ui.web.http_requests.post", return_value=mock_resp), \
             patch("ui.web.id_token.verify_oauth2_token", return_value=mock_idinfo):
            result = authenticator.exchange_code_for_token("code", "verifier", "http://localhost/callback")
        assert result == "user@example.com"

    def test_exchange_code_failure(self, authenticator):
        with patch("ui.web.http_requests.post", side_effect=Exception("network error")):
            result = authenticator.exchange_code_for_token("code", "verifier", "http://localhost/callback")
        assert result is None

    def test_exchange_code_no_id_token(self, authenticator):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"access_token": "xxx"}
        mock_resp.raise_for_status = MagicMock()

        with patch("ui.web.http_requests.post", return_value=mock_resp):
            result = authenticator.exchange_code_for_token("code", "verifier", "http://localhost/callback")
        assert result is None
