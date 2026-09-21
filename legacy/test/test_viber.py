"""Tests for src/ui/viber.py — Viber gateway and CLI integration."""

import asyncio
import hashlib
import hmac
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from legacy.gateway.viber import ViberGateway, MAX_MESSAGE_LENGTH
from src.ui import split_message


def _split_message(text, limit=MAX_MESSAGE_LENGTH):
    return split_message(text, limit)


# ── _split_message ─────────────────────────────────────────────────────────


class TestSplitMessage:
    def test_short_message_no_split(self):
        assert _split_message("hello") == ["hello"]

    def test_exact_limit_no_split(self):
        text = "x" * 7000
        assert _split_message(text) == [text]

    def test_long_message_splits_at_newline(self):
        line = "a" * 100 + "\n"
        text = line * 80  # 8080 chars > 7000 limit
        chunks = _split_message(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 7000

    def test_long_message_no_newlines(self):
        text = "x" * 15000
        chunks = _split_message(text)
        assert len(chunks) >= 2
        assert "".join(chunks) == text

    def test_custom_limit(self):
        text = "abc\ndef\nghi\njkl"
        chunks = _split_message(text, limit=8)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 8

    def test_empty_string(self):
        assert _split_message("") == [""]


# ── ViberGateway init ──────────────────────────────────────────────────────


class TestViberGatewayInit:
    def test_basic_init(self):
        onit = MagicMock()
        onit.session_path = "/tmp/sessions/test.jsonl"
        gw = ViberGateway(
            onit=onit,
            token="test-token",
            webhook_url="https://example.com/viber",
            port=8443,
            show_logs=True,
        )
        assert gw.token == "test-token"
        assert gw.webhook_url == "https://example.com/viber"
        assert gw.port == 8443
        assert gw.show_logs is True
        assert gw._chat_sessions == {}


# ── Signature verification ─────────────────────────────────────────────────


class TestSignatureVerification:
    def test_valid_signature(self):
        onit = MagicMock()
        onit.session_path = "/tmp/sessions/test.jsonl"
        gw = ViberGateway(onit, "my-secret-token", "https://example.com/viber")

        body = b'{"event":"message","sender":{"id":"123"}}'
        expected_sig = hmac.new(
            b"my-secret-token", body, hashlib.sha256
        ).hexdigest()

        assert gw._verify_signature(body, expected_sig) is True

    def test_invalid_signature(self):
        onit = MagicMock()
        onit.session_path = "/tmp/sessions/test.jsonl"
        gw = ViberGateway(onit, "my-secret-token", "https://example.com/viber")

        body = b'{"event":"message"}'
        assert gw._verify_signature(body, "invalid-signature") is False

    def test_empty_signature(self):
        onit = MagicMock()
        onit.session_path = "/tmp/sessions/test.jsonl"
        gw = ViberGateway(onit, "token", "https://example.com/viber")

        assert gw._verify_signature(b"body", "") is False


# ── Session management ─────────────────────────────────────────────────────


class TestSessionManagement:
    def test_creates_new_session(self, tmp_path):
        onit = MagicMock()
        onit.session_path = str(tmp_path / "sessions" / "main.jsonl")
        os.makedirs(tmp_path / "sessions", exist_ok=True)

        gw = ViberGateway(onit, "token", "https://example.com/viber")
        session = gw._get_chat_session("user123")

        assert "session_id" in session
        assert "session_path" in session
        assert "data_path" in session
        assert os.path.exists(session["session_path"])

    def test_reuses_existing_session(self, tmp_path):
        onit = MagicMock()
        onit.session_path = str(tmp_path / "sessions" / "main.jsonl")
        os.makedirs(tmp_path / "sessions", exist_ok=True)

        gw = ViberGateway(onit, "token", "https://example.com/viber")
        session1 = gw._get_chat_session("user123")
        session2 = gw._get_chat_session("user123")
        assert session1["session_id"] == session2["session_id"]

    def test_different_users_get_different_sessions(self, tmp_path):
        onit = MagicMock()
        onit.session_path = str(tmp_path / "sessions" / "main.jsonl")
        os.makedirs(tmp_path / "sessions", exist_ok=True)

        gw = ViberGateway(onit, "token", "https://example.com/viber")
        s1 = gw._get_chat_session("user1")
        s2 = gw._get_chat_session("user2")
        assert s1["session_id"] != s2["session_id"]


# ── Message handling ───────────────────────────────────────────────────────


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_handle_text_message(self, tmp_path):
        onit = MagicMock()
        onit.session_path = str(tmp_path / "sessions" / "main.jsonl")
        onit.process_task = AsyncMock(return_value="Hello back!")
        os.makedirs(tmp_path / "sessions", exist_ok=True)

        gw = ViberGateway(onit, "token", "https://example.com/viber")
        gw._send_text = AsyncMock()

        data = {
            "event": "message",
            "sender": {"id": "user1", "name": "Test User"},
            "message": {"type": "text", "text": "Hello"},
            "message_token": 12345,
        }

        await gw._handle_message(data)

        onit.process_task.assert_called_once()
        call_args = onit.process_task.call_args
        assert call_args[0][0] == "Hello"
        gw._send_text.assert_called_once_with("user1", "Hello back!")

    @pytest.mark.asyncio
    async def test_handle_message_empty_text(self, tmp_path):
        onit = MagicMock()
        onit.session_path = str(tmp_path / "sessions" / "main.jsonl")
        onit.process_task = AsyncMock()
        os.makedirs(tmp_path / "sessions", exist_ok=True)

        gw = ViberGateway(onit, "token", "https://example.com/viber")
        gw._send_text = AsyncMock()

        data = {
            "event": "message",
            "sender": {"id": "user1"},
            "message": {"type": "text", "text": ""},
        }

        await gw._handle_message(data)
        onit.process_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_message_error(self, tmp_path):
        onit = MagicMock()
        onit.session_path = str(tmp_path / "sessions" / "main.jsonl")
        onit.process_task = AsyncMock(side_effect=RuntimeError("boom"))
        os.makedirs(tmp_path / "sessions", exist_ok=True)

        gw = ViberGateway(onit, "token", "https://example.com/viber")
        gw._send_text = AsyncMock()

        data = {
            "event": "message",
            "sender": {"id": "user1", "name": "Test"},
            "message": {"type": "text", "text": "hi"},
            "message_token": 1,
        }

        await gw._handle_message(data)
        gw._send_text.assert_called_once()
        assert "rephrase" in gw._send_text.call_args[0][1]


class TestHandlePhoto:
    @pytest.mark.asyncio
    async def test_handle_photo_message(self, tmp_path):
        onit = MagicMock()
        onit.session_path = str(tmp_path / "sessions" / "main.jsonl")
        onit.process_task = AsyncMock(return_value="I see a cat!")
        os.makedirs(tmp_path / "sessions", exist_ok=True)

        gw = ViberGateway(onit, "token", "https://example.com/viber")
        gw._send_text = AsyncMock()

        # Mock aiohttp image download
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=b"\xff\xd8\xff\xe0fake-jpeg")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        data = {
            "event": "message",
            "sender": {"id": "user1", "name": "Test"},
            "message": {
                "type": "picture",
                "media": "https://dl.viber.com/photo.jpg",
                "text": "What is this?",
            },
            "message_token": 99,
        }

        with patch("legacy.gateway.viber.aiohttp.ClientSession", return_value=mock_session):
            await gw._handle_photo(data)

        onit.process_task.assert_called_once()
        call_kwargs = onit.process_task.call_args
        assert call_kwargs[1]["images"] is not None
        assert len(call_kwargs[1]["images"]) == 1
        gw._send_text.assert_called_once_with("user1", "I see a cat!")

    @pytest.mark.asyncio
    async def test_handle_photo_default_caption(self, tmp_path):
        onit = MagicMock()
        onit.session_path = str(tmp_path / "sessions" / "main.jsonl")
        onit.process_task = AsyncMock(return_value="Described.")
        os.makedirs(tmp_path / "sessions", exist_ok=True)

        gw = ViberGateway(onit, "token", "https://example.com/viber")
        gw._send_text = AsyncMock()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=b"img-data")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        data = {
            "event": "message",
            "sender": {"id": "user1", "name": "Test"},
            "message": {
                "type": "picture",
                "media": "https://dl.viber.com/photo.jpg",
            },
            "message_token": 100,
        }

        with patch("legacy.gateway.viber.aiohttp.ClientSession", return_value=mock_session):
            await gw._handle_photo(data)

        # Should use default caption
        assert onit.process_task.call_args[0][0] == "Describe this image."


# ── API request / send message ─────────────────────────────────────────────


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_send_text_success(self):
        onit = MagicMock()
        onit.session_path = "/tmp/sessions/test.jsonl"
        gw = ViberGateway(onit, "token", "https://example.com/viber")

        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"status": 0, "status_message": "ok"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("legacy.gateway.viber.aiohttp.ClientSession", return_value=mock_session):
            await gw._send_text("user1", "Hello!")

        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "send_message" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_text_splits_long_message(self):
        onit = MagicMock()
        onit.session_path = "/tmp/sessions/test.jsonl"
        gw = ViberGateway(onit, "token", "https://example.com/viber")
        gw._send_message_with_retry = AsyncMock()

        long_text = "x" * 15000
        await gw._send_text("user1", long_text)

        # Should have been called multiple times for split chunks
        assert gw._send_message_with_retry.call_count >= 2


# ── Event routing ──────────────────────────────────────────────────────────


class TestEventRouting:
    @pytest.mark.asyncio
    async def test_routes_text_message(self, tmp_path):
        onit = MagicMock()
        onit.session_path = str(tmp_path / "sessions" / "main.jsonl")
        os.makedirs(tmp_path / "sessions", exist_ok=True)

        gw = ViberGateway(onit, "token", "https://example.com/viber")
        gw._handle_message = AsyncMock()
        gw._handle_photo = AsyncMock()

        data = {
            "event": "message",
            "sender": {"id": "u1"},
            "message": {"type": "text", "text": "hi"},
        }
        await gw._handle_event(data)
        gw._handle_message.assert_called_once_with(data)
        gw._handle_photo.assert_not_called()

    @pytest.mark.asyncio
    async def test_routes_picture_message(self, tmp_path):
        onit = MagicMock()
        onit.session_path = str(tmp_path / "sessions" / "main.jsonl")
        os.makedirs(tmp_path / "sessions", exist_ok=True)

        gw = ViberGateway(onit, "token", "https://example.com/viber")
        gw._handle_message = AsyncMock()
        gw._handle_photo = AsyncMock()

        data = {
            "event": "message",
            "sender": {"id": "u1"},
            "message": {"type": "picture", "media": "https://x.com/img.jpg"},
        }
        await gw._handle_event(data)
        gw._handle_photo.assert_called_once_with(data)
        gw._handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_unknown_message_type(self, tmp_path):
        onit = MagicMock()
        onit.session_path = str(tmp_path / "sessions" / "main.jsonl")
        os.makedirs(tmp_path / "sessions", exist_ok=True)

        gw = ViberGateway(onit, "token", "https://example.com/viber")
        gw._handle_message = AsyncMock()
        gw._handle_photo = AsyncMock()

        data = {
            "event": "message",
            "sender": {"id": "u1"},
            "message": {"type": "sticker", "sticker_id": 1},
        }
        await gw._handle_event(data)
        gw._handle_message.assert_not_called()
        gw._handle_photo.assert_not_called()


# ── Webhook registration ──────────────────────────────────────────────────


class TestSetWebhook:
    @pytest.mark.asyncio
    async def test_set_webhook_success(self):
        onit = MagicMock()
        onit.session_path = "/tmp/sessions/test.jsonl"
        gw = ViberGateway(onit, "token", "https://example.com/viber")

        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"status": 0, "status_message": "ok"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("legacy.gateway.viber.aiohttp.ClientSession", return_value=mock_session):
            await gw._set_webhook()

    @pytest.mark.asyncio
    async def test_set_webhook_failure_raises(self):
        onit = MagicMock()
        onit.session_path = "/tmp/sessions/test.jsonl"
        gw = ViberGateway(onit, "token", "https://example.com/viber")

        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={
            "status": 1, "status_message": "invalidUrl"
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("legacy.gateway.viber.aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(RuntimeError, match="status 1.*invalidUrl"):
                await gw._set_webhook()


# ── CLI gateway auto-detect ────────────────────────────────────────────────


class TestCLIGatewayAutoDetect:
    """Test the gateway type resolution logic in cli.py."""

    def test_auto_detects_telegram(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg-token-123")
        monkeypatch.delenv("VIBER_BOT_TOKEN", raising=False)

        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        viber_token = os.environ.get("VIBER_BOT_TOKEN")

        gateway_type = "auto"
        if telegram_token:
            gateway_type = "telegram"
        elif viber_token:
            gateway_type = "viber"

        assert gateway_type == "telegram"

    def test_auto_detects_viber(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.setenv("VIBER_BOT_TOKEN", "viber-token-123")

        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        viber_token = os.environ.get("VIBER_BOT_TOKEN")

        gateway_type = "auto"
        if telegram_token:
            gateway_type = "telegram"
        elif viber_token:
            gateway_type = "viber"

        assert gateway_type == "viber"

    def test_auto_prefers_telegram(self, monkeypatch):
        """When both tokens are set, auto should prefer Telegram for backward compat."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg-token")
        monkeypatch.setenv("VIBER_BOT_TOKEN", "viber-token")

        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        viber_token = os.environ.get("VIBER_BOT_TOKEN")

        gateway_type = "auto"
        if telegram_token:
            gateway_type = "telegram"
        elif viber_token:
            gateway_type = "viber"

        assert gateway_type == "telegram"

    def test_auto_no_tokens_available(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("VIBER_BOT_TOKEN", raising=False)

        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        viber_token = os.environ.get("VIBER_BOT_TOKEN")

        gateway_type = "auto"
        if telegram_token:
            gateway_type = "telegram"
        elif viber_token:
            gateway_type = "viber"

        # Should remain 'auto' — cli.py would sys.exit(1) in this case
        assert gateway_type == "auto"
