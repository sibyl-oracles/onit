"""Viber bot gateway for OnIt.

Allows remote interaction with an OnIt agent via a Viber bot.
Usage: onit --gateway viber  (requires VIBER_BOT_TOKEN and VIBER_WEBHOOK_URL env vars)

Unlike Telegram (which uses long-polling), Viber requires a webhook (HTTPS)
endpoint. This gateway uses FastAPI + uvicorn to serve the webhook.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from fastapi import FastAPI, Request, Response
import uvicorn

logger = logging.getLogger(__name__)

VIBER_API_URL = "https://chatapi.viber.com/pa"
MAX_MESSAGE_LENGTH = 7000


def _split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a long message into chunks that fit within Viber's limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind('\n', 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip('\n')
    return chunks


class ViberGateway:
    """Viber bot that routes messages through an OnIt agent instance."""

    MAX_RETRIES = 3

    def __init__(self, onit, token: str, webhook_url: str,
                 port: int = 8443, show_logs: bool = False):
        self.onit = onit
        self.token = token
        self.webhook_url = webhook_url
        self.port = port
        self.show_logs = show_logs
        self._chat_sessions: dict[str, dict] = {}

        @asynccontextmanager
        async def lifespan(app):
            # Delay webhook registration so uvicorn is accepting connections
            # before Viber sends its verification event back to our endpoint.
            async def _register_webhook():
                await asyncio.sleep(1)
                await self._set_webhook()
            asyncio.create_task(_register_webhook())
            yield

        self._app = FastAPI(lifespan=lifespan)
        self._setup_routes()

    def _setup_routes(self):
        """Register FastAPI webhook route."""
        @self._app.post("/viber")
        async def viber_webhook(request: Request):
            body = await request.body()
            signature = request.headers.get("X-Viber-Content-Signature", "")
            if not self._verify_signature(body, signature):
                return Response(status_code=403)
            try:
                data = await request.json()
            except Exception:
                logger.warning("Failed to parse webhook JSON body")
                return Response(status_code=200)
            event = data.get("event")
            if event == "message":
                asyncio.create_task(self._handle_event(data))
            elif event == "webhook":
                logger.info("Webhook registered successfully")
            elif event == "conversation_started":
                # Viber expects the welcome message in the response body
                # (user is not yet subscribed, so send_message API won't work)
                welcome = {
                    "sender": {"name": "OnIt"},
                    "type": "text",
                    "text": (
                        "Hey there! I'm your friendly AI assistant. "
                        "Feel free to send me a message or a photo and "
                        "I'll be happy to help!\n\n"
                        "May produce inaccurate information. "
                        "Verify important details independently."
                    ),
                }
                return Response(
                    content=json.dumps(welcome),
                    media_type="application/json",
                    status_code=200,
                )
            return Response(status_code=200)

    def _verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify Viber webhook signature using HMAC-SHA256."""
        expected = hmac.new(
            self.token.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def _get_chat_session(self, user_id: str) -> dict:
        """Get or create session state for a Viber user."""
        if user_id not in self._chat_sessions:
            session_id = str(uuid.uuid4())
            sessions_dir = os.path.dirname(self.onit.session_path)
            session_path = os.path.join(sessions_dir, f"{session_id}.jsonl")
            if not os.path.exists(session_path):
                with open(session_path, "w", encoding="utf-8") as f:
                    f.write("")
            data_path = str(Path(tempfile.gettempdir()) / "onit" / "data" / session_id)
            os.makedirs(data_path, exist_ok=True)
            self._chat_sessions[user_id] = {
                "session_id": session_id,
                "session_path": session_path,
                "data_path": data_path,
            }
            logger.info("Created new session %s for Viber user %s", session_id, user_id)
        return self._chat_sessions[user_id]

    async def _api_request(self, endpoint: str, payload: dict) -> dict:
        """Make an authenticated request to the Viber API."""
        headers = {"X-Viber-Auth-Token": self.token, "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{VIBER_API_URL}/{endpoint}", json=payload, headers=headers
            ) as resp:
                return await resp.json()

    async def _send_text(self, to: str, text: str) -> None:
        """Send a text message to a Viber user with retry logic."""
        for chunk in _split_message(text):
            await self._send_message_with_retry(to, {
                "type": "text",
                "text": chunk,
            })

    async def _send_message_with_retry(self, to: str, message: dict) -> None:
        """Send a message with exponential backoff retry."""
        payload = {
            "receiver": to,
            "min_api_version": 1,
            "sender": {"name": "OnIt"},
            **message,
        }
        for attempt in range(self.MAX_RETRIES):
            try:
                result = await self._api_request("send_message", payload)
                status = result.get("status")
                if status == 0:
                    return
                logger.warning("Viber send failed (status %s, attempt %d/%d): %s",
                               status, attempt + 1, self.MAX_RETRIES,
                               result.get("status_message"))
            except Exception as e:
                logger.warning("Viber send failed (attempt %d/%d): %s",
                               attempt + 1, self.MAX_RETRIES, e)
            if attempt < self.MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error("Gave up sending message after %d attempts", self.MAX_RETRIES)

    async def _handle_event(self, data: dict) -> None:
        """Route incoming message events."""
        message = data.get("message", {})
        msg_type = message.get("type", "")
        if msg_type == "text":
            await self._handle_message(data)
        elif msg_type == "picture":
            await self._handle_photo(data)

    async def _handle_message(self, data: dict) -> None:
        """Handle incoming text messages."""
        sender = data.get("sender", {})
        user_id = sender.get("id")
        text = data.get("message", {}).get("text", "")
        if not user_id or not text:
            return

        session = self._get_chat_session(user_id)

        if self.show_logs:
            name = sender.get("name", user_id)
            print(f"[MSG] {name}: {text}")

        try:
            response = await self.onit.process_task(
                text,
                session_path=session["session_path"],
                data_path=session["data_path"],
            )
        except Exception as e:
            logger.error("Error processing task: %s", e)
            response = "I am sorry \U0001f614. Could you please rephrase your question?"

        if self.show_logs:
            print(f"[BOT] {response}")

        await self._send_text(user_id, response)

    async def _handle_photo(self, data: dict) -> None:
        """Handle incoming photo messages."""
        sender = data.get("sender", {})
        user_id = sender.get("id")
        message = data.get("message", {})
        media_url = message.get("media")
        caption = message.get("text") or "Describe this image."

        if not user_id or not media_url:
            return

        session = self._get_chat_session(user_id)

        if self.show_logs:
            name = sender.get("name", user_id)
            print(f"[IMG] {name}: {caption}")

        # Download the image
        image_path = os.path.join(
            session["data_path"],
            f"viber_{data.get('message_token', uuid.uuid4().hex)}.jpg"
        )
        for attempt in range(self.MAX_RETRIES):
            try:
                async with aiohttp.ClientSession() as http_session:
                    async with http_session.get(media_url) as resp:
                        if resp.status == 200:
                            with open(image_path, "wb") as f:
                                f.write(await resp.read())
                            break
                        logger.warning("Image download HTTP %s (attempt %d/%d)",
                                       resp.status, attempt + 1, self.MAX_RETRIES)
            except Exception as e:
                logger.warning("Image download failed (attempt %d/%d): %s",
                               attempt + 1, self.MAX_RETRIES, e)
            if attempt < self.MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                await self._send_text(user_id, "Sorry, failed to download the photo. Please try again.")
                return

        try:
            response = await self.onit.process_task(
                caption,
                images=[image_path],
                session_path=session["session_path"],
                data_path=session["data_path"],
            )
        except Exception as e:
            logger.error("Error processing image task: %s", e)
            response = "I am sorry \U0001f614. Could you please rephrase your question?"

        if self.show_logs:
            print(f"[BOT] {response}")

        await self._send_text(user_id, response)

    async def _set_webhook(self) -> None:
        """Register the webhook URL with Viber."""
        result = await self._api_request("set_webhook", {
            "url": self.webhook_url,
            "event_types": ["message", "conversation_started"],
        })
        status = result.get("status")
        status_msg = result.get("status_message", "unknown error")
        if status == 0:
            logger.info("Viber webhook registered: %s", self.webhook_url)
        else:
            raise RuntimeError(
                f"Failed to register Viber webhook (status {status}): {status_msg}\n"
                f"  Webhook URL: {self.webhook_url}\n"
                f"  Ensure the URL is publicly reachable over HTTPS and the "
                f"VIBER_BOT_TOKEN is valid."
            )

    def run_sync(self) -> None:
        """Start the Viber gateway with uvicorn (blocking).

        Registers the webhook on startup, then serves the FastAPI app.
        """
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.WARNING)
            lg.propagate = False

        print(f"Viber gateway running on port {self.port} (Ctrl+C to stop)")
        uvicorn.run(self._app, host="0.0.0.0", port=self.port, log_level="warning")
