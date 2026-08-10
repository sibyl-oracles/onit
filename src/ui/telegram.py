"""Telegram bot gateway for OnIt.

Allows remote interaction with an OnIt agent via a Telegram bot.
Usage: onit --gateway  (requires TELEGRAM_BOT_TOKEN env var)
"""

import asyncio
import logging

from . import split_message
import os
import uuid
from pathlib import Path
import tempfile

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import TimedOut, NetworkError
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)

# Telegram message length limit
MAX_MESSAGE_LENGTH = 4096


class TelegramGateway:
    """Telegram bot that routes messages through an OnIt agent instance."""

    MAX_RETRIES = 3

    def __init__(self, onit, token: str, show_logs: bool = False):
        self.onit = onit
        self.token = token
        self.show_logs = show_logs
        # Per-chat session state: chat_id -> {session_id, session_path, data_path}
        self._chat_sessions: dict[int, dict] = {}

    def _get_chat_session(self, chat_id: int) -> dict:
        """Get or create session state for a Telegram chat."""
        if chat_id not in self._chat_sessions:
            session_id = str(uuid.uuid4())
            sessions_dir = os.path.dirname(self.onit.session_path)
            session_path = os.path.join(sessions_dir, f"{session_id}.jsonl")
            if not os.path.exists(session_path):
                with open(session_path, "w", encoding="utf-8") as f:
                    f.write("")
            data_path = str(Path(tempfile.gettempdir()) / "onit" / "data" / session_id)
            os.makedirs(data_path, exist_ok=True)
            self._chat_sessions[chat_id] = {
                "session_id": session_id,
                "session_path": session_path,
                "data_path": data_path,
            }
            logger.info("Created new session %s for Telegram chat %s", session_id, chat_id)
        return self._chat_sessions[chat_id]

    def _correction_sender(self, message):
        """Deliver a fact-check that finished after the answer was sent.

        As a follow-up carrying the corrected answer, not a note on its own:
        the wrong figure is still on their screen above it, and a bare "3.1M,
        not 4.2M" leaves them holding two numbers with no way to tell which
        one the answer now says.
        """
        def _on_correction(answer: str, note: str) -> None:
            async def _send() -> None:
                try:
                    text = f"\u21bb Correction to my last answer ({note}):\n\n{answer}"
                    for chunk in split_message(text, MAX_MESSAGE_LENGTH):
                        await self._reply_with_retry(message, chunk)
                except Exception as e:
                    logger.warning("Could not send the correction: %s", e)
            asyncio.ensure_future(_send())
        return _on_correction

    async def _reply_with_retry(self, message, text):
        """Send a reply, retrying on Telegram network/timeout errors."""
        for attempt in range(self.MAX_RETRIES):
            try:
                await message.reply_text(text)
                return
            except (TimedOut, NetworkError) as e:
                logger.warning("Telegram send failed (attempt %d/%d): %s",
                               attempt + 1, self.MAX_RETRIES, e)
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("Gave up sending message after %d attempts", self.MAX_RETRIES)

    async def _start_command(self, update: Update, context) -> None:
        """Handle /start command."""
        name = "Assistant"
        await self._reply_with_retry(
            update.message,
            f"Hey there! I'm {name}, your friendly AI assistant. "
            f"Feel free to send me a message or a photo and I'll be happy to help!\n\n"
            f"⚠️ May produce inaccurate information. Verify important details independently."
        )

    async def _handle_message(self, update: Update, context) -> None:
        """Handle incoming text messages."""
        text = update.message.text
        if not text:
            return

        chat_id = update.message.chat.id
        session = self._get_chat_session(chat_id)

        if self.show_logs:
            user = update.message.from_user
            name = user.username or user.first_name or str(user.id)
            print(f"[MSG] {name}: {text}")

        # Show typing indicator
        try:
            await update.message.chat.send_action(ChatAction.TYPING)
        except (TimedOut, NetworkError):
            pass

        # Keep sending typing action while processing
        stop_typing = asyncio.Event()

        async def typing_loop():
            while not stop_typing.is_set():
                try:
                    await update.message.chat.send_action(ChatAction.TYPING)
                except Exception:
                    pass
                await asyncio.sleep(5)

        typing_task = asyncio.create_task(typing_loop())

        try:
            response = await self.onit.process_task(
                text,
                session_path=session["session_path"],
                data_path=session["data_path"],
                session_id=session["session_id"],
                correction_callback=self._correction_sender(update.message),
            )
        except Exception as e:
            logger.error("Error processing task: %s", e)
            response = "I am sorry \U0001f614. Could you please rephrase your question?"
        finally:
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        if self.show_logs:
            print(f"[BOT] {response}")

        # Send response, splitting if needed
        for chunk in split_message(response, MAX_MESSAGE_LENGTH):
            await self._reply_with_retry(update.message, chunk)

    async def _handle_photo(self, update: Update, context) -> None:
        """Handle incoming photo messages (for vision models)."""
        caption = update.message.caption or "Describe this image."

        chat_id = update.message.chat.id
        session = self._get_chat_session(chat_id)

        if self.show_logs:
            user = update.message.from_user
            name = user.username or user.first_name or str(user.id)
            print(f"[IMG] {name}: {caption}")

        # Show typing indicator
        try:
            await update.message.chat.send_action(ChatAction.TYPING)
        except (TimedOut, NetworkError):
            pass

        # Download the highest resolution photo
        photo = update.message.photo[-1]
        image_path = os.path.join(session["data_path"], f"telegram_{photo.file_unique_id}.jpg")
        for attempt in range(self.MAX_RETRIES):
            try:
                file = await context.bot.get_file(photo.file_id)
                await file.download_to_drive(image_path)
                break
            except (TimedOut, NetworkError) as e:
                logger.warning("Photo download failed (attempt %d/%d): %s",
                               attempt + 1, self.MAX_RETRIES, e)
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    await self._reply_with_retry(update.message, "Sorry, failed to download the photo. Please try again.")
                    return

        # Keep sending typing action while processing
        stop_typing = asyncio.Event()

        async def typing_loop():
            while not stop_typing.is_set():
                try:
                    await update.message.chat.send_action(ChatAction.TYPING)
                except Exception:
                    pass
                await asyncio.sleep(5)

        typing_task = asyncio.create_task(typing_loop())

        try:
            response = await self.onit.process_task(
                caption,
                images=[image_path],
                session_path=session["session_path"],
                data_path=session["data_path"],
                session_id=session["session_id"],
                correction_callback=self._correction_sender(update.message),
            )
        except Exception as e:
            logger.error("Error processing image task: %s", e)
            response = "I am sorry \U0001f614. Could you please rephrase your question?"
        finally:
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        if self.show_logs:
            print(f"[BOT] {response}")

        for chunk in split_message(response, MAX_MESSAGE_LENGTH):
            await self._reply_with_retry(update.message, chunk)

    def run_sync(self) -> None:
        """Start the Telegram bot with its own event loop via run_polling().

        This must be called instead of `run()` because python-telegram-bot's
        run_polling() manages its own asyncio event loop.  OnIt should call
        this from a dedicated thread or before entering its own asyncio.run().
        """
        # Suppress noisy library logs; verbose mode uses print() for messages
        for name in ("telegram", "telegram.ext", "telegram.bot",
                      "httpx", "httpcore"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.WARNING)
            lg.propagate = False

        app = (
            Application.builder()
            .token(self.token)
            .concurrent_updates(256)
            .build()
        )

        app.add_handler(CommandHandler("start", self._start_command))
        app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        print("Telegram gateway running (Ctrl+C to stop)")
        app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
