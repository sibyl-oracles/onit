'''
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


Text-based UI components for the application.

If the `rich` library is available, it will be used to enhance the console output.
If not, it will fall back to standard print statements.

'''
import asyncio
import json
import os
import sys
import time
import threading

if sys.platform != "win32":
    import tty
    import termios
    import select as _select
else:
    import msvcrt
from collections import deque
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.theme import Theme
from rich.prompt import Prompt
from rich.status import Status
from rich.box import Box
from rich.box import ROUNDED, HEAVY
from datetime import datetime
import re

from typing import Callable, Optional, Any, Literal, Union


HORIZONTAL_THICK = Box(
    "━━━━\n"
    "    \n"
    "━━━━\n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "━━━━\n"
)

@dataclass
class Message:
    """Strongly typed message structure"""
    role: Literal["user", "assistant", "system", "tool_call", "tool_result"]
    content: str
    timestamp: str
    elapsed: str = ""
    name: str = ""

class ChatUI:
    def __init__(
        self,
        theme: str = "white",
        agent_cursor: str = "OnIt",
        max_messages: int = 100,
        display_messages: int = 15,
        max_history: int = 100,
        max_logs: int = 100,
        display_logs: int = 10,
        show_logs: bool = False,
        banner_title: str = "OnIt Chat Interface"
    ) -> None:
        """
        Initialize ChatUI with bounded memory and configuration options.

        Args:
            theme: Color theme ("white" or "dark"/"black")
            agent_cursor: Name of the AI agent
            max_messages: Maximum number of messages to keep in memory (default: 100)
            display_messages: Number of recent messages to display (default: 15)
            max_history: Maximum number of input history items to keep (default: 100)
            max_logs: Maximum number of execution logs to keep in memory (default: 100)
            display_logs: Number of recent logs to display (default: 10)
            show_logs: Whether to show the execution logs panel (default: False)
            banner_title: Title text shown in the startup banner (default: "OnIt Chat Interface")
        """
        self.banner_title = banner_title
        # Use deque for bounded memory - automatically removes old messages
        self.messages = deque(maxlen=max_messages)
        self.display_messages = display_messages
        self.agent_cursor = agent_cursor
        # Input history for arrow key navigation (like bash)
        self.input_history: deque[str] = deque(maxlen=max_history)
        self.history_index = -1  # -1 means not browsing history
        # Execution logs for debugging
        self.execution_logs: deque[dict] = deque(maxlen=max_logs)
        self.display_logs = display_logs
        self.show_logs = show_logs
        self.set_theme(theme)
        self.console = Console(theme=self.theme)
        self.status = Status(
            f"[{self.theme.styles.get('prompt', 'bold dark_blue')}] 💡 {agent_cursor}...[/]",
            spinner="dots",
            console=self.console
        )
        self._spinner_messages = [
            "On it...",
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
        self._spinner_step = 0
        self._spinner_timer: Optional[threading.Timer] = None
        self._spinner_stop_event: Optional[threading.Event] = None
        self._thinking_stop_event: Optional[threading.Event] = None
        self._thinking_thread: Optional[threading.Thread] = None
        self.model_name = ""  # auto-detected model name, set by chat()
        self.data_path = ""  # set when code files are saved to a data directory
        self._context_pct: float = 0.0  # context window usage 0-100, updated after each LLM call
        self._context_max_tokens: int = 0  # max context window size in tokens
        self._stream_header_printed = False  # lazy: header deferred until first visible token
        self._stream_pending = ""  # buffer tokens until first non-whitespace
        self._stream_think_started = False  # True while a think block is open
        self._tag_buf = ""  # buffer for partial tag detection across tokens
        self._trail_buf = ""  # buffer whitespace-only tokens to suppress trailing blank lines
        self._stream_cursor_shown = False  # blinking block cursor during streaming
        self._link_buf = ""  # buffer for detecting markdown links during streaming
        self._link_state = 0  # 0=normal, 1=in label [..., 2=after ](, eating URL
        self._stream_token_count = 0  # token counter for tok/s calculation
        self._stream_start_time = 0.0  # monotonic time when streaming started
        self.initialize()
        
    def set_theme(self, theme: str) -> None:
        """Set theme for the agent UI"""
        
        if "black" in theme or "dark" in theme:
            self.theme = Theme({
                "info": "bold dark_cyan",
                "warning": "bold yellow",
                "error": "bold red",
                "prompt": "bold white",
                "timestamp": "white"
            })
        else:
            self.theme = Theme({
                "info": "bold dark_cyan",
                "warning": "bold dark_green",
                "error": "dark_orange3",
                "prompt": "bold dark_blue",
                "timestamp": "cyan"
            })
            
    def format_timestamp(self):
        return datetime.now().strftime("%I:%M %p %d %b")

    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        response: str,
        elapsed: str = ""
    ) -> None:
        """
        Add a message to the chat history.

        Args:
            role: The role of the message sender
            response: The message content (parameter name matches chat.py usage)
            elapsed: Optional elapsed time string
        """
        try:
            msg = Message(
                role=role,
                content=response,
                timestamp=self.format_timestamp(),
                elapsed=elapsed
            )
            self.messages.append(msg)
        except Exception as e:
            # Fallback: still add message even if there's an error
            # This ensures messages are never lost due to type issues
            self.messages.append({
                "role": role,
                "content": response,
                "time": self.format_timestamp(),
                "elapsed": elapsed
            })

    def update_last_message(self, additional_content: str) -> None:
        """
        Update the last message by appending content (useful for streaming responses).

        Args:
            additional_content: Content to append to the last message
        """
        if self.messages:
            last_msg = self.messages[-1]
            if isinstance(last_msg, Message):
                # Create new Message with updated content (dataclasses are immutable by default)
                self.messages[-1] = Message(
                    role=last_msg.role,
                    content=last_msg.content + additional_content,
                    timestamp=last_msg.timestamp,
                    elapsed=last_msg.elapsed
                )
            else:
                # Legacy dict support
                last_msg["content"] += additional_content

    def clear_messages(self, keep_last: int = 0) -> None:
        """
        Clear messages, optionally keeping the last N messages.

        Args:
            keep_last: Number of recent messages to keep (default: 0 clears all)
        """
        if keep_last > 0:
            messages_to_keep = list(self.messages)[-keep_last:]
            self.messages.clear()
            self.messages.extend(messages_to_keep)
        else:
            self.messages.clear()

    def add_tool_call(self, name: str, arguments: dict) -> None:
        """Add a tool invocation message inline in the chat history."""
        if not self.show_logs:
            return
        self.messages.append(Message(
            role="tool_call",
            content=json.dumps(arguments),
            timestamp=self.format_timestamp(),
            name=name,
        ))

    def add_tool_result(self, name: str, result: str, truncate: int = 300) -> None:
        """Add a tool result message inline in the chat history."""
        if not self.show_logs:
            return
        display = result if len(result) <= truncate else result[:truncate] + "…"
        self.messages.append(Message(
            role="tool_result",
            content=display,
            timestamp=self.format_timestamp(),
            name=name,
        ))

    def add_log(
        self,
        message: str,
        level: Literal["info", "warning", "error", "debug"] = "info"
    ) -> None:
        """
        Add a log entry to the execution logs.

        Args:
            message: The log message
            level: Log level - "info", "warning", "error", or "debug"
        """
        log_entry = {
            "message": message,
            "level": level,
            "timestamp": self.format_timestamp()
        }
        self.execution_logs.append(log_entry)

    def clear_logs(self, keep_last: int = 0) -> None:
        """
        Clear execution logs, optionally keeping the last N entries.

        Args:
            keep_last: Number of recent logs to keep (default: 0 clears all)
        """
        if keep_last > 0:
            logs_to_keep = list(self.execution_logs)[-keep_last:]
            self.execution_logs.clear()
            self.execution_logs.extend(logs_to_keep)
        else:
            self.execution_logs.clear()

    def set_show_logs(self, show: bool) -> None:
        """
        Enable or disable the execution logs panel.

        Args:
            show: True to show logs panel, False to hide
        """
        self.show_logs = show

    def render_logs_panel(self) -> Panel:
        """
        Render the execution logs panel.

        Returns:
            Panel: A Rich Panel containing the execution logs
        """
        if not self.execution_logs:
            content = Text("No execution logs yet.", style="dim")
        else:
            content = Text()
            display_logs = list(self.execution_logs)[-self.display_logs:]

            level_styles = {
                "info": "cyan",
                "warning": "yellow",
                "error": "red",
                "debug": "dim white"
            }
            level_icons = {
                "info": "ℹ️ ",
                "warning": "⚠️ ",
                "error": "❌",
                "debug": "🔍"
            }

            for log in display_logs:
                level = log.get("level", "info")
                style = level_styles.get(level, "white")
                icon = level_icons.get(level, "•")
                timestamp = log.get("timestamp", "")
                message = log.get("message", "")

                content.append(f"{icon} ", style=style)
                content.append(f"[{timestamp}] ", style="dim")
                content.append(f"{message}\n", style=style)

        return Panel(
            content,
            title="[bold white]🔧 Execution Logs[/]",
            title_align="left",
            border_style="yellow",
            box=ROUNDED,
            padding=(0, 1),
            subtitle=f"Showing last {min(len(self.execution_logs), self.display_logs)} of {len(self.execution_logs)} logs" if self.execution_logs else None,
            subtitle_align="right"
        )

    def render_messages(self) -> Panel:
        """
        Render all messages in the upper panel with error handling.

        Returns:
            Panel: A Rich Panel containing the rendered messages
        """
        try:
            if not self.messages:
                return self._render_welcome_panel()

            content = Text()
            # Show last N messages (configurable)
            display_msgs = list(self.messages)[-self.display_messages:]

            for msg in display_msgs:
                # Handle both Message dataclass and dict (for backward compatibility)
                role = msg.role if isinstance(msg, Message) else msg["role"]
                msg_content = msg.content if isinstance(msg, Message) else msg["content"]
                msg_time = msg.timestamp if isinstance(msg, Message) else msg["time"]
                msg_elapsed = msg.elapsed if isinstance(msg, Message) else msg.get("elapsed", "")

                msg_name = msg.name if isinstance(msg, Message) else msg.get("name", "")

                if role == "user":
                    self._render_user_message(content, msg_content, msg_time)
                elif role == "tool_call":
                    if self.show_logs:
                        self._render_tool_call_message(content, msg_name, msg_content, msg_time)
                elif role == "tool_result":
                    if self.show_logs:
                        self._render_tool_result_message(content, msg_name, msg_content, msg_time)
                else:
                    self._render_assistant_message(content, msg_content, msg_time, msg_elapsed)

            if self.data_path:
                subtitle = f"📁 {self.data_path}"
                subtitle_align = "left"
            else:
                subtitle = f"Showing last {len(display_msgs)} of {len(self.messages)} messages"
                subtitle_align = "right"
            panel = Panel(
                content,
                title="[bold white]💬 Chat History[/]",
                title_align="left",
                border_style="blue",
                box=HORIZONTAL_THICK,
                padding=(1, 0),
                subtitle=subtitle,
                subtitle_align=subtitle_align,
            )

            return panel

        except Exception as e:
            # Fallback rendering on error - ensures UI doesn't crash
            return Panel(
                Text(f"Error rendering messages: {e}", style="error"),
                title="[bold red]⚠️  Rendering Error[/]",
                border_style="red",
                box=ROUNDED
            )

    def _render_welcome_panel(self) -> Panel:
        """Render welcome message when no messages exist."""
        welcome = Text()
        welcome.append("🤖 ", style="bold")
        welcome.append("Welcome! ", style=self.theme.styles.get("assistant", "bold magenta"))
        welcome.append("Type a message below to start chatting.\n\n", style=self.theme.styles.get("assistant", "bold magenta"))
        welcome.append("Commands: ", style=self.theme.styles.get("user", "bold cyan"))
        welcome.append("'Ctl+c' to close", style=self.theme.styles.get("user", "bold cyan"))

        return Panel(
            Align.center(welcome, vertical="middle"),
            title="[bold white]💬 Chat History[/]",
            title_align="left",
            border_style="blue",
            box=HORIZONTAL_THICK,
            padding=(1, 0)
        )

    def _render_tool_call_message(self, content: Text, name: str, args_str: str, msg_time: str) -> None:
        content.append(f"┌─ ⚙️  {name} ", style="bold dark_orange3")
        content.append(f"[{msg_time}]", style=self.theme.styles.get("timestamp", "cyan"))
        if self._context_pct > 0:
            content.append(f"  {self._fmt_ctx_label()}", style="bright_cyan")
        content.append("\n", style="default")
        content.append(f"{args_str}\n", style="bright_white")
        content.append("└" + "─" * 40 + "\n", style="dark_orange3")

    def _render_tool_result_message(self, content: Text, name: str, result_str: str, msg_time: str) -> None:
        content.append(f"┌─ ↩  {name} ", style="bold green")
        content.append(f"[{msg_time}]\n", style=self.theme.styles.get("timestamp", "cyan"))
        content.append(f"{result_str}\n", style="bright_white")
        content.append("└" + "─" * 40 + "\n", style="green")

    def _render_user_message(self, content: Text, msg_content: str, msg_time: str) -> None:
        """
        Render a user message to the content Text object.

        Args:
            content: The Rich Text object to append to
            msg_content: The message content
            msg_time: The message timestamp
        """
        content.append(f"┌─ 👤 You ", style=self.theme.styles.get("user", "bold cyan"))
        content.append(f"[{msg_time}]\n", style=self.theme.styles.get("timestamp", "cyan"))
        content.append(f"{msg_content}\n", style="white")
        content.append("└" + "─" * 40 + "\n", style=self.theme.styles.get("user", "cyan"))

    @staticmethod
    def _strip_markdown_links(text: str) -> str:
        """Replace markdown links with just their display text.

        Handles regular URLs and data-URIs so the terminal doesn't show
        huge base64 blobs.  ``[label](url)`` → ``label``
        """
        return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    def _render_assistant_message(self, content: Text, msg_content: str, msg_time: str, msg_elapsed: str) -> None:
        """
        Render an assistant message to the content Text object.

        Args:
            content: The Rich Text object to append to
            msg_content: The message content
            msg_time: The message timestamp
            msg_elapsed: The elapsed time string (optional)
        """
        _ai_label = f"┌─ 🤖 AI ({self.model_name}) " if self.model_name else "┌─ 🤖 AI "
        content.append(_ai_label, style=self.theme.styles.get("assistant", "bold magenta"))
        content.append(f"[{msg_time}]", style=self.theme.styles.get("timestamp", "dim magenta"))
        if msg_elapsed:
            content.append(f" - {msg_elapsed}", style=self.theme.styles.get("warning", "dim magenta"))
        content.append("\n", style=self.theme.styles.get("warning", "dim magenta"))
        msg_content = self._strip_markdown_links(msg_content)
        content.append(f"{msg_content}\n", style="bright_white")
        content.append("└" + "─" * 40 + "\n", style=self.theme.styles.get("assistant", "magenta"))

    def render_thinking_panel(self) -> Panel:
        """Render thinking indicator in input panel"""
        thinking = Text()
        thinking.append("🤖 ", style="bold")
        thinking.append("Assistant is thinking", style="bright_white")
        thinking.append("...", style="bright_white blink")
        
        return Panel(
            thinking,
            title="[bold white]⌨️  Please Wait[/]",
            title_align="left",
            border_style="magenta",
            box=HEAVY,
            padding=(0, 2),
            height=3
        )
    
    def start_status(self) -> None:
        """Start the status spinner with rotating messages."""
        self._spinner_step = 0
        self._spinner_stop_event = threading.Event()
        self._update_spinner_text()
        self.status.start()
        self._schedule_spinner_rotation()

    def _update_spinner_text(self) -> None:
        """Update the spinner text to the current message."""
        msg = self._spinner_messages[self._spinner_step % len(self._spinner_messages)]
        style = self.theme.styles.get('prompt', 'bold dark_blue')
        self.status.update(f"[{style}] 💡 {msg}[/]")

    def _schedule_spinner_rotation(self) -> None:
        """Schedule the next spinner message rotation."""
        if self._spinner_stop_event and self._spinner_stop_event.is_set():
            return
        self._spinner_timer = threading.Timer(3.0, self._rotate_spinner)
        self._spinner_timer.daemon = True
        self._spinner_timer.start()

    def _rotate_spinner(self) -> None:
        """Advance to the next spinner message and schedule the next rotation."""
        if self._spinner_stop_event and self._spinner_stop_event.is_set():
            return
        self._spinner_step += 1
        self._update_spinner_text()
        self._schedule_spinner_rotation()

    def stop_status(self) -> None:
        """
        Safely stop the status spinner and cancel message rotation.

        Handles cases where status might not be initialized or already stopped.
        """
        if self._spinner_stop_event:
            self._spinner_stop_event.set()
        try:
            if self._spinner_timer:
                self._spinner_timer.cancel()
                self._spinner_timer = None
        except Exception:
            pass
        try:
            if self.status and hasattr(self.status, 'stop'):
                self.status.stop()
        except Exception:
            # Silently fail - status stopping is not critical
            pass

    def _start_thinking_spinner(self) -> None:
        """Start a background thread that shows an animated spinner with rotating messages."""
        # Stop any existing spinner to prevent zombie threads from accumulating
        self._stop_thinking_spinner()
        self._thinking_stop_event = threading.Event()

        _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        _stop_event = self._thinking_stop_event  # capture local ref; avoids NoneType if cleared

        def _animate() -> None:
            # Show blinking block cursor while thinking
            sys.stdout.write("\033[?25h\033[1 q")
            sys.stdout.flush()
            msg_idx = 0
            frame_idx = 0
            ticks = 0
            while True:
                frame = _FRAMES[frame_idx % len(_FRAMES)]
                msg = self._spinner_messages[msg_idx % len(self._spinner_messages)]
                line = f"\r\033[1;96m{frame} {msg}\033[0m\033[K"
                sys.stdout.write(line)
                sys.stdout.flush()
                frame_idx += 1
                ticks += 1
                # Rotate message every ~30 frames (~2.4s at 80ms/frame)
                if ticks % 30 == 0:
                    msg_idx += 1
                if _stop_event.wait(0.08):
                    break

        self._thinking_thread = threading.Thread(target=_animate, daemon=True)
        self._thinking_thread.start()

    def _stop_thinking_spinner(self) -> None:
        """Stop the rotating thinking display and clear the line."""
        stop_event = self._thinking_stop_event
        thread = self._thinking_thread
        if stop_event:
            stop_event.set()
        if thread:
            thread.join(timeout=1.0)
        self._thinking_stop_event = None
        self._thinking_thread = None
        # Erase the entire current line and restore blinking block cursor
        sys.stdout.write("\r\033[2K\033[?25h\033[1 q")
        sys.stdout.flush()

    def start_thinking(self) -> None:
        """Public API: start the animated thinking spinner."""
        self._start_thinking_spinner()

    def stop_thinking(self) -> None:
        """Public API: stop the animated thinking spinner."""
        self._stop_thinking_spinner()

    # ── Context usage display ──────────────────────────────────────

    def _fmt_ctx_label(self) -> str:
        """Return a compact ctx label like 'ctx:42%/128k'."""
        label = f"ctx: {self._context_pct:.0f}%"
        if self._context_max_tokens:
            k = f"{self._context_max_tokens // 1000:,}"
            label += f", max toks: {k}k"
        return label

    def set_context_usage(self, pct: float, max_tokens: int = 0) -> None:
        """Update the context window usage percentage (0–100) and max token count."""
        self._context_pct = pct
        if max_tokens:
            self._context_max_tokens = max_tokens

    def show_context_compaction(self, orig_msg_count: int, summary_chars: int) -> None:
        """Print an inline notification when the context window is compacted."""
        self.console.print(
            f"  ⚡ Context compacted  ({orig_msg_count} messages → {summary_chars:,} char summary)",
            style="bold yellow",
        )

    # ── Inline tool-call display (streaming mode) ──────────────────

    def show_tool_start(self, name: str, arguments: dict) -> None:
        """Print a bordered tool-call block inline (mirrors chat history style)."""
        if not self.show_logs:
            return
        self._erase_stream_cursor()  # pause blinking cursor during tool execution
        ts = self.format_timestamp()
        args_str = json.dumps(arguments, ensure_ascii=False)
        t = Text()
        ctx_suffix = f"  {self._fmt_ctx_label()}" if self._context_pct > 0 else ""
        t.append(f"┌─ ⚙️  {name} ", style="bold dark_orange3")
        t.append(f"[{ts}]", style=self.theme.styles.get("timestamp", "cyan"))
        if ctx_suffix:
            t.append(ctx_suffix, style="bright_cyan")
        t.append("\n", style="default")
        t.append(f"{args_str}\n", style="bright_white")
        t.append("└" + "─" * 40, style="dark_orange3")
        self.console.print(t)

    def start_tool_spinner(self, name: str, arguments: dict) -> None:
        pass  # Bordered output provides visual feedback; spinner not needed here

    def stop_tool_spinner(self) -> None:
        pass

    def tool_log(self, name: str, data: str, level: str = "info") -> None:
        """Display real-time log messages from MCP tools (e.g. sandbox output)."""
        if not self.show_logs:
            return
        style = "bright_yellow" if level == "warning" else "bright_red" if level == "error" else "bright_cyan"
        self.console.print(f"  [{name}] {data}", style=style)

    def tool_progress(self, name: str, elapsed_seconds: int) -> None:
        """No-op for text UI; spinner already indicates progress."""
        pass

    def show_tool_done(self, name: str, result: str, success: bool = True) -> None:
        """Print a bordered tool-result block inline (mirrors chat history style)."""
        if not self.show_logs:
            return
        ts = self.format_timestamp()
        display = result if len(result) <= 300 else result[:300] + "…"
        border_style = "bold green" if success else "bold red"
        t = Text()
        t.append(f"┌─ ↩  {name} ", style=border_style)
        t.append(f"[{ts}]\n", style=self.theme.styles.get("timestamp", "cyan"))
        t.append(f"{display}\n", style="bright_white")
        t.append("└" + "─" * 40, style="green" if success else "red")
        self.console.print(t)

    # ── Blinking cursor helpers ────────────────────────────────────

    def _show_stream_cursor(self) -> None:
        """Hide terminal cursor and show a blinking white block."""
        if not self._stream_cursor_shown:
            sys.stdout.write("\033[?25l")          # hide terminal cursor
            sys.stdout.write("\033[5m\u2588\033[0m")  # blinking white block
            sys.stdout.flush()
            self._stream_cursor_shown = True

    def _erase_stream_cursor(self) -> None:
        """Remove the blinking block and restore the terminal cursor."""
        if self._stream_cursor_shown:
            sys.stdout.write("\b \b")              # erase the block char
            sys.stdout.write("\033[?25h")           # restore terminal cursor
            sys.stdout.flush()
            self._stream_cursor_shown = False

    # ── Token streaming ───────────────────────────────────────────

    def stream_start(self) -> None:
        """Prepare for streaming. Resets all stream state; header is deferred lazily."""
        self.stop_status()  # safety: stop spinner if running
        self._stop_thinking_spinner()  # stop rotating lobby messages
        self._streaming_content = ""
        self._stream_header_printed = False
        self._stream_pending = ""
        self._stream_think_started = False
        self._tag_buf = ""
        self._link_buf = ""
        self._link_state = 0
        self._stream_token_count = 0
        self._stream_start_time = time.monotonic()

    def stream_think_token(self, token: str) -> None:
        """Print a reasoning/thinking token in dim italic, opening a think block if needed."""
        if not self._stream_think_started:
            self._stream_think_started = True
            self.console.print(
                f"┌─ 💭 Thinking [{self.format_timestamp()}]",
                style="bright_white",
            )
        self.console.print(token, end="", style="bright_white")

    def stream_think_end(self) -> None:
        """Close the thinking block if it is open."""
        if self._stream_think_started:
            print()  # newline after last think token
            self.console.print("└" + "─" * 40, style="bright_white")
            self._stream_think_started = False

    def stream_token(self, token: str) -> None:
        """Stream an answer token. Auto-closes any open think block on first call."""
        self._streaming_content += token
        self._stream_token_count += 1
        if self._stream_think_started:
            self.stream_think_end()  # reasoning just finished, close think block
        display = self._filter_display_token(token)
        display = self._filter_markdown_links(display)
        if not display:
            return
        self._erase_stream_cursor()
        if not self._stream_header_printed:
            self._stream_pending += display
            if self._stream_pending.strip():
                # First real answer content — print header then flush buffer
                self._stream_header_printed = True
                self.console.print(
                    f"┌─ 🤖 AI ({self.model_name}) [{self.format_timestamp()}]" if self.model_name else f"┌─ 🤖 AI [{self.format_timestamp()}]",
                    style=self.theme.styles.get("assistant", "bold magenta"),
                )
                print(self._stream_pending.lstrip(), end="", flush=True)
                self._stream_pending = ""
                self._show_stream_cursor()
        else:
            if display.strip():
                # Non-whitespace: flush any buffered trailing whitespace then print
                if self._trail_buf:
                    print(self._trail_buf, end="", flush=True)
                    self._trail_buf = ""
                print(display, end="", flush=True)
                self._show_stream_cursor()
            else:
                # Whitespace-only token: buffer it; discard if stream ends before real content
                self._trail_buf += display
                self._show_stream_cursor()

    def _filter_display_token(self, token: str) -> str:
        """Strip <answer>/<answer> wrapper tags, buffering a partial '<' across tokens."""
        buf = self._tag_buf + token
        buf = buf.replace("<answer>", "").replace("</answer>", "")
        if buf.endswith("<"):
            self._tag_buf = "<"
            return buf[:-1]
        self._tag_buf = ""
        return buf

    def _filter_markdown_links(self, text: str) -> str:
        """Filter markdown links from streaming tokens.

        Buffers ``[label](url)`` patterns across token boundaries and emits
        only the label text, suppressing the URL (which may be a huge
        data-URI).  State machine: 0=normal, 1=inside label ``[…``,
        2=consuming URL after ``](``.
        """
        out = []
        for ch in text:
            if self._link_state == 0:
                if ch == '[':
                    self._link_state = 1
                    self._link_buf = ""
                else:
                    out.append(ch)
            elif self._link_state == 1:
                if ch == ']':
                    # Peek-ahead not possible across tokens; buffer the label
                    # and move to a transition state.  We abuse _link_state=2
                    # but first need to see '('.
                    self._link_state = 2
                elif ch == '\n':
                    # Newline inside bracket — not a link, flush as literal
                    out.append('[')
                    out.append(self._link_buf)
                    out.append(ch)
                    self._link_buf = ""
                    self._link_state = 0
                else:
                    self._link_buf += ch
            elif self._link_state == 2:
                if ch == '(':
                    # Confirmed markdown link — consume URL until ')'
                    self._link_state = 3
                else:
                    # Not a markdown link — flush buffered label as literal
                    out.append('[')
                    out.append(self._link_buf)
                    out.append(']')
                    out.append(ch)
                    self._link_buf = ""
                    self._link_state = 0
            elif self._link_state == 3:
                if ch == ')':
                    # End of URL — emit only the label
                    out.append(self._link_buf)
                    self._link_buf = ""
                    self._link_state = 0
                # else: swallow URL characters
        return "".join(out)

    def stream_end(self, elapsed: str = "") -> None:
        """Close the streamed block. Message saving is handled by onit.py (has correct elapsed)."""
        self._erase_stream_cursor()
        self.stream_think_end()  # close think block if still open
        if not self._stream_header_printed:
            # Only whitespace or tool-call-only response — skip the empty block
            self._streaming_content = ""
            self._stream_pending = ""
            self._trail_buf = ""
            self._tag_buf = ""
            self._link_buf = ""
            self._link_state = 0
            return
        # Discard trailing whitespace — just emit a single newline before the footer
        self._trail_buf = ""
        print()  # final newline after the last token
        footer = "└" + "─" * 40
        if elapsed:
            footer += f"  {elapsed}"
        # Append average tokens/sec if we tracked any tokens
        stream_elapsed = time.monotonic() - self._stream_start_time if self._stream_start_time else 0
        if stream_elapsed > 0 and self._stream_token_count > 0:
            tok_s = self._stream_token_count / stream_elapsed
            footer += f"  ({tok_s:.1f} tok/s)"
        if self._context_pct > 0:
            t = Text()
            t.append(footer, style=self.theme.styles.get("assistant", "magenta"))
            t.append(f"  {self._fmt_ctx_label()}", style="bright_cyan")
            self.console.print(t)
        else:
            self.console.print(footer, style=self.theme.styles.get("assistant", "magenta"))
        # Warn when context is getting full (≥75%)
        if self._context_pct >= 90:
            self.console.print(
                f"  ⚠  Context window {self._context_pct:.0f}% full — compaction imminent",
                style="bold red",
            )
        elif self._context_pct >= 75:
            self.console.print(
                f"  ⚠  Context window {self._context_pct:.0f}% full",
                style="bold yellow",
            )
        # Save to history so intermediate AI turns appear in the chat panel.
        # Strip all XML-style tags (same as remove_tags) so the final turn's
        # content matches onit.py's comparison and gets elapsed time updated.
        content = re.sub(r"<[^>]+>", "", self._streaming_content).strip()
        if content:
            self.add_message("assistant", content)
        self._streaming_content = ""
        self._stream_pending = ""
        self._tag_buf = ""
        self._link_buf = ""
        self._link_state = 0
        self._stream_header_printed = False

    def render(self, thinking: bool = False) -> Union[Panel, None]:
        """
        Render complete layout including chat messages and optionally execution logs.

        Args:
            thinking: If True, could show thinking indicator (currently unused)

        Returns:
            Panel: The rendered chat UI panel (logs panel is printed separately if enabled)
        """
        messages_panel = self.render_messages()

        if self.show_logs:
            # Print logs panel first, then messages panel
            self.console.print(self.render_logs_panel())

        return messages_panel
    
    
    def initialize(self):
        """Initialize the chat UI"""
        self.console.clear()
        
        # Show header
        header = Text()
        header.append("╔" + "═" * 50 + "╗\n", style=self.theme.styles.get("prompt", "bold blue"))
        header.append("║", style=self.theme.styles.get("prompt", "bold blue"))
        title_text = f"🌟 {self.banner_title} 🌟"
        padding = max(0, 50 - len(title_text))
        left_pad = padding // 2 - 2
        right_pad = padding - left_pad - 2
        header.append(" " * left_pad + title_text + " " * right_pad, style=self.theme.styles.get("prompt", "bold white on blue"))
        header.append("║\n", style=self.theme.styles.get("prompt", "bold blue"))
        header.append("╚" + "═" * 50 + "╝", style=self.theme.styles.get("prompt", "bold blue"))
        self.console.print(Align.center(header))
        self.console.print()
        self.console.print(Align.center(Text("OnIt may produce inaccurate information. Verify important details independently.", style="bright_white")))

    def _input_with_history(self, prompt: str = "➤ ") -> str:
        """Dispatch to the platform-appropriate raw-input implementation."""
        if sys.platform == "win32":
            return self._input_with_history_windows(prompt)
        return self._input_with_history_unix(prompt)

    @staticmethod
    def _redraw_line(prompt: str, current_input: list[str], prev_lines: int = 1) -> int:
        """Clear and redraw the prompt with the given input (supports multiline).

        Returns:
            Number of display lines used.
        """
        # Move cursor up to the first line of the current display
        if prev_lines > 1:
            sys.stdout.write(f'\033[{prev_lines - 1}A')
        # Clear from start of first line to end of screen
        sys.stdout.write('\r\033[J')

        text = ''.join(current_input)
        lines = text.split('\n')
        sys.stdout.write(f"\033[1;32m{prompt}\033[0m{lines[0]}")
        for line in lines[1:]:
            sys.stdout.write(f"\r\n\033[1;32m  \033[0m{line}")
        sys.stdout.flush()
        return len(lines)

    def _handle_arrow_keys(
        self,
        code: str,
        prompt: str,
        current_input: list[str],
        cursor_pos: int,
        temp_history_index: int,
        saved_input: str,
        num_display_lines: int = 1,
    ) -> tuple[list[str], int, int, str, int]:
        """Handle up/down/left/right arrow key navigation.

        Returns:
            Updated (current_input, cursor_pos, temp_history_index, saved_input, num_display_lines).
        """
        if code == 'A' and self.input_history:  # Up arrow
            if temp_history_index == -1:
                saved_input = ''.join(current_input)
                temp_history_index = len(self.input_history) - 1
            elif temp_history_index > 0:
                temp_history_index -= 1

            if temp_history_index >= 0:
                current_input = list(self.input_history[temp_history_index])
                cursor_pos = len(current_input)
                num_display_lines = self._redraw_line(prompt, current_input, num_display_lines)

        elif code == 'B' and temp_history_index != -1:  # Down arrow
            temp_history_index += 1
            if temp_history_index >= len(self.input_history):
                temp_history_index = -1
                current_input = list(saved_input)
            else:
                current_input = list(self.input_history[temp_history_index])
            cursor_pos = len(current_input)
            num_display_lines = self._redraw_line(prompt, current_input, num_display_lines)

        elif code == 'C' and cursor_pos < len(current_input):  # Right arrow
            cursor_pos += 1
            sys.stdout.write('\033[C')
            sys.stdout.flush()

        elif code == 'D' and cursor_pos > 0:  # Left arrow
            cursor_pos -= 1
            sys.stdout.write('\033[D')
            sys.stdout.flush()

        return current_input, cursor_pos, temp_history_index, saved_input, num_display_lines

    @staticmethod
    def _handle_delete_key(
        current_input: list[str],
        cursor_pos: int,
        fd: int,
    ) -> tuple[list[str], int]:
        """Handle the Delete key (ESC [ 3 ~), removing the char at cursor_pos."""
        rlist, _, _ = _select.select([fd], [], [], 0.05)
        if not rlist:
            return current_input, cursor_pos
        next3 = os.read(fd, 1).decode('utf-8', errors='replace')
        if next3 == '~' and cursor_pos < len(current_input):
            del current_input[cursor_pos]
            sys.stdout.write(''.join(current_input[cursor_pos:]) + ' ')
            sys.stdout.write('\033[' + str(len(current_input) - cursor_pos + 1) + 'D')
            sys.stdout.flush()
        return current_input, cursor_pos

    @staticmethod
    def _drain_escape_sequence(fd: int) -> None:
        """Drain remaining bytes of an unrecognized escape sequence without blocking.

        Reads until no byte arrives within 20 ms or a typical sequence terminator
        ('~' or an uppercase ASCII letter) is seen — whichever comes first.
        This prevents stray bytes from unhandled sequences (PgUp, PgDn, Fn keys,
        etc.) from leaking into normal character input.
        """
        while True:
            rlist, _, _ = _select.select([fd], [], [], 0.02)
            if not rlist:
                break
            ch = os.read(fd, 1).decode('utf-8', errors='replace')
            if ch == '~' or ('A' <= ch <= 'Z'):
                break

    @staticmethod
    def _handle_backspace(
        current_input: list[str],
        cursor_pos: int,
    ) -> tuple[list[str], int]:
        """Handle the Backspace key, removing the char before cursor_pos.

        Returns:
            Updated (current_input, cursor_pos).
        """
        if cursor_pos > 0:
            cursor_pos -= 1
            del current_input[cursor_pos]
            sys.stdout.write('\b' + ''.join(current_input[cursor_pos:]) + ' ')
            sys.stdout.write('\033[' + str(len(current_input) - cursor_pos + 1) + 'D')
            sys.stdout.flush()
        return current_input, cursor_pos

    @staticmethod
    def _handle_printable(
        char: str,
        current_input: list[str],
        cursor_pos: int,
    ) -> tuple[list[str], int]:
        """Insert a printable character at cursor_pos and update the display.

        Returns:
            Updated (current_input, cursor_pos).
        """
        current_input.insert(cursor_pos, char)
        cursor_pos += 1
        sys.stdout.write(char + ''.join(current_input[cursor_pos:]))
        if cursor_pos < len(current_input):
            sys.stdout.write('\033[' + str(len(current_input) - cursor_pos) + 'D')
        sys.stdout.flush()
        return current_input, cursor_pos

    def _input_with_history_unix(self, prompt: str = "➤ ") -> str:
        """
        Get user input with history navigation using up/down arrow keys.
        Uses termios/tty for raw terminal control (Unix/Linux/macOS only).
        Supports bracketed paste mode for pasting multi-line text.

        Args:
            prompt: The prompt string to display

        Returns:
            The user's input string
        """
        # Enable bracketed paste mode + show blinking bar cursor + bold green prompt
        sys.stdout.write(f"\033[?2004h\033[?25h\033[1 q\033[1;32m{prompt}\033[0m")
        sys.stdout.flush()

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)  # type: ignore[name-defined]

        try:
            tty.setraw(fd)  # type: ignore[name-defined]

            current_input: list[str] = []
            cursor_pos = 0
            temp_history_index = -1
            saved_input = ""
            in_paste = False
            num_display_lines = 1

            while True:
                char = os.read(fd, 1).decode('utf-8', errors='replace')

                if char in ('\r', '\n'):
                    if in_paste:
                        # Insert literal newline during paste
                        current_input.insert(cursor_pos, '\n')
                        cursor_pos += 1
                        num_display_lines += 1
                        sys.stdout.write('\r\n\033[1;32m  \033[0m')
                        sys.stdout.flush()
                    else:
                        # Submit input
                        sys.stdout.write('\r\n')
                        sys.stdout.flush()
                        result = ''.join(current_input)
                        if result.strip() and (not self.input_history or self.input_history[-1] != result):
                            self.input_history.append(result)
                        self.history_index = -1
                        return result

                elif char == '\x03':  # Ctrl+C
                    sys.stdout.write('\r\n')
                    sys.stdout.flush()
                    raise KeyboardInterrupt

                elif char == '\x04':  # Ctrl+D
                    sys.stdout.write('\r\n')
                    sys.stdout.flush()
                    raise EOFError

                elif char == '\x1b':  # Escape sequence
                    # Use select with a 50 ms timeout to distinguish a standalone
                    # ESC key from the start of a multi-byte terminal sequence.
                    # Without this, a bare ESC press would block the next read(1)
                    # call and silently consume (drop) the following keypress.
                    rlist, _, _ = _select.select([fd], [], [], 0.05)
                    if not rlist:
                        continue  # Standalone ESC — ignore, nothing to drop

                    next1 = os.read(fd, 1).decode('utf-8', errors='replace')
                    if next1 == '[':
                        rlist, _, _ = _select.select([fd], [], [], 0.05)
                        if not rlist:
                            continue
                        next2 = os.read(fd, 1).decode('utf-8', errors='replace')
                        if next2 in ('A', 'B', 'C', 'D'):  # Arrow keys only
                            current_input, cursor_pos, temp_history_index, saved_input, num_display_lines = (
                                self._handle_arrow_keys(
                                    next2, prompt, current_input, cursor_pos,
                                    temp_history_index, saved_input, num_display_lines,
                                )
                            )
                        elif next2 == '2':
                            # \e[2~  = Insert key
                            # \e[200~ = bracketed-paste start
                            # \e[201~ = bracketed-paste end
                            rlist, _, _ = _select.select([fd], [], [], 0.05)
                            if not rlist:
                                continue
                            next3 = os.read(fd, 1).decode('utf-8', errors='replace')
                            if next3 == '~':
                                pass  # Insert key — ignore
                            elif next3 == '0':
                                # Could be \e[20~  (F9) or \e[200~/\e[201~ (paste)
                                rlist, _, _ = _select.select([fd], [], [], 0.05)
                                if not rlist:
                                    continue
                                next4 = os.read(fd, 1).decode('utf-8', errors='replace')
                                if next4 == '~':
                                    pass  # \e[20~ = F9 — ignore (was the bug: previously
                                          # read one extra byte here, dropping user input)
                                elif next4 in ('0', '1'):
                                    # \e[200~ or \e[201~: one more byte expected ('~')
                                    rlist, _, _ = _select.select([fd], [], [], 0.05)
                                    if rlist:
                                        next5 = os.read(fd, 1).decode('utf-8', errors='replace')
                                        if next5 == '~':
                                            in_paste = (next4 == '0')
                                else:
                                    self._drain_escape_sequence(fd)
                            else:
                                self._drain_escape_sequence(fd)
                        elif next2 == '3':  # Delete key: \e[3~
                            current_input, cursor_pos = self._handle_delete_key(
                                current_input, cursor_pos, fd,
                            )
                        else:
                            # Unrecognised sequence (Home \e[H/\e[1~, End \e[F/\e[4~,
                            # PgUp \e[5~, PgDn \e[6~, Fn keys, etc.).
                            # Drain any remaining bytes so they don't appear as text.
                            self._drain_escape_sequence(fd)

                elif char in ('\x7f', '\b'):  # Backspace
                    current_input, cursor_pos = self._handle_backspace(
                        current_input, cursor_pos,
                    )

                elif ' ' <= char <= '~':  # Printable characters
                    current_input, cursor_pos = self._handle_printable(
                        char, current_input, cursor_pos,
                    )

        finally:
            # Disable bracketed paste mode and restore terminal
            sys.stdout.write("\033[?2004l")
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # type: ignore[name-defined]

    def _input_with_history_windows(self, prompt: str = "➤ ") -> str:
        """
        Get user input with history navigation using up/down arrow keys.
        Uses msvcrt for raw character reading (Windows only).

        Arrow keys on Windows emit two bytes via msvcrt.getwch():
          prefix \x00 or \xe0, then a scan code:
            \x48 = Up, \x50 = Down, \x4d = Right, \x4b = Left, \x53 = Delete

        Args:
            prompt: The prompt string to display

        Returns:
            The user's input string
        """
        # Show blinking bar cursor + bold green prompt
        sys.stdout.write(f"\033[?25h\033[5 q\033[1;32m{prompt}\033[0m")
        sys.stdout.flush()

        current_input = []
        cursor_pos = 0
        temp_history_index = -1
        saved_input = ""

        while True:
            char = msvcrt.getwch()

            if char in ('\r', '\n'):  # Enter key
                sys.stdout.write('\r\n')
                sys.stdout.flush()
                result = ''.join(current_input)
                if result.strip() and (not self.input_history or self.input_history[-1] != result):
                    self.input_history.append(result)
                self.history_index = -1
                return result

            elif char == '\x03':  # Ctrl+C
                sys.stdout.write('\r\n')
                sys.stdout.flush()
                raise KeyboardInterrupt

            elif char == '\x04':  # Ctrl+D
                sys.stdout.write('\r\n')
                sys.stdout.flush()
                raise EOFError

            elif char in ('\x00', '\xe0'):  # Special key prefix (arrows, Delete, etc.)
                scan = msvcrt.getwch()

                if scan == '\x48':  # Up arrow
                    if self.input_history:
                        if temp_history_index == -1:
                            saved_input = ''.join(current_input)
                            temp_history_index = len(self.input_history) - 1
                        elif temp_history_index > 0:
                            temp_history_index -= 1

                        if temp_history_index >= 0:
                            sys.stdout.write('\r' + ' ' * (len(prompt) + len(current_input) + 5) + '\r')
                            sys.stdout.write(f"\033[1;32m{prompt}\033[0m")
                            current_input = list(self.input_history[temp_history_index])
                            cursor_pos = len(current_input)
                            sys.stdout.write(''.join(current_input))
                            sys.stdout.flush()

                elif scan == '\x50':  # Down arrow
                    if temp_history_index != -1:
                        temp_history_index += 1
                        sys.stdout.write('\r' + ' ' * (len(prompt) + len(current_input) + 5) + '\r')
                        sys.stdout.write(f"\033[1;32m{prompt}\033[0m")

                        if temp_history_index >= len(self.input_history):
                            temp_history_index = -1
                            current_input = list(saved_input)
                        else:
                            current_input = list(self.input_history[temp_history_index])

                        cursor_pos = len(current_input)
                        sys.stdout.write(''.join(current_input))
                        sys.stdout.flush()

                elif scan == '\x4d':  # Right arrow
                    if cursor_pos < len(current_input):
                        cursor_pos += 1
                        sys.stdout.write('\033[C')
                        sys.stdout.flush()

                elif scan == '\x4b':  # Left arrow
                    if cursor_pos > 0:
                        cursor_pos -= 1
                        sys.stdout.write('\033[D')
                        sys.stdout.flush()

                elif scan == '\x53':  # Delete key
                    if cursor_pos < len(current_input):
                        del current_input[cursor_pos]
                        sys.stdout.write(''.join(current_input[cursor_pos:]) + ' ')
                        sys.stdout.write('\033[' + str(len(current_input) - cursor_pos + 1) + 'D')
                        sys.stdout.flush()

            elif char in ('\x7f', '\x08'):  # Backspace
                if cursor_pos > 0:
                    cursor_pos -= 1
                    del current_input[cursor_pos]
                    sys.stdout.write('\b' + ''.join(current_input[cursor_pos:]) + ' ')
                    sys.stdout.write('\033[' + str(len(current_input) - cursor_pos + 1) + 'D')
                    sys.stdout.flush()

            elif char >= ' ' and char <= '~':  # Printable characters
                current_input.insert(cursor_pos, char)
                cursor_pos += 1
                sys.stdout.write(char + ''.join(current_input[cursor_pos:]))
                if cursor_pos < len(current_input):
                    sys.stdout.write('\033[' + str(len(current_input) - cursor_pos) + 'D')
                sys.stdout.flush()

    def get_user_input(self) -> str:
        """Get user input from the console with history support (up/down arrows)"""
        self.stop_status()
        self.stop_thinking()  # ensure no spinner thread is writing to stdout while user types
        while True:
            self.console.clear()
            self.console.print(self.render())
            self.console.print(Align.center(Text("OnIt may produce inaccurate information. Verify important details independently.", style="bright_white")))
            try:
                self.console.print()
                user_input = self._input_with_history("➤ ")
                self.add_message("user", user_input)
                self.console.clear()

                # Show panel with the user's message, then a plain (non-Live)
                # thinking indicator so the console is free for streaming tokens.
                self.console.print(self.render())
                self.console.print(Align.center(Text("OnIt may produce inaccurate information. Verify important details independently.", style="bright_white")))
                self.console.print()
                return user_input
            # capture control-d
            except EOFError:
                return "\\bye"
            except KeyboardInterrupt:
                return "\\bye"
    

### Legacy
async def text_prompt(console: Console = None,
                status: Status = None,
                message: str = "Type '\\bye' to stop.",
                prompt_color: str = "bold blue",
                cursor: str = "[bold]User[/bold]:",
                tool_registry: Any = None,
                is_voice: bool = False,
                say: Optional[Callable] = None,
                listen: Optional[Callable] = None,
                prompted_once: bool = True,
                is_safety_task: bool = False) -> str:
    assert Console is not None, "Console must be provided for CLI prompt."
    if is_voice:
        if not prompted_once:
            console.print(message, style="info")
            if say:
                await say(text=message)
        try:
            if is_safety_task:
                prompt = f"[{prompt_color}] 💡 On-it... Say 'stop' to stop all tasks.[/]"
            else:
                prompt = f"[{prompt_color}] 🎤 Listening... Say 'goodbye' to end the session.[/]"
            status.update(prompt) if status else None
            status.start() if status else None
            task = await listen(tool_registry=tool_registry, status=status, prompt_color=prompt_color, is_safety_task=is_safety_task)
            status.stop() if status else None
            if is_safety_task:
                console.print(f"Stopping all tasks...", style="warning")
            elif task is not None:
                console.print(f"{cursor} : {task}", style="prompt")
            return task
        
        except Exception as e:
            console.print(f"An error occurred: {str(e)}. Please try again", style="error")
            return None                   
    else:                  
        # put status on upper layout
        if status and isinstance(status, Status):
            status.stop() if status else None
        if not prompted_once and console:
            console.print(message, style="info")
        loop = asyncio.get_running_loop()
        prompt = f"[bold]{cursor}[/bold]"
        if is_safety_task:
            prompt = f"[{prompt_color}] 🎤 On-it... Press 'Ctl+c' key to stop all tasks.[/]"
            status.update(prompt)
            status.start()
        try:
            console.clear()                                                                                                                                                           
            task = await loop.run_in_executor(None, lambda: Prompt.ask(prompt, console=console))
        except KeyboardInterrupt:
            task = None
        if is_safety_task:
            if status:
                status.stop()
            console.print(f"Stopping all tasks...", style="warning")
        return task if task is not None else ""
    


def main():
    """Entry point"""
    # Check for rich library
    try:
        from rich import __version__
        console = Console()
        console.print(f"[dim]Rich version: {__version__}[/]\n")
    except ImportError:
        print("Please install rich: pip install rich")
        return
    
    chat = ChatUI()
    #chat.run()


if __name__ == "__main__":
    main()