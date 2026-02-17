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
import sys
import tty
import termios
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
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str
    elapsed: str = ""

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

                if role == "user":
                    self._render_user_message(content, msg_content, msg_time)
                else:
                    self._render_assistant_message(content, msg_content, msg_time, msg_elapsed)

            panel = Panel(
                content,
                title="[bold white]💬 Chat History[/]",
                title_align="left",
                border_style="blue",
                box=HORIZONTAL_THICK,
                padding=(1, 0),
                subtitle=f"Showing last {len(display_msgs)} of {len(self.messages)} messages",
                subtitle_align="right"
            )

            # Scroll to the last message by printing enough newlines to push content up
            if self.messages:
                self.console.print()  # Add spacing after panel

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

    def _render_assistant_message(self, content: Text, msg_content: str, msg_time: str, msg_elapsed: str) -> None:
        """
        Render an assistant message to the content Text object.

        Args:
            content: The Rich Text object to append to
            msg_content: The message content
            msg_time: The message timestamp
            msg_elapsed: The elapsed time string (optional)
        """
        content.append(f"┌─ 🤖 AI ", style=self.theme.styles.get("assistant", "bold magenta"))
        content.append(f"[{msg_time}] - ", style=self.theme.styles.get("timestamp", "dim magenta"))
        if msg_elapsed:
            content.append(f"{msg_elapsed}\n", style=self.theme.styles.get("warning", "dim magenta"))
        else:
            content.append("\n", style=self.theme.styles.get("warning", "dim magenta"))
        content.append(f"{msg_content}\n", style="bright_white")
        content.append("└" + "─" * 40 + "\n", style=self.theme.styles.get("assistant", "magenta"))

    def render_thinking_panel(self) -> Panel:
        """Render thinking indicator in input panel"""
        thinking = Text()
        thinking.append("🤖 ", style="bold")
        thinking.append("Assistant is thinking", style="bold magenta")
        thinking.append("...", style="magenta blink")
        
        return Panel(
            thinking,
            title="[bold white]⌨️  Please Wait[/]",
            title_align="left",
            border_style="magenta",
            box=HEAVY,
            padding=(0, 2),
            height=3
        )
    
    def stop_status(self) -> None:
        """
        Safely stop the status spinner.

        Handles cases where status might not be initialized or already stopped.
        """
        try:
            if self.status and hasattr(self.status, 'stop'):
                self.status.stop()
        except Exception:
            # Silently fail - status stopping is not critical
            pass

    def render(self, thinking: bool = False) -> Union[Panel, None]:
        """
        Render complete layout including chat messages and optionally execution logs.

        Args:
            thinking: If True, could show thinking indicator (currently unused)

        Returns:
            Panel: The rendered chat UI panel (logs panel is printed separately if enabled)
        """
        self.stop_status()
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
        self.console.print(Align.center(Text("OnIt may produce inaccurate information. Verify important details independently.", style="dim italic")))
    
    def _input_with_history(self, prompt: str = "➤ ") -> str:
        """
        Get user input with history navigation using up/down arrow keys.

        Args:
            prompt: The prompt string to display

        Returns:
            The user's input string
        """
        # Print prompt
        sys.stdout.write(f"\033[1;32m{prompt}\033[0m")  # Bold green prompt
        sys.stdout.flush()

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)

            current_input = []
            cursor_pos = 0
            temp_history_index = -1
            saved_input = ""  # Save current input when browsing history

            while True:
                char = sys.stdin.read(1)

                if char == '\r' or char == '\n':  # Enter key
                    sys.stdout.write('\r\n')
                    sys.stdout.flush()
                    result = ''.join(current_input)
                    # Add to history if non-empty and different from last entry
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

                elif char == '\x1b':  # Escape sequence (arrow keys)
                    next1 = sys.stdin.read(1)
                    if next1 == '[':
                        next2 = sys.stdin.read(1)

                        if next2 == 'A':  # Up arrow
                            if self.input_history:
                                if temp_history_index == -1:
                                    # Save current input before browsing
                                    saved_input = ''.join(current_input)
                                    temp_history_index = len(self.input_history) - 1
                                elif temp_history_index > 0:
                                    temp_history_index -= 1

                                if temp_history_index >= 0:
                                    # Clear current line
                                    sys.stdout.write('\r' + ' ' * (len(prompt) + len(current_input) + 5) + '\r')
                                    sys.stdout.write(f"\033[1;32m{prompt}\033[0m")
                                    # Show history item
                                    current_input = list(self.input_history[temp_history_index])
                                    cursor_pos = len(current_input)
                                    sys.stdout.write(''.join(current_input))
                                    sys.stdout.flush()

                        elif next2 == 'B':  # Down arrow
                            if temp_history_index != -1:
                                temp_history_index += 1
                                # Clear current line
                                sys.stdout.write('\r' + ' ' * (len(prompt) + len(current_input) + 5) + '\r')
                                sys.stdout.write(f"\033[1;32m{prompt}\033[0m")

                                if temp_history_index >= len(self.input_history):
                                    # Restore saved input
                                    temp_history_index = -1
                                    current_input = list(saved_input)
                                else:
                                    current_input = list(self.input_history[temp_history_index])

                                cursor_pos = len(current_input)
                                sys.stdout.write(''.join(current_input))
                                sys.stdout.flush()

                        elif next2 == 'C':  # Right arrow
                            if cursor_pos < len(current_input):
                                cursor_pos += 1
                                sys.stdout.write('\033[C')
                                sys.stdout.flush()

                        elif next2 == 'D':  # Left arrow
                            if cursor_pos > 0:
                                cursor_pos -= 1
                                sys.stdout.write('\033[D')
                                sys.stdout.flush()

                        elif next2 == '3':  # Delete key (might have another char)
                            next3 = sys.stdin.read(1)
                            if next3 == '~' and cursor_pos < len(current_input):
                                # Delete character at cursor
                                del current_input[cursor_pos]
                                # Rewrite line from cursor
                                sys.stdout.write(''.join(current_input[cursor_pos:]) + ' ')
                                # Move cursor back
                                sys.stdout.write('\033[' + str(len(current_input) - cursor_pos + 1) + 'D')
                                sys.stdout.flush()

                elif char == '\x7f' or char == '\b':  # Backspace
                    if cursor_pos > 0:
                        cursor_pos -= 1
                        del current_input[cursor_pos]
                        # Move cursor back, rewrite rest of line, add space to clear last char
                        sys.stdout.write('\b' + ''.join(current_input[cursor_pos:]) + ' ')
                        # Move cursor back to position
                        sys.stdout.write('\033[' + str(len(current_input) - cursor_pos + 1) + 'D')
                        sys.stdout.flush()

                elif char >= ' ' and char <= '~':  # Printable characters
                    current_input.insert(cursor_pos, char)
                    cursor_pos += 1
                    # Write char and rest of line
                    sys.stdout.write(char + ''.join(current_input[cursor_pos:]))
                    # Move cursor back if not at end
                    if cursor_pos < len(current_input):
                        sys.stdout.write('\033[' + str(len(current_input) - cursor_pos) + 'D')
                    sys.stdout.flush()

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def get_user_input(self) -> str:
        """Get user input from the console with history support (up/down arrows)"""
        while True:
            self.console.print(self.render())
            self.console.print(Align.center(Text("OnIt may produce inaccurate information. Verify important details independently.", style="dim italic")))
            try:
                self.console.print()
                user_input = self._input_with_history("➤ ")
                self.add_message("user", user_input)
                self.console.clear()

                # Show thinking indicator
                self.console.print(self.render())
                self.console.print(Align.center(Text("OnIt may produce inaccurate information. Verify important details independently.", style="dim italic")))
                self.console.print()
                self.status.update(f"[{self.theme.styles.get('prompt', 'bold dark_blue')}] 💡 On it...[/]")
                self.status.start()
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