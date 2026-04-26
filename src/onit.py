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

OnIt: An intelligent agent framework for task automation and assistance.

"""

import asyncio
import base64
import os
import time
import tempfile
import yaml
import json
import uuid

from pathlib import Path
from typing import Union, Any
from pydantic import BaseModel, ConfigDict, Field
from fastmcp import Client

import logging
import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings:.*")

logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from httpx/httpcore (used by FastMCP client)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from .lib.tools import discover_tools
from .lib.text import remove_tags
from .lib.files import has_code_files, zip_code_files
from .ui import ChatUI
from .model.serving.chat import chat

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Part, TaskState, TaskStatus, TaskStatusUpdateEvent
from a2a.helpers.proto_helpers import new_text_message

AGENT_CURSOR = "OnIt"
USER_CURSOR = "You"
STOP_TAG = "<stop></stop>"


async def _call_sandbox_stop(tool_registry, session_id: str = "", sandbox: bool = False) -> None:
    """Call sandbox_stop via the tool registry if sandbox mode is enabled and the tool is available."""
    if not sandbox or not tool_registry or "sandbox_stop" not in tool_registry.tools:
        return
    try:
        handler = tool_registry["sandbox_stop"]
        if handler:
            kwargs = {}
            if session_id:
                kwargs["session_id"] = session_id
            await asyncio.wait_for(handler(**kwargs), timeout=10)
    except Exception as e:
        logger.warning("sandbox_stop failed: %s", e)


class StreamingAdapter:
    """Minimal chat_ui adapter that forwards streaming tokens to a callback.

    Implements the subset of the ChatUI interface used by ``chat()`` so that
    ``process_task`` callers (web UI, A2A) can receive tokens incrementally
    without a full terminal UI.
    """

    def __init__(self, on_token=None, on_complete=None, show_logs=False,
                 throttle_tokens=0, on_tool_status=None):
        self.on_token = on_token
        self.on_complete = on_complete
        self.show_logs = show_logs
        self._throttle_tokens = throttle_tokens
        self._on_tool_status = on_tool_status
        self._content = ""
        self._tag_buf = ""
        self._token_count = 0
        self._total_tokens = 0
        self._start_time = 0.0
        self._pending: list[asyncio.Task] = []
        self.messages = []

    # ── streaming ────────────────────────────────────────────────
    def stream_start(self):
        self._content = ""
        self._tag_buf = ""
        self._token_count = 0
        self._total_tokens = 0
        self._start_time = time.monotonic()

    def stream_token(self, token):
        # Strip <answer></answer> wrapper tags
        buf = self._tag_buf + token
        buf = buf.replace("<answer>", "").replace("</answer>", "")
        if buf.endswith("<"):
            self._tag_buf = "<"
            token = buf[:-1]
        else:
            self._tag_buf = ""
            token = buf
        self._content += token
        self._total_tokens += 1
        if not (self.on_token and token):
            return
        self._token_count += 1
        # Throttle: skip intermediate tokens when configured
        if self._throttle_tokens and (self._token_count % self._throttle_tokens != 0):
            return
        result = self.on_token(token, self._content)
        # Support async callbacks (e.g. A2A event_queue) — track the
        # futures so they can be flushed before the caller returns.
        if asyncio.iscoroutine(result):
            task = asyncio.ensure_future(result)
            self._pending.append(task)

    def stream_think_token(self, token):
        pass  # skip think tokens for external clients

    @property
    def tokens_per_second(self) -> float:
        """Return average tokens/sec for the current or last stream."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        if elapsed > 0 and self._total_tokens > 0:
            return self._total_tokens / elapsed
        return 0.0

    def stream_end(self, elapsed=""):
        if self.on_complete:
            self.on_complete(self._content, self.tokens_per_second)
        self._content = ""
        self._tag_buf = ""
        self._token_count = 0

    async def flush(self):
        """Await all pending async callbacks so no events are lost."""
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
            self._pending.clear()
        # Send one final callback with the latest content so the client
        # is guaranteed to see the full accumulated text before the final
        # response message arrives.
        if self.on_token and self._throttle_tokens and self._content:
            result = self.on_token("", self._content)
            if asyncio.iscoroutine(result):
                await result

    # ── tool display (no-ops for external clients) ───────────────
    def add_tool_call(self, name, arguments):
        pass

    def show_tool_start(self, name, arguments):
        if self.show_logs:
            print(f"{name}({arguments})")

    def start_tool_spinner(self, name, arguments):
        if self._on_tool_status:
            self._on_tool_status(f"{name}({arguments})")

    def stop_tool_spinner(self):
        if self._on_tool_status:
            self._on_tool_status("")

    def show_tool_done(self, name, result, success=True):
        if self._on_tool_status:
            self._on_tool_status("")
        if self.show_logs:
            truncated = result[:500] + "..." if len(result) > 500 else result
            print(f"{name} returned: {truncated}")

    def tool_log(self, name: str, data: str, level: str = "info") -> None:
        """Called when a tool emits a log/notification message (e.g. sandbox output)."""
        if self._on_tool_status:
            self._on_tool_status(f"[{name}] {data}")
        if self.on_token:
            # Forward sandbox output as a streaming token so the UI shows it in real-time
            log_line = f"\n[{name}] {data}\n"
            self._content += log_line
            result = self.on_token(log_line, self._content)
            if asyncio.iscoroutine(result):
                task = asyncio.ensure_future(result)
                self._pending.append(task)

    def tool_progress(self, name, elapsed_seconds):
        """Called periodically during long-running tool calls to keep SSE alive."""
        if self._on_tool_status:
            self._on_tool_status(f"{name} running… ({elapsed_seconds}s)")
        if self.on_token:
            # Send an empty-content SSE event as a keepalive heartbeat
            result = self.on_token("", self._content)
            if asyncio.iscoroutine(result):
                task = asyncio.ensure_future(result)
                self._pending.append(task)

    def add_tool_result(self, name, result, truncate=300):
        pass

    def add_log(self, message, level="info"):
        if self.show_logs:
            print(f"[{level}] {message}")

    def set_context_usage(self, pct: float) -> None:
        """No-op for external clients; context % is informational only."""
        pass

    def show_context_compaction(self, orig_msg_count: int, summary_chars: int) -> None:
        """Forward context compaction notice as a streaming token."""
        if self.on_token:
            msg = f"\n[Context compacted: {orig_msg_count} messages → {summary_chars:,} char summary]\n"
            self._content += msg
            result = self.on_token(msg, self._content)
            if asyncio.iscoroutine(result):
                task = asyncio.ensure_future(result)
                self._pending.append(task)


class OnItA2AExecutor(AgentExecutor):
    """A2A executor that delegates task processing to an OnIt instance.

    Each A2A context (client conversation) gets its own isolated session
    with separate chat history, data directory, and safety queue — following
    the same pattern as the Telegram and Viber gateways.
    """

    def __init__(self, onit):
        self.onit = onit
        # Per-context session state: context_key -> {session_id, session_path, data_path, safety_queue}
        self._sessions: dict[str, dict] = {}
        # Track active safety_queue per asyncio task for disconnect middleware
        self._active_safety_queues: dict[int, asyncio.Queue] = {}

    def _get_session(self, context: 'RequestContext') -> dict:
        """Get or create session state for an A2A context."""
        # Use context_id to group related tasks from the same client,
        # fall back to task_id for one-off requests
        key = context.context_id or context.task_id or str(uuid.uuid4())
        if key not in self._sessions:
            session_id = str(uuid.uuid4())
            sessions_dir = os.path.dirname(self.onit.session_path)
            session_path = os.path.join(sessions_dir, f"{session_id}.jsonl")
            if not os.path.exists(session_path):
                with open(session_path, "w", encoding="utf-8") as f:
                    f.write("")
            configured_data_path = self.onit.config_data.get('data_path')
            if configured_data_path:
                data_path = str(Path(configured_data_path).expanduser().resolve())
            else:
                data_path = str(Path(tempfile.gettempdir()) / "onit" / "data" / session_id)
            os.makedirs(data_path, exist_ok=True)
            self._sessions[key] = {
                "session_id": session_id,
                "session_path": session_path,
                "data_path": data_path,
                "safety_queue": asyncio.Queue(maxsize=10),
            }
            logger.info("Created new A2A session %s for context %s", session_id, key)
        return self._sessions[key]

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.get_user_input()
        if not context.message:
            raise Exception('No message provided')

        session = self._get_session(context)

        # Extract inline file parts from the A2A message and save to session data folder
        image_paths = []
        file_paths = []
        for part in context.message.parts:
            if part.HasField('raw') and part.raw:
                safe_name = os.path.basename(part.filename or 'file')
                filepath = os.path.join(session["data_path"], safe_name)
                with open(filepath, 'wb') as f:
                    f.write(part.raw)
                if part.media_type and part.media_type.startswith('image/'):
                    image_paths.append(filepath)
                else:
                    file_paths.append(filepath)

        # Append file references to task so the agent knows about them
        if file_paths:
            file_refs = "\n".join(f"- {fp}" for fp in file_paths)
            task = f"{task}\n\nFiles uploaded to data folder:\n{file_refs}"

        # Register safety_queue for disconnect middleware
        current_task_id = id(asyncio.current_task())
        self._active_safety_queues[current_task_id] = session["safety_queue"]

        # Stream partial progress back to the A2A client as "working" status events
        _task_id = context.task_id or ""
        _context_id = context.context_id or ""

        async def _a2a_stream_callback(_token, full_content):
            try:
                event = TaskStatusUpdateEvent(
                    task_id=_task_id,
                    context_id=_context_id,
                    status=TaskStatus(
                        state=TaskState.TASK_STATE_WORKING,
                        message=new_text_message(full_content),
                    ),
                )
                await event_queue.enqueue_event(event)
            except Exception:
                pass  # best-effort streaming

        try:
            _stats = {}
            _task_start = time.monotonic()
            result = await self.onit.process_task(
                task,
                images=image_paths if image_paths else None,
                session_path=session["session_path"],
                data_path=session["data_path"],
                safety_queue=session["safety_queue"],
                stream_callback=_a2a_stream_callback,
                stream_throttle=10,
                stats=_stats,
                session_id=session["session_id"],
            )
            _task_elapsed = time.monotonic() - _task_start
        except asyncio.CancelledError:
            session["safety_queue"].put_nowait(STOP_TAG)
            raise
        finally:
            self._active_safety_queues.pop(current_task_id, None)

        # Append elapsed time and tokens/sec to the final response text
        tok_s = _stats.get("tokens_per_second", 0)
        _footer_parts = []
        if _task_elapsed > 0:
            _footer_parts.append(f"{_task_elapsed:.2f}s")
        if tok_s > 0:
            _footer_parts.append(f"{tok_s:.1f} tok/s")
        if _footer_parts:
            result = f"{result}\n\n({' · '.join(_footer_parts)})"

        message = new_text_message(result)

        # Attach codebase zip when code files were generated
        zip_path = zip_code_files(session["data_path"])
        if zip_path:
            with open(zip_path, "rb") as zf:
                zip_bytes = zf.read()
            zip_name = os.path.basename(zip_path)
            message.parts.add(
                raw=zip_bytes,
                media_type="application/zip",
                filename=zip_name,
            )

        await event_queue.enqueue_event(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        session = self._get_session(context)
        session["safety_queue"].put_nowait(STOP_TAG)
        await _call_sandbox_stop(self.onit.tool_registry, session["session_id"], sandbox=self.onit.sandbox)


class ClientDisconnectMiddleware:
    """ASGI middleware that signals safety_queue when a client disconnects mid-request."""

    def __init__(self, app, executor: OnItA2AExecutor):
        self.app = app
        self.executor = executor

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Skip disconnect detection for file upload/download routes;
        # these are normal HTTP transfers, not client task cancellations.
        path = scope.get("path", "")
        if path.startswith("/uploads"):
            await self.app(scope, receive, send)
            return

        # Read the full request body upfront
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return  # client already gone
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

        # Provide buffered body to the inner app
        body_delivered = False
        async def buffered_receive():
            nonlocal body_delivered
            if not body_delivered:
                body_delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            # Block until cancelled (app shouldn't need receive again)
            await asyncio.Future()

        # Monitor the real receive for client disconnect
        async def disconnect_watcher():
            msg = await receive()
            if msg.get("type") == "http.disconnect":
                # Signal the safety_queue for the current request's task
                task_id = id(asyncio.current_task())
                sq = self.executor._active_safety_queues.get(task_id)
                if sq:
                    sq.put_nowait(STOP_TAG)

        watcher = asyncio.create_task(disconnect_watcher())
        try:
            await self.app(scope, buffered_receive, send)
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass


class OnIt(BaseModel):
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
    status: str = Field(default="idle")
    config_data: dict[str, Any] = Field(default_factory=dict)
    mcp_servers: list[Any] = Field(default_factory=list)
    tool_registry: Any | None = Field(default=None)
    theme: str | None = Field(default="white", exclude=True)
    messages: dict[str, str] = Field(default_factory=dict)
    stop_commands: list[str] = Field(default_factory=lambda: ['\\goodbye', '\\bye', '\\quit', '\\exit'])
    model_serving: dict[str, Any] = Field(default_factory=dict)
    user_id: str = Field(default="default_user")
    input_queue: asyncio.Queue | None = Field(default=None, exclude=True)
    output_queue: asyncio.Queue | None = Field(default=None, exclude=True)
    safety_queue: asyncio.Queue | None = Field(default=None, exclude=True)
    verbose: bool = Field(default=True)
    session_id: str | None = Field(default=None)
    session_path: str = Field(default="~/.onit/sessions")
    data_path: str = Field(default="")
    template_path: str | None = Field(default=None)
    documents_path: str | None = Field(default=None)
    topic: str | None = Field(default=None)
    prompt_intro: str | None = Field(default=None)
    timeout: int | None = Field(default=None)
    show_logs: bool = Field(default=False)
    stream: bool = Field(default=True)
    loop: bool = Field(default=False)
    period: float = Field(default=10.0)
    task: str | None = Field(default=None)
    web: bool = Field(default=False)
    web_port: int = Field(default=9000)
    web_google_client_id: str | None = Field(default=None)
    web_google_client_secret: str | None = Field(default=None)
    web_allowed_emails: list[str] | None = Field(default=None)
    web_title: str = Field(default="OnIt Chat")
    a2a: bool = Field(default=False)
    a2a_port: int = Field(default=9001)
    a2a_name: str = Field(default="OnIt")
    a2a_description: str = Field(default="An intelligent agent for task automation and assistance.")
    gateway: str | None = Field(default=None)
    gateway_token: str | None = Field(default=None, exclude=True)
    viber_webhook_url: str | None = Field(default=None)
    viber_port: int = Field(default=8443)
    sandbox: bool = Field(default=False)
    prompt_url: str | None = Field(default=None, exclude=True)
    file_server_url: str | None = Field(default=None, exclude=True)
    chat_ui: Any | None = Field(default=None, exclude=True)

    @property
    def sandbox_available(self) -> bool:
        """Check if sandbox mode is enabled."""
        return self.sandbox

    def __init__(self, config: Union[str, os.PathLike[str], dict[str, Any], None] = None) -> None :
        super().__init__()

        if config is not None:
            if isinstance(config, (str, os.PathLike)):
                cfg_path = Path(config).expanduser()
                if not cfg_path.exists():
                    raise FileNotFoundError(f"Config file {cfg_path} not found.")
                with cfg_path.open("r", encoding="utf-8") as f:
                    self.config_data = yaml.safe_load(f) or {}
            elif isinstance(config, dict):
                self.config_data = config
            else:
                raise TypeError("config must be a path-like object or dict.")

        self.initialize()
        if not self.loop:
            if self.web:
                from .ui.web import WebChatUI
                self.chat_ui = WebChatUI(
                    theme=self.theme,
                    data_path=self.data_path,
                    show_logs=self.show_logs,
                    server_port=self.web_port,
                    google_client_id=self.web_google_client_id,
                    google_client_secret=self.web_google_client_secret,
                    allowed_emails=self.web_allowed_emails,
                    session_path=self.session_path,
                    title=self.web_title,
                    verbose=self.verbose,
                )
                self.chat_ui._onit = self
            else:
                if self.a2a:
                    banner = "OnIt Agent to Agent Server"
                elif self.gateway:
                    banner = f"OnIt {self.gateway.capitalize()} Gateway"
                else:
                    banner = "OnIt Chat Interface"
                self.chat_ui = ChatUI(self.theme, show_logs=self.show_logs, banner_title=banner)
                # When resuming a session, pre-populate input history for arrow-key nav
                if self.config_data.get('resume_session_id'):
                    history = self.load_session_history(max_turns=100)
                    for entry in history:
                        task = entry.get("task", "").strip()
                        if task and (not self.chat_ui.input_history
                                     or self.chat_ui.input_history[-1] != task):
                            self.chat_ui.input_history.append(task)

    def initialize(self):
        self._setup_mcp_servers()
        self._setup_tool_registry()
        self._setup_model_serving()
        self._setup_session()
        self._setup_file_server_url()
        self._setup_config_fields()

    _DEFAULT_MCP_SERVERS = [
        {
            'name': 'PromptsMCPServer',
            'description': 'Provides prompt templates for instruction generation',
            'url': 'http://127.0.0.1:18200/sse',
            'enabled': True,
        },
        {
            'name': 'ToolsMCPServer',
            'description': 'Web search, bash commands, file operations, and document tools',
            'url': 'http://127.0.0.1:18201/sse',
            'enabled': True,
        },
    ]

    def _setup_mcp_servers(self) -> None:
        """Parse MCP server list from config and resolve the prompts server URL."""
        self.mcp_servers = self.config_data['mcp']['servers'] if 'mcp' in self.config_data and 'servers' in self.config_data['mcp'] else []
        # Ensure default servers are present if missing from config
        existing_names = {s.get('name') for s in self.mcp_servers}
        for default_server in self._DEFAULT_MCP_SERVERS:
            if default_server['name'] not in existing_names:
                self.mcp_servers.insert(0, dict(default_server))
        # Override MCP server URL hosts if mcp_host is configured
        mcp_host = self.config_data.get('mcp', {}).get('mcp_host')
        if mcp_host:
            from urllib.parse import urlparse, urlunparse
            for server in self.mcp_servers:
                url = server.get('url')
                if url:
                    parsed = urlparse(url)
                    server['url'] = urlunparse(parsed._replace(netloc=f"{mcp_host}:{parsed.port}" if parsed.port else mcp_host))
        # Find the prompts server URL from the MCP servers list
        for server in self.mcp_servers:
            if server.get('name') == 'PromptsMCPServer' and server.get('enabled', True):
                self.prompt_url = server.get('url')
                break
        if not self.prompt_url:
            raise ValueError(
                "PromptsMCPServer not found or disabled in MCP server config. "
                "Ensure it is listed under mcp.servers with a valid URL."
            )

    def _setup_tool_registry(self) -> None:
        """Discover tools from MCP servers (excluding the prompts server)."""
        tool_servers = [s for s in self.mcp_servers if s.get('name') != 'PromptsMCPServer']
        self.tool_registry = asyncio.run(discover_tools(tool_servers))
        # List discovered tools
        for tool_name in self.tool_registry:
            print(f"  - {tool_name}")
        print(f"  Total: {len(self.tool_registry)} tools discovered")

    def _setup_model_serving(self) -> None:
        """Configure model serving host and related settings."""
        self.theme = self.config_data.get('theme', 'white')
        self.messages = self.config_data.get('messages', {})
        self.stop_commands = list(self.config_data.get('stop_command', self.stop_commands))
        self.model_serving = self.config_data.get('serving', {})
        # resolve host: CLI/config > env var ONIT_HOST
        if 'host' not in self.model_serving or not self.model_serving['host']:
            env_host = os.environ.get('ONIT_HOST')
            if env_host:
                self.model_serving['host'] = env_host
            else:
                raise ValueError(
                    "No serving host configured. Set it via:\n"
                    "  - ONIT_HOST environment variable\n"
                    "  - --host CLI flag\n"
                    "  - serving.host in the config YAML"
                )
        self.user_id = self.config_data.get('user_id', 'default_user')
        self.status = "initialized"
        self.verbose = self.config_data.get('verbose', False)
        # Suppress noisy logs unless verbose
        if not self.verbose:
            logging.getLogger("src.lib.tools").setLevel(logging.WARNING)
            logging.getLogger("lib.tools").setLevel(logging.WARNING)
            logging.getLogger("type.tools").setLevel(logging.WARNING)
            logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
            logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    def _setup_session(self) -> None:
        """Create session ID, session file, and data directory.

        If ``config_data['resume_session_id']`` is set, resume that session
        instead of creating a new one.
        """
        from .sessions import register_session

        sessions_base = self.config_data.get('session_path', '~/.onit/sessions')
        sessions_base = os.path.expanduser(sessions_base)

        resume_id = self.config_data.get('resume_session_id')
        if resume_id:
            # Resume an existing session
            self.session_id = resume_id
            self.session_path = os.path.join(sessions_base, f"{self.session_id}.jsonl")
            if not os.path.exists(self.session_path):
                raise FileNotFoundError(
                    f"Session file not found: {self.session_path}\n"
                    f"Cannot resume session '{resume_id}'."
                )
        else:
            # Create a new session
            self.session_id = str(uuid.uuid4())
            self.session_path = os.path.join(sessions_base, f"{self.session_id}.jsonl")
            os.makedirs(sessions_base, exist_ok=True)
            if not os.path.exists(self.session_path):
                with open(self.session_path, "w", encoding="utf-8") as f:
                    f.write("")
            register_session(self.session_id, sessions_base)

        configured_data_path = self.config_data.get('data_path')
        if configured_data_path:
            self.data_path = str(Path(configured_data_path).expanduser().resolve())
        else:
            self.data_path = str(Path(tempfile.gettempdir()) / "onit" / "data" / self.session_id)
        os.makedirs(self.data_path, exist_ok=True)

    def _setup_file_server_url(self) -> None:
        """Compute file_server_url for file transfer via callback_url."""
        self.file_server_url = None
        mcp_host = self.config_data.get('mcp', {}).get('mcp_host')
        if mcp_host and self.config_data.get('web', False):
            import socket
            web_port = self.config_data.get('web_port', 9000)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect((mcp_host, 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = "127.0.0.1"
            self.file_server_url = f"http://{local_ip}:{web_port}"
        elif self.config_data.get('a2a', False):
            # In A2A mode, serve files through the A2A server itself
            import socket
            a2a_port = self.config_data.get('a2a_port', 9001)
            if mcp_host:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect((mcp_host, 80))
                    local_ip = s.getsockname()[0]
                    s.close()
                except Exception:
                    local_ip = "127.0.0.1"
            else:
                local_ip = "127.0.0.1"
            self.file_server_url = f"http://{local_ip}:{a2a_port}"

    def _setup_config_fields(self) -> None:
        """Assign remaining configuration fields from config_data."""
        self.template_path = self.config_data.get('template_path', None)
        self.documents_path = self.config_data.get('documents_path', None)
        self.topic = self.config_data.get('topic', None)
        self.prompt_intro = self.config_data.get('prompt_intro', None)
        self.timeout = self.config_data.get('timeout', None)  # default timeout 300 seconds
        if self.timeout is not None and self.timeout < 0:
            self.timeout = None  # no timeout
        self.show_logs = self.config_data.get('show_logs', False)
        self.stream = self.config_data.get('stream', True)
        self.loop = self.config_data.get('loop', False)
        self.period = float(self.config_data.get('period', 20.0))
        self.task = self.config_data.get('task', None)
        self.web = self.config_data.get('web', False)
        self.web_port = self.config_data.get('web_port', 9000)
        self.web_google_client_id = self.config_data.get('web_google_client_id', None)
        self.web_google_client_secret = self.config_data.get('web_google_client_secret', None)
        # Nullify placeholder credentials so auth is cleanly disabled
        for attr in ('web_google_client_id', 'web_google_client_secret'):
            val = getattr(self, attr, None)
            if val and "YOUR_" in str(val).upper():
                setattr(self, attr, None)
        self.web_allowed_emails = self.config_data.get('web_allowed_emails', None)
        self.web_title = self.config_data.get('web_title', 'OnIt Chat')
        self.a2a = self.config_data.get('a2a', False)
        self.a2a_port = self.config_data.get('a2a_port', 9001)
        self.a2a_name = self.config_data.get('a2a_name', 'OnIt')
        self.a2a_description = self.config_data.get('a2a_description', 'An intelligent agent for task automation and assistance.')
        self.gateway = self.config_data.get('gateway', None) or None
        self.gateway_token = self.config_data.get('gateway_token', None)
        self.viber_webhook_url = self.config_data.get('viber_webhook_url', None)
        self.viber_port = self.config_data.get('viber_port', 8443)
        self.sandbox = self.config_data.get('sandbox', False)
    def load_session_history(self, max_turns: int = 20, session_path: str | None = None) -> list[dict]:
        """Load recent session history from the JSONL session file.

        Args:
            max_turns: Maximum number of recent task/response pairs to return.
            session_path: Optional override path to the session file.

        Returns:
            A list of dicts with 'task' and 'response' keys, oldest first.
        """
        effective_path = session_path or self.session_path
        history = []
        try:
            if os.path.exists(effective_path):
                with open(effective_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                if "task" in entry and "response" in entry:
                                    history.append(entry)
                            except json.JSONDecodeError:
                                continue
        except Exception:
            pass
        # return only the most recent turns
        return history[-max_turns:]

    async def run(self) -> None:
        """Run the OnIt agent session"""
        try:
            self.input_queue = asyncio.Queue(maxsize=10)
            self.output_queue = asyncio.Queue(maxsize=10)
            self.safety_queue = asyncio.Queue(maxsize=10)
            # safety_queue is used by non-web modes; web uses per-session queues
            self.status = "running"
            if self.a2a:
                await self.run_a2a()
            elif self.loop:
                await self.run_loop()
            else:
                if self.web and hasattr(self.chat_ui, 'launch'):
                    self.chat_ui.launch(asyncio.get_event_loop())
                    # Web sessions call process_task() directly; keep loop alive
                    while self.status == "running":
                        await asyncio.sleep(1)
                else:
                    client_to_agent_task = asyncio.create_task(self.client_to_agent())
                    await asyncio.gather(client_to_agent_task)
        except Exception:
            pass
        finally:
            self.status = "stopped"

    async def process_task(self, task: str, images: list[str] | None = None,
                           session_path: str | None = None,
                           data_path: str | None = None,
                           safety_queue: asyncio.Queue | None = None,
                           stream_callback=None,
                           stream_complete_callback=None,
                           stream_throttle: int = 0,
                           stats: dict | None = None,
                           tool_status_callback=None,
                           session_id: str | None = None) -> str:
        """Process a single task and return the response string.

        Args:
            task: The user task/message to process.
            images: Optional list of image file paths.
            session_path: Optional override for session history file path.
            data_path: Optional override for data directory path.
            safety_queue: Optional per-session safety queue (e.g. per-tab in web UI).
            stream_callback: Optional callback ``(token, full_content) -> None``
                called for each streamed token so callers can deliver incremental
                updates to their clients (web UI, A2A, etc.).
            stream_complete_callback: Optional callback ``(content, tok_s) -> None``
                called when a streaming phase ends (before tool calls begin).
            stream_throttle: When > 0, only invoke ``stream_callback`` every N
                tokens to avoid flooding (useful for A2A SSE).
            tool_status_callback: Optional callback ``(status_text) -> None``
                called when a tool starts/stops to show activity indicators.
        """
        # Use per-chat overrides if provided, otherwise fall back to instance defaults
        effective_session_path = session_path or self.session_path
        effective_data_path = data_path or self.data_path
        effective_safety_queue = safety_queue or self.safety_queue

        while not effective_safety_queue.empty():
            effective_safety_queue.get_nowait()

        prompt_client = Client(self.prompt_url)
        async with prompt_client:
            instruction = await prompt_client.get_prompt("assistant", {
                "task": task,
                "data_path": effective_data_path,
                "template_path": self.template_path,
                "file_server_url": self.file_server_url,
                "documents_path": self.documents_path,
                "topic": self.topic,
                "sandbox_available": self.sandbox_available,
            })
            instruction = instruction.messages[0].content.text

        # Use a StreamingAdapter when streaming tokens or tracking tool status.
        _adapter = None
        if stream_callback and self.stream:
            _adapter = StreamingAdapter(
                on_token=stream_callback,
                on_complete=stream_complete_callback,
                show_logs=self.show_logs,
                throttle_tokens=stream_throttle,
                on_tool_status=tool_status_callback,
            )

        effective_session_id = session_id or self.session_id
        kwargs = {
            'console': None,
            'chat_ui': _adapter,
            'cursor': AGENT_CURSOR, 'memories': None,
            'verbose': self.verbose or self.show_logs,
            'data_path': effective_data_path,
            'session_id': effective_session_id,
            'max_tokens': self.model_serving.get('max_tokens', 8192),
            'max_context_tokens': self.model_serving.get('max_context_tokens', None),
            'session_history': self.load_session_history(session_path=effective_session_path),
            'stream': self.stream,
        }
        for _k in ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty'):
            if _k in self.model_serving:
                kwargs[_k] = self.model_serving[_k]
        if self.prompt_intro:
            kwargs['prompt_intro'] = self.prompt_intro
        MAX_PROCESS_RETRIES = 3
        last_response = None
        for _pt_attempt in range(1, MAX_PROCESS_RETRIES + 1):
            if not effective_safety_queue.empty():
                break
            last_response = await chat(
                host=self.model_serving["host"],
                host_key=self.model_serving.get("host_key", "EMPTY"),
                model=self.model_serving.get("model"),
                instruction=instruction,
                images=images,
                tool_registry=self.tool_registry,
                safety_queue=effective_safety_queue,
                think=self.model_serving.get("think", False),
                timeout=self.timeout,
                **kwargs,
            )
            _usable = last_response and remove_tags(last_response).strip()
            if _usable:
                break
            if _pt_attempt < MAX_PROCESS_RETRIES and effective_safety_queue.empty():
                kind = "Empty" if last_response is not None else "No"
                retry_msg = f"{kind} response from model, retrying ({_pt_attempt}/{MAX_PROCESS_RETRIES})..."
                logger.warning(retry_msg)
                if hasattr(self, 'chat_ui') and self.chat_ui and hasattr(self.chat_ui, 'add_log'):
                    self.chat_ui.add_log(retry_msg, level="warning")
                await asyncio.sleep(min(2 ** _pt_attempt, 10))

        # If the safety queue fired, stop the sandbox container.
        if not effective_safety_queue.empty():
            await _call_sandbox_stop(self.tool_registry, effective_session_id, sandbox=self.sandbox)

        # Flush any pending async streaming events (e.g. A2A) before
        # returning, so the client sees all partial updates before the
        # final completed message.
        if _adapter:
            await _adapter.flush()
            if stats is not None:
                stats["tokens_per_second"] = _adapter.tokens_per_second

        if not last_response or not remove_tags(last_response).strip():
            logger.error("chat() returned empty/None after %d retries. "
                         "Host: %s, Model: auto-detected",
                         MAX_PROCESS_RETRIES, self.model_serving["host"])
            return "I am sorry \U0001f614. Could you please rephrase your question?"

        response = remove_tags(last_response)
        try:
            with open(effective_session_path, "a", encoding="utf-8") as f:
                session_data = {
                    "task": task,
                    "response": response,
                    "timestamp": asyncio.get_event_loop().time(),
                }
                f.write(json.dumps(session_data) + "\n")
        except Exception:
            pass
        # Update session index (auto-tag on first message, track turns)
        try:
            from .sessions import update_session
            effective_sid = session_id or self.session_id
            sessions_dir = os.path.dirname(effective_session_path)
            update_session(effective_sid, task=task, sessions_dir=sessions_dir)
        except Exception:
            pass
        return response

    async def run_loop(self) -> None:
        """Run the OnIt agent in loop mode, executing a task repeatedly."""
        if not self.task:
            raise ValueError("Loop mode requires a 'task' to be set in the config.")

        print(f"Loop mode: task='{self.task}', period={self.period}s (Ctrl+C to stop)")
        prompt_client = Client(self.prompt_url)
        iteration = 0

        while True:
            try:
                iteration += 1
                start_time = asyncio.get_event_loop().time()

                # clear safety queue
                while not self.safety_queue.empty():
                    self.safety_queue.get_nowait()

                # build instruction via MCP prompt
                print(f"--- Iteration {iteration} ---")
                async with prompt_client:
                    instruction = await prompt_client.get_prompt("assistant", {"task": self.task,
                                                                                "data_path": self.data_path,
                                                                                "template_path": self.template_path,
                                                                                "file_server_url": self.file_server_url,
                                                                                "documents_path": self.documents_path,
                                                                                "topic": self.topic,
                                                                                "sandbox_available": self.sandbox_available})
                    instruction = instruction.messages[0]
                    instruction = instruction.content.text

                # call chat directly (no queues needed)
                kwargs = {'console': None,
                          'chat_ui': None,
                          'cursor': AGENT_CURSOR,
                          'memories': None,
                          'verbose': self.verbose,
                          'data_path': self.data_path,
                          'session_id': self.session_id,
                          'max_tokens': self.model_serving.get('max_tokens', 8192),
                          'max_context_tokens': self.model_serving.get('max_context_tokens', None),
                          'session_history': self.load_session_history()}
                for _k in ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty'):
                    if _k in self.model_serving:
                        kwargs[_k] = self.model_serving[_k]
                last_response = await chat(host=self.model_serving["host"],
                                            host_key=self.model_serving.get("host_key", "EMPTY"),
                                            model=self.model_serving.get("model"),
                                            instruction=instruction,
                                            tool_registry=self.tool_registry,
                                            safety_queue=self.safety_queue,
                                            think=self.model_serving.get("think", False),
                                            timeout=self.timeout,
                                            **kwargs)

                if last_response is not None:
                    elapsed_time = asyncio.get_event_loop().time() - start_time
                    response = remove_tags(last_response)
                    print(f"\n[{AGENT_CURSOR}] ({elapsed_time:.2f}s)\n{response}\n")

                    # save to session JSONL
                    try:
                        with open(self.session_path, "a", encoding="utf-8") as f:
                            session_data = {
                                "task": self.task,
                                "response": response,
                                "timestamp": asyncio.get_event_loop().time()
                            }
                            f.write(json.dumps(session_data) + "\n")
                    except Exception:
                        pass

                # countdown timer before next iteration
                remaining = int(self.period)
                while remaining > 0:
                    print(f"\rNext in {remaining}s (Ctrl+C to stop)  ", end="", flush=True)
                    await asyncio.sleep(1)
                    remaining -= 1
                # sleep any fractional remainder
                frac = self.period - int(self.period)
                if frac > 0:
                    await asyncio.sleep(frac)
                print("\r" + " " * 40 + "\r", end="", flush=True)

            except asyncio.CancelledError:
                return
            except KeyboardInterrupt:
                return
            except Exception:
                await asyncio.sleep(self.period)

    async def run_a2a(self) -> None:
        """Run OnIt as an A2A server, accepting tasks from other agents."""
        import uvicorn
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.tasks import InMemoryTaskStore
        from a2a.server.routes import create_jsonrpc_routes, create_agent_card_routes
        from a2a.types import AgentCard, AgentCapabilities, AgentSkill
        from starlette.applications import Starlette

        agent_card = AgentCard(
            name=self.a2a_name,
            description=self.a2a_description,
            url=f"http://0.0.0.0:{self.a2a_port}/",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=self.stream),
            skills=[AgentSkill(
                id="general",
                name="General Task",
                description="Process any task using OnIt's tools and LLM capabilities.",
                tags=["general", "automation"],
            )],
        )

        executor = OnItA2AExecutor(self)
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
            agent_card=agent_card,
        )
        routes = create_agent_card_routes(agent_card) + create_jsonrpc_routes(request_handler, rpc_url='/')
        starlette_app = Starlette(routes=routes)

        # Add file upload/download routes so MCP tools can send files
        # back through the A2A server instead of requiring a separate file server
        from starlette.requests import Request
        from starlette.responses import FileResponse, Response, JSONResponse
        from starlette.routing import Route

        def _find_session_data_path(session_id: str) -> str | None:
            """Look up per-context data_path by session_id."""
            for session in executor._sessions.values():
                if session["session_id"] == session_id:
                    return session["data_path"]
            return None

        async def serve_upload(request: Request) -> Response:
            session_id = request.path_params["session_id"]
            session_data_path = _find_session_data_path(session_id)
            if session_data_path is None:
                return Response(content="Session not found", status_code=404)
            filename = request.path_params["filename"]
            safe_name = os.path.basename(filename)
            filepath = os.path.join(session_data_path, safe_name)
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

        async def receive_upload(request: Request) -> Response:
            session_id = request.path_params["session_id"]
            session_data_path = _find_session_data_path(session_id)
            if session_data_path is None:
                return Response(content="Session not found", status_code=404)
            from starlette.formparsers import MultiPartParser
            os.makedirs(session_data_path, exist_ok=True)
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                return JSONResponse({"error": "No file provided"}, status_code=400)
            safe_name = os.path.basename(upload.filename)
            filepath = os.path.join(session_data_path, safe_name)
            content = await upload.read()
            with open(filepath, "wb") as f:
                f.write(content)
            await form.close()
            return JSONResponse({"filename": safe_name, "status": "ok"})

        starlette_app.routes.insert(0, Route("/uploads/{session_id}/{filename}", serve_upload, methods=["GET"]))
        starlette_app.routes.insert(0, Route("/uploads/{session_id}/", receive_upload, methods=["POST"]))

        # Wrap app with disconnect detection middleware
        wrapped_app = ClientDisconnectMiddleware(starlette_app, executor)

        print(f"A2A server running at http://0.0.0.0:{self.a2a_port}/ (Ctrl+C to stop)")

        _verbose_or_logs = self.verbose or self.show_logs
        config = uvicorn.Config(wrapped_app, host="0.0.0.0", port=self.a2a_port, log_level="info" if _verbose_or_logs else "warning", access_log=_verbose_or_logs)
        server = uvicorn.Server(config)
        await server.serve()

    def run_gateway_sync(self) -> None:
        """Run OnIt as a messaging gateway (blocking, owns the event loop).

        Supports Telegram and Viber gateways based on ``self.gateway`` value.
        """
        self.input_queue = asyncio.Queue(maxsize=10)
        self.output_queue = asyncio.Queue(maxsize=10)
        self.safety_queue = asyncio.Queue(maxsize=10)
        self.status = "running"

        if self.gateway == "viber":
            from .ui.viber import ViberGateway

            if not self.gateway_token:
                raise ValueError(
                    "Viber gateway requires a bot token. Set VIBER_BOT_TOKEN "
                    "environment variable or gateway_token in config."
                )
            if not self.viber_webhook_url:
                raise ValueError(
                    "Viber gateway requires a webhook URL. Set VIBER_WEBHOOK_URL "
                    "environment variable or --viber-webhook-url CLI option."
                )
            gw = ViberGateway(
                self, self.gateway_token,
                webhook_url=self.viber_webhook_url,
                port=self.viber_port,
                show_logs=self.show_logs,
            )
        else:
            from .ui.telegram import TelegramGateway

            if not self.gateway_token:
                raise ValueError(
                    "Telegram gateway requires a bot token. Set TELEGRAM_BOT_TOKEN "
                    "environment variable or gateway_token in config."
                )
            gw = TelegramGateway(self, self.gateway_token, show_logs=self.show_logs)

        gw.run_sync()

    async def _get_user_task(self, loop: asyncio.AbstractEventLoop) -> str:
        """Get user input from the appropriate UI (web or text)."""
        if self.web:
            return await self.chat_ui.get_user_input_async()
        return await loop.run_in_executor(None, self.chat_ui.get_user_input)

    async def _build_instruction(self, prompt_client: Client, task: str) -> str:
        """Build the agent instruction via the MCP prompts server."""
        async with prompt_client:
            instruction = await prompt_client.get_prompt("assistant", {"task": task,
                                                                        "data_path": self.data_path,
                                                                        "template_path": self.template_path,
                                                                        "file_server_url": self.file_server_url,
                                                                        "documents_path": self.documents_path,
                                                                        "topic": self.topic,
                                                                        "sandbox_available": self.sandbox_available})
            instruction = instruction.messages[0]
            instruction = instruction.content.text
        return instruction

    def _setup_enter_key_listener(self, loop: asyncio.AbstractEventLoop):
        """Set up Enter-key stop listener for text UI.

        Returns the callback so callers can pass it to
        ``_restore_enter_key_listener`` without storing it on the instance.
        Returns ``None`` in web mode (no listener needed).
        """
        if self.web:
            return None
        import sys
        safety_warning = self.messages.get('safety_warning', "Press 'Enter' key to stop all tasks.")
        self.chat_ui.console.print(safety_warning, style="dim")
        self.chat_ui.start_thinking()
        def _on_enter():
            sys.stdin.readline()
            self.safety_queue.put_nowait(STOP_TAG)
        try:
            loop.add_reader(sys.stdin.fileno(), _on_enter)
        except NotImplementedError:
            pass  # Windows ProactorEventLoop does not support add_reader
        return _on_enter

    def _cleanup_enter_key_listener(self, loop: asyncio.AbstractEventLoop) -> None:
        """Clean up Enter-key listener for text UI."""
        if self.web:
            return
        import sys
        try:
            loop.remove_reader(sys.stdin.fileno())
        except Exception:
            pass

    def _restore_enter_key_listener(self, loop: asyncio.AbstractEventLoop,
                                    callback) -> None:
        """Re-attach Enter-key listener after removing it (e.g. for retry prompt)."""
        if self.web or callback is None:
            return
        import sys
        try:
            loop.add_reader(sys.stdin.fileno(), callback)
        except NotImplementedError:
            pass  # Windows ProactorEventLoop does not support add_reader

    def _handle_successful_response(self, response: str, task: str,
                                    elapsed_time: str,
                                    loop: asyncio.AbstractEventLoop) -> None:
        """Process a successful agent response: display it and save to session."""
        response = remove_tags(response).strip()
        # Skip empty responses — model returned nothing useful.
        if not response:
            # Surface any error from logs so user knows what went wrong
            error_detail = ""
            if hasattr(self.chat_ui, 'execution_logs') and self.chat_ui.execution_logs:
                last_log = self.chat_ui.execution_logs[-1]
                if last_log.get("level") in ("error", "warning"):
                    error_detail = f": {last_log['message']}"
            self.chat_ui.add_message(
                "system",
                f"The model returned an empty response{error_detail}. Try rephrasing or providing more detail.",
                elapsed=elapsed_time,
            )
            return
        # If streaming already persisted the message via stream_end(),
        # just update the elapsed time instead of adding a duplicate.
        _last = self.chat_ui.messages[-1] if self.chat_ui.messages else None
        if _last and hasattr(_last, 'role') and _last.role == "assistant" and _last.content == response:
            from src.ui.text import Message
            self.chat_ui.messages[-1] = Message(
                role=_last.role, content=_last.content,
                timestamp=_last.timestamp, elapsed=elapsed_time,
                name=getattr(_last, 'name', ''),
            )
        else:
            self.chat_ui.add_message("assistant", response, elapsed=elapsed_time)
        # Show local codebase path when sandbox generated code files
        if has_code_files(self.data_path):
            self.chat_ui.data_path = self.data_path
        try:
            with open(self.session_path, "a", encoding="utf-8") as f:
                session_data = {
                    "task": task,
                    "response": response,
                    "timestamp": loop.time()
                }
                f.write(json.dumps(session_data) + "\n")
        except Exception:
            pass

    def _format_elapsed_time(self, elapsed_secs: float) -> str:
        """Format elapsed time string, including tokens/sec if available."""
        _tok_s = ""
        if hasattr(self.chat_ui, '_stream_token_count') and hasattr(self.chat_ui, '_stream_start_time'):
            _st = self.chat_ui._stream_start_time
            _tc = self.chat_ui._stream_token_count
            if _st > 0 and _tc > 0:
                _tok_s = f" ({_tc / (time.monotonic() - _st):.1f} tok/s)"
        return f"{elapsed_secs:.2f} secs{_tok_s}"

    async def client_to_agent(self) -> None:
        """Handle client to agent communication"""

        prompt_client = Client(self.prompt_url)
        agent_task = None
        loop = asyncio.get_event_loop()

        while True:
            task = await self._get_user_task(loop)

            if task.lower().strip() in self.stop_commands:
                if not self.web:
                    self.chat_ui.console.print("Exiting chat session...", style="warning")
                if agent_task and not agent_task.done():
                    agent_task.cancel()
                break
            if not task or len(task) == 0:
                task = None
                continue

            # clear all queues
            while not self.input_queue.empty():
                self.input_queue.get_nowait()
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
            while not self.safety_queue.empty():
                self.safety_queue.get_nowait()

            instruction = await self._build_instruction(prompt_client, task)

            on_enter_cb = self._setup_enter_key_listener(loop)

            # submit instruction with retry on API error
            start_time = loop.time()
            while True:
                while not self.safety_queue.empty():
                    self.safety_queue.get_nowait()

                agent_task = asyncio.create_task(self.agent_session())
                await self.input_queue.put(instruction)

                final_answer_task = asyncio.create_task(self.output_queue.get())
                done, pending = await asyncio.wait([final_answer_task],
                                                   return_when=asyncio.FIRST_COMPLETED)

                for t in pending:
                    t.cancel()

                if final_answer_task not in done:
                    await self.safety_queue.put(STOP_TAG)
                    while not agent_task.done():
                        await asyncio.sleep(0.1)
                    break

                response = final_answer_task.result()

                # User-initiated stop
                if response == STOP_TAG:
                    await _call_sandbox_stop(self.tool_registry, self.session_id, sandbox=self.sandbox)
                    self.chat_ui.add_message("system", "Task stopped by user.")
                    break

                if response is None:
                    # API error — auto-retry with backoff before giving up
                    error_detail = ""
                    if hasattr(self.chat_ui, 'execution_logs') and self.chat_ui.execution_logs:
                        last_log = self.chat_ui.execution_logs[-1]
                        if last_log.get("level") in ("error", "warning"):
                            error_detail = f" ({last_log['message']})"

                    retry_delays = [30, 60, 120]
                    retry_succeeded = False
                    for attempt, delay in enumerate(retry_delays, 1):
                        self.chat_ui.add_message("system", f"Unable to get a response from the model{error_detail}. Retrying in {delay}s (attempt {attempt}/{len(retry_delays)})...")
                        await asyncio.sleep(delay)

                        # Clear queues before retry
                        while not self.safety_queue.empty():
                            self.safety_queue.get_nowait()

                        agent_task = asyncio.create_task(self.agent_session())
                        await self.input_queue.put(instruction)

                        final_answer_task = asyncio.create_task(self.output_queue.get())
                        done, pending = await asyncio.wait([final_answer_task],
                                                           return_when=asyncio.FIRST_COMPLETED)
                        for t in pending:
                            t.cancel()

                        if final_answer_task in done:
                            response = final_answer_task.result()
                            if response is not None and response != STOP_TAG:
                                retry_succeeded = True
                                break
                            if response == STOP_TAG:
                                await _call_sandbox_stop(self.tool_registry, self.session_id, sandbox=self.sandbox)
                                self.chat_ui.add_message("system", "Task stopped by user.")
                                break

                    if retry_succeeded:
                        # success on retry
                        elapsed_time = self._format_elapsed_time(loop.time() - start_time)
                        self._handle_successful_response(response, task, elapsed_time, loop)
                        break

                    if response == STOP_TAG:
                        break

                    # All retries exhausted — give up
                    self._cleanup_enter_key_listener(loop)
                    self.chat_ui.add_message("system", f"Unable to get a response from the model{error_detail} after {len(retry_delays)} retries. Giving up.")
                    break

                # success
                elapsed_time = self._format_elapsed_time(loop.time() - start_time)
                self._handle_successful_response(response, task, elapsed_time, loop)
                break

            self._cleanup_enter_key_listener(loop)
            
    async def agent_session(self) -> None:
        """Start the agent session with automatic retry on transient failures."""
        MAX_AGENT_RETRIES = 3
        while True:
            try:
                instruction = await self.input_queue.get()
                if not self.safety_queue.empty():
                    await _call_sandbox_stop(self.tool_registry, self.session_id, sandbox=self.sandbox)
                    await self.output_queue.put(STOP_TAG)
                    break
                kwargs = {'console': self.chat_ui.console,
                          'chat_ui': self.chat_ui,
                          'cursor': AGENT_CURSOR,
                          'memories': None,
                          'verbose': self.verbose,
                          'data_path': self.data_path,
                          'session_id': self.session_id,
                          'max_tokens': self.model_serving.get('max_tokens', 8192),
                          'max_context_tokens': self.model_serving.get('max_context_tokens', None),
                          'session_history': self.load_session_history(),
                          'stream': self.stream}
                for _k in ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty'):
                    if _k in self.model_serving:
                        kwargs[_k] = self.model_serving[_k]
                if self.prompt_intro:
                    kwargs['prompt_intro'] = self.prompt_intro

                last_response = None
                for attempt in range(1, MAX_AGENT_RETRIES + 1):
                    if not self.safety_queue.empty():
                        break
                    last_response = await chat(
                        host=self.model_serving["host"],
                        host_key=self.model_serving.get("host_key", "EMPTY"),
                        model=self.model_serving.get("model"),
                        instruction=instruction,
                        tool_registry=self.tool_registry,
                        safety_queue=self.safety_queue,
                        think=self.model_serving.get("think", False),
                        timeout=self.timeout,
                        **kwargs)
                    # Treat empty/whitespace-only responses as failures too
                    _usable = last_response and remove_tags(last_response).strip()
                    if _usable:
                        break
                    if attempt < MAX_AGENT_RETRIES and self.safety_queue.empty():
                        kind = "Empty" if last_response is not None else "No"
                        retry_msg = f"{kind} response from model, retrying ({attempt}/{MAX_AGENT_RETRIES})..."
                        logger.warning(retry_msg)
                        if self.chat_ui and hasattr(self.chat_ui, 'add_log'):
                            self.chat_ui.add_log(retry_msg, level="warning")
                        await asyncio.sleep(min(2 ** attempt, 10))

                # Normalize: if final response is empty/whitespace, treat as None
                if not last_response or not remove_tags(last_response).strip():
                    last_response = None

                if last_response is None and self.safety_queue.empty():
                    await self.output_queue.put(None)
                    return
                if not self.safety_queue.empty():
                    await _call_sandbox_stop(self.tool_registry, self.session_id, sandbox=self.sandbox)
                    await self.output_queue.put(STOP_TAG)
                    break
                await self.output_queue.put(f"<answer>{last_response}</answer>")
                return
            except asyncio.CancelledError:
                logger.warning("Agent session cancelled.")
                await _call_sandbox_stop(self.tool_registry, self.session_id, sandbox=self.sandbox)
                await self.output_queue.put(None)
                return
            except Exception as e:
                logger.error("Error in agent session: %s", e)
                if self.chat_ui and hasattr(self.chat_ui, 'add_log'):
                    self.chat_ui.add_log(f"Agent error: {e}", level="error")
                await self.output_queue.put(None)
                return