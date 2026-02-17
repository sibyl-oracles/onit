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
import os
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

# Suppress noisy HTTP request logs from httpx/httpcore (used by FastMCP client)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from .lib.tools import discover_tools
from .lib.text import remove_tags
from .ui import ChatUI
from .model.serving.chat import chat

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import FilePart, FileWithBytes
from a2a.utils import new_agent_text_message

AGENT_CURSOR = "OnIt"
USER_CURSOR = "You"
STOP_TAG = "<stop></stop>"


class OnItA2AExecutor(AgentExecutor):
    """A2A executor that delegates task processing to an OnIt instance."""

    def __init__(self, onit):
        self.onit = onit

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.get_user_input()
        if not context.message:
            raise Exception('No message provided')

        # Extract inline image parts from the A2A message and save to temp folder
        import base64
        image_paths = []
        for part in context.message.parts:
            if isinstance(part.root, FilePart) and isinstance(part.root.file, FileWithBytes):
                file_obj = part.root.file
                if file_obj.mime_type and file_obj.mime_type.startswith('image/'):
                    safe_name = os.path.basename(file_obj.name or 'image.png')
                    filepath = os.path.join(self.onit.data_path, safe_name)
                    with open(filepath, 'wb') as f:
                        f.write(base64.b64decode(file_obj.bytes))
                    image_paths.append(filepath)

        try:
            result = await self.onit.process_task(
                task, images=image_paths if image_paths else None
            )
        except asyncio.CancelledError:
            self.onit.safety_queue.put_nowait(STOP_TAG)
            raise
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        self.onit.safety_queue.put_nowait(STOP_TAG)


class ClientDisconnectMiddleware:
    """ASGI middleware that signals safety_queue when a client disconnects mid-request."""

    def __init__(self, app, onit):
        self.app = app
        self.onit = onit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
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
                self.onit.safety_queue.put_nowait(STOP_TAG)

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
    persona: str = Field(default="assistant")
    user_id: str = Field(default="default_user")
    input_queue: asyncio.Queue | None = Field(default=None, exclude=True)
    output_queue: asyncio.Queue | None = Field(default=None, exclude=True)
    safety_queue: asyncio.Queue | None = Field(default=None, exclude=True)
    verbose: bool = Field(default=True)
    session_id: str | None = Field(default=None)
    session_path: str = Field(default="~/.onit/sessions")
    data_path: str = Field(default="")
    template_path: str | None = Field(default=None)
    timeout: int | None = Field(default=None)
    show_logs: bool = Field(default=False)
    loop: bool = Field(default=False)
    period: float = Field(default=10.0)
    task: str | None = Field(default=None)
    web: bool = Field(default=False)
    web_port: int = Field(default=9000)
    web_share: bool = Field(default=False)
    web_google_client_id: str | None = Field(default=None)
    web_google_client_secret: str | None = Field(default=None)
    web_allowed_emails: list[str] | None = Field(default=None)
    web_title: str = Field(default="OnIt Chat")
    a2a: bool = Field(default=False)
    a2a_port: int = Field(default=9001)
    a2a_name: str = Field(default="OnIt")
    a2a_description: str = Field(default="An intelligent agent for task automation and assistance.")
    prompt_url: str | None = Field(default=None, exclude=True)
    file_server_url: str | None = Field(default=None, exclude=True)
    chat_ui: Any | None = Field(default=None, exclude=True)

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
                    share=self.web_share,
                    google_client_id=self.web_google_client_id,
                    google_client_secret=self.web_google_client_secret,
                    allowed_emails=self.web_allowed_emails,
                    session_path=self.session_path,
                    title=self.web_title,
                )
            else:
                banner = "OnIt Agent to Agent Server" if self.a2a else "OnIt Chat Interface"
                self.chat_ui = ChatUI(self.theme, show_logs=self.show_logs, banner_title=banner)
        
    def initialize(self):
        self.mcp_servers = self.config_data['mcp']['servers'] if 'mcp' in self.config_data and 'servers' in self.config_data['mcp'] else []
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
        self.tool_registry = asyncio.run(discover_tools(self.mcp_servers))
        # List discovered tools
        for tool_name in self.tool_registry:
            print(f"  - {tool_name}")
        print(f"  Total: {len(self.tool_registry)} tools discovered")
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
        self.persona = self.config_data.get('persona', 'assistant')
        self.user_id = self.config_data.get('user_id', 'default_user')
        self.status = "initialized"
        self.verbose = self.config_data.get('verbose', False)
        # Suppress tool discovery and MCP client logs unless verbose
        if not self.verbose:
            logging.getLogger("src.lib.tools").setLevel(logging.WARNING)
            logging.getLogger("lib.tools").setLevel(logging.WARNING)
            logging.getLogger("type.tools").setLevel(logging.WARNING)
        # append session id to sessions path
        self.session_id = str(uuid.uuid4())
        self.session_path = os.path.join(self.config_data.get('session_path', '~/.onit/sessions'), f"{self.session_id}.jsonl")
        # create the sessions directory and file if not exists. expand ~ to home directory
        self.session_path = os.path.expanduser(self.session_path)
        sessions_dir = os.path.dirname(self.session_path)
        os.makedirs(sessions_dir, exist_ok=True)
        if not os.path.exists(self.session_path):
            with open(self.session_path, "w", encoding="utf-8") as f:
                f.write("")
        self.data_path = str(Path(tempfile.gettempdir()) / "onit" / "data" / self.session_id)
        os.makedirs(self.data_path, exist_ok=True)
        # Compute file_server_url for file transfer via callback_url
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
        self.template_path = self.config_data.get('template_path', None)
        self.timeout = self.config_data.get('timeout', None)  # default timeout 300 seconds
        if self.timeout is not None and self.timeout < 0:
            self.timeout = None  # no timeout
        self.show_logs = self.config_data.get('show_logs', False)
        self.loop = self.config_data.get('loop', False)
        self.period = float(self.config_data.get('period', 20.0))
        self.task = self.config_data.get('task', None)
        self.web = self.config_data.get('web', False)
        self.web_port = self.config_data.get('web_port', 9000)
        self.web_share = self.config_data.get('web_share', False)
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
    def load_session_history(self, max_turns: int = 20) -> list[dict]:
        """Load recent session history from the JSONL session file.

        Args:
            max_turns: Maximum number of recent task/response pairs to return.

        Returns:
            A list of dicts with 'task' and 'response' keys, oldest first.
        """
        history = []
        try:
            if os.path.exists(self.session_path):
                with open(self.session_path, "r", encoding="utf-8") as f:
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
            if self.web and self.chat_ui:
                self.chat_ui.safety_queue = self.safety_queue
            self.status = "running"
            if self.a2a:
                await self.run_a2a()
            elif self.loop:
                await self.run_loop()
            else:
                if self.web and hasattr(self.chat_ui, 'launch'):
                    self.chat_ui.launch(asyncio.get_event_loop())
                client_to_agent_task = asyncio.create_task(self.client_to_agent())
                await asyncio.gather(client_to_agent_task)
        except Exception:
            pass
        finally:
            self.status = "stopped"

    async def process_task(self, task: str, images: list[str] | None = None) -> str:
        """Process a single task and return the response string."""
        while not self.safety_queue.empty():
            self.safety_queue.get_nowait()

        prompt_client = Client(self.prompt_url)
        async with prompt_client:
            instruction = await prompt_client.get_prompt(self.persona, {
                "task": task,
                "session_id": self.session_id,
                "template_path": self.template_path,
                "file_server_url": self.file_server_url,
            })
            instruction = instruction.messages[0].content.text

        kwargs = {
            'console': None, 'chat_ui': None,
            'cursor': AGENT_CURSOR, 'memories': None,
            'verbose': self.verbose,
            'max_tokens': self.model_serving.get('max_tokens', 262144),
            'session_history': self.load_session_history(),
        }
        last_response = await chat(
            host=self.model_serving["host"],
            host_key=self.model_serving.get("host_key", "EMPTY"),
            model=self.model_serving["model"],
            instruction=instruction,
            images=images,
            tool_registry=self.tool_registry,
            safety_queue=self.safety_queue,
            think=self.model_serving["think"],
            timeout=self.timeout,
            **kwargs,
        )

        if last_response is not None:
            response = remove_tags(last_response)
            try:
                with open(self.session_path, "a", encoding="utf-8") as f:
                    session_data = {
                        "task": task,
                        "response": response,
                        "timestamp": asyncio.get_event_loop().time(),
                    }
                    f.write(json.dumps(session_data) + "\n")
            except Exception:
                pass
            return response
        return "Error: No response from model."

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
                    instruction = await prompt_client.get_prompt(self.persona, {"task": self.task,
                                                                                "session_id": self.session_id,
                                                                                "template_path": self.template_path,
                                                                                "file_server_url": self.file_server_url})
                    instruction = instruction.messages[0]
                    instruction = instruction.content.text

                # call chat directly (no queues needed)
                kwargs = {'console': None,
                          'chat_ui': None,
                          'cursor': AGENT_CURSOR,
                          'memories': None,
                          'verbose': self.verbose,
                          'max_tokens': self.model_serving.get('max_tokens', 262144),
                          'session_history': self.load_session_history()}
                last_response = await chat(host=self.model_serving["host"],
                                            host_key=self.model_serving.get("host_key", "EMPTY"),
                                            model=self.model_serving["model"],
                                            instruction=instruction,
                                            tool_registry=self.tool_registry,
                                            safety_queue=self.safety_queue,
                                            think=self.model_serving["think"],
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
        from a2a.server.apps import A2AStarletteApplication
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.tasks import InMemoryTaskStore
        from a2a.server.events import InMemoryQueueManager
        from a2a.types import AgentCard, AgentCapabilities, AgentSkill

        agent_card = AgentCard(
            name=self.a2a_name,
            description=self.a2a_description,
            url=f"http://0.0.0.0:{self.a2a_port}/",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=False),
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
            queue_manager=InMemoryQueueManager(),
        )
        a2a_app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
        starlette_app = a2a_app.build()

        # Add file upload/download routes so MCP tools can send files
        # back through the A2A server instead of requiring a separate file server
        from starlette.requests import Request
        from starlette.responses import FileResponse, Response, JSONResponse
        from starlette.routing import Route

        data_path = self.data_path

        async def serve_upload(request: Request) -> Response:
            filename = request.path_params["filename"]
            safe_name = os.path.basename(filename)
            filepath = os.path.join(data_path, safe_name)
            if os.path.isfile(filepath):
                return FileResponse(filepath)
            return Response(content="File not found", status_code=404)

        async def receive_upload(request: Request) -> Response:
            from starlette.formparsers import MultiPartParser
            os.makedirs(data_path, exist_ok=True)
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                return JSONResponse({"error": "No file provided"}, status_code=400)
            safe_name = os.path.basename(upload.filename)
            filepath = os.path.join(data_path, safe_name)
            content = await upload.read()
            with open(filepath, "wb") as f:
                f.write(content)
            await form.close()
            return JSONResponse({"filename": safe_name, "status": "ok"})

        starlette_app.routes.insert(0, Route("/uploads/{filename}", serve_upload, methods=["GET"]))
        starlette_app.routes.insert(0, Route("/uploads/", receive_upload, methods=["POST"]))

        # Wrap app with disconnect detection middleware
        wrapped_app = ClientDisconnectMiddleware(starlette_app, self)

        print(f"A2A server running at http://0.0.0.0:{self.a2a_port}/ (Ctrl+C to stop)")

        config = uvicorn.Config(wrapped_app, host="0.0.0.0", port=self.a2a_port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    async def client_to_agent(self) -> None:
        """Handle client to agent communication"""

        prompt_client = Client(self.prompt_url)
        agent_task = None
        loop = asyncio.get_event_loop()
        safety_warning = self.messages.get('safety_warning', "Press 'Enter' key to stop all tasks.")

        while True:
            if self.web:
                task = await self.chat_ui.get_user_input_async()
            else:
                task = await loop.run_in_executor(None, self.chat_ui.get_user_input)

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

            # prompt engineering
            async with prompt_client:
                instruction = await prompt_client.get_prompt(self.persona, {"task": task,
                                                                            "session_id": self.session_id,
                                                                            "template_path": self.template_path,
                                                                            "file_server_url": self.file_server_url})
                instruction = instruction.messages[0]
                instruction = instruction.content.text
                
            # Set up Enter-key stop listener for text UI
            if not self.web:
                import sys
                self.chat_ui.console.print(safety_warning, style="dim")
                def _on_enter():
                    sys.stdin.readline()
                    self.safety_queue.put_nowait(STOP_TAG)
                loop.add_reader(sys.stdin.fileno(), _on_enter)

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
                    self.chat_ui.add_message("system", "Task stopped by user.")
                    break

                if response is None:
                    # API error — ask user whether to retry
                    if not self.web:
                        loop.remove_reader(sys.stdin.fileno())
                    self.chat_ui.add_message("system", "Unable to get a response from the model. Would you like to retry? (yes/no)")
                    if self.web:
                        retry_input = await self.chat_ui.get_user_input_async()
                    else:
                        retry_input = await loop.run_in_executor(
                            None, self.chat_ui.get_user_input)
                    if retry_input.lower().strip() in ('yes', 'y'):
                        if not self.web:
                            loop.add_reader(sys.stdin.fileno(), _on_enter)
                        continue
                    break

                # success
                elapsed_time = loop.time() - start_time
                elapsed_time = f"{elapsed_time:.2f} secs"
                response = remove_tags(response)
                self.chat_ui.add_message("assistant", response, elapsed=elapsed_time)
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
                break

            # Clean up Enter-key listener for text UI
            if not self.web:
                import sys
                try:
                    loop.remove_reader(sys.stdin.fileno())
                except Exception:
                    pass
            
    async def agent_session(self) -> None:
        """Start the agent session"""
        while True:
            try:
                instruction = await self.input_queue.get()
                if not self.safety_queue.empty():
                    await self.output_queue.put(STOP_TAG)
                    break
                kwargs = {'console': self.chat_ui.console,
                          'chat_ui': self.chat_ui,
                          'cursor': AGENT_CURSOR,
                          'memories': None,
                          'verbose': self.verbose,
                          'max_tokens': self.model_serving.get('max_tokens', 262144),
                          'session_history': self.load_session_history()}
                last_response = await chat(host=self.model_serving["host"],
                                            host_key=self.model_serving.get("host_key", "EMPTY"),
                                            model=self.model_serving["model"],
                                            instruction=instruction,
                                            tool_registry=self.tool_registry,
                                            safety_queue=self.safety_queue,
                                            think=self.model_serving["think"],
                                            timeout=self.timeout,
                                            **kwargs)
                if last_response is None and self.safety_queue.empty():
                    await self.output_queue.put(None)
                    return
                if not self.safety_queue.empty():
                    await self.output_queue.put(STOP_TAG)
                    break
                await self.output_queue.put(f"<answer>{last_response}</answer>")
                return
            except asyncio.CancelledError:
                await self.output_queue.put(f"Error: Session cancelled.")
                return
            except Exception as e:
                await self.output_queue.put(f"Error in agent session:\nInstruction:\n{instruction}\nError:\n{str(e)}")
                return