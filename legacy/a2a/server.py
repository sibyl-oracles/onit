"""Legacy A2A protocol server.

Moved verbatim from src/onit.py when A2A support moved to legacy/
(see ../README.md). The executor and the disconnect middleware are
unchanged; ``run_a2a_server`` is ``OnIt.run_a2a`` with the instance
passed in instead of bound.
"""

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent
from a2a.helpers.proto_helpers import new_text_message

from src.lib.files import zip_code_files

logger = logging.getLogger(__name__)

STOP_TAG = "<stop></stop>"


async def _call_sandbox_stop(tool_registry, session_id: str = "") -> None:
    """Stop what this session had running: its interpreter, and the sandbox.

    Kept in sync with src/onit.py, which owns the active copy.
    """
    try:
        from src.model.serving.interpreter import shutdown_session
        await shutdown_session(session_id)
    except Exception as e:  # a stop that fails must not fail the stop
        logger.debug("interpreter not shut down for %s: %s", session_id, e)
    if not tool_registry or "sandbox_stop" not in tool_registry.tools:
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
                base_path = str(Path(configured_data_path).expanduser().resolve())
            else:
                base_path = str(Path.home() / "sandbox")
            data_path = os.path.join(base_path, session_id)
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
        await _call_sandbox_stop(self.onit.tool_registry, session["session_id"])


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


async def run_a2a_server(onit) -> None:
    """Run OnIt as an A2A server, accepting tasks from other agents."""
    import uvicorn
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.server.routes import create_jsonrpc_routes, create_agent_card_routes
    from a2a.types import AgentCard, AgentCapabilities, AgentSkill
    from starlette.applications import Starlette

    agent_card = AgentCard(
        name=onit.a2a_name,
        description=onit.a2a_description,
        url=f"http://0.0.0.0:{onit.a2a_port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=onit.stream),
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

    print(f"A2A server running at http://0.0.0.0:{onit.a2a_port}/ (Ctrl+C to stop)")

    _verbose_or_logs = onit.verbose or onit.show_logs
    config = uvicorn.Config(wrapped_app, host="0.0.0.0", port=onit.a2a_port, log_level="info" if _verbose_or_logs else "warning", access_log=_verbose_or_logs)
    server = uvicorn.Server(config)
    await server.serve()
