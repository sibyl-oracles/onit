"""Inspect AI model provider that drives the real OnIt agent.

Registers an ``onit`` provider so benchmarks can use ``model="onit/<label>"``.
Each :meth:`OnItAPI.generate` call runs one task end-to-end through
:meth:`OnIt.process_task`, exercising OnIt's real prompt engineering, MCP tool
registry, and tool loop — not just the underlying LLM.

Integration notes:
    * One ``OnIt`` instance is built lazily and shared across samples. Inspect
      runs samples concurrently, so per-call state (safety queue, session and
      data directories) is created per ``generate`` call for isolation.
    * ``OnIt.__init__`` discovers tools via ``asyncio.run(...)``, which cannot
      run inside Inspect's event loop. We therefore build the agent in a worker
      thread (no running loop there) the first time it is needed.
    * The eval target (host/model) comes from the environment via
      :func:`benchmarks.config.resolve_serving`.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from . import config as bench_config

# Shared OnIt instance, built once on first use.
_agent: Any | None = None
_agent_lock: asyncio.Lock | None = None


def base_config_data() -> dict[str, Any]:
    """Build the headless OnIt config used to drive benchmarks.

    ``loop=True`` skips terminal-UI setup; streaming is off. The eval target
    (host/model) and per-request timeout come from the environment.
    """
    sessions_dir = Path(tempfile.gettempdir()) / "onit-bench-sessions"
    data_dir = Path(tempfile.gettempdir()) / "onit-bench-data"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return {
        "serving": bench_config.resolve_serving(),
        # Headless: no streaming, no terminal ChatUI (loop=True skips UI setup).
        "stream": False,
        "loop": True,
        "verbose": False,
        "show_logs": False,
        # Bounded per-request timeout so a stalled endpoint fails the sample
        # instead of hanging the whole run (override via ONIT_BENCH_TIMEOUT).
        "timeout": bench_config.bench_timeout(),
        "session_path": str(sessions_dir),
        "data_path": str(data_dir),
        "mcp": {
            "servers": [
                {"name": "PromptsMCPServer", "url": "http://127.0.0.1:18200/sse",
                 "enabled": True},
                {"name": "ToolsMCPServer", "url": "http://127.0.0.1:18201/sse",
                 "enabled": True},
            ]
        },
    }


def _build_agent_blocking(config_overrides: dict[str, Any] | None = None) -> Any:
    """Construct an OnIt agent. Must run in a thread with no running loop.

    Starts the MCP servers (idempotent — skips ports already bound) then builds
    the agent, which discovers tools against those servers. ``config_overrides``
    are shallow-merged onto :func:`base_config_data` (e.g. ``{"sandbox": True}``
    to delegate code execution to OnIt's MCP sandbox provider).
    """
    # Imported lazily so importing this module never pulls in the whole agent
    # stack (keeps the provider unit-testable with a stub).
    from src.cli import _ensure_mcp_servers
    from src.onit import OnIt

    config_data = base_config_data()
    if config_overrides:
        config_data.update(config_overrides)

    _ensure_mcp_servers(config_data)
    return OnIt(config_data)


async def _get_agent() -> Any:
    """Return the shared OnIt agent, building it once off the event loop."""
    global _agent, _agent_lock
    if _agent is not None:
        return _agent
    if _agent_lock is None:
        _agent_lock = asyncio.Lock()
    async with _agent_lock:
        if _agent is None:
            loop = asyncio.get_running_loop()
            _agent = await loop.run_in_executor(None, _build_agent_blocking)
    return _agent


def _messages_to_task(messages: list[ChatMessage]) -> str:
    """Flatten Inspect chat messages into a single task string for OnIt.

    Benchmarks express each item as a system prompt plus a user prompt; OnIt's
    own prompt engineering wraps this into its assistant instruction. Tool and
    prior-assistant turns are included so multi-turn tasks keep their context.
    """
    parts: list[str] = []
    for msg in messages:
        text = (msg.text or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


@modelapi(name="onit")
class OnItAPI(ModelAPI):
    """Inspect model provider backed by ``OnIt.process_task``."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, api_key_vars or [], config)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        agent = await _get_agent()
        task = _messages_to_task(input)

        # Per-sample isolation: unique session + data directory.
        run_id = uuid.uuid4().hex[:12]
        sessions_dir = Path(tempfile.gettempdir()) / "onit-bench-sessions"
        data_dir = Path(tempfile.gettempdir()) / "onit-bench-data" / run_id
        data_dir.mkdir(parents=True, exist_ok=True)

        stats: dict[str, Any] = {}
        answer = await agent.process_task(
            task,
            session_path=str(sessions_dir / f"{run_id}.jsonl"),
            data_path=str(data_dir),
            safety_queue=asyncio.Queue(),
            stats=stats,
        )

        return ModelOutput.from_content(model=self.model_name, content=answer or "")
