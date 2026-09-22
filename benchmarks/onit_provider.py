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
    * The local ``src/mcp`` package shadows the PyPI ``mcp`` SDK whenever
      ``src/`` is on sys.path, which breaks fastmcp's lazy imports (its
      ``import mcp.types`` resolves to the local package and fails).  The
      pre-imports below pin the real SDK modules in sys.modules before the
      agent stack is loaded, so the shadow cannot bite.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

# Must precede any import of the agent stack (see module docstring).
# The PyPI mcp SDK and fastmcp must be imported BEFORE src/ is put on
# sys.path: once "src" is on the path, the local src/mcp package shadows the
# SDK, and fastmcp's lazy "import mcp.types" then resolves to the shadow
# (which has no types module) and raises "FastMCP server support is not
# installed".  Importing the SDK first pins the real modules in sys.modules,
# so the shadow cannot bite anything imported afterwards.  This is safe for
# the agent stack: it reaches the local package only through relative imports
# (e.g. "from .mcp.prompts.prompts import ..."), which do not consult
# sys.path, and its one bare SDK import (src/type/tools.py) wants the real
# SDK anyway.
#
# The SDK-first import alone is not enough in multiprocessing spawn children:
# they re-import this module with the parent's sys.path *already* containing
# "src" (it was inserted before the servers were spawned), so the very
# `import mcp.types` below would resolve to the shadow and every MCP server
# child crash-loops (observed 2026-09-05). Demote the src entry to the tail
# first: absolute `src.*` imports still resolve (the entry is present, just
# not first), and the bare-SDK import below then finds the PyPI package.
import os as _os

_src_dir = _os.path.realpath(
    _os.path.join(_os.path.dirname(__file__), _os.pardir, "src"))
_demoted = [p for p in sys.path if p and _os.path.realpath(p) == _src_dir]
for _p in _demoted:
    sys.path.remove(_p)
sys.path.extend(_demoted)

import mcp.types  # noqa: F401
import fastmcp.server.context  # noqa: F401
import fastmcp.server  # noqa: F401
import fastmcp.client  # noqa: F401

sys.path.insert(0, ".")
sys.path.insert(0, "src")

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
# Cross-sample concurrency cap (see generate()); None = no cap.
_concurrency_sem: asyncio.Semaphore | None = None


def base_config_data() -> dict[str, Any]:
    """Build the headless OnIt config used to drive benchmarks.

    ``loop=True`` skips terminal-UI setup; streaming is off. The eval target
    (host/model) and per-request timeout come from the environment.
    """
    sessions_dir = Path(tempfile.gettempdir()) / "onit-bench-sessions"
    data_dir = bench_config.bench_data_root()
    sessions_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return {
        "serving": bench_config.resolve_serving(),
        # Off unless ONIT_BENCH/ONIT_LEARN asked otherwise: a baseline row has
        # to be the scaffold as shipped, or it is not a baseline.
        "learn": bench_config.resolve_learn(),
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
            # Names must match the split defaults in src/lib/tools.py
            # (DEFAULT_MCP_SERVERS): apply_default_mcp_servers() adds any
            # missing default server, and _ensure_mcp_servers() registers a
            # stdio launch spec for every server. A name the agent no longer
            # knows (the old combined "ToolsMCPServer") gets no spec, so it
            # stays unspawnable and discovery times out on it.
            "servers": [
                {"name": "PromptsMCPServer", "transport": "stdio",
                 "module": "src.mcp.prompts.prompts", "enabled": True},
                # The current split tools servers (with real modules), not the
                # legacy combined "ToolsMCPServer" which has no module and so
                # is never spawned. Both are stdio (spawned by the MCP client
                # on first use): ToolsLocal carries the data_path tools,
                # ToolsNet the stateless ones. apply_default_mcp_servers()
                # sees these names and adds nothing, so the list stays exactly
                # as written.
                {"name": "ToolsLocalMCPServer", "transport": "stdio",
                 "module": "tasks.tools", "options": {"profile": "local"},
                 "enabled": True},
                {"name": "ToolsNetMCPServer", "transport": "stdio",
                 "module": "tasks.tools", "options": {"profile": "net"},
                 "enabled": True},
            ]
        },
    }


def _build_agent_blocking(config_overrides: dict[str, Any] | None = None) -> Any:
    """Construct an OnIt agent. Must run in a thread with no running loop.

    Registers the MCP stdio launch specs (idempotent) then builds
    the agent, which discovers tools against those servers. ``config_overrides``
    are shallow-merged onto :func:`base_config_data` (e.g. ``{"data_path": ...}``
    to root the agent's file tools at a per-instance workspace).
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
        data_dir = bench_config.bench_data_root() / run_id
        data_dir.mkdir(parents=True, exist_ok=True)

        stats: dict[str, Any] = {}
        # Cloud endpoints rate-limit concurrent requests (Ollama cloud
        # answers 429 "too many concurrent requests" when the tier's
        # max_connections samples hit it at once, each sample's tool loop
        # firing several requests). A module-level semaphore caps the
        # agent requests in flight across the whole run; the cap is
        # ONIT_BENCH_MAX_CONNECTIONS (0 = no cap, the default, which is
        # what local vLLM endpoints want). Set it to 1 or 2 for cloud
        # endpoints.
        global _concurrency_sem
        if _concurrency_sem is None:
            max_conc = int(os.environ.get("ONIT_BENCH_MAX_CONNECTIONS", "0") or 0)
            _concurrency_sem = asyncio.Semaphore(max_conc) if max_conc > 0 else None
        sem = _concurrency_sem
        if sem is not None:
            async with sem:
                answer = await agent.process_task(
                    task,
                    session_path=str(sessions_dir / f"{run_id}.jsonl"),
                    data_path=str(data_dir),
                    safety_queue=asyncio.Queue(),
                    stats=stats,
                )
        else:
            answer = await agent.process_task(
                task,
                session_path=str(sessions_dir / f"{run_id}.jsonl"),
                data_path=str(data_dir),
                safety_queue=asyncio.Queue(),
                stats=stats,
            )

        # Wall time and tokens per sample (S1).  process_task fills
        # stats["metrics"] with the TurnMetrics sink; publishing it here puts
        # it in the sample's metadata, where report.py's wall/token columns
        # and the S1 gate read it from.  A run that failed before any turn
        # has an empty sink — publish zeros so the column always exists.
        _m = stats.get("metrics") or {}
        return ModelOutput.from_content(model=self.model_name, content=answer or "",
                                        metadata={
                                            "wall_s": round(float(_m.get("model_s", 0.0))
                                                            + float(_m.get("tool_s", 0.0))
                                                            + float(_m.get("compaction_s", 0.0))
                                                            + float(_m.get("verify_s", 0.0))
                                                            + float(_m.get("instruction_s", 0.0)), 3),
                                            "model_s": float(_m.get("model_s", 0.0)),
                                            "prefill_s": float(_m.get("prefill_s", 0.0)),
                                            "decode_s": float(_m.get("decode_s", 0.0)),
                                            "tool_s": float(_m.get("tool_s", 0.0)),
                                            "ttft_s": float(_m.get("ttft_s", 0.0) or 0.0),
                                            "prompt_tokens_max": int(_m.get("prompt_tokens_max", 0)),
                                            "completion_tokens": int(_m.get("completion_tokens", 0)),
                                            "cached_tokens": int(_m.get("cached_tokens", 0)),
                                            "compactions": int(_m.get("compactions", 0)),
                                            "turn_count": int(_m.get("turn_count", 0)),
                                            "retries": int(_m.get("api_retries", 0)),
                                        })
