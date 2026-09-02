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
"""Live self-checks for a running session — the text UI's ``\\doctor`` command.

The pytest suite proves the code is correct in the abstract: mocks stand in
for servers, no model is called, nothing listens on a port.  ``\\doctor``
proves the complement — that *this session's* stack is actually wired.  The
MCP servers this process started answer on their ports, the tools they serve
execute a real round trip, the model endpoint lists models and answers a
probe, and the harness's own tools work against this session's data path.
It is the thing to run after pulling changes, before trusting the session
with real work:

    \\doctor          fast checks — servers, tools, prompts, endpoint.  No
                      tokens spent, a few seconds.
    \\doctor deep     everything above plus a live model reply and a full
                      tool-calling turn.  Costs a few hundred tokens and
                      up to a couple of minutes on a slow endpoint.

Design rules the checks follow:

* **Each check is time-boxed.**  A hung MCP server costs its own timeout,
  never the session — the same bargain ``discover_tools`` strikes.
* **A check that cannot run here is a skip, not a failure.**  No
  ``data_path`` means the file-tool round trip has nowhere to run; that is
  a fact about the configuration, not evidence that the code broke.
* **Failures name the layer.**  "endpoint did not list models" and "bash
  tool returned an error" point at different halves of the stack, and a
  check that cannot tell them apart is a check that cannot be acted on.
* **Probes clean up after themselves.**  Every file, note and result a
  check writes is removed before the check returns, best effort.
"""

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass

# The registry type lives beside ``src`` and is imported the way
# src/lib/tools.py imports it — the package root is on sys.path by the time
# any of this runs (onit.py imports lib.tools at module scope; conftest.py
# guarantees it under pytest).
try:
    from type.tools import _STDIO_SPECS, is_stdio_url
except ImportError:  # pragma: no cover - standalone import of this module
    from ..type.tools import _STDIO_SPECS, is_stdio_url

from ..lib.tools import _wait_for_port

# Per-check ceilings, in seconds.  Generous where a cold start is legitimate
# (a stdio server spawns a Python subprocess on first contact; a model turn
# queues behind whatever else the endpoint is serving), tight where a hang
# means broken (a TCP port that will not open).
FAST_TIMEOUT_S = 30.0
PORT_TIMEOUT_S = 8.0
TOOL_CALL_TIMEOUT_S = 45.0
MODEL_REPLY_TIMEOUT_S = 90.0
MODEL_TOOL_TURN_TIMEOUT_S = 240.0

# The toolset the shipped servers serve between them (see the header of
# src/mcp/servers/tasks/tools/mcp_server.py).  A session running a deliberately
# minimal config will fail this check by design — that is the check saying
# "this is not the toolset the code ships", which is exactly what a
# post-refactor run wants to know.
DEFAULT_TOOLS = (
    "bash", "read_file", "write_file", "edit_file", "grep", "search_document",
    "local_search", "index_documents", "send_file", "serve",
    "search", "fetch_content", "get_weather", "github_repo",
)

REPLY_PROBE = (
    "This is an automated self-check, not a real task. "
    "Reply with exactly this single line and nothing else: DOCTOR-OK"
)

TOOL_TURN_INSTRUCTION = (
    "This is an automated self-check. Use the bash tool to run exactly: "
    "echo DOCTOR-TOOL-OK — then reply with that command's exact output "
    "and nothing else."
)


# ── results ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CheckResult:
    """One check's outcome, in the shape the report renders."""
    name: str
    state: str            # "pass" | "fail" | "skip"
    detail: str = ""
    elapsed: float = 0.0

    @property
    def mark(self) -> str:
        return {"pass": "✓", "fail": "✗", "skip": "–"}[self.state]


def _result(name: str, ok: bool, detail: str, start: float,
            skip: bool = False) -> CheckResult:
    """Build a result, timing it from ``start`` (``time.monotonic()``)."""
    state = "skip" if skip else ("pass" if ok else "fail")
    return CheckResult(name, state, detail, time.monotonic() - start)


def _short(text: str, limit: int = 120) -> str:
    """One line, trimmed — probe output goes in failure details."""
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[:limit - 1] + "…"


# ── the checks ───────────────────────────────────────────────────────────────

async def check_config(agent) -> CheckResult:
    """The loaded config has the three things a session cannot run without."""
    start = time.monotonic()
    cfg = getattr(agent, "config_data", None)
    if not isinstance(cfg, dict) or not cfg:
        return _result("config", False, "no config loaded", start)
    serving = cfg.get("serving") or {}
    # Two config shapes carry an endpoint: the endpoints list (one entry per
    # server, with optional per-entry model/priority) and the legacy
    # host/host2 pair.  A session works through either, so the check does.
    endpoints = serving.get("endpoints") or []
    if endpoints:
        hosts = [e.get("host") if isinstance(e, dict) else e
                 for e in endpoints if e]
        hosts = [h for h in hosts if h]
        if not hosts:
            return _result("config", False,
                           "serving.endpoints is set but no entry has a host",
                           start)
        detail = f"{len(hosts)} endpoint(s) from serving.endpoints"
    else:
        if not serving.get("host"):
            return _result("config", False, "serving.host is not set", start)
        detail = f"endpoint {serving['host']}"
    servers = (cfg.get("mcp") or {}).get("servers") or []
    if not servers:
        return _result("config", False, "no mcp.servers configured", start)
    from .. import setup as onit_setup
    path = onit_setup.CONFIG_PATH
    where = path if os.path.isfile(path) else "built-in defaults"
    return _result("config", True,
                   f"{detail}, {len(servers)} MCP server(s) ({where})", start)


async def check_mcp_servers(agent) -> CheckResult:
    """Every enabled MCP server is reachable — on its port, or spawnable."""
    start = time.monotonic()
    servers = [s for s in getattr(agent, "mcp_servers", [])
               if s.get("enabled", True)]
    if not servers:
        return _result("mcp-servers", False, "no enabled MCP servers", start)

    parts, broken = [], []
    for s in servers:
        name = s.get("name") or "?"
        url = s.get("url")
        if not url:
            parts.append(f"{name}: no url")
            broken.append(name)
            continue
        if is_stdio_url(url):
            # A stdio server has no port to poll: "reachable" means its
            # launch spec is registered, so the client can spawn it.
            ok = url in _STDIO_SPECS
            parts.append(f"{name}: {'stdio spec ok' if ok else 'stdio spec MISSING'}")
        else:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 80
            ok = await _wait_for_port(host, port, timeout=PORT_TIMEOUT_S)
            parts.append(f"{name}:{port} {'up' if ok else 'DOWN'}")
        if not ok:
            broken.append(name)

    detail = " · ".join(parts)
    if broken:
        return _result("mcp-servers", False,
                       f"{detail} — unreachable: {', '.join(broken)}", start)
    return _result("mcp-servers", True, detail, start)


async def check_tool_registry(agent) -> CheckResult:
    """Discovery ran and registered something."""
    start = time.monotonic()
    reg = getattr(agent, "tool_registry", None)
    if reg is None:
        return _result("tool-registry", False,
                       "no tool registry — discovery never ran", start)
    n = len(reg)
    if n == 0:
        return _result("tool-registry", False,
                       "0 tools discovered — MCP servers up but discovery "
                       "returned nothing", start)
    collisions = getattr(reg, "collisions", [])
    detail = f"{n} tools"
    if collisions:
        names = ", ".join(sorted({c[0] for c in collisions}))
        detail += (f", {len(collisions)} name collision(s) resolved by "
                   f"registration order: {names}")
    return _result("tool-registry", True, detail, start)


async def check_default_tools(agent) -> CheckResult:
    """Every tool the shipped servers serve was discovered."""
    start = time.monotonic()
    reg = getattr(agent, "tool_registry", None)
    if reg is None:
        return _result("default-tools", False, "no tool registry", start)
    missing = [t for t in DEFAULT_TOOLS if t not in reg.tools]
    if missing:
        return _result("default-tools", False,
                       f"missing {len(missing)} of {len(DEFAULT_TOOLS)}: "
                       f"{', '.join(missing)}", start)
    return _result("default-tools", True,
                   f"all {len(DEFAULT_TOOLS)} default tools discovered", start)


async def _call_tool(registry, name: str, **kwargs):
    """One tool round trip through the live registry.

    The single funnel for the probe calls, so the tests can stand in for the
    network at one seam (``doctor._call_tool``) rather than one per tool.
    """
    handler = registry[name]
    if handler is None:
        raise RuntimeError(f"tool {name!r} has no handler")
    out = await handler(**kwargs)
    if out is None:
        raise RuntimeError(f"tool {name!r} returned nothing")
    return out


def _tool_json(out: str, name: str) -> dict:
    """Parse a tool's JSON reply, or raise with the raw text attached."""
    try:
        data = json.loads(out)
    except (TypeError, ValueError):
        raise RuntimeError(f"{name} returned non-JSON: {_short(out)}")
    if not isinstance(data, dict):
        raise RuntimeError(f"{name} returned unexpected payload: {_short(out)}")
    return data


async def check_tool_bash(agent) -> CheckResult:
    """The bash tool executes — the whole MCP path, end to end."""
    start = time.monotonic()
    reg = getattr(agent, "tool_registry", None)
    if reg is None or "bash" not in reg.tools:
        return _result("tool-bash", False, "bash tool not discovered", start,
                       skip=True)
    probe = f"onit-doctor-{int(time.time())}"
    data = _tool_json_safe(await _call_tool(reg, "bash", command=f"echo {probe}"),
                           "bash")
    if data.get("returncode") != 0:
        return _result("tool-bash", False,
                       f"exit {data.get('returncode')}: "
                       f"{_short(data.get('stderr') or data.get('stdout') or '')}",
                       start)
    if probe not in (data.get("stdout") or ""):
        return _result("tool-bash", False,
                       f"stdout did not carry the probe: "
                       f"{_short(data.get('stdout') or '')}", start)
    return _result("tool-bash", True, f"echo round trip ok ({probe})", start)


def _tool_json_safe(out, name: str) -> dict:
    """``_tool_json`` for checks that treat a bad reply as a failure, not a raise."""
    try:
        return _tool_json(out, name)
    except RuntimeError as e:
        return {"returncode": -1, "stderr": str(e)}


async def check_tool_files(agent) -> CheckResult:
    """write_file → read_file → edit_file round trip inside the data path.

    The probe uses a *relative* path on purpose: the server resolves it
    against the session's own working directory (``_validate_write_path``
    resolves relative paths against the jail root), which is exactly the
    path handling a real task rides.  An absolute temp path would be
    rejected by the server's jail whenever it sits outside the server's
    data directory — a configuration fact, not a code break.
    """
    start = time.monotonic()
    reg = getattr(agent, "tool_registry", None)
    if reg is None or not {"write_file", "read_file", "edit_file"} <= reg.tools:
        return _result("tool-files", False,
                       "file tools not all discovered", start, skip=True)
    data_path = getattr(agent, "data_path", "") or ""
    if not data_path:
        return _result("tool-files", False,
                       "no data_path this session — nowhere to probe", start,
                       skip=True)

    probe = f"doctor-probe-{os.getpid()}-{int(time.time())}.txt"
    marker = f"DOCTOR-PROBE-{int(time.time())}"
    try:
        w = _tool_json_safe(await _call_tool(reg, "write_file", path=probe,
                                             content=marker, data_path=data_path),
                            "write_file")
        path = os.path.join(data_path, probe)
        if not os.path.isfile(path):
            return _result("tool-files", False,
                           f"write_file reported {w.get('status', '?')} but "
                           f"no file on disk at {probe}", start)
        r = await _call_tool(reg, "read_file", path=probe, data_path=data_path)
        if marker not in str(r):
            return _result("tool-files", False,
                           f"read_file did not return what write_file wrote: "
                           f"{_short(r)}", start)
        edited = marker + "-EDITED"
        e = _tool_json_safe(await _call_tool(reg, "edit_file", path=probe,
                                             old_string=marker,
                                             new_string=edited,
                                             data_path=data_path),
                            "edit_file")
        if e.get("replacements", 0) < 1:
            return _result("tool-files", False,
                           f"edit_file replaced nothing: {_short(str(e))}", start)
        r2 = await _call_tool(reg, "read_file", path=probe, data_path=data_path)
        if edited not in str(r2):
            return _result("tool-files", False,
                           "edit_file claimed success but the file did not change",
                           start)
        return _result("tool-files", True,
                       f"write/read/edit round trip ok ({probe})", start)
    finally:
        try:
            os.remove(os.path.join(data_path, probe))
        except OSError:
            pass


async def check_harness_tools(agent) -> CheckResult:
    """The harness's own tools — note_write/note_read/context_status — work.

    Unlike the MCP tool probes this one runs entirely in process, so a
    temporary directory is a faithful stand-in for the session's data path:
    the note tools jail to whatever ``data_path`` they are handed.
    """
    start = time.monotonic()
    try:
        from ..model.serving.harness import HarnessTools
    except Exception as e:
        return _result("harness-tools", False,
                       f"import failed: {type(e).__name__}: {e}", start)

    data_path = getattr(agent, "data_path", "") or ""
    in_temp = not data_path
    if in_temp:
        data_path = tempfile.mkdtemp(prefix="onit-doctor-")
    h = HarnessTools(data_path=data_path, enabled=True)
    if "note_write" not in h.names or "context_status" not in h.names:
        return _result("harness-tools", False,
                       f"harness offers {', '.join(h.names) or 'nothing'}; "
                       f"note tools missing", start)
    try:
        w = h.note_write("doctor_probe", "ok")
        if '"saved"' not in str(w):
            return _result("harness-tools", False, f"note_write: {_short(str(w))}",
                           start)
        r = h.note_read("doctor_probe")
        if r != "ok":
            return _result("harness-tools", False,
                           f"note_read returned {_short(str(r))!r}", start)
        cs = json.loads(h.context_status())
        if "doctor_probe" not in (cs.get("notes_saved") or []):
            return _result("harness-tools", False,
                           "context_status does not list the probe note", start)
        where = "temp dir" if in_temp else "data path"
        return _result("harness-tools", True,
                       f"note write/read + context_status ok ({where})", start)
    finally:
        note = os.path.join(data_path, ".onit", "notes", "doctor_probe.md")
        try:
            os.remove(note)
        except OSError:
            pass


async def check_prompts(agent) -> CheckResult:
    """The agent instruction assembles — in process, or over MCP when so configured.

    ``build_assistant_instruction`` is a pure function (it creates the data
    directory it is pointed at and assembles a string), so a temp dir is a
    legitimate stand-in when this session has no data path.
    """
    start = time.monotonic()
    data_path = getattr(agent, "data_path", "") or ""
    in_temp = not data_path
    if in_temp:
        data_path = tempfile.mkdtemp(prefix="onit-doctor-")
    try:
        from ..mcp.prompts.prompts import build_assistant_instruction
        text = await build_assistant_instruction(task="doctor self-check probe",
                                                 data_path=data_path)
    except Exception as e:
        return _result("prompts", False,
                       f"{type(e).__name__}: {e}", start)
    if not text or len(text) < 200:
        return _result("prompts", False,
                       f"instruction suspiciously short ({len(text or '')} chars)",
                       start)

    detail = f"instruction built ({len(text):,} chars"
    if in_temp:
        detail += ", temp data dir"
    detail += ")"
    # When the prompt is served over MCP rather than called in process, that
    # round trip is the real path — exercise it too, or a broken prompt server
    # would surface only as a stall on the next real task.
    if not getattr(agent, "prompt_in_process", True):
        url = getattr(agent, "prompt_url", None)
        if not url:
            return _result("prompts", False,
                           "prompt_in_process is off but no prompt_url", start)
        try:
            from fastmcp import Client
            async with Client(url) as client:
                result = await client.get_prompt("assistant", {
                    "task": "doctor self-check probe", "data_path": data_path})
            served = result.messages[0].content.text
        except Exception as e:
            return _result("prompts", False,
                           f"prompt server {url}: {type(e).__name__}: {e}", start)
        if served != text:
            return _result("prompts", False,
                           "served prompt differs from the in-process one", start)
        detail += ", served over MCP and identical"
    return _result("prompts", True, detail, start)


async def check_load_balancer(agent) -> CheckResult:
    """Endpoints exist, one is assigned, and not all of them are cooling down."""
    start = time.monotonic()
    lb = getattr(agent, "load_balancer", None)
    endpoints = getattr(lb, "endpoints", None) if lb is not None else None
    if not endpoints:
        return _result("load-balancer", False, "no endpoints configured", start)
    ep = lb.assigned(getattr(agent, "session_id", None))
    if ep is None:
        return _result("load-balancer", False,
                       "no endpoint assigned to this session", start)
    down = [e.name or e.host for e in endpoints if not e.is_healthy()]
    detail = (f"{len(endpoints)} endpoint(s), serving via "
              f"{ep.name or ep.host}")
    if down:
        if len(down) == len(endpoints):
            return _result("load-balancer", False,
                           f"all endpoints cooling down: {', '.join(down)}", start)
        detail += f" (cooling down: {', '.join(down)})"
    return _result("load-balancer", True, detail, start)


async def check_endpoint(agent) -> CheckResult:
    """The serving endpoint answers a model listing, and serves its configured model."""
    start = time.monotonic()
    lb = getattr(agent, "load_balancer", None)
    ep = lb.assigned(getattr(agent, "session_id", None)) if lb is not None else None
    if ep is None:
        return _result("endpoint", False, "no endpoint assigned", start)
    from ..model.serving.chat import list_models
    try:
        names = await list_models(ep.host, ep.host_key)
    except Exception as e:
        return _result("endpoint", False,
                       f"{ep.host}: {type(e).__name__}: {e}", start)
    if not names:
        return _result("endpoint", False,
                       f"{ep.host} listed no models", start)
    # Ollama cloud lists its models without the ":cloud" suffix but accepts
    # the suffixed name in a chat request (verified against the live API), so
    # a configured "x:cloud" is served when "x" is listed.  A verbatim
    # comparison here would flag a working endpoint as broken.
    if ep.model and ep.model not in names:
        bare = ep.model.removesuffix(":cloud")
        if not (bare != ep.model and bare in names):
            shown = ", ".join(names[:5]) + ("…" if len(names) > 5 else "")
            return _result("endpoint", False,
                           f"{ep.host} does not serve configured model "
                           f"{ep.model!r} (serves: {shown})", start)
    detail = f"{ep.host} serves {len(names)} model(s)"
    if ep.model:
        detail += f", {ep.model} listed"
    return _result("endpoint", True, detail, start)


async def check_session_history(agent) -> CheckResult:
    """The session file is readable and the sessions directory is live."""
    start = time.monotonic()
    history = agent.load_session_history(max_turns=1)
    if history is None:
        return _result("session-history", False,
                       "load_session_history returned None", start)
    path = getattr(agent, "session_path", "") or ""
    detail = "history readable"
    if path:
        detail += f" ({path})"
    return _result("session-history", True, detail, start)


async def check_commands(agent) -> CheckResult:
    """Every backslash command parses and is listed by \\help."""
    start = time.monotonic()
    from . import commands
    text = commands.render_help()
    missing = [c.name for c in commands.COMMANDS if c.usage not in text]
    if missing:
        return _result("commands", False,
                       f"\\help does not list: {', '.join(missing)}", start)
    if commands.parse("\\help") != ("help", ""):
        return _result("commands", False, "parse(\\\\help) is broken", start)
    for word in commands.STOP_ALIASES:
        if commands.parse(f"\\{word}") is None:
            return _result("commands", False,
                           f"stop alias \\{word} no longer parses", start)
    return _result("commands", True,
                   f"{len(commands.COMMANDS)} commands parse and list", start)


async def check_learn(agent) -> CheckResult:
    """Trajectory recording is configured coherently and its directory is writable."""
    start = time.monotonic()
    try:
        from ..learn import config as learn_config
    except Exception as e:
        return _result("learn", False,
                       f"import failed: {type(e).__name__}: {e}", start)
    cfg = getattr(agent, "config_data", None) or {}
    level = learn_config.autonomy(cfg)
    if level < learn_config.OBSERVE:
        return _result("learn", False, "recording off (learn.autonomy: off)",
                       start, skip=True)
    where = learn_config.trajectory_dir(cfg)
    probe = os.path.join(where, f".doctor-probe-{os.getpid()}")
    try:
        os.makedirs(where, exist_ok=True)
        with open(probe, "w", encoding="utf-8") as f:
            f.write("probe")
        os.remove(probe)
    except OSError as e:
        return _result("learn", False,
                       f"{where} is not writable: {e}", start)
    return _result("learn", True,
                   f"recording on ({learn_config.level_name(level)}), "
                   f"{where} writable", start)


# ── the deep checks: one live model turn each ────────────────────────────────

async def check_model_reply(agent) -> CheckResult:
    """The endpoint answers through chat() — the plumbing a real task rides."""
    start = time.monotonic()
    lb = getattr(agent, "load_balancer", None)
    ep = lb.assigned(getattr(agent, "session_id", None)) if lb is not None else None
    if ep is None:
        return _result("model-reply", False, "no endpoint assigned", start)
    from ..model.serving.chat import chat
    out = await chat(
        host=ep.host, host_key=ep.host_key, model=ep.model,
        instruction=REPLY_PROBE,
        tool_registry=None,                      # no tools: one plain turn
        safety_queue=asyncio.Queue(),
        chat_ui=None, verbose=False, stream=False,
        think=bool((getattr(agent, "model_serving", {}) or {}).get("think", False)),
        max_tokens=256,
        verify_answers=False,                    # a probe has nothing to fact-check
        session_history=[],
        session_id=f"doctor-{int(time.time())}",
    )
    if out and "DOCTOR-OK" in out:
        return _result("model-reply", True,
                       f"{ep.name or ep.host} replied through chat()", start)
    return _result("model-reply", False,
                   f"model did not echo the probe — got: {_short(out or 'nothing')}",
                   start)


async def check_model_tool_turn(agent) -> CheckResult:
    """The full agent loop: the model calls a tool and reports its result."""
    start = time.monotonic()
    reg = getattr(agent, "tool_registry", None)
    if reg is None or "bash" not in reg.tools:
        return _result("model-tool-turn", False,
                       "needs the bash tool", start, skip=True)
    lb = getattr(agent, "load_balancer", None)
    ep = lb.assigned(getattr(agent, "session_id", None)) if lb is not None else None
    if ep is None:
        return _result("model-tool-turn", False, "no endpoint assigned", start)
    from ..model.serving.chat import chat
    # The session's own data_path, exactly as a real task passes it — the MCP
    # server jails data_path inside its own data directory, so a temp-dir
    # stand-in here would make the model's bash call fail for a reason that
    # has nothing to do with the code under test.  An empty value is fine:
    # the server falls back to its own data directory.
    data_path = getattr(agent, "data_path", "") or ""
    out = await chat(
        host=ep.host, host_key=ep.host_key, model=ep.model,
        instruction=TOOL_TURN_INSTRUCTION,
        tool_registry=reg,
        safety_queue=asyncio.Queue(),
        chat_ui=None, verbose=False, stream=False,
        think=bool((getattr(agent, "model_serving", {}) or {}).get("think", False)),
        max_tokens=1024,
        data_path=data_path,
        verify_answers=False,
        session_history=[],
        session_id=f"doctor-{int(time.time())}",
    )
    if out and "DOCTOR-TOOL-OK" in out:
        return _result("model-tool-turn", True,
                       "model called bash and reported its output", start)
    return _result("model-tool-turn", False,
                   "the tool-calling turn did not land the probe output — "
                   f"got: {_short(out or 'nothing')}", start)


# ── the runner and the report ────────────────────────────────────────────────

# (name, check, timeout) in report order.  The names double as the row labels,
# so they stay short and layer-named rather than sentence-shaped.
CHECKS = (
    ("config", check_config, FAST_TIMEOUT_S),
    ("commands", check_commands, FAST_TIMEOUT_S),
    ("load-balancer", check_load_balancer, FAST_TIMEOUT_S),
    ("mcp-servers", check_mcp_servers, PORT_TIMEOUT_S * 4),
    ("tool-registry", check_tool_registry, FAST_TIMEOUT_S),
    ("default-tools", check_default_tools, FAST_TIMEOUT_S),
    ("tool-bash", check_tool_bash, TOOL_CALL_TIMEOUT_S),
    ("tool-files", check_tool_files, TOOL_CALL_TIMEOUT_S),
    ("harness-tools", check_harness_tools, FAST_TIMEOUT_S),
    ("prompts", check_prompts, FAST_TIMEOUT_S),
    ("endpoint", check_endpoint, FAST_TIMEOUT_S),
    ("session-history", check_session_history, FAST_TIMEOUT_S),
    ("learn", check_learn, FAST_TIMEOUT_S),
)

DEEP_CHECKS = (
    ("model-reply", check_model_reply, MODEL_REPLY_TIMEOUT_S),
    ("model-tool-turn", check_model_tool_turn, MODEL_TOOL_TURN_TIMEOUT_S),
)


async def run_checks(agent, deep: bool = False, on_start=None) -> list[CheckResult]:
    """Run the battery, each check inside its own timeout.

    ``on_start(name)`` is called just before each check begins — the text UI
    uses it to show progress while a deep check holds the session for a
    model turn.  A check that raises or outlives its timeout is recorded as
    a failure carrying the exception or the timeout, never propagated: a
    self-check that crashes the session would be a poor self-check.
    """
    selected = CHECKS + (DEEP_CHECKS if deep else ())
    results = []
    for name, fn, timeout in selected:
        if on_start:
            try:
                on_start(name)
            except Exception:
                pass
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(fn(agent), timeout=timeout)
        except asyncio.TimeoutError:
            result = CheckResult(name, "fail", f"timed out after {timeout:g}s",
                                 time.monotonic() - start)
        except Exception as e:
            result = CheckResult(name, "fail", f"{type(e).__name__}: {e}",
                                 time.monotonic() - start)
        results.append(result)
    return results


def render_report(results: list[CheckResult], deep: bool = False) -> str:
    """The report as plain text — it lands in the transcript as a system message."""
    if not results:
        return "Self-check ran nothing — no checks selected."

    width = max(len(r.name) for r in results)
    lines = []
    for r in results:
        suffix = f"  {r.detail}" if r.detail else ""
        lines.append(f"  {r.mark} {r.name:<{width}}{suffix}")

    passed = sum(1 for r in results if r.state == "pass")
    failed = sum(1 for r in results if r.state == "fail")
    skipped = sum(1 for r in results if r.state == "skip")
    elapsed = sum(r.elapsed for r in results)

    parts = [f"{passed} passed"]
    if failed:
        parts.append(f"{failed} failed")
    if skipped:
        parts.append(f"{skipped} skipped")
    header = f"Self-check ({'fast + deep' if deep else 'fast'}) — " \
             f"{', '.join(parts)} in {elapsed:.1f}s"

    out = [header, ""] + lines
    if failed:
        out += ["", f"{failed} check(s) failed. Fix, then run \\doctor again."]
        if not deep:
            out.append("\\doctor deep also exercises a live model reply and a "
                       "tool-calling turn (costs tokens).")
    elif skipped:
        out += ["", "All checks that ran passed. Skipped ones could not run "
                "here — see their lines above."]
    else:
        out += ["", "Nothing is broken."]
    return "\n".join(out)