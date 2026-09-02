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

    \\doctor          fast checks — servers, every shipped tool, prompts,
                      endpoint.  No tokens spent, a few seconds.
    \\doctor deep     everything above plus live model turns: a plain reply,
                      a tool-calling turn, and a file-reading turn.  Costs a
                      few hundred tokens and up to a couple of minutes on a
                      slow endpoint.

Design rules the checks follow:

* **Each check is time-boxed.**  A hung MCP server costs its own timeout,
  never the session — the same bargain ``discover_tools`` strikes.
* **A check that cannot run here is a skip, not a failure.**  No
  ``data_path`` means the file-tool round trip has nowhere to run; a tool
  that needs an API key this machine does not have cannot be exercised.
  Those are facts about the configuration, not evidence that the code
  broke.
* **Failures name the layer.**  "endpoint did not list models" and "bash
  tool returned an error" point at different halves of the stack, and a
  check that cannot tell them apart is a check that cannot be acted on.
* **Probes clean up after themselves.**  Every file, note, index entry,
  managed process and result a check writes is removed or stopped before
  the check returns, best effort.  The one deliberate exception is the
  local-search index file itself: it is the session's own index, so the
  probe re-indexes the emptied corpus instead of deleting the file.
* **Every shipped tool is exercised, not merely listed.**  Discovery
  proves a tool's name arrived; a round trip proves it still works.  The
  fast battery runs one against each of the fourteen, in the cheapest
  safe form each supports — and the tools that would reach out to a
  third party (weather, GitHub) skip cleanly when their key is absent
  rather than counting a missing credential as a broken tool.
"""

import asyncio
import base64
import json
import os
import shutil
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
# The retrieval probes do real work \u2014 chunking, embedding dispatch, a
# subprocess grep \u2014 so they get a wider box than a single tool call.
RAG_TIMEOUT_S = 90.0
SERVE_TIMEOUT_S = 60.0
WEB_TIMEOUT_S = 60.0
KEYED_TIMEOUT_S = 60.0
MODEL_REPLY_TIMEOUT_S = 90.0
MODEL_TOOL_TURN_TIMEOUT_S = 240.0
MODEL_FILE_TURN_TIMEOUT_S = 120.0

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

# The deep file-reading turn: the harness writes a probe file, then asks the
# model to read it with the read_file tool and echo its contents.  This is
# the read half of the tool loop — check_model_tool_turn exercises the
# bash half — and it is the path every document task rides.
FILE_TURN_INSTRUCTION = (
    "This is an automated self-check. Use the read_file tool to read the "
    "file at the path given below, then reply with that file's exact "
    "contents and nothing else.\nPath: {path}"
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


async def _call_tool(registry, tool: str, **kwargs):
    """One tool round trip through the live registry.

    The single funnel for the probe calls, so the tests can stand in for the
    network at one seam (``doctor._call_tool``) rather than one per tool.
    The tool-name parameter is ``tool`` rather than ``name`` because several
    tools take a ``name`` argument of their own (serve's process label) —
    a colliding parameter would make every such probe a TypeError.
    """
    handler = registry[tool]
    if handler is None:
        raise RuntimeError(f"tool {tool!r} has no handler")
    out = await handler(**kwargs)
    if out is None:
        raise RuntimeError(f"tool {tool!r} returned nothing")
    return out


def _tool_json(out: str, name: str) -> dict:
    """Parse a tool's JSON reply, or raise with the raw text attached.

    A bare JSON array — the web search tool's success shape — is wrapped as
    ``{"results": [...]}`` so the checks can treat one payload shape.
    """
    try:
        data = json.loads(out)
    except (TypeError, ValueError):
        raise RuntimeError(f"{name} returned non-JSON: {_short(out)}")
    if isinstance(data, list):
        return {"results": data}
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


# ── the remaining shipped tools: one probe each ────────────────────────────
#
# Discovery proves a tool's *name* arrived; these prove each tool still
# *works*.  Each probe is the cheapest safe round trip the tool supports,
# and each cleans up after itself.  A probe that cannot run here — the tool
# was not discovered, or it needs a credential this machine does not have —
# is a skip, not a failure: a missing key is a fact about the configuration,
# not evidence that the code broke.


def _reg_skip(agent, name: str, needed: tuple, start: float) -> CheckResult | None:
    """The shared skip shape for the per-tool probes."""
    reg = getattr(agent, "tool_registry", None)
    if reg is None or not set(needed) <= reg.tools:
        return _result(name, False, f"{', '.join(needed)} not discovered",
                       start, skip=True)
    return None


async def check_tool_grep(agent) -> CheckResult:
    """grep finds a planted marker in a planted file — the subprocess path."""
    start = time.monotonic()
    if early := _reg_skip(agent, "tool-grep", ("grep",), start):
        return early
    reg = agent.tool_registry
    data_path = getattr(agent, "data_path", "") or ""
    if not data_path:
        return _result("tool-grep", False,
                       "no data_path this session — nowhere to probe", start,
                       skip=True)
    marker = f"DOCTOR-GREP-{os.getpid()}"
    probe_dir = os.path.join(data_path, f".doctor-grep-{os.getpid()}")
    os.makedirs(probe_dir, exist_ok=True)
    try:
        with open(os.path.join(probe_dir, "probe.txt"), "w",
                  encoding="utf-8") as f:
            f.write(f"line one\n{marker}\nline three\n")
        out = _tool_json_safe(
            await _call_tool(reg, "grep", path=probe_dir, pattern=marker,
                             data_path=data_path),
            "grep")
        if out.get("total_matches", 0) < 1:
            return _result("tool-grep", False,
                           f"grep found the planted marker 0 times: "
                           f"{_short(str(out))}", start)
        return _result("tool-grep", True,
                       f"grep found the planted marker ({out.get('total_matches')} "
                       f"match)", start)
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


async def check_tool_search_document(agent) -> CheckResult:
    """search_document regex mode over a planted file — no index involved."""
    start = time.monotonic()
    if early := _reg_skip(agent, "tool-search-document",
                          ("search_document",), start):
        return early
    reg = agent.tool_registry
    data_path = getattr(agent, "data_path", "") or ""
    if not data_path:
        return _result("tool-search-document", False,
                       "no data_path this session — nowhere to probe", start,
                       skip=True)
    marker = f"DOCTOR-DOC-{os.getpid()}"
    probe = os.path.join(data_path, f".doctor-doc-{os.getpid()}.md")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write(f"# probe\n\nThe {marker} token lives here.\n")
        out = _tool_json_safe(
            await _call_tool(reg, "search_document", path=probe,
                             mode="pattern", pattern=marker,
                             data_path=data_path),
            "search_document")
        if out.get("total_matches", 0) < 1:
            return _result("tool-search-document", False,
                           f"pattern search missed the planted marker: "
                           f"{_short(str(out))}", start)
        return _result("tool-search-document", True,
                       f"pattern search found the planted marker "
                       f"({out.get('total_matches')} match)", start)
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass


async def check_tool_index_and_search(agent) -> CheckResult:
    """index_documents then local_search over a dedicated probe corpus.

    The probe corpus is its own directory under the data path, so the
    session's real index is never touched.  The index it creates lives in
    the corpus directory itself; the cleanup re-indexes the emptied
    directory (dropping the probe's entries) before removing it.
    """
    start = time.monotonic()
    if early := _reg_skip(agent, "tool-local-search",
                          ("index_documents", "local_search"), start):
        return early
    reg = agent.tool_registry
    data_path = getattr(agent, "data_path", "") or ""
    if not data_path:
        return _result("tool-local-search", False,
                       "no data_path this session — nowhere to probe", start,
                       skip=True)
    marker = f"quantum{os.getpid()}teapot"
    # A plain (non-dot) directory: the corpus walk skips dot-components, so
    # a hidden probe corpus would index zero documents and the search half
    # of this probe would fail for a reason that has nothing to do with the
    # tools under test.
    probe_dir = os.path.join(data_path, f"doctor-corpus-{os.getpid()}")
    os.makedirs(probe_dir, exist_ok=True)
    try:
        with open(os.path.join(probe_dir, "probe.md"), "w",
                  encoding="utf-8") as f:
            f.write(f"# Probe\n\nThe secret passphrase is {marker}.\n")
        idx = _tool_json_safe(
            await _call_tool(reg, "index_documents", path=probe_dir,
                             data_path=data_path),
            "index_documents")
        if idx.get("status") != "success":
            return _result("tool-local-search", False,
                           f"index_documents failed: {_short(str(idx))}", start)
        if idx.get("total_documents", 0) < 1:
            return _result("tool-local-search", False,
                           f"index_documents indexed nothing: "
                           f"{_short(str(idx))}", start)
        found = _tool_json_safe(
            await _call_tool(reg, "local_search", query=marker, top_k=3,
                             method="bm25", path=probe_dir,
                             data_path=data_path),
            "local_search")
        hits = json.dumps(found.get("results") or [])
        if marker not in hits:
            return _result("tool-local-search", False,
                           f"local_search did not retrieve the planted "
                           f"passphrase: {_short(hits)}", start)
        return _result("tool-local-search", True,
                       f"index + local_search round trip ok "
                       f"({idx.get('total_documents')} doc indexed, "
                       f"retrieved)", start)
    finally:
        # Drop the probe's entries from the index, then remove the corpus.
        # method="bm25" keeps the probe off the embedding path; the re-index
        # of an emptied directory is cheap by construction.
        try:
            await _call_tool(reg, "index_documents", path=probe_dir,
                             data_path=data_path)
        except Exception:
            pass
        shutil.rmtree(probe_dir, ignore_errors=True)


async def check_tool_send_file(agent) -> CheckResult:
    """send_file returns the probe file's bytes — the base64, no-upload path."""
    start = time.monotonic()
    if early := _reg_skip(agent, "tool-send-file", ("send_file",), start):
        return early
    reg = agent.tool_registry
    data_path = getattr(agent, "data_path", "") or ""
    if not data_path:
        return _result("tool-send-file", False,
                       "no data_path this session — nowhere to probe", start,
                       skip=True)
    probe = f"doctor-send-{os.getpid()}.txt"
    payload = f"DOCTOR-SEND-{os.getpid()}-{int(time.time())}"
    try:
        with open(os.path.join(data_path, probe), "w", encoding="utf-8") as f:
            f.write(payload)
        out = _tool_json_safe(
            await _call_tool(reg, "send_file", path=probe, data_path=data_path),
            "send_file")
        if out.get("status") != "success":
            return _result("tool-send-file", False,
                           f"send_file failed: {_short(str(out))}", start)
        if base64.b64encode(payload.encode()).decode() != \
                out.get("file_data_base64"):
            return _result("tool-send-file", False,
                           "send_file's base64 does not decode to the "
                           "planted payload", start)
        return _result("tool-send-file", True,
                       f"base64 round trip ok ({probe})", start)
    finally:
        try:
            os.remove(os.path.join(data_path, probe))
        except OSError:
            pass


async def check_tool_serve(agent) -> CheckResult:
    """serve starts, reports, and stops a managed process — the full cycle."""
    start = time.monotonic()
    if early := _reg_skip(agent, "tool-serve", ("serve",), start):
        return early
    reg = agent.tool_registry
    data_path = getattr(agent, "data_path", "") or ""
    name = f"doctor-probe-{os.getpid()}"
    try:
        started = _tool_json_safe(
            await _call_tool(reg, "serve", action="start",
                             command="sleep 30", name=name,
                             data_path=data_path),
            "serve")
        # "started" is the success shape; "already_running" is fine too —
        # a leftover probe from a crashed run is not a broken serve.
        if started.get("status") not in ("started", "already_running"):
            return _result("tool-serve", False,
                           f"serve start did not report started: "
                           f"{_short(str(started))}", start)
        status = _tool_json_safe(
            await _call_tool(reg, "serve", action="status", name=name,
                             data_path=data_path),
            "serve")
        if status.get("status") not in ("running", "already_running"):
            return _result("tool-serve", False,
                           f"serve status did not see the probe running: "
                           f"{_short(str(status))}", start)
        return _result("tool-serve", True,
                       f"start/status/stop cycle ok (pid {started.get('pid')})",
                       start)
    finally:
        try:
            await _call_tool(reg, "serve", action="stop", name=name,
                             data_path=data_path)
        except Exception:
            pass


async def check_tool_search(agent) -> CheckResult:
    """search reaches the web and returns results — one cheap query.

    The web search path returns a bare JSON array of results (or an
    ``{"error": ...}`` object) — there is no ``status`` key to check.
    """
    start = time.monotonic()
    if early := _reg_skip(agent, "tool-search", ("search",), start):
        return early
    out = _tool_json_safe(
        await _call_tool(agent.tool_registry, "search",
                         query="example domain", type="web", max_results=1),
        "search")
    if "error" in out:
        return _result("tool-search", False,
                       f"search failed: {_short(str(out))}", start)
    results = out.get("results") if isinstance(out, dict) else out
    if not results:
        return _result("tool-search", False,
                       "search returned no results", start)
    return _result("tool-search", True,
                   f"web search returned {len(results)} result(s)",
                   start)


async def check_tool_fetch_content(agent) -> CheckResult:
    """fetch_content retrieves a known page — the network fetch path.

    A successful fetch carries the page's known text; there is no
    ``status`` key on the success shape, so the content is the verdict.
    """
    start = time.monotonic()
    if early := _reg_skip(agent, "tool-fetch-content", ("fetch_content",),
                          start):
        return early
    out = _tool_json_safe(
        await _call_tool(agent.tool_registry, "fetch_content",
                         url="https://example.com", extract_media=False),
        "fetch_content")
    if "error" in out:
        return _result("tool-fetch-content", False,
                       f"fetch_content failed: {_short(str(out))}", start)
    if "Example Domain" not in str(out.get("content") or ""):
        return _result("tool-fetch-content", False,
                       f"the page's known text did not come back: "
                       f"{_short(str(out.get('content')))}", start)
    return _result("tool-fetch-content", True,
                   "fetched example.com and found its known text", start)


def _keyed_skip(name: str, env: tuple, start: float) -> CheckResult | None:
    """A tool whose probe needs a credential skips cleanly without one.

    The same env names the tool's server checks (see the weather and GitHub
    registrations in src/mcp/servers/tasks/tools/mcp_server.py), so a key
    present for the server is present for the probe.
    """
    if any(os.environ.get(v) for v in env):
        return None
    return _result(name, False,
                   f"no {' or '.join(env)} set — probe cannot run", start,
                   skip=True)


async def check_tool_weather(agent) -> CheckResult:
    """get_weather answers for a known city — needs OPENWEATHER_API_KEY."""
    start = time.monotonic()
    if early := _reg_skip(agent, "tool-weather", ("get_weather",), start):
        return early
    if early := _keyed_skip("tool-weather",
                            ("OPENWEATHER_API_KEY", "OPENWEATHERMAP_API_KEY"),
                            start):
        return early
    out = _tool_json_safe(
        await _call_tool(agent.tool_registry, "get_weather",
                         place="Tokyo, Japan"),
        "get_weather")
    if "error" in out:
        return _result("tool-weather", False,
                       f"get_weather failed: {_short(str(out))}", start)
    # The success shape has no status key — the current conditions are the
    # verdict.  The description is the one field every provider fills.
    if not (out.get("current") or {}).get("description"):
        return _result("tool-weather", False,
                       f"no current conditions in the reply: "
                       f"{_short(str(out))}", start)
    return _result("tool-weather", True,
                   f"conditions for {out.get('location', 'Tokyo')} came back",
                   start)


async def check_tool_github(agent) -> CheckResult:
    """github_repo lists the token's repositories — the read-only action."""
    start = time.monotonic()
    if early := _reg_skip(agent, "tool-github", ("github_repo",), start):
        return early
    if early := _keyed_skip("tool-github", ("GITHUB_TOKEN",), start):
        return early
    out = _tool_json_safe(
        await _call_tool(agent.tool_registry, "github_repo", action="list",
                         per_page=1),
        "github_repo")
    if out.get("status") != "ok":
        return _result("tool-github", False,
                       f"github_repo list failed: {_short(str(out))}", start)
    return _result("tool-github", True,
                   f"listed {out.get('count', '?')} repo(s) for the token",
                   start)


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


async def check_model_file_turn(agent) -> CheckResult:
    """The read half of the tool loop: the model reads a file and reports it.

    The harness plants a probe file through the write_file tool — the same
    path a real task's documents ride — then asks the model to read_file it
    and echo the contents.  ``check_model_tool_turn`` exercises the bash
    half of the loop; this exercises the half every document task rides.
    """
    start = time.monotonic()
    reg = getattr(agent, "tool_registry", None)
    if reg is None or not {"write_file", "read_file"} <= reg.tools:
        return _result("model-file-turn", False,
                       "needs the file tools", start, skip=True)
    lb = getattr(agent, "load_balancer", None)
    ep = lb.assigned(getattr(agent, "session_id", None)) if lb is not None else None
    if ep is None:
        return _result("model-file-turn", False, "no endpoint assigned", start)
    from ..model.serving.chat import chat
    data_path = getattr(agent, "data_path", "") or ""
    if not data_path:
        return _result("model-file-turn", False,
                       "no data_path this session — nowhere to plant the probe",
                       start, skip=True)
    probe = f"doctor-read-{os.getpid()}.txt"
    marker = f"DOCTOR-READ-{os.getpid()}-{int(time.time())}"
    try:
        w = _tool_json_safe(
            await _call_tool(reg, "write_file", path=probe, content=marker,
                             data_path=data_path),
            "write_file")
        if w.get("status") != "success":
            return _result("model-file-turn", False,
                           f"planting the probe file failed: "
                           f"{_short(str(w))}", start)
        out = await chat(
            host=ep.host, host_key=ep.host_key, model=ep.model,
            instruction=FILE_TURN_INSTRUCTION.format(
                path=os.path.join(data_path, probe)),
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
        if out and marker in out:
            return _result("model-file-turn", True,
                           "model read the probe file and echoed its contents",
                           start)
        return _result("model-file-turn", False,
                       "the file-reading turn did not land the probe "
                       f"contents — got: {_short(out or 'nothing')}", start)
    finally:
        try:
            os.remove(os.path.join(data_path, probe))
        except OSError:
            pass


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
    ("tool-grep", check_tool_grep, TOOL_CALL_TIMEOUT_S),
    ("tool-search-document", check_tool_search_document, TOOL_CALL_TIMEOUT_S),
    ("tool-local-search", check_tool_index_and_search, RAG_TIMEOUT_S),
    ("tool-send-file", check_tool_send_file, TOOL_CALL_TIMEOUT_S),
    ("tool-serve", check_tool_serve, SERVE_TIMEOUT_S),
    ("tool-search", check_tool_search, WEB_TIMEOUT_S),
    ("tool-fetch-content", check_tool_fetch_content, WEB_TIMEOUT_S),
    ("tool-weather", check_tool_weather, KEYED_TIMEOUT_S),
    ("tool-github", check_tool_github, KEYED_TIMEOUT_S),
    ("harness-tools", check_harness_tools, FAST_TIMEOUT_S),
    ("prompts", check_prompts, FAST_TIMEOUT_S),
    ("endpoint", check_endpoint, FAST_TIMEOUT_S),
    ("session-history", check_session_history, FAST_TIMEOUT_S),
    ("learn", check_learn, FAST_TIMEOUT_S),
)

DEEP_CHECKS = (
    ("model-reply", check_model_reply, MODEL_REPLY_TIMEOUT_S),
    ("model-tool-turn", check_model_tool_turn, MODEL_TOOL_TURN_TIMEOUT_S),
    ("model-file-turn", check_model_file_turn, MODEL_FILE_TURN_TIMEOUT_S),
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
            out.append("\\doctor deep also exercises a live model reply, a "
                       "tool-calling turn, and a file-reading turn "
                       "(costs tokens).")
    elif skipped:
        out += ["", "All checks that ran passed. Skipped ones could not run "
                "here — see their lines above."]
    else:
        out += ["", "Nothing is broken."]
    return "\n".join(out)