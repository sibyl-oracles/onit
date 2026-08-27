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

Code as action — a persistent Python interpreter the model can drive.

Every step of a multi-step operation costs a full model round trip.  The
parallel path helps only when every call in the batch is read-only; anything
containing a write is strictly sequential, correctly so — a batch of those is a
script, and a script has an order.  The answer is to let the model write the
script:

    hits = local_search("Q3 revenue")[:3]
    totals = {h["title"]: extract_tables(h["path"]) for h in hits}
    print(totals)

One block instead of six turns, each of which would have paid prefill on a
context that grew since the last one.  Only what the code ``print``s enters the
conversation; everything else stays as live objects in the interpreter's
namespace, which is pass-by-reference bought without giving up isolation.

**Where the code actually runs.**  In a child process, never in the harness.
The namespace lives there across calls, so `x` defined in one ``run_code`` is
still `x` in the next; a wedged one is killed and the session recovers with a
fresh namespace rather than a hung loop.  Under ``onit --container`` that child
is inside the container, which is the isolation the deployment already bought.

**Tool calls go the other way.**  A generated binding in the child does not
reach an MCP server — it sends a request up the pipe, and the *parent* runs the
real ``ToolHandler``.  That is what keeps ``session_id`` and ``data_path``
harness-owned: they are bound in the parent on every call and stripped from
anything the child sends, so no amount of cleverness inside the interpreter can
reach another session's files.  They are dropped from the generated signatures
entirely, so there is nothing to override in the first place.

**Off by default.**  This runs model-written Python with the privileges of the
OnIt process, which is a different posture from the bash tool's AST allowlist
and path jail.  ``serving.code_execution: true`` is a deployment's explicit
choice, and small models write worse Python than they write JSON tool calls —
see HARNESS_CAPABILITIES.md §7.3 step 5 before making it a default anywhere.
"""

import asyncio
import json
import keyword
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

# One exec, from the request to the "done" line.  Tool calls made from inside
# the code count against it: the model asked for a script, and the script's
# wall time is what the user waits for.
DEFAULT_CODE_TIMEOUT = 120.0

# What one tool call inside the interpreter may take.  Separate from the budget
# above so a single slow tool is reported as a slow tool rather than surfacing
# as the whole block timing out with nothing to show for it.
DEFAULT_TOOL_TIMEOUT = 120.0

# Printed output is capped in the child, before it is ever sent up the pipe: a
# runaway ``while True: print(x)`` should cost a bounded amount of memory in a
# child that is about to be killed, not an unbounded amount in the parent.
# What survives the cap still goes through the result store, so this is a
# backstop and not the thing that keeps output out of the context.
MAX_STDOUT_CHARS = 400_000

# Interpreters alive at once, across every session this process serves.  A web
# deployment has as many sessions as it has tabs, and each one holding a Python
# process is how a machine runs out of them.  Least recently used is evicted;
# its session gets a fresh namespace next time, which is the same thing that
# happens after a timeout.
MAX_LIVE_INTERPRETERS = 8

# Parameters the harness owns.  Never accepted from inside the interpreter, and
# never present in a generated signature — see the module docstring.
# Parameters the parent binds on every call, whatever the child asked for.
#
# The approval pair is here for a sharper reason than the other two. A tool
# result that needs a person's approval carries the ticket id — the model can
# read it, because it is in the payload. If code could then pass that id back
# as ``approval_token``, the model would be approving its own commands and the
# prompt would be decoration. Stripping them at the boundary means a ticket
# minted for a call from inside the interpreter can only ever be redeemed by
# the harness, which is the thing that asks the human.
APPROVAL_PARAMS = ("approval_token", "approval_scope")
INJECTED_PARAMS = ("session_id", "data_path") + APPROVAL_PARAMS

# Anything the bindings would shadow or that would not survive being written as
# a Python identifier.
_RESERVED = frozenset(keyword.kwlist) | {"call_tool", "ToolError", "print"}


def _as_data(value):
    """A tool's answer as something code can index, where it is one.

    MCP tools answer with a string, and most of them answer with JSON in it.
    Handing that back as text means the model writes ``json.loads`` around
    every call, which is a line of boilerplate per call and a whole failed turn
    when it forgets.  Only objects and arrays are parsed: a tool that answers
    ``"42"`` means the string, and quietly turning it into an int is the kind
    of helpfulness that shows up later as a type error.
    """
    if not isinstance(value, str):
        return value
    stripped = value.lstrip()
    if stripped[:1] not in ("{", "["):
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return value


def _is_identifier(name) -> bool:
    return (isinstance(name, str) and name.isidentifier()
            and name not in _RESERVED and not name.startswith("_"))


def bindings_for(tool_items: list) -> list[dict]:
    """Python signatures for the registry's tools, from their schemas alone.

    Schema-driven so a newly discovered MCP server is callable from code with
    no extra work here.  A tool whose name is not a Python identifier is
    skipped rather than mangled — it is still reachable through ``call_tool``,
    and a mangled name is one the model cannot guess from the tool list.
    """
    bindings = []
    for item in tool_items or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not _is_identifier(name):
            continue
        schema = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        params = []
        for pname in props:
            # The injected ones are bound in the parent on every call.  Leaving
            # them in the signature would advertise an override that silently
            # does nothing, which is worse than not offering it.
            if pname in INJECTED_PARAMS or not _is_identifier(pname):
                continue
            params.append({"name": pname, "required": pname in required})
        # Required first: Python will not accept a non-default parameter after
        # a defaulted one, and a schema is under no obligation to order them.
        params.sort(key=lambda p: not p["required"])
        bindings.append({
            "name": name,
            "params": params,
            "doc": (fn.get("description") or "")[:400],
        })
    return bindings


# ── the child process ───────────────────────────────────────────────────────
#
# Self-contained by construction: it is started with ``python -c`` and imports
# nothing from this repository, so nothing it can reach is worth reaching.
# Protocol is line-delimited JSON on the real stdout, which is taken away from
# user code at the first opportunity — ``print`` writes to a capture buffer
# from the moment this starts, so no ordinary output can corrupt the stream.

_CHILD_SOURCE = r'''
import ast, io, json, sys, traceback

_OUT = sys.stdout
sys.stdout = io.StringIO()
MAX_STDOUT = %(max_stdout)d


def _send(obj):
    _OUT.write(json.dumps(obj, default=repr) + "\n")
    _OUT.flush()


def _recv():
    line = sys.stdin.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


class ToolError(RuntimeError):
    """A tool call that failed. Catch it like any other exception."""


class Row(dict):
    """A dict that also answers to attribute access, for tool output."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


def _wrap(value):
    if isinstance(value, dict):
        return Row((k, _wrap(v)) for k, v in value.items())
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


_seq = [0]


def call_tool(_name, **kwargs):
    """Call any registered tool by name, including ones with awkward names."""
    _seq[0] += 1
    ident = _seq[0]
    _send({"t": "call", "id": ident, "tool": _name, "kwargs": kwargs})
    while True:
        msg = _recv()
        if msg.get("t") == "reply" and msg.get("id") == ident:
            if not msg.get("ok"):
                raise ToolError(msg.get("error") or "tool call failed")
            return _wrap(msg.get("value"))


NS = {"__name__": "__main__", "__builtins__": __builtins__,
      "call_tool": call_tool, "ToolError": ToolError, "Row": Row}


# Accepted and thrown away, rather than raising like any other keyword the
# signature does not have. A model that has seen a ``needs_approval`` payload
# knows an ``approval_token`` exists, and writing one into a call from code
# was a TypeError from a function that never declared it — a hard failure,
# for a value the parent discards anyway, and one the model answers by giving
# up on the command and trying a smaller one.
#
# Only this pair. ``data_path`` and ``session_id`` are the isolation boundary,
# and code reaching for those should hear about it: they stay a TypeError, as
# does every actual typo.
_DISCARDED = %(approval_params)r


def _bind(specs):
    for spec in specs:
        args = []
        for p in spec["params"]:
            args.append(p["name"] if p["required"] else p["name"] + "=None")
        args.append("**_harness")
        names = [p["name"] for p in spec["params"]]
        # Optional parameters left at None are dropped rather than sent: a
        # server that distinguishes "absent" from "null" should see absent.
        # _DISCARDED is written into the source rather than read from the
        # module: these run with NS as their globals, and NS is swept of
        # everything that is not a callable between statements.
        src = (
            "def %%s(%%s):\n"
            "    _extra = [k for k in _harness if k not in %%r]\n"
            "    if _extra:\n"
            "        raise TypeError(%%r + '() got an unexpected keyword '\n"
            "                        'argument %%%%r' %%%% _extra[0])\n"
            "    _a = {}\n"
            "%%s"
            "    return call_tool(%%r, **_a)\n"
        ) %% (spec["name"], ", ".join(args), _DISCARDED, spec["name"],
             "".join("    if %%s is not None: _a[%%r] = %%s\n" %% (n, n, n)
                     for n in names),
             spec["name"])
        try:
            exec(compile(src, "<bindings>", "exec"), NS)
            NS[spec["name"]].__doc__ = spec.get("doc") or ""
        except SyntaxError:
            continue


def _clean_traceback(exc):
    """The model's frames only. Ours are noise it cannot act on."""
    if isinstance(exc, SyntaxError):
        # Raised inside ast.parse, so its traceback is our call stack and not
        # the model's. The exception alone already points at the bad line.
        return "".join(traceback.format_exception_only(type(exc), exc)).strip()
    out = []
    for line in traceback.format_exception(type(exc), exc, exc.__traceback__):
        if 'File "<string>"' in line or 'File "<bindings>"' in line:
            continue
        out.append(line)
    return "".join(out).strip()


def _execute(code):
    buf = io.StringIO()
    sys.stdout = buf
    try:
        try:
            tree = ast.parse(code, "<run_code>", "exec")
        except SyntaxError as e:
            return False, "", _clean_traceback(e)
        # A trailing expression is echoed, the way a REPL would: the model
        # writing `local_search("x")` and seeing nothing is a wasted turn.
        tail = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tail = ast.Expression(tree.body.pop().value)
        try:
            if tree.body:
                exec(compile(tree, "<run_code>", "exec"), NS)
            if tail is not None:
                value = eval(compile(ast.fix_missing_locations(tail),
                                     "<run_code>", "eval"), NS)
                if value is not None:
                    print(repr(value))
            return True, buf.getvalue(), None
        except SystemExit:
            return True, buf.getvalue(), None
        except BaseException as e:
            return False, buf.getvalue(), _clean_traceback(e)
    finally:
        sys.stdout = io.StringIO()


while True:
    try:
        message = _recv()
    except SystemExit:
        break
    except Exception:
        break
    kind = message.get("t")
    if kind == "bind":
        _bind(message.get("specs") or [])
        _send({"t": "bound", "names": sorted(
            k for k in NS if not k.startswith("_") and callable(NS.get(k)))})
    elif kind == "exec":
        ok, out, err = _execute(message.get("code") or "")
        if len(out) > MAX_STDOUT:
            out = out[:MAX_STDOUT] + "\n... [output capped at %%d characters]" %% MAX_STDOUT
        _send({"t": "done", "ok": ok, "stdout": out, "error": err,
               "names": sorted(k for k in NS if not k.startswith("_"))})
    elif kind == "reset":
        for key in [k for k in NS if not k.startswith("__")
                    and k not in ("call_tool", "ToolError", "Row")]:
            if not callable(NS[key]):
                NS.pop(key, None)
        _send({"t": "done", "ok": True, "stdout": "", "error": None, "names": []})
    else:
        break
'''


def child_source(max_stdout: int = MAX_STDOUT_CHARS) -> str:
    return _CHILD_SOURCE % {"max_stdout": int(max_stdout),
                            "approval_params": APPROVAL_PARAMS}


class InterpreterError(RuntimeError):
    """The interpreter could not be started or could not be spoken to."""


class PythonInterpreter:
    """A child Python process, its namespace, and the pipe to it.

    One per session.  Started on first use rather than on construction: a run
    that never calls ``run_code`` should not pay for a process, and most runs
    never call it.
    """

    def __init__(self, session_id: str = "", data_path: str = "",
                 tool_items: list | None = None,
                 dispatch=None,
                 tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
                 max_stdout: int = MAX_STDOUT_CHARS):
        self.session_id = session_id or ""
        self.data_path = data_path or ""
        self.tool_items = list(tool_items or [])
        # ``dispatch(name, kwargs) -> awaitable[str]``.  Supplied by the owner
        # so the parent, not this object and never the child, decides what a
        # tool call actually runs with.
        self.dispatch = dispatch
        self.tool_timeout = float(tool_timeout)
        self.max_stdout = int(max_stdout)
        self._proc = None
        self._lock = asyncio.Lock()
        self.bound_names: list[str] = []
        self.last_used = time.monotonic()
        self.restarts = 0

    # ── lifecycle ───────────────────────────────────────────────────────────

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        """Spawn the child and bind the tools into its namespace."""
        if self.alive:
            return
        cwd = self.data_path if self.data_path and os.path.isdir(self.data_path) else None
        env = dict(os.environ)
        # Unbuffered, so a line written is a line the parent can read; and the
        # session's own directory is where relative paths land.
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if self.data_path:
            env["ONIT_DATA_PATH"] = self.data_path
        if self.session_id:
            env["ONIT_SESSION_ID"] = self.session_id
        try:
            self._proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-c", child_source(self.max_stdout),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=cwd, env=env,
            )
        except (OSError, ValueError) as e:
            self._proc = None
            raise InterpreterError(f"could not start a Python interpreter: {e}") from e
        await self._send({"t": "bind", "specs": bindings_for(self.tool_items)})
        reply = await self._read_until(("bound",), timeout=30.0)
        self.bound_names = list(reply.get("names") or [])

    async def stop(self) -> None:
        """Kill the child.  Idempotent, and safe on an already-dead one."""
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.kill()
        except (ProcessLookupError, OSError):  # pragma: no cover - already gone
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, Exception):  # pragma: no cover
            pass

    async def restart(self) -> None:
        await self.stop()
        self.restarts += 1
        await self.start()

    # ── the pipe ────────────────────────────────────────────────────────────

    async def _send(self, obj: dict) -> None:
        if not self.alive:
            raise InterpreterError("the interpreter is not running")
        self._proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def _read_until(self, kinds: tuple, timeout: float) -> dict:
        """Read protocol lines, answering tool calls, until one of ``kinds``.

        Unparseable lines are skipped rather than fatal.  Code that spawns a
        subprocess inheriting fd 1 can write into this stream, and losing the
        whole block to someone else's ``echo`` would be a poor trade.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            line = await asyncio.wait_for(self._proc.stdout.readline(),
                                          timeout=remaining)
            if not line:
                raise InterpreterError("the interpreter exited unexpectedly")
            try:
                message = json.loads(line.decode("utf-8", "replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(message, dict):
                continue
            kind = message.get("t")
            if kind in kinds:
                return message
            if kind == "call":
                await self._answer_call(message)
                continue
            # Anything else is a child that has lost the protocol.
            raise InterpreterError(f"unexpected message from the interpreter: {kind!r}")

    async def _answer_call(self, message: dict) -> None:
        """Run one tool call on the child's behalf, in the parent.

        Everything arriving here is model input that has been through an
        interpreter, so the injected parameters are stripped before the owner's
        dispatcher ever sees them — even though that dispatcher strips them
        again.  The generated signatures have nowhere to put them, but
        ``call_tool`` takes ``**kwargs`` and would otherwise be the way around
        the whole scheme.  Two checks, for the same reason a jailed path is
        resolved after a regex has already made it safe.
        """
        name = message.get("tool")
        kwargs = message.get("kwargs")
        kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
        for injected in INJECTED_PARAMS:
            kwargs.pop(injected, None)
        reply = {"t": "reply", "id": message.get("id")}
        try:
            if self.dispatch is None:
                raise InterpreterError("this interpreter has no tools")
            value = await asyncio.wait_for(self.dispatch(name, kwargs),
                                           timeout=self.tool_timeout)
            reply.update(ok=True, value=_as_data(value))
        except asyncio.TimeoutError:
            reply.update(ok=False,
                         error=f"{name} timed out after {self.tool_timeout:g}s")
        except Exception as e:
            # Raised as ToolError inside the code, where it can be caught: a
            # script that loops over ten documents should be able to survive
            # one of them failing.
            reply.update(ok=False, error=f"{name}: {e}")
        await self._send(reply)

    # ── running code ────────────────────────────────────────────────────────

    async def run(self, code: str, timeout: float = DEFAULT_CODE_TIMEOUT) -> dict:
        """Execute ``code`` in the persistent namespace.

        Returns ``{ok, stdout, error, timed_out}``.  Never raises for anything
        the code did: an exception inside it is a result the model can act on,
        not a failure of the harness.
        """
        async with self._lock:
            self.last_used = time.monotonic()
            try:
                await self.start()
            except InterpreterError as e:
                return {"ok": False, "stdout": "", "error": str(e), "timed_out": False}
            try:
                await self._send({"t": "exec", "code": code})
                done = await self._read_until(("done",), timeout=timeout)
            except asyncio.TimeoutError:
                # A wedged interpreter is not recoverable in place — the child
                # is inside the model's own loop and will not answer again.
                await self.stop()
                return {
                    "ok": False, "stdout": "", "timed_out": True,
                    "error": (f"the code did not finish within {timeout:g}s and was "
                              "stopped. The interpreter has been restarted, so "
                              "variables from earlier calls are gone. Break the work "
                              "into smaller pieces, or check for a loop that does "
                              "not end."),
                }
            except InterpreterError as e:
                await self.stop()
                return {"ok": False, "stdout": "", "error": str(e), "timed_out": False}
            return {
                "ok": bool(done.get("ok")),
                "stdout": done.get("stdout") or "",
                "error": done.get("error"),
                "timed_out": False,
                "names": done.get("names") or [],
            }


# ── one interpreter per session, for as long as the session lasts ───────────
#
# Held here rather than on ``HarnessTools`` because that object is built per
# ``chat()`` call — one task — and the namespace is meant to outlive a task the
# way a session does.

_INTERPRETERS: dict = {}


def get_interpreter(session_id: str, **kwargs) -> PythonInterpreter:
    """The interpreter for ``session_id``, started lazily on first use."""
    key = session_id or "default"
    existing = _INTERPRETERS.get(key)
    if existing is not None:
        # Tools and paths can change between tasks (a new MCP server, a
        # per-task data_path); the namespace should not be thrown away for it.
        existing.tool_items = list(kwargs.get("tool_items") or existing.tool_items)
        existing.dispatch = kwargs.get("dispatch") or existing.dispatch
        existing.last_used = time.monotonic()
        return existing
    interpreter = PythonInterpreter(session_id=session_id, **kwargs)
    _INTERPRETERS[key] = interpreter
    return interpreter


async def evict_stale(limit: int = MAX_LIVE_INTERPRETERS) -> None:
    """Stop the least recently used interpreters past ``limit``."""
    while len(_INTERPRETERS) > max(1, limit):
        oldest = min(_INTERPRETERS, key=lambda k: _INTERPRETERS[k].last_used)
        await shutdown_session(oldest)


async def shutdown_session(session_id: str) -> None:
    """Stop and forget the interpreter for one session."""
    interpreter = _INTERPRETERS.pop(session_id or "default", None)
    if interpreter is not None:
        await interpreter.stop()


async def shutdown_all() -> None:
    """Stop every interpreter this process started."""
    for key in list(_INTERPRETERS):
        await shutdown_session(key)
