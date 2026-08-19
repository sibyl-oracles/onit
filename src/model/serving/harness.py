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

The harness's own tools — the ones the model can call to see and manage the
context it is running in.

Everything else OnIt offers is an MCP tool: a schema discovered from a server,
dispatched over SSE, answered by a process that knows nothing about this run.
These three cannot be, because their answers *are* this run — how full the
context window is, how many turns have passed, what was written down two
compactions ago.  So they live here and are dispatched inside ``_execute_tool``
ahead of the registry lookup, the same way ``sandbox_download_file`` is.

The gap they close: the loop knows it is at 85% of the window and compacts
without warning, and the model — the only party that knows which finding is
worth keeping — is never told.  ``context_status`` is the reading; ``note_write``
is somewhere to put things before the summarizer eats them; ``note_read`` gets
them back.

**Scope is deliberate and narrow.**  Notes are session state: they live under
``data_path``, which is the session isolation boundary, and they die with it.
Cross-session learning is a different thing with different rules — see
``SELF_IMPROVEMENT.md``, Loops A and B — and the path from one to the other is
that document's offline, gated Reflector, never a shared directory.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

from .results import DEFAULT_READ_CHARS, MAX_READ_CHARS, ResultStore
from .interpreter import (DEFAULT_CODE_TIMEOUT, DEFAULT_TOOL_TIMEOUT,
                          INJECTED_PARAMS, evict_stale, get_interpreter)

try:
    from ...lib.schema import coerce_arguments, validate_arguments
except ImportError:  # imported with src/ itself on sys.path (tests, scripts)
    from lib.schema import coerce_arguments, validate_arguments

logger = logging.getLogger(__name__)

# Notes live beside the session's files rather than in it: a task that lists
# its working directory should not find the model's own scratchpad mixed in
# with the artifacts it produced.
NOTES_SUBDIR = os.path.join(".onit", "notes")
NOTE_SUFFIX = ".md"

# A note is the model's own conclusion, not a place to park a tool result.
# The cap is what enforces that: 8k characters is several paragraphs of
# findings and nowhere near a file dump, and a write over it is refused with
# an explanation rather than silently truncated — a memory store that quietly
# drops the end of what it was given is worse than one that says no.
MAX_NOTE_CHARS = 8000
# Distinct keys per session.  Overwrites are always allowed; this bounds only
# how many different things can be kept at once, so a model looping on
# note_write("finding_1"), note_write("finding_2")… cannot fill the disk.
MAX_NOTES = 64
# Conservative on purpose: this string becomes a filename under data_path, and
# the only safe answer to "which characters are fine in a path component" is a
# short allow-list.  Traversal (``..``, ``/``, ``\``) fails to match, and the
# realpath check in ``_note_path`` is the second line rather than the first.
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_EMPTY_OBJECT = {"type": "object", "properties": {}, "required": [],
                 "additionalProperties": False}

_NOTE_KEY_SCHEMA = {
    "type": "string",
    "description": "Short identifier, letters/digits/._- only (e.g. 'findings', "
                   "'remaining_work').",
}

# Schemas are written the way an MCP server's would be discovered — ``required``
# and ``additionalProperties`` present — because the same validator runs against
# both.  See ``HarnessTools.dispatch``.
HARNESS_TOOL_ITEMS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "context_status",
            "description": (
                "How full your context window is, what this run has done so far, "
                "and which notes you hold. Takes no arguments."
            ),
            "parameters": dict(_EMPTY_OBJECT),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note_write",
            "description": (
                "Save a short note that survives context summarization. Writing a "
                "key again replaces what was there."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": _NOTE_KEY_SCHEMA,
                    "text": {"type": "string",
                             "description": f"The note, up to {MAX_NOTE_CHARS} characters."},
                },
                "required": ["key", "text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note_read",
            "description": (
                "Read back a note saved with note_write. context_status lists the keys."
            ),
            "parameters": {
                "type": "object",
                "properties": {"key": _NOTE_KEY_SCHEMA},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    },
]

_RESULT_HANDLE_SCHEMA = {
    "type": "string",
    "description": "The handle from a [result:NNNN …] line, e.g. '0007'.",
}

HARNESS_TOOL_ITEMS += [
    {
        "type": "function",
        "function": {
            "name": "result_read",
            "description": (
                "Read a window of a large tool result that was stored instead of "
                "being pasted in full. The [result:NNNN …] line above a truncated "
                "result carries its handle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "handle": _RESULT_HANDLE_SCHEMA,
                    "offset": {"type": "integer", "minimum": 0,
                               "description": "Character to start from. Default 0."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_READ_CHARS,
                              "description": f"Characters to return, up to "
                                             f"{MAX_READ_CHARS}. Default {DEFAULT_READ_CHARS}."},
                },
                "required": ["handle"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "result_grep",
            "description": (
                "Find lines matching a pattern in a stored tool result, with "
                "surrounding context. Faster than paging through it with result_read "
                "when you know what you are looking for."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "handle": _RESULT_HANDLE_SCHEMA,
                    "pattern": {"type": "string",
                                "description": "Regular expression, or a plain string."},
                    "context": {"type": "integer", "minimum": 0, "maximum": 10,
                                "description": "Lines to show around each match. Default 3."},
                },
                "required": ["handle", "pattern"],
                "additionalProperties": False,
            },
        },
    },
]

HARNESS_TOOL_ITEMS += [
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": (
                "Run Python in a session that keeps its variables between calls. "
                "Every tool you have is available as a function with the same name. "
                "Use it when a task is several dependent steps — search, filter, read "
                "each, combine — so they run as one block instead of one turn each. "
                "Only what you print comes back."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string",
                             "description": "Python source. print() what you need to see."},
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
]

_SPEC_BY_NAME = {item["function"]["name"]: item for item in HARNESS_TOOL_ITEMS}

# context_status needs nothing but the loop's own counters; the note tools need
# somewhere on disk to write.  A run without a data_path gets the first and not
# the other two — offering a tool that can only fail is worse than not offering it.
ALWAYS_AVAILABLE = ("context_status",)
NEEDS_DATA_PATH = ("note_write", "note_read", "result_read", "result_grep")
# Off unless a deployment asks for it: this one runs model-written Python.
NEEDS_CODE_EXECUTION = ("run_code",)

# Appended to the compacted conversation so the model learns what just happened
# to its context.  The UI already gets ``show_context_compaction``; this is the
# model-facing twin, and without it compaction is an event only the human sees.
COMPACTION_NOTICE = (
    "[The conversation above was summarized to free context. Anything not in "
    "the summary is gone. Notes you saved with note_write are unaffected, and "
    "so are stored tool results — call context_status for the note keys and "
    "the result handles, note_read or result_read to get one back, and "
    "note_write now for anything you still need later.]"
)


def _error(message: str) -> str:
    """A refusal in the shape the tool-call error path already uses."""
    return f"Error: {message}"


class HarnessTools:
    """The harness tools, and the live run state their answers are drawn from.

    Owned by ``chat()`` and mutated in place through :meth:`observe` — the
    pattern ``TurnMetrics`` already uses with its metrics sink, rather than a
    second one invented for this.  The loop keeps the counters current; this
    object only reports them.

    Not a home for run state generally.  ``HARNESS_CAPABILITIES.md`` Phase 6
    gives ``iteration_count``, ``tool_call_history`` and the continuation
    budgets a typed ``RunState``; what is here is the subset the model can ask
    about, and it should fold into that object when it lands rather than
    growing into a parallel copy of it.
    """

    def __init__(self, data_path: str = "", max_context_tokens: int | None = None,
                 enabled: bool = True, result_store: bool = True,
                 code_execution: bool = False, session_id: str = "",
                 tool_registry=None, tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
                 code_timeout: float = DEFAULT_CODE_TIMEOUT):
        self.enabled = bool(enabled)
        self.data_path = data_path or ""
        self.max_context_tokens = max_context_tokens
        # Code as action (interpreter.py).  Held here for the same reason the
        # result store is: ``_execute_tool`` already receives this object, and
        # threading a third one through four signatures to reach the same
        # place is churn without a reader.
        self.session_id = session_id or ""
        self.tool_registry = tool_registry
        self.code_execution = bool(enabled) and bool(code_execution)
        self.tool_timeout = float(tool_timeout)
        self.code_timeout = float(code_timeout)
        # Large tool results, kept on disk and reached by handle (results.py).
        # Owned here rather than beside here so the loop threads one object:
        # ``_execute_tool`` already receives this one, and it is the function
        # that has a result to store.
        self.results = ResultStore(self.data_path,
                                   enabled=self.enabled and bool(result_store))
        # Run state, kept current by the loop.
        self.prompt_tokens = 0
        self.turns = 0
        self.tools_called = 0
        self.compactions = 0

    # ── availability ────────────────────────────────────────────────────────

    @property
    def names(self) -> tuple[str, ...]:
        """The tools this run actually offers.

        The result tools need the store switched on as well as a data_path:
        with ``serving.result_store: false`` nothing is ever stored, so a
        handle could never resolve and the two schemas would be paid for on
        every request to buy nothing.
        """
        if not self.enabled:
            return ()
        if not self.data_path:
            # run_code needs somewhere to run, not somewhere to write: a
            # session with no data_path still benefits from doing five steps
            # in one turn.
            return ALWAYS_AVAILABLE + (NEEDS_CODE_EXECUTION if self.code_execution else ())
        offered = ALWAYS_AVAILABLE + NEEDS_DATA_PATH
        if not self.results.enabled:
            offered = tuple(n for n in offered if not n.startswith("result_"))
        if self.code_execution:
            offered += NEEDS_CODE_EXECUTION
        return offered

    def handles(self, name: str) -> bool:
        """Whether ``name`` is a harness tool this run offers.

        Exactly the names in :attr:`names`: a ``note_write`` on a run with no
        ``data_path`` is not intercepted, so it reaches the ordinary "tool not
        found" path and the model is told the truth about it.
        """
        return name in self.names

    def tool_items(self) -> list[dict]:
        """Tool records to add to the API payload, in registry shape."""
        return [_SPEC_BY_NAME[name] for name in self.names]

    # ── run state ───────────────────────────────────────────────────────────

    def observe(self, *, prompt_tokens: int | None = None,
                max_context_tokens: int | None = None,
                turns: int | None = None,
                tools_called: int | None = None,
                compactions: int | None = None) -> None:
        """Update the counters ``context_status`` reports.

        Every argument is optional and None means "unchanged", so a caller
        updating one number does not have to restate the others.
        """
        if prompt_tokens is not None:
            self.prompt_tokens = int(prompt_tokens)
        if max_context_tokens is not None:
            self.max_context_tokens = max_context_tokens
        if turns is not None:
            self.turns = int(turns)
        if tools_called is not None:
            self.tools_called = int(tools_called)
        if compactions is not None:
            self.compactions = int(compactions)

    # ── notes on disk ───────────────────────────────────────────────────────

    @property
    def notes_dir(self) -> Path:
        return Path(self.data_path) / NOTES_SUBDIR

    def _note_path(self, key: str) -> Path | None:
        """The file for ``key``, or None if it would land outside the jail.

        ``_KEY_RE`` has already rejected every key that could escape, so this
        check is redundant by design: ``data_path`` is a session isolation
        boundary, and a boundary with one check has one bug between it and the
        next session's files.
        """
        root = self.notes_dir
        candidate = root / f"{key}{NOTE_SUFFIX}"
        try:
            resolved_root = root.resolve()
            resolved = candidate.resolve()
            resolved.relative_to(resolved_root)
        except (ValueError, OSError):
            return None
        return candidate

    def note_keys(self) -> list[str]:
        """Every key saved this session, sorted. Empty when nothing is saved."""
        if not self.data_path:
            return []
        try:
            return sorted(p.stem for p in self.notes_dir.glob(f"*{NOTE_SUFFIX}")
                          if p.is_file())
        except OSError:
            return []

    # ── the tools ───────────────────────────────────────────────────────────

    def context_status(self) -> str:
        used = self.prompt_tokens
        limit = self.max_context_tokens
        status: dict = {
            "used_tokens": used if used > 0 else None,
            "max_tokens": limit or None,
            "pct_used": round(used / limit * 100) if used > 0 and limit else None,
            "turns_taken": self.turns,
            "tool_calls_made": self.tools_called,
            "compactions": self.compactions,
            "notes_saved": self.note_keys(),
            # Same reason notes_saved is here rather than a note_list tool: a
            # model that needs a handle after a compaction should not spend a
            # turn discovering that handles exist.
            "results_stored": self.results.stored(),
        }
        # The numbers come from the previous API response, so there is nothing
        # to report on the opening turn or on the turn straight after a
        # compaction.  Say so: a model handed ``used_tokens: null`` with no
        # explanation reads it as an error and calls the tool again.
        if status["used_tokens"] is None:
            status["detail"] = ("No token count yet — it is measured from the last "
                                "API response, and appears from the next turn onward.")
        elif status["max_tokens"] is None:
            status["detail"] = ("This provider does not report a context window size, "
                                "so only the used count is known.")
        elif status["pct_used"] >= 70:
            status["detail"] = ("Context is filling. Save anything you still need with "
                                "note_write now — a summarization will drop the detail.")
        return json.dumps(status, indent=2)

    def note_write(self, key: str, text: str) -> str:
        if not isinstance(key, str) or not _KEY_RE.match(key):
            return _error(
                f"'{key}' is not a usable note key. Use 1-64 characters, letters "
                "or digits to start, then letters, digits, '.', '_' or '-'.")
        if not isinstance(text, str):
            return _error(f"note '{key}' must be given as text.")
        if len(text) > MAX_NOTE_CHARS:
            return _error(
                f"note '{key}' is {len(text):,} characters, over the {MAX_NOTE_CHARS:,} "
                "limit. It was not saved. A note is for your own conclusions, not a "
                "copy of a tool result — write the summary, or split it across keys.")
        path = self._note_path(key)
        if path is None:
            return _error(f"'{key}' does not resolve to a path inside this session.")
        existing = self.note_keys()
        if key not in existing and len(existing) >= MAX_NOTES:
            return _error(
                f"this session already holds {MAX_NOTES} notes, the limit. Overwrite "
                f"one of the existing keys instead: {', '.join(existing)}.")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as e:
            logger.warning("note_write(%s) failed: %s", key, e)
            return _error(f"could not save note '{key}': {e}")
        return json.dumps({
            "status": "saved",
            "key": key,
            "chars": len(text),
            "replaced": key in existing,
            "notes_saved": self.note_keys(),
        }, indent=2)

    def note_read(self, key: str) -> str:
        if not isinstance(key, str) or not _KEY_RE.match(key):
            return _error(f"'{key}' is not a usable note key.")
        path = self._note_path(key)
        if path is None:
            return _error(f"'{key}' does not resolve to a path inside this session.")
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Name what does exist.  The same reasoning as the argument
            # validator's "here are the parameters this tool takes": a model
            # that guessed a key can fix the guess from this, where a bare
            # "not found" costs another turn to discover the same thing.
            saved = self.note_keys()
            available = ", ".join(saved) if saved else "none saved yet"
            return _error(f"no note saved under '{key}'. Saved keys: {available}.")
        except OSError as e:
            logger.warning("note_read(%s) failed: %s", key, e)
            return _error(f"could not read note '{key}': {e}")

    # ── code as action ──────────────────────────────────────────────────────

    async def _run_tool(self, name: str, kwargs: dict) -> str:
        """One tool call made from inside the interpreter, run out here.

        This function is the trust boundary.  The child asks for a tool by
        name; the *parent* decides what that call actually runs with, so
        ``session_id`` and ``data_path`` are stripped from whatever arrived and
        re-bound from the harness's own values.  They are absent from the
        generated signatures too, so there is normally nothing to strip — this
        is the second check, for the same reason ``_note_path`` resolves a key
        that the regex has already made safe.
        """
        registry = self.tool_registry
        if not registry or name not in getattr(registry, "tools", ()):
            raise LookupError(f"no tool named {name!r} — call context_status to see "
                              "what this run has")
        kwargs = {k: v for k, v in (kwargs or {}).items() if k not in INJECTED_PARAMS}
        if self.session_id and registry.tool_accepts_param(name, "session_id"):
            kwargs["session_id"] = self.session_id
        if self.data_path and registry.tool_accepts_param(name, "data_path"):
            kwargs["data_path"] = self.data_path
        handler = registry[name]
        if handler is None:
            raise LookupError(f"no handler registered for {name!r}")
        result = await handler(**kwargs)
        return "" if result is None else str(result)

    @property
    def interpreter(self):
        """This session's interpreter.  Started on first use, not here."""
        return get_interpreter(
            self.session_id,
            data_path=self.data_path,
            tool_items=(self.tool_registry.get_tool_items()
                        if self.tool_registry else []),
            dispatch=self._run_tool,
            tool_timeout=self.tool_timeout,
        )

    async def run_code(self, code: str) -> str:
        """Execute ``code`` and return what it printed.

        The result goes through the store like any other large tool output, so
        a print that turns out to be a whole file is addressed rather than
        pasted — which is why Phase 4 came first.
        """
        if not isinstance(code, str) or not code.strip():
            return _error("run_code needs Python source in `code`.")
        interpreter = self.interpreter
        await evict_stale()
        started = time.monotonic()
        outcome = await interpreter.run(code, timeout=self.code_timeout)
        elapsed = time.monotonic() - started

        stdout = outcome.get("stdout") or ""
        status = "ok" if outcome.get("ok") else (
            "timed out" if outcome.get("timed_out") else "raised")
        body = f"[run_code · {status} · {elapsed:.1f}s]"
        if stdout:
            body += "\n" + stdout.rstrip()
        elif outcome.get("ok"):
            # An empty result reads as a failure, and the model re-runs the
            # same block to find out what happened.  Say what actually did.
            body += ("\nThe code ran and printed nothing. Only what you print comes "
                     "back — add print() for the values you need to see. Variables "
                     "you set are still there for the next call.")
        if outcome.get("error"):
            body += "\n" + str(outcome["error"]).rstrip()
        stored = self.results.put("run_code", body)
        return stored if stored is not None else body

    # ── dispatch ────────────────────────────────────────────────────────────

    def _prepare(self, name: str, arguments: dict):
        """Coerced arguments for a harness call, or an error string.

        Split out of ``dispatch`` so the async tool below is checked the same
        way: these tools bypass ``_execute_tool``'s validation by intercepting
        ahead of it, and one of them skipping it too would make it the single
        place in the harness where a wrong-typed argument reaches the
        implementation.
        """
        arguments = dict(arguments or {})
        schema = _SPEC_BY_NAME[name]["function"]["parameters"]
        coerced, _ = coerce_arguments(schema, arguments)
        arguments.update(coerced)
        problems = validate_arguments(schema, arguments)
        if problems:
            return _error(f"{name} was called with arguments that do not match its "
                          f"schema: {'; '.join(problems)}. The call was not run.")
        return arguments

    async def adispatch(self, name: str, arguments: dict) -> str | None:
        """Every harness tool, including the one that has to be awaited.

        ``run_code`` drives a child process and answers tool calls from it, so
        it cannot be synchronous.  Everything else still is, and goes through
        :meth:`dispatch` unchanged.
        """
        if name != "run_code" or not self.handles(name):
            return self.dispatch(name, arguments)
        prepared = self._prepare(name, arguments)
        if isinstance(prepared, str):
            return prepared
        try:
            return await self.run_code(prepared["code"])
        except Exception as e:  # an interpreter that will not start is a tool error
            logger.warning("run_code failed: %s", e)
            return _error(f"the code could not be run: {e}")

    def dispatch(self, name: str, arguments: dict) -> str | None:
        """Run a harness tool and return its result, or None if ``name`` is not one."""
        if not self.handles(name):
            return None
        if name == "run_code":
            # Reached only by a caller that has not moved to adispatch.
            return _error("run_code must be dispatched asynchronously.")
        prepared = self._prepare(name, arguments)
        if isinstance(prepared, str):
            return prepared
        arguments = prepared
        if name == "context_status":
            return self.context_status()
        if name == "note_write":
            return self.note_write(arguments["key"], arguments["text"])
        if name == "result_read":
            return self.results.read(arguments["handle"],
                                     arguments.get("offset", 0),
                                     arguments.get("limit", DEFAULT_READ_CHARS))
        if name == "result_grep":
            return self.results.grep(arguments["handle"], arguments["pattern"],
                                     arguments.get("context", 3))
        return self.note_read(arguments["key"])
