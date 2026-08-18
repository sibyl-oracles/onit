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
from pathlib import Path

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

_SPEC_BY_NAME = {item["function"]["name"]: item for item in HARNESS_TOOL_ITEMS}

# context_status needs nothing but the loop's own counters; the note tools need
# somewhere on disk to write.  A run without a data_path gets the first and not
# the other two — offering a tool that can only fail is worse than not offering it.
ALWAYS_AVAILABLE = ("context_status",)
NEEDS_DATA_PATH = ("note_write", "note_read")

# Appended to the compacted conversation so the model learns what just happened
# to its context.  The UI already gets ``show_context_compaction``; this is the
# model-facing twin, and without it compaction is an event only the human sees.
COMPACTION_NOTICE = (
    "[The conversation above was summarized to free context. Anything not in "
    "the summary is gone. Notes you saved with note_write are unaffected — "
    "call context_status for the keys, note_read to get one back, and "
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
                 enabled: bool = True):
        self.enabled = bool(enabled)
        self.data_path = data_path or ""
        self.max_context_tokens = max_context_tokens
        # Run state, kept current by the loop.
        self.prompt_tokens = 0
        self.turns = 0
        self.tools_called = 0
        self.compactions = 0

    # ── availability ────────────────────────────────────────────────────────

    @property
    def names(self) -> tuple[str, ...]:
        """The tools this run actually offers."""
        if not self.enabled:
            return ()
        if not self.data_path:
            return ALWAYS_AVAILABLE
        return ALWAYS_AVAILABLE + NEEDS_DATA_PATH

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

    # ── dispatch ────────────────────────────────────────────────────────────

    def dispatch(self, name: str, arguments: dict) -> str | None:
        """Run a harness tool and return its result, or None if ``name`` is not one.

        Arguments are coerced and validated here against the same schemas and
        with the same validator the MCP path uses — these tools bypass
        ``_execute_tool``'s checks by intercepting ahead of them, and skipping
        validation along with the dispatch would make them the one place in the
        harness where a wrong-typed argument reaches the implementation.
        """
        if not self.handles(name):
            return None
        arguments = dict(arguments or {})
        schema = _SPEC_BY_NAME[name]["function"]["parameters"]
        coerced, _ = coerce_arguments(schema, arguments)
        arguments.update(coerced)
        problems = validate_arguments(schema, arguments)
        if problems:
            return _error(f"{name} was called with arguments that do not match its "
                          f"schema: {'; '.join(problems)}. The call was not run.")
        if name == "context_status":
            return self.context_status()
        if name == "note_write":
            return self.note_write(arguments["key"], arguments["text"])
        return self.note_read(arguments["key"])
