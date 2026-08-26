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
from urllib.parse import urlparse
from pydantic import BaseModel, ConfigDict, Field
from fastmcp import Client

import logging
import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings:.*")

logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from httpx/httpcore (used by FastMCP client)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from .lib.tools import (discover_tools, register_stdio_servers,
                        apply_default_mcp_servers)
from .lib.text import remove_tags
from .mcp.prompts.prompts import DEFAULT_MAX_DOCUMENTS, build_assistant_instruction
from .lib.files import has_code_files, zip_code_files
from .ui import ChatUI
from .model.serving.chat import chat, decode_rate, summarize_metrics
from .model.serving.state import RunState, state_path_for
from .model.serving.balancer import (DEFAULT_PRIORITY, LoadBalancer,
                                     ServerEndpoint)

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Part, TaskState, TaskStatus, TaskStatusUpdateEvent
from a2a.helpers.proto_helpers import new_text_message

AGENT_CURSOR = "OnIt"
USER_CURSOR = "You"
STOP_TAG = "<stop></stop>"

# ``serving:`` keys forwarded to chat() only when the config sets them, so
# chat()'s own default stays the single place each one is defined.
SERVING_PASSTHROUGH = ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty',
                       'repetition_penalty', 'num_ctx', 'think_tool_turns',
                       # Loop policy — the ceilings on the ways the agent loop
                       # can fail to terminate.  Defaults live in chat().
                       'max_chat_iterations', 'max_repeated_tool_calls',
                       'max_api_retries', 'max_planning_continuations',
                       'max_ack_continuations', 'max_final_continuations',
                       # Fact-checking the finished answer.  Defaults live in
                       # chat(): `verify_answers: false` turns it off,
                       # `verify_timeout_s` caps what the user waits for, and
                       # the rest govern the check that keeps going behind the
                       # answer once it has been handed over.
                       'verify_answers', 'verify_max_tool_turns',
                       'verify_timeout_s', 'verify_background',
                       'verify_trusted_domains',
                       # The harness's own tools — context_status and the note
                       # scratchpad.  Off turns them off everywhere: the prompt
                       # block below is gated on the same key.
                       'harness_tools',
                       # Large tool results kept on disk and passed by handle.
                       # Off restores the old behavior: a hard cut at
                       # MAX_TOOL_RESPONSE with the middle gone for good.
                       'result_store',
                       # Code as action — run_code and its per-session Python
                       # interpreter.  Default off: it runs model-written
                       # Python with this process's privileges, and small
                       # models write worse Python than they write JSON.
                       'code_execution', 'code_timeout')


async def _call_sandbox_stop(tool_registry, session_id: str = "") -> None:
    """Stop what this session had running: its interpreter, and the sandbox.

    The interpreter goes first and unconditionally — it belongs to the session
    rather than to sandbox mode, and a child process left behind by a stopped
    session is one nobody will ever stop.

    A provider that registered ``sandbox_stop`` is the whole signal that there
    is a sandbox to stop; there is no separate flag to disagree with it.
    """
    try:
        from .model.serving.interpreter import shutdown_session
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


# Substring → layman-friendly status line, first match wins. Raw tool log
# output (pip, git, curl …) is too noisy to show users verbatim.
_TOOL_LOG_PHRASES = [
    ("requirement already satisfied", "Checking installed components…"),
    ("successfully installed", "Finishing installation…"),
    ("installing", "Installing components…"),
    ("building wheel", "Installing components…"),
    ("preparing metadata", "Installing components…"),
    ("collecting", "Downloading required files…"),
    ("downloading", "Downloading required files…"),
    ("fetching", "Downloading required files…"),
    ("cloning", "Downloading source code…"),
    ("receiving objects", "Downloading source code…"),
    ("resolving deltas", "Downloading source code…"),
    ("uploading", "Uploading files…"),
]


# Tools whose arguments and raw output are never shown to the client.  A shell
# command and its stdout describe how the agent works, not what the user asked
# for, so the web UI reports that work is happening and nothing more.  The
# phrases above still apply — they are a gist ("Installing components…"), not a
# reflection of the call.
_OPAQUE_TOOLS = ("bash",)
_OPAQUE_STATUS = "Working…"


def friendly_tool_status(name: str, data) -> str:
    """Reduce a raw tool log payload to one short, human-readable line.

    MCP log notifications carry structured data (e.g. ``{'msg': ..., 'extra':
    None}``); unwrap it, keep the first non-empty line, and translate common
    operations into plain language. Returns "" when there is nothing to show.
    """
    if isinstance(data, dict):
        text = str(data.get("msg") or data.get("message") or "")
    else:
        text = str(data or "")
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not line:
        return ""
    low = line.lower()
    for needle, phrase in _TOOL_LOG_PHRASES:
        if needle in low:
            return phrase
    if name in _OPAQUE_TOOLS:
        # Nothing recognizable to summarize, and the raw line is command
        # output — leave the current status alone rather than echo it.
        return ""
    if len(line) > 100:
        line = line[:99] + "…"
    return f"{name}: {line}"


# Arguments worth naming in a status line, in the order they are looked for.
# "Reading scholarship-policy.pdf" is a different wait from "Running
# read_file…": one reads as progress, the other as a stall.
_STATUS_ARG_KEYS = ("query", "path", "file", "file_path", "url", "command")

_TOOL_VERBS = {
    "read_file": "Reading", "search_document": "Reading",
    "get_document_context": "Reading", "extract_tables": "Reading",
    "local_search": "Searching documents for", "search": "Searching the web for",
    "fetch_content": "Fetching", "grep": "Searching for",
    "find_files": "Looking for", "search_directory": "Looking for",
    "write_file": "Writing", "edit_file": "Editing",
}

# What a tool is doing, said without its arguments.  Used when several tools
# run at once: naming five subjects would not fit on one line, but "Searching
# the web and reading documents" is the part a waiting user wants anyway.
_TOOL_ACTIVITIES = {
    "search": "searching the web",
    "fetch_content": "reading web pages",
    "local_search": "searching your documents",
    "search_document": "reading documents",
    "get_document_context": "reading documents",
    "extract_tables": "reading documents",
    "read_file": "reading files",
    "write_file": "writing files",
    "edit_file": "editing files",
    "grep": "searching files",
    "find_files": "looking through files",
    "search_directory": "looking through files",
    "bash": "working",
}

# Said while the model is generating rather than calling tools.  "Thinking" is
# true of every one of these moments and so distinguishes none of them; what
# the model is thinking *about* is what makes the wait legible.
_STATUS_THINKING = "Thinking…"
_STATUS_REVIEWING = "Going through what it found…"
_STATUS_VERIFYING = "Checking the facts…"


def _short_subject(key: str, value: str) -> str:
    """Trim one argument value down to something that fits a status line."""
    subject = value.strip()
    if key in ("path", "file", "file_path"):
        return os.path.basename(subject.rstrip("/")) or subject
    if key == "url":
        # A full URL is mostly query string and tracking; the site is the part
        # that tells the user where the answer is coming from.
        host = urlparse(subject).netloc
        if host:
            return host[4:] if host.startswith("www.") else host
    return subject


def tool_status_text(name: str, arguments: dict | None) -> str:
    """One line describing what a tool is doing right now.

    Falls back to the tool's own name when nothing in the arguments is worth
    showing, so an unrecognized tool still reports that it is running.
    """
    if name in _OPAQUE_TOOLS:
        return _OPAQUE_STATUS
    verb = _TOOL_VERBS.get(name)
    subject = ""
    for key in _STATUS_ARG_KEYS:
        value = (arguments or {}).get(key)
        if isinstance(value, str) and value.strip():
            subject = _short_subject(key, value)
            break
    if not subject:
        return f"Running {name}…"
    if len(subject) > 60:
        subject = subject[:59] + "…"
    return f"{verb} {subject}…" if verb else f"{name}: {subject}…"


def _as_pairs(calls) -> list[tuple[str, dict | None]]:
    """Accept either tool names or ``(name, arguments)`` pairs."""
    return [(c, None) if isinstance(c, str) else (c[0], c[1]) for c in calls]


def tool_batch_text(calls) -> str:
    """The line shown when a batch of concurrent tool calls starts.

    Says as much as fits: one call is described in full, several calls of the
    same kind name the first subject and count the rest, and a mixed batch
    falls back to the kinds of work in it.
    """
    pairs = _as_pairs(calls)
    if not pairs:
        return ""
    if len(pairs) == 1:
        return tool_status_text(pairs[0][0], pairs[0][1])
    first = tool_status_text(*pairs[0])
    if (len({name for name, _ in pairs}) == 1
            and not first.startswith(("Running ", _OPAQUE_STATUS))):
        # Six searches in one turn are six angles on one question; showing the
        # first is a truer picture of the wait than "searching the web" alone.
        return f"{first.rstrip('… ')} (+{len(pairs) - 1} more)…"
    return f"{tool_batch_activity(calls).rstrip('… ')} — {len(pairs)} at once…"


def tool_batch_activity(calls) -> str:
    """The kinds of work in a batch, short enough to prefix a progress count."""
    pairs = _as_pairs(calls)
    if not pairs:
        return ""
    # dict, not set: the order tools were called in is the order they are named
    activities = list(dict.fromkeys(
        _TOOL_ACTIVITIES.get(name, "working") for name, _ in pairs))
    # "Reading files and reading documents" says "reading" twice for no gain;
    # activities sharing a verb are folded into one ("reading files and
    # documents"), which is how a person would say it.
    by_verb: dict[str, list[str]] = {}
    for activity in activities:
        verb, _, rest = activity.partition(" ")
        by_verb.setdefault(verb, [])
        if rest and rest not in by_verb[verb]:
            by_verb[verb].append(rest)
    phrases = [f"{verb} {_join_words(rests)}" if rests else verb
               for verb, rests in by_verb.items()]
    phrase = _join_words(phrases)
    return f"{phrase[0].upper()}{phrase[1:]}…"


def _join_words(parts: list[str]) -> str:
    """"a", "a and b", "a, b and c"."""
    if len(parts) <= 2:
        return " and ".join(parts)
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


class StreamingAdapter:
    """Minimal chat_ui adapter that forwards streaming tokens to a callback.

    Implements the subset of the ChatUI interface used by ``chat()`` so that
    ``process_task`` callers (web UI, A2A) can receive tokens incrementally
    without a full terminal UI.
    """

    def __init__(self, on_token=None, on_complete=None, show_logs=False,
                 throttle_tokens=0, on_tool_status=None, on_tool_result=None,
                 on_answer_start=None, on_correction=None, on_think=None,
                 on_approval=None):
        self.on_token = on_token
        self.on_complete = on_complete
        self.on_think = on_think
        # Present only when the caller has a person it can put the question
        # to. chat() checks for the method, so a caller that passes nothing
        # here gets a UI with no ask_approval at all and every gated command
        # is refused — the right answer for a run nobody is watching.
        self._on_approval = on_approval
        self.show_logs = show_logs
        self._throttle_tokens = throttle_tokens
        self._on_tool_status = on_tool_status
        self._on_tool_result = on_tool_result
        self._on_answer_start = on_answer_start
        self._on_correction = on_correction
        self._metrics: dict = {}  # live TurnMetrics sink; the source of tok/s
        # Set by the chat loop before each turn; the answer is the prose that
        # starts once tools have run, and the client wants to know which of
        # several streamed phases that is.
        self._tools_run = 0
        self._answer_announced = False
        self._batch_total = 0
        self._batch_done = 0
        self._batch_label = ""
        self._current_label = ""
        # Which phase the run is in, so the gaps between tool calls — where
        # most of a long run is actually spent — say something.
        self._saw_tool = False
        self._verifying = False
        self._content = ""
        self._tag_buf = ""
        self._token_count = 0
        self._pending: list[asyncio.Task] = []
        self.messages = []

    # ── approvals ────────────────────────────────────────────────
    def __getattr__(self, name):
        # ask_approval exists only when a callback was supplied. Defining it
        # unconditionally and returning "deny" would look the same to the
        # model but not to the person: chat() would report that someone was
        # asked and said no, when in truth nobody was asked.
        if name == "ask_approval" and self.__dict__.get("_on_approval"):
            return self.__dict__["_on_approval"]
        raise AttributeError(name)

    # ── streaming ────────────────────────────────────────────────
    def set_metrics(self, sink: dict) -> None:
        """Adopt the run's live token/timing accounting (see TurnMetrics)."""
        self._metrics = sink

    def stream_start(self):
        self._content = ""
        self._tag_buf = ""
        self._token_count = 0
        self._answer_announced = False

    def set_turn_context(self, tools_run: int = 0) -> None:
        """Told by the chat loop, before each turn, how much work preceded it."""
        self._tools_run = tools_run

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
        if not (self.on_token and token):
            return
        if not self._answer_announced and self._tools_run:
            # Prose after the tools have run: the model has stopped gathering
            # and started writing.  Announced once per stream — a phase that
            # turns out to end in another tool call is corrected by the next
            # announcement, which is the same correction the UI already makes
            # when a phase ends.
            self._answer_announced = True
            if self._on_answer_start:
                self._on_answer_start()
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
        """Forward a reasoning token to the client.

        Reasoning is the longest silence in a run, and a client told only
        "thinking…" through it cannot tell a model working from one that has
        hung.  Sent on its own channel rather than mixed into the answer, so
        the client can show it while it happens and fold it away once the
        answer it produced arrives.
        """
        if not (self.on_think and token):
            return
        # A model that keeps its reasoning inline in content brings the tags
        # along with it; what the client is shown is the reasoning, never the
        # markup that delimited it.
        token = token.replace("<think>", "").replace("</think>", "")
        if not token:
            return
        result = self.on_think(token)
        if asyncio.iscoroutine(result):
            self._pending.append(asyncio.ensure_future(result))

    @property
    def tokens_per_second(self) -> float:
        """Generation rate for the run so far, from the provider's own counts.

        Reasoning tokens included: the client is told how fast the model
        generated, not how fast the subset of it worth displaying arrived.
        """
        return decode_rate(self._metrics)

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

    # ── phase reporting ──────────────────────────────────────────
    def _idle_status(self) -> str:
        """What to say between tool calls, when the model is generating.

        A tool ending is not the run ending: the model is now reading what
        came back and deciding what to do next, which on a multi-turn run is
        where most of the waiting happens.  Saying so beats going blank.
        """
        if self._verifying:
            return _STATUS_VERIFYING
        if self._saw_tool:
            return _STATUS_REVIEWING
        return _STATUS_THINKING

    def _tool_status(self, name: str, arguments) -> str:
        """Status for one tool call, in the context of the current phase."""
        text = tool_status_text(name, arguments)
        if self._verifying and text.startswith("Running "):
            # The fact-check sometimes calls things this map has no words for
            # (and the model occasionally invents a name outright).  "Running
            # issues…" tells the user nothing; the phase does.
            return _STATUS_VERIFYING
        return text

    def start_tool_batch(self, calls):
        """Several read-only tools are about to run at once.

        ``calls`` may be tool names or ``(name, arguments)`` pairs.
        """
        self._batch_total = len(calls)
        self._batch_done = 0
        self._saw_tool = True
        # The opener says the most it can; the progress counts that follow it
        # keep the short form, so the line does not grow past reading width.
        self._batch_label = tool_batch_activity(calls)
        if self._on_tool_status:
            self._on_tool_status(tool_batch_text(calls))

    def end_tool_batch(self):
        self._batch_total = 0
        self._batch_done = 0
        self._batch_label = ""
        if self._on_tool_status:
            self._on_tool_status(self._idle_status())

    def start_tool_spinner(self, name, arguments):
        self._saw_tool = True
        if not self._batch_total:
            self._current_label = self._tool_status(name, arguments)
            if self._on_tool_status:
                self._on_tool_status(self._current_label)

    def stop_tool_spinner(self):
        # Inside a batch the calls finish in any order, so one of them going
        # quiet says nothing about the others; the batch counter reports
        # instead, and clearing here would only flicker.
        if not self._batch_total:
            self._current_label = ""
            if self._on_tool_status:
                self._on_tool_status(self._idle_status())

    def show_tool_done(self, name, result, success=True):
        if self._on_tool_status:
            if self._batch_total:
                self._batch_done += 1
                done = f"{self._batch_done} of {self._batch_total} done"
                stem = self._batch_label.rstrip("… ")
                self._on_tool_status(
                    f"{stem} — {done}…" if stem else f"{done}…")
            else:
                self._on_tool_status(self._idle_status())
        if self.show_logs:
            truncated = result[:500] + "..." if len(result) > 500 else result
            print(f"{name} returned: {truncated}")

    def tool_log(self, name: str, data, level: str = "info") -> None:
        """Called when a tool emits a log/notification message (e.g. sandbox output).

        Shown only as a one-line status update — raw output is never streamed
        into the message text (it would fill the chat with pip/git noise).
        """
        if self._on_tool_status:
            status = friendly_tool_status(name, data)
            if status:
                self._on_tool_status(status)

    def tool_progress(self, name, elapsed_seconds):
        """Called periodically during long-running tool calls to keep SSE alive."""
        if self._on_tool_status:
            # A slow call is exactly when the user needs to be told what is
            # slow, so repeat what it is doing rather than its function name.
            base = self._current_label or tool_status_text(name, None)
            self._on_tool_status(f"{base.rstrip('… ')}… ({elapsed_seconds}s)")
        if self.on_token:
            # Send an empty-content SSE event as a keepalive heartbeat
            result = self.on_token("", self._content)
            if asyncio.iscoroutine(result):
                task = asyncio.ensure_future(result)
                self._pending.append(task)

    def add_tool_result(self, name, result, truncate=300):
        """Not displayed (raw tool output would swamp the chat), but forwarded
        when a caller asked for it — the web UI uses tool output as the ground
        truth for which emails and facts actually came from a source."""
        if self._on_tool_result:
            try:
                self._on_tool_result(name, result)
            except Exception:
                pass

    def add_log(self, message, level="info"):
        if self.show_logs:
            print(f"[{level}] {message}")

    def set_context_usage(self, pct: float, max_tokens: int = 0) -> None:
        """No-op for external clients; context % is informational only."""
        pass

    # ── fact-check (runs after the answer has streamed) ──────────
    def verification_start(self) -> None:
        """The answer is written and is now being checked against its sources.

        Reported as tool status rather than as answer text: the draft is
        already on the client's screen, and this is work happening behind it.
        """
        self._verifying = True
        if self._on_tool_status:
            self._on_tool_status(_STATUS_VERIFYING)

    def verification_end(self, answer: str, note: str) -> None:
        """The fast check is done.

        Nothing to send: a corrected answer is the answer this run returns, and
        every client already replaces what it streamed with the final response.
        """
        self._verifying = False
        if self._on_tool_status:
            self._on_tool_status("")

    def verification_correction(self, answer: str, note: str) -> None:
        """A correction from the check that kept running after the answer.

        Forwarded rather than handled: by now this run has returned and the
        client has rendered what it returned, so the only thing that can amend
        what is on their screen is the client itself.
        """
        if self._on_correction and note:
            try:
                self._on_correction(answer, note)
            except Exception as e:
                logger.warning("Could not forward the correction: %s", e)

    def show_context_compaction(self, orig_msg_count: int, summary_chars: int) -> None:
        """Forward context compaction notice as a streaming token."""
        self._emit_notice(
            f"\n[Context compacted: {orig_msg_count} messages → {summary_chars:,} char summary]\n")

    def show_turn_limit(self, limit: int) -> None:
        """Forward the turn-limit stop as a streaming token.

        The run ends here, so a client that only renders streamed tokens would
        otherwise show the last step as though it were the answer.
        """
        self._emit_notice(
            f"\n[Stopped at the {limit}-step turn limit — the task did not finish]\n")

    def _emit_notice(self, msg: str) -> None:
        """Push an out-of-band notice into the token stream."""
        if not self.on_token:
            return
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
    topic: str | None = Field(default=None)
    prompt_intro: str | None = Field(default=None)
    # How many documents a research answer may open. Each one is a tool round
    # trip whose result then rides along in every later prompt, so this is the
    # dial between recall and how long an answer takes to arrive. Defaults to
    # the number of documents a result page describes, so the budget never cuts
    # the list short of what local_search put in front of the model.
    max_documents: int = Field(default=DEFAULT_MAX_DOCUMENTS)
    # Prior task/response pairs replayed into each request. They are re-sent
    # every turn, so a long tail costs prompt tokens on all of them.
    history_turns: int = Field(default=10)
    # Build the instruction by calling the prompt function directly instead of
    # reaching PromptsMCPServer over the network. Set false when that server
    # runs a customized prompt rather than the one shipped here.
    prompt_in_process: bool = Field(default=True)
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
    web_require_auth: bool = Field(default=True)
    web_title: str = Field(default="OnIt Chat")
    web_ga_measurement_id: str | None = Field(default=None)
    web_html_preview: bool = Field(default=True)
    # Full-duplex speech-to-speech via a NemotronLabs VoiceChat container.
    # See src/ui/voice.py:VoiceConfig for the keys.
    voice: dict[str, Any] = Field(default_factory=dict)
    agent_name: str = Field(default="OnIt")
    developer: str = Field(default="Rowel Atienza")
    a2a: bool = Field(default=False)
    a2a_port: int = Field(default=9001)
    a2a_name: str = Field(default="OnIt")
    a2a_description: str = Field(default="An intelligent agent for task automation and assistance.")
    gateway: str | None = Field(default=None)
    gateway_token: str | None = Field(default=None, exclude=True)
    viber_webhook_url: str | None = Field(default=None)
    viber_port: int = Field(default=8443)
    prompt_url: str | None = Field(default=None, exclude=True)
    file_server_url: str | None = Field(default=None, exclude=True)
    chat_ui: Any | None = Field(default=None, exclude=True)
    load_balancer: Any | None = Field(default=None, exclude=True)
    # Telemetry for the run in flight.  The interactive path answers in one
    # method and persists in another, so the sink chat() fills has to be
    # reachable from both; process_task keeps its own and never touches this.
    last_metrics: dict[str, Any] = Field(default_factory=dict, exclude=True)
    # What this session has already done — tools run, turns spent, how the last
    # attempt ended.  The session file keeps what was *said*; a resumed session
    # replaying only that has no idea which tools ran or what was tried and
    # failed.  Loaded in _setup_session, folded into after every task, and
    # written beside the session file (see model/serving/state.py).
    run_state: Any = Field(default=None, exclude=True)
    # The interactive path answers in one method and persists in another, so
    # the state chat() fills has to be reachable from both — the same reason
    # last_metrics above is a field.
    last_run_state: Any = Field(default=None, exclude=True)
    # Fact-checks still running behind answers that have already been handed
    # over, one per session.  Held here because this is the only object that
    # sees the next task arrive, and the next task is what ends them.
    background_checks: dict[str, Any] = Field(default_factory=dict, exclude=True)
    # The interactive path answers in one method and saves in another, so the
    # deep check chat() hands over waits here in between — it corrects the
    # saved answer, and there is nothing saved yet when it arrives.
    pending_deep_checks: list[Any] = Field(default_factory=list, exclude=True)

    @property
    def sandbox_available(self) -> bool:
        """Check if an MCP provider registered the sandbox execution tools.

        Derived, never configured: the routing block this gates tells the model
        to run *all* code in the sandbox, so asserting it without a provider
        would point the model at tools that do not exist.
        """
        return self.tool_registry is not None and "sandbox_run_code" in self.tool_registry.tools

    @property
    def local_search_available(self) -> bool:
        """Check if the local document search tool was discovered."""
        return self.tool_registry is not None and "local_search" in self.tool_registry.tools

    @property
    def document_search_available(self) -> bool:
        """Check if the within-document search tool was discovered."""
        return (self.tool_registry is not None
                and "search_document" in self.tool_registry.tools)

    @property
    def web_search_available(self) -> bool:
        """Check if the web search tool was discovered."""
        return self.tool_registry is not None and "search" in self.tool_registry.tools

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
                from .ui.api import WebApiUI
                self.chat_ui = WebApiUI(
                    theme=self.theme,
                    data_path=self.data_path,
                    show_logs=self.show_logs,
                    server_port=self.web_port,
                    google_client_id=self.web_google_client_id,
                    google_client_secret=self.web_google_client_secret,
                    allowed_emails=self.web_allowed_emails,
                    session_path=self.session_path,
                    title=self.web_title,
                    ga_measurement_id=self.web_ga_measurement_id,
                    verbose=self.verbose,
                    require_auth=self.web_require_auth,
                    html_preview=self.web_html_preview,
                    voice=self.voice,
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

    def _setup_mcp_servers(self) -> None:
        """Parse MCP server list from config and resolve the prompts server URL."""
        self.mcp_servers = self.config_data['mcp']['servers'] if 'mcp' in self.config_data and 'servers' in self.config_data['mcp'] else []
        # Ensure default servers are present if missing from config. Normally
        # the CLI has already done this, before it allocated ports for them.
        apply_default_mcp_servers(self.mcp_servers)
        # Override MCP server URL hosts if mcp_host is configured. A stdio
        # server is a subprocess of this process, so there is no host to move.
        mcp_host = self.config_data.get('mcp', {}).get('mcp_host')
        if mcp_host:
            from urllib.parse import urlparse, urlunparse
            for server in self.mcp_servers:
                url = server.get('url')
                if url and not str(url).startswith('stdio://'):
                    parsed = urlparse(url)
                    server['url'] = urlunparse(parsed._replace(netloc=f"{mcp_host}:{parsed.port}" if parsed.port else mcp_host))
        # Give every stdio server its launch spec. The CLI does this too, but
        # OnIt is also built directly — by the web, A2A and gateway servers and
        # by anything embedding it — and a stdio server that was never
        # registered has no address and no way to start.
        register_stdio_servers(self.mcp_servers, self._mcp_data_path())

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

    def _mcp_data_path(self) -> str:
        """Session root handed to MCP servers, resolved the way the CLI does.

        Read straight from the config rather than from ``self.data_path``,
        which is not set until later in initialize().
        """
        configured = self.config_data.get('data_path')
        base = Path(configured) if configured else Path.home() / "sandbox"
        return str(base.expanduser().resolve())

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
        # resolve host: CLI/config > env var ONIT_HOST. Skipped when
        # serving.endpoints supplies the hosts instead; _setup_load_balancer
        # raises if that list turns out to hold nothing usable.
        if (not self.model_serving.get('endpoints')
                and not self.model_serving.get('host')):
            env_host = os.environ.get('ONIT_HOST')
            if env_host:
                self.model_serving['host'] = env_host
            else:
                raise ValueError(
                    "No serving host configured. Set it via:\n"
                    "  - ONIT_HOST environment variable\n"
                    "  - --host CLI flag\n"
                    "  - serving.host (or serving.endpoints) in the config YAML"
                )
        self._setup_load_balancer()
        # Mirror the preferred endpoint onto serving.host so config readers
        # that expect a single host keep working under an endpoints list.
        self.model_serving.setdefault('host', self.load_balancer.preferred.host)
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

    def _setup_load_balancer(self) -> None:
        """Build the endpoint load balancer from serving config.

        Two config shapes are accepted. ``serving.endpoints`` is a list of any
        number of model servers, each entry taking ``host`` plus optional
        ``name``, ``model``, ``host_key``, and ``priority``. The legacy
        ``serving.host`` / ``serving.host2`` pair (or ONIT_HOST / ONIT_HOST2)
        is desugared into the same list, so existing configs are unaffected.

        Requests are distributed per ``serving.load_balancer`` (sticky assigns
        each new session a random host; round_robin, random, or least_busy
        distribute per request), and a failing endpoint is cooled down so
        another server takes over automatically.

        ``priority`` ranks endpoints — lower is preferred, equal numbers share
        a tier and load-balance against each other, and a higher number serves
        only while every better tier is cooling down. Without any priority the
        implicit default applies: Ollama endpoints are fallback-only, serving
        requests only while no vLLM/OpenRouter endpoint is healthy, which
        ``serving.ollama_fallback_only: false`` disables.
        """
        serving = self.model_serving
        endpoints = self._parse_endpoint_list(serving.get('endpoints'))
        if not endpoints:
            if serving.get('endpoints') and not serving.get('host'):
                raise ValueError(
                    "serving.endpoints is configured but no entry has a "
                    "usable 'host'. Give each entry a host URL, or fall back "
                    "to serving.host in the config YAML."
                )
            endpoints = self._legacy_endpoints(serving)
        self.load_balancer = LoadBalancer(
            endpoints, serving.get('load_balancer', 'sticky'),
            ollama_fallback_only=serving.get('ollama_fallback_only', True))
        if len(endpoints) > 1:
            print(f"  Load balancing ({self.load_balancer.algorithm}) across: "
                  f"{self.load_balancer.describe()}")

    @staticmethod
    def _parse_endpoint_list(raw) -> list:
        """Build ServerEndpoints from a ``serving.endpoints`` config list.

        Entries without a ``host`` are skipped, as is any host already claimed
        by an earlier entry — a duplicate would otherwise take a second share
        of the rotation. Returns [] when no list is configured, which sends
        the caller to the legacy host/host2 path.
        """
        if not isinstance(raw, list):
            return []
        endpoints, seen = [], set()
        for i, entry in enumerate(raw):
            if isinstance(entry, str):
                entry = {'host': entry}
            if not isinstance(entry, dict):
                logger.warning("Ignoring serving.endpoints[%d]: expected a "
                               "mapping or URL string, got %r", i, entry)
                continue
            host = str(entry.get('host') or '').strip()
            if not host:
                logger.warning("Ignoring serving.endpoints[%d]: no host", i)
                continue
            if host in seen:
                logger.warning("Ignoring serving.endpoints[%d]: duplicate "
                               "host %s", i, host)
                continue
            seen.add(host)
            try:
                priority = int(entry.get('priority', DEFAULT_PRIORITY))
            except (TypeError, ValueError):
                logger.warning("serving.endpoints[%d]: priority %r is not an "
                               "integer, using %d", i, entry.get('priority'),
                               DEFAULT_PRIORITY)
                priority = DEFAULT_PRIORITY
            endpoints.append(ServerEndpoint(
                host=host,
                # chat() resolves a provider-specific key (OLLAMA_API_KEY,
                # VLLM_API_KEY, ...) when this stays "EMPTY".
                host_key=str(entry.get('host_key') or 'EMPTY'),
                model=entry.get('model'),
                name=str(entry.get('name') or f'server{i + 1}'),
                priority=priority,
            ))
        return endpoints

    @staticmethod
    def _legacy_endpoints(serving: dict) -> list:
        """Desugar the host/host2 config pair into an endpoint list."""
        endpoints = [ServerEndpoint(
            host=serving['host'],
            host_key=serving.get('host_key', 'EMPTY'),
            model=serving.get('model'),
            name='server1',
        )]
        host2 = serving.get('host2') or os.environ.get('ONIT_HOST2')
        if host2 and host2 != serving['host']:
            host2_key = serving.get('host2_key')
            if not host2_key:
                # Env var ONIT_HOST2_KEY or OS keychain; chat() falls back to
                # provider-specific keys (e.g. OLLAMA_API_KEY) when "EMPTY".
                try:
                    from .setup import get_secret
                    host2_key = get_secret('host2_key')
                except Exception:
                    host2_key = None
            endpoints.append(ServerEndpoint(
                host=host2,
                host_key=host2_key or 'EMPTY',
                model=serving.get('model2'),
                name='server2',
            ))
        return endpoints

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

        # The other half of the session: what was done, not what was said.  A
        # new session gets an empty one rather than None, so every path below
        # can fold into it without a null check.
        self.run_state = RunState.load(state_path_for(self.session_path))

        configured_data_path = self.config_data.get('data_path')
        if configured_data_path:
            self.data_path = str(Path(configured_data_path).expanduser().resolve())
        else:
            self.data_path = str(Path.home() / "sandbox")
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
        self.topic = self.config_data.get('topic', None)
        self.prompt_intro = self.config_data.get('prompt_intro', None)
        self.max_documents = int(self.config_data.get('max_documents', DEFAULT_MAX_DOCUMENTS))
        self.history_turns = int(self.config_data.get('history_turns', 10))
        self.prompt_in_process = bool(self.config_data.get('prompt_in_process', True))
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
        self.web_require_auth = bool(self.config_data.get('web_require_auth', True))
        self.web_title = self.config_data.get('web_title', 'OnIt Chat')
        self.web_ga_measurement_id = self.config_data.get('web_ga_measurement_id', None)
        # Generated pages run in a sandboxed frame in the reply; set false to
        # keep them as source and a download instead.
        self.web_html_preview = bool(self.config_data.get('web_html_preview', True))
        _voice = self.config_data.get('voice') or {}
        self.voice = _voice if isinstance(_voice, dict) else {}
        self.agent_name = self.config_data.get('agent_name', 'OnIt')
        self.developer = self.config_data.get('developer', 'Rowel Atienza')
        self.a2a = self.config_data.get('a2a', False)
        self.a2a_port = self.config_data.get('a2a_port', 9001)
        self.a2a_name = self.config_data.get('a2a_name', 'OnIt')
        self.a2a_description = self.config_data.get('a2a_description', 'An intelligent agent for task automation and assistance.')
        self.gateway = self.config_data.get('gateway', None) or None
        self.gateway_token = self.config_data.get('gateway_token', None)
        self.viber_webhook_url = self.config_data.get('viber_webhook_url', None)
        self.viber_port = self.config_data.get('viber_port', 8443)

    def load_session_history(self, max_turns: int | None = None, session_path: str | None = None) -> list[dict]:
        """Load recent session history from the JSONL session file.

        Args:
            max_turns: Maximum number of recent task/response pairs to return.
                Defaults to the configured ``history_turns``. Every pair is
                replayed in full on every request, so the depth is a running
                cost on each turn of every task, not a one-off.
            session_path: Optional override path to the session file.

        Returns:
            A list of dicts with 'task' and 'response' keys, oldest first.
        """
        if max_turns is None:
            max_turns = self.history_turns
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
                           tool_result_callback=None,
                           answer_start_callback=None,
                           correction_callback=None,
                           think_callback=None,
                           approval_callback=None,
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
            tool_result_callback: Optional callback ``(tool_name, result) -> None``
                called with each tool's raw output, letting callers treat it as
                sourced material (the web UI verifies emails against it).
            answer_start_callback: Optional callback ``() -> None`` fired when
                the model starts writing prose after its tools have run — the
                moment the answer begins, as opposed to the moment the run ends.
            correction_callback: Optional callback ``(answer, note) -> None``
                called if the fact-check that keeps running after this returns
                finds something. Passing it is what turns that check on: a
                caller with no way to show a late correction should not be
                paying for one. It is called at most once, and never after the
                same session starts another task.
            think_callback: Optional callback ``(token) -> None`` called for
                each reasoning token, kept apart from ``stream_callback`` so
                the caller can show the model's working without it landing in
                the answer.
            approval_callback: Optional ``async (request) -> "once"|"session"
                |"deny"`` asked before a command the policy will not run on
                its own authority. Passing it is what makes those commands
                possible at all: a caller that cannot reach a person leaves it
                unset and every such command is refused, which is what a
                scheduled or headless run should do.
        """
        # Use per-chat overrides if provided, otherwise fall back to instance defaults
        effective_session_path = session_path or self.session_path
        effective_data_path = data_path or self.data_path
        effective_safety_queue = safety_queue or self.safety_queue

        while not effective_safety_queue.empty():
            effective_safety_queue.get_nowait()

        # Still timed: it sits in front of the first token of every task, and
        # nothing else measures it.
        _instruction_start = time.monotonic()
        instruction = await self._assistant_instruction(
            task, effective_data_path, session_path=effective_session_path)
        _instruction_s = time.monotonic() - _instruction_start

        # Use a StreamingAdapter when streaming tokens or tracking tool status.
        _adapter = None
        if (((stream_callback or think_callback) and self.stream)
                or tool_result_callback or approval_callback):
            _adapter = StreamingAdapter(
                on_token=stream_callback,
                on_complete=stream_complete_callback,
                on_think=think_callback,
                show_logs=self.show_logs,
                throttle_tokens=stream_throttle,
                on_tool_status=tool_status_callback,
                on_tool_result=tool_result_callback,
                on_answer_start=answer_start_callback,
                on_correction=correction_callback,
                on_approval=approval_callback,
            )

        effective_session_id = session_id or self.session_id
        # Whatever was still being checked for this session is about to be
        # answering a question nobody asked any more.
        self._cancel_background_check(effective_session_id)
        _metrics: dict = {}
        # This task's run state, read back below to persist what it did.  Fresh
        # per task on purpose: the repeated-call ceiling counts against it, and
        # a session's accumulated history would spend that budget before the
        # task made its first call.
        _run_state = RunState()
        kwargs = {
            'metrics': _metrics,
            'run_state': _run_state,
            'console': None,
            'chat_ui': _adapter,
            'cursor': AGENT_CURSOR, 'memories': None,
            'verbose': self.verbose or self.show_logs,
            'data_path': effective_data_path,
            'session_id': effective_session_id,
            'max_tokens': self.model_serving.get('max_tokens', 32768),
            'max_context_tokens': self.model_serving.get('max_context_tokens', None),
            'session_history': self.load_session_history(session_path=effective_session_path),
            'stream': self.stream,
        }
        for _k in SERVING_PASSTHROUGH:
            if _k in self.model_serving:
                kwargs[_k] = self.model_serving[_k]
        if self.prompt_intro:
            kwargs['prompt_intro'] = self.prompt_intro
        # Collected rather than started: the deep check amends the answer this
        # method has not saved yet, so it is handed to the loop below only once
        # there is a saved answer for it to amend.
        _deep_checks: list = []
        if correction_callback:
            kwargs['background_verify'] = _deep_checks.append
        MAX_PROCESS_RETRIES = 3
        last_response = None
        for _pt_attempt in range(1, MAX_PROCESS_RETRIES + 1):
            if not effective_safety_queue.empty():
                break
            endpoint = self.load_balancer.acquire(key=effective_session_id)
            _usable = None
            try:
                last_response = await chat(
                    host=endpoint.host,
                    host_key=endpoint.host_key,
                    model=endpoint.model,
                    instruction=instruction,
                    images=images,
                    tool_registry=self.tool_registry,
                    safety_queue=effective_safety_queue,
                    think=self.model_serving.get("think", False),
                    timeout=self.timeout,
                    **kwargs,
                )
                _usable = last_response and remove_tags(last_response).strip()
            finally:
                # A stop request is not an endpoint failure — don't cool down.
                self.load_balancer.release(
                    endpoint,
                    success=bool(_usable) or not effective_safety_queue.empty())
            if _usable:
                break
            if _pt_attempt < MAX_PROCESS_RETRIES and effective_safety_queue.empty():
                kind = "Empty" if last_response is not None else "No"
                retry_msg = (f"{kind} response from {endpoint.name or endpoint.host}, "
                             f"retrying ({_pt_attempt}/{MAX_PROCESS_RETRIES})...")
                logger.warning(retry_msg)
                if hasattr(self, 'chat_ui') and self.chat_ui and hasattr(self.chat_ui, 'add_log'):
                    self.chat_ui.add_log(retry_msg, level="warning")
                await asyncio.sleep(min(2 ** _pt_attempt, 10))

        # If the safety queue fired, stop the sandbox container.
        if not effective_safety_queue.empty():
            await _call_sandbox_stop(self.tool_registry, effective_session_id)

        # Flush any pending async streaming events (e.g. A2A) before
        # returning, so the client sees all partial updates before the
        # final completed message.
        if _adapter:
            await _adapter.flush()

        # Telemetry is reported whether or not the task succeeded — a run that
        # ended in a retry loop is exactly the one worth looking at.
        _metrics["instruction_s"] = round(_instruction_s, 3)
        if stats is not None:
            stats["metrics"] = _metrics
            stats["tokens_per_second"] = decode_rate(_metrics)
        logger.info("task timing: instruction %.2fs | %s",
                    _instruction_s, summarize_metrics(_metrics))

        if not last_response or not remove_tags(last_response).strip():
            logger.error("chat() returned empty/None after %d retries "
                         "across hosts: %s",
                         MAX_PROCESS_RETRIES, ", ".join(self.load_balancer.hosts))
            for _stale in _deep_checks:
                _stale.close()
            # A run that spent thirty turns and came back with nothing is
            # exactly the run the next task should know about.
            self._persist_run_state(_run_state, effective_session_path)
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
        self._persist_run_state(_run_state, effective_session_path)
        self._record_trajectory(task, response, _metrics,
                                effective_session_path,
                                session_id or self.session_id, _adapter,
                                run_state=_run_state)
        # The answer is saved, so the check behind it now has something to
        # correct.  Anything the run collected earlier belonged to an attempt
        # that was retried over, and is closed rather than run.
        for _stale in _deep_checks[:-1]:
            _stale.close()
        if _deep_checks and effective_safety_queue.empty():
            self._schedule_background_check(
                effective_session_id, _deep_checks[-1], effective_session_path)
        return response

    # ── fact-checks that outlive the answer ────────────────────────────────
    #
    # The fast check decides what the user reads; this owns the slow one that
    # keeps going behind it.  Two rules make it safe to run at all: it belongs
    # to exactly one session, and the next thing that session does ends it.  A
    # correction to an answer the user has already moved past is not worth the
    # tokens, and arriving late next to a newer answer it does not describe is
    # worse than not arriving.

    def _schedule_background_check(self, session_id: str, coro,
                                   session_path: str) -> None:
        """Own a deep check for this session, replacing any still running.

        Showing the correction is not this method's job — the check tells the
        UI itself, through the same object that was told it had started.  What
        happens here is the half a UI cannot do: making the saved conversation
        agree with what the user was just shown.
        """
        self._cancel_background_check(session_id)

        async def _run():
            try:
                answer, note = await coro
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Background fact-check failed: %s", e)
                return
            if not note:
                return
            # The session file holds what the next turn is built from, so a
            # correction that is not written there is one the model will
            # contradict later, citing itself.
            self._amend_last_response(session_path, answer)

        task = asyncio.ensure_future(_run())
        self.background_checks[session_id] = task
        task.add_done_callback(
            lambda t, sid=session_id: (
                self.background_checks.pop(sid, None)
                if self.background_checks.get(sid) is t else None))

    def _cancel_background_check(self, session_id: str) -> None:
        """End the check still running for this session, if there is one."""
        task = self.background_checks.pop(session_id, None)
        if task and not task.done():
            task.cancel()

    def cancel_background_checks(self) -> None:
        """End every one of them — shutdown, or a session being torn down."""
        for session_id in list(self.background_checks):
            self._cancel_background_check(session_id)

    @staticmethod
    def _amend_last_response(session_path: str, answer: str) -> None:
        """Rewrite the last saved answer in place.

        The session file is append-only JSONL and the correction belongs to the
        turn already in it, not to a new one: a second record would replay to
        the model as though the assistant had answered twice.
        """
        if not session_path or not answer:
            return
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            while lines and not lines[-1].strip():
                lines.pop()
            if not lines:
                return
            record = json.loads(lines[-1])
            record["response"] = answer
            record["fact_checked"] = True
            lines[-1] = json.dumps(record)
            with open(session_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            logger.warning("Could not amend the saved answer: %s", e)

    # ── run state: what the session has already done ───────────────────────
    #
    # The session file records the conversation and ``_record_trajectory``
    # records the run for offline learning.  Neither is readable by the agent
    # itself on the next task, which is what this closes: a resumed session
    # that knows it already ran ``local_search`` eleven times and stopped at
    # the turn limit does not start over from the same place.

    def _persist_run_state(self, state: Any, session_path: str) -> None:
        """Fold a finished run into its session's state file.

        Best-effort throughout, like the trajectory writer above: a task that
        answered must not be reported as failed because a bookkeeping file
        could not be written.
        """
        path = state_path_for(session_path)
        if state is None or not path:
            return
        try:
            # One OnIt serves many sessions in web mode, so the file — not the
            # in-memory copy — is the source of truth for any session but the
            # one this process was started on.
            own = bool(self.session_path
                       and os.path.abspath(session_path)
                       == os.path.abspath(self.session_path))
            session_state = (self.run_state if own and self.run_state is not None
                             else RunState.load(path))
            session_state.merge(state)
            session_state.save(path)
            if own:
                self.run_state = session_state
        except Exception as e:
            logger.debug("run state not persisted: %s", e)

    def _prior_run_note(self, session_path: str | None = None) -> str:
        """What to tell the next task about the ones before it, or ""."""
        try:
            effective = session_path or self.session_path
            own = bool(self.session_path and effective
                       and os.path.abspath(effective)
                       == os.path.abspath(self.session_path))
            state = (self.run_state if own and self.run_state is not None
                     else RunState.load(state_path_for(effective)))
            return state.resume_note()
        except Exception as e:
            logger.debug("prior run note unavailable: %s", e)
            return ""

    def _record_trajectory(self, task: str, response: str, metrics: dict,
                           session_path: str, session_id: str, adapter,
                           run_state: Any = None) -> None:
        """Write what this task actually did to the trajectory store.

        The session file above keeps the conversation; this keeps the run — the
        tool calls and their outcomes, the turn-by-turn token counts, the
        retries.  Nothing downstream of it exists yet, which is exactly why it
        has to start now: the loops in ``docs/SELF_IMPROVEMENT.md`` can only
        learn from runs that were recorded before they were written.

        Off unless the autonomy level is ``observe`` or higher, and best-effort
        either way — a task that answered must not be reported as failed
        because a log line could not be written.
        """
        try:
            from .learn import record_task, recording_enabled
            if not recording_enabled(self.config_data):
                return
            from .sessions import _turn_count_from_jsonl, get_session_owner
            sessions_dir = os.path.dirname(session_path)
            record_task(
                session_id=session_id,
                # The session file was just appended to, so its line count is
                # this task's turn number.
                turn=_turn_count_from_jsonl(session_path),
                task=task,
                response=response,
                metrics=metrics,
                config_data=self.config_data,
                tools_available=sorted(self.tool_registry.tools) if self.tool_registry else [],
                owner=get_session_owner(session_id, sessions_dir),
                topic_hint=self.topic,
                # The configured model may be unset and resolved from the
                # endpoint; the adapter carries whatever actually answered.
                model=getattr(adapter, "model_name", None) or self.model_serving.get("model"),
                # Whether the loop finished or gave up — the one signal the
                # metrics cannot supply.  Taken from the run's own state rather
                # than re-derived, so the two records of how a run ended cannot
                # drift apart (docs/HARNESS_CAPABILITIES.md §8.2, step 6).
                stop_reason=getattr(run_state, "stop_reason", None),
            )
        except Exception as e:
            logger.debug("trajectory not recorded: %s", e)

    async def run_loop(self) -> None:
        """Run the OnIt agent in loop mode, executing a task repeatedly."""
        if not self.task:
            raise ValueError("Loop mode requires a 'task' to be set in the config.")

        print(f"Loop mode: task='{self.task}', period={self.period}s (Ctrl+C to stop)")
        iteration = 0

        while True:
            try:
                iteration += 1
                start_time = asyncio.get_event_loop().time()

                # clear safety queue
                while not self.safety_queue.empty():
                    self.safety_queue.get_nowait()

                print(f"--- Iteration {iteration} ---")
                instruction = await self._assistant_instruction(self.task)

                # call chat directly (no queues needed)
                _metrics: dict = {}
                _run_state = RunState()
                kwargs = {'console': None,
                          'run_state': _run_state,
                          'chat_ui': None,
                          'cursor': AGENT_CURSOR,
                          'memories': None,
                          'metrics': _metrics,
                          'verbose': self.verbose,
                          'data_path': self.data_path,
                          'session_id': self.session_id,
                          'max_tokens': self.model_serving.get('max_tokens', 32768),
                          'max_context_tokens': self.model_serving.get('max_context_tokens', None),
                          'session_history': self.load_session_history()}
                for _k in SERVING_PASSTHROUGH:
                    if _k in self.model_serving:
                        kwargs[_k] = self.model_serving[_k]
                endpoint = self.load_balancer.acquire(key=self.session_id)
                last_response = None
                try:
                    last_response = await chat(host=endpoint.host,
                                                host_key=endpoint.host_key,
                                                model=endpoint.model,
                                                instruction=instruction,
                                                tool_registry=self.tool_registry,
                                                safety_queue=self.safety_queue,
                                                think=self.model_serving.get("think", False),
                                                timeout=self.timeout,
                                                **kwargs)
                finally:
                    self.load_balancer.release(
                        endpoint, success=last_response is not None)

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
                    # Loop mode runs one task over and over, which is the
                    # workload where comparing run to run says the most.
                    self._persist_run_state(_run_state, self.session_path)
                    self._record_trajectory(self.task, response, _metrics,
                                            self.session_path, self.session_id, None,
                                            run_state=_run_state)

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

    async def _assistant_instruction(self, task: str,
                                     data_path: str | None = None,
                                     session_path: str | None = None) -> str:
        """The agent instruction for ``task``.

        PromptsMCPServer hosts a single prompt, and that prompt is a pure
        function in this package: it assembles a string, holds no state, and
        does no I/O beyond creating the data directory.  Reaching it over MCP
        put a connection, an initialize handshake and a round trip in front of
        the first token of every task — the cost ``instruction_s`` exists to
        measure.  Call it directly, and leave the server up for clients outside
        this process.

        ``prompt_in_process: false`` restores the round trip, for a
        PromptsMCPServer running a customized prompt rather than this one.
        """
        args = {
            "task": task,
            "data_path": data_path or self.data_path,
            "template_path": self.template_path,
            "file_server_url": self.file_server_url,
            "topic": self.topic,
            "sandbox_available": self.sandbox_available,
            "local_search_available": self.local_search_available,
            "document_search_available": self.document_search_available,
            "web_search_available": self.web_search_available,
            # Only claimed when chat() will actually offer them: the note tools
            # need a data_path to write under, and the toolset as a whole is
            # switched off by `serving.harness_tools: false`.
            "harness_tools_available": (bool(data_path or self.data_path)
                                        and self.model_serving.get('harness_tools', True)),
            # Same rule again, and one more switch: the store can be turned off
            # without withdrawing the note tools.
            "result_store_available": (bool(data_path or self.data_path)
                                       and self.model_serving.get('harness_tools', True)
                                       and self.model_serving.get('result_store', True)),
            # Needs no data_path: an interpreter needs somewhere to run, not
            # somewhere to write.
            "code_execution_available": (self.model_serving.get('harness_tools', True)
                                         and self.model_serving.get('code_execution', False)),
            "agent_name": self.agent_name,
            "developer": self.developer,
            "max_documents": self.max_documents,
            # What this session already did, from the run state beside its
            # session file.  Empty on the first task of a new session, which
            # is the common case and costs nothing.
            "prior_attempts": self._prior_run_note(session_path),
        }
        if self.prompt_in_process:
            return await build_assistant_instruction(**args)
        async with Client(self.prompt_url) as prompt_client:
            result = await prompt_client.get_prompt("assistant", args)
        return result.messages[0].content.text

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
        fd = sys.stdin.fileno()
        def _on_enter():
            # Drain the available bytes with a raw, non-buffered read. Two reasons:
            #   1. ``sys.stdin.readline()`` *blocks* until a full line is available;
            #      running it inside this event-loop callback would freeze the loop
            #      (and the streaming UI) whenever stdin is readable mid-line.
            #   2. ``sys.stdin`` is buffered — readline() pulls bytes into a Python
            #      userspace buffer that the raw ``os.read(fd, 1)`` input reader can
            #      never see, silently swallowing the start of the user's next message.
            #   ``os.read`` on a fd the selector just reported readable returns
            #   immediately and leaves any unconsumed bytes in the kernel tty buffer,
            #   where the text-UI input reader can still pick them up.
            try:
                data = os.read(fd, 4096)
            except OSError:
                return
            if b"\n" in data or b"\r" in data:
                self.safety_queue.put_nowait(STOP_TAG)
        try:
            loop.add_reader(fd, _on_enter)
        except NotImplementedError:
            pass  # Windows ProactorEventLoop does not support add_reader
        return _on_enter

    def _cleanup_enter_key_listener(self, loop: asyncio.AbstractEventLoop) -> None:
        """Clean up Enter-key listener for text UI."""
        if self.web:
            return
        import sys
        if hasattr(self, 'chat_ui') and self.chat_ui and hasattr(self.chat_ui, 'stop_thinking'):
            self.chat_ui.stop_thinking()
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
        # If streaming already persisted the message via stream_end(), replace
        # it rather than adding a duplicate.  Matched on prefix, not equality,
        # and in both directions: the run can append to what it streamed — a
        # turn-limit notice, a resumed final answer — and it can also hand back
        # less than it streamed, because the answer is tag-stripped on the way
        # here and the strip can remove something stream_end() kept.  Either
        # way it is the same turn's answer, and an equality check posts it to
        # the panel a second time — the shortened copy landing under the full
        # one, which reads as an answer that was cut off.
        _last = self.chat_ui.messages[-1] if self.chat_ui.messages else None
        if (_last and hasattr(_last, 'role') and _last.role == "assistant"
                and _last.content and (response.startswith(_last.content)
                                       or _last.content.startswith(response))):
            from src.ui.text import Message
            self.chat_ui.messages[-1] = Message(
                role=_last.role, content=response,
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
        try:
            from .sessions import update_session
            update_session(self.session_id, task=task,
                           sessions_dir=os.path.dirname(self.session_path))
        except Exception:
            pass
        self._persist_run_state(self.last_run_state, self.session_path)
        self._record_trajectory(task, response, self.last_metrics,
                                self.session_path, self.session_id, self.chat_ui,
                                run_state=self.last_run_state)
        if self.pending_deep_checks:
            self._schedule_background_check(
                self.session_id, self.pending_deep_checks.pop(),
                self.session_path)
            for _stale in self.pending_deep_checks:
                _stale.close()
            self.pending_deep_checks.clear()

    def _format_elapsed_time(self, elapsed_secs: float) -> str:
        """Format elapsed time string, including tokens/sec if available."""
        # Same accounting the streamed footer used, so re-rendering a message
        # from history does not quote a different rate than the run did.
        _tok_s = decode_rate(self.last_metrics)
        # Same shape as the web UI's per-answer meta line.
        if hasattr(self.chat_ui, 'format_meta'):
            return self.chat_ui.format_meta(elapsed_secs, _tok_s)
        return f"{elapsed_secs:.2f}s"

    async def client_to_agent(self) -> None:
        """Handle client to agent communication"""

        agent_task = None
        loop = asyncio.get_event_loop()

        while True:
            task = await self._get_user_task(loop)

            if task.lower().strip() in self.stop_commands:
                if not self.web:
                    self.chat_ui.console.print("Exiting chat session...", style="warning")
                if agent_task and not agent_task.done():
                    agent_task.cancel()
                self.cancel_background_checks()
                break
            if not task or len(task) == 0:
                task = None
                continue

            # A new question ends the check still running behind the last
            # answer.  Anything it had to say is about a screen the user has
            # already scrolled past, and it would land under the wrong answer.
            self._cancel_background_check(self.session_id)

            # clear all queues
            while not self.input_queue.empty():
                self.input_queue.get_nowait()
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
            while not self.safety_queue.empty():
                self.safety_queue.get_nowait()

            instruction = await self._assistant_instruction(task)

            on_enter_cb = self._setup_enter_key_listener(loop)

            # submit instruction with retry on API error
            start_time = loop.time()
            # Start the UI's own clock so streamed blocks can print elapsed
            # time before the turn returns here.
            if hasattr(self.chat_ui, "turn_start"):
                self.chat_ui.turn_start()
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
                    await _call_sandbox_stop(self.tool_registry, self.session_id)
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
                                await _call_sandbox_stop(self.tool_registry, self.session_id)
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
                    await _call_sandbox_stop(self.tool_registry, self.session_id)
                    await self.output_queue.put(STOP_TAG)
                    break
                self.last_metrics.clear()
                self.last_run_state = RunState()
                kwargs = {'console': self.chat_ui.console,
                          'run_state': self.last_run_state,
                          'chat_ui': self.chat_ui,
                          'cursor': AGENT_CURSOR,
                          'memories': None,
                          'metrics': self.last_metrics,
                          'verbose': self.verbose,
                          'data_path': self.data_path,
                          'session_id': self.session_id,
                          'max_tokens': self.model_serving.get('max_tokens', 32768),
                          'max_context_tokens': self.model_serving.get('max_context_tokens', None),
                          'session_history': self.load_session_history(),
                          'stream': self.stream}
                for _k in SERVING_PASSTHROUGH:
                    if _k in self.model_serving:
                        kwargs[_k] = self.model_serving[_k]
                if self.prompt_intro:
                    kwargs['prompt_intro'] = self.prompt_intro
                # The terminal is handed its correction by chat() itself — the
                # UI object there is the real one — so all that is collected
                # here is the check, to be started once the answer is saved.
                for _stale in self.pending_deep_checks:
                    _stale.close()
                self.pending_deep_checks.clear()
                kwargs['background_verify'] = self.pending_deep_checks.append

                last_response = None
                for attempt in range(1, MAX_AGENT_RETRIES + 1):
                    if not self.safety_queue.empty():
                        break
                    endpoint = self.load_balancer.acquire(key=self.session_id)
                    _usable = None
                    try:
                        last_response = await chat(
                            host=endpoint.host,
                            host_key=endpoint.host_key,
                            model=endpoint.model,
                            instruction=instruction,
                            tool_registry=self.tool_registry,
                            safety_queue=self.safety_queue,
                            think=self.model_serving.get("think", False),
                            timeout=self.timeout,
                            **kwargs)
                        # Treat empty/whitespace-only responses as failures too
                        _usable = last_response and remove_tags(last_response).strip()
                    finally:
                        # A stop request is not an endpoint failure — don't cool down.
                        self.load_balancer.release(
                            endpoint,
                            success=bool(_usable) or not self.safety_queue.empty())
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
                    await _call_sandbox_stop(self.tool_registry, self.session_id)
                    await self.output_queue.put(STOP_TAG)
                    break
                await self.output_queue.put(f"<answer>{last_response}</answer>")
                return
            except asyncio.CancelledError:
                logger.warning("Agent session cancelled.")
                await _call_sandbox_stop(self.tool_registry, self.session_id)
                await self.output_queue.put(None)
                return
            except Exception as e:
                logger.error("Error in agent session: %s", e)
                if self.chat_ui and hasattr(self.chat_ui, 'add_log'):
                    self.chat_ui.add_log(f"Agent error: {e}", level="error")
                await self.output_queue.put(None)
                return