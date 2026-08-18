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

Tool results kept on disk, passed by reference into the context.

Every byte a tool produced used to be serialized into the context window and
then progressively destroyed to make room for the next one: a hard cut at
``MAX_TOOL_RESPONSE`` that threw the middle away, a decay pass that reduced
older results to their opening, and a compaction that summarized whatever
survived.  The recovery path for any of it was **running the tool again** — a
network round trip to recover bytes the harness had already been given and
deliberately deleted.

Here the full output goes to a file and a bounded preview goes into the
message, carrying a handle.  Truncation stops being lossy: the head is still
what the model reads first, but the rest is one local file read away rather
than one tool re-execution away.  ``result_read`` and ``result_grep`` in
``harness.py`` are how it is read back.

**Session isolation is the load-bearing constraint.**  Results live under the
session's ``data_path`` and die with it — no global cache, no shared directory,
however tempting a cross-session result cache looks.  ``data_path`` is a trust
boundary, so every handle is jailed under it twice: a regex that admits nothing
but digits, and a resolve-and-compare against the results directory.
"""

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Beside the notes, and under the same dot-directory: a task that lists its
# working directory should not find the harness's bookkeeping mixed in with the
# artifacts it produced.
RESULTS_SUBDIR = os.path.join(".onit", "results")
RESULT_SUFFIX = ".txt"

# How much of a stored result goes into the message.
#
# Deliberately equal to ``TOOL_RESULT_DECAY_CHARS`` in chat.py, and for the
# reason recorded there: at 1,500 characters the surviving head of a
# local_search result was its document summaries alone — every matched passage
# fell outside it, leaving the model a list of titles and no quotes.  6,000 is
# where that stopped happening.  A preview is the same object as a decayed
# result seen from the other end, so it must clear the same bar.
RESULT_PREVIEW_CHARS = 6000

# Below this, a result is passed through untouched.  The preview would save at
# most 2,000 characters, which does not pay for a file, a handle line in every
# later turn, and a possible extra tool call to read back what would otherwise
# have been in front of the model already.
RESULT_STORE_THRESHOLD = 8000

# One window of a stored result.  Large enough to be worth the round trip,
# small enough that a model reading a 200k file cannot pull it all into the
# context in two calls.
DEFAULT_READ_CHARS = 4000
MAX_READ_CHARS = 8000

MAX_GREP_MATCHES = 40
MAX_GREP_CHARS = 8000
MAX_GREP_CONTEXT = 10

# A long session accrues one file per large result.  Pruned rather than
# refused: a note may be turned away because the model chose to write it, but a
# tool result arrives whether or not there is room, and refusing to store one
# would put the lossy truncation back exactly where this replaced it.
MAX_STORED_RESULTS = 200

# Zero-padded so the directory sorts the way the run happened.
_HANDLE_DIGITS = 4
_HANDLE_RE = re.compile(r"^[0-9]{1,6}$")
_UNSAFE_TOOL = re.compile(r"[^A-Za-z0-9._-]")

# Opens the preview line of a stored result, and how the handle is recovered
# from a message the loop is about to decay.
_HEADER_RE = re.compile(r"^\[result:([0-9]{1,6})\b")

# Every trailer this module writes starts with it, and so does chat.py's decay
# marker.  One prefix is what lets "has this message already been trimmed?" be
# a single check rather than a list of known sentences.
CONTINUATION_PREFIX = "… ["
# The trailer a decayed result ends with, distinguished from the one a fresh
# preview ends with.  "Already trimmed?" cannot be answered by the shared
# prefix alone: a preview also ends in a continuation line, and treating one as
# the other would either decay nothing or re-decay a message whose head has
# already been cut — the second pass slicing the previous trailer back into the
# body it is supposed to follow.
DECAY_MARK = "trimmed: older tool result"


def _error(message: str) -> str:
    """A refusal in the shape the tool-call error path already uses.

    Duplicated from harness.py rather than imported: harness.py imports this
    module, and a two-line helper is a smaller price than a cycle.
    """
    return f"Error: {message}"


def handle_of(content) -> str | None:
    """The handle a tool message carries, or None if it carries none.

    Read from the message rather than tracked alongside it, so a conversation
    that survived a compaction — or arrived from a session file — still knows
    which of its results are recoverable.
    """
    if not isinstance(content, str):
        return None
    match = _HEADER_RE.match(content.lstrip())
    return match.group(1) if match else None


def _last_line(content) -> str:
    if not isinstance(content, str):
        return ""
    return content.rstrip().rsplit("\n", 1)[-1]


def is_continued(content: str) -> bool:
    """Whether this message already ends in a "there is more" trailer."""
    return _last_line(content).startswith(CONTINUATION_PREFIX)


def is_decayed(content: str) -> bool:
    """Whether this message has already been cut by the decay pass."""
    return _last_line(content).startswith(CONTINUATION_PREFIX + DECAY_MARK)


class ResultStore:
    """Full tool outputs on disk, addressed by handle.

    Owned by ``HarnessTools`` and reached through it, so the loop threads one
    object rather than two.  Disabled without a ``data_path``: there is nowhere
    to write, and a store that silently keeps nothing would hand out handles
    that never resolve.
    """

    def __init__(self, data_path: str = "",
                 preview_chars: int = RESULT_PREVIEW_CHARS,
                 threshold: int = RESULT_STORE_THRESHOLD,
                 enabled: bool = True):
        self.data_path = data_path or ""
        self.preview_chars = max(500, int(preview_chars))
        self.threshold = max(self.preview_chars, int(threshold))
        self.enabled = bool(enabled) and bool(self.data_path)
        # Continues where the session left off rather than restarting at 1: a
        # second task in the same session would otherwise write 0001 over the
        # first task's 0001, and every handle already in the conversation would
        # quietly start resolving to the wrong result.
        self._next = self._highest_existing() + 1

    # ── the directory ───────────────────────────────────────────────────────

    @property
    def results_dir(self) -> Path:
        return Path(self.data_path) / RESULTS_SUBDIR

    def _stored_files(self) -> list:
        """Every stored result, oldest handle first."""
        if not self.enabled:
            return []
        try:
            return sorted(p for p in self.results_dir.glob(f"*{RESULT_SUFFIX}")
                          if p.is_file() and _HANDLE_RE.match(p.name.split("-")[0]))
        except OSError:
            return []

    def _highest_existing(self) -> int:
        highest = 0
        for path in self._stored_files():
            try:
                highest = max(highest, int(path.name.split("-")[0]))
            except (ValueError, IndexError):
                continue
        return highest

    def _path_for(self, handle: str) -> Path | None:
        """The file behind ``handle``, or None — unknown, or outside the jail.

        The regex above has already rejected everything that could escape, so
        the resolve-and-compare is redundant by design.  ``data_path`` is a
        session isolation boundary, and a boundary with one check has one bug
        between it and the next session's files.
        """
        if not self.enabled or not isinstance(handle, str):
            return None
        handle = handle.strip()
        if not _HANDLE_RE.match(handle):
            return None
        root = self.results_dir
        try:
            resolved_root = root.resolve()
        except OSError:
            return None
        for candidate in root.glob(f"{int(handle):0{_HANDLE_DIGITS}d}-*{RESULT_SUFFIX}"):
            try:
                resolved = candidate.resolve()
                resolved.relative_to(resolved_root)
            except (ValueError, OSError):
                # A symlink pointing out of the jail is the case this catches:
                # the name is fine, the target is not.
                continue
            if resolved.is_file():
                return resolved
        return None

    def _prune(self) -> None:
        """Drop the oldest results once the cap is passed."""
        files = self._stored_files()
        for path in files[:-MAX_STORED_RESULTS] if len(files) > MAX_STORED_RESULTS else []:
            try:
                path.unlink()
            except OSError:  # pragma: no cover - permissions, races
                pass

    # ── what the model sees ─────────────────────────────────────────────────

    def stored(self) -> list[dict]:
        """One record per stored result: handle, tool, size.  For discovery —
        ``context_status`` reports it, so finding a handle costs no extra tool."""
        out = []
        for path in self._stored_files():
            handle, _, rest = path.name.partition("-")
            try:
                size = path.stat().st_size
            except OSError:
                continue
            out.append({"handle": handle,
                        "tool": rest[:-len(RESULT_SUFFIX)] or "unknown",
                        "chars": size})
        return out

    def put(self, tool: str, text: str) -> str | None:
        """Store ``text`` and return the message body to put in the context.

        ``None`` means "not stored, use what you already have" — no store, a
        result small enough not to need one, or a write that failed.  One
        return value for all three because the caller does the same thing with
        each: fall back to the ordinary path.  A bookkeeping failure must never
        turn a tool result into an error.
        """
        if not self.enabled or not isinstance(text, str) or len(text) <= self.threshold:
            return None
        handle = f"{self._next:0{_HANDLE_DIGITS}d}"
        name = f"{handle}-{_UNSAFE_TOOL.sub('_', str(tool or 'tool'))[:48]}{RESULT_SUFFIX}"
        path = self.results_dir / name
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as e:
            logger.warning("result not stored for %s: %s", tool, e)
            return None
        self._next += 1
        self._prune()
        return self.preview(handle, tool, text)

    def preview(self, handle: str, tool: str, text: str) -> str:
        """The message body for a stored result: header, head, how to get more."""
        head = text[: self.preview_chars]
        return (
            f"[result:{handle} · {tool} · {len(text):,} chars · "
            f"showing the first {len(head):,}]\n"
            f"{head.rstrip()}\n"
            f'{CONTINUATION_PREFIX}rest of this result: result_read("{handle}", '
            f'offset={len(head)}) or result_grep("{handle}", "pattern")]'
        )

    def decay_trailer(self, handle: str, offset: int) -> str:
        """The marker a decayed result ends with when it has a handle.

        The point of the whole phase in one line: the old marker told the model
        to run the tool again, which is a network round trip to recover bytes
        this directory is already holding.
        """
        return (f'{CONTINUATION_PREFIX}{DECAY_MARK} — '
                f'result_read("{handle}", offset={offset}) for the rest, or '
                f'result_grep("{handle}", "pattern")]')

    # ── the tools ───────────────────────────────────────────────────────────

    def read(self, handle: str, offset: int = 0,
             limit: int = DEFAULT_READ_CHARS) -> str:
        """A window of a stored result."""
        text = self._text(handle)
        if isinstance(text, str) and text.startswith("Error:"):
            return text
        offset = max(0, int(offset or 0))
        limit = max(1, min(int(limit or DEFAULT_READ_CHARS), MAX_READ_CHARS))
        total = len(text)
        if offset >= total:
            return _error(
                f"offset {offset:,} is past the end of result {handle} "
                f"({total:,} chars). Read from an offset below that.")
        window = text[offset:offset + limit]
        end = offset + len(window)
        header = (f"[result:{handle} · {total:,} chars · "
                  f"showing {offset:,}–{end:,}]")
        if end < total:
            return (f"{header}\n{window}\n"
                    f'{CONTINUATION_PREFIX}more: result_read("{handle}", offset={end})]')
        return f"{header}\n{window}\n[end of result {handle}]"

    def grep(self, handle: str, pattern: str, context: int = 3) -> str:
        """Matching lines of a stored result, with surrounding context.

        The reason this exists next to ``read``: a model looking for one number
        in a 200k-character result would otherwise page through it a window at
        a time, and every window is a turn.
        """
        text = self._text(handle)
        if isinstance(text, str) and text.startswith("Error:"):
            return text
        if not isinstance(pattern, str) or not pattern.strip():
            return _error("result_grep needs a pattern to look for.")
        context = max(0, min(int(context or 0), MAX_GREP_CONTEXT))
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # A model that meant a literal string and wrote something the regex
            # engine rejects gets what it meant, not a lecture about syntax.
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        lines = text.splitlines()
        hits = [i for i, line in enumerate(lines) if regex.search(line)]
        if not hits:
            return (f"[result:{handle} · no line matches {pattern!r}. "
                    f'Read it instead: result_read("{handle}")]')

        # Overlapping context windows are merged, so a cluster of matches reads
        # as one passage rather than the same lines repeated per hit.
        spans: list[list[int]] = []
        for i in hits[:MAX_GREP_MATCHES]:
            lo, hi = max(0, i - context), min(len(lines), i + context + 1)
            if spans and lo <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], hi)
            else:
                spans.append([lo, hi])

        out = [f"[result:{handle} · {len(hits)} matching line(s) for {pattern!r}"
               + (f", showing the first {MAX_GREP_MATCHES}]"
                  if len(hits) > MAX_GREP_MATCHES else "]")]
        chars = len(out[0])
        for lo, hi in spans:
            block = "\n".join(f"{n + 1}: {lines[n]}" for n in range(lo, hi))
            if chars + len(block) > MAX_GREP_CHARS:
                out.append(f'{CONTINUATION_PREFIX}output truncated — narrow the '
                           f'pattern, or read around a line with result_read("{handle}")]')
                break
            out.append(block)
            chars += len(block)
        return "\n--\n".join(out)

    def _text(self, handle: str):
        """The stored text, or an error string naming what does exist."""
        if not self.enabled:
            return _error("this run has no result store, so there is nothing to read.")
        path = self._path_for(handle)
        if path is None:
            saved = [r["handle"] for r in self.stored()]
            available = ", ".join(saved) if saved else "none stored yet"
            return _error(f"no stored result under handle '{handle}'. "
                          f"Stored handles: {available}.")
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("result %s could not be read: %s", handle, e)
            return _error(f"could not read result '{handle}': {e}")
