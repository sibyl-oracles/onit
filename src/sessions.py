"""
Session index management for OnIt.

Maintains a lightweight JSON index (~/.onit/sessions/index.json) that maps
session IDs to human-readable tags, timestamps, and preview text.  The JSONL
history files are not touched — this module only manages metadata.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone

DEFAULT_SESSIONS_DIR = os.path.expanduser("~/.onit/sessions")
INDEX_FILENAME = "index.json"


def _index_path(sessions_dir: str = DEFAULT_SESSIONS_DIR) -> str:
    return os.path.join(sessions_dir, INDEX_FILENAME)


def _load_index(sessions_dir: str = DEFAULT_SESSIONS_DIR) -> dict:
    """Load the session index from disk.  Returns {session_id: metadata}."""
    path = _index_path(sessions_dir)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_index(index: dict, sessions_dir: str = DEFAULT_SESSIONS_DIR) -> None:
    os.makedirs(sessions_dir, exist_ok=True)
    path = _index_path(sessions_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def _first_task_from_jsonl(jsonl_path: str) -> str | None:
    """Read the first user task from a JSONL session file."""
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if "task" in entry:
                        return entry["task"]
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _turn_count_from_jsonl(jsonl_path: str) -> int:
    """Count the number of turns in a JSONL session file."""
    count = 0
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except OSError:
        pass
    return count


# Filler that carries no topic: politeness, pronouns, auxiliaries, articles,
# prepositions and question words.  Dropping these turns "can you please help
# me fix the login bug" into "fix-login-bug".
_TAG_STOPWORDS = frozenset("""
a an the and or but if then than so as of to in on at for with from by into
about over under again between within without across after before during
through out up down off onto upon per via
is are was were be been being am do does did doing done
can could would should will shall may might must have has had
i me my mine we us our ours you your yours it its they them their he she his her
this that these those there here what which who whom whose when where why how
please pls kindly hi hey hello thanks thank just some any all very really
actually maybe let lets like want wants need needs help sure okay ok also
http https www com org net io co html htm
""".split())

# A sidebar entry is only a couple of inches wide — four words is the most that
# stays readable, and the character cap keeps long identifiers from blowing it.
_TAG_MAX_WORDS = 4
_TAG_MAX_CHARS = 32
_TAG_MAX_WORD_LEN = 20


def _tag_words(task: str) -> list[str]:
    """Extract ordered, de-duplicated content words from *task*.

    Only the first line/sentence is considered: long multi-paragraph tasks
    describe their subject up front, and the rest is detail.
    """
    head = task.strip().split("\n", 1)[0]
    for sep in (". ", "? ", "! ", "; "):
        head = head.split(sep, 1)[0]

    raw, seen = [], set()
    for token in head.split():
        # Keep internal dashes (well-known, re-run) but drop other punctuation,
        # which also strips URLs and paths down to their readable parts.
        word = "".join(c if c.isalnum() or c == "-" else " " for c in token)
        for part in word.split():
            part = part.strip("-").lower()
            # Overlong tokens are hashes, URLs or base64 — never descriptive.
            if not part or len(part) > _TAG_MAX_WORD_LEN or part in seen:
                continue
            seen.add(part)
            raw.append(part)

    content = [w for w in raw if w not in _TAG_STOPWORDS]
    # A task made entirely of stopwords ("what is this?") still needs a name.
    return content or raw


def _fit_tag(words: list[str]) -> str:
    """Join *words* into a slug that respects the word and character caps."""
    tag = ""
    for word in words[:_TAG_MAX_WORDS]:
        candidate = f"{tag}-{word}" if tag else word
        if tag and len(candidate) > _TAG_MAX_CHARS:
            break
        tag = candidate
    return tag[:_TAG_MAX_CHARS].strip("-")


def _make_auto_tag(task: str) -> str:
    """Generate a short, descriptive auto-tag from the first user message."""
    return _fit_tag(_tag_words(task)) or "unnamed"


def _auto_tag_candidates(task: str) -> list[str]:
    """Auto-tag options for *task*, best first.

    When the preferred name is taken, later candidates slide the word window
    forward so the sibling session gets a name that actually says something
    different, rather than the same name with a number bolted on.
    """
    words = _tag_words(task)
    candidates, seen = [], set()
    for start in range(max(len(words) - 1, 1)):
        tag = _fit_tag(words[start:])
        if tag and tag not in seen:
            seen.add(tag)
            candidates.append(tag)
    return candidates or ["unnamed"]


def _existing_tags(index: dict, exclude_sid: str | None = None) -> set[str]:
    """Return the set of all tags in the index (lowercased), optionally excluding one session."""
    tags = set()
    for sid, meta in index.items():
        if sid == exclude_sid:
            continue
        t = meta.get("tag")
        if t:
            tags.add(t.lower())
    return tags


def _ensure_unique_tag(tag: str, index: dict, exclude_sid: str | None = None,
                       alternates: list[str] | None = None) -> str:
    """Return the first free name for a session.

    *tag* is preferred; *alternates* are tried next so a collision produces a
    differently-worded name instead of a numbered one.  A numeric suffix is
    the last resort.
    """
    used = _existing_tags(index, exclude_sid)
    for candidate in [tag, *(alternates or [])]:
        if candidate and candidate.lower() not in used:
            return candidate
    for i in range(2, 10000):
        candidate = f"{tag}-{i}"
        if candidate.lower() not in used:
            return candidate
    return f"{tag}-{int(time.time())}"


# ---- Public API ----

def register_session(session_id: str, sessions_dir: str = DEFAULT_SESSIONS_DIR,
                     tag: str | None = None) -> None:
    """Register a new session in the index."""
    index = _load_index(sessions_dir)
    index[session_id] = {
        "tag": tag,
        "created": time.time(),
        "updated": time.time(),
        "preview": None,
        "turns": 0,
    }
    _save_index(index, sessions_dir)


def update_session(session_id: str, task: str | None = None,
                   sessions_dir: str = DEFAULT_SESSIONS_DIR) -> None:
    """Update session metadata after a new turn.  Auto-tags if no tag set."""
    index = _load_index(sessions_dir)
    meta = index.get(session_id)
    if meta is None:
        # Session was created before index existed — bootstrap it
        meta = {
            "tag": None,
            "created": time.time(),
            "updated": time.time(),
            "preview": None,
            "turns": 0,
        }
    meta["updated"] = time.time()
    meta["turns"] = meta.get("turns", 0) + 1
    if task and not meta.get("preview"):
        meta["preview"] = task[:120]
    if task and not meta.get("tag"):
        options = _auto_tag_candidates(task)
        meta["tag"] = _ensure_unique_tag(options[0], index, exclude_sid=session_id,
                                         alternates=options[1:])
    index[session_id] = meta
    _save_index(index, sessions_dir)


def get_session_owner(session_id: str,
                      sessions_dir: str = DEFAULT_SESSIONS_DIR) -> str | None:
    """Return the owner (authenticated email) recorded for a session, if any."""
    return _load_index(sessions_dir).get(session_id, {}).get("owner")


def set_session_owner(session_id: str, owner: str,
                      sessions_dir: str = DEFAULT_SESSIONS_DIR) -> None:
    """Record the owner (authenticated email) of a session, bootstrapping
    an index entry if the session predates the index."""
    index = _load_index(sessions_dir)
    meta = index.get(session_id)
    if meta is None:
        meta = {
            "tag": None,
            "created": time.time(),
            "updated": time.time(),
            "preview": None,
            "turns": 0,
        }
    meta["owner"] = owner
    index[session_id] = meta
    _save_index(index, sessions_dir)


def tag_session(session_id: str, tag: str,
                sessions_dir: str = DEFAULT_SESSIONS_DIR) -> bool | str:
    """Set or overwrite the tag for a session.

    Returns True on success, False if session not found, or an error string
    if the tag is already taken by another session.
    """
    index = _load_index(sessions_dir)
    if session_id not in index:
        return False
    # Check uniqueness (exclude the session being tagged)
    used = _existing_tags(index, exclude_sid=session_id)
    if tag.lower() in used:
        return f"Tag '{tag}' is already in use by another session."
    index[session_id]["tag"] = tag
    _save_index(index, sessions_dir)
    return True


def delete_session(session_id: str,
                   sessions_dir: str = DEFAULT_SESSIONS_DIR) -> bool:
    """Delete a session's JSONL history file and its index entry.

    Returns True if either the file or the index entry existed.
    """
    found = False
    index = _load_index(sessions_dir)
    if session_id in index:
        del index[session_id]
        _save_index(index, sessions_dir)
        found = True
    jsonl_path = os.path.join(sessions_dir, f"{session_id}.jsonl")
    if os.path.isfile(jsonl_path):
        try:
            os.remove(jsonl_path)
            found = True
        except OSError:
            pass
    # Sidecars written alongside the history (e.g. <sid>.emails.json, the web
    # UI's grounded-address list) go with it — otherwise they outlive the
    # session and a recycled id would inherit them.
    for sidecar in ("emails.json",):
        path = os.path.join(sessions_dir, f"{session_id}.{sidecar}")
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
    return found


def find_session_by_tag(tag: str,
                        sessions_dir: str = DEFAULT_SESSIONS_DIR) -> str | None:
    """Find a session ID by its tag (case-insensitive prefix match).

    Returns the session_id or None.  If multiple match, returns the most
    recently updated one.
    """
    index = _load_index(sessions_dir)
    tag_lower = tag.lower()
    candidates = []
    for sid, meta in index.items():
        session_tag = (meta.get("tag") or "").lower()
        if session_tag == tag_lower or session_tag.startswith(tag_lower):
            candidates.append((sid, meta))
    if not candidates:
        return None
    # Sort by updated time descending
    candidates.sort(key=lambda x: x[1].get("updated", 0), reverse=True)
    return candidates[0][0]


def find_session_by_id_prefix(prefix: str,
                              sessions_dir: str = DEFAULT_SESSIONS_DIR) -> str | None:
    """Find a session by UUID prefix match."""
    index = _load_index(sessions_dir)
    prefix_lower = prefix.lower()
    for sid in index:
        if sid.lower().startswith(prefix_lower):
            return sid
    # Also check JSONL files on disk (for sessions created before indexing)
    sessions_path = Path(sessions_dir)
    for jsonl_file in sessions_path.glob("*.jsonl"):
        if jsonl_file.stem.lower().startswith(prefix_lower):
            return jsonl_file.stem
    return None


def resolve_session(identifier: str,
                    sessions_dir: str = DEFAULT_SESSIONS_DIR) -> str | None:
    """Resolve a tag, UUID, or UUID prefix to a session_id.

    Special value 'last' returns the most recently updated session.
    """
    if identifier.lower() == "last":
        return get_last_session(sessions_dir)
    # Try tag match first
    result = find_session_by_tag(identifier, sessions_dir)
    if result:
        return result
    # Try UUID / prefix match
    result = find_session_by_id_prefix(identifier, sessions_dir)
    if result:
        return result
    return None


def get_last_session(sessions_dir: str = DEFAULT_SESSIONS_DIR) -> str | None:
    """Return the most recently updated session ID."""
    index = _load_index(sessions_dir)
    sessions_path = Path(sessions_dir)

    if not index:
        jsonl_files = list(sessions_path.glob("*.jsonl"))
        if not jsonl_files:
            return None
        jsonl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return jsonl_files[0].stem

    # Use max(index updated, file mtime) so sessions modified by text-mode chat
    # (which doesn't call update_session) are still ranked by actual last use.
    def _effective_ts(sid: str, meta: dict) -> float:
        idx_ts = meta.get("updated", 0)
        try:
            return max(idx_ts, (sessions_path / f"{sid}.jsonl").stat().st_mtime)
        except OSError:
            return idx_ts

    sorted_sessions = sorted(index.items(),
                             key=lambda x: _effective_ts(x[0], x[1]),
                             reverse=True)
    return sorted_sessions[0][0] if sorted_sessions else None


def list_sessions(sessions_dir: str = DEFAULT_SESSIONS_DIR,
                  limit: int = 20, owner: str | None = None) -> list[dict]:
    """Return a list of sessions sorted by most recently updated.

    Each entry has: session_id, tag, created, updated, preview, turns.
    When *owner* is given, only sessions recorded as owned by that email are
    returned — unowned (pre-auth) sessions stay hidden until claimed.
    """
    index = _load_index(sessions_dir)

    # If index is empty, bootstrap from existing JSONL files on disk
    if not index:
        index = rebuild_index(sessions_dir)

    sessions = []
    for sid, meta in index.items():
        if owner is not None and meta.get("owner") != owner:
            continue
        sessions.append({
            "session_id": sid,
            "tag": meta.get("tag"),
            "created": meta.get("created"),
            "updated": meta.get("updated"),
            "preview": meta.get("preview"),
            "turns": meta.get("turns", 0),
        })
    sessions.sort(key=lambda x: x.get("updated") or 0, reverse=True)
    return sessions[:limit]


def rebuild_index(sessions_dir: str = DEFAULT_SESSIONS_DIR) -> dict:
    """Scan existing JSONL files and rebuild the index from scratch."""
    sessions_path = Path(sessions_dir)
    index = _load_index(sessions_dir)
    for jsonl_file in sessions_path.glob("*.jsonl"):
        sid = jsonl_file.stem
        if sid in index:
            continue
        first_task = _first_task_from_jsonl(str(jsonl_file))
        turns = _turn_count_from_jsonl(str(jsonl_file))
        if turns == 0:
            continue  # skip empty sessions
        stat = jsonl_file.stat()
        options = _auto_tag_candidates(first_task) if first_task else []
        index[sid] = {
            "tag": _ensure_unique_tag(options[0], index, exclude_sid=sid,
                                      alternates=options[1:]) if options else None,
            "created": stat.st_birthtime if hasattr(stat, 'st_birthtime') else stat.st_mtime,
            "updated": stat.st_mtime,
            "preview": (first_task[:120] if first_task else None),
            "turns": turns,
        }
    _save_index(index, sessions_dir)
    return index


def clear_sessions(sessions_dir: str = DEFAULT_SESSIONS_DIR) -> int:
    """Delete all session JSONL files and the index.  Returns count of files removed."""
    sessions_path = Path(sessions_dir)
    count = 0
    for jsonl_file in sessions_path.glob("*.jsonl"):
        jsonl_file.unlink()
        count += 1
    # Remove the index file
    idx = _index_path(sessions_dir)
    if os.path.isfile(idx):
        os.remove(idx)
    return count


def format_sessions_table(sessions: list[dict]) -> str:
    """Format session list as a human-readable table for CLI output."""
    if not sessions:
        return "No sessions found."

    lines = []
    lines.append(f"{'#':<4} {'Tag':<30} {'Turns':<6} {'Updated':<20} {'Preview'}")
    lines.append("-" * 100)
    for i, s in enumerate(sessions, 1):
        tag = s.get("tag") or s["session_id"][:8]
        turns = s.get("turns", 0)
        updated = s.get("updated")
        if updated:
            updated_str = datetime.fromtimestamp(updated, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        else:
            updated_str = "unknown"
        preview = (s.get("preview") or "")[:40]
        lines.append(f"{i:<4} {tag:<30} {turns:<6} {updated_str:<20} {preview}")

    lines.append("")
    lines.append("Resume a session with: onit --resume <tag>")
    return "\n".join(lines)
