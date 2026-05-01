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


def _make_auto_tag(task: str) -> str:
    """Generate a short auto-tag from the first user message."""
    # Take the first ~6 words, lowercase, replace spaces with dashes
    words = task.strip().split()[:6]
    tag = "-".join(w.lower() for w in words)
    # Remove non-alphanumeric characters except dashes
    tag = "".join(c for c in tag if c.isalnum() or c == "-")
    # Truncate to 50 chars
    return tag[:50] or "unnamed"


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


def _ensure_unique_tag(tag: str, index: dict, exclude_sid: str | None = None) -> str:
    """Append a numeric suffix if *tag* already exists in the index."""
    used = _existing_tags(index, exclude_sid)
    if tag.lower() not in used:
        return tag
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
        meta["tag"] = _ensure_unique_tag(_make_auto_tag(task), index, exclude_sid=session_id)
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
                  limit: int = 20) -> list[dict]:
    """Return a list of sessions sorted by most recently updated.

    Each entry has: session_id, tag, created, updated, preview, turns.
    """
    index = _load_index(sessions_dir)

    # If index is empty, bootstrap from existing JSONL files on disk
    if not index:
        index = rebuild_index(sessions_dir)

    sessions = []
    for sid, meta in index.items():
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
        index[sid] = {
            "tag": _ensure_unique_tag(_make_auto_tag(first_task), index, exclude_sid=sid) if first_task else None,
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
