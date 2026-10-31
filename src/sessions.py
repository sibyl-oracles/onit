"""
Session index management for OnIt.

Maintains a lightweight JSON index (~/.onit/sessions/index.json) that maps
session IDs to human-readable tags, timestamps, and preview text.  The JSONL
history files are not touched — this module only manages metadata.
"""

import json
import os
import re
import time
from pathlib import Path
from datetime import datetime, timezone

DEFAULT_SESSIONS_DIR = os.path.expanduser("~/.onit/sessions")
INDEX_FILENAME = "index.json"

# A row range: digits on both sides of a single dash, either side optional,
# and nothing else.  Deliberately not "anything with a dash" -- session names
# are full of dashes -- and deliberately not tolerant of trailing words, so
# '2-4 7' (a range and a row) stays two identifiers instead of becoming 2-47.
_RANGE_RE = re.compile(r"\s*(\d*)\s*-\s*(\d*)\s*")


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


# Names that mean "no name yet" rather than a topic.  A session carrying one
# is still waiting for its first real message, so a later turn may replace it --
# unlike a name the user typed, which is never overwritten.
_PLACEHOLDER_TAGS = frozenset({"unnamed", "untitled", "new-session"})


def _has_topic(task: str) -> bool:
    """True when *task* carries a content word a name could be built from.

    A session is named from its first message, and the first message is often
    "hi".  Naming that session "hi" makes it unfindable an hour later, so a
    content-free opener passes without naming anything and the next real
    message gets to do it.
    """
    return any(w not in _TAG_STOPWORDS for w in _tag_words(task))


def _needs_name(tag, task: str, allow_rename: bool = True,
                named_by_user: bool = False) -> bool:
    """Whether this turn should (re)name a session currently called *tag*.

    *allow_rename* is False for the preview, which -- unlike the name -- may be
    replaced even when a person typed the name themselves: they chose what the
    session is called, not what its first message was.

    *named_by_user* settles the one case the tag alone cannot: a name someone
    typed that reads like a placeholder.  "untitled" from the auto-namer means
    "no name yet"; "untitled" typed by a person is the name they wanted.
    """
    if named_by_user and allow_rename:
        return False
    if not _has_topic(task):
        return False
    if not tag:
        return True
    if not allow_rename:
        return str(tag).lower() in _PLACEHOLDER_TAGS or not _has_topic(str(tag))
    return str(tag).lower() in _PLACEHOLDER_TAGS




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


# A user-chosen name is typed, quoted and copied, so it may carry spaces and
# mixed case -- unlike an auto-tag, which is a slug.  What it may not carry is
# a character that would break the one-line index it is stored in, or the
# terminal it is printed back to.
_TAG_FORBIDDEN = frozenset("\r\n\t\x00")
TAG_MAX_CHARS = 64


def normalize_tag(tag: str) -> str:
    """Clean a user-supplied session name, or return '' when it is unusable.

    Whitespace is collapsed rather than rejected: ``my  session`` is a typing
    accident, not a request for a name with a double space.  A control
    character becomes a space instead of vanishing -- dropping it would run the
    words on either side together, turning a stray tab into a different name.
    """
    if not tag:
        return ""
    spaced = "".join(" " if c in _TAG_FORBIDDEN else c for c in str(tag))
    cleaned = " ".join(spaced.split())
    return cleaned[:TAG_MAX_CHARS].strip()


def _tag_holder(tag: str, index: dict, exclude_sid: str | None = None) -> str | None:
    """The session already holding *tag*, or None.

    Compared case-insensitively because ``resolve_session`` matches that way:
    two names differing only in case would be one name you cannot tell apart
    and cannot reliably resolve.
    """
    wanted = tag.lower()
    for sid, meta in index.items():
        if sid == exclude_sid:
            continue
        if (meta.get("tag") or "").lower() == wanted:
            return sid
    return None


# ---- Public API ----

def name_for_new_session(tag: str | None,
                         sessions_dir: str = DEFAULT_SESSIONS_DIR) -> str:
    """The name a session created right now with *tag* would be given.

    A caller that says "call it X" needs to print the name the session will
    actually have, and that is not always *tag*: a name already held by another
    session gets a suffix.  Both the message and the creation go through this,
    so the printed name is the one `--resume-session` resolves.
    """
    name = normalize_tag(tag) if tag else ""
    if not name:
        return ""
    # An explicitly named session keeps the name the user chose.  A collision
    # is disambiguated rather than refused: the session is already being
    # created, and losing the name entirely would be worse than a suffix.
    return _ensure_unique_tag(name, _load_index(sessions_dir))


def register_session(session_id: str, sessions_dir: str = DEFAULT_SESSIONS_DIR,
                     tag: str | None = None) -> str:
    """Register a new session in the index, returning the name it got.

    A name given here is the name the session has: the caller prints it back
    and expects `--resume-session <that>` to find this session, so a name the
    index had to change is returned rather than left for the caller to guess.
    """
    index = _load_index(sessions_dir)
    name = name_for_new_session(tag, sessions_dir)
    index[session_id] = {
        "tag": name or None,
        "created": time.time(),
        "updated": time.time(),
        "preview": None,
        "turns": 0,
        # A name the user typed is theirs, even when it reads like a
        # placeholder.  Without this the auto-namer treats "untitled" as "no
        # name yet" and replaces it with a slug from the first message, so the
        # name given to --new-session is not the name the session keeps.
        "named_by_user": bool(name),
    }
    _save_index(index, sessions_dir)
    return name


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
    # The preview is the other half of recognising a session in the list, so it
    # follows the same rule as the name: a greeting is not what the session was
    # about, and the first message that says something replaces it.
    if task and _needs_name(meta.get("preview"), task, allow_rename=False):
        meta["preview"] = task[:120]
    if task and _needs_name(meta.get("tag"), task,
                            named_by_user=meta.get("named_by_user", False)):
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
    tag = normalize_tag(tag)
    if not tag:
        return "A session name cannot be empty."
    holder = _tag_holder(tag, index, exclude_sid=session_id)
    if holder is not None:
        return f"'{tag}' is already the name of session {holder[:8]}."
    index[session_id]["tag"] = tag
    # A rename is a person choosing the name, so the auto-namer must not
    # replace it later even if it reads like a placeholder.
    index[session_id]["named_by_user"] = True
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
    for sidecar in ("emails.json", "state.json"):
        path = os.path.join(sessions_dir, f"{session_id}.{sidecar}")
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
    return found


def exact_session_name(name: str,
                       sessions_dir: str = DEFAULT_SESSIONS_DIR) -> str | None:
    """The session whose name is exactly *name*, or None.

    Case-sensitive first, then case-insensitive.  The order matters: names are
    unique only *ignoring* case, so ``Notes`` and ``notes`` can never both
    exist -- but an older index can still hold two that differ only in case,
    and then the spelling the person typed has to win over the one it merely
    resembles.
    """
    index = _load_index(sessions_dir)
    wanted = name.lower()
    folded = None
    for sid, meta in index.items():
        stored = meta.get("tag") or ""
        if stored == name:
            return sid
        if folded is None and stored.lower() == wanted:
            folded = sid
    return folded


def match_session_name(name: str,
                       sessions_dir: str = DEFAULT_SESSIONS_DIR) -> tuple[str | None, list]:
    """Resolve a name the way a person typed it; return ``(session_id, candidates)``.

    Three tiers, most specific first:

    1. an exact match, case-sensitive -- the name exactly as it is stored;
    2. an exact match ignoring case;
    3. a unique prefix, so ``--resume-session chapter`` still finds
       ``chapter 3``.

    A prefix that matches more than one session resolves to nothing and hands
    back the candidates instead of choosing one.  Choosing by recency is how
    ``--delete`` removed the wrong session: seven sessions named
    ``marikina-river-water-level-latest`` and ``-2`` through ``-6`` all render
    inside the name column, every one of them matched what was on screen, and
    the newest was deleted without a word.  A name that cannot name one
    session is a question for the person who typed it, not a guess.
    """
    index = _load_index(sessions_dir)

    exact = exact_session_name(name, sessions_dir)
    if exact is not None:
        return exact, []

    wanted = name.lower()
    prefixed = []
    for sid, meta in index.items():
        stored = (meta.get("tag") or "").lower()
        if stored and stored.startswith(wanted):
            prefixed.append((sid, meta.get("tag") or ""))

    if len(prefixed) == 1:
        return prefixed[0][0], []
    if len(prefixed) > 1:
        # Ordered the way the listing is, so the candidates read like rows.
        prefixed.sort(key=lambda x: index[x[0]].get("updated", 0), reverse=True)
        return None, [{"session_id": sid, "tag": tag} for sid, tag in prefixed]
    return None, []


def find_session_by_tag(tag: str,
                        sessions_dir: str = DEFAULT_SESSIONS_DIR) -> str | None:
    """Find a session ID by its tag (case-insensitive prefix match).

    Returns the session_id or None.  An exact name wins over a longer name
    that merely starts with it, however old it is; among several *prefix*
    matches the most recently updated one is returned.  Callers that act on
    what they get back -- ``--delete`` above all -- should use
    :func:`match_session_name`, which refuses to choose between candidates.
    """
    sid, _ = match_session_name(tag, sessions_dir)
    return sid


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


def is_session_range(text: str,
                     sessions_dir: str = DEFAULT_SESSIONS_DIR) -> bool:
    """Whether *text* should be read as a row range rather than a name.

    The question the CLI asks before it prints an error, so that a range which
    names no rows is told apart from a name that matches nothing -- and the
    question :func:`resolve_session_range` asks before it expands, so the two
    cannot disagree about what a range is.
    """
    text = (text or "").strip()
    if not _RANGE_RE.fullmatch(text):
        return False
    first, last = _RANGE_RE.fullmatch(text).groups()
    if not first and not last:
        # A bare "-" is not a range, it is a typo.  Reading it as "everything"
        # is the one expansion that could empty the whole store.
        return False
    # A session may be *called* '2-4'.  The name wins, as it does over a row
    # number, so a chosen name is never unreachable.
    return exact_session_name(text, sessions_dir) is None


def resolve_session_range(text: str,
                          sessions_dir: str = DEFAULT_SESSIONS_DIR,
                          limit: int | None = None,
                          include_empty: bool = True) -> list | None:
    """Expand a row range like ``20-32`` into session ids, in row order.

    Returns None when *text* is not a range at all, so the caller can carry on
    treating it as a name.  That test comes before the expansion because a
    name is allowed to contain a dash -- ``marikina-river-water-level-latest``
    has four -- and only digits on both sides make it a range.

    A session *named* ``2-4`` also comes before the expansion, for the same
    reason a session named ``3`` comes before row 3: the name a person chose
    is the address they will reach for, and a range that shadows it would make
    it the one session on the listing that could not be named.

    Rows are the ones the listing prints: the same ``#`` a person read off
    the screen.  A range is inclusive at both ends, ``20-32`` is thirteen
    rows, because that is how a range is read out loud.  Either end may be
    open -- ``20-`` is row 20 to the last, ``-32`` is the first to row 32 --
    since "everything from here on" is the usual reason to want a range.

    A range is expanded against *one* listing, taken here, so its rows cannot
    drift from the rows that were confirmed.  An end past the last row is
    clamped: ``-32`` on a store of twenty means all twenty, not an error,
    while a start past the last row is empty -- there is nothing there.

    An end before its start returns ``[]``, which the caller reports as
    nothing being deleted.  Reversing them silently would delete rows the
    person did not name.
    """
    if not is_session_range(text, sessions_dir):
        return None
    match = _RANGE_RE.fullmatch((text or "").strip())
    first, last = match.group(1), match.group(2)
    rows = list_sessions(sessions_dir,
                         limit=1_000_000 if limit is None else limit,
                         include_empty=include_empty)
    start = int(first) if first else 1
    end = int(last) if last else len(rows)
    if end > len(rows):
        end = len(rows)
    if start < 1 or start > end:
        return []
    return [row["session_id"] for row in rows[start - 1:end]]


def resolve_session_choice(identifier: str,
                           sessions_dir: str = DEFAULT_SESSIONS_DIR,
                           limit: int | None = None,
                           include_empty: bool = True,
                           candidates: list | None = None) -> str | None:
    """Resolve what a person typed at a shell prompt to a session id.

    The order is deliberate, and it is what makes ``--delete <name>`` mean the
    session that has that name:

    1. a name that matches exactly (see :func:`match_session_name`);
    2. the ``#`` shown next to a row in the listing, so ``--rename 3`` and
       ``--delete 3`` act on the third line;
    3. a unique name prefix;
    4. ``last``/``-1``, the most recently updated session;
    5. a UUID or UUID prefix.

    A name comes before the row number because a session can be *called* ``3``
    or ``2026``: with the number first, ``--delete 2026`` was read as a row
    past the end and reported that no such session existed, while the session
    of that name sat right there in the listing.  ``last`` moves to fourth for
    the same reason, so a session named ``last`` can still be addressed as one.

    *candidates*, when given, is filled with the sessions a *prefix* matched
    when it matched more than one.  That is not a failure to report as "no
    session matches": the caller can say which ones were meant and let the
    person pick, which is the only honest answer when a name fits six rows.

    The listing prints every session, so the default view here is every
    session too -- a ``#`` has to mean the row it was printed next to.  The
    filtering defaults used to be the other way round, so the listing and
    the resume flag could disagree about what row 3 was.  *limit* and
    *include_empty* remain for callers that deliberately want a narrower
    view; ``None`` means no truncation.
    """
    text = (identifier or "").strip()
    if not text:
        return None

    sid, ambiguous = match_session_name(text, sessions_dir)
    if sid:
        return sid
    if ambiguous:
        if candidates is not None:
            candidates.extend(ambiguous)
        return None

    if text.isdigit():
        # The same view 'onit sessions' prints, or the number means a different
        # session than the row it was read off.
        rows = list_sessions(sessions_dir,
                             limit=1_000_000 if limit is None else limit,
                             include_empty=include_empty)
        n = int(text)
        if 1 <= n <= len(rows):
            return rows[n - 1]["session_id"]

    if text.lower() in ("last", "-1"):
        return get_last_session(sessions_dir)

    return find_session_by_id_prefix(text, sessions_dir)


def get_last_session(sessions_dir: str = DEFAULT_SESSIONS_DIR) -> str | None:
    """Return the most recently updated session that has something in it.

    A session that was opened and never used is still a row in the listing --
    it is the accidental one, and it needs a ``#`` to delete it by -- but it is
    not a conversation to continue.  It is nonetheless the *newest* entry by
    definition, so ranking purely by recency made a bare ``onit`` resume it:
    the person asked to continue where they left off and got an empty prompt.

    ``onit doctor`` made that happen on purpose.  It builds a real agent to
    check a real agent, then deletes the session it created -- and if it dies
    before the cleanup, the row stays, and every bare ``onit`` afterwards
    resumes it.  Skipping empty sessions here is the difference between "the
    last thing I did" and "the last thing that happened to the store".

    Only sessions with nothing at all are skipped.  A session whose name or
    preview was never set but which recorded a turn is a real conversation and
    keeps its place; so does one that is empty *now* when it is the only entry
    there is, because falling back to no session at all would lose it.
    """
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

    def _has_content(meta: dict) -> bool:
        return bool(meta.get("turns") or meta.get("preview") or meta.get("tag"))

    ranked = sorted(index.items(), key=lambda x: _effective_ts(x[0], x[1]),
                    reverse=True)
    for sid, meta in ranked:
        if _has_content(meta):
            return sid
    # Nothing has been said in any of them: the newest empty session is a
    # better answer than none, because it is the one the person is looking at.
    return ranked[0][0] if ranked else None


def list_sessions(sessions_dir: str = DEFAULT_SESSIONS_DIR,
                  limit: int | None = 20, owner: str | None = None,
                  include_empty: bool = True) -> list:
    """Return a list of sessions sorted by most recently updated.

    Each entry has: session_id, tag, created, updated, preview, turns.
    When *owner* is given, only sessions recorded as owned by that email are
    returned -- unowned (pre-auth) sessions stay hidden until claimed.
    *include_empty* keeps sessions that were opened and never used: they have
    no name, no preview and nothing to recall, and on a long-lived install they
    outnumber the real ones enough to bury them.
    *limit* is ``None`` for no truncation.  The sort happens before the slice,
    so a limit only ever shortens the list from the bottom: row *n* of the
    full listing is row *n* of any listing long enough to reach it.
    """
    index = _load_index(sessions_dir)

    # If index is empty, bootstrap from existing JSONL files on disk
    if not index:
        index = rebuild_index(sessions_dir)

    sessions = []
    for sid, meta in index.items():
        if owner is not None and meta.get("owner") != owner:
            continue
        if not include_empty and not meta.get("turns"):
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
    return sessions if limit is None else sessions[:limit]


def count_sessions(sessions_dir: str = DEFAULT_SESSIONS_DIR,
                   owner: str | None = None,
                   include_empty: bool = True) -> int:
    """How many sessions there are, ignoring *limit*."""
    return len(list_sessions(sessions_dir, limit=None, owner=owner,
                             include_empty=include_empty))


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


def empty_sessions(sessions_dir: str = DEFAULT_SESSIONS_DIR) -> list:
    """Every session that was opened and never used, newest first.

    "Never used" is exact: no name, no preview, no turn.  A session that was
    named by hand but never spoken to is not empty -- the name is the reason
    it exists -- and one that recorded a turn is a conversation whatever its
    name says.

    These are the rows a test run used to create in bulk: an agent built by a
    test that named no ``session_path`` registered itself in the real store.
    They are interleaved with the real sessions by date, so a row range cannot
    name them and deleting them one at a time is not a chore anyone finishes.
    """
    return [row for row in list_sessions(sessions_dir, limit=None)
            if not row.get("tag") and not row.get("preview")
            and not row.get("turns")]


def clear_sessions(sessions_dir: str = DEFAULT_SESSIONS_DIR) -> int:
    """Delete all session JSONL files and the index.  Returns count of files removed."""
    sessions_path = Path(sessions_dir)
    count = 0
    for jsonl_file in sessions_path.glob("*.jsonl"):
        jsonl_file.unlink()
        count += 1
    # The sidecars go with the histories they belong to, or a recycled session
    # id inherits another session's run state.  Not counted: the count is of
    # conversations cleared, and a session with a state file is still one.
    for sidecar in list(sessions_path.glob("*.state.json")) + \
            list(sessions_path.glob("*.emails.json")):
        try:
            sidecar.unlink()
        except OSError:
            pass
    # Remove the index file
    idx = _index_path(sessions_dir)
    if os.path.isfile(idx):
        os.remove(idx)
    return count


def format_sessions_table(sessions: list, total: int | None = None) -> str:
    """Format session list as a human-readable table for CLI output.

    The ``#`` column is not decoration: ``onit sessions --rename 3 <name>``
    and ``onit --resume-session 3`` take it, so a session can be acted on
    without copying a
    UUID out of the terminal.  *total* is the number of sessions that exist,
    when the list shown is a limited view of them.

    The Name column is not truncated either.  It used to be cut to 32
    characters, and the names that needed the room most are the ones that
    differ only at the end: ``marikina-river-water-level-latest`` and its
    ``-2`` through ``-6`` all printed as the same 32 characters, so a person
    deleting "the one I can see" deleted whichever of the six happened to be
    newest.  A name is only useful if typing it back selects the session it
    was printed next to.
    """
    if not sessions:
        return ("No sessions found.\n"
                "Start one with: onit --new-session")

    width = max((len(s.get("tag") or "(unnamed)") for s in sessions),
                default=len("Name"))
    width = max(width, len("Name"))

    lines = []
    lines.append(f"{'#':<4} {'Name':<{width}} {'Turns':<6} {'Updated':<17} Preview")
    lines.append("-" * (4 + 1 + width + 1 + 6 + 1 + 17 + 1 + 38))
    for i, s in enumerate(sessions, 1):
        tag = s.get("tag") or "(unnamed)"
        turns = s.get("turns", 0)
        updated = s.get("updated")
        if updated:
            updated_str = datetime.fromtimestamp(
                updated, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        else:
            updated_str = "unknown"
        preview = (s.get("preview") or "").replace("\n", " ")[:38]
        lines.append(f"{i:<4} {tag:<{width}} {turns:<6} {updated_str:<17} {preview}")

    lines.append("")
    if total is not None and total > len(sessions):
        lines.append(f"Showing {len(sessions)} of {total} sessions.")
    lines.append("Continue: onit --resume-session <#>   "
                 "Rename: onit sessions --rename <#> <name>")
    lines.append("Delete: onit sessions --delete <#> [<#> ...] or <a>-<b>")
    lines.append("Start a named session: onit --new-session <name>")
    return "\n".join(lines)
