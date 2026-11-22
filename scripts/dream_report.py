#!/usr/bin/env python3
"""Measure the dreaming pass on its own terms, from the artifacts it leaves.

The A/B gate (``onit learn --eval-run --wiki-ab``) asks whether the wiki
moves task outcomes.  This script asks the questions that decide whether
the pass itself is sound, without running a model:

  1. Extraction yield   -- candidates/verified/rejected per pass, from
                           ``state.json`` and ``dream.log``.
  2. Provenance integrity -- every fact's ``session#line`` citation is
                           resolved against the session files on disk, and
                           the cited line is re-checked for a shared
                           content token with the fact (the same check the
                           extractor applies).  A wiki whose provenance
                           does not resolve is a wiki that cannot be
                           audited.
  3. Hygiene            -- pages past their TTL, pages past the prune age,
                           open contradictions, index size vs the cap,
                           facts per page, kinds.
  4. Cost               -- seconds and counters from the last pass.

Usage:

    python3 scripts/dream_report.py PATH/TO/memory-dir
    python3 scripts/dream_report.py PATH/TO/memory-dir --json

``PATH/TO/memory-dir`` is the directory holding ``wiki/``, ``INDEX.md``,
``state.json`` and ``dream.log`` -- e.g.
``tmp/eval-runs/<run>/store/wiki-on/memory``.  The session and trajectory
directories are resolved relative to its parent (the store root).

Read-only.  Exits 0 even when the store is empty or broken; the report
says so.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timezone

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from onit.learn._search import tokenize as _tokenize
except Exception:  # pragma: no cover - fallback for a bare checkout
    _TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

    def _tokenize(text):
        return _TOKEN_RE.findall(str(text or "").lower())


CHARS_PER_TOKEN = 4
MAX_INDEX_LINES = 200
MAX_FACTS_PER_PAGE = 60
DEFAULT_TTL_DAYS = 30
DEFAULT_PRUNE_AFTER_DAYS = 45


def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return default
    if default is None:
        return data
    return data if isinstance(data, type(default)) else default


def _read_pages(wiki_dir):
    pages = []
    try:
        names = sorted(os.listdir(wiki_dir))
    except OSError:
        return pages
    for name in names:
        if not name.endswith(".json") or name.endswith(".prev.json"):
            continue
        page = _load_json(os.path.join(wiki_dir, name), None)
        if isinstance(page, dict) and page.get("slug"):
            pages.append(page)
    return pages


def _transcript_lines(records):
    """Numbered lines exactly as ``dream._transcript_text`` numbers them."""
    lines = []
    for record in records:
        task = " ".join(str(record.get("task") or "").split())
        response = " ".join(str(record.get("response") or "").split())
        if task:
            lines.append(task)
        if response:
            lines.append(response[:1200])
    return lines


def _load_sessions(store_root):
    """session_id -> list of records, from both on-disk layouts."""
    sessions = {}

    def add(directory, shape):
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            return
        for name in names:
            if not name.endswith(".jsonl"):
                continue
            sid = name[: -len(".jsonl")]
            records = []
            try:
                with open(os.path.join(directory, name), "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(rec, dict):
                            continue
                        if shape == "replay":
                            if not {"task", "response", "timestamp"} <= set(rec):
                                continue
                            if rec.get("response") is None:
                                continue
                        else:
                            if rec.get("response") is None:
                                continue
                        records.append(rec)
            except OSError:
                continue
            if records:
                sessions.setdefault(sid, records)

    add(os.path.join(store_root, "trajectories"), "trajectory")
    add(os.path.join(store_root, "sessions"), "replay")
    return sessions


def _days_since(day):
    try:
        then = date.fromisoformat(str(day)[:10])
    except (TypeError, ValueError):
        return 10**6
    return (date.today() - then).days


def _parse_log(log_path):
    passes = []
    current = None
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line.startswith("== pass "):
                    if current:
                        passes.append(current)
                    current = {"header": line, "lines": []}
                elif current is not None:
                    current["lines"].append(line)
        if current:
            passes.append(current)
    except OSError:
        pass
    return passes


def build_report(memory_dir):
    store_root = os.path.dirname(os.path.abspath(memory_dir))
    wiki_dir = os.path.join(memory_dir, "wiki")
    state = _load_json(os.path.join(memory_dir, "state.json"), {})
    pages = _read_pages(wiki_dir)
    sessions = _load_sessions(store_root)
    log_passes = _parse_log(os.path.join(memory_dir, "dream.log"))

    facts = [f for p in pages for f in (p.get("facts") or [])]
    prov_entries = [p for f in facts for p in (f.get("provenance") or [])]

    # Provenance resolution + re-verification.
    resolved = 0
    unresolved_session = 0
    bad_line = 0
    unparseable = 0
    reverified = 0
    for entry in prov_entries:
        if "#" not in entry:
            unparseable += 1
            continue
        sid, _, line_s = entry.rpartition("#")
        try:
            line_no = int(line_s)
        except ValueError:
            unparseable += 1
            continue
        records = sessions.get(sid)
        if records is None:
            unresolved_session += 1
            continue
        lines = _transcript_lines(records)
        if not (1 <= line_no <= len(lines)):
            bad_line += 1
            continue
        resolved += 1
        # Re-verify: does the cited line share a content token with the fact?
        # (The fact text is not available here per-entry, so this is done
        # per fact below; this branch only counts resolvable citations.)

    # Per-fact re-verification against its cited lines.
    fact_checks = {"checked": 0, "supported": 0, "unsupported": 0}
    for fact in facts:
        text = str(fact.get("text") or "")
        content = [t for t in _tokenize(text) if len(t) > 3]
        if not content:
            continue
        cited_lines = []
        for entry in fact.get("provenance") or []:
            if "#" not in entry:
                continue
            sid, _, line_s = entry.rpartition("#")
            try:
                line_no = int(line_s)
            except ValueError:
                continue
            records = sessions.get(sid)
            if not records:
                continue
            lines = _transcript_lines(records)
            if 1 <= line_no <= len(lines):
                cited_lines.append(lines[line_no - 1])
        if not cited_lines:
            continue
        hay = set(_tokenize(" ".join(cited_lines)))
        fact_checks["checked"] += 1
        if any(t in hay for t in content):
            fact_checks["supported"] += 1
        else:
            fact_checks["unsupported"] += 1

    # Hygiene.
    ttl = DEFAULT_TTL_DAYS
    prune_after = DEFAULT_PRUNE_AFTER_DAYS
    stale = sum(1 for p in pages if _days_since(p.get("verified")) > ttl)
    prunable = sum(1 for p in pages if _days_since(p.get("updated")) > prune_after)
    open_contradictions = sum(len(p.get("contradictions") or []) for p in pages)
    overfull = sum(1 for p in pages if len(p.get("facts") or []) > MAX_FACTS_PER_PAGE)
    empty_pages = sum(1 for p in pages if not (p.get("facts") or []))
    kinds = {}
    for f in facts:
        kinds[f.get("kind", "?")] = kinds.get(f.get("kind", "?"), 0) + 1

    index_path = os.path.join(memory_dir, "INDEX.md")
    try:
        with open(index_path, "r", encoding="utf-8") as fh:
            index_text = fh.read()
        index_lines = index_text.count("\n") + (0 if index_text.endswith("\n") else 1)
        index_bytes = len(index_text.encode("utf-8"))
    except OSError:
        index_text, index_lines, index_bytes = "", 0, 0

    last_stats = state.get("last_stats") or {}
    report = {
        "memory_dir": memory_dir,
        "store_root": store_root,
        "passes": state.get("passes"),
        "watermark_ts": state.get("watermark_ts"),
        "last_run": state.get("last_run"),
        "last_stats": last_stats,
        "log_passes": len(log_passes),
        "pages": len(pages),
        "facts": len(facts),
        "facts_per_page": round(len(facts) / len(pages), 2) if pages else 0.0,
        "fact_kinds": kinds,
        "fact_chars": sum(len(str(f.get("text") or "")) for f in facts),
        "empty_pages": empty_pages,
        "pages_over_fact_cap": overfull,
        "open_contradictions": open_contradictions,
        "stale_pages_ttl": stale,
        "prunable_pages": prunable,
        "index_lines": index_lines,
        "index_bytes": index_bytes,
        "index_tokens_est": index_bytes // CHARS_PER_TOKEN,
        "index_line_cap": MAX_INDEX_LINES,
        "provenance": {
            "entries": len(prov_entries),
            "resolved": resolved,
            "unresolved_session": unresolved_session,
            "bad_line": bad_line,
            "unparseable": unparseable,
            "resolved_pct": round(100.0 * resolved / len(prov_entries), 1) if prov_entries else None,
        },
        "fact_reverification": fact_checks,
        "sessions_seen": len(sessions),
    }
    return report


def render(report):
    p = []
    p.append("dream report: %s" % report["memory_dir"])
    p.append("")
    p.append("pass")
    p.append("  passes recorded      : %s" % report["passes"])
    p.append("  watermark            : %s" % report["watermark_ts"])
    p.append("  last run             : %s" % report["last_run"])
    p.append("  passes in dream.log  : %d" % report["log_passes"])
    ls = report["last_stats"]
    if ls:
        p.append("  last pass            : sessions=%s candidates=%s verified=%s rejected=%s"
                 % (ls.get("sessions"), ls.get("candidates"), ls.get("verified"), ls.get("rejected")))
        p.append("                         parsed_items=%s empty_replies=%s pages_created=%s pages_updated=%s"
                 % (ls.get("parsed_items"), ls.get("empty_replies"), ls.get("pages_created"), ls.get("pages_updated")))
        p.append("                         pruned=%s contradictions_cleared=%s seconds=%s"
                 % (ls.get("pruned"), ls.get("contradictions_cleared"), ls.get("seconds")))
    p.append("")
    p.append("wiki")
    p.append("  pages                : %d (%d empty, %d over the %d-fact cap)"
             % (report["pages"], report["empty_pages"], report["pages_over_fact_cap"], MAX_FACTS_PER_PAGE))
    p.append("  facts                : %d (%.2f/page, %d chars) kinds=%s"
             % (report["facts"], report["facts_per_page"], report["fact_chars"], report["fact_kinds"]))
    p.append("  open contradictions  : %d" % report["open_contradictions"])
    p.append("  stale (>%dd)         : %d" % (DEFAULT_TTL_DAYS, report["stale_pages_ttl"]))
    p.append("  prunable (>%dd)      : %d" % (DEFAULT_PRUNE_AFTER_DAYS, report["prunable_pages"]))
    p.append("  INDEX.md             : %d lines, %d bytes (~%d tokens, cap %d lines)"
             % (report["index_lines"], report["index_bytes"], report["index_tokens_est"], report["index_line_cap"]))
    p.append("")
    prov = report["provenance"]
    p.append("provenance")
    p.append("  entries              : %d" % prov["entries"])
    p.append("  resolved             : %d (%s%%)" % (prov["resolved"], prov["resolved_pct"]))
    p.append("  unresolvable session : %d" % prov["unresolved_session"])
    p.append("  line out of range    : %d" % prov["bad_line"])
    p.append("  unparseable          : %d" % prov["unparseable"])
    fv = report["fact_reverification"]
    p.append("  re-verified          : %d/%d supported by their cited lines (%d unsupported)"
             % (fv["supported"], fv["checked"], fv["unsupported"]))
    p.append("  sessions on disk     : %d" % report["sessions_seen"])
    p.append("")
    p.append("reading")
    if prov["entries"] and prov["resolved"] == 0:
        p.append("  PROVENANCE IS BROKEN: no citation resolves. The wiki cannot be audited.")
    elif prov["entries"] and prov["resolved"] < prov["entries"]:
        p.append("  provenance is partial: %d of %d citations do not resolve."
                 % (prov["entries"] - prov["resolved"], prov["entries"]))
    else:
        p.append("  provenance resolves.")
    if report["open_contradictions"]:
        p.append("  %d open contradiction(s): those pages are out of the fast path until a pass clears them."
                 % report["open_contradictions"])
    if report["index_lines"] > report["index_line_cap"]:
        p.append("  INDEX.md exceeds the line cap; the injected top layer is being truncated.")
    if not report["pages"]:
        p.append("  empty store: nothing to measure.")
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("memory_dir", help="the memory directory (holds wiki/, state.json, dream.log)")
    ap.add_argument("--json", action="store_true", help="emit the raw report as JSON")
    args = ap.parse_args()
    memory_dir = os.path.abspath(os.path.expanduser(args.memory_dir))
    if not os.path.isdir(memory_dir):
        print("not a directory: %s" % memory_dir, file=sys.stderr)
        return 2
    report = build_report(memory_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
