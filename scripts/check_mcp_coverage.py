#!/usr/bin/env python3
"""Fail when an MCP module drops below its own coverage floor.

The suite-wide floor in ``pyproject.toml`` is a ratchet on the *total*, and
that is the wrong shape for a tree this uneven: the well-tested ``learn/`` and
``model/`` packages carry the average, so a module can sit at 17% and cost the
total almost nothing.  That is exactly how ``web/search`` and ``github`` went
untested for months while the build stayed green.

This runs *after* pytest, against the coverage data pytest wrote.  It cannot be
a test: coverage flushes its data file when the run ends, so a test reading
``.coverage`` mid-run sees zeros.

Usage:
    pytest onit/test/            # writes .coverage
    python scripts/check_mcp_coverage.py

Exit status is 1 when any module is below its floor, so CI can gate on it.
"""

from __future__ import annotations

import sys

try:
    from coverage import Coverage
except ImportError:  # pragma: no cover - coverage is a test dependency
    print("coverage is not installed; run: pip install -e '.[test]'", file=sys.stderr)
    raise SystemExit(2)

# module path -> (floor, why it is worth a floor of its own)
#
# Floors sit a few points under the measured value, for the same reason the
# global one does: a floor one good week away from the measurement turns
# unrelated changes into red builds.  Raise them as the numbers rise; never
# lower one to make a build pass.
FLOORS: dict[str, tuple[float, str]] = {
    # Fetches arbitrary model-supplied URLs and holds the path jail.
    "onit/mcp/servers/tasks/web/search/mcp_server.py": (78, "fetches untrusted URLs"),
    # The provider fallback chain every web search runs through.
    "onit/mcp/servers/tasks/web/search/web_search.py": (82, "web search chain"),
    # Talks to the GitHub API with the user's token.
    "onit/mcp/servers/tasks/github/mcp_server.py": (95, "holds the user's token"),
    # The SSRF guard and the shared tool bodies every server imports.
    "onit/mcp/servers/tasks/shared.py": (58, "SSRF guard + shared bodies"),
    # The local document toolkit.
    "onit/mcp/servers/tasks/local/search/mcp_server.py": (77, "local search server"),
    "onit/mcp/servers/tasks/local/search/toolkit.py": (70, "document toolkit"),
}


def percent(cov: Coverage, path: str) -> float | None:
    """Measured coverage for *path*, or None when the run never imported it."""
    try:
        _, statements, _, missing, _ = cov.analysis2(path)
    except Exception:
        return None
    if not statements:
        return 100.0
    return 100.0 * (len(statements) - len(missing)) / len(statements)


def main() -> int:
    cov = Coverage(data_file=".coverage")
    try:
        cov.load()
    except Exception:
        print("no .coverage file; run pytest first", file=sys.stderr)
        return 2

    failures = []
    unmeasured = []
    width = max(len(p) for p in FLOORS)

    for path, (floor, why) in sorted(FLOORS.items()):
        measured = percent(cov, path)
        if measured is None:
            unmeasured.append(path)
            print(f"{path:<{width}}  NOT MEASURED  (floor {floor:>5.1f}%)  {why}")
            continue
        ok = measured >= floor
        mark = "ok  " if ok else "FAIL"
        print(f"{path:<{width}}  {mark} {measured:>5.1f}%  (floor {floor:>5.1f}%)  {why}")
        if not ok:
            failures.append((path, measured, floor))

    print()
    if unmeasured:
        print(
            "These modules have a floor but were not imported by the run, so the "
            "floor never fired:",
            file=sys.stderr,
        )
        for p in unmeasured:
            print(f"  {p}", file=sys.stderr)
        return 1

    if failures:
        print(f"{len(failures)} module(s) below their coverage floor:", file=sys.stderr)
        for path, measured, floor in failures:
            print(f"  {path}: {measured:.1f}% < {floor:.1f}%", file=sys.stderr)
        print(
            "\nAdd tests, or lower the floor in scripts/check_mcp_coverage.py "
            "with a note saying why.",
            file=sys.stderr,
        )
        return 1

    print(f"all {len(FLOORS)} modules meet their floors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
