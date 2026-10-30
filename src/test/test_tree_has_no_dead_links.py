"""The tree has no references into a directory that is not there.

``legacy/`` held the A2A server and the Telegram/Viber gateways: 16 files,
1,482 lines of production code and four test files.  ``pyproject.toml``
points ``testpaths`` at ``src/test``, so those four were never collected,
and nothing in ``src/`` imported the rest -- they were maintained by hand
and verified by nobody.  They were deleted, and this file keeps the
deletion honest: a doc that still tells a reader to run
``python -m legacy.a2a_client`` is worse than no doc.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


class TestLegacyIsGone:
    def test_the_directory_does_not_exist(self):
        assert not os.path.exists(os.path.join(ROOT, "legacy"))

    def test_nothing_in_src_imports_it(self):
        offenders = []
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "src")):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if re.search(r"^\s*(from|import)\s+legacy\b", line):
                            offenders.append(f"{path}:{i}: {line.strip()}")
        assert not offenders, offenders


class TestDocsPointAtRealFiles:
    """A markdown link to a path in this repo must resolve."""

    DOCS = ["README.md", "RELEASE.md", "docs"]

    def _markdown_files(self):
        for entry in self.DOCS:
            path = os.path.join(ROOT, entry)
            if os.path.isfile(path):
                yield path
            else:
                for dirpath, _, filenames in os.walk(path):
                    for name in filenames:
                        if name.endswith(".md"):
                            yield os.path.join(dirpath, name)

    def test_no_relative_link_points_into_a_missing_path(self):
        broken = []
        for doc in self._markdown_files():
            with open(doc, errors="ignore") as fh:
                text = fh.read()
            for target in re.findall(r"\]\((\.{0,2}/?[^)\s#]+)\)", text):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                # Only paths: a link with no slash and no extension is prose
                # or code (``__builtins__['open'](p)`` is not a link), and
                # this test is about files that moved, not about syntax.
                if "/" not in target and not os.path.splitext(target)[1]:
                    continue
                resolved = os.path.normpath(
                    os.path.join(os.path.dirname(doc), target))
                if not os.path.exists(resolved):
                    broken.append(f"{os.path.relpath(doc, ROOT)} -> {target}")
        assert not broken, "links to paths that do not exist:\n" + "\n".join(broken)

    def test_no_doc_tells_a_reader_to_run_the_removed_module(self):
        """Text, not just links: an instruction is a reference too."""
        offenders = []
        for doc in self._markdown_files():
            if os.path.basename(doc) in ("test_tree_has_no_dead_links.py",):
                continue
            if "archive" in doc.split(os.sep):
                continue        # history, explicitly labelled as such
            with open(doc, errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if re.search(r"python -m legacy\.", line):
                        offenders.append(
                            f"{os.path.relpath(doc, ROOT)}:{i}: {line.strip()}")
        assert not offenders, offenders
