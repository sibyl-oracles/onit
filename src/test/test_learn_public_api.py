"""What ``src.learn`` re-exports, and what it deliberately does not.

``learn/__init__.py`` was a 120-name facade over 14 modules.  Twenty-three of
those names were not in ``__all__`` and had no reader anywhere in the repo:
they were reachable only as ``src.learn.load`` or by a
``from src.learn import load`` that nobody wrote.  They were removed rather
than kept "just in case", and this file is the record of the decision --
so that re-adding one is a deliberate act, not a side effect of pasting an
import block.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import src.learn as learn


REMOVED = [
    # .playbook -- all reachable as src.learn.playbook.<name>
    "contrast", "curate", "decay", "global_path", "load", "make_bullet",
    "owner_path", "playbook_dir", "read_playbook", "reflect", "render",
    "select", "tool_signature",
    # .skills
    "pinned_path", "pinned_skills",
    # .skills_gate
    "ALL_CHECKS", "CHECK_DEPS", "CHECK_MANIFEST", "CHECK_REGRESSION",
    "CHECK_SOURCE", "CHECK_TESTS", "GateResult",
    # .skills_mining
    "MIN_SESSIONS",
]


class TestRemovedNames:
    def test_none_of_them_is_on_the_package(self):
        still_there = [n for n in REMOVED if hasattr(learn, n)]
        assert not still_there, (
            "re-exported again without a reader: %s" % still_there)

    @pytest.mark.parametrize("name,module", [
        ("load", "playbook"),
        ("reflect", "playbook"),
        ("GateResult", "skills_gate"),
        ("MIN_SESSIONS", "skills_mining"),
    ])
    def test_the_owning_module_still_has_them(self, name, module):
        """Removed from the facade, not from the codebase: callers that need
        one import it from the module that defines it."""
        import importlib
        mod = importlib.import_module(f"src.learn.{module}")
        assert hasattr(mod, name)


class TestTheFacadeIsConsistent:
    def test_every_advertised_name_resolves(self):
        missing = [n for n in learn.__all__ if not hasattr(learn, n)]
        assert not missing, "__all__ names that do not exist: %s" % missing

    def test_nothing_is_exported_that_is_not_advertised(self):
        """A public name with no ``__all__`` entry is one a wildcard import
        silently drops -- the same ambiguity that hid the 23 above.

        Submodules are attributes too (``learn.playbook`` after any
        ``import src.learn.playbook``), and they are not part of the
        facade's contract, so they are excluded: only names the package
        imports or defines count.
        """
        import types
        public = {n for n in dir(learn)
                  if not n.startswith("_")
                  and not isinstance(getattr(learn, n), types.ModuleType)}
        extra = sorted(public - set(learn.__all__))
        assert not extra, (
            "public but unadvertised (add to __all__ or drop): %s" % extra)
