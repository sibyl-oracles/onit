"""The harness must not depend on any front end.

``onit/contract/`` is the only module both sides may import.  The harness
(``onit/onit.py``, ``onit/model/serving/``) names ``ChatUIProtocol`` and
``UISettings`` from there and knows nothing else about a UI; each front end
(``onit/ui/text.py``, ``onit/ui/api.py``, and a future native app)
implements the protocol without the harness importing it.

That property is what lets the harness change without breaking a UI, and a
UI change without touching the harness.  It is also easy to lose: one
convenience ``from .ui import ChatUI`` at the top of ``onit.py`` and the
agent loop is a compile-time dependency of the terminal again.  This test
fails the build instead of leaving it to a reviewer.

Only *module-level* imports are checked.  A function-body import is the
sanctioned way to reach a front end from the harness \u2014 it runs only on the
path that needs that UI, and it keeps ``import onit.onit`` free of every
front end.  ``onit.py`` uses one to build the terminal UI.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent

#: Harness modules: the agent loop and the serving layer beneath it.
HARNESS_MODULES = [
    PACKAGE / "onit.py",
    *sorted((PACKAGE / "model" / "serving").glob("*.py")),
]

#: The one package both sides may import.
ALLOWED = "onit.contract"


def _module_level_imports(tree: ast.Module):
    """Yield (lineno, module_name) for every import not inside a function.

    Walks only the module body and the bodies of ``if``/``try`` blocks at
    that level, which is where an import still runs at import time.  Class
    and function bodies are skipped: those run later, on a path that has
    already chosen a front end.
    """
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            # Relative imports: resolve against the package root so
            # ``from .ui import ChatUI`` in onit/onit.py reads as "onit.ui".
            if node.level:
                name = "." * node.level + (node.module or "")
            else:
                name = node.module or ""
            yield node.lineno, name
        elif isinstance(node, (ast.If, ast.Try)):
            stack.extend(node.body)
            stack.extend(getattr(node, "orelse", []))
            stack.extend(getattr(node, "finalbody", []))
            for handler in getattr(node, "handlers", []):
                stack.extend(handler.body)


def _resolves_to_ui(name: str) -> bool:
    """Whether an import name reaches the UI package.

    Handles both absolute (``onit.ui.text``) and relative (``.ui.text``,
    ``..ui``) spellings, since ``onit.py`` uses the relative form.
    """
    cleaned = name.lstrip(".")
    return cleaned == "ui" or cleaned.startswith("ui.")


@pytest.mark.parametrize("path", HARNESS_MODULES, ids=lambda p: p.name)
def test_harness_does_not_import_a_ui_at_module_level(path: Path):
    """No harness module may import onit.ui at import time."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = [
        (lineno, name)
        for lineno, name in _module_level_imports(tree)
        if _resolves_to_ui(name)
    ]
    assert not offenders, (
        f"{path.relative_to(PACKAGE)} imports a front end at module level: "
        f"{offenders}.  Move it inside the function that needs it, or put "
        f"the shared name in {ALLOWED}."
    )


def test_contract_package_imports_without_any_ui():
    """onit.contract must be importable with onit.ui unavailable.

    The contract is the seam; if it ever imports a front end, the seam is
    gone and both sides are coupled again.
    """
    tree = ast.parse(
        (PACKAGE / "contract" / "__init__.py").read_text(encoding="utf-8")
    )
    for module in (PACKAGE / "contract").glob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        offenders = [
            (lineno, name)
            for lineno, name in _module_level_imports(tree)
            if _resolves_to_ui(name)
        ]
        assert not offenders, f"{module.name} imports a front end: {offenders}"


def test_ui_settings_is_the_declared_surface():
    """The fields a UI may read are named here, not guessed with getattr.

    Guards the rename case: if the harness drops ``config_data``, this fails
    at the boundary rather than every front end silently reading None.
    """
    from onit.contract import UISettings

    fields = set(UISettings.__dataclass_fields__)
    assert fields == {"config_data", "background_checks", "cancel_background_check"}
