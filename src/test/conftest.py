"""Shared fixtures for OnIt test suite."""

import asyncio
import json
import os
import sys

import pytest

# Pin the PyPI ``mcp`` SDK in sys.modules before any test module's sys.path
# insert can shadow it with the local ``src/mcp`` package. fastmcp imports
# ``mcp.types`` lazily; when the shadow wins that resolution, fastmcp raises
# "server support is not installed" at collection. The eager
# ``from .onit import OnIt`` in ``src/__init__.py`` used to pin the SDK as a
# side effect of the first ``import src``; that import is lazy now (PEP 562),
# so the pin has to be explicit. Harmless when the SDK is genuinely absent.
try:
    import mcp.types  # noqa: F401
except ImportError:  # pragma: no cover - SDK extras genuinely absent
    pass

# Ensure src/ is importable for the bare-import fallbacks (``lib.*``/``type.*``/
# ``mcp.*`` without a package prefix, used when server modules run outside the
# package). Appended, not inserted: putting this directory at position 0 lets
# the local ``src/mcp`` package shadow the PyPI ``mcp`` SDK. The eager
# ``from .onit import OnIt`` in ``src/__init__.py`` used to pin the SDK in
# ``sys.modules`` before this insert ran, hiding the hazard; since that import
# became lazy (PEP 562), the first fastmcp import now happens after it and
# resolves ``mcp.types`` to the shadow — fastmcp then raises "server support
# not installed" at collection. Appending keeps the fallbacks working while
# site-packages keeps priority for the SDK.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Deployment-mode env flags
# ---------------------------------------------------------------------------

# Process-global flags that change what the bash policy allows and what the
# assistant prompt announces. A value inherited from the developer's shell (or
# left behind by another test) silently rewrites refusal messages and flips
# install tests, so clear them for every test and let each opt back in.
_MODE_ENV_VARS = (
    "ONIT_WEB_UI",
    "ONIT_CONTAINER",
    "ONIT_ALLOW_PACKAGE_INSTALL",
    "ONIT_COMMAND_ALLOWLIST",
    # Same hazard, newer knobs: a developer running with recording off would
    # otherwise see the trajectory tests silently assert nothing, and one
    # running with a custom path would write test data into it.
    "ONIT_LEARN",
    "ONIT_LEARN_PATH",
    "ONIT_LEARN_REDACT",
)


@pytest.fixture(autouse=True)
def _clean_mode_env(monkeypatch, tmp_path):
    for var in _MODE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Backstop, not a convenience. Recording is on by default, so any test that
    # runs a task through a config without a `learn.path` writes a trajectory —
    # and without this it writes into the developer's real ~/.onit/learned.
    # A config that names its own path still wins; this only catches the ones
    # that say nothing.
    monkeypatch.setenv("ONIT_LEARN_PATH", str(tmp_path / "learned-default"))


# ---------------------------------------------------------------------------
# The developer's keychain
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_real_secrets(monkeypatch, tmp_path):
    """No test reads a secret the machine running it happens to hold.

    ``get_secret`` asks the OS keychain, then ``~/.onit/secrets.yaml``, so
    either one makes the machine part of the test's input.  CI caught a test
    that had only ever passed here: ``test_learn_llm`` builds an Ollama client
    at ``https://ollama.com``, a provider that refuses to answer without a
    key.  It raised on a runner with no keychain and resolved happily on a
    laptop with an ``ollama_api_key`` stored -- the same commit, two results.

    Both sources are closed, not just the keychain, or the file is the next
    one to differ.  A test that wants a secret supplies one; ``test_setup``'s
    ``_fake_keystore`` replaces ``get_secret`` itself and is unaffected.
    """
    import src.setup as setup_mod
    monkeypatch.setattr(setup_mod, "KEYRING_AVAILABLE", False)
    monkeypatch.setattr(setup_mod, "_SECRETS_PATH",
                        str(tmp_path / "secrets.yaml"))


# ---------------------------------------------------------------------------
# The developer's session store
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_real_session_store(monkeypatch, tmp_path_factory):
    """No test writes a session into the store the person running it uses.

    ``OnIt._setup_session`` falls back to ``~/.onit/sessions`` when the config
    does not name one, and it registers the session it creates.  A test that
    builds a real ``OnIt`` from a bare ``{"serving": {...}}`` therefore adds a
    row to the developer's own ``onit sessions`` listing -- unnamed, zero
    turns, and, being the newest entry, the one a bare ``onit`` would resume
    next.  ``test_max_tokens_default`` did exactly that three times per run
    and is why one laptop had 360 such rows.

    Two things are moved, because moving one is not enough.

    ``HOME`` covers ``Path.home()``, ``expanduser("~")`` and the default in
    ``_setup_session``, which is read at call time.  It does *not* cover
    ``DEFAULT_SESSIONS_DIR``: that is a module constant, and it is already
    bound as the default argument of every function in ``sessions`` by the
    time the first test runs, since an earlier test imported the module.  So
    the constant and those defaults are rebound too -- otherwise a call that
    names no directory, which is the leak this fixture exists for, still
    writes to the real store.  A test that names its own ``session_path``
    still wins.

    The directory is made through ``tmp_path_factory`` and not under
    ``tmp_path``: several doctor tests assert their ``tmp_path`` is empty
    afterwards, and a home directory parked inside it is the one thing they
    would find.
    """
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    # Windows spells it USERPROFILE; the fallback is not exercised on the CI
    # runners, but a test that reads it should not reach the real home either.
    monkeypatch.setenv("USERPROFILE", str(home))

    import sys

    import sessions as sessions_mod

    old = sessions_mod.DEFAULT_SESSIONS_DIR
    new = str(home / ".onit" / "sessions")
    monkeypatch.setattr(sessions_mod, "DEFAULT_SESSIONS_DIR", new,
                        raising=False)
    for fn in list(vars(sessions_mod).values()):
        defaults = getattr(fn, "__defaults__", None)
        if defaults and old in defaults:
            rebound = tuple(new if d == old else d for d in defaults)
            monkeypatch.setattr(fn, "__defaults__", rebound, raising=False)
    # ``OnIt._setup_session`` and the CLI resolve their own default string.
    for mod_name in ("src.onit", "src.cli"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "DEFAULT_SESSIONS_DIR"):
            monkeypatch.setattr(mod, "DEFAULT_SESSIONS_DIR", new,
                                raising=False)


# ---------------------------------------------------------------------------
# Minimal valid config dict (avoids real MCP / network calls)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config_dict(tmp_path):
    """Minimal config dict that satisfies OnIt.__init__ validation."""
    return {
        "serving": {
            "host": "http://localhost:8000/v1",
            "think": False,
            "max_tokens": 1024,
        },
        "mcp": {
            "servers": [
                {
                    "name": "PromptsMCPServer",
                    "url": "stdio://PromptsMCPServer",
                    "enabled": True,
                },
                {
                    "name": "ToolsMCPServer",
                    "url": "http://127.0.0.1:18201/sse",
                    "enabled": True,
                },
            ],
        },
        "session_path": str(tmp_path / "sessions"),
        "theme": "white",
        "verbose": False,
    }


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def make_tool_item():
    """Factory to build a tool_item dict."""
    def _make(name="test_tool", description="A test tool", url="http://127.0.0.1:18201/sse"):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                "returns": {},
            },
        }
    return _make


@pytest.fixture
def mock_tool_registry(make_tool_item):
    """Pre-populated ToolRegistry with two fake tools."""
    from type.tools import ToolRegistry, ToolHandler

    registry = ToolRegistry()
    for name, url in [("search", "http://127.0.0.1:18201/sse"),
                      ("bash", "http://127.0.0.1:18201/sse")]:
        handler = ToolHandler(url=url, tool_item=make_tool_item(name=name, url=url))
        registry.register(handler)
    return registry


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def safety_queue():
    return asyncio.Queue(maxsize=10)


# ---------------------------------------------------------------------------
# Temp session file
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_session_file(tmp_path):
    """Create a temp JSONL session file with sample entries."""
    path = tmp_path / "session.jsonl"
    entries = [
        {"task": "hello", "response": "Hi there!", "timestamp": 1.0},
        {"task": "weather", "response": "Sunny today.", "timestamp": 2.0},
    ]
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return str(path)
