"""The endpoint and key surface a graphical front end needs.

The terminal has ``\\host``, ``\\model`` and ``\\key``; the web SPA has none of
them, and a native app had no way to name a model, add a vLLM server or store
an API key at all.  This module is the missing half of that: the same
operations, expressed as data a front end can render, with the rules about
where a key lives kept in one place.

Two properties are load-bearing.

**A key is never returned.**  Every read reports *where* a key comes from and
at most its last four characters \u2014 the same ``\u2022\u2022\u2022\u2022abcd`` the terminal
prints.  A response that carried the key itself would put it in a browser
history, a screenshot and a log, which is the one place ``onit setup``
deliberately keeps it out of.

**A key is never written to the config file.**  It goes to the OS keychain
(``onit.setup.store_endpoint_key``), falling back to the 0600 secrets file
when no keychain is reachable.  The config file keeps the endpoint list and
nothing secret, so it stays safe to paste into an issue.

The functions here do no I/O beyond what ``onit.setup`` already does, and they
take the config mapping rather than reaching for a global: the routes in
``api.py`` hand over the harness's own ``config_data`` so a save is visible to
the running agent without a restart.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from onit import setup as onit_setup

#: The integrations a person configures by hand, in the order a front end
#: should show them.  Each is ``(keyring key, label, env var, blurb)``.
#:
#: ``ollama_api_key`` is here rather than in the endpoint list because it is
#: two things at once \u2014 the credential for Ollama cloud *and* what switches
#: web search on \u2014 and a user looking for "why can't it search the web" will
#: look under integrations, not under model endpoints.
INTEGRATIONS = (
    (
        "tavily_api_key",
        "Tavily",
        "TAVILY_API_KEY",
        "Web search, first tier. Without it search falls back to Ollama, then DuckDuckGo.",
    ),
    (
        "ollama_api_key",
        "Ollama",
        "OLLAMA_API_KEY",
        "Ollama cloud models, and the web-search tool. Without it, search is off.",
    ),
    (
        "openweathermap_api_key",
        "OpenWeatherMap",
        "OPENWEATHERMAP_API_KEY",
        "The weather tool.",
    ),
    (
        "github_token",
        "GitHub",
        "GITHUB_TOKEN",
        "Git operations and the GitHub MCP server.",
    ),
    (
        "huggingface_token",
        "HuggingFace",
        "HF_TOKEN",
        "Model hub downloads.",
    ),
)

#: Presets offered when adding an endpoint: ``(label, url, needs_key)``.
#:
#: The same list the wizard shows, plus the two local servers a person is
#: most likely to have running.  ``needs_key`` is a hint for the form, not a
#: rule \u2014 a local vLLM can require one and an Ollama cloud host always does.
PRESETS = (
    ("Ollama (local)", "http://localhost:11434/v1", False),
    ("vLLM", "http://localhost:8000/v1", False),
    ("SGLang", "http://localhost:30000/v1", False),
    ("Ollama Cloud", "https://ollama.com", True),
    ("OpenRouter", "https://openrouter.ai/api/v1", True),
    ("OpenAI", "https://api.openai.com/v1", True),
    ("Anthropic", "https://api.anthropic.com/v1", True),
    ("Vercel AI Gateway", "https://ai-gateway.vercel.sh/v1", True),
)


# \u2500\u2500 Reading \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def _mask(value: Optional[str]) -> Optional[str]:
    """The last four characters of a secret, never the secret.

    Four is enough to tell two keys apart in a list and not enough to use
    one.  A value shorter than eight characters is masked whole: a five
    character key shown as ``\u2022\u2022\u2022\u2022abcde`` is not masked at all.
    """
    if not value:
        return None
    if len(value) < 8:
        return "\u2022\u2022\u2022\u2022"
    return "\u2022\u2022\u2022\u2022" + value[-4:]


def _key_status(host: str, literal: bool = False) -> dict:
    """Where the key for *host* comes from, and its masked tail.

    ``literal`` says the config file itself carries a key for this endpoint.
    That case is reported first because it is the one that wins at request
    time and the one a user cannot see: a key stored in the keychain is
    silently ignored while a literal one is present.
    """
    if literal:
        return {"source": "config", "masked": None, "label": "in config file"}
    if not host:
        return {"source": None, "masked": None, "label": "none"}
    value = onit_setup.get_endpoint_key(host)
    if value:
        return {"source": "keychain", "masked": _mask(value), "label": "stored"}
    keyring_key, env_var, label, _required = onit_setup.legacy_key_for(host)
    for candidate in (keyring_key,):
        legacy = onit_setup.get_secret(candidate)
        if legacy:
            return {
                "source": "legacy",
                "masked": _mask(legacy),
                "label": f"from {candidate}",
            }
    if env_var and os.environ.get(env_var):
        return {"source": "env", "masked": None, "label": f"from ${env_var}"}
    return {"source": None, "masked": None, "label": "none"}


def _entry_rows(config: dict) -> list:
    """The configured endpoints, in preference order, as rows.

    Reads both config shapes through ``onit.setup`` so a config written by
    the wizard (``serving.host``) and one written by ``\\save``
    (``serving.endpoints``) present identically.
    """
    entries = onit_setup._endpoint_list(config) or onit_setup._entries_from_host(config)
    order = sorted(range(len(entries)), key=lambda i: onit_setup._entry_priority(entries[i]))
    rows = []
    for position, index in enumerate(order):
        entry = entries[index]
        host = onit_setup.normalize_host(str(entry.get("host", "")))
        literal = bool(entry.get("api_key") or entry.get("host_key"))
        status = _key_status(host, literal=literal)
        rows.append(
            {
                "id": host,
                "host": host,
                "model": entry.get("model") or "",
                "name": entry.get("name") or "",
                "priority": onit_setup._entry_priority(entry),
                "position": position,
                "key": status,
                "notes": _notes_for(host, entry.get("model") or "", literal),
            }
        )
    return rows


def _notes_for(host: str, model: str, literal: bool) -> list:
    """What is wrong with this endpoint, in the words the wizard uses.

    The same two questions ``onit.setup._provider_notes`` asks \u2014 is a
    required key missing, and is a model named where one is required \u2014 but
    scoped to one row so a front end can put the note next to the field that
    fixes it instead of in a paragraph at the bottom.
    """
    notes = []
    _keyring_key, env_var, label, required = onit_setup.legacy_key_for(host)
    status = _key_status(host, literal=literal)
    if required and status["source"] is None:
        notes.append(f"{label} needs an API key.")
        if env_var:
            notes.append(f"Or set {env_var} in the environment.")
    if not model and label in ("OpenRouter", "OpenAI"):
        notes.append(f"{label} needs an explicit model name.")
    return notes


def describe(config: dict) -> dict:
    """Everything a settings screen needs, in one response.

    ``endpoints`` is the configured list; ``presets`` is what the add form
    offers; ``integrations`` is the key/value pairs that are not tied to a
    model server.  No key value appears anywhere in the result.
    """
    return {
        "endpoints": _entry_rows(config),
        "presets": [
            {"label": label, "host": host, "needs_key": needs_key}
            for label, host, needs_key in PRESETS
        ],
        "integrations": [
            {
                "key": key,
                "label": label,
                "env_var": env_var,
                "blurb": blurb,
                **_integration_status(key, env_var),
            }
            for key, label, env_var, blurb in INTEGRATIONS
        ],
        "config_path": onit_setup.CONFIG_PATH,
        "keychain": bool(onit_setup.KEYRING_AVAILABLE),
    }


def _integration_status(key: str, env_var: Optional[str]) -> dict:
    """Where an integration's secret comes from, masked."""
    if env_var and os.environ.get(env_var):
        return {"source": "env", "masked": None, "label": f"from ${env_var}"}
    value = onit_setup.get_secret(key)
    if value:
        return {"source": "keychain", "masked": _mask(value), "label": "stored"}
    return {"source": None, "masked": None, "label": "not set"}


# \u2500\u2500 Writing \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def save_endpoints(config: dict, rows: list) -> dict:
    """Replace the endpoint list, keeping each endpoint's key attached.

    *rows* is what the front end sends back: ``{host, model, name,
    priority}`` per endpoint, in the order it wants them tried.  A row's
    ``priority`` is its position when the caller does not send one, so a
    list dragged into order is saved in that order.

    Keys are **not** part of this call.  They are addressed by URL and stored
    separately (``store_endpoint_key``), so reordering or renaming an
    endpoint cannot lose one \u2014 and a save cannot write a secret into the
    config file by accident.

    An endpoint that disappears from the list keeps its stored key: a person
    who removes a host and adds it back a minute later should not have to
    find the key again.  ``forget_endpoint_key`` is how a key is removed.
    """
    cleaned = []
    seen = set()
    for position, row in enumerate(rows):
        host = onit_setup.normalize_host(str(row.get("host") or ""))
        if not host or host in seen:
            continue
        seen.add(host)
        entry: dict[str, Any] = {"host": host}
        model = str(row.get("model") or "").strip()
        if model:
            entry["model"] = model
        name = str(row.get("name") or "").strip()
        if name:
            entry["name"] = name
        priority = row.get("priority")
        entry["priority"] = int(priority) if priority is not None else position
        cleaned.append(entry)

    onit_setup._write_endpoints(config, cleaned)
    onit_setup._save_config(config)
    return describe(config)


def store_key(host: str, value: str) -> dict:
    """Store one endpoint's API key, addressed by its URL.

    A key typed for a host that is not in the config yet is still stored:
    the add form asks for the key before the endpoint is saved, and losing it
    at that point would be the same bug as not asking.
    """
    host = onit_setup.normalize_host(host)
    if not host:
        raise ValueError("An endpoint URL is required.")
    if not value or not value.strip():
        raise ValueError("The key is empty.")
    onit_setup.store_endpoint_key(host, value.strip())
    return _key_status(host)


def forget_key(host: str) -> dict:
    """Remove an endpoint's stored key, leaving any legacy fallback in place."""
    host = onit_setup.normalize_host(host)
    if not host:
        raise ValueError("An endpoint URL is required.")
    onit_setup.delete_endpoint_key(host)
    return _key_status(host)


def store_integration(key: str, value: str) -> dict:
    """Store one integration secret (Tavily, Ollama, GitHub, \u2026)."""
    known = {k for k, _, _, _ in INTEGRATIONS}
    if key not in known:
        raise ValueError(f"Unknown integration: {key}")
    if not value or not value.strip():
        raise ValueError("The key is empty.")
    onit_setup.store_secret(key, value.strip())
    env_var = next(env for k, _, env, _ in INTEGRATIONS if k == key)
    return _integration_status(key, env_var)


def forget_integration(key: str) -> dict:
    """Remove one integration secret."""
    known = {k for k, _, _, _ in INTEGRATIONS}
    if key not in known:
        raise ValueError(f"Unknown integration: {key}")
    onit_setup.delete_secret(key)
    env_var = next(env for k, _, env, _ in INTEGRATIONS if k == key)
    return _integration_status(key, env_var)
