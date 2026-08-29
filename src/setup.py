"""
Interactive setup wizard for OnIt.

Guides the user through configuring API keys, hosts, and preferences.
Secrets are stored in the OS keychain via the `keyring` library;
non-secret settings are written to ~/.onit/config.yaml.

Usage:
    onit setup          # interactive wizard
    onit setup --show   # display current configuration
"""

import getpass
import os
import re
import sys

import yaml

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

SERVICE_NAME = "onit"
CONFIG_DIR = os.path.expanduser("~/.onit")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")
_SECRETS_PATH = os.path.join(CONFIG_DIR, "secrets.yaml")

# ── Configurable secrets ────────────────────────────────────────────
# (keyring_key, prompt_label, env_var_name)
# Split by what the key is for so the wizard can ask for model-serving
# credentials alongside the endpoints they authenticate, and keep unrelated
# integrations in their own section.
# An endpoint's API key is now asked for and stored per endpoint, addressed by
# its URL (see endpoint_secret_name below), because that is what a key actually
# belongs to: two vLLM servers with different keys could not be expressed by a
# single ``vllm_api_key``, and nothing but the URL decided which of these four
# a request used.
#
# These stay readable so an install predating that keeps working — a host with
# no key of its own falls back to whichever of them its URL selects — but the
# wizard no longer offers them, and nothing writes them any more.
LEGACY_SERVING_SECRETS = [
    ("host_key",               "OpenRouter API key (legacy fallback)",
     "OPENROUTER_API_KEY"),
    ("vllm_api_key",           "vLLM API key (legacy fallback)",
     "VLLM_API_KEY"),
    ("host2_key",              "Second model server key (legacy fallback)",
     "ONIT_HOST2_KEY"),
]

INTEGRATION_SECRETS = [
    # Still prompted, and not only a serving credential: without it the web
    # search tool is switched off (see cli.py), which is why it did not move
    # into the per-endpoint keys with the rest.
    ("ollama_api_key",         "Ollama API key (web search + Ollama hosts)",
     "OLLAMA_API_KEY"),
    ("openweathermap_api_key", "OpenWeatherMap API key (enables weather tool)",
     "OPENWEATHERMAP_API_KEY"),
    ("telegram_bot_token",     "Telegram bot token (for gateway mode)",
     "TELEGRAM_BOT_TOKEN"),
    ("viber_bot_token",        "Viber bot token (for gateway mode)",
     "VIBER_BOT_TOKEN"),
    ("web_google_client_id",   "Google OAuth2 client ID (for web UI auth)",
     None),
    ("web_google_client_secret", "Google OAuth2 client secret (for web UI auth)",
     None),
    ("github_token",           "GitHub personal access token (for git operations)",
     "GITHUB_TOKEN"),
    ("huggingface_token",      "HuggingFace access token (for model hub access)",
     "HF_TOKEN"),
]

SECRETS = LEGACY_SERVING_SECRETS + INTEGRATION_SECRETS

# Lookup table: keyring_key → env_var_name (for keys that have one).
# Used by get_secret() to check env vars before keyring/file — critical in
# container mode where the OS keychain is unreachable but the container
# launcher injects secrets as env vars (e.g. GITHUB_TOKEN).
_SECRET_ENV_MAP: dict[str, str] = {
    key: env for key, _, env in SECRETS if env
}

# ── Non-secret settings ────────────────────────────────────────────
# (dot_path, prompt_label, default_value)
# An empty default marks an optional setting; entering "-" at the prompt
# clears a previously saved value.

# Host/model settings are no longer prompted one by one — the endpoint editor
# collects them as a ranked list and writes back whichever shape fits. They
# stay here so show_config and any config reader still knows their labels.
HOST_SETTINGS = [
    ("serving.host",  "LLM endpoint URL (vLLM / OpenRouter / Ollama)",
     "http://localhost:8000/v1"),
    ("serving.model", "Model name (blank = auto-detect from endpoint)", ""),
    ("serving.host2", "Second LLM endpoint URL (optional, enables load balancing)", ""),
    ("serving.model2", "Model name on second server (blank = auto-detect)", ""),
]

SERVING_SETTINGS = [
    ("serving.load_balancer",
     "Load balancing algorithm (sticky / round_robin / random / least_busy)",
     "sticky"),
]

GENERAL_SETTINGS = [
    ("theme",        "UI theme (dark / white)",  "dark"),
    ("web_port",     "Web UI port",              "9000"),
    ("timeout",      "Request timeout in seconds (-1 = none)", "-1"),
]

SETTINGS = HOST_SETTINGS + SERVING_SETTINGS + GENERAL_SETTINGS

_HOST_SETTINGS = {dotpath for dotpath, _, _ in HOST_SETTINGS}

# Default host offered when configuring the very first endpoint.
DEFAULT_HOST = "http://localhost:8000/v1"

# Example endpoints shown at the top of the wizard, one per provider.
ENDPOINT_EXAMPLES = (
    "vLLM: http://localhost:8000/v1  |  "
    "Ollama cloud: https://api.ollama.com  |  "
    "Ollama local: http://localhost:11434/v1  |  "
    "OpenRouter: https://openrouter.ai/api/v1"
)


# ── Helpers ─────────────────────────────────────────────────────────

def _get_nested(data: dict, dotpath: str):
    """Get a value from a nested dict using 'a.b.c' notation."""
    keys = dotpath.split(".")
    for k in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(k)
    return data


def _set_nested(data: dict, dotpath: str, value):
    """Set a value in a nested dict using 'a.b.c' notation."""
    keys = dotpath.split(".")
    for k in keys[:-1]:
        data = data.setdefault(k, {})
    data[keys[-1]] = value


def _load_config() -> dict:
    """Load existing config from ~/.onit/config.yaml or return empty dict."""
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_config(data: dict):
    """Write config dict to ~/.onit/config.yaml."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _file_store_secret(key: str, value: str):
    """Fallback: persist secret in ~/.onit/secrets.yaml (owner-only perms)."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    data = {}
    if os.path.isfile(_SECRETS_PATH):
        with open(_SECRETS_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
    data[key] = value
    with open(_SECRETS_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    os.chmod(_SECRETS_PATH, 0o600)


def _file_get_secret(key: str) -> str | None:
    """Fallback: read secret from ~/.onit/secrets.yaml."""
    if not os.path.isfile(_SECRETS_PATH):
        return None
    # A bind-mounted secrets.yaml from the host may be unreadable inside a
    # container due to UID mismatch — treat as "not present" rather than crash.
    try:
        with open(_SECRETS_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
    except OSError:
        return None
    return data.get(key)


def store_secret(key: str, value: str):
    """Store a secret in the OS keychain, falling back to file storage."""
    if KEYRING_AVAILABLE:
        try:
            keyring.set_password(SERVICE_NAME, key, value)
            return
        except Exception:
            pass
    _file_store_secret(key, value)


def delete_secret(key: str):
    """Remove a secret from the OS keychain and the file fallback."""
    if KEYRING_AVAILABLE:
        try:
            keyring.delete_password(SERVICE_NAME, key)
        except Exception:
            pass
    if os.path.isfile(_SECRETS_PATH):
        try:
            with open(_SECRETS_PATH, "r") as f:
                data = yaml.safe_load(f) or {}
            if key in data:
                del data[key]
                with open(_SECRETS_PATH, "w") as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                os.chmod(_SECRETS_PATH, 0o600)
        except OSError:
            pass


def get_secret(key: str) -> str | None:
    """Retrieve a secret: env var > OS keychain > file fallback.

    Checking the env var first makes this work inside containers where the
    host OS keychain is unreachable but the launcher injected secrets as env
    vars (e.g. GITHUB_TOKEN bridged in by container_launcher).
    """
    env_var = _SECRET_ENV_MAP.get(key)
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val
    if KEYRING_AVAILABLE:
        try:
            val = keyring.get_password(SERVICE_NAME, key)
            if val is not None:
                return val
        except Exception:
            pass
    return _file_get_secret(key)


# ── Per-endpoint API keys ───────────────────────────────────────────
# One key per endpoint, addressed by its URL. The URL is the identity because
# it is the only field an endpoint cannot be without: ``name`` is optional and
# editable, and an index moves the moment a server is added above it.

_ENDPOINT_SECRET_PREFIX = "endpoint_key:"

# Which provider-named secret still authenticates a host that has no key of
# its own. First matching fragment wins; the last entry is the default, and
# matches nothing explicitly. ``required`` marks a provider that cannot answer
# at all without a key, where a missing one is worth an error rather than an
# attempt.
LEGACY_ENDPOINT_KEYS = (
    (("openrouter.ai",), "host_key", "OPENROUTER_API_KEY", "OpenRouter", True),
    (("ollama.com", "ollama.ai"), "ollama_api_key", "OLLAMA_API_KEY",
     "Ollama cloud", True),
    ((), "vllm_api_key", "VLLM_API_KEY", "vLLM", False),
)


def normalize_host(host: str) -> str:
    """The form of a URL that addresses its key.

    A trailing slash is the one difference between two spellings of the same
    endpoint that a user will not notice and that would otherwise hide the key
    they just stored.
    """
    return (host or "").strip().rstrip("/")


def endpoint_secret_name(host: str) -> str:
    """Keychain entry name holding the API key for ``host``."""
    return f"{_ENDPOINT_SECRET_PREFIX}{normalize_host(host)}"


def endpoint_env_var(host: str) -> str:
    """Env var an endpoint's key may be injected under.

    Needed where the OS keychain is unreachable — inside a container, or a
    deployment that ships secrets as environment. Two URLs differing only in
    punctuation slug to the same name; the keychain entry keeps the URL whole,
    so only this form can collide, and only between endpoints that are nearly
    the same address already.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", normalize_host(host)).strip("_").upper()
    return f"ONIT_ENDPOINT_KEY_{slug}"


def get_endpoint_key(host: str) -> str | None:
    """The API key stored for one endpoint: env var > keychain > file."""
    if not host:
        return None
    value = os.environ.get(endpoint_env_var(host))
    if value:
        return value
    return get_secret(endpoint_secret_name(host))


def store_endpoint_key(host: str, value: str) -> None:
    store_secret(endpoint_secret_name(host), value)


def delete_endpoint_key(host: str) -> None:
    delete_secret(endpoint_secret_name(host))


def move_endpoint_key(old_host: str, new_host: str) -> None:
    """Follow a key to an endpoint's new URL.

    Editing a host would otherwise strand its key under the old address and
    silently leave the endpoint unauthenticated.
    """
    if not old_host or normalize_host(old_host) == normalize_host(new_host):
        return
    value = get_secret(endpoint_secret_name(old_host))
    if value:
        store_endpoint_key(new_host, value)
    delete_endpoint_key(old_host)


def legacy_key_for(host: str) -> tuple[str, str, str, bool]:
    """(keyring key, env var, provider label, key is mandatory) for ``host``.

    The rule that used to be spelled out in chat.py's key resolution and again
    in the wizard's sanity notes; both read it from here now.
    """
    host = host or ""
    for fragments, keyring_key, env_var, label, required in LEGACY_ENDPOINT_KEYS:
        if any(f in host for f in fragments):
            return keyring_key, env_var, label, required
    return LEGACY_ENDPOINT_KEYS[-1][1:]


def endpoint_key_label(host: str, in_config: bool = False,
                       extra_keys: tuple = ()) -> str:
    """One cell's account of what authenticates ``host``.

    Names the legacy secret rather than saying "fallback", because that case
    is the one a user cannot guess: the endpoint has no key of its own and is
    running on something they set up for a different server.
    """
    if in_config:
        return "in config"
    source = endpoint_key_source(host, extra_keys)
    if source is None:
        return "none"
    return "set" if source == "endpoint" else source


def endpoint_key_source(host: str, extra_keys: tuple = ()) -> str | None:
    """Where ``host`` gets its key from: 'endpoint', a legacy secret's name,
    or None when nothing authenticates it.

    ``extra_keys`` names secrets that apply to this endpoint by position
    rather than by URL — ``host2_key`` for the legacy second server.
    """
    if get_endpoint_key(host):
        return "endpoint"
    keyring_key, env_var, _, _ = legacy_key_for(host)
    for key in (keyring_key,) + tuple(extra_keys):
        if get_secret(key):
            return key
    return env_var if os.environ.get(env_var) else None


def resolve_credential(cli_value: str | None,
                       env_var: str | None,
                       keyring_key: str) -> str | None:
    """Resolve a credential: CLI arg > env var > keyring > None."""
    if cli_value:
        return cli_value
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val
    return get_secret(keyring_key)


# ── Provider-specific sanity checks ─────────────────────────────────

def _endpoint_list(config: dict) -> list[dict]:
    """Return ``serving.endpoints`` normalized to dicts, or [] if unset.

    Bare URL strings are a documented shorthand for ``{host: <url>}``.
    """
    raw = _get_nested(config, "serving.endpoints")
    if not isinstance(raw, list):
        return []
    out = []
    for entry in raw:
        if isinstance(entry, str):
            entry = {"host": entry}
        if isinstance(entry, dict) and entry.get("host"):
            out.append(entry)
    return out


def _entry_priority(entry: dict) -> int:
    """An endpoint entry's priority, defaulting to 0 when unset or invalid."""
    try:
        return int(entry.get("priority", 0))
    except (TypeError, ValueError):
        return 0


def _entries_from_host_pair(config: dict) -> list[dict]:
    """Read the legacy serving.host / serving.host2 pair as endpoint entries.

    Lets the editor present one list regardless of which shape the config is
    currently written in.
    """
    entries = []
    for host_path, model_path, key_path in (
        ("serving.host",  "serving.model",  "serving.host_key"),
        ("serving.host2", "serving.model2", "serving.host2_key"),
    ):
        host = str(_get_nested(config, host_path) or "").strip()
        if not host:
            continue
        entry = {"host": host}
        model = _get_nested(config, model_path)
        if model:
            entry["model"] = model
        key = _get_nested(config, key_path)
        if key and key != "EMPTY":
            entry["host_key"] = key
        entries.append(entry)
    return entries


def _fits_host_pair(entries: list[dict]) -> bool:
    """True when entries carry nothing the legacy host/host2 pair can't hold.

    Keeps a plain one- or two-server config in its short, familiar form
    instead of rewriting it as a list the moment the wizard is run.
    """
    return (len(entries) <= 2
            and all(_entry_priority(e) == 0 and not e.get("name")
                    for e in entries))


def _write_endpoints(config: dict, entries: list[dict]) -> None:
    """Persist entries in the simplest config shape that can express them.

    Whichever shape is written, the other is cleared, so exactly one of them
    is ever live and the running agent can't read a stale host.
    """
    serving = config.setdefault("serving", {})
    if _fits_host_pair(entries):
        serving.pop("endpoints", None)
        for i, (host_key, model_key, key_key) in enumerate(
                (("host", "model", "host_key"),
                 ("host2", "model2", "host2_key"))):
            if i < len(entries):
                serving[host_key] = entries[i]["host"]
                serving[model_key] = entries[i].get("model") or ""
                literal = entries[i].get("api_key") or entries[i].get("host_key")
                if literal:
                    serving[key_key] = literal
            else:
                for k in (host_key, model_key, key_key):
                    serving.pop(k, None)
        return
    # A key written by hand into the YAML is carried across under the current
    # spelling; the wizard's own keys live in the keychain and appear here as
    # nothing at all.
    serving["endpoints"] = [
        {k: v for k, v in (("name", e.get("name")),
                           ("host", e["host"]),
                           ("model", e.get("model")),
                           ("api_key", e.get("api_key") or e.get("host_key")),
                           ("priority", _entry_priority(e)))
         if v not in (None, "")}
        for e in entries
    ]
    for k in ("host", "model", "host2", "model2", "host2_key"):
        serving.pop(k, None)


def _configured_endpoints(config: dict) -> list[tuple]:
    """Enumerate configured endpoints as (host_label, model_label, host,
    model, key_names) tuples.

    Covers both config shapes: a ``serving.endpoints`` list, or the legacy
    ``serving.host`` / ``serving.host2`` pair. The labels name the setting in
    the form the user wrote it, so notes point at something they can edit.
    """
    entries = _endpoint_list(config)
    if entries:
        # Label by the entry's name (or host) rather than a list index — it
        # survives reordering and is what the user recognizes.
        return [(f"endpoint '{e.get('name') or e['host']}'",
                 "its 'model' key", str(e["host"]), e.get("model"),
                 ("api_key" in e or "host_key" in e), ())
                for e in entries]
    found = []
    for host_path, model_path, key_path, extra_keys in (
        ("serving.host",  "serving.model",  "serving.host_key", ()),
        # The second legacy server takes its own key by position rather than
        # by URL, so name it alongside whichever one the URL selects.
        ("serving.host2", "serving.model2", "serving.host2_key", ("host2_key",)),
    ):
        host = str(_get_nested(config, host_path) or "")
        if host:
            found.append((host_path, model_path, host,
                          _get_nested(config, model_path),
                          bool(_get_nested(config, key_path)), extra_keys))
    return found


def _provider_notes(config: dict) -> list[str]:
    """Check host/model/key combinations and return human-readable notes.

    Ollama cloud and OpenRouter endpoints need an API key. OpenRouter also
    needs an explicit model name (auto-detection would pick an arbitrary
    entry from its huge model list); for Ollama cloud an explicit model is
    recommended since auto-detection takes the first available model.

    A key counts wherever it comes from — the endpoint's own, one written into
    the YAML, or a legacy provider secret — so the note fires when a request
    would actually fail, not when one particular place is empty.
    """
    notes = []
    for host_path, model_path, host, model, in_config, extra_keys in \
            _configured_endpoints(config):
        _, env_var, label, required = legacy_key_for(host)
        if not required:
            continue
        if not in_config and not endpoint_key_source(host, extra_keys):
            notes.append(f"Note: {host_path} is a{'n' if label[0] in 'AEIOU' else ''} "
                         f"{label} endpoint but no API key is set (rerun "
                         f"'onit setup' or export {env_var}).")
        if not model:
            if label == "OpenRouter":
                notes.append(f"Note: OpenRouter requires an explicit model name — set "
                             f"{model_path} (e.g. google/gemini-2.5-pro).")
            else:
                notes.append(f"Note: {model_path} is not set — the first model available at "
                             f"{host} will be used. Set it to choose (e.g. glm-5.3:cloud).")
    return notes


# ── Endpoint editor ─────────────────────────────────────────────────

_ENDPOINT_HELP = (
    "  Commands: [a]dd  [e]dit N  [d]elete N  [p]riority N  [Enter] done")


def _print_endpoint_table(entries: list[dict], indent: str = "  ") -> None:
    """Print the endpoints in preference order, each with a stable row number.

    Rows are listed best-first so the routing order is visible at a glance,
    but the number identifies the endpoint itself and never moves — otherwise
    changing one row's priority would silently renumber the others out from
    under the next command.
    """
    if not entries:
        print(f"{indent}No endpoints configured yet.")
        return
    order = sorted(range(len(entries)), key=lambda i: _entry_priority(entries[i]))
    models = [e.get("model") or "auto-detect" for e in entries]
    keys = [entry_key_label(e) for e in entries]
    width = max(len("HOST"), max(len(e.get("host", "")) for e in entries))
    mw = max(len("MODEL"), max(len(m) for m in models))
    kw = max(len("KEY"), max(len(k) for k in keys))
    print(f"{indent}{'#':>2}  {'PRIO':<4}  {'HOST':<{width}}  "
          f"{'MODEL':<{mw}}  {'KEY':<{kw}}  NAME")
    for i in order:
        e = entries[i]
        print(f"{indent}{i + 1:>2}  {_entry_priority(e):<4}  "
              f"{e.get('host', ''):<{width}}  "
              f"{models[i]:<{mw}}  {keys[i]:<{kw}}  "
              f"{e.get('name') or ''}")
    if len(entries) > 1 and all(_entry_priority(e) == 0 for e in entries):
        print(f"{indent}All endpoints share one tier — requests are spread "
              f"across them. Use 'p N' to rank one ahead.")


def entry_key_label(entry: dict) -> str:
    """How a config entry's endpoint is authenticated, for the table."""
    return endpoint_key_label(
        entry.get("host", ""),
        in_config=bool(entry.get("api_key") or entry.get("host_key")))


def _prompt_endpoint_key(host: str, previous_host: str | None) -> None:
    """Ask for one endpoint's API key and store it in the keychain.

    The key never goes into config.yaml — that file is world-readable on a
    shared host, and is the one thing here that gets copied around and pasted
    into issues.
    """
    move_endpoint_key(previous_host or "", host)
    existing = get_endpoint_key(host)
    if existing:
        hint = "••••" + existing[-4:]
    else:
        source = endpoint_key_source(host)
        hint = f"unset, falling back to {source}" if source else "not set"
    value = getpass.getpass(f"    API key [{hint}]: ").strip()
    if value == "-":
        delete_endpoint_key(host)
    elif value:
        store_endpoint_key(host, value)


def _prompt_entry(entry: dict | None, is_first: bool) -> dict | None:
    """Collect one endpoint's fields, pre-filled when editing.

    Returns None if no host was given, which cancels an add.
    """
    entry = dict(entry or {})
    previous_host = entry.get("host")
    host_default = entry.get("host") or (DEFAULT_HOST if is_first else "")
    host = input(f"    Endpoint URL [{host_default or 'required'}]: ").strip()
    host = host or host_default
    if not host:
        print("    No URL given — nothing added.")
        return None
    entry["host"] = host
    _prompt_endpoint_key(host, previous_host)

    model = input(f"    Model name [{entry.get('model') or 'auto-detect'}]: ").strip()
    if model == "-":
        entry.pop("model", None)
    elif model:
        entry["model"] = model

    name = input(f"    Label for logs [{entry.get('name') or 'optional'}]: ").strip()
    if name == "-":
        entry.pop("name", None)
    elif name:
        entry["name"] = name

    entry["priority"] = _prompt_priority(entry)
    return entry


def _prompt_priority(entry: dict) -> int:
    """Ask for an endpoint's rank, keeping the current value on blank input."""
    current = _entry_priority(entry)
    raw = input(f"    Priority — lower is preferred, equal numbers "
                f"load-balance [{current}]: ").strip()
    if not raw:
        return current
    try:
        return int(raw)
    except ValueError:
        print(f"    '{raw}' is not a whole number — keeping {current}.")
        return current


def _resolve_index(arg: str, entries: list[dict]) -> int | None:
    """Map a table row number back to an index in entries."""
    try:
        row = int(arg)
    except (TypeError, ValueError):
        print(f"  {arg or '(nothing)'!r} is not a row number — "
              f"try e.g. 'p 2'.")
        return None
    if not 1 <= row <= len(entries):
        print(f"  No row {row} — pick 1..{len(entries)}.")
        return None
    return row - 1


def _edit_endpoints(config: dict) -> None:
    """Interactively edit the model endpoints, then write them into config.

    Presents one list whether the config currently uses ``serving.endpoints``
    or the legacy host/host2 pair, and saves back into whichever shape the
    result fits.
    """
    entries = _endpoint_list(config) or _entries_from_host_pair(config)
    if not entries:
        print("  No endpoint configured yet — let's add the first one.")
        first = _prompt_entry(None, is_first=True)
        if first:
            entries.append(first)

    while True:
        print()
        _print_endpoint_table(entries)
        print(_ENDPOINT_HELP)
        parts = input("  endpoints> ").strip().split()
        if not parts:
            break
        cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

        if cmd in ("a", "add"):
            new = _prompt_entry(None, is_first=not entries)
            if not new:
                continue
            if any(e["host"] == new["host"] for e in entries):
                print(f"  {new['host']} is already configured.")
                continue
            entries.append(new)
        elif cmd in ("e", "edit"):
            idx = _resolve_index(arg, entries)
            if idx is not None:
                updated = _prompt_entry(entries[idx], is_first=False)
                if updated:
                    entries[idx] = updated
        elif cmd in ("d", "delete", "rm"):
            idx = _resolve_index(arg, entries)
            if idx is None:
                continue
            if len(entries) == 1:
                print("  At least one endpoint is required — "
                      "edit this one instead.")
                continue
            print(f"  Removed {entries.pop(idx)['host']}")
        elif cmd in ("p", "priority"):
            idx = _resolve_index(arg, entries)
            if idx is not None:
                entries[idx]["priority"] = _prompt_priority(entries[idx])
        else:
            print(f"  Unknown command {parts[0]!r}.")

    if entries:
        _write_endpoints(config, entries)


# ── Show current configuration ──────────────────────────────────────

def _print_settings(config: dict, settings: list[tuple]) -> None:
    """Print each setting's effective value, falling back to its default."""
    for dotpath, label, default in settings:
        current = _get_nested(config, dotpath)
        if current in (None, ""):
            current = default if default != "" else "not set"
        print(f"  {label:.<40s} {current}")


def secret_status(key: str, env_var: str | None) -> str:
    """One line describing a secret: masked value and where it came from.

    Shared with the in-session ``\\setup`` command so the wizard and the chat
    UI never disagree about whether a key is set — the question a user asks
    precisely when a request has just failed for want of one.
    """
    source = None
    value = None

    # Check keyring
    kr_val = get_secret(key)
    if kr_val:
        value = kr_val
        source = "keychain"

    # Check env var (takes precedence for display)
    if env_var:
        env_val = os.environ.get(env_var)
        if env_val:
            value = env_val
            source = "env var"

    if not value:
        return "not set"
    return f"••••{value[-4:]} ({source})"


def _print_secrets(secrets: list[tuple]) -> None:
    """Print each secret masked, annotated with where it was found."""
    for key, label, env_var in secrets:
        print(f"  {label:.<40s} {secret_status(key, env_var)}")


def show_config():
    """Print current configuration with masked secrets."""
    config = _load_config()

    print("\n  OnIt Configuration")
    print("  " + "─" * 50)

    print("\n  Model serving")
    entries = _endpoint_list(config) or _entries_from_host_pair(config)
    _print_endpoint_table(entries, indent="    ")
    print()
    _print_settings(config, SERVING_SETTINGS)
    # Only shown once one is actually holding something up: on a fresh install
    # every endpoint carries its own key and this section is noise.
    legacy = [entry for entry in LEGACY_SERVING_SECRETS
              if get_secret(entry[0])]
    if legacy:
        print("\n  Legacy fallback keys — used only by an endpoint that has "
              "none of its own")
        _print_secrets(legacy)

    print("\n  Preferences")
    _print_settings(config, GENERAL_SETTINGS)

    print("\n  Integrations")
    _print_secrets(INTEGRATION_SECRETS)

    print()
    for note in _provider_notes(config):
        print(f"  {note}")


# ── Interactive setup wizard ────────────────────────────────────────

def _prompt_settings(config: dict, settings: list[tuple]) -> None:
    """Prompt for each setting, writing answers into config."""
    for dotpath, label, default in settings:
        current = _get_nested(config, dotpath)
        display = current if current not in (None, "") else (default or "not set")
        value = input(f"  {label} [{display}]: ").strip()
        if value == "-":
            # Clear optional settings; reset required ones to their default
            _set_nested(config, dotpath, "" if default == "" else default)
        elif value:
            # Convert numeric strings to int where appropriate
            if dotpath in ("web_port", "timeout"):
                try:
                    value = int(value)
                except ValueError:
                    pass
            _set_nested(config, dotpath, value)
        elif current is None:
            _set_nested(config, dotpath, default)


def _prompt_secrets(secrets: list[tuple]) -> None:
    """Prompt for each secret, storing or clearing it in the keychain."""
    for key, label, env_var in secrets:
        existing = get_secret(key)
        if existing:
            hint = "••••" + existing[-4:]
        elif env_var and os.environ.get(env_var):
            hint = "set via env var"
        else:
            hint = "not set"

        value = getpass.getpass(f"  {label} [{hint}]: ").strip()
        if value == "-":
            delete_secret(key)
        elif value:
            store_secret(key, value)


def run_setup(show_only: bool = False):
    """Run the interactive setup wizard."""
    if show_only:
        show_config()
        return

    if not KEYRING_AVAILABLE:
        print("Warning: 'keyring' package not installed. "
              "Secrets will be stored in plaintext in the config file.",
              file=sys.stderr)
        print("Install it with: pip install keyring\n", file=sys.stderr)

    print("\n  OnIt Setup")
    print("  " + "─" * 50)
    print("  Press Enter to keep the current value; type '-' to clear it.")

    config = _load_config()

    # ── Model serving ────────────────────────────────────────────
    # Endpoints, how requests are spread across them, and the keys that
    # authenticate them — everything about talking to a model, in one place.
    print("\n  Model serving")
    print("  " + "─" * 50)
    print(f"  Endpoint examples — {ENDPOINT_EXAMPLES}")
    print("  Model examples — vLLM/local: auto-detect; "
          "Ollama cloud: glm-5.3:cloud; OpenRouter: google/gemini-2.5-pro")
    print("  Each endpoint takes its own API key; leave it blank where the "
          "server wants none.")
    _edit_endpoints(config)
    print()
    _prompt_settings(config, SERVING_SETTINGS)

    # ── Preferences ──────────────────────────────────────────────
    print("\n  Preferences")
    print("  " + "─" * 50)
    _prompt_settings(config, GENERAL_SETTINGS)

    # ── Integrations ─────────────────────────────────────────────
    print("\n  Integrations (optional — press Enter to skip)")
    print("  " + "─" * 50)
    _prompt_secrets(INTEGRATION_SECRETS)

    _save_config(config)

    print()
    for note in _provider_notes(config):
        print(f"  {note}")
    print()
    print("  Setup complete!")
    print(f"  Config saved to {CONFIG_PATH}")
    if KEYRING_AVAILABLE:
        print(f"  Secrets stored in OS keychain (service: '{SERVICE_NAME}')")
    else:
        print("  Warning: Secrets stored in plaintext in config file.")
        print("  Install 'keyring' for secure storage: pip install keyring")
    print()
    print("  Run 'onit setup --show' to review your configuration.")
    print("  Run 'onit' to start chatting.\n")
