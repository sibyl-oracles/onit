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
SECRETS = [
    ("host_key",               "OpenRouter API key",
     "OPENROUTER_API_KEY"),
    ("ollama_api_key",         "Ollama API key (enables web search and cloud LLM access)",
     "OLLAMA_API_KEY"),
    ("vllm_api_key",           "vLLM API key (required if vLLM is started with --api-key)",
     "VLLM_API_KEY"),
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
    ("host2_key",              "LLM API key for the second model server (optional)",
     "ONIT_HOST2_KEY"),
    ("github_token",           "GitHub personal access token (for git operations)",
     "GITHUB_TOKEN"),
    ("huggingface_token",      "HuggingFace access token (for model hub access)",
     "HF_TOKEN"),
]

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
SETTINGS = [
    ("serving.host",  "LLM endpoint URL (vLLM / OpenRouter / Ollama)",
     "http://localhost:8000/v1"),
    ("serving.model", "Model name (blank = auto-detect from endpoint)", ""),
    ("serving.host2", "Second LLM endpoint URL (optional, enables load balancing)", ""),
    ("serving.model2", "Model name on second server (blank = auto-detect)", ""),
    ("serving.load_balancer",
     "Load balancing algorithm (sticky / round_robin / random / least_busy)",
     "sticky"),
    ("theme",        "UI theme (dark / white)",  "dark"),
    ("web_port",     "Web UI port",              "9000"),
    ("timeout",      "Request timeout in seconds (-1 = none)", "-1"),
]

# Example endpoints shown at the top of the wizard, one per provider.
ENDPOINT_EXAMPLES = (
    "vLLM: http://localhost:8000/v1  |  "
    "Ollama cloud: https://ollama.com  |  "
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

def _provider_notes(config: dict) -> list[str]:
    """Check host/model/key combinations and return human-readable notes.

    Ollama cloud and OpenRouter endpoints need an API key. OpenRouter also
    needs an explicit model name (auto-detection would pick an arbitrary
    entry from its huge model list); for Ollama cloud an explicit model is
    recommended since auto-detection takes the first available model.
    """
    notes = []
    for host_path, model_path, key_names in (
        ("serving.host",  "serving.model",  ("host_key",)),
        ("serving.host2", "serving.model2", ("host2_key", "host_key")),
    ):
        host = str(_get_nested(config, host_path) or "")
        if not host:
            continue
        model = _get_nested(config, model_path)
        if "ollama.com" in host or "ollama.ai" in host:
            if not get_secret("ollama_api_key"):
                notes.append(f"Note: {host_path} is an Ollama cloud endpoint but no "
                             "Ollama API key is set (rerun 'onit setup' or export OLLAMA_API_KEY).")
            if not model:
                notes.append(f"Note: {model_path} is not set — the first model available at "
                             f"{host} will be used. Set it to choose (e.g. glm-5.1:cloud).")
        elif "openrouter.ai" in host:
            if not any(get_secret(k) for k in key_names):
                notes.append(f"Note: {host_path} is an OpenRouter endpoint but no "
                             "LLM API key is set (rerun 'onit setup' or export OPENROUTER_API_KEY).")
            if not model:
                notes.append(f"Note: OpenRouter requires an explicit model name — set "
                             f"{model_path} (e.g. google/gemini-2.5-pro).")
    return notes


# ── Show current configuration ──────────────────────────────────────

def show_config():
    """Print current configuration with masked secrets."""
    config = _load_config()

    print("\n  OnIt Configuration")
    print("  " + "─" * 50)

    # Settings
    for dotpath, label, default in SETTINGS:
        current = _get_nested(config, dotpath)
        if current in (None, ""):
            current = default if default != "" else "not set"
        print(f"  {label:.<40s} {current}")

    print()

    # Secrets
    for key, label, env_var in SECRETS:
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

        if value:
            masked = "••••" + value[-4:]
            print(f"  {label:.<40s} {masked} ({source})")
        else:
            print(f"  {label:.<40s} not set")

    print()
    for note in _provider_notes(config):
        print(f"  {note}")


# ── Interactive setup wizard ────────────────────────────────────────

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
    print(f"  Endpoint examples — {ENDPOINT_EXAMPLES}")
    print("  Model examples — vLLM/local: auto-detect; "
          "Ollama cloud: glm-5.1:cloud; OpenRouter: google/gemini-2.5-pro\n")

    config = _load_config()

    # ── Non-secret settings ──────────────────────────────────────
    for dotpath, label, default in SETTINGS:
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

    print()

    # ── Secrets ──────────────────────────────────────────────────
    for key, label, env_var in SECRETS:
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
