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
    ("host_key",               "LLM API key (for OpenRouter or remote vLLM)",
     "OPENROUTER_API_KEY"),
    ("ollama_api_key",         "Ollama API key (enables web search and cloud LLM access)",
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

# ── Non-secret settings ────────────────────────────────────────────
# (dot_path, prompt_label, default_value)
SETTINGS = [
    ("serving.host", "LLM endpoint URL",        "http://localhost:8000/v1"),
    ("theme",        "UI theme (dark / white)",  "dark"),
    ("web_port",     "Web UI port",              "9000"),
    ("timeout",      "Request timeout in seconds (-1 = none)", "-1"),
]


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


def get_secret(key: str) -> str | None:
    """Retrieve a secret from the OS keychain, falling back to file storage."""
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


# ── Show current configuration ──────────────────────────────────────

def show_config():
    """Print current configuration with masked secrets."""
    config = _load_config()

    print("\n  OnIt Configuration")
    print("  " + "─" * 50)

    # Settings
    for dotpath, label, default in SETTINGS:
        current = _get_nested(config, dotpath) or default
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
    print("  Press Enter to keep the current value.\n")

    config = _load_config()

    # ── Non-secret settings ──────────────────────────────────────
    for dotpath, label, default in SETTINGS:
        current = _get_nested(config, dotpath)
        display = current if current is not None else default
        value = input(f"  {label} [{display}]: ").strip()
        if value:
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
        if value:
            if KEYRING_AVAILABLE:
                store_secret(key, value)
            else:
                # Fallback: store in config file (plaintext)
                _set_nested(config, f"_secrets.{key}", value)

    _save_config(config)

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
