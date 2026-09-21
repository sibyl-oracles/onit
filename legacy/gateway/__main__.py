"""Run a legacy Telegram or Viber gateway from the command line.

    python -m legacy.gateway [telegram|viber|auto] [--webhook-url URL] [--port N]

Token resolution mirrors what ``onit serve gateway`` used to do: the
``TELEGRAM_BOT_TOKEN`` / ``VIBER_BOT_TOKEN`` environment variables, or
``gateway_token`` / ``telegram_bot_token`` / ``viber_bot_token`` in the
OnIt config.
"""

import argparse
import asyncio
import os
import sys


def _resolve_token(config: dict, key: str, env_var: str) -> str | None:
    value = os.environ.get(env_var)
    if value:
        return value
    value = config.get(key)
    if value:
        return value
    try:
        from src.setup import get_secret
        return get_secret(key)
    except Exception:
        return None


def _load_config() -> dict:
    try:
        import yaml
        from src.setup import CONFIG_PATH
        path = CONFIG_PATH
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m legacy.gateway",
        description="Run the legacy Telegram or Viber gateway.",
    )
    parser.add_argument("gateway_type", nargs="?", choices=["telegram", "viber", "auto"],
                        default="auto", help="Gateway type (default: auto-detect from env).")
    parser.add_argument("--webhook-url", type=str, default=None,
                        help="Public HTTPS URL for the Viber webhook "
                             "(or set VIBER_WEBHOOK_URL).")
    parser.add_argument("--port", type=int, default=None,
                        help="Local port for the Viber webhook server (default: 8443).")
    args = parser.parse_args()

    config = _load_config()
    telegram_token = _resolve_token(config, "telegram_bot_token", "TELEGRAM_BOT_TOKEN")
    viber_token = _resolve_token(config, "viber_bot_token", "VIBER_BOT_TOKEN")

    gateway_type = args.gateway_type
    if gateway_type == "auto":
        if telegram_token:
            gateway_type = "telegram"
        elif viber_token:
            gateway_type = "viber"
        else:
            print("Error: python -m legacy.gateway requires TELEGRAM_BOT_TOKEN or "
                  "VIBER_BOT_TOKEN in the environment.", file=sys.stderr)
            sys.exit(1)

    from src.onit import OnIt

    config_data = dict(config)
    config_data["gateway"] = gateway_type
    if gateway_type == "viber":
        if not viber_token:
            print("Error: the Viber gateway requires VIBER_BOT_TOKEN.",
                  file=sys.stderr)
            sys.exit(1)
        config_data["gateway_token"] = viber_token
        webhook_url = (args.webhook_url
                       or config_data.get("viber_webhook_url")
                       or os.environ.get("VIBER_WEBHOOK_URL"))
        if not webhook_url:
            print("Error: the Viber gateway requires a webhook URL. "
                  "Set VIBER_WEBHOOK_URL or pass --webhook-url.", file=sys.stderr)
            sys.exit(1)
        config_data["viber_webhook_url"] = webhook_url
        if args.port is not None:
            config_data["viber_port"] = args.port
    else:
        if not telegram_token:
            print("Error: the Telegram gateway requires TELEGRAM_BOT_TOKEN.",
                  file=sys.stderr)
            sys.exit(1)
        config_data["gateway_token"] = telegram_token

    onit = OnIt(config=config_data)

    if gateway_type == "viber":
        from .viber import ViberGateway
        gw = ViberGateway(
            onit, onit.gateway_token,
            webhook_url=onit.viber_webhook_url,
            port=onit.viber_port,
            show_logs=onit.show_logs,
        )
    else:
        from .telegram import TelegramGateway
        gw = TelegramGateway(onit, onit.gateway_token, show_logs=onit.show_logs)

    gw.run_sync()


if __name__ == "__main__":
    main()