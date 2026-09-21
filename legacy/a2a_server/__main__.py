"""Run the legacy A2A protocol server from the command line.

    python -m legacy.a2a_server [--port 9001]

Builds the same ``OnIt`` agent the active front ends use and serves it over
the A2A JSON-RPC protocol. ``onit ask`` (kept in the active CLI) can send
tasks to it:

    python -m legacy.a2a_server --port 9001 &
    onit ask "hello" --server http://localhost:9001
"""

import argparse
import asyncio
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m legacy.a2a_server",
        description="Run the legacy A2A protocol server.",
    )
    parser.add_argument("--port", type=int, default=None,
                        help="A2A server port (default: 9001, or a2a_port in config).")
    args = parser.parse_args()

    config_data = {}
    try:
        import yaml
        from src.setup import CONFIG_PATH
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
    except Exception:
        pass

    config_data["a2a"] = True
    if args.port is not None:
        config_data["a2a_port"] = args.port

    from src.onit import OnIt
    from .server import run_a2a_server

    onit = OnIt(config=config_data)
    asyncio.run(run_a2a_server(onit))


if __name__ == "__main__":
    main()