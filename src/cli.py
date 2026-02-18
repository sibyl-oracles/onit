"""
CLI entry point for the OnIt agent.

Usage:
    onit                        # interactive terminal chat
    onit --web                  # Gradio web UI
    onit --config my.yaml       # custom config
    onit --a2a                  # A2A server mode
    onit --a2a-client --a2a-task "question"  # send task to A2A server (default: localhost:9001)
    onit --mcp                  # run MCP servers (default config)
    onit --mcp --config mcp.yaml  # run MCP servers with custom config
"""

import argparse
import asyncio
import base64
import json
import os
import sys

import requests
import yaml

from .onit import OnIt


def _download_files(text: str, server_url: str) -> str:
    """Download any files referenced in the response text from the A2A server."""
    import re
    # Match /uploads/filename patterns in the text
    pattern = re.compile(r'/uploads/([^\s\)\]"\'<>]+)')
    downloaded = []
    for match in pattern.finditer(text):
        filename = match.group(1)
        download_url = f"{server_url.rstrip('/')}/uploads/{filename}"
        try:
            resp = requests.get(download_url, timeout=60)
            resp.raise_for_status()
            local_path = os.path.join(os.getcwd(), filename)
            with open(local_path, "wb") as f:
                f.write(resp.content)
            downloaded.append(local_path)
        except Exception as e:
            downloaded.append(f"Failed to download {filename}: {e}")
    if downloaded:
        text += "\n\nDownloaded files:\n" + "\n".join(f"  - {p}" for p in downloaded)
    return text


def _upload_file(url: str, filepath: str) -> str:
    """Upload a file to the A2A server and return the uploaded filename."""
    filepath = os.path.abspath(os.path.expanduser(filepath))
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    filename = os.path.basename(filepath)
    with open(filepath, 'rb') as f:
        resp = requests.post(
            f"{url.rstrip('/')}/uploads/",
            files={'file': (filename, f)},
            timeout=60,
        )
        resp.raise_for_status()
    return filename


def _send_task(url: str, task: str, file: str = None, image: str = None) -> str:
    """Send a task to an OnIt A2A server and return the answer."""
    import threading
    import time

    # Upload file first if provided, then reference it in the task
    if file:
        filename = _upload_file(url, file)
        task = f"{task}\n\nFile uploaded to server: /uploads/{filename}"

    parts = [{"kind": "text", "text": task}]

    # Embed image inline as a FilePart in the same JSON-RPC payload
    if image:
        mime_types = {
            '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.gif': 'image/gif', '.bmp': 'image/bmp', '.webp': 'image/webp',
            '.tiff': 'image/tiff', '.tif': 'image/tiff',
        }
        ext = os.path.splitext(image)[1].lower()
        mime_type = mime_types.get(ext, 'image/png')
        with open(image, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        parts.append({
            "kind": "file",
            "file": {
                "bytes": image_data,
                "mimeType": mime_type,
                "name": os.path.basename(image),
            }
        })

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": parts,
                "messageId": "client-001",
            }
        },
    }

    # Elapsed time indicator while waiting for response
    stop_timer = threading.Event()
    start = time.monotonic()

    def _show_elapsed():
        while not stop_timer.is_set():
            elapsed = int(time.monotonic() - start)
            h, remainder = divmod(elapsed, 3600)
            m, s = divmod(remainder, 60)
            sys.stderr.write(f"\rWaiting... {h:02d}:{m:02d}:{s:02d}")
            sys.stderr.flush()
            stop_timer.wait(1.0)

    # Print initial message immediately on main thread
    sys.stderr.write("\rWaiting... 00:00:00")
    sys.stderr.flush()

    timer_thread = threading.Thread(target=_show_elapsed, daemon=True)
    timer_thread.start()
    try:
        resp = requests.post(url.rstrip("/"), json=payload, timeout=None)
        resp.raise_for_status()
    finally:
        stop_timer.set()
        timer_thread.join()
        # Clear the timer line
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
    data = resp.json()

    error = data.get("error")
    if error:
        return f"Error: {error}"

    result = data.get("result", {})
    text = None
    # A2A Task response (has status + artifacts)
    if "status" in result:
        for artifact in result.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    text = part["text"]
                    break
            if text:
                break
        # fallback: check result field
        if not text:
            task_result = result.get("result")
            if task_result:
                for part in task_result.get("parts", []):
                    if part.get("kind") == "text":
                        text = part["text"]
                        break
    # Direct Message response
    if not text and "parts" in result:
        for part in result.get("parts", []):
            if part.get("kind") == "text":
                text = part["text"]
                break

    if not text:
        return json.dumps(result, indent=2)

    # Download any files referenced in the response
    if "/uploads/" in text:
        text = _download_files(text, url)

    return text


def _find_default_config() -> str:
    """Locate the default config file, checking common locations."""
    candidates = [
        "configs/default.yaml",
        os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"),
        os.path.expanduser("~/.onit/config.yaml"),
        # Bundled config inside the installed package (pip install)
        os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return "configs/default.yaml"


def main():
    parser = argparse.ArgumentParser(
        prog="onit",
        description="OnIt — an intelligent agent for task automation and assistance.",
    )
    # General options
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the configuration YAML file.')
    parser.add_argument('--host', type=str, default=None,
                        help='LLM serving host URL (e.g. http://localhost:8000/v1). Overrides config and ONIT_HOST env var.')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (e.g. google/gemini-2.5-pro). Overrides serving.model in config.')
    parser.add_argument('--verbose', action='store_true', default=None,
                        help='Enable verbose logging.')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Request timeout in seconds (-1 for no timeout).')
    parser.add_argument('--template-path', type=str, default=None,
                        help='Path to custom prompt template YAML file.')

    # Text UI options
    parser.add_argument('--text-theme', type=str, default=None,
                        help='Text UI theme (e.g. "white", "dark").')
    parser.add_argument('--text-show-logs', action='store_true', default=None,
                        help='Show execution logs in text UI.')

    # Web UI options
    parser.add_argument('--web', action='store_true', default=None,
                        help='Launch Gradio web chat UI.')
    parser.add_argument('--web-port', type=int, default=None,
                        help='Port for Gradio web UI (default: 9000).')

    # A2A options
    parser.add_argument('--a2a', action='store_true', default=None,
                        help='Run as an A2A protocol server.')
    parser.add_argument('--a2a-port', type=int, default=None,
                        help='A2A server port (default: 9001).')
    parser.add_argument('--a2a-client', action='store_true', default=False,
                        help='Client mode: send a task to a remote OnIt A2A server and print the answer.')
    parser.add_argument('--a2a-host', type=str, default='http://localhost:9001',
                        help='A2A server URL for client mode (default: http://localhost:9001).')
    parser.add_argument('--a2a-task', type=str, default=None,
                        help='Task to execute in A2A loop or client mode.')
    parser.add_argument('--a2a-file', type=str, default=None,
                        help='File to upload to the A2A server along with the task.')
    parser.add_argument('--a2a-image', type=str, default=None,
                        help='Image file to send to the A2A server for vision processing.')
    parser.add_argument('--a2a-loop', action='store_true', default=None,
                        help='Enable A2A loop mode.')
    parser.add_argument('--a2a-period', type=float, default=None,
                        help='Period in seconds between A2A loop iterations (default: 10).')

    # MCP options
    parser.add_argument('--mcp', action='store_true', default=False,
                        help='Run MCP servers.')
    parser.add_argument('--mcp-host', type=str, default=None,
                        help='Override the host/IP in all MCP server URLs (e.g. 192.168.1.100).')
    parser.add_argument('--mcp-log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='MCP server log level.')
    args = parser.parse_args()

    # MCP server mode: run MCP servers and exit
    if args.mcp:
        from .mcp.servers.run import run_servers
        run_servers(config_path=args.config, log_level=args.mcp_log_level)
        return

    # Client mode: send task to remote A2A server and exit
    if args.a2a_client:
        if not args.a2a_task:
            print("Error: --a2a-client requires --a2a-task", file=sys.stderr)
            sys.exit(1)
        # Validate image file if provided
        if args.a2a_image:
            valid_image_ext = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
            image_path = os.path.abspath(os.path.expanduser(args.a2a_image))
            if not os.path.isfile(image_path):
                print(f"Error: Image file not found: {image_path}", file=sys.stderr)
                sys.exit(1)
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in valid_image_ext:
                print(f"Error: Invalid image file. Supported formats: {', '.join(sorted(valid_image_ext))}", file=sys.stderr)
                sys.exit(1)
            args.a2a_image = image_path
        try:
            answer = _send_task(args.a2a_host, args.a2a_task, file=args.a2a_file, image=args.a2a_image)
            print(answer)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # resolve config file
    config_path = args.config or _find_default_config()
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    else:
        config_data = {}
        if args.config:
            print(f"Warning: config file '{args.config}' not found, using defaults.",
                  file=sys.stderr)

    # override config with CLI args (only if explicitly provided)
    arg_to_config = {
        'a2a_loop': 'loop',
        'a2a_period': 'period',
        'a2a_task': 'task',
        'verbose': 'verbose',
        'text_theme': 'theme',
        'timeout': 'timeout',
        'text_show_logs': 'show_logs',
        'web': 'web',
        'web_port': 'web_port',
        'template_path': 'template_path',
        'a2a': 'a2a',
        'a2a_port': 'a2a_port',
    }
    for arg_name, config_key in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config_data[config_key] = value

    # --host overrides serving.host in config
    if args.host:
        config_data.setdefault('serving', {})['host'] = args.host

    # --model overrides serving.model in config
    if args.model:
        config_data.setdefault('serving', {})['model'] = args.model

    # --mcp-host overrides mcp.mcp_host in config
    if args.mcp_host:
        config_data.setdefault('mcp', {})['mcp_host'] = args.mcp_host

    onit = OnIt(config=config_data)
    asyncio.run(onit.run())


if __name__ == "__main__":
    main()
