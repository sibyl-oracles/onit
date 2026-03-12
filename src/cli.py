"""
CLI entry point for the OnIt agent.

Usage:
    onit                        # interactive terminal chat
    onit setup                  # interactive setup wizard
    onit setup --show           # show current configuration
    onit --web                  # Gradio web UI
    onit --gateway              # Telegram bot gateway
    onit --config my.yaml       # custom config
    onit --a2a                  # A2A server mode
    onit --client --a2a-task "question"  # send task to A2A server (default: localhost:9001)
"""

import argparse
import asyncio
import base64
import json
import os
import socket
import sys
import time
import threading

import requests
import yaml

from .onit import OnIt


def _download_files(text: str, server_url: str) -> str:
    """Download any files referenced in the response text from the A2A server."""
    import re
    # Match /uploads/filename patterns in the text
    pattern = re.compile(r'/uploads/([^\s\)\]"\'<>`*]+)')
    downloaded = []
    for match in pattern.finditer(text):
        filename = match.group(1)
        download_url = f"{server_url.rstrip('/')}/uploads/{filename}"
        try:
            resp = requests.get(download_url, timeout=60)
            resp.raise_for_status()
            local_path = os.path.join(os.getcwd(), os.path.basename(filename))
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


def _build_a2a_parts(task: str, file: str = None, image: str = None) -> list:
    """Build the A2A message parts list from task text and optional files."""
    import mimetypes as _mimetypes

    parts = [{"kind": "text", "text": task}]

    if file:
        filepath = os.path.abspath(os.path.expanduser(file))
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        mime_type = _mimetypes.guess_type(filepath)[0] or 'application/octet-stream'
        with open(filepath, 'rb') as f:
            file_data = base64.b64encode(f.read()).decode('utf-8')
        parts.append({
            "kind": "file",
            "file": {
                "bytes": file_data,
                "mimeType": mime_type,
                "name": os.path.basename(filepath),
            }
        })

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

    return parts


def _extract_a2a_text(result: dict) -> str | None:
    """Extract text from an A2A result dict (Task or Message)."""
    text = None
    if "status" in result:
        for artifact in result.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    text = part["text"]
                    break
            if text:
                break
        if not text:
            task_result = result.get("result")
            if task_result:
                for part in task_result.get("parts", []):
                    if part.get("kind") == "text":
                        text = part["text"]
                        break
    if not text and "parts" in result:
        for part in result.get("parts", []):
            if part.get("kind") == "text":
                text = part["text"]
                break
    return text


class _StreamState:
    """Mutable state shared between SSE streaming helpers."""

    def __init__(self, stop_timer: threading.Event, timer_thread: threading.Thread):
        self.stop_timer = stop_timer
        self.timer_thread = timer_thread
        self.printed_len: int = 0
        self.final_text: str | None = None
        self.raw_result: dict = {}
        self.spinner_cleared: bool = False
        self.cursor_shown: bool = False

    def erase_cursor(self) -> None:
        """Remove the blinking block cursor and restore the terminal cursor."""
        if self.cursor_shown:
            sys.stdout.write("\b \b")
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            self.cursor_shown = False

    def show_cursor(self) -> None:
        """Hide terminal cursor and show a blinking white block instead."""
        if not self.cursor_shown:
            sys.stdout.write("\033[?25l")
            sys.stdout.write("\033[5m\u2588\033[0m")
            sys.stdout.flush()
            self.cursor_shown = True

    def clear_spinner(self) -> None:
        """Stop the elapsed-time spinner and clear its line."""
        if not self.spinner_cleared:
            self.stop_timer.set()
            self.timer_thread.join()
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
            self.spinner_cleared = True


def _handle_sse_events(resp: requests.Response, state: _StreamState) -> None:
    """Process SSE event lines from a streaming A2A response.

    Updates *state* in place with streamed text deltas, the final text,
    and the raw result dict.
    """
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue

        result = event.get("result", {})
        status = result.get("status", {})
        event_state = status.get("state", "")

        if event_state == "working":
            msg = status.get("message", {})
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    full = part["text"]
                    if len(full) > state.printed_len:
                        state.clear_spinner()
                        state.erase_cursor()
                        sys.stdout.write(full[state.printed_len:])
                        sys.stdout.flush()
                        state.printed_len = len(full)
                        state.show_cursor()
                    break

        elif event_state == "completed":
            state.raw_result = result
            state.final_text = _extract_a2a_text(result)
            if not state.final_text:
                msg = status.get("message", {})
                for part in msg.get("parts", []):
                    if part.get("kind") == "text":
                        state.final_text = part["text"]
                        break

        elif "parts" in result:
            state.raw_result = result
            state.final_text = _extract_a2a_text(result)


def _format_output(state: _StreamState, url: str) -> str:
    """Produce the final return value after streaming/response is complete.

    Handles the JSON-dump fallback, trailing-text flush for streamed
    responses, and file downloads.
    """
    if state.final_text is None:
        return json.dumps(state.raw_result, indent=2)

    if state.printed_len > 0:
        remaining = state.final_text[state.printed_len:]
        if remaining:
            sys.stdout.write(remaining)
        sys.stdout.write("\n")
        sys.stdout.flush()

    if "/uploads/" in state.final_text:
        state.final_text = _download_files(state.final_text, url)

    if state.printed_len > 0:
        return ""

    return state.final_text


def _send_task(url: str, task: str, file: str = None, image: str = None) -> str:
    """Send a task to an OnIt A2A server using SSE streaming.

    Uses ``message/stream`` so the server can push incremental
    ``TaskStatusUpdateEvent`` (state=working) events.  Each event
    carries the accumulated text so far; the client prints only the
    new delta.  The "Waiting ..." spinner is replaced by live output
    as soon as the first token arrives.

    Falls back to the non-streaming ``message/send`` path if the SSE
    request fails (e.g. older server without streaming support).
    """
    parts = _build_a2a_parts(task, file=file, image=image)

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/stream",
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

    sys.stderr.write("\rWaiting... 00:00:00")
    sys.stderr.flush()

    timer_thread = threading.Thread(target=_show_elapsed, daemon=True)
    timer_thread.start()

    state = _StreamState(stop_timer, timer_thread)

    try:
        resp = requests.post(
            url.rstrip("/"),
            json=payload,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=None,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            _handle_sse_events(resp, state)
        else:
            # Non-streaming JSON response (fallback)
            data = resp.json()
            error = data.get("error")
            if error:
                state.clear_spinner()
                return f"Error: {error}"
            state.raw_result = data.get("result", {})
            state.final_text = _extract_a2a_text(state.raw_result)

    except requests.RequestException:
        # SSE failed — fall back to non-streaming message/send
        state.clear_spinner()
        payload["method"] = "message/send"
        resp = requests.post(url.rstrip("/"), json=payload, timeout=None)
        resp.raise_for_status()
        data = resp.json()
        error = data.get("error")
        if error:
            return f"Error: {error}"
        state.raw_result = data.get("result", {})
        state.final_text = _extract_a2a_text(state.raw_result)
    finally:
        state.erase_cursor()
        state.clear_spinner()

    return _format_output(state, url)


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


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


def _mcp_servers_ready(config_data: dict, timeout: float = 15.0) -> bool:
    """Wait for all enabled MCP servers to be reachable.

    Parses the agent config's mcp.servers list and checks each URL's port.
    Returns True if all servers respond within timeout, False otherwise.
    """
    from urllib.parse import urlparse

    servers = config_data.get('mcp', {}).get('servers', [])
    endpoints = []
    for s in servers:
        if s.get('enabled', True) and s.get('url'):
            parsed = urlparse(s['url'])
            host = parsed.hostname or '127.0.0.1'
            port = parsed.port or 80
            endpoints.append((host, port, s.get('name', 'Unknown')))

    if not endpoints:
        return True

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        all_up = True
        for host, port, _ in endpoints:
            if not _is_port_open(host, port):
                all_up = False
                break
        if all_up:
            return True
        time.sleep(0.5)
    return False


def _start_mcp_servers_background(log_level='ERROR'):
    """Start MCP servers in a daemon thread. Blocks forever (runs in background)."""
    from .mcp.servers.run import run_servers
    try:
        run_servers(log_level=log_level)
    except Exception:
        pass


def _ensure_mcp_servers(config_data: dict, log_level='ERROR'):
    """Start MCP servers if they are not already running, then wait for readiness."""
    from urllib.parse import urlparse

    # Propagate documents_path to MCP servers via environment variable
    docs_path = config_data.get('documents_path', '')
    if docs_path:
        os.environ['ONIT_DOCUMENTS_PATH'] = docs_path

    # Check if servers are already running by probing the first enabled server port
    servers = config_data.get('mcp', {}).get('servers', [])
    already_running = True
    for s in servers:
        if s.get('enabled', True) and s.get('url'):
            parsed = urlparse(s['url'])
            host = parsed.hostname or '127.0.0.1'
            port = parsed.port or 80
            if not _is_port_open(host, port, timeout=0.3):
                already_running = False
                break

    if already_running and servers:
        return

    # Start MCP servers in a daemon thread
    mcp_thread = threading.Thread(
        target=_start_mcp_servers_background,
        args=(log_level,),
        daemon=True,
    )
    mcp_thread.start()

    # Wait for all servers to be reachable
    if not _mcp_servers_ready(config_data, timeout=15.0):
        print("Warning: some MCP servers may not have started in time.",
              file=sys.stderr)


def _merge_base(override: dict, base: dict):
    """Recursively merge *override* into *base* (in-place).

    Values from *override* take precedence.  For nested dicts the merge
    is recursive so that e.g. ``serving.host`` from the override replaces
    only that key, not the entire ``serving`` block.
    """
    for key, value in override.items():
        if (key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)):
            _merge_base(value, base[key])
        else:
            base[key] = value


def _build_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="onit",
        description="OnIt — an intelligent agent for task automation and assistance.",
    )

    # Subcommands (setup)
    subparsers = parser.add_subparsers(dest="command")
    setup_parser = subparsers.add_parser("setup",
                                         help="Interactive setup wizard.")
    setup_parser.add_argument("--show", action="store_true",
                              help="Display current configuration.")

    # General options
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the configuration YAML file.')
    parser.add_argument('--host', type=str, default=None,
                        help='LLM serving host URL (e.g. http://localhost:8000/v1). Overrides config and ONIT_HOST env var.')
    parser.add_argument('--verbose', action='store_true', default=None,
                        help='Enable verbose logging.')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Request timeout in seconds (-1 for no timeout).')
    parser.add_argument('--template-path', type=str, default=None,
                        help='Path to custom prompt template YAML file.')
    parser.add_argument('--documents-path', type=str, default=None,
                        help='Path to local documents directory. The model will search here before the web.')
    parser.add_argument('--topic', type=str, default=None,
                        help='Default topic context (e.g. "machine learning"). The model will assume this topic unless specified otherwise.')
    parser.add_argument('--prompt-intro', type=str, default=None,
                        help='Custom system prompt intro for the model (default: "I am a helpful AI assistant. My name is OnIt.").')
    # Text UI options
    parser.add_argument('--text-theme', type=str, default=None,
                        help='Text UI theme (e.g. "white", "dark").')
    parser.add_argument('--show-logs', action='store_true', default=None,
                        help='Show execution logs.')
    parser.add_argument('--no-stream', action='store_true', default=None,
                        dest='no_stream',
                        help='Disable streaming of tokens (streaming is enabled by default for text, web and a2a modes).')

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
    parser.add_argument('--client', '--a2a-client', action='store_true', default=False,
                        dest='a2a_client',
                        help='Client mode: send a task to a remote OnIt A2A server and print the answer.')
    parser.add_argument('--a2a-host', type=str, default='http://localhost:9001',
                        help='A2A server URL for client mode (default: http://localhost:9001).')
    parser.add_argument('--a2a-task', '--task', type=str, default=None,
                        help='Task to execute in A2A loop or client mode.')
    parser.add_argument('--a2a-file', '--file', type=str, default=None,
                        help='File to upload to the A2A server along with the task.')
    parser.add_argument('--a2a-image', '--image', type=str, default=None,
                        help='Image file to send to the A2A server for vision processing (model is a VLM).')
    parser.add_argument('--a2a-loop', '--loop', action='store_true', default=None,
                        help='Enable A2A loop mode.')
    parser.add_argument('--a2a-period', '--period', type=float, default=None,
                        help='Period in seconds between A2A loop iterations (default: 10).')

    # Gateway options
    parser.add_argument('--gateway', nargs='?', const='auto', default=None,
                        choices=['telegram', 'viber', 'auto'],
                        help='Run as a messaging gateway. Options: telegram, viber, auto '
                             '(auto-detect from env vars). Default when flag used alone: auto.')
    parser.add_argument('--viber-webhook-url', type=str, default=None,
                        help='Public HTTPS URL for Viber webhook (or set VIBER_WEBHOOK_URL env var).')
    parser.add_argument('--viber-port', type=int, default=None,
                        help='Local port for Viber webhook server (default: 8443).')

    # MCP options
    parser.add_argument('--mcp-host', type=str, default=None,
                        help='Override the host/IP in all MCP server URLs (e.g. 192.168.1.100).')
    parser.add_argument('--ollama-api-key', type=str, default=None,
                        help='Ollama API key for web search. Overrides OLLAMA_API_KEY env var.')
    parser.add_argument('--openweathermap-api-key', type=str, default=None,
                        help='OpenWeatherMap API key for weather tool. Overrides OPENWEATHERMAP_API_KEY env var.')
    parser.add_argument('--mcp-sse', type=str, action='append', default=None,
                        help='URL of an external MCP tools server using SSE transport (can be repeated). '
                             'Example: --mcp-sse http://localhost:8080/sse')

    return parser


def _parse_and_resolve_config(args: argparse.Namespace) -> dict:
    """Load the config file, merge setup defaults, and apply CLI overrides.

    Returns the fully-resolved config dict ready for use.
    """
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

    # Merge ~/.onit/config.yaml (from 'onit setup') as a base layer.
    # Setup values fill in gaps but never override the project/user config.
    from .setup import CONFIG_PATH as _setup_config_path, resolve_credential
    _resolved_config = os.path.realpath(config_path) if os.path.isfile(config_path) else None
    _setup_resolved = os.path.realpath(_setup_config_path)
    if (_resolved_config != _setup_resolved
            and os.path.isfile(_setup_config_path)):
        with open(_setup_config_path, 'r') as f:
            setup_data = yaml.safe_load(f) or {}
        # Deep-merge: setup_data is the base, config_data overrides
        _merge_base(config_data, setup_data)
        config_data = setup_data

    # override config with CLI args (only if explicitly provided)
    arg_to_config = {
        'a2a_loop': 'loop',
        'a2a_period': 'period',
        'a2a_task': 'task',
        'verbose': 'verbose',
        'text_theme': 'theme',
        'timeout': 'timeout',
        'show_logs': 'show_logs',
        'web': 'web',
        'web_port': 'web_port',
        'template_path': 'template_path',
        'documents_path': 'documents_path',
        'topic': 'topic',
        'prompt_intro': 'prompt_intro',
        'a2a': 'a2a',
        'a2a_port': 'a2a_port',
        'gateway': 'gateway',
        'viber_webhook_url': 'viber_webhook_url',
        'viber_port': 'viber_port',
    }
    for arg_name, config_key in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config_data[config_key] = value

    # --no-stream explicitly disables streaming (default is True)
    if args.no_stream:
        config_data['stream'] = False

    # --host overrides serving.host in config
    if args.host:
        config_data.setdefault('serving', {})['host'] = args.host

    # --mcp-host overrides mcp.mcp_host in config
    if args.mcp_host:
        config_data.setdefault('mcp', {})['mcp_host'] = args.mcp_host

    # --mcp-sse adds external MCP servers to the servers list
    if args.mcp_sse:
        servers = config_data.setdefault('mcp', {}).setdefault('servers', [])
        for i, url in enumerate(args.mcp_sse):
            servers.append({
                'name': f'ExternalSSE_{i}',
                'description': f'External MCP server at {url}',
                'url': url,
                'enabled': True,
            })

    # Check that essential environment variables are set
    serving = config_data.get('serving', {})
    host = serving.get('host') or os.environ.get('ONIT_HOST')
    host_key = serving.get('host_key', '')

    # Resolve host_key from keyring if not set via config/env
    if not host_key or host_key == 'EMPTY':
        kr_key = resolve_credential(None, 'OPENROUTER_API_KEY', 'host_key')
        if kr_key:
            host_key = kr_key
            config_data.setdefault('serving', {})['host_key'] = host_key

    missing = []
    if not host:
        missing.append('ONIT_HOST (or set serving.host in config, or run: onit setup)')
    elif 'openrouter' in (host or '').lower():
        if not host_key:
            missing.append('OPENROUTER_API_KEY (or run: onit setup)')

    if missing:
        print("Error: missing required configuration:", file=sys.stderr)
        for var in missing:
            print(f"  - {var}", file=sys.stderr)
        print("\nSet via environment variable, CLI option (--host), config YAML, "
              "or run: onit setup", file=sys.stderr)
        sys.exit(1)

    # Check OLLAMA_API_KEY for web search support
    ollama_api_key = resolve_credential(
        args.ollama_api_key, 'OLLAMA_API_KEY', 'ollama_api_key')
    if ollama_api_key:
        os.environ['OLLAMA_API_KEY'] = ollama_api_key
    else:
        os.environ['ONIT_DISABLE_WEB_SEARCH'] = '1'

    # Check OPENWEATHERMAP_API_KEY for weather tool support
    weather_api_key = resolve_credential(
        args.openweathermap_api_key, 'OPENWEATHERMAP_API_KEY',
        'openweathermap_api_key')
    if not weather_api_key:
        weather_api_key = resolve_credential(
            None, 'OPENWEATHER_API_KEY', 'openweathermap_api_key')
    if weather_api_key:
        os.environ['OPENWEATHERMAP_API_KEY'] = weather_api_key
    else:
        os.environ['ONIT_DISABLE_WEATHER'] = '1'

    # Resolve gateway type and token
    gateway_type = config_data.get('gateway')
    if gateway_type:
        telegram_token = resolve_credential(
            None, 'TELEGRAM_BOT_TOKEN', 'telegram_bot_token')
        viber_token = resolve_credential(
            None, 'VIBER_BOT_TOKEN', 'viber_bot_token')

        if gateway_type == 'auto':
            # Auto-detect: prefer Telegram for backward compat, fall back to Viber
            if telegram_token:
                gateway_type = 'telegram'
            elif viber_token:
                gateway_type = 'viber'
            else:
                print("Error: --gateway requires TELEGRAM_BOT_TOKEN or VIBER_BOT_TOKEN "
                      "environment variable.", file=sys.stderr)
                sys.exit(1)

        if gateway_type == 'viber':
            if not viber_token:
                print("Error: --gateway viber requires VIBER_BOT_TOKEN environment variable.",
                      file=sys.stderr)
                sys.exit(1)
            config_data['gateway_token'] = viber_token
            # Resolve webhook URL
            webhook_url = (config_data.get('viber_webhook_url')
                           or os.environ.get('VIBER_WEBHOOK_URL'))
            if not webhook_url:
                print("Error: Viber gateway requires a webhook URL. "
                      "Set VIBER_WEBHOOK_URL env var or --viber-webhook-url.",
                      file=sys.stderr)
                sys.exit(1)
            config_data['viber_webhook_url'] = webhook_url
        else:  # telegram
            if not telegram_token:
                print("Error: --gateway telegram requires TELEGRAM_BOT_TOKEN "
                      "environment variable.", file=sys.stderr)
                sys.exit(1)
            config_data['gateway_token'] = telegram_token

        config_data['gateway'] = gateway_type

    return config_data


def _setup_servers(config_data: dict) -> None:
    """Start MCP servers and print tool-availability warnings."""
    _ensure_mcp_servers(
        config_data,
        log_level='DEBUG' if config_data.get('verbose') else 'ERROR',
    )

    # Print tool availability warnings before launching any mode
    if os.environ.get('ONIT_DISABLE_WEATHER'):
        print("Warning: OPENWEATHERMAP_API_KEY is not set. Weather tool is unavailable.",
              file=sys.stderr)
        print("  Set via env var, --openweathermap-api-key, or run: onit setup\n",
              file=sys.stderr)
    if os.environ.get('ONIT_DISABLE_WEB_SEARCH'):
        print("WARNING: OLLAMA_API_KEY is not set or invalid. "
              "Internet search is DISABLED.", file=sys.stderr)
        print("  OnIt will NOT be able to search the web in this session.",
              file=sys.stderr)
        print("  Set via env var, --ollama-api-key, or run: onit setup\n",
              file=sys.stderr)


def _dispatch_mode(config_data: dict) -> None:
    """Instantiate OnIt and launch the appropriate run mode."""
    onit = OnIt(config=config_data)
    if config_data.get('gateway'):
        onit.run_gateway_sync()
    else:
        asyncio.run(onit.run())


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Setup wizard
    if args.command == "setup":
        from .setup import run_setup
        run_setup(show_only=args.show)
        return

    # Client mode: send task to remote A2A server and exit
    if args.a2a_client:
        if not args.a2a_task:
            print("Error: --client requires --a2a-task", file=sys.stderr)
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

    config_data = _parse_and_resolve_config(args)
    _setup_servers(config_data)
    _dispatch_mode(config_data)


if __name__ == "__main__":
    main()
