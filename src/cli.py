"""
CLI entry point for the OnIt agent.

Usage:
    onit                        # interactive terminal chat
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
    args = parser.parse_args()

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

    # --host overrides serving.host in config
    if args.host:
        config_data.setdefault('serving', {})['host'] = args.host

    # --model overrides serving.model in config
    if args.model:
        config_data.setdefault('serving', {})['model'] = args.model

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

    missing = []
    if not host:
        missing.append('ONIT_HOST (or set serving.host in config)')
    elif 'openrouter' in (host or '').lower():
        if not host_key and not os.environ.get('OPENROUTER_API_KEY'):
            missing.append('OPENROUTER_API_KEY (or set serving.host_key in config)')

    if missing:
        print("Error: missing required configuration:", file=sys.stderr)
        for var in missing:
            print(f"  - {var}", file=sys.stderr)
        print("\nSet via environment variable, CLI option (--host), or in your config YAML.", file=sys.stderr)
        sys.exit(1)

    # Check OLLAMA_API_KEY for web search support
    ollama_api_key = args.ollama_api_key or os.environ.get('OLLAMA_API_KEY')
    if ollama_api_key:
        os.environ['OLLAMA_API_KEY'] = ollama_api_key
    else:
        print("Warning: OLLAMA_API_KEY is not set. Web search tool will be disabled.",
              file=sys.stderr)
        print("Set via environment variable or --ollama-api-key CLI option.",
              file=sys.stderr)
        os.environ['ONIT_DISABLE_WEB_SEARCH'] = '1'

    # Check OPENWEATHERMAP_API_KEY for weather tool support
    weather_api_key = (args.openweathermap_api_key
                       or os.environ.get('OPENWEATHERMAP_API_KEY')
                       or os.environ.get('OPENWEATHER_API_KEY'))
    if weather_api_key:
        os.environ['OPENWEATHERMAP_API_KEY'] = weather_api_key
    else:
        print("Warning: OPENWEATHERMAP_API_KEY is not set. Weather tool will be disabled.",
              file=sys.stderr)
        print("Set via environment variable or --openweathermap-api-key CLI option.",
              file=sys.stderr)
        os.environ['ONIT_DISABLE_WEATHER'] = '1'

    # Resolve gateway type and token
    gateway_type = config_data.get('gateway')
    if gateway_type:
        telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        viber_token = os.environ.get('VIBER_BOT_TOKEN')

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

    # Auto-start MCP servers if not already running
    _ensure_mcp_servers(
        config_data,
        log_level='DEBUG' if config_data.get('verbose') else 'ERROR',
    )

    onit = OnIt(config=config_data)
    if config_data.get('gateway'):
        onit.run_gateway_sync()
    else:
        asyncio.run(onit.run())


if __name__ == "__main__":
    main()
