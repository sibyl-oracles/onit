"""
CLI entry point for the OnIt agent.

Usage:
    onit                                          # interactive terminal chat (resumes last session)
    onit --restart-session                        # interactive terminal chat, fresh session
    onit setup                                    # interactive setup wizard
    onit setup --show                             # show current configuration
    onit sessions                                 # list previous sessions
    onit doctor [--deep]                          # run the live self-check battery
    onit resume [TAG_OR_ID]                       # resume a previous session
    onit serve web [--port 9000]                  # launch the web UI
    onit serve loop "task" [--period 60]          # repeat a task on a timer
    onit --config my.yaml                         # custom config file
    onit --container                              # run in a hardened Docker container
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import threading
from pathlib import Path

import yaml
from fastmcp import Client

from .onit import OnIt
from .lib.tools import (is_stdio_server as _is_stdio_server,
                        register_stdio_servers, apply_default_mcp_servers)


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



def _is_external_server(server: dict) -> bool:
    """True for a server OnIt must not start or re-port: it lives elsewhere.

    Marked by ``external: true`` in the config (the old --mcp-sse /
    --mcp-server flags wrote the same thing as generated names).
    """
    if server.get('external'):
        return True
    name = server.get('name', '')
    return name.startswith('ExternalSSE_') or name.startswith('ExternalMCP_')


def _mcp_servers_ready(config_data: dict, timeout: float = 15.0) -> bool:
    """Wait for all locally-managed MCP servers to be ready to serve MCP requests.

    Probes each server with an actual list_tools() MCP call rather than a raw
    TCP port check.  A server is considered ready only when it can respond to
    MCP protocol requests, which happens after the ASGI app is fully initialized
    — port-open alone is not sufficient.

    External servers (``external: true`` in the config) are excluded since
    they are not managed by this process.
    Returns True if all servers respond within timeout, False otherwise.
    """
    servers = config_data.get('mcp', {}).get('servers', [])
    urls = [
        s['url']
        for s in servers
        if not _is_external_server(s) and not _is_stdio_server(s)
        and s.get('enabled', True) and s.get('url')
    ]

    if not urls:
        return True

    async def _probe(url: str) -> bool:
        try:
            async with Client(url) as client:
                await client.list_tools()
                return True
        except Exception:
            return False

    async def _all_ready() -> bool:
        results = await asyncio.gather(*[_probe(url) for url in urls])
        return all(results)

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if asyncio.run(_all_ready()):
            return True
        time.sleep(0.5)
    return False


def _start_mcp_servers_background(log_level='ERROR', port_overrides=None):
    """Start MCP servers in a daemon thread. Blocks forever (runs in background)."""
    from .mcp.servers.run import run_servers
    try:
        run_servers(log_level=log_level, port_overrides=port_overrides)
    except Exception as exc:
        print(f"ERROR: MCP server background thread failed: {exc}", file=sys.stderr)


def _assign_free_ports(servers: list, config_data: dict) -> dict:
    """Point every socket-served MCP server at a free port, and report the map.

    Ports are searched for at or above 18200 rather than fixed, so that two
    people running OnIt on one machine each get their own servers instead of
    the second silently attaching to the first one's — which used to run their
    tools under the first user's account, inside the first user's sandbox.

    Set ``mcp.fixed_ports: true`` to keep the configured ports as written.
    """
    from urllib.parse import urlparse, urlunparse
    from src.mcp.servers.run import find_free_ports

    targets = [s for s in servers
               if s.get('enabled', True) and s.get('url')
               and not _is_stdio_server(s) and not _is_external_server(s)]
    if not targets:
        return {}

    if config_data.get('mcp', {}).get('fixed_ports'):
        return {s['name']: urlparse(s['url']).port for s in targets
                if s.get('name') and urlparse(s['url']).port}

    port_base = config_data.get('mcp', {}).get('port_base')
    ports = find_free_ports(len(targets), base=port_base)
    assigned: dict = {}
    for server, port in zip(targets, ports):
        parsed = urlparse(server['url'])
        server['url'] = urlunparse(
            parsed._replace(netloc=f"{parsed.hostname or '127.0.0.1'}:{port}"))
        if server.get('name'):
            assigned[server['name']] = port
    return assigned


def _ensure_mcp_servers(config_data: dict, log_level='ERROR'):
    """Start this process's MCP servers and wait for the socket-served ones.

    Every OnIt process gets its own servers. The stdio ones are spawned by the
    MCP client on first use; the rest are started here on ports found free at
    startup.
    """
    # Propagate data_path to MCP servers via an environment variable.
    # Fall back to OnIt's own default (~/sandbox) when unset so the MCP servers write
    # to the same absolute local path the UI displays, instead of their /tmp fallback.
    data_path = config_data.get('data_path') or str(Path.home() / "sandbox")
    data_path = str(Path(data_path).expanduser().resolve())
    os.environ['ONIT_DATA_PATH'] = data_path

    # Fill in the default servers first. A config that names none still gets
    # them — from OnIt, further down startup — and allocating ports before
    # they exist would leave those servers pointing at ports nothing listens
    # on. The list must be complete here, where the ports are chosen.
    config_data.setdefault('mcp', {}).setdefault('servers', [])
    servers = config_data['mcp']['servers']
    apply_default_mcp_servers(servers)

    register_stdio_servers(servers, data_path, log_level)
    port_overrides = _assign_free_ports(servers, config_data)

    # Start the socket-served MCP servers in a daemon thread. Always: the
    # runner has its own config and may serve more than the client lists (the
    # VLM tools server, for one). It allocates ports for anything not named in
    # the overrides, and returns immediately if there is nothing to start.
    mcp_thread = threading.Thread(
        target=_start_mcp_servers_background,
        args=(log_level, port_overrides),
        daemon=True,
    )
    mcp_thread.start()

    # Wait for all servers to be reachable (spawn start method on Linux is slower)
    if not _mcp_servers_ready(config_data, timeout=30.0):
        print("Warning: some MCP servers may not have started in time.",
              file=sys.stderr)


# What the web UI serves with, over and above the shared ``serving:`` block.
# A browser turn is a chat: one iteration, no tool loop to amortise a reasoning
# pass over, and someone watching the composer while a 27B model deliberates —
# so the reasoning is the whole wait.  The terminal is where runs are long
# enough for thinking to earn its latency, and it keeps it.  Sampling moves
# with the switch because thinking and instruct mode want different numbers
# (see the table in docs/MODEL_SERVING.md); leaving the thinking-mode values on
# a model told not to think is the wrong half of the pair.
WEB_SERVING_DEFAULTS = {
    'think': False,
    'temperature': 0.7,
    'top_p': 0.8,
}


def _apply_web_serving_defaults(config_data: dict, is_web: bool,
                                force_think: bool = False):
    """Layer :data:`WEB_SERVING_DEFAULTS` over ``serving:`` for a web run.

    These beat a plain ``serving.think`` rather than filling in for it: one
    config serves both UIs, so a ``serving:`` block that turns reasoning on
    for the terminal would otherwise turn it on for the browser too, and the
    split would never take effect for anyone who had asked for thinking at
    all.  ``serving.web`` is the way back — any key set there wins over the
    default beside it, so ``serving.web.think: true`` restores reasoning in
    the browser and leaves the terminal alone.

    The defaults land as one trade rather than three independent settings:
    thinking off, and the sampling an instruct-mode turn wants.  Put the
    reasoning back — through ``serving.web`` or a ``--think`` on the command
    line, which *force_think* reports — and the sampling half goes with it,
    falling through to ``serving:``, which is already tuned for a model that
    thinks.  Sampling named explicitly under ``serving.web`` applies either
    way; it is a stated preference, not half of a default.

    The ``serving.web`` block is consumed here on every run, web or not, so
    that a terminal session never carries it into chat().
    """
    serving_cfg = config_data.get('serving')
    web_cfg = (serving_cfg.pop('web', None)
               if isinstance(serving_cfg, dict) else None)
    if not is_web:
        return
    web_cfg = dict(web_cfg) if isinstance(web_cfg, dict) else {}
    overrides = ({} if (force_think or web_cfg.get('think'))
                 else dict(WEB_SERVING_DEFAULTS))
    overrides.update(web_cfg)
    config_data.setdefault('serving', {}).update(overrides)


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


def _token_count(text: str) -> int:
    """Parse a token count written as a plain number or with a k/M suffix.

    ``32768``, ``128k`` and ``1M`` are all accepted (case-insensitive, with
    underscores or commas allowed as digit separators), because the numbers
    these flags take are the kind nobody wants to spell out in full.
    """
    cleaned = text.strip().replace(",", "").replace("_", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmM]?)", cleaned)
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid token count: {text!r} (use e.g. 32768, 128k, or 1M)")
    value = float(match.group(1)) * {"": 1, "k": 1_000, "m": 1_000_000}[
        match.group(2).lower()]
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"token count must be at least 1, got {text!r}")
    return int(value)


def _build_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="onit",
        description="OnIt — an agent harness for task automation, in the terminal or on the web.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # setup
    setup_parser = subparsers.add_parser("setup", help="Interactive setup wizard.")
    setup_parser.add_argument("--show", action="store_true",
                              help="Display current configuration.")

    # sessions
    sessions_parser = subparsers.add_parser("sessions",
                                            help="List and manage previous sessions.")
    sessions_parser.add_argument("--limit", type=int, default=20,
                                 help="Maximum number of sessions to list (default: 20).")
    sessions_parser.add_argument("--rebuild", action="store_true",
                                 help="Rebuild the session index from existing JSONL files.")
    sessions_parser.add_argument("--tag", type=str, nargs=2, metavar=("SESSION", "TAG"),
                                 help="Tag a session: --tag <session-id-or-tag> <new-tag>")
    sessions_parser.add_argument("--clear", action="store_true",
                                 help="Delete all previous sessions and the index.")

    # learn: inspect what the agent has recorded about its own runs
    learn_parser = subparsers.add_parser(
        "learn",
        help="Show what OnIt has recorded about its own runs.")
    learn_parser.add_argument("--session", type=str, default=None,
                              metavar="ID",
                              help="Print one session's trajectory records as JSON.")
    learn_parser.add_argument("--json", action="store_true",
                              help="Print the summary as JSON instead of a table.")
    learn_parser.add_argument("--events", action="store_true",
                              help="Print the loop event log (tool lifecycle and "
                                   "the like) instead of the task summary.")

    # doctor: run the live self-check battery from the shell, no session needed
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run the live self-check battery (same as \\doctor in the text UI).")
    doctor_parser.add_argument("--deep", action="store_true", default=False,
                               help="Also exercise a live model reply and a full "
                                    "tool-calling turn (costs tokens, adds up to a "
                                    "couple of minutes on a slow endpoint).")
    doctor_parser.add_argument("--json", action="store_true", default=False,
                               help="Print the report as JSON instead of text.")
    doctor_parser.add_argument("--keep-session", action="store_true", default=False,
                               help="Keep the throwaway session the check creates "
                                    "(visible in 'onit sessions').")

    # resume
    resume_parser = subparsers.add_parser("resume", help="Resume a previous session.")
    resume_parser.add_argument("session", nargs="?", default="last",
                               help='Session tag, UUID, or "last" (default: last).')

    # serve: run OnIt in a server or daemon mode
    serve_parser = subparsers.add_parser("serve",
                                         help="Run OnIt in a server or daemon mode.")
    serve_sub = serve_parser.add_subparsers(dest="serve_mode", metavar="MODE")
    serve_sub.required = True

    # serve web
    web_p = serve_sub.add_parser("web", help="Launch the web UI.")
    web_p.add_argument("--port", type=int, default=None,
                       help="Web UI port (default: 9000, or web_port in config).")
    web_p.add_argument("--no-login", action="store_true", dest="no_login",
                       help="Run the web UI without requiring Google login "
                            "(sessions are open to anyone who can reach the port).")
    web_p.add_argument("--voice", action="store_true", dest="voice",
                       help="Enable full-duplex speech-to-speech. Requires a "
                            "NemotronLabs VoiceChat container (see docs/VOICE.md).")

    # serve loop
    loop_p = serve_sub.add_parser("loop",
                                   help="Repeat a task on a configurable timer.")
    loop_p.add_argument("task", type=str,
                        help="Task to execute repeatedly.")
    loop_p.add_argument("--period", type=float, default=None,
                        help="Seconds between iterations (default: 10, or period in config).")

    # ── General options ──────────────────────────────────────────────────────
    parser.add_argument("--resume", type=str, default=None, metavar="TAG_OR_ID",
                        help='Resume a previous session by tag, UUID, or "last" for the most recent.')
    parser.add_argument("--restart-session", "--new-session", action="store_true",
                        default=False, dest="restart_session",
                        help="Start a fresh session instead of resuming the last "
                             "one (terminal chat resumes the last session by default).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the configuration YAML file.")
    parser.add_argument("--host", type=str, default=None,
                        help="LLM serving host URL (e.g. http://localhost:8000/v1). "
                             "Overrides config and ONIT_HOST env var.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to use (e.g. Qwen/Qwen3-30B-A3B-Instruct-2507). "
                             "Skips auto-detection from endpoint.")
    parser.add_argument("--load-balancer", type=str, default=None,
                        dest="load_balancer",
                        choices=["sticky", "round_robin", "random", "least_busy"],
                        help="Load balancing algorithm across the two hosts "
                             "(default: sticky — new sessions are assigned "
                             "round-robin, then each session stays on its "
                             "host unless a timeout/error fails it over).")
    parser.add_argument("--ollama-fallback-only", default=None,
                        dest="ollama_fallback_only",
                        action=argparse.BooleanOptionalAction,
                        help="Whether Ollama endpoints only serve while no "
                             "vLLM/OpenRouter endpoint is healthy (default: "
                             "true). Use --no-ollama-fallback-only to put "
                             "Ollama endpoints in normal load-balancing "
                             "rotation. Overrides serving.ollama_fallback_only "
                             "in the config YAML.")
    parser.add_argument("--max-tokens", type=_token_count, default=None,
                        dest="max_tokens", metavar="N",
                        help="Max output tokens per model response, e.g. 32768, "
                             "128k or 1M. Each request is still clamped to what "
                             "is left of the context window. Overrides "
                             "serving.max_tokens in the config YAML "
                             "(default: 131072).")
    parser.add_argument("--max-context-tokens", type=_token_count, default=None,
                        dest="max_context_tokens", metavar="N",
                        help="Context window size in tokens, e.g. 128k or 1M. "
                             "Normally detected from the endpoint; set it when "
                             "the server does not report its own, or to hold the "
                             "agent to a smaller window than the model allows. "
                             "On Ollama this also sizes num_ctx unless "
                             "serving.num_ctx says otherwise. Overrides "
                             "serving.max_context_tokens in the config YAML "
                             "(default: whatever the endpoint reports, else "
                             "262144).")
    parser.add_argument("--verbose", action="store_true", default=None,
                        help="Enable verbose logging.")
    parser.add_argument("--think", action="store_true", default=None,
                        help="Enable thinking/reasoning mode (CoT).")
    parser.add_argument("--no-stream", action="store_true", default=None, dest="no_stream",
                        help="Disable streaming of tokens (streaming is on by default).")
    parser.add_argument("--show-logs", action="store_true", default=None,
                        help="Show tool execution logs.")

    # ── Isolation ────────────────────────────────────────────────────────────
    parser.add_argument("--data-path", type=str, default=None, dest="data_path",
                        help="Working directory for agent files (default: ~/sandbox). "
                             "Overrides data_path in the config YAML.")
    parser.add_argument("--unrestricted", action="store_true", default=False,
                        help="Run with unrestricted host filesystem access. "
                             "The agent can read/write any path, use any working directory, "
                             "and install packages freely. Use only in trusted environments.")
    # Default on for a run that belongs to the person who started it, off for
    # one that serves other people (see the approval wiring further down), so
    # the flag is passed with a value only to override that.
    parser.add_argument("--auto", action="store_true", default=None, dest="auto",
                        help="Answer yes to every command approval prompt, so a run "
                             "never stops to ask. On by default for your own runs "
                             "(terminal chat, --loop, one-shot); pass it explicitly "
                             "to get the same for a web deployment, "
                             "which serves other people and so keeps asking. Only "
                             "questions that would have been put to a person are "
                             "answered: privilege escalation, container and "
                             "remote-shell tools, operator deny rules, and the "
                             "session path jail on a shared web deployment are still "
                             "refused.")
    parser.add_argument("--no-auto", "--ask", action="store_false", dest="auto",
                        help="Ask before running a command policy will not run on "
                             "its own, instead of approving it automatically.")
    parser.add_argument("--container", action="store_true", default=False,
                        help="Run the entire OnIt process inside a hardened Docker container "
                             "so a breach cannot reach the host OS.")
    parser.add_argument("--container-gpus", type=str, default=None, dest="container_gpus",
                        help='Pass GPUs into the container (e.g. "all" or "device=0,1"). '
                             "Requires the NVIDIA Container Toolkit on the host.")
    parser.add_argument("--container-mount", type=str, action="append", default=None,
                        dest="container_mount",
                        help="Extra bind mount for the container, e.g. "
                             "/host/path:/container/path:ro. Repeatable.")
    parser.add_argument("--container-memory", type=str, default=None,
                        dest="container_memory",
                        help="Hard memory cap for the container (e.g. 16g).")
    parser.add_argument("--container-shm-size", type=str, default=None,
                        dest="container_shm_size",
                        help="/dev/shm size inside the container (default: 4g).")
    parser.add_argument("--container-tmp-size", type=str, default=None,
                        dest="container_tmp_size",
                        help="/tmp tmpfs size inside the container (default: 16g).")
    parser.add_argument("--container-allow-installs", action="store_true",
                        default=False, dest="container_allow_installs",
                        help="Permit package installs inside the container. "
                             "Installs are still restricted to version-pinned "
                             "packages (e.g. pip install name==1.2.3).")

    # ── External MCP servers ─────────────────────────────────────────────────
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

    # Map top-level CLI flags that still exist
    for arg_name, config_key in [
        ('verbose', 'verbose'),
        ('show_logs', 'show_logs'),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            config_data[config_key] = value

    # serve subcommand: inject mode-specific settings into config_data
    if getattr(args, 'command', None) == 'serve':
        serve_mode = getattr(args, 'serve_mode', None)
        if serve_mode == 'web':
            config_data['web'] = True
            if args.port is not None:
                config_data['web_port'] = args.port
            if getattr(args, 'no_login', False):
                config_data['web_require_auth'] = False
            if getattr(args, 'voice', False):
                voice_cfg = dict(config_data.get('voice') or {})
                voice_cfg['enabled'] = True
                config_data['voice'] = voice_cfg
        elif serve_mode == 'loop':
            config_data['loop'] = True
            config_data['task'] = args.task
            if getattr(args, 'period', None) is not None:
                config_data['period'] = args.period

    # Keyed off the resolved flag, not the subcommand, so a config file that
    # says ``web: true`` gets the same treatment as ``onit serve web``.  The
    # --think flag is only read here to keep the sampling consistent with it;
    # the flag itself is applied below, so it still wins outright.
    _apply_web_serving_defaults(config_data, bool(config_data.get('web')),
                                bool(getattr(args, 'think', False)))

    # --no-stream explicitly disables streaming (default is True)
    if args.no_stream:
        config_data['stream'] = False

    # --host, --model, --think override serving config
    if args.host:
        serving_cfg = config_data.setdefault('serving', {})
        serving_cfg['host'] = args.host
        # An explicit --host means a single endpoint: drop any second host
        # from config/env so the load balancer can't route requests to a
        # leftover server (e.g. a vLLM host2 shadowing an explicitly
        # requested Ollama host, which is fallback-only).
        for key in ('host2', 'model2', 'host2_key'):
            serving_cfg.pop(key, None)
        os.environ.pop('ONIT_HOST2', None)
        # Same reasoning for a configured endpoints list — it would otherwise
        # take precedence over the host the user just named on the CLI.
        serving_cfg.pop('endpoints', None)
    if args.model:
        config_data.setdefault('serving', {})['model'] = args.model
    if getattr(args, 'load_balancer', None):
        config_data.setdefault('serving', {})['load_balancer'] = args.load_balancer
    if getattr(args, 'ollama_fallback_only', None) is not None:
        config_data.setdefault('serving', {})['ollama_fallback_only'] = \
            args.ollama_fallback_only
    if getattr(args, 'max_tokens', None) is not None:
        config_data.setdefault('serving', {})['max_tokens'] = args.max_tokens
    if getattr(args, 'max_context_tokens', None) is not None:
        serving_cfg = config_data.setdefault('serving', {})
        serving_cfg['max_context_tokens'] = args.max_context_tokens
        # Ollama allocates its KV cache from num_ctx and otherwise auto-sizes it
        # to a ceiling well below a 1M window, so an explicit context size has to
        # reach that knob too — unless the config names num_ctx itself, which is
        # the more specific setting and keeps precedence.
        serving_cfg.setdefault('num_ctx', args.max_context_tokens)
    if args.think:
        config_data.setdefault('serving', {})['think'] = True
    if args.data_path:
        config_data['data_path'] = args.data_path

    # Check that essential environment variables are set
    serving = config_data.get('serving', {})
    host = serving.get('host') or os.environ.get('ONIT_HOST')
    host_key = serving.get('host_key', '')
    # serving.endpoints supplies the hosts on its own; OnIt validates the list
    # and reports per-entry problems when it builds the load balancer.
    endpoints_configured = bool(serving.get('endpoints'))

    # Resolve host_key from keyring if not set via config/env. Only for
    # OpenRouter/OpenAI hosts — the stored legacy key is a key for that
    # provider, and injecting it into serving.host_key for a vLLM host would
    # shadow VLLM_API_KEY / the vllm_api_key keychain entry with the wrong key.
    #
    # A key stored for this endpoint's own URL is more specific than the
    # provider-wide one, and chat() will find it; injecting the legacy key
    # here would put it in front and authenticate as the wrong account.
    from src.setup import endpoint_key_source, legacy_key_for
    endpoint_key = bool(host) and endpoint_key_source(host) == 'endpoint'
    if (not host_key or host_key == 'EMPTY') and not endpoint_key and host:
        _keyring_key, _env_var, _label, _required = legacy_key_for(host)
        # Ollama and vLLM keys are not serving credentials in the same sense:
        # Ollama's doubles as the web-search key, and a vLLM host wants no
        # key at all — neither must land in serving.host_key.
        if _label not in ('Ollama cloud', 'vLLM'):
            kr_key = resolve_credential(None, _env_var, _keyring_key)
            if kr_key:
                host_key = kr_key
                config_data.setdefault('serving', {})['host_key'] = host_key

    missing = []
    if not host and not endpoints_configured:
        missing.append('ONIT_HOST (or set serving.host in config, or run: onit setup)')
    elif 'openrouter' in (host or '').lower():
        if not host_key and not endpoint_key:
            missing.append('OPENROUTER_API_KEY (or run: onit setup)')
    elif 'api.openai.com' in (host or '').lower():
        if not host_key and not endpoint_key:
            missing.append('OPENAI_API_KEY (or run: onit setup)')

    if missing:
        print("Error: missing required configuration:", file=sys.stderr)
        for var in missing:
            print(f"  - {var}", file=sys.stderr)
        print("\nSet via environment variable, CLI option (--host), config YAML, "
              "or run: onit setup", file=sys.stderr)
        sys.exit(1)

    # API keys: resolved from env vars and keyring only
    ollama_api_key = resolve_credential(None, 'OLLAMA_API_KEY', 'ollama_api_key')
    if ollama_api_key:
        os.environ['OLLAMA_API_KEY'] = ollama_api_key
    else:
        os.environ['ONIT_DISABLE_WEB_SEARCH'] = '1'

    weather_api_key = resolve_credential(None, 'OPENWEATHERMAP_API_KEY', 'openweathermap_api_key')
    if not weather_api_key:
        weather_api_key = resolve_credential(None, 'OPENWEATHER_API_KEY', 'openweathermap_api_key')
    if weather_api_key:
        os.environ['OPENWEATHERMAP_API_KEY'] = weather_api_key
    else:
        os.environ['ONIT_DISABLE_WEATHER'] = '1'

    # Web UI login: Google OAuth2 credentials live in the keyring (stored by
    # 'onit setup'); a value in the config file takes precedence.
    if config_data.get('web'):
        for key, env_var in [('web_google_client_id', 'GOOGLE_CLIENT_ID'),
                             ('web_google_client_secret', 'GOOGLE_CLIENT_SECRET')]:
            if not config_data.get(key):
                val = resolve_credential(None, env_var, key)
                if val:
                    config_data[key] = val

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
        print("  Set via env var or run: onit setup\n", file=sys.stderr)
    if os.environ.get('ONIT_DISABLE_WEB_SEARCH'):
        print("WARNING: OLLAMA_API_KEY is not set or invalid. "
              "Internet search is DISABLED.", file=sys.stderr)
        print("  OnIt will NOT be able to search the web in this session.",
              file=sys.stderr)
        print("  Set via env var or run: onit setup\n", file=sys.stderr)


def _dispatch_mode(config_data: dict) -> None:
    """Instantiate OnIt and launch the appropriate run mode."""
    onit = OnIt(config=config_data)
    asyncio.run(onit.run())


def _run_doctor(args: argparse.Namespace, config_data: dict) -> int:
    """Run the self-check battery against a throwaway session; return exit code.

    The battery is written for a live session, so it needs a live agent: this
    builds one exactly the way a real session would — MCP servers started,
    tools discovered, a session registered — and deletes the session again
    afterwards.  The checks then read the same attributes (config_data,
    tool_registry, load_balancer, ...) they read in the text UI, which is
    the point: a pass here means the same thing a pass in the session means.
    """
    from .sessions import delete_session
    from .ui.doctor import render_report, run_checks

    # A doctor run is a diagnostic, not a deployment: strip the mode keys so
    # OnIt builds the plain terminal-chat shape (no web server, no loop) and
    # never auto-resumes someone's last real session. The a2a/gateway keys
    # are legacy config; they are stripped so old config files cannot turn a
    # diagnostic into a server.
    for key in ('web', 'a2a', 'gateway', 'loop', 'resume_session_id',
                'web_require_auth'):
        config_data.pop(key, None)
    # The battery's own probes are allowlisted, but the deep tool-calling turn
    # asks the model to run bash — give it the same standing a terminal chat
    # would, so the check fails on a broken loop rather than on a refused
    # approval nobody is there to answer.
    os.environ['ONIT_APPROVAL_CHANNEL'] = '1'
    os.environ['ONIT_AUTO_APPROVE'] = '1'

    try:
        # OnIt() prints the banner, the tool list and the balancer summary as
        # it starts.  In --json mode those prints would sit in front of the
        # JSON on stdout and make it unparseable, so they are captured and
        # dropped there; text mode keeps them as the run's startup trail.
        import contextlib
        import io
        sink = io.StringIO() if args.json else None
        with contextlib.redirect_stdout(sink if sink is not None
                                        else sys.__stdout__):
            agent = OnIt(config=config_data)
    except Exception as e:
        print(f"Error: could not start the agent for self-check: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    session_id = getattr(agent, "session_id", None)
    sessions_dir = os.path.expanduser(
        config_data.get('session_path', '~/.onit/sessions'))
    try:
        results = asyncio.run(run_checks(agent, deep=args.deep))
    finally:
        # The session existed only to give the battery something to check.
        # Kept only under --keep-session, for reading a failure's details in
        # the JSONL; the index entry goes with it either way.
        if session_id and not args.keep_session:
            delete_session(session_id, sessions_dir)

    if args.json:
        print(json.dumps({
            "deep": args.deep,
            "results": [vars(r) if not hasattr(r, "__dataclass_fields__")
                        else {"name": r.name, "state": r.state,
                              "detail": r.detail, "elapsed": r.elapsed}
                        for r in results],
        }, indent=2))
    else:
        print(render_report(results, deep=args.deep))

    failed = sum(1 for r in results if r.state == "fail")
    return 1 if failed else 0


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # --container: re-exec the whole process inside a hardened Docker container.
    # Must happen before any config load or server setup touches the host.
    if getattr(args, 'container', False):
        from .container_launcher import run as _container_run, strip_launcher_args
        sys.exit(_container_run(
            strip_launcher_args(sys.argv[1:]),
            gpus=getattr(args, 'container_gpus', None),
            mounts=getattr(args, 'container_mount', None) or [],
            memory=getattr(args, 'container_memory', None),
            shm_size=getattr(args, 'container_shm_size', None),
            tmp_size=getattr(args, 'container_tmp_size', None),
            allow_installs=getattr(args, 'container_allow_installs', False),
        ))

    # Setup wizard
    if args.command == "setup":
        from .setup import run_setup
        run_setup(show_only=args.show)
        return

    # Sessions management
    if args.command == "sessions":
        from .sessions import (list_sessions, format_sessions_table,
                               rebuild_index, resolve_session, tag_session,
                               clear_sessions)
        sessions_dir = os.path.expanduser("~/.onit/sessions")
        if args.clear:
            answer = input("This will delete ALL session history. Are you sure? (yes/no): ")
            if answer.strip().lower() in ("yes", "y"):
                count = clear_sessions(sessions_dir)
                print(f"Deleted {count} session(s).")
            else:
                print("Cancelled.")
            return
        if args.rebuild:
            print("Rebuilding session index...")
            rebuild_index(sessions_dir)
            print("Done.")
        if args.tag:
            identifier, new_tag = args.tag
            sid = resolve_session(identifier, sessions_dir)
            if not sid:
                print(f"Error: Session '{identifier}' not found.", file=sys.stderr)
                sys.exit(1)
            result = tag_session(sid, new_tag, sessions_dir)
            if result is True:
                print(f"Tagged session {sid[:8]}... as '{new_tag}'")
            elif isinstance(result, str):
                print(f"Error: {result}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Error: Session not found.", file=sys.stderr)
                sys.exit(1)
            return
        sessions = list_sessions(sessions_dir, limit=args.limit)
        print(format_sessions_table(sessions))
        return

    # Trajectory store: read-only, and deliberately not gated on a config file
    # or a reachable model — "is anything being recorded" has to be answerable
    # when the rest of the stack is down.
    if args.command == "learn":
        from .learn import format_status, read_session, summarize
        # Only the learn block is needed, so the file is read directly rather
        # than run through the full resolver — that one reaches for the
        # keychain and the MCP hosts to answer a question about a directory.
        _learn_config_path = args.config or _find_default_config()
        config_data = {}
        if os.path.isfile(_learn_config_path):
            with open(_learn_config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        if args.session:
            records = read_session(args.session, config_data)
            if not records:
                print(f"No trajectory recorded for session '{args.session}'.",
                      file=sys.stderr)
                sys.exit(1)
            print(json.dumps(records, indent=2))
            return
        if args.json:
            summary = summarize(config_data)
            summary["models"] = dict(summary["models"])
            summary["ratings"] = dict(summary["ratings"])
            summary["totals"] = dict(summary["totals"])
            print(json.dumps(summary, indent=2))
            return
        if args.events:
            from .learn.events import summarize_events, tool_timeline
            events = summarize_events(config_data)
            if not events["total"]:
                print("No loop events recorded yet. They appear when the "
                      "tool lifecycle starts emitting (registry loads, "
                      "tool loads/updates/archives).")
                return
            print(json.dumps(events, indent=2))
            return
        print(format_status(config_data))
        return

    # doctor: the live self-check battery, from the shell.  Runs the same
    # checks as \doctor in the text UI against a throwaway session, prints
    # the report and exits non-zero when anything failed, so it can gate a
    # deploy or sit at the end of an update script.  The battery is the
    # product here, not a conversation: nothing is dispatched to a mode.
    if args.command == "doctor":
        # A machine that has never been set up must still be diagnosable —
        # "your config is missing serving.host" is exactly the kind of answer
        # a self-check exists to give.  Resolution exits when the config is
        # incomplete; catching that leaves an empty dict for the battery to
        # report on instead of dying before it ran anything.
        try:
            config_data = _parse_and_resolve_config(args)
        except SystemExit:
            config_data = {}
        # The servers are the thing under test: start this process's own
        # before the battery polls their ports.
        _setup_servers(config_data)
        sys.exit(_run_doctor(args, config_data))

    # resume subcommand: translate to --resume flag and continue normal startup
    if args.command == "resume":
        args.resume = args.session

    config_data = _parse_and_resolve_config(args)

    # Session selection.  The terminal chat continues where it left off, so an
    # explicit --resume is optional: without one we resume the last session.
    # --restart-session opts out and starts from scratch.  Server modes
    # (web/loop) manage their own sessions and never auto-resume.
    resume_target = args.resume
    auto_resume = False
    if not resume_target and not args.restart_session and not (
            config_data.get('web') or config_data.get('loop')):
        resume_target = "last"
        auto_resume = True

    if resume_target:
        from .sessions import resolve_session
        sessions_dir = os.path.expanduser(
            config_data.get('session_path', '~/.onit/sessions'))
        sid = resolve_session(resume_target, sessions_dir)
        # An index entry can outlive its JSONL file; treat that as "no session".
        if sid and not os.path.exists(os.path.join(sessions_dir, f"{sid}.jsonl")):
            sid = None
        if not sid:
            # Nothing to resume is only an error when the user asked for a
            # specific session; the automatic path just starts a new one.
            if not auto_resume:
                print(f"Error: Session '{resume_target}' not found.", file=sys.stderr)
                print("Use 'onit sessions' to list available sessions.", file=sys.stderr)
                sys.exit(1)
        else:
            config_data['resume_session_id'] = sid
            if auto_resume:
                print(f"Resuming last session: {sid[:8]}... "
                      "(use --restart-session to start a new one)")
            else:
                print(f"Resuming session: {sid[:8]}...")

    # Must be set before MCP servers are spawned so child processes inherit it.
    if args.unrestricted:
        os.environ['ONIT_UNRESTRICTED'] = '1'
        print("Warning: running in unrestricted mode — agent has full host filesystem access.",
              file=sys.stderr)

    # Web UI mode: enforce ~/.onit/settings.json permission rules in the bash
    # MCP server. The text UI ignores the default settings file and runs
    # privileged. In the container this flag additionally seals off package
    # installs (see command_policy.installs_sealed). Must be set before MCP
    # servers are spawned.
    if config_data.get('web'):
        os.environ['ONIT_WEB_UI'] = '1'

    # Declare, before the tool servers are spawned and inherit it, whether
    # this run can answer at all about a command the policy will not run on
    # its own. Only the two interactive front ends have a person to ask: the
    # terminal chat and the web UI, both of which implement the prompt. A
    # --loop run has nobody at the other end, so unless something else is
    # answering it keeps refusing outright rather than minting approval
    # tickets nobody will ever answer. Set here rather than inferred inside
    # the servers, because "is anyone watching" is a fact about how OnIt was
    # started and nothing downstream can recover it.
    #
    # Automatic approval is a channel of its own: the answer comes from the
    # switch rather than from a person, so it works in the modes that have no
    # one attached — which is where an unattended run actually needs it.
    _interactive = not config_data.get('loop')

    # Whether the prompts are answered by the flag rather than by a person.
    # Unset means take the default, and the default turns on whose run this
    # is. A terminal chat, a --loop and a one-shot belong to whoever started
    # them: they already hold the shell OnIt is running commands from, so
    # stopping to ask them protects nobody, and the prompt is one more thing
    # to get stuck on. A web deployment answers to people who are not the
    # operator — on a shared host the approval prompt is part of what keeps
    # one session out of another's way — so it keeps asking unless --auto
    # says otherwise in as many words.
    _serves_others = bool(config_data.get('web'))
    _auto = (not _serves_others) if args.auto is None else args.auto
    _auto_by_default = _auto and args.auto is None

    if _interactive or _auto:
        os.environ['ONIT_APPROVAL_CHANNEL'] = '1'
    else:
        os.environ.pop('ONIT_APPROVAL_CHANNEL', None)

    if _auto:
        # Set for the harness, which is what answers the prompts; the tool
        # servers never read it. Approving is still answering a question the
        # policy chose to ask, so this cannot reach anything the policy would
        # have refused outright — see docs/ISOLATION.md, "Command Approvals".
        os.environ['ONIT_AUTO_APPROVE'] = '1'
        if _auto_by_default:
            # Short, because it prints on every ordinary run. Each approval
            # is still reported as it happens, so this only has to say which
            # way the switch is set and how to move it.
            print("Command approvals: approving automatically "
                  "(--no-auto to be asked instead).", file=sys.stderr)
        else:
            _where = ("every session of this deployment"
                      if config_data.get('web') else "this run")
            print(f"Warning: --auto approves command prompts automatically for "
                  f"{_where}. Commands outside the session directory, unlisted "
                  f"executables and unpinned installs will run without asking.",
                  file=sys.stderr)
        if os.environ.get('ONIT_ASK_APPROVAL') == '0':
            print("Warning: ONIT_ASK_APPROVAL=0 is also set, which refuses "
                  "those commands instead of asking. Nothing is left for "
                  "--auto to approve.", file=sys.stderr)
    else:
        os.environ.pop('ONIT_AUTO_APPROVE', None)

    _setup_servers(config_data)
    _dispatch_mode(config_data)


if __name__ == "__main__":
    main()
