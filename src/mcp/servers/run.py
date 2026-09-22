'''
Run one MCP server over stdio, in the foreground.

An MCP client that wants a server it owns spawns this process and speaks the
protocol over stdin/stdout. Nothing is written to stdout by anything except
the protocol itself: logs go to a file under ~/.onit/logs, and stderr is
pointed there too.

Usage (the launch spec the MCP client builds, see lib/tools.py):
    python3 -m src.mcp.servers.run --stdio --name NAME --module MODULE

rowel.atienza@up.edu.ph
2025
'''

import os
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def _demote_src_on_syspath() -> None:
    """Move any sys.path entry pointing at this repo's src/ to the tail.

    The stdio server is spawned by the MCP client, which inherits the parent's
    sys.path. When the parent is the benchmark runner,
    ``benchmarks/onit_provider.py`` has put "src" at position 0 so its absolute
    ``src.*`` imports resolve — and in the child that makes the local
    ``src/mcp`` package shadow the PyPI ``mcp`` SDK: fastmcp's lazy
    ``import mcp.types`` then raises ModuleNotFoundError and the server
    crash-loops (observed 2026-09-05: Prompts/ToolsNet/VLMTools exited code 1
    every 10 s for the whole run). The ``55b7d0e`` pre-import fix pins the SDK
    in the parent's sys.modules only; a spawned child starts with a fresh module
    table but the same poisoned path, so the fix must live here, where every
    child executes it.

    Demoting rather than deleting keeps the bare-import fallbacks working
    (``src/lib/tools.py`` and ``src/mcp/prompts/prompts.py`` resolve ``lib.*`` /
    ``type.*`` / ``mcp.*`` without a package prefix when run outside the
    package) — they only need the entry present, not first. In a parent that
    imports this module the demotion is harmless: package imports are relative
    and anything needing the bare names appends the entry itself.
    """
    src_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    demoted = [p for p in sys.path if p and os.path.realpath(p) == src_dir]
    for p in demoted:
        sys.path.remove(p)
    sys.path.extend(demoted)


_demote_src_on_syspath()

# Pin the real SDK modules before any server module is imported. After the
# demotion a bare ``import mcp`` resolves to the PyPI SDK in a poisoned child;
# pinning here also covers parents where some other path entry still shadows.
try:
    import mcp.types  # noqa: F401
    import fastmcp.server  # noqa: F401
except ImportError:  # pragma: no cover - SDK extras genuinely absent
    pass  # servers that need them will fail with their own honest error


def run_server(name: str,
               module: str,
               options: dict = {}) -> bool:
    """
    Serve one MCP server over stdio until the client closes the pipe.

    Args:
        name: Server name, used for the log file name.
        module: Python module path to import and run.
        options: Passed through to the module's ``run()`` (e.g. ``profile``).

    Returns:
        bool: True if the server ran, False otherwise.
    """
    # Redirect this child process's logging to a file so MCP server output
    # never appears on the onit terminal that spawned it.
    _log_dir = os.path.expanduser('~/.onit/logs')
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.path.join(_log_dir, f'mcp_{name}.log')
    _log_level = logging.DEBUG if options.get('verbose') else logging.ERROR
    logging.basicConfig(
        filename=_log_file,
        level=_log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )
    # Anything a server writes straight to stderr (the FastMCP startup banner,
    # rich-formatted library logs, third-party prints) has no logger to
    # reroute, so point the file descriptor itself at the log. Only when
    # spawned: run_server() is also called in-process by the tests, where
    # stealing stderr would be wrong.
    if os.environ.get('ONIT_STDIO_SPAWNED') == '1':
        try:
            _stderr_sink = open(_log_file, 'a', buffering=1)
            os.dup2(_stderr_sink.fileno(), sys.stderr.fileno())
            sys.stderr = _stderr_sink
        except (OSError, AttributeError, ValueError):
            pass  # No usable stderr fd; logging still goes to the file

    try:
        if not module:
            logger.error(f"No module specified for server {name}")
            return False

        # Import the server module dynamically
        # Built-in shorthand: names starting with "tasks." or "src." are
        # resolved relative to the onit package.  Everything else is treated
        # as an absolute Python import path so that pip-installed third-party
        # packages work out of the box.
        if module.startswith("src."):
            full_module = module
        elif module.startswith("tasks."):
            full_module = f"src.mcp.servers.{module}"
        else:
            full_module = module  # third-party absolute module path
        server_module = __import__(full_module, fromlist=['run'])

        # basicConfig() above only owns the root logger.  fastmcp installs its
        # own stderr RichHandler when it is imported (just now, under 'spawn')
        # and sets propagate=False, so its records — one per ctx.log() call,
        # i.e. every streamed line of bash stdout, relabelled DEBUG — would
        # bypass the file handler.  Re-point them at it, at the level asked for
        # here rather than fastmcp's own INFO default.
        _fastmcp_logger = logging.getLogger('fastmcp')
        for _hdlr in _fastmcp_logger.handlers[:]:
            _fastmcp_logger.removeHandler(_hdlr)
        _fastmcp_logger.propagate = True
        _fastmcp_logger.setLevel(_log_level)

        # Run the server. stdout carries the protocol here: no socket, no
        # banner, no uvicorn.
        logger.info(f"Starting {name} server using stdio transport")
        server_module.run(transport='stdio', options=options)

        logger.info(f"Server {name} started successfully")
        return True

    except ImportError as e:
        logger.error(f"Failed to import module {module} for server {name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error starting {name} server: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP stdio server")
    parser.add_argument('--log-level', default='ERROR',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    # Serving one server on stdio, in the foreground, is how an MCP client
    # starts a server it owns: the client spawns this process and speaks the
    # protocol over its stdin/stdout. Nothing is written to stdout by anything
    # except the protocol itself.
    parser.add_argument('--stdio', action='store_true',
                        help='Serve a single server over stdio')
    parser.add_argument('--name', default='StdioMCPServer',
                        help='Server name, used for log file naming')
    parser.add_argument('--module', help='Module to serve')
    parser.add_argument('--profile', help='Tool profile to serve, for servers '
                                          'that support one')
    args = parser.parse_args()

    if not args.stdio:
        parser.error('only --stdio serving is supported; the server pool was '
                     'removed — every built-in server runs over stdio now')
    if not args.module:
        parser.error('--stdio requires --module')
    stdio_options = {}
    if args.profile:
        stdio_options['profile'] = args.profile
    if args.log_level == 'DEBUG':
        stdio_options['verbose'] = True
    os.environ['ONIT_STDIO_SPAWNED'] = '1'
    ok = run_server(name=args.name, module=args.module, options=stdio_options)
    sys.exit(0 if ok else 1)