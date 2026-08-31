''''
Manage and run multiple MCP servers based on a configuration file.

Usage:
    python3 run.py

rowel.atienza@up.edu.ph
2025
'''

import os
import sys
import tempfile
import yaml
import time
import multiprocessing
from multiprocessing import Pool
import logging
import argparse
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


# Where the search for a free port starts. Every OnIt process on a shared
# machine allocates its own ports from here upward, so two users never contend
# for a fixed number.
DEFAULT_PORT_BASE = 18200

# How far above the base to look before giving up.
PORT_SEARCH_LIMIT = 400


def _port_claim_dir() -> str:
    """Where port claims live: one directory shared by every user on the host.

    Deliberately not ``tempfile.gettempdir()``. That honours $TMPDIR, which is
    per-user on macOS and can be set per-user anywhere — and OnIt's own bash
    tool points TMPDIR at the session jail. A per-user claim directory would
    still appear to work while excluding nobody, which is the worst outcome:
    the collision it exists to prevent would come back silently.
    """
    if os.name != 'nt' and os.path.isdir('/tmp'):
        return '/tmp/onit-mcp-ports'
    return os.path.join(tempfile.gettempdir(), 'onit-mcp-ports')


# Ports this process has claimed. A claim is an advisory lock on a file named
# after the port. Holding the descriptor open for the life of the process is
# what keeps the claim; the kernel drops it when the process ends, however it
# ends, so a crash leaves nothing to clean up.
_HELD_CLAIMS: list = []

# Set once the claim directory proves unusable, after which the bind test
# stands alone rather than every port being refused.
_CLAIMS_DISABLED = False


def _ensure_claim_dir() -> str | None:
    """The claim directory, created world-writable, or None if unusable."""
    global _CLAIMS_DISABLED
    if _CLAIMS_DISABLED:
        return None

    path = _port_claim_dir()
    try:
        if not os.path.isdir(path):
            # Sticky and world-writable, like /tmp itself: any user may add
            # their own claim, nobody may remove somebody else's.
            os.makedirs(path, exist_ok=True)
            os.chmod(path, 0o1777)
        return path
    except OSError as exc:
        logger.warning("Port claims disabled (%s unusable: %s); relying on the "
                       "bind test alone", path, exc)
        _CLAIMS_DISABLED = True
        return None


def _claim_port(port: int) -> bool:
    """Reserve ``port`` for this process against other OnIt processes.

    Binding a socket to test a port cannot reserve it: the test socket has to
    be closed before the server can bind, and in that gap another OnIt that
    tested the same port takes it. The two processes then run one server
    between them, which is the sharing this whole arrangement exists to stop.
    A lock held across that gap closes it.

    Returns True when the port is ours. Without ``flock``, or without a usable
    claim directory, there is no claim to make and the caller falls back to
    the bind test alone.
    """
    try:
        import fcntl
    except ImportError:
        return True  # Windows: bind test only

    claim_dir = _ensure_claim_dir()
    if claim_dir is None:
        return True

    path = os.path.join(claim_dir, str(port))
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        # The mode above is cut by the umask, typically to 0644, which would
        # leave this file unopenable for writing by anyone else — and since a
        # released claim leaves the file behind, the first user to touch a
        # port would own it forever. Widen it back.
        try:
            os.chmod(path, 0o666)
        except OSError:
            pass  # Not ours to widen; the read-only path below still works.
    except OSError:
        try:
            # flock does not need write access. An older claim file owned by
            # another user is still lockable, so a port they have finished
            # with comes back into circulation.
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return False

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return False

    _HELD_CLAIMS.append(fd)
    return True


def find_free_ports(count: int, base: int | None = None,
                    host: str = '127.0.0.1',
                    limit: int = PORT_SEARCH_LIMIT) -> list[int]:
    if base is None:
        base = DEFAULT_PORT_BASE
    """Return ``count`` ports at or above ``base`` for this process to use.

    A candidate has to pass twice: it must be claimable against other OnIt
    processes, and nothing outside OnIt may already be listening on it. The
    claim is kept for the life of the process, so two people starting OnIt at
    the same moment cannot be handed the same port.
    """
    import socket
    held: list[socket.socket] = []
    ports: list[int] = []
    try:
        for candidate in range(base, base + limit):
            if len(ports) == count:
                break
            if not _claim_port(candidate):
                continue
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind((host, candidate))
            except OSError:
                # Claimed but occupied by something that is not an OnIt MCP
                # server. Keep the claim (harmless) and move on.
                sock.close()
                continue
            held.append(sock)
            ports.append(candidate)
        if len(ports) < count:
            raise RuntimeError(
                f"No free port found for {count - len(ports)} MCP server(s) in "
                f"{base}-{base + limit}")
        return ports
    finally:
        for sock in held:
            sock.close()


def run_server(name:str,
               transport:str,
               host:str,
               port:int,
               path:str,
               module:str,
               options: dict={}) -> bool:
    """
    Start a server with the provided configuration.

    Args:
        name (str): Server name for identification
        transport (str): Transport protocol (e.g., 'sse')
        host (str): Host address to bind the server
        port (int): Port number to listen on
        path (str): URL path for the server endpoint
        module (str): Python module path to import and run
        model (str, optional): Model name if applicable
        model_url (str, optional): URL to download model if applicable

    Returns:
        bool: True if server started successfully, False otherwise
    """
    # Redirect this child process's logging to a file so MCP server output
    # never appears on the onit terminal that spawned the pool — especially
    # important when a second onit instance shares the already-running servers.
    _log_dir = os.path.expanduser('~/.onit/logs')
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.path.join(_log_dir, f'mcp_{name}_{port}.log')
    _log_level = logging.DEBUG if options.get('verbose') else logging.ERROR
    logging.basicConfig(
        filename=_log_file,
        level=_log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )
    # Anything a server writes straight to stderr (the FastMCP startup banner,
    # rich-formatted library logs, third-party prints) has no logger to
    # reroute, so point the file descriptor itself at the log.  Pool workers
    # only: run_server() is also called in-process by the tests, where
    # stealing stderr would be wrong.
    if multiprocessing.parent_process() is not None:
        try:
            _stderr_sink = open(_log_file, 'a', buffering=1)
            os.dup2(_stderr_sink.fileno(), sys.stderr.fileno())
            sys.stderr = _stderr_sink
        except (OSError, AttributeError, ValueError):
            pass  # No usable stderr fd; logging still goes to the file

    try:
        # Suppress noisy uvicorn logs in child processes unless verbose.
        # Override LOGGING_CONFIG *before* uvicorn is imported so that
        # dictConfig() never resets the access logger back to INFO.
        if 'verbose' not in options:
            import uvicorn.config
            uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
            logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
            logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

        if 'stdio' in transport:
            logger.info(f"Starting {name} server using stdio transport")
        else:
            logger.info(f"Starting {name} server at {host}:{port} with path {path} using transport {transport}")

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

        # Run the server
        server_module.run(
            transport=transport, 
            host=host, 
            port=port, 
            path=path, 
            options=options
        )
        
        logger.info(f"Server {name} started successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import module {module} for server {name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error starting {name} server: {e}")
        return False

def load_config(config_path=None):
    """
    Load server configuration from YAML file.

    Args:
        config_path (str, optional): Path to the base configuration file.
            Defaults to the built-in ``configs/default.yaml``.

    Returns:
        dict: Configuration with a ``servers`` list.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")

    logger.info(f"Loading configuration from {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

    return config

def prepare_server_args(config, port_overrides: dict | None = None):
    """Extract server arguments from configuration.

    Args:
        config: Parsed server configuration.
        port_overrides: ``{server name: port}`` chosen by the caller. A name
            present here overrides the port in the config file, which is how
            each OnIt process gets its own set (see ``find_free_ports``).

    Servers declaring ``transport: stdio`` are left out: their process is
    spawned by the MCP client that connects to it, not by this pool, so that
    it runs as the user who started OnIt and dies with them.
    """
    server_args = []
    port_overrides = port_overrides or {}

    for server in config.get('servers', []):
        name = server.get('name')
        if not name:
            logger.warning("Skipping server with no name defined")
            continue
            
        transport = server.get('transport', 'sse')
        # Loopback, not 0.0.0.0: these servers have no authentication of their
        # own, so they must not be reachable from off the machine.
        host = server.get('host', '127.0.0.1')
        port = port_overrides.get(name, server.get('port', 18201))
        path = server.get('path', f'/{name.lower()}')
        enabled = server.get('enabled', True)
        module = server.get('module')
        
        if not module:
            logger.warning(f"Skipping {name} server: No module specified")
            continue

        if 'stdio' in transport:
            logger.info(f"Skipping {name}: stdio servers are spawned by the client")
            continue

        options = {}
        if 'options' in server:
            options = server.get('options', {})
            #model = kwargs.get('model')
            #model_url = kwargs.get('model_url')
            
        if enabled:
            server_args.append([name, transport, host, port, path, module, options])
            logger.info(f"Preparing to start {name} server at {host}:{port} with path {path}")
        else:
            logger.info(f"Skipping {name} server as it is disabled in the configuration.")

    # A server the caller did not name still needs a port of its own — the VLM
    # tools server is in this config but not in the client's list, and on a
    # shared machine its fixed port collides just as readily as the rest.
    unassigned = [args for args in server_args if args[0] not in port_overrides]
    if unassigned:
        for args, port in zip(unassigned, find_free_ports(len(unassigned))):
            args[3] = port
            logger.info(f"Assigned free port {port} to {args[0]}")

    return [tuple(args) for args in server_args]

def run_servers(config_path=None, log_level='INFO', port_overrides: dict | None = None):
    """Run MCP servers based on a configuration file.

    Args:
        config_path (str, optional): Path to YAML config. Defaults to built-in default.
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        port_overrides (dict, optional): ``{server name: port}`` to start on
            instead of the configured ports.
    """
    logging.getLogger().setLevel(getattr(logging, log_level))
    # Suppress noisy uvicorn access logs unless in DEBUG mode
    if log_level != 'DEBUG':
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    try:
        config = load_config(config_path)
        # Propagate verbose flag into each server's options so child processes
        # (which get fresh logging configs) can suppress uvicorn access logs.
        if log_level == 'DEBUG':
            for server in config.get('servers', []):
                server.setdefault('options', {})['verbose'] = True
        server_args = prepare_server_args(config, port_overrides)

        if not server_args:
            logger.warning("No enabled servers found in configuration")
            return

        # Use 'spawn' on Linux to avoid fork-from-thread deadlocks
        # when run_servers() is called from a daemon thread.
        ctx = multiprocessing.get_context('spawn') if sys.platform != 'darwin' else multiprocessing
        with ctx.Pool(processes=len(server_args)) as pool:
            results = [pool.apply_async(run_server, args) for args in server_args]

            logger.info(f"Starting {len(server_args)} servers...")

            try:
                while True:
                    for i, result in enumerate(results):
                        if result.ready() and not result.get():
                            name = server_args[i][0]
                            logger.error(f"Server {name} failed to start or crashed")
                    time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Received KeyboardInterrupt, terminating all servers...")

    except Exception as e:
        logger.error(f"Failed to start servers: {e}")
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MCP Server Manager")
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    # Serving one server on stdio, in the foreground, is how an MCP client
    # starts a server it owns: the client spawns this process and speaks the
    # protocol over its stdin/stdout. Nothing is written to stdout by anything
    # except the protocol itself.
    parser.add_argument('--stdio', action='store_true',
                        help='Serve a single server over stdio instead of '
                             'starting the configured pool')
    parser.add_argument('--name', default='StdioMCPServer',
                        help='Server name, used for log file naming (--stdio)')
    parser.add_argument('--module', help='Module to serve (--stdio)')
    parser.add_argument('--profile', help='Tool profile to serve, for servers '
                                          'that support one (--stdio)')
    args = parser.parse_args()

    if args.stdio:
        if not args.module:
            parser.error('--stdio requires --module')
        stdio_options = {}
        if args.profile:
            stdio_options['profile'] = args.profile
        if args.log_level == 'DEBUG':
            stdio_options['verbose'] = True
        ok = run_server(name=args.name, transport='stdio', host='', port=0,
                        path='', module=args.module, options=stdio_options)
        sys.exit(0 if ok else 1)

    run_servers(config_path=args.config, log_level=args.log_level)
