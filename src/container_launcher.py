"""
Container launcher for ``onit --container``.

Runs the entire OnIt process inside a Docker container so the agent's
filesystem / apt / pip operations stay off the host. The launcher:

  * Verifies Docker is available.
  * Builds the ``onit:local`` image on first run.
  * Invokes ``docker run`` with resource limits tuned for ML workloads
    (generous /tmp + /dev/shm, high pids-limit, unlimited memory by default)
    and no host mounts beyond the user's config/secrets (read-only), a named
    data volume, and any ``--container-mount`` bind mounts.
  * Bridges host secrets (OS keychain / ``~/.onit/secrets.yaml``) into the
    container as ephemeral env vars, never baked into the image.
  * Proxies signals so Ctrl-C tears the container down cleanly.

The container keeps namespace / filesystem isolation from the host but does
not drop capabilities or mount the rootfs read-only — the agent can run
``sudo apt install`` and build native extensions. For per-tool sandboxing
(stricter isolation with no sudo), use ``--sandbox`` instead.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

# Secret keys bridged from host keychain/secrets.yaml into the container env.
# Kept in sync with src/setup.py SECRETS entries that have an env_var name.
_SECRET_ENV_KEYS = [
    ("host_key", "OPENROUTER_API_KEY"),
    ("ollama_api_key", "OLLAMA_API_KEY"),
    ("openweathermap_api_key", "OPENWEATHERMAP_API_KEY"),
    ("telegram_bot_token", "TELEGRAM_BOT_TOKEN"),
    ("viber_bot_token", "VIBER_BOT_TOKEN"),
    ("github_token", "GITHUB_TOKEN"),
    ("huggingface_token", "HF_TOKEN"),
]

IMAGE_TAG = "onit:local"
DATA_VOLUME = "onit-data"
# Default resource caps tuned for long-running ML experiments. Memory is left
# unset (host's default = unlimited) so training jobs aren't OOM-killed at 2g.
# Users can clamp with --container-memory. pids-limit is high enough for
# DataLoader workers + torch.distributed + subprocess tools.
DEFAULT_MEMORY: str | None = None
DEFAULT_PIDS_LIMIT = "4096"
# /dev/shm for PyTorch DataLoader IPC — Docker's 64m default causes "Bus error"
# with num_workers>0. 4g covers typical training workloads.
DEFAULT_SHM_SIZE = "4g"
# /tmp tmpfs — wheel extraction, scratch files, dataset shuffling. Backed by
# host RAM, so keep reasonable but bigger than the old 4g.
DEFAULT_TMP_SIZE = "16g"

# CUDA runtime version baked into the image (see Dockerfile FROM line). Used to
# warn the user when the host driver's supported CUDA version is older — host
# driver CUDA must be >= image CUDA for GPU ops to work.
_IMAGE_CUDA = (12, 6)


def _check_docker() -> str:
    """Return the docker binary path, or exit with a helpful message."""
    docker = shutil.which("docker")
    if not docker:
        sys.stderr.write(
            "Error: 'docker' not found on PATH.\n"
            "Install Docker Desktop (https://docs.docker.com/get-docker/) "
            "and retry with `onit --container`.\n"
        )
        sys.exit(1)
    try:
        subprocess.run(
            [docker, "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.stderr.write(
            "Error: Docker daemon is not reachable. Start Docker Desktop "
            "(or the docker service) and retry.\n"
        )
        sys.exit(1)
    return docker


def _repo_root() -> Path:
    """Return the directory containing the Dockerfile we ship with."""
    return Path(__file__).resolve().parent.parent


def _image_exists(docker: str) -> bool:
    result = subprocess.run(
        [docker, "image", "inspect", IMAGE_TAG],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _ensure_data_volume_writable(docker: str) -> None:
    """Make the named data volume writable by any UID.

    The Dockerfile chmods ``/home/onit/data`` to 0777 so a freshly-created
    volume inherits those perms. But a volume that predates that change keeps
    its original 0755 + onit:onit ownership — and when the launcher later
    runs the container as the host user's UID/GID (to match bind mounts), the
    container can't write to that volume. Fix it idempotently by chmodding
    from a one-shot root container; no-op if the volume doesn't exist yet.
    """
    inspect = subprocess.run(
        [docker, "volume", "inspect", DATA_VOLUME],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if inspect.returncode != 0:
        return  # volume will be created with 0777 from the image
    # -R because the volume already contains subdirs like .pip-cache and
    # .huggingface created (on first use) as onit:onit 0755 — those would
    # still reject writes from other UIDs even after we fix the top level.
    subprocess.run(
        [docker, "run", "--rm", "-u", "0:0", "--entrypoint", "chmod",
         "-v", f"{DATA_VOLUME}:/v", IMAGE_TAG, "-R", "0777", "/v"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _build_image(docker: str) -> None:
    root = _repo_root()
    if not (root / "Dockerfile").is_file():
        sys.stderr.write(
            f"Error: Dockerfile not found in {root}. The --container flag "
            "only works from a source checkout of onit.\n"
        )
        sys.exit(1)
    print(f"Building {IMAGE_TAG} (first run, may take a few minutes)...",
          file=sys.stderr)
    subprocess.run(
        [docker, "build", "-t", IMAGE_TAG, str(root)],
        check=True,
    )


def _collect_secret_env() -> list[str]:
    """Read host secrets and return ``-e KEY=VALUE`` args (ephemeral)."""
    try:
        from .setup import get_secret  # type: ignore[no-redef]
    except Exception:
        get_secret = lambda _key: None  # noqa: E731

    env_args: list[str] = []
    for keyring_key, env_name in _SECRET_ENV_KEYS:
        # CLI > env > keychain. Env takes precedence if set on the host.
        value = os.environ.get(env_name) or get_secret(keyring_key)
        if value:
            env_args.extend(["-e", f"{env_name}={value}"])
    return env_args


def _config_mount_args() -> list[str]:
    """Mount ``~/.onit/config.yaml`` and ``secrets.yaml`` read-only if present."""
    mounts: list[str] = []
    onit_dir = Path(os.path.expanduser("~/.onit"))
    cfg = onit_dir / "config.yaml"
    if cfg.is_file():
        mounts.extend(["-v", f"{cfg}:/home/onit/.onit/config.yaml:ro"])
    secrets = onit_dir / "secrets.yaml"
    if secrets.is_file():
        mounts.extend(["-v", f"{secrets}:/home/onit/.onit/secrets.yaml:ro"])
    return mounts


def _host_user_args(bind_mount_count: int) -> list[str]:
    """Return ``--user HOST_UID:HOST_GID`` when bind mounts are in play.

    Bind-mounted host paths keep their host ownership inside the container.
    The Dockerfile's baked ``onit`` user (UID/GID 1000) doesn't match most
    hosts, so without this the in-container process can't read or write the
    mount. We only apply this on Linux; Docker Desktop (macOS/Windows) runs
    inside a VM that handles UID translation automatically.
    """
    if bind_mount_count <= 0:
        return []
    if sys.platform != "linux":
        return []
    try:
        uid, gid = os.getuid(), os.getgid()
    except AttributeError:
        return []
    return ["--user", f"{uid}:{gid}"]


def _runtime_args(
    *, memory: str | None = None, shm_size: str | None = None,
    tmp_size: str | None = None,
) -> list[str]:
    """Runtime flags for the container.

    We keep namespace/filesystem isolation (separate rootfs, pid/net
    namespaces) but relax the defense-in-depth flags (--read-only, cap-drop,
    no-new-privileges) so the agent can run `sudo apt install` and ordinary
    build tooling without fighting the sandbox. For stricter isolation use
    ``--sandbox`` (tool-level MCP sandbox) instead.
    """
    args: list[str] = []
    # Session transcripts tmpfs — small, still useful since sessions are
    # ephemeral within a container run and persistent ones live on the volume.
    args.extend(["--tmpfs", "/home/onit/.onit/sessions:rw,size=64m"])
    # /tmp sized generously for ML scratch (wheel extraction, dataset shards).
    args.extend(["--tmpfs", f"/tmp:rw,size={tmp_size or DEFAULT_TMP_SIZE}"])
    # /dev/shm for DataLoader workers — default 64m breaks multiprocess IPC.
    args.extend(["--shm-size", shm_size or DEFAULT_SHM_SIZE])
    args.extend(["--pids-limit", DEFAULT_PIDS_LIMIT])
    if memory or DEFAULT_MEMORY:
        args.extend(["--memory", memory or DEFAULT_MEMORY])
    return args


def _detect_gpu_support(docker: str) -> tuple[bool, str | None]:
    """Check if ``docker run --gpus all`` is viable on this host.

    Returns ``(available, warning)``. ``available`` is True only when both
    ``nvidia-smi`` resolves and Docker has the nvidia runtime registered.
    ``warning`` is set when the host driver's supported CUDA is older than the
    image's CUDA runtime — GPU ops may fail even if --gpus attaches.
    """
    if not shutil.which("nvidia-smi"):
        return (False, None)
    try:
        smi = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=5, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return (False, None)

    # Docker needs the nvidia runtime registered (nvidia-container-toolkit).
    try:
        info = subprocess.run(
            [docker, "info"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=5, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return (False, None)
    if "nvidia" not in info.lower():
        return (False, None)

    # Parse "CUDA Version: X.Y" from the nvidia-smi header — this is the max
    # CUDA the installed driver supports.
    m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", smi)
    if m:
        host_cuda = (int(m.group(1)), int(m.group(2)))
        if host_cuda < _IMAGE_CUDA:
            warn = (
                f"host NVIDIA driver supports CUDA {host_cuda[0]}.{host_cuda[1]} "
                f"but the container image uses CUDA "
                f"{_IMAGE_CUDA[0]}.{_IMAGE_CUDA[1]} — GPU ops may fail with "
                f"'CUDA driver version is insufficient'. Update the host driver."
            )
            return (True, warn)
    return (True, None)


def _extract_flag_value(args: list[str], flag: str) -> str | None:
    """Return the string value following ``flag`` in ``args``, or ``None``.

    Accepts both ``--flag VALUE`` and ``--flag=VALUE`` forms.
    """
    prefix = f"{flag}="
    for i, tok in enumerate(args):
        if tok == flag and i + 1 < len(args):
            return args[i + 1]
        if tok.startswith(prefix):
            return tok.split("=", 1)[1]
    return None


def _auto_path_mounts(forwarded_args: list[str]) -> list[str]:
    """Bind-mount host paths referenced by forwarded path flags.

    The container has a read-only rootfs, so a ``--data-path /home/user/tts``
    passed through to the in-container onit would fail to create its target
    directory. Detect absolute host paths on these flags and mount them at
    the same path inside the container so the in-container process can use
    them unchanged.
    """
    mounts: list[str] = []
    # data_path is written to (session state, tool output) — rw.
    data_path = _extract_flag_value(forwarded_args, "--data-path")
    if data_path:
        expanded = os.path.abspath(os.path.expanduser(data_path))
        if expanded.startswith("/"):
            os.makedirs(expanded, exist_ok=True)
            mounts.extend(["-v", f"{expanded}:{expanded}:rw"])
    # documents_path is read-only (search corpus).
    docs_path = _extract_flag_value(forwarded_args, "--documents-path")
    if docs_path:
        expanded = os.path.abspath(os.path.expanduser(docs_path))
        if expanded.startswith("/"):
            mounts.extend(["-v", f"{expanded}:{expanded}:ro"])
    return mounts


def _extract_port(args: list[str], flag: str, default: int) -> int:
    """Return the integer value following ``flag`` in ``args``, or ``default``.

    Accepts both ``--flag 9500`` and ``--flag=9500`` forms.
    """
    prefix = f"{flag}="
    for i, tok in enumerate(args):
        if tok == flag and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                return default
        if tok.startswith(prefix):
            try:
                return int(tok.split("=", 1)[1])
            except ValueError:
                return default
    return default


def _port_args(forwarded_args: list[str]) -> list[str]:
    """Map only the ports needed by the selected mode, honoring user overrides."""
    ports: list[str] = []
    if "--web" in forwarded_args:
        p = _extract_port(forwarded_args, "--web-port", 9000)
        ports.extend(["-p", f"{p}:{p}"])
    if "--a2a" in forwarded_args:
        p = _extract_port(forwarded_args, "--a2a-port", 9001)
        ports.extend(["-p", f"{p}:{p}"])
    if "--gateway" in forwarded_args or any(
        t == "--viber-port" or t.startswith("--viber-port=") for t in forwarded_args
    ):
        p = _extract_port(forwarded_args, "--viber-port", 8443)
        ports.extend(["-p", f"{p}:{p}"])
    return ports


def _is_server_mode(forwarded_args: list[str]) -> bool:
    return any(flag in forwarded_args for flag in ("--web", "--a2a", "--gateway"))


def _run_docker(
    docker: str,
    forwarded_args: list[str],
    *,
    gpus: str | None = None,
    mounts: list[str] | None = None,
    memory: str | None = None,
    shm_size: str | None = None,
    tmp_size: str | None = None,
) -> int:
    """Run a single containerised onit process via ``docker run``."""
    cmd = [docker, "run", "--rm"]

    # Terminal mode needs an interactive TTY; server mode just attaches.
    if not _is_server_mode(forwarded_args) and sys.stdin.isatty():
        cmd.extend(["-it"])
    else:
        cmd.append("-i")

    cmd.extend(_runtime_args(memory=memory, shm_size=shm_size, tmp_size=tmp_size))
    if gpus:
        available, warn = _detect_gpu_support(docker)
        if not available:
            print(
                "Note: --container-gpus requested but no usable GPU/"
                "nvidia-container-toolkit found on this host — starting "
                "without GPU attachment (torch will run on CPU).",
                file=sys.stderr,
            )
        else:
            if warn:
                print(f"Warning: {warn}", file=sys.stderr)
            cmd.extend(["--gpus", gpus])
    cmd.extend(_port_args(forwarded_args))
    cmd.extend(["-v", f"{DATA_VOLUME}:/home/onit/data"])
    # Anchor the agent's cwd in the persistent writable volume so shell tools
    # that write relative paths don't hit the read-only rootfs.
    cmd.extend(["--workdir", "/home/onit/data"])
    cmd.extend(_config_mount_args())
    auto_mounts = _auto_path_mounts(forwarded_args)
    cmd.extend(auto_mounts)
    for m in mounts or []:
        cmd.extend(["-v", m])
    # Match host UID/GID so bind-mounted directories are writable from inside.
    total_bind_mounts = (len(auto_mounts) // 2) + len(mounts or [])
    cmd.extend(_host_user_args(total_bind_mounts))
    cmd.extend(_collect_secret_env())
    cmd.append(IMAGE_TAG)
    cmd.extend(forwarded_args)

    # Forward signals to the docker client so the container tears down cleanly.
    proc = subprocess.Popen(cmd)

    def _forward(signum, _frame):
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    old_int = signal.signal(signal.SIGINT, _forward)
    old_term = signal.signal(signal.SIGTERM, _forward)
    try:
        return proc.wait()
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def build_run_command(
    docker: str,
    forwarded_args: list[str],
    *,
    config_mounts: list[str] | None = None,
    secret_env: list[str] | None = None,
    gpus: str | None = None,
    mounts: list[str] | None = None,
    memory: str | None = None,
    shm_size: str | None = None,
    tmp_size: str | None = None,
) -> list[str]:
    """Assemble the ``docker run`` argv. Extracted for unit testing."""
    cmd = [docker, "run", "--rm"]
    if not _is_server_mode(forwarded_args):
        cmd.append("-it")
    else:
        cmd.append("-i")
    cmd.extend(_runtime_args(memory=memory, shm_size=shm_size, tmp_size=tmp_size))
    if gpus:
        cmd.extend(["--gpus", gpus])
    cmd.extend(_port_args(forwarded_args))
    cmd.extend(["-v", f"{DATA_VOLUME}:/home/onit/data"])
    cmd.extend(["--workdir", "/home/onit/data"])
    cmd.extend(config_mounts if config_mounts is not None else _config_mount_args())
    auto_mounts = _auto_path_mounts(forwarded_args)
    cmd.extend(auto_mounts)
    for m in mounts or []:
        cmd.extend(["-v", m])
    total_bind_mounts = (len(auto_mounts) // 2) + len(mounts or [])
    cmd.extend(_host_user_args(total_bind_mounts))
    cmd.extend(secret_env if secret_env is not None else _collect_secret_env())
    cmd.append(IMAGE_TAG)
    cmd.extend(forwarded_args)
    return cmd


def run(
    forwarded_args: list[str],
    *,
    gpus: str | None = None,
    mounts: list[str] | None = None,
    memory: str | None = None,
    shm_size: str | None = None,
    tmp_size: str | None = None,
) -> int:
    """Entry point called from ``cli.main`` when ``--container`` is passed.

    ``forwarded_args`` are the remaining argv tokens (launcher-only flags
    already stripped) that should be handed to the in-container ``onit``
    entrypoint. ``gpus`` and ``mounts`` are launcher-only pass-throughs.
    ``memory``/``shm_size``/``tmp_size`` override the resource defaults.
    """
    docker = _check_docker()
    if not _image_exists(docker):
        _build_image(docker)
    _ensure_data_volume_writable(docker)
    return _run_docker(
        docker, forwarded_args,
        gpus=gpus, mounts=mounts,
        memory=memory, shm_size=shm_size, tmp_size=tmp_size,
    )


# Launcher-only flags consumed by the launcher, NOT forwarded into the container.
_LAUNCHER_VALUE_FLAGS = frozenset({
    "--container-gpus", "--container-mount",
    "--container-memory", "--container-shm-size", "--container-tmp-size",
})


def strip_launcher_args(argv: list[str]) -> list[str]:
    """Remove ``--container`` and its launcher-only siblings (and their values)."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--container":
            i += 1
            continue
        if tok in _LAUNCHER_VALUE_FLAGS:
            i += 2  # skip flag and its value
            continue
        if any(tok.startswith(f + "=") for f in _LAUNCHER_VALUE_FLAGS):
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


# Backward-compatible alias.
def strip_container_flag(argv: list[str]) -> list[str]:
    return strip_launcher_args(argv)
