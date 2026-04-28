'''
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Bash MCP Server - System and Document Operations

Execute shell commands, read/write files, and search documents on the local system.

Requires:
    pip install fastmcp pypdf pdfplumber

12 Core Tools:
1. bash - Execute bash/shell commands with timeout and directory control
2. read_file - Read files (text, PDF returns content; binary files return metadata only)
3. write_file - Write content to files (supports write/append modes)
4. edit_file - Edit a file by replacing an exact string (targeted in-place edit)
5. serve - Launch/stop/monitor background processes (web servers, daemons)
6. send_file - Send a file to a remote client via callback URL
7. search_document - Search for patterns in a document (text, PDF, markdown)
8. search_directory - Search for patterns across files in a directory
9. extract_tables - Extract tables from documents (PDF, markdown)
10. find_files - Find files matching patterns
11. transform_text - Transform text using sed/awk/tr operations
12. get_document_context - Extract relevant context from a document
'''

import asyncio
import base64
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional, Dict, Any
from pydantic import Field
from fastmcp import FastMCP, Context

from src.mcp.servers.tasks.shared import (
    truncate_output as _truncate_output,
    secure_makedirs as _secure_makedirs,
    validate_required as _validate_required,
    search_document_impl,
    search_directory_impl,
    extract_tables_impl,
    find_files_impl,
    transform_text_impl,
    get_document_context_impl,
)

import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"

# Locate a usable bash executable on Windows (Git Bash, WSL, etc.)
_BASH_EXE = None
if IS_WINDOWS:
    _BASH_EXE = shutil.which("bash")
    if not _BASH_EXE:
        # Common Git Bash locations
        for candidate in (
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Windows\System32\bash.exe",
        ):
            if os.path.isfile(candidate):
                _BASH_EXE = candidate
                break

mcp = FastMCP("Bash MCP Server")

# Constants
DEFAULT_TIMEOUT = 300
MAX_OUTPUT_SIZE = 100000  # 100KB max output

# Data path for file creation (set via options['data_path'] in run())
# All file writes are confined to this directory. Never use home folder.
DATA_PATH = os.path.join(tempfile.gettempdir(), "onit", "data")

# Optional read-only documents directory (set via options or env var in run())
DOCUMENTS_PATH = None

# Cached sandbox environment for subprocess calls (reset when globals change)
_SANDBOX_ENV = None

# Binary file extensions (return description only, not content)
BINARY_EXTENSIONS = {
    # Images
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.ico', '.tiff', '.tif', '.svg',
    # Audio
    '.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma',
    # Video
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm',
    # Archives
    '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2',
    # Executables/binaries
    '.exe', '.dll', '.so', '.dylib', '.bin', '.o', '.a',
    # Documents (binary)
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    # Other
    '.pyc', '.class', '.wasm', '.ttf', '.otf', '.woff', '.woff2'
}


def _in_container() -> bool:
    """True when running inside the onit container (Dockerfile sets ONIT_CONTAINER=1).

    In that case the container itself is the filesystem boundary, so MCP path
    allowlists are relaxed — the agent can read/write any path the container
    can see (including directories mapped via ``--container-mount``).
    """
    return os.environ.get("ONIT_CONTAINER") == "1"


def _in_unrestricted() -> bool:
    """True when the user passed ``--unrestricted`` (sets ONIT_UNRESTRICTED=1)."""
    return os.environ.get("ONIT_UNRESTRICTED") == "1"


def _path_restrictions_relaxed() -> bool:
    """True when filesystem path allowlisting should be skipped.

    Container mode: the container itself is the isolation boundary.
    Unrestricted mode: user explicitly opted into full host access.
    """
    return _in_container() or _in_unrestricted()


def _validate_write_path(file_path: str) -> str:
    """Validate that the write path is within DATA_PATH. Returns absolute path.
    Relative paths are resolved against DATA_PATH (not CWD)."""
    if not os.path.isabs(os.path.expanduser(file_path)):
        file_path = os.path.join(DATA_PATH, file_path)
    abs_path = os.path.realpath(os.path.expanduser(file_path))
    if _path_restrictions_relaxed():
        return abs_path
    abs_data = os.path.realpath(os.path.expanduser(DATA_PATH))
    if not abs_path.startswith(abs_data + os.sep) and abs_path != abs_data:
        raise ValueError(
            f"Write path must be within the designated data directory: {abs_data}. "
            f"Got: {abs_path}"
        )
    return abs_path


def _validate_read_path(file_path: str) -> str:
    """Validate that the read path is within DATA_PATH or DOCUMENTS_PATH.
    Relative paths are resolved against DATA_PATH (not CWD).
    Returns the resolved absolute path. Raises ValueError if outside allowed directories."""
    if not os.path.isabs(os.path.expanduser(file_path)):
        file_path = os.path.join(DATA_PATH, file_path)
    abs_path = os.path.realpath(os.path.expanduser(file_path))
    if _path_restrictions_relaxed():
        return abs_path
    abs_data = os.path.realpath(os.path.expanduser(DATA_PATH))

    if abs_path.startswith(abs_data + os.sep) or abs_path == abs_data:
        return abs_path

    if DOCUMENTS_PATH:
        abs_docs = os.path.realpath(os.path.expanduser(DOCUMENTS_PATH))
        if abs_path.startswith(abs_docs + os.sep) or abs_path == abs_docs:
            return abs_path

    allowed = abs_data
    if DOCUMENTS_PATH:
        allowed += f" or {os.path.realpath(os.path.expanduser(DOCUMENTS_PATH))}"
    raise ValueError(
        f"Read access denied. Path must be within: {allowed}. Got: {abs_path}"
    )


def _validate_dir_path(dir_path: str) -> str:
    """Validate a directory path is within allowed directories.
    Relative paths are resolved against DATA_PATH (not CWD).
    Returns resolved absolute path."""
    if not os.path.isabs(os.path.expanduser(dir_path)):
        dir_path = os.path.join(DATA_PATH, dir_path)
    abs_path = os.path.realpath(os.path.expanduser(dir_path))
    if _path_restrictions_relaxed():
        return abs_path
    abs_data = os.path.realpath(os.path.expanduser(DATA_PATH))

    if abs_path.startswith(abs_data + os.sep) or abs_path == abs_data:
        return abs_path

    if DOCUMENTS_PATH:
        abs_docs = os.path.realpath(os.path.expanduser(DOCUMENTS_PATH))
        if abs_path.startswith(abs_docs + os.sep) or abs_path == abs_docs:
            return abs_path

    allowed = abs_data
    if DOCUMENTS_PATH:
        allowed += f" or {os.path.realpath(os.path.expanduser(DOCUMENTS_PATH))}"
    raise ValueError(
        f"Directory access denied. Path must be within: {allowed}. Got: {abs_path}"
    )


def _get_sandbox_env() -> dict:
    """Build a minimal environment dict for sandboxed shell execution.
    Strips most inherited env vars but preserves HOME and credential-related
    variables so that keyring/keychain tools and credential helpers work.

    In container mode (``ONIT_CONTAINER=1``, set by the Dockerfile) the full
    container env is forwarded — the Dockerfile curates that env (PIP_TARGET,
    HF_HOME, MPLCONFIGDIR, XDG_*, …) and the container itself is the boundary,
    so the allowlist-style stripping that makes sense on the host is just an
    obstacle here.
    """
    global _SANDBOX_ENV
    if _SANDBOX_ENV is not None:
        return dict(_SANDBOX_ENV)

    abs_data = os.path.realpath(os.path.expanduser(DATA_PATH))
    tmp_dir = os.path.join(abs_data, "tmp")
    _secure_makedirs(tmp_dir)

    # Use the real HOME so credential helpers (osxkeychain, gh, git config) work.
    real_home = os.path.expanduser("~")

    if _path_restrictions_relaxed():
        env = dict(os.environ)
        env.update({
            "TERM": "dumb",
            "LANG": env.get("LANG", "en_US.UTF-8"),
            "LC_ALL": env.get("LC_ALL", "en_US.UTF-8"),
            "HOME": real_home,
            "TMPDIR": tmp_dir,
            "DATA_PATH": abs_data,
        })
        target_env_bin = os.environ.get("ONIT_TARGET_ENV_BIN")
        if target_env_bin and os.path.isdir(target_env_bin):
            env["PATH"] = target_env_bin + os.pathsep + env.get("PATH", "")
        _SANDBOX_ENV = env
        return dict(env)

    env = {
        "TERM": "dumb",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "HOME": real_home,
        "TMPDIR": tmp_dir,
        "DATA_PATH": abs_data,
    }

    # Pass through credential-related env vars if set in the parent environment.
    for var in ("GITHUB_TOKEN", "GH_TOKEN", "GH_CONFIG_DIR",
                "SSH_AUTH_SOCK", "SSH_AGENT_PID",
                "GIT_ASKPASS", "GIT_CREDENTIAL_HELPER",
                "ONIT_TARGET_ENV_BIN"):
        val = os.environ.get(var)
        if val:
            env[var] = val

    if IS_WINDOWS:
        # Build PATH that includes Git Bash Unix tools + essential Windows paths
        path_parts = []
        # Git Bash provides Unix coreutils (wc, grep, sed, awk, etc.)
        if _BASH_EXE:
            git_root = os.path.dirname(os.path.dirname(_BASH_EXE))
            for subdir in ("usr/bin", "bin", "mingw64/bin"):
                d = os.path.join(git_root, subdir)
                if os.path.isdir(d):
                    path_parts.append(d)
        # Essential Windows system paths (for python, pip, etc.)
        sys_root = os.environ.get("SystemRoot", r"C:\Windows")
        path_parts.extend([
            os.path.join(sys_root, "System32"),
            sys_root,
        ])
        env.update({
            "TEMP": tmp_dir,
            "TMP": tmp_dir,
            "PATH": os.pathsep.join(path_parts),
            "SystemRoot": sys_root,
            "COMSPEC": os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe"),
        })
    else:
        # Include conda, container venv, and Homebrew paths so python/pip resolve
        # to the environment where onit installed torch et al. /opt/venv is the
        # venv created by our Dockerfile for --container runs; missing paths are
        # filtered out below, so it's harmless on hosts without it.
        path_parts = [
            "/opt/venv/bin",
            "/opt/miniconda3/bin",
            "/opt/miniconda3/condabin",
            "/usr/local/bin",
            "/opt/homebrew/bin",
            "/usr/bin",
            "/bin",
        ]
        base_path = ":".join(p for p in path_parts if os.path.isdir(p))
        target_env_bin = os.environ.get("ONIT_TARGET_ENV_BIN")
        if target_env_bin and os.path.isdir(target_env_bin):
            env["PATH"] = target_env_bin + ":" + base_path
        else:
            env["PATH"] = base_path

    if DOCUMENTS_PATH:
        env["DOCUMENTS_PATH"] = os.path.realpath(os.path.expanduser(DOCUMENTS_PATH))

    _SANDBOX_ENV = env
    return dict(env)


# Patterns blocked even in unrestricted/container modes (catastrophic ops only).
_ALWAYS_BLOCKED_PATTERNS = [
    # Destructive filesystem operations
    (re.compile(r'\brm\s+(-[^\s]*\s+)*-r\s*f?\s+/', re.IGNORECASE), "rm -rf on root"),
    (re.compile(r'\brm\s+(-[^\s]*\s+)*-f\s*r?\s+/', re.IGNORECASE), "rm -rf on root"),
    (re.compile(r'\brm\s+(-[^\s]*\s+)*/\s*$', re.IGNORECASE), "rm on root"),
    (re.compile(r'>\s*/dev/(sd|hd|nvme|vd|xvd)', re.IGNORECASE), "write to block device"),
    (re.compile(r'>\s*/dev/mem', re.IGNORECASE), "write to /dev/mem"),
    (re.compile(r'\bdd\b.*\bof\s*=\s*/dev/', re.IGNORECASE), "dd to device"),
    (re.compile(r'\bmkfs\b', re.IGNORECASE), "mkfs command"),
    (re.compile(r'\bfdisk\b', re.IGNORECASE), "fdisk command"),
    (re.compile(r'\bparted\b', re.IGNORECASE), "parted command"),
    # System shutdown / state
    (re.compile(r'\bshutdown\b', re.IGNORECASE), "shutdown command"),
    (re.compile(r'\breboot\b', re.IGNORECASE), "reboot command"),
    (re.compile(r'\bpoweroff\b', re.IGNORECASE), "poweroff command"),
    (re.compile(r'\bhalt\b', re.IGNORECASE), "halt command"),
    (re.compile(r'\binit\s+[06]\b', re.IGNORECASE), "init runlevel change"),
    # Kernel / boot
    (re.compile(r'\binsmod\b', re.IGNORECASE), "insmod command"),
    (re.compile(r'\brmmod\b', re.IGNORECASE), "rmmod command"),
    (re.compile(r'\bmodprobe\b', re.IGNORECASE), "modprobe command"),
    (re.compile(r'\bsysctl\s+-w\b', re.IGNORECASE), "sysctl write"),
    # Windows-specific catastrophic ops
    (re.compile(r'\bformat\b\s+[A-Za-z]:', re.IGNORECASE), "format drive"),
    (re.compile(r'\bdiskpart\b', re.IGNORECASE), "diskpart command"),
    (re.compile(r'\bbcdedit\b', re.IGNORECASE), "bcdedit command"),
]

# Blocked command patterns for the bash tool
_BLOCKED_PATTERNS = [
    # Environment variable access
    (re.compile(r'\benv\b', re.IGNORECASE), "env command"),
    (re.compile(r'\bprintenv\b', re.IGNORECASE), "printenv command"),
    (re.compile(r'\bexport\s', re.IGNORECASE), "export command"),
    (re.compile(r'/proc/self/environ', re.IGNORECASE), "/proc/self/environ"),
    # Process listings
    (re.compile(r'\bps\b', re.IGNORECASE), "ps command"),
    (re.compile(r'\btop\b', re.IGNORECASE), "top command"),
    (re.compile(r'\bhtop\b', re.IGNORECASE), "htop command"),
    (re.compile(r'/proc/\d+', re.IGNORECASE), "/proc access"),
    (re.compile(r'/proc/self\b', re.IGNORECASE), "/proc/self access"),
    # Sensitive system files
    (re.compile(r'/etc/passwd', re.IGNORECASE), "/etc/passwd"),
    (re.compile(r'/etc/shadow', re.IGNORECASE), "/etc/shadow"),
    (re.compile(r'/etc/sudoers', re.IGNORECASE), "/etc/sudoers"),
    # Destructive filesystem operations
    (re.compile(r'\brm\s+(-[^\s]*\s+)*-r\s*f?\s+/', re.IGNORECASE), "rm -rf on root"),
    (re.compile(r'\brm\s+(-[^\s]*\s+)*-f\s*r?\s+/', re.IGNORECASE), "rm -rf on root"),
    (re.compile(r'\brm\s+(-[^\s]*\s+)*/\s*$', re.IGNORECASE), "rm on root"),
    (re.compile(r'>\s*/dev/(sd|hd|nvme|vd|xvd)', re.IGNORECASE), "write to block device"),
    (re.compile(r'>\s*/dev/mem', re.IGNORECASE), "write to /dev/mem"),
    (re.compile(r'\bdd\b.*\bof\s*=\s*/dev/', re.IGNORECASE), "dd to device"),
    (re.compile(r'\bmkfs\b', re.IGNORECASE), "mkfs command"),
    (re.compile(r'\bfdisk\b', re.IGNORECASE), "fdisk command"),
    (re.compile(r'\bparted\b', re.IGNORECASE), "parted command"),
    # Privilege escalation
    (re.compile(r'\bsudo\b', re.IGNORECASE), "sudo command"),
    (re.compile(r'\bsu\b\s', re.IGNORECASE), "su command"),
    (re.compile(r'\bdoas\b', re.IGNORECASE), "doas command"),
    (re.compile(r'\bchmod\s+[0-7]*s', re.IGNORECASE), "setuid/setgid chmod"),
    (re.compile(r'\bchown\b', re.IGNORECASE), "chown command"),
    # System state / shutdown
    (re.compile(r'\bshutdown\b', re.IGNORECASE), "shutdown command"),
    (re.compile(r'\breboot\b', re.IGNORECASE), "reboot command"),
    (re.compile(r'\bpoweroff\b', re.IGNORECASE), "poweroff command"),
    (re.compile(r'\bhalt\b', re.IGNORECASE), "halt command"),
    (re.compile(r'\binit\s+[06]\b', re.IGNORECASE), "init runlevel change"),
    (re.compile(r'\bsystemctl\s+(start|stop|restart|disable|enable|mask|reboot|poweroff|halt)', re.IGNORECASE), "systemctl service control"),
    # Network / exfiltration
    (re.compile(r'\bcurl\b.*\|\s*(ba)?sh', re.IGNORECASE), "curl piped to shell"),
    (re.compile(r'\bwget\b.*\|\s*(ba)?sh', re.IGNORECASE), "wget piped to shell"),
    (re.compile(r'\bnc\b\s+-[el]', re.IGNORECASE), "netcat listener"),
    (re.compile(r'\bncat\b', re.IGNORECASE), "ncat command"),
    (re.compile(r'\bsocat\b', re.IGNORECASE), "socat command"),
    (re.compile(r'\biptables\b', re.IGNORECASE), "iptables command"),
    (re.compile(r'\bufw\b', re.IGNORECASE), "ufw command"),
    # Kernel / boot
    (re.compile(r'\binsmod\b', re.IGNORECASE), "insmod command"),
    (re.compile(r'\brmmod\b', re.IGNORECASE), "rmmod command"),
    (re.compile(r'\bmodprobe\b', re.IGNORECASE), "modprobe command"),
    (re.compile(r'\bsysctl\s+-w\b', re.IGNORECASE), "sysctl write"),
    # Package managers (prevent installs that could alter the system)
    (re.compile(r'\bapt(-get)?\s+(install|remove|purge|dist-upgrade)', re.IGNORECASE), "apt package modification"),
    (re.compile(r'\byum\s+(install|remove|erase|update)', re.IGNORECASE), "yum package modification"),
    (re.compile(r'\bdnf\s+(install|remove|erase|upgrade)', re.IGNORECASE), "dnf package modification"),
    (re.compile(r'\bpacman\s+-[SRU]', re.IGNORECASE), "pacman package modification"),
    (re.compile(r'\bbrew\s+(install|uninstall|remove)', re.IGNORECASE), "brew package modification"),
    # Windows-specific dangerous commands
    (re.compile(r'\bformat\b\s+[A-Za-z]:', re.IGNORECASE), "format drive"),
    (re.compile(r'\bdiskpart\b', re.IGNORECASE), "diskpart command"),
    (re.compile(r'\breg\s+(add|delete|import)\b', re.IGNORECASE), "registry modification"),
    (re.compile(r'\bnet\s+(user|localgroup|stop|start)\b', re.IGNORECASE), "net service/user command"),
    (re.compile(r'\bsc\s+(delete|stop|config)\b', re.IGNORECASE), "sc service command"),
    (re.compile(r'\bbcdedit\b', re.IGNORECASE), "bcdedit command"),
    (re.compile(r'\bschtasks\s+/(create|delete)\b', re.IGNORECASE), "scheduled task modification"),
]

# Matches a heredoc construct and its body for stripping before security checks.
# Uses [^\n]*\n per-line matching instead of .*? with DOTALL to avoid ReDoS.
_HEREDOC_RE = re.compile(r"""<<-?\s*['"]?(\w+)['"]?\n(?:[^\n]*\n)*?\1\b""")


def _exec_shell(command: str, cwd: str, timeout: int) -> subprocess.CompletedProcess:
    """Run a command in a sandboxed shell, routing through Git Bash on Windows."""
    if IS_WINDOWS and _BASH_EXE:
        return subprocess.run(
            [_BASH_EXE, "-c", command],
            capture_output=True, text=True,
            cwd=cwd, timeout=timeout, env=_get_sandbox_env()
        )
    return subprocess.run(
        command, shell=True,
        capture_output=True, text=True,
        cwd=cwd, timeout=timeout, env=_get_sandbox_env()
    )


async def _exec_shell_streaming(command: str, cwd: str, timeout: int,
                                ctx: Context | None = None) -> dict:
    """Run a command asynchronously, streaming stdout/stderr lines via ctx.log().

    Returns a dict with stdout, stderr, and returncode.
    Falls back to synchronous _exec_shell if ctx is None.
    """
    if ctx is None:
        result = _exec_shell(command, cwd, timeout)
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }

    env = _get_sandbox_env()
    if IS_WINDOWS and _BASH_EXE:
        proc = await asyncio.create_subprocess_exec(
            _BASH_EXE, "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd, env=env,
        )
    else:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd, env=env,
        )

    stdout_lines = []
    stderr_lines = []
    _MAX_LINES = 10_000  # Cap collected lines to prevent OOM on verbose commands

    async def _read_stream(stream, collector, level):
        async for raw_line in stream:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            if len(collector) < _MAX_LINES:
                collector.append(line)
            await ctx.log(level=level, message=line)

    try:
        await asyncio.wait_for(
            asyncio.gather(
                _read_stream(proc.stdout, stdout_lines, "info"),
                _read_stream(proc.stderr, stderr_lines, "warning"),
            ),
            timeout=timeout,
        )
        await proc.wait()
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass  # Process didn't respond to SIGKILL; nothing more we can do
        return {
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
            "returncode": -1,
            "timeout": True,
        }

    return {
        "stdout": "\n".join(stdout_lines),
        "stderr": "\n".join(stderr_lines),
        "returncode": proc.returncode,
    }


def _strip_heredoc_bodies(command: str) -> str:
    """Remove heredoc body text from a command string before security checking.

    Heredoc bodies are written to files as data, not executed as shell code.
    Scanning them for blocked patterns produces false positives (e.g. the word
    'env' in document content triggering the env-command block).

    Handles all three quoting styles: << 'DELIM', << "DELIM", << DELIM.
    The delimiter word (without quotes) is used to find the closing marker.
    """
    return _HEREDOC_RE.sub(r"<< '\1'", command)


def _validate_bash_command(command: str) -> str | None:
    """Check command for blocked patterns and path references outside allowed dirs.
    Returns error message or None if command is allowed."""
    check_command = _strip_heredoc_bodies(command)

    # In unrestricted mode only the catastrophic always-blocked ops are checked;
    # path restrictions and package-manager/env blocks are lifted.
    if _in_unrestricted():
        for pattern, description in _ALWAYS_BLOCKED_PATTERNS:
            if pattern.search(check_command):
                return f"Command blocked: {description} is not allowed."
        return None

    # Check blocked command patterns
    for pattern, description in _BLOCKED_PATTERNS:
        if pattern.search(check_command):
            return f"Command blocked: {description} is not allowed."

    # In container mode the container itself is the filesystem boundary —
    # skip the path allowlist so agent commands can reach --container-mount
    # directories and any other path visible to the container.
    if _in_container():
        return None

    # Check for absolute path references outside allowed directories
    abs_data = os.path.realpath(os.path.expanduser(DATA_PATH))
    abs_docs = os.path.realpath(os.path.expanduser(DOCUMENTS_PATH)) if DOCUMENTS_PATH else None

    # Normalize for comparison (on Windows, realpath uses backslashes)
    abs_data_lower = abs_data.lower() if IS_WINDOWS else abs_data
    abs_docs_lower = abs_docs.lower() if (IS_WINDOWS and abs_docs) else abs_docs

    target_env_bin = os.environ.get("ONIT_TARGET_ENV_BIN")
    # Allow the entire env prefix (parent of bin/) so scripts can reference
    # env_B's lib/, include/, share/, etc. with absolute paths.
    target_env_prefix = os.path.normpath(os.path.dirname(target_env_bin)) if target_env_bin else None

    def _is_allowed_path(p: str) -> bool:
        """Check if a path is within allowed directories or is a standard tool path."""
        norm = os.path.normpath(p)
        if IS_WINDOWS:
            norm_lower = norm.lower()
            if norm_lower.startswith(abs_data_lower):
                return True
            if abs_docs_lower and norm_lower.startswith(abs_docs_lower):
                return True
        else:
            if norm.startswith(abs_data):
                return True
            if abs_docs and norm.startswith(abs_docs):
                return True
        # Allow the target conda env (bin/, lib/, share/, etc.)
        if target_env_prefix and norm.startswith(target_env_prefix):
            return True
        # Allow standard Unix tool/device paths (check against original path
        # since os.path.normpath on Windows converts /usr/bin to \usr\bin)
        if p.startswith(('/usr/bin/', '/usr/local/bin/', '/bin/',
                         '/dev/null', '/dev/stdout', '/dev/stderr',
                         '/dev/stdin', '/dev/fd/')):
            return True
        return False

    # Match Unix-style absolute paths (/...)
    for match in re.finditer(r'(?:^|(?<=\s)|(?<=[;|&><]))(/[^\s;|&><"\'`\)]+)', check_command):
        if not _is_allowed_path(match.group(1)):
            path = os.path.normpath(match.group(1))
            return (f"Command blocked: references path '{path}' outside allowed directories. "
                    f"Commands must operate within DATA_PATH ({abs_data})"
                    + (f" or DOCUMENTS_PATH ({abs_docs})" if abs_docs else "")
                    + ".")

    # Match Windows-style absolute paths (C:\... or C:/...)
    if IS_WINDOWS:
        for match in re.finditer(r'(?:^|(?<=\s)|(?<=["]))([A-Za-z]:[/\\][^\s;|&><"\'`\)]*)', check_command):
            if not _is_allowed_path(match.group(1)):
                path = os.path.normpath(match.group(1))
                return (f"Command blocked: references path '{path}' outside allowed directories. "
                        f"Commands must operate within DATA_PATH ({abs_data})"
                        + (f" or DOCUMENTS_PATH ({abs_docs})" if abs_docs else "")
                        + ".")

    return None


# =============================================================================
# TOOL 1: RUN COMMAND
# =============================================================================

@mcp.tool(
    title="Run Shell Command",
    description="""Execute a bash/shell command. Captures stdout, stderr, and return code.

Args:
- command: Shell command to run (e.g., "ls -la", "python script.py", "grep -r 'TODO' .")
- cwd: FULL absolute path to working directory. Always use the complete working directory path from your system prompt (e.g., "/tmp/onit/data/<session_id>") - never use relative paths.
- timeout: Max seconds to wait (default: 300)

Returns JSON: {stdout, stderr, returncode, cwd, command, status}

Container mode (ONIT_CONTAINER=1): heavy ML packages (torch, transformers, accelerate, datasets, tokenizers, safetensors, hf_transfer, einops, phonemizer) are NOT preinstalled. Install them on demand with `onit-install-ml [torch|hf|extras|all]` — it picks CUDA vs CPU wheels automatically and writes to the persistent data volume, so the install happens once per host."""
)
async def bash(
    command: Optional[str] = None,
    cwd: str = ".",
    timeout: int = DEFAULT_TIMEOUT,
    ctx: Context = None,
) -> str:
    if err := _validate_required(command=command):
        return err
    try:
        # Validate command against blocklist and path restrictions
        if err_msg := _validate_bash_command(command):
            return json.dumps({
                "error": err_msg,
                "command": command,
                "status": "blocked"
            })

        # Validate and normalize working directory
        # Default to DATA_PATH when cwd is "." (default) to avoid
        # rejecting commands when process cwd is outside allowed dirs
        if cwd == ".":
            cwd = DATA_PATH
        work_dir = os.path.abspath(os.path.expanduser(cwd))
        try:
            work_dir = _validate_dir_path(work_dir)
        except ValueError as e:
            return json.dumps({
                "error": str(e),
                "command": command,
                "status": "error"
            })
        if not os.path.isdir(work_dir):
            return json.dumps({
                "error": f"Working directory does not exist: {work_dir}",
                "command": command
            })

        # Cap timeout at 5 minutes
        timeout = min(timeout, 300)

        # Execute command with streaming output via ctx.log()
        result = await _exec_shell_streaming(command, work_dir, timeout, ctx=ctx)

        stdout = _truncate_output(result["stdout"])
        stderr = result["stderr"]
        returncode = result["returncode"]

        if result.get("timeout"):
            return json.dumps({
                "error": f"Command timed out after {timeout} seconds",
                "stdout": stdout if stdout else None,
                "stderr": stderr if stderr else None,
                "command": command,
                "cwd": work_dir,
                "status": "timeout"
            })

        # Provide helpful message for empty output
        if not stdout and not stderr and returncode == 0:
            stdout = "Command completed successfully (no output)"

        return json.dumps({
            "stdout": stdout,
            "stderr": stderr if stderr else None,
            "returncode": returncode,
            "cwd": work_dir,
            "command": command,
            "status": "success" if returncode == 0 else "failed"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "command": command,
            "status": "error"
        })


# =============================================================================
# TOOL 2: READ FILE
# =============================================================================

@mcp.tool(
    title="Read File",
    description="""Read file contents. Supports text files and PDFs. Binary files (images, audio, video) return metadata only.

Args:
- path: FULL absolute file path (e.g., "/tmp/onit/data/<session_id>/report.pdf"). Always use the complete working directory path from your system prompt - never use relative paths.
- encoding: Text encoding (default: utf-8)
- max_chars: Max characters to read (default: 100000)

Returns JSON: {content, path, size_bytes, format, status} or {path, size_bytes, format, type} for binary files"""
)
def read_file(
    path: Optional[str] = None,
    encoding: str = "utf-8",
    max_chars: int = 100000
) -> str:
    if err := _validate_required(path=path):
        return err
    try:
        # Validate path is within allowed directories
        # (relative paths are resolved against DATA_PATH by _validate_read_path)
        file_path = _validate_read_path(path)

        # Check if file exists
        if not os.path.isfile(file_path):
            return json.dumps({
                "error": f"File not found: {file_path}",
                "path": path
            })

        # Get file info
        file_size = os.path.getsize(file_path)
        file_ext = Path(file_path).suffix.lower()

        # Handle binary files (return description only)
        if file_ext in BINARY_EXTENSIONS:
            return _read_binary(file_path, file_size, file_ext)

        # Handle PDF files
        if file_ext == '.pdf':
            return _read_pdf(file_path, file_size, max_chars)

        # Handle text files
        return _read_text(file_path, file_size, file_ext, encoding, max_chars)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "path": path
        })


def _read_binary(file_path: str, file_size: int, file_ext: str) -> str:
    """Return metadata for binary files without reading content."""
    # Categorize binary file types
    type_map = {
        # Images
        '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.gif': 'image',
        '.bmp': 'image', '.webp': 'image', '.ico': 'image', '.tiff': 'image',
        '.tif': 'image', '.svg': 'image',
        # Audio
        '.mp3': 'audio', '.wav': 'audio', '.ogg': 'audio', '.flac': 'audio',
        '.aac': 'audio', '.m4a': 'audio', '.wma': 'audio',
        # Video
        '.mp4': 'video', '.avi': 'video', '.mov': 'video', '.mkv': 'video',
        '.wmv': 'video', '.flv': 'video', '.webm': 'video',
        # Archives
        '.zip': 'archive', '.tar': 'archive', '.gz': 'archive', '.rar': 'archive',
        '.7z': 'archive', '.bz2': 'archive',
        # Executables
        '.exe': 'executable', '.dll': 'library', '.so': 'library',
        '.dylib': 'library', '.bin': 'binary', '.o': 'object', '.a': 'archive',
        # Documents
        '.doc': 'document', '.docx': 'document', '.xls': 'spreadsheet',
        '.xlsx': 'spreadsheet', '.ppt': 'presentation', '.pptx': 'presentation',
        # Other
        '.pyc': 'bytecode', '.class': 'bytecode', '.wasm': 'bytecode',
        '.ttf': 'font', '.otf': 'font', '.woff': 'font', '.woff2': 'font'
    }

    file_type = type_map.get(file_ext, 'binary')
    file_name = os.path.basename(file_path)

    return json.dumps({
        "path": file_path,
        "filename": file_name,
        "size_bytes": file_size,
        "format": file_ext.lstrip('.'),
        "type": file_type,
        "note": f"Binary file ({file_type}). Content not returned.",
        "status": "success"
    }, indent=2)


def _read_pdf(file_path: str, file_size: int, max_chars: int) -> str:
    """Extract text from PDF file."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages = []
        total_chars = 0

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if total_chars + len(text) > max_chars:
                # Truncate this page
                remaining = max_chars - total_chars
                text = text[:remaining] + f"\n\n... [TRUNCATED at page {i+1}/{len(reader.pages)}]"
                pages.append(text)
                break
            pages.append(text)
            total_chars += len(text)

        content = "\n\n".join(pages)

        return json.dumps({
            "content": content,
            "path": file_path,
            "size_bytes": file_size,
            "format": "pdf",
            "pages": len(reader.pages),
            "status": "success"
        }, indent=2)

    except ImportError:
        return json.dumps({
            "error": "pypdf not installed. Run: pip install pypdf",
            "path": file_path
        })
    except Exception as e:
        return json.dumps({
            "error": f"Failed to read PDF: {str(e)}",
            "path": file_path
        })


def _read_text(file_path: str, file_size: int, file_ext: str, encoding: str, max_chars: int) -> str:
    """Read text file content."""
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read(max_chars)

        truncated = file_size > max_chars

        # Detect format
        format_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.html': 'html',
            '.css': 'css',
            '.xml': 'xml',
            '.txt': 'text',
            '.sh': 'shell',
            '.bash': 'shell',
            '.sql': 'sql',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c-header',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
        }
        file_format = format_map.get(file_ext, 'text')

        result = {
            "content": content,
            "path": file_path,
            "size_bytes": file_size,
            "format": file_format,
            "encoding": encoding,
            "status": "success"
        }

        if truncated:
            result["truncated"] = True
            result["truncated_at"] = max_chars

        return json.dumps(result, indent=2)

    except UnicodeDecodeError:
        # Try with latin-1 as fallback
        if encoding != 'latin-1':
            return _read_text(file_path, file_size, file_ext, 'latin-1', max_chars)
        return json.dumps({
            "error": f"Could not decode file with encoding: {encoding}",
            "path": file_path
        })
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "path": file_path
        })


# =============================================================================
# TOOL 3: WRITE FILE
# =============================================================================

@mcp.tool(
    title="Write File",
    description="""Write content to a file. Creates directories if needed.
Files are created within the working directory with owner-only access.

Args:
- path: FULL absolute file path (e.g., "/tmp/onit/data/<session_id>/output.txt"). Always use the complete working directory path from your system prompt - never use relative paths.
- content: Text content to write (required)
- mode: "write" (overwrite) or "append" (add to end) (default: "write")
- encoding: Text encoding (default: utf-8)

Returns JSON: {path, size_bytes, mode, status}"""
)
def write_file(
    path: Optional[str] = None,
    content: Optional[str] = None,
    mode: str = "write",
    encoding: str = "utf-8"
) -> str:
    if err := _validate_required(path=path, content=content):
        return err
    try:
        file_path = _validate_write_path(_normalize_to_data_path(path))

        # Create directory with owner-only permissions
        _secure_makedirs(os.path.dirname(file_path))

        # Determine file mode
        file_mode = 'a' if mode == "append" else 'w'

        # Write content
        fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | (os.O_APPEND if mode == "append" else os.O_TRUNC), 0o600)
        with os.fdopen(fd, file_mode, encoding=encoding) as f:
            f.write(content)

        # Get final file size
        file_size = os.path.getsize(file_path)

        return json.dumps({
            "path": file_path,
            "size_bytes": file_size,
            "mode": mode,
            "encoding": encoding,
            "status": "success"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "path": path,
            "status": "failed"
        })


def _normalize_to_data_path(path: str) -> str:
    """Resolve path into DATA_PATH: relative paths are placed under it; absolute
    paths outside it are re-rooted under it (strips leading separator).

    In container/unrestricted mode we don't re-root — absolute paths are
    honored as-is so the agent can reach any visible location.
    """
    expanded = os.path.expanduser(path)
    if _path_restrictions_relaxed():
        if os.path.isabs(expanded):
            return os.path.abspath(expanded)
        return os.path.join(os.path.abspath(os.path.expanduser(DATA_PATH)), expanded)
    abs_data = os.path.abspath(os.path.expanduser(DATA_PATH))
    if not os.path.isabs(expanded) or not os.path.abspath(expanded).startswith(abs_data):
        return os.path.join(abs_data, expanded.lstrip(os.sep))
    return os.path.abspath(expanded)


def _kill_process(pid: int, grace_timeout: float = 3.0) -> None:
    """Send SIGTERM to the process group, wait up to grace_timeout seconds, then SIGKILL."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_timeout
    while time.monotonic() < deadline:
        if not _process_running(pid):
            return
        time.sleep(0.5)
    if _process_running(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            os.kill(pid, signal.SIGKILL)


# =============================================================================
# TOOL 3b: EDIT FILE (TARGETED STRING REPLACEMENT)
# =============================================================================

@mcp.tool(
    title="Edit File",
    description="""Edit a file by replacing an exact string with new content. The file must exist.
Files must be within the designated data directory.

Args:
- path: FULL absolute file path (e.g., "/tmp/onit/data/<session_id>/main.py"). Always use the complete working directory path — never use relative paths.
- old_string: The exact string to find and replace (must be unique in the file, or use replace_all)
- new_string: The replacement string
- replace_all: Replace every occurrence of old_string (default: false, replaces first occurrence only)
- encoding: Text encoding (default: utf-8)

Returns JSON: {path, replacements, status}"""
)
def edit_file(
    path: Optional[str] = None,
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    replace_all: bool = False,
    encoding: str = "utf-8",
) -> str:
    if err := _validate_required(path=path, old_string=old_string, new_string=new_string):
        return err
    try:
        file_path = _validate_write_path(_normalize_to_data_path(path))

        if not os.path.isfile(file_path):
            return json.dumps({"error": f"File not found: {file_path}", "path": path, "status": "error"})

        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            return json.dumps({"error": "old_string not found in file", "path": file_path, "status": "error"})

        if replace_all:
            new_content = content.replace(old_string, new_string)
            replacements = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replacements = 1

        fd = os.open(file_path, os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(new_content)

        return json.dumps({"path": file_path, "replacements": replacements, "status": "success"}, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "path": path, "status": "error"})


# =============================================================================
# TOOL 3c: SERVE — BACKGROUND PROCESS MANAGER FOR WEB SERVERS
# =============================================================================

_SERVE_REGISTRY_FILE = "serve_registry.json"


def _serve_registry_path() -> str:
    return os.path.join(os.path.abspath(os.path.expanduser(DATA_PATH)), _SERVE_REGISTRY_FILE)


def _load_serve_registry() -> dict:
    path = _serve_registry_path()
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_serve_registry(registry: dict):
    path = _serve_registry_path()
    _secure_makedirs(os.path.dirname(path))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(registry, f, indent=2)


def _process_running(pid: int) -> bool:
    """Return True only if the process exists and is not a zombie."""
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    # On Unix, check /proc/<pid>/status for zombie state (Linux)
    # On macOS, use ps to detect zombie
    try:
        status_path = f"/proc/{pid}/status"
        if os.path.isfile(status_path):
            with open(status_path) as f:
                for line in f:
                    if line.startswith("State:") and "Z" in line:
                        return False
            return True
    except Exception:
        pass
    # macOS fallback: check via ps
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2
        )
        stat = result.stdout.strip()
        return bool(stat) and "Z" not in stat
    except Exception:
        return True


def _tail_file(path: str, lines: int) -> str:
    """Return last `lines` lines of a file without loading it all into memory."""
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return ""
            buf = b""
            pos = size
            chunk_size = 8192
            newlines_found = 0
            while pos > 0 and newlines_found <= lines:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size)
                buf = chunk + buf
                newlines_found += chunk.count(b"\n")
        return b"\n".join(buf.split(b"\n")[-(lines + 1):]).decode("utf-8", errors="replace")
    except Exception:
        return ""


@mcp.tool(
    title="Serve",
    description="""Manage long-running background processes such as web servers.

Actions:
- start   : Launch a command as a background process. Returns name, pid, and log paths.
- stop    : Stop a running process by name or pid.
- status  : Check if a process is running (name or pid).
- logs    : Tail stdout/stderr logs for a process (name or pid).
- list    : List all managed processes with running/stopped status.
- restart : Stop then re-start a named process using its saved command.

Args:
- action  : One of "start", "stop", "status", "logs", "list", "restart" (required)
- command : Shell command to run — required for "start" (e.g., "uvicorn main:app --port 8080")
- name    : Human-readable label for the process (default: auto-generated). Used to reference it later.
- pid     : Process ID — alternative to name for stop/status/logs
- cwd     : Working directory for the process (default: DATA_PATH). Can be any accessible directory.
- lines   : Number of log lines to return for "logs" action (default: 50)

Returns JSON with process details and, for "logs", stdout/stderr tail."""
)
def serve(
    action: Optional[str] = None,
    command: Optional[str] = None,
    name: Optional[str] = None,
    pid: Optional[int] = None,
    cwd: Optional[str] = None,
    lines: int = 50,
) -> str:
    if err := _validate_required(action=action):
        return err

    action = action.lower().strip()

    # ── START ──────────────────────────────────────────────────────────
    if action == "start":
        if err := _validate_required(command=command):
            return err

        registry = _load_serve_registry()

        # Auto-generate name from command if not provided
        if not name:
            base = command.split()[0].split("/")[-1]
            name = f"{base}-{int(time.time())}"

        # Reject if already running
        if name in registry and _process_running(registry[name]["pid"]):
            entry = registry[name]
            return json.dumps({
                "error": f"Process '{name}' is already running (pid {entry['pid']}). "
                         "Use action='stop' first or choose a different name.",
                "name": name, "pid": entry["pid"], "status": "already_running"
            })

        # Resolve cwd — allow any accessible directory (servers live outside DATA_PATH)
        work_dir = os.path.abspath(os.path.expanduser(cwd)) if cwd else os.path.abspath(os.path.expanduser(DATA_PATH))
        if not os.path.isdir(work_dir):
            return json.dumps({"error": f"Working directory not found: {work_dir}", "status": "error"})

        # Create log directory under DATA_PATH
        log_dir = os.path.join(os.path.abspath(os.path.expanduser(DATA_PATH)), "serve_logs", name)
        _secure_makedirs(log_dir)
        stdout_log = os.path.join(log_dir, "stdout.log")
        stderr_log = os.path.join(log_dir, "stderr.log")

        try:
            with open(stdout_log, "w") as out, open(stderr_log, "w") as err_f:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=work_dir,
                    stdout=out,
                    stderr=err_f,
                    start_new_session=True,  # detach from MCP server session
                )

            # Brief pause to catch immediate startup failures
            time.sleep(0.5)
            if not _process_running(proc.pid):
                err_tail = _tail_file(stderr_log, 20)
                return json.dumps({
                    "error": "Process exited immediately after launch.",
                    "stderr": err_tail,
                    "name": name, "status": "failed"
                })

            registry[name] = {
                "pid": proc.pid,
                "command": command,
                "cwd": work_dir,
                "started_at": datetime.now().isoformat(),
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
            }
            _save_serve_registry(registry)

            return json.dumps({
                "name": name,
                "pid": proc.pid,
                "command": command,
                "cwd": work_dir,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "status": "started",
            }, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "name": name, "status": "error"})

    # ── STOP ───────────────────────────────────────────────────────────
    elif action == "stop":
        registry = _load_serve_registry()
        entry = None

        if name and name in registry:
            entry = registry[name]
        elif pid is not None:
            entry = next((e for e in registry.values() if e["pid"] == pid), None)
            name = next((n for n, e in registry.items() if e["pid"] == pid), str(pid))

        if not entry:
            return json.dumps({"error": f"No managed process found for name='{name}' / pid={pid}.", "status": "error"})

        target_pid = entry["pid"]
        if not _process_running(target_pid):
            return json.dumps({"name": name, "pid": target_pid, "status": "already_stopped"})

        try:
            _kill_process(target_pid)
        except Exception as e:
            if _process_running(target_pid):
                return json.dumps({"error": str(e), "name": name, "pid": target_pid, "status": "error"})

        return json.dumps({"name": name, "pid": target_pid, "status": "stopped"}, indent=2)

    # ── STATUS ─────────────────────────────────────────────────────────
    elif action == "status":
        registry = _load_serve_registry()
        entry = None

        if name and name in registry:
            entry = registry[name]
        elif pid is not None:
            entry = next((e for e in registry.values() if e["pid"] == pid), None)
            name = next((n for n, e in registry.items() if e["pid"] == pid), str(pid))

        if not entry:
            return json.dumps({"error": f"No managed process found for name='{name}' / pid={pid}.", "status": "error"})

        running = _process_running(entry["pid"])
        return json.dumps({
            "name": name,
            "pid": entry["pid"],
            "command": entry["command"],
            "cwd": entry["cwd"],
            "started_at": entry["started_at"],
            "stdout_log": entry["stdout_log"],
            "stderr_log": entry["stderr_log"],
            "status": "running" if running else "stopped",
        }, indent=2)

    # ── LOGS ───────────────────────────────────────────────────────────
    elif action == "logs":
        registry = _load_serve_registry()
        entry = None

        if name and name in registry:
            entry = registry[name]
        elif pid is not None:
            entry = next((e for e in registry.values() if e["pid"] == pid), None)
            name = next((n for n, e in registry.items() if e["pid"] == pid), str(pid))

        if not entry:
            return json.dumps({"error": f"No managed process found for name='{name}' / pid={pid}.", "status": "error"})

        stdout_tail = _tail_file(entry["stdout_log"], lines)
        stderr_tail = _tail_file(entry["stderr_log"], lines)
        running = _process_running(entry["pid"])

        return json.dumps({
            "name": name,
            "pid": entry["pid"],
            "status": "running" if running else "stopped",
            "stdout": stdout_tail or "(empty)",
            "stderr": stderr_tail or "(empty)",
        }, indent=2)

    # ── LIST ───────────────────────────────────────────────────────────
    elif action == "list":
        registry = _load_serve_registry()
        if not registry:
            return json.dumps({"processes": [], "total": 0, "status": "ok"})

        processes = []
        dirty = False
        for n, entry in list(registry.items()):
            running = _process_running(entry["pid"])
            processes.append({
                "name": n,
                "pid": entry["pid"],
                "command": entry["command"],
                "cwd": entry["cwd"],
                "started_at": entry["started_at"],
                "status": "running" if running else "stopped",
            })
            if not running:
                del registry[n]
                dirty = True

        if dirty:
            _save_serve_registry(registry)

        return json.dumps({"processes": processes, "total": len(processes), "status": "ok"}, indent=2)

    # ── RESTART ────────────────────────────────────────────────────────
    elif action == "restart":
        registry = _load_serve_registry()

        if not name:
            return json.dumps({"error": "name is required for restart.", "status": "error"})
        if name not in registry:
            return json.dumps({"error": f"No managed process named '{name}'.", "status": "error"})

        entry = registry[name]
        saved_command = entry["command"]
        saved_cwd = entry["cwd"]

        # Stop if running
        target_pid = entry["pid"]
        if _process_running(target_pid):
            try:
                _kill_process(target_pid)
            except Exception:
                pass

        # Re-start with saved command and cwd (override allowed via args)
        # start checks _process_running before re-using the name, so no pre-removal needed
        return serve(
            action="start",
            command=command or saved_command,
            name=name,
            cwd=cwd or saved_cwd,
            lines=lines,
        )

    else:
        return json.dumps({
            "error": f"Unknown action '{action}'. Use: start, stop, status, logs, list, restart.",
            "status": "error"
        })


# =============================================================================
# TOOL 4: SEND FILE TO REMOTE CLIENT
# =============================================================================

MAX_BASE64_SIZE = 10 * 1024 * 1024  # 10MB limit for base64 transfer

@mcp.tool(
    title="Send File",
    description="""Send a file from this host to a remote client.

If callback_url is provided, uploads the file via HTTP POST and returns the download URL.
Otherwise, returns the file content as base64-encoded data (max 10MB).

Args:
- path: FULL absolute file path (e.g., "/tmp/onit/data/<session_id>/file.pdf"). Always use the complete working directory path - never use relative paths. (required)
- callback_url: Full upload URL prefix (e.g., "http://host:9000/uploads/session_id"). File is POSTed to {callback_url}/ (optional)

Returns JSON: {path, filename, size_bytes, download_url, status} or {path, filename, size_bytes, file_data_base64, status}"""
)
def send_file(
    path: Optional[str] = None,
    callback_url: Optional[str] = None
) -> str:
    if err := _validate_required(path=path):
        return err
    try:
        # Validate path is within allowed directories
        # (relative paths are resolved against DATA_PATH by _validate_read_path)
        file_path = _validate_read_path(path)

        if not os.path.isfile(file_path):
            return json.dumps({"error": f"File not found: {file_path}", "path": path})

        file_size = os.path.getsize(file_path)
        filename = os.path.basename(file_path)

        if callback_url:
            # Strip trailing slash for consistent URL construction
            cb = callback_url.rstrip("/")
            # Upload to remote server
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': (filename, f)}
                    resp = requests.post(
                        f"{cb}/", files=files, timeout=60
                    )
                    resp.raise_for_status()
                return json.dumps({
                    "path": file_path,
                    "filename": filename,
                    "size_bytes": file_size,
                    "download_url": f"{cb}/{filename}",
                    "status": "uploaded"
                }, indent=2)
            except Exception as e:
                return json.dumps({
                    "error": f"Upload failed: {str(e)}",
                    "path": file_path,
                    "filename": filename,
                    "status": "failed"
                })

        # No callback_url — return base64 content
        if file_size > MAX_BASE64_SIZE:
            return json.dumps({
                "error": f"File too large for base64 transfer ({file_size} bytes, max {MAX_BASE64_SIZE}). Provide a callback_url instead.",
                "path": file_path,
                "filename": filename,
                "size_bytes": file_size,
                "status": "failed"
            })

        with open(file_path, 'rb') as f:
            content = base64.b64encode(f.read()).decode('ascii')

        return json.dumps({
            "path": file_path,
            "filename": filename,
            "size_bytes": file_size,
            "file_data_base64": content,
            "status": "success"
        })

    except Exception as e:
        return json.dumps({"error": str(e), "path": path, "status": "failed"})


# =============================================================================
# DOCUMENT SEARCH HELPERS
# =============================================================================

MAX_CONTEXT_LINES = 5


def _run_command(command: str, cwd: str = ".", timeout: int = 60) -> Dict[str, Any]:
    """Execute a shell command and return results."""
    try:
        work_dir = os.path.abspath(os.path.expanduser(cwd))
        if not os.path.isdir(work_dir):
            return {"error": f"Directory does not exist: {work_dir}", "status": "error"}

        result = _exec_shell(command, work_dir, timeout)

        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "status": "success" if result.returncode == 0 else "failed"
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout} seconds", "status": "timeout"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


# =============================================================================
# TOOL 5: SEARCH DOCUMENT
# =============================================================================

@mcp.tool(
    title="Search Document",
    description="""Search for a regex pattern in a single document file. Supports text, PDF, and markdown files.
Uses grep-like regex pattern matching and returns matching lines with surrounding context.

IMPORTANT - Required parameters:
- path: FULL absolute file path (e.g., "/tmp/onit/data/<session_id>/report.pdf"). Always use the complete working directory path from your system prompt — never use relative paths.
- pattern: Regex search pattern to find in the document (e.g., "error.*timeout", "subjects")
  Do NOT use 'query' - the parameter name is 'pattern'.

Optional parameters:
- case_sensitive: Whether search is case-sensitive (default: false)
- context_lines: Number of lines of context before/after each match (default: 3).
  Do NOT use 'context_chars' - the parameter name is 'context_lines'.
- max_matches: Maximum number of matches to return (default: 50).
  Do NOT use 'max_sections' - the parameter name is 'max_matches'.

Example: search_document(path="/tmp/onit/data/<session_id>/report.pdf", pattern="conclusion")

Returns JSON: {matches, total_matches, file, format, status}
Each match includes: {line_number, match, context_before, context_after}"""
)
def search_document(
    path: Annotated[Optional[str], Field(description="FULL absolute file path to search")] = None,
    pattern: Annotated[Optional[str], Field(description="Regex search pattern to find in the document (e.g., 'error.*timeout', 'subjects')")] = None,
    case_sensitive: Annotated[bool, Field(description="Whether search is case-sensitive")] = False,
    context_lines: Annotated[int, Field(description="Number of lines of context before/after each match")] = 3,
    max_matches: Annotated[int, Field(description="Maximum number of matches to return")] = 50
) -> str:
    return search_document_impl(
        path=path, pattern=pattern, case_sensitive=case_sensitive,
        context_lines=context_lines, max_matches=max_matches,
        validate_read_path=_validate_read_path,
    )


# =============================================================================
# TOOL 6: SEARCH DIRECTORY
# =============================================================================

@mcp.tool(
    title="Search Directory",
    description="""Search for patterns across files in a directory using grep.
Recursively searches text files matching the file pattern.

Args:
- directory: FULL absolute directory path (e.g., "/tmp/onit/data/<session_id>"). Always use the complete working directory path from your system prompt — never use relative paths.
- pattern: Search pattern (regex with -E flag)
- file_pattern: File glob pattern (default: "*" for all files)
- case_sensitive: Case-sensitive search (default: false)
- include_hidden: Include hidden files (default: false)
- max_results: Maximum results to return (default: 100)

Returns JSON: {results, total_files, total_matches, status}
Each result includes: {file, line_number, content}"""
)
def search_directory(
    directory: Optional[str] = None,
    pattern: Optional[str] = None,
    file_pattern: str = "*",
    case_sensitive: bool = False,
    include_hidden: bool = False,
    max_results: int = 100
) -> str:
    # Pre-resolve before validation (bash server convention)
    resolved = os.path.abspath(os.path.expanduser(directory)) if directory else directory
    return search_directory_impl(
        directory=resolved, pattern=pattern, file_pattern=file_pattern,
        case_sensitive=case_sensitive, include_hidden=include_hidden,
        max_results=max_results,
        validate_dir_path=_validate_dir_path,
        run_command=_run_command,
    )


# =============================================================================
# TOOL 7: EXTRACT TABLES
# =============================================================================

@mcp.tool(
    title="Extract Tables",
    description="""Extract tables from documents. Supports PDF and markdown files.
Tables are returned in a structured format with headers and rows.

Args:
- path: FULL absolute file path (e.g., "/tmp/onit/data/<session_id>/report.pdf"). Always use the complete working directory path from your system prompt — never use relative paths.
- table_index: Specific table index to extract (1-based, default: all)
- output_format: Output format - "json" or "markdown" (default: "json")

Returns JSON: {tables, total_tables, file, format, status}
Each table includes: {headers, rows, row_count, page (for PDF)}"""
)
def extract_tables(
    path: Optional[str] = None,
    table_index: Optional[int] = None,
    output_format: str = "json"
) -> str:
    return extract_tables_impl(
        path=path, table_index=table_index, output_format=output_format,
        validate_read_path=_validate_read_path,
    )


# =============================================================================
# TOOL 8: FIND FILES
# =============================================================================

@mcp.tool(
    title="Find Files",
    description="""Find files matching patterns using the find command.
Searches recursively from the specified directory.

Args:
- directory: FULL absolute directory path (e.g., "/tmp/onit/data/<session_id>"). Always use the complete working directory path from your system prompt — never use relative paths.
- name_pattern: File name pattern (glob, e.g., "*.py", "test_*")
- file_type: Type filter - "f" (file), "d" (directory), or None (all)
- max_depth: Maximum directory depth (default: unlimited)
- size_filter: Size filter (e.g., "+1M", "-100k", "50k")
- modified_days: Modified within N days (e.g., 7 for last week)
- max_results: Maximum results (default: 100)

Returns JSON: {files, total_files, directory, status}"""
)
def find_files(
    directory: str = ".",
    name_pattern: Optional[str] = None,
    file_type: Optional[str] = None,
    max_depth: Optional[int] = None,
    size_filter: Optional[str] = None,
    modified_days: Optional[int] = None,
    max_results: int = 100
) -> str:
    # Pre-resolve before validation (bash server convention)
    resolved = os.path.abspath(os.path.expanduser(directory))
    return find_files_impl(
        directory=resolved, name_pattern=name_pattern, file_type=file_type,
        max_depth=max_depth, size_filter=size_filter, modified_days=modified_days,
        max_results=max_results,
        validate_dir_path=_validate_dir_path,
        run_command=_run_command,
    )


# =============================================================================
# TOOL 9: TRANSFORM TEXT
# =============================================================================

@mcp.tool(
    title="Transform Text",
    description="""Transform text using sed, awk, or tr commands.
Useful for extracting, replacing, or reformatting text content.

Args:
- input_text: Text to transform (or path to file if is_file=true)
- is_file: If true, input_text is treated as a file path (default: false)
- operation: Transformation type - "sed", "awk", or "tr"
- expression: The sed/awk/tr expression to apply
  - sed: e.g., "s/old/new/g", "/pattern/d"
  - awk: e.g., "{print $1}", "NR==1", "/pattern/{print}"
  - tr: e.g., "a-z A-Z" (translate), "-d '\\n'" (delete)

Returns JSON: {output, operation, expression, status}"""
)
def transform_text(
    input_text: Optional[str] = None,
    operation: Optional[str] = None,
    expression: Optional[str] = None,
    is_file: bool = False
) -> str:
    # Pre-resolve file path before validation (bash server convention)
    def _bash_validate_read_path(p: str) -> str:
        return _validate_read_path(os.path.abspath(os.path.expanduser(p)))

    return transform_text_impl(
        input_text=input_text, operation=operation, expression=expression,
        is_file=is_file, data_path=DATA_PATH,
        validate_read_path=_bash_validate_read_path,
        run_command=_run_command,
    )


# =============================================================================
# TOOL 10: GET DOCUMENT CONTEXT
# =============================================================================

@mcp.tool(
    title="Get Document Context",
    description="""Extract relevant context from a document for answering questions.
Searches for keywords and returns surrounding context that can support answers.

Args:
- path: FULL absolute file path (e.g., "/tmp/onit/data/<session_id>/document.pdf"). Always use the complete working directory path — never use relative paths.
- query: The question or topic to find context for
- keywords: Additional keywords to search (comma-separated)
- context_chars: Characters of context around matches (default: 500)
- max_sections: Maximum context sections to return (default: 5)

Returns JSON: {sections, query, file, status}
Each section includes: {content, relevance_keywords, position}"""
)
def get_document_context(
    path: Optional[str] = None,
    query: Optional[str] = None,
    keywords: Optional[str] = None,
    context_chars: int = 500,
    max_sections: int = 5
) -> str:
    return get_document_context_impl(
        path=path, query=query, keywords=keywords,
        context_chars=context_chars, max_sections=max_sections,
        validate_read_path=_validate_read_path,
    )


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18202,
    path: str = "/sse",
    options: dict = {}
) -> None:
    """Run the MCP server."""
    global DATA_PATH, DOCUMENTS_PATH, _SANDBOX_ENV

    if 'verbose' in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    if 'data_path' in options:
        DATA_PATH = options['data_path']
    elif os.environ.get('ONIT_DATA_PATH'):
        DATA_PATH = os.environ['ONIT_DATA_PATH']
    _secure_makedirs(os.path.abspath(os.path.expanduser(DATA_PATH)))

    if 'documents_path' in options:
        DOCUMENTS_PATH = options['documents_path']
    elif os.environ.get('ONIT_DOCUMENTS_PATH'):
        DOCUMENTS_PATH = os.environ['ONIT_DOCUMENTS_PATH']

    # Reset sandbox env cache so it picks up new DATA_PATH/DOCUMENTS_PATH
    _SANDBOX_ENV = None

    logger.info(f"Starting Bash MCP Server at {host}:{port}{path}")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info("10 Core Tools: bash, read_file, write_file, send_file, search_document, search_directory, extract_tables, find_files, transform_text, get_document_context")

    quiet = 'verbose' not in options
    if quiet:
        import uvicorn.config
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    mcp.run(transport=transport, host=host, port=port, path=path,
            uvicorn_config={"access_log": False, "log_level": "warning"} if quiet else {})

if __name__ == "__main__":
    run()
    
