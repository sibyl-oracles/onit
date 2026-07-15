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

Command allowlisting via bash AST parsing.

Parses a shell command string into simple-command nodes (including commands
nested inside $(...), `...`, <(...), pipelines, &&/||/; lists, loops and
subshells) and checks every executable against an allowlist. Package-manager
invocations are additionally gated: blocked by default, and when explicitly
allowed every requested package must carry a pinned version.

The parser is deliberately FAIL-CLOSED: any construct it cannot statically
analyze (case statements, function definitions, arithmetic commands, dynamic
command names like $CMD, unbalanced quoting) raises CommandParseError and the
caller blocks the command. Over-blocking is acceptable; under-blocking is not.

Known, accepted gap: interpreters (python, bash script.sh) can execute
arbitrary code from files. The allowlist is defense-in-depth on top of the
container boundary, not a substitute for it.
'''

import re
from dataclasses import dataclass, field

_MAX_DEPTH = 8


class CommandParseError(ValueError):
    """Command uses shell constructs the static parser cannot analyze."""


@dataclass
class Word:
    text: str = ""
    quoted: bool = False           # any part quoted or backslash-escaped
    has_expansion: bool = False    # $VAR, ${...}, $((...)), brace expansion
    has_substitution: bool = False  # $(...), `...`, <(...), >(...)
    lit_prefix: int = -1           # chars of text before the first quoted part
                                   # (-1: fully unquoted so far)


@dataclass
class SimpleCommand:
    words: list = field(default_factory=list)

    @property
    def argv(self):
        return [w.text for w in self.words]


# Keywords that are transparent prefixes of a real command
# ("if cmd", "while cmd", "! cmd", "{ cmd; }").
_TRANSPARENT_KEYWORDS = {
    "if", "then", "else", "elif", "fi", "while", "until",
    "do", "done", "!", "{", "}",
}
# Constructs we refuse to analyze (fail closed).
_UNSUPPORTED_KEYWORDS = {"case", "select", "function", "coproc"}

_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_]\w*=")
_VAR_NAME_RE = re.compile(r"[A-Za-z_]\w*")
_SPECIAL_PARAMS = set("?#@*$!-0123456789")


def _find_closing_paren(text: str, open_idx: int) -> int:
    """Index of the ')' matching text[open_idx] == '(', skipping quoted spans."""
    i = open_idx + 1
    depth = 1
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "'":
            j = text.find("'", i + 1)
            if j == -1:
                raise CommandParseError("unterminated single quote")
            i = j + 1
            continue
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            if i >= n:
                raise CommandParseError("unterminated double quote")
            i += 1
            continue
        if c == "`":
            j = i + 1
            while j < n and text[j] != "`":
                j += 2 if text[j] == "\\" else 1
            if j >= n:
                raise CommandParseError("unterminated backquote")
            i = j + 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise CommandParseError("unbalanced parentheses")


def _collect_substitutions(text: str, commands: list, depth: int) -> None:
    """Scan data text (arithmetic/parameter-expansion bodies) for embedded
    $(...) / `...` command substitutions and parse them too."""
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
        elif c == "'":
            j = text.find("'", i + 1)
            i = n if j == -1 else j + 1
        elif c == "$" and text[i + 1:i + 2] == "(" and text[i + 2:i + 3] != "(":
            close = _find_closing_paren(text, i + 1)
            _parse_into(text[i + 2:close], commands, depth + 1)
            i = close + 1
        elif c == "`":
            j = i + 1
            while j < n and text[j] != "`":
                j += 2 if text[j] == "\\" else 1
            if j >= n:
                raise CommandParseError("unterminated backquote")
            _parse_into(text[i + 1:j].replace("\\`", "`").replace("\\$", "$"),
                        commands, depth + 1)
            i = j + 1
        else:
            i += 1


def _parse_into(text: str, commands: list, depth: int) -> None:
    """Parse a script fragment, appending every SimpleCommand found."""
    if depth > _MAX_DEPTH:
        raise CommandParseError("command nesting too deep")

    n = len(text)
    i = 0
    words: list[Word] = []
    cur = Word()
    cur_started = False
    expect_redirect_target = False

    def flush_word():
        nonlocal cur, cur_started, expect_redirect_target
        if not cur_started:
            return
        if expect_redirect_target:
            expect_redirect_target = False  # redirect targets are data
        else:
            words.append(cur)
        cur = Word()
        cur_started = False

    def end_command():
        nonlocal words
        flush_word()
        if expect_redirect_target:
            raise CommandParseError("redirection without a target")
        if words:
            cmd = _finish_command(words)
            if cmd is not None:
                commands.append(cmd)
        words = []

    def mark(started=True, quoted=False, expansion=False, substitution=False):
        nonlocal cur_started
        cur_started = cur_started or started
        if quoted and not cur.quoted:
            cur.lit_prefix = len(cur.text)  # call mark() BEFORE appending
        cur.quoted = cur.quoted or quoted
        cur.has_expansion = cur.has_expansion or expansion
        cur.has_substitution = cur.has_substitution or substitution

    def handle_dollar(s: str, j: int) -> int:
        """Handle '$' at s[j] (shared by unquoted and double-quoted states).
        Returns index after the consumed construct."""
        nxt = s[j + 1:j + 2]
        if nxt == "(":
            if s[j + 2:j + 3] == "(":  # $(( arithmetic ))
                k = j + 3
                d = 2
                while k < len(s) and d > 0:
                    if s[k] == "(":
                        d += 1
                    elif s[k] == ")":
                        d -= 1
                    k += 1
                if d > 0:
                    raise CommandParseError("unbalanced arithmetic expansion")
                _collect_substitutions(s[j + 3:k - 1], commands, depth)
                mark(expansion=True)
                return k
            close = _find_closing_paren(s, j + 1)
            _parse_into(s[j + 2:close], commands, depth + 1)
            mark(substitution=True)
            return close + 1
        if nxt == "{":
            k = j + 2
            d = 1
            while k < len(s) and d > 0:
                if s[k] == "\\":
                    k += 1
                elif s[k] == "{":
                    d += 1
                elif s[k] == "}":
                    d -= 1
                k += 1
            if d > 0:
                raise CommandParseError("unterminated parameter expansion")
            _collect_substitutions(s[j + 2:k - 1], commands, depth)
            mark(expansion=True)
            return k
        if nxt and (nxt in _SPECIAL_PARAMS or _VAR_NAME_RE.match(nxt)):
            m = _VAR_NAME_RE.match(s, j + 1)
            mark(expansion=True)
            return m.end() if m else j + 2
        # lone '$' is a literal
        cur.text += "$"
        mark()
        return j + 1

    while i < n:
        c = text[i]

        if c in " \t":
            flush_word()
            i += 1
        elif c == "\\":
            if i + 1 >= n:
                raise CommandParseError("trailing backslash")
            if text[i + 1] == "\n":  # line continuation
                i += 2
            else:
                mark(quoted=True)
                cur.text += text[i + 1]
                i += 2
        elif c == "'":
            j = text.find("'", i + 1)
            if j == -1:
                raise CommandParseError("unterminated single quote")
            mark(quoted=True)
            cur.text += text[i + 1:j]
            i = j + 1
        elif c == '"':
            mark(quoted=True)
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    if i + 1 >= n:
                        raise CommandParseError("unterminated double quote")
                    cur.text += text[i + 1]
                    i += 2
                elif text[i] == "$":
                    i = handle_dollar(text, i)
                elif text[i] == "`":
                    j = i + 1
                    while j < n and text[j] != "`":
                        j += 2 if text[j] == "\\" else 1
                    if j >= n:
                        raise CommandParseError("unterminated backquote")
                    _parse_into(
                        text[i + 1:j].replace("\\`", "`").replace("\\$", "$"),
                        commands, depth + 1)
                    mark(substitution=True)
                    i = j + 1
                else:
                    cur.text += text[i]
                    i += 1
            if i >= n:
                raise CommandParseError("unterminated double quote")
            i += 1
        elif c == "#" and not cur_started:
            j = text.find("\n", i)
            i = n if j == -1 else j
        elif c == "\n":
            end_command()
            i += 1
        elif c == ";":
            if text[i:i + 2] in (";;", ";&"):
                raise CommandParseError("case terminators are not supported")
            end_command()
            i += 1
        elif c == "&":
            if text[i:i + 3] == "&>>" or text[i:i + 2] == "&>":
                flush_word()
                expect_redirect_target = True
                i += 3 if text[i:i + 3] == "&>>" else 2
            else:
                end_command()
                i += 2 if text[i:i + 2] in ("&&",) else 1
        elif c == "|":
            end_command()
            i += 2 if text[i:i + 2] in ("||", "|&") else 1
        elif c == "(":
            if cur_started:
                raise CommandParseError(
                    "function definitions are not supported")
            if text[i + 1:i + 2] == "(":
                raise CommandParseError(
                    "arithmetic commands '((...))' are not supported")
            end_command()  # subshell — treat '(' as a command boundary
            i += 1
        elif c == ")":
            end_command()
            i += 1
        elif c in "<>":
            if text[i + 1:i + 2] == "(":  # process substitution <(...) >(...)
                close = _find_closing_paren(text, i + 1)
                _parse_into(text[i + 2:close], commands, depth + 1)
                mark(substitution=True)
                i = close + 1
                continue
            # A word made purely of digits before a redirect is an fd number.
            if cur_started and cur.text.isdigit() and not cur.quoted:
                cur = Word()
                cur_started = False
            else:
                flush_word()
            if text[i:i + 3] == "<<<":
                i += 3
            elif text[i:i + 3] == "<<-":
                i += 3
            elif text[i:i + 2] == "<<":
                i += 2
            elif text[i:i + 2] in (">>", "<>", ">|"):
                i += 2
            elif text[i:i + 2] in (">&", "<&"):
                i += 2
                # >&2, >&- style: consume fd/dash directly, no target word
                m = re.match(r"\d+-?|-", text[i:])
                if m:
                    i += m.end()
                    continue
            else:
                i += 1
            expect_redirect_target = True
        elif c == "$":
            i = handle_dollar(text, i)
        elif c == "`":
            j = i + 1
            while j < n and text[j] != "`":
                j += 2 if text[j] == "\\" else 1
            if j >= n:
                raise CommandParseError("unterminated backquote")
            _parse_into(text[i + 1:j].replace("\\`", "`").replace("\\$", "$"),
                        commands, depth + 1)
            mark(substitution=True)
            i = j + 1
        elif c == "{" or c == "}":
            boundary = (i + 1 >= n) or text[i + 1] in " \t\n;"
            if not cur_started and boundary:
                end_command()  # brace group delimiter
                i += 1
            else:
                cur.text += c
                mark(expansion=not cur.quoted)  # possible brace expansion
                i += 1
        else:
            cur.text += c
            mark()
            i += 1

    end_command()


def _finish_command(words: list) -> SimpleCommand | None:
    """Strip transparent keywords and env assignments; classify the rest."""
    idx = 0
    while idx < len(words):
        w = words[idx]
        m = _ASSIGNMENT_RE.match(w.text)
        if not w.quoted and w.text in _TRANSPARENT_KEYWORDS:
            idx += 1
        # Assignment only when the NAME= prefix itself is unquoted ("FOO=1"
        # runs a program named FOO=1; FOO="a b" is a real assignment). Value
        # substitutions were already parsed during scanning.
        elif m and (not w.quoted or w.lit_prefix >= m.end()):
            idx += 1
        else:
            break
    rest = words[idx:]
    if not rest:
        return None
    first = rest[0]
    if not first.quoted:
        if first.text in _UNSUPPORTED_KEYWORDS:
            raise CommandParseError(
                f"'{first.text}' statements are not supported")
        if first.text == "for":
            return None  # loop head is data; the body arrives after 'do'
        if first.text == "[[":
            return None  # conditional expression, no command executed
    return SimpleCommand(rest)


def parse_commands(command: str) -> list:
    """Parse a shell command string into every SimpleCommand it executes.

    Includes commands nested in substitutions, pipelines, lists, loops and
    subshells. Raises CommandParseError on constructs that cannot be
    statically analyzed.
    """
    commands: list[SimpleCommand] = []
    _parse_into(command, commands, 0)
    return commands


# =============================================================================
# ALLOWLIST + PACKAGE-MANAGER POLICY
# =============================================================================

# Curated default allowlist. Extend via ONIT_ALLOWED_COMMANDS (comma-separated)
# or permissions.allowedCommands in ~/.onit/settings.json. System package
# managers (apt, yum, dnf, pacman, brew, apk, snap, dpkg) are deliberately
# absent; language package managers (pip, npm, ...) are present but their
# install subcommands are gated by the package policy below.
DEFAULT_ALLOWED_COMMANDS = frozenset({
    # shell builtins / control
    "cd", "pwd", "true", "false", "test", "[", "echo", "printf", "read",
    "exit", "set", "unset", "shift", "local", "declare", "typeset", "let",
    "type", "which", "hash", "wait", "kill", "umask", "getopts", "trap",
    # shells & wrappers
    "bash", "sh", "dash", "zsh", "env", "nohup", "time", "timeout", "nice",
    "stdbuf", "xargs", "command", "builtin", "exec",
    # core file / text utilities
    "ls", "cat", "tac", "head", "tail", "wc", "sort", "uniq", "cut", "tr",
    "tee", "dirname", "basename", "realpath", "readlink", "stat", "file",
    "du", "df", "touch", "mkdir", "rmdir", "rm", "cp", "mv", "ln", "chmod",
    "find", "grep", "egrep", "fgrep", "rg", "sed", "awk", "gawk", "mawk",
    "diff", "cmp", "comm", "paste", "join", "split", "csplit", "nl", "od",
    "xxd", "hexdump", "strings", "shuf", "seq", "sleep", "date", "cal",
    "yes", "fold", "fmt", "expand", "unexpand", "column", "rev", "less",
    "more", "install", "patch", "mktemp",
    # archives / hashing / encoding
    "tar", "gzip", "gunzip", "zcat", "bzip2", "bunzip2", "xz", "unxz",
    "zstd", "unzstd", "zip", "unzip", "7z", "md5sum", "shasum", "sha1sum",
    "sha256sum", "sha512sum", "b2sum", "cksum", "base64", "base32",
    # data / network clients (curl|sh piping is blocked separately)
    "jq", "curl", "wget", "openssl", "sqlite3", "pandoc",
    # dev toolchain
    "git", "git-lfs", "python", "pip", "node", "npm", "npx", "pnpm", "yarn",
    "uv", "pipx", "gem", "cargo", "rustc", "go", "gofmt", "ruby", "perl",
    "java", "javac", "make", "cmake", "ninja", "gcc", "g++", "cc", "c++",
    "clang", "clang++", "ld", "ar", "nm", "ranlib", "strip", "objdump",
    "readelf", "ldd", "pkg-config", "nvcc", "nvidia-smi",
    # python tooling
    "pytest", "ruff", "black", "mypy", "flake8", "pylint", "isort",
    # app servers
    "uvicorn", "gunicorn", "flask",
    # media
    "ffmpeg", "ffprobe", "convert", "magick",
    # system info
    "uname", "id", "whoami", "groups", "hostname", "tty", "locale", "clear",
    # onit itself
    "onit",
})

# Wrappers that execute another command given as their arguments.
_SIMPLE_WRAPPERS = {"command", "builtin", "exec", "nohup", "time"}
# Wrapper flags that consume the following token as their value.
_WRAPPER_VALUE_FLAGS = {
    "env": {"-u", "-C", "-S"},
    "timeout": {"-k", "--kill-after", "-s", "--signal"},
    "nice": {"-n", "--adjustment"},
    "stdbuf": {"-i", "-o", "-e"},
    "xargs": {"-a", "-d", "-E", "-e", "-I", "-i", "-L", "-l", "-n", "-P", "-s"},
}

_VERSIONED_EXE_RE = re.compile(r"^(python|pip)[0-9.]*$")
_SUFFIXED_EXE_RE = re.compile(r"^(gcc|g\+\+|cc|c\+\+|clang|clang\+\+|node)-[\d.]+$")

# pip-style pinned requirement: name[extras]==version
_PIP_PIN_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._\s-]*\])?==[A-Za-z0-9.!+*_-]+$")
# apt/conda-style pin: name=version (or name==version for conda)
_EQ_PIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*={1,2}[A-Za-z0-9.:~+*_-]+$")

# pip flags whose value is the next token (needed to find positional args).
_PIP_VALUE_FLAGS = {
    "-i", "--index-url", "--extra-index-url", "-f", "--find-links",
    "-t", "--target", "--platform", "--python-version", "--implementation",
    "--abi", "--root", "--prefix", "--src", "--upgrade-strategy",
    "--install-option", "--global-option", "--config-settings", "--no-binary",
    "--only-binary", "--progress-bar", "--proxy", "--retries", "--timeout",
    "--trusted-host", "--cert", "--client-cert", "--cache-dir", "--log",
    "--python", "--report", "-C",
}
_PIP_FORBIDDEN_FLAGS = {"-r", "--requirement", "-e", "--editable"}

_NPM_INSTALL_SUBCOMMANDS = {"install", "i", "add", "update", "up", "upgrade",
                            "uninstall", "remove", "rm", "unlink", "link"}
_SYSTEM_PKG_MANAGERS = {"apt", "apt-get", "aptitude", "dpkg", "yum", "dnf",
                        "pacman", "apk", "brew", "snap", "zypper", "emerge"}


def _basename(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if name.lower().endswith(".exe"):
        name = name[:-4]
    return name


def _normalize_exe(name: str) -> str:
    m = _VERSIONED_EXE_RE.match(name)
    if m:
        return m.group(1)
    m = _SUFFIXED_EXE_RE.match(name)
    if m:
        return m.group(1)
    return name


def _positionals(args: list, value_flags: set) -> list:
    """Extract positional (non-flag) Words from an argument list."""
    out = []
    skip_next = False
    no_more_flags = False
    for w in args:
        if skip_next:
            skip_next = False
            continue
        t = w.text
        if not no_more_flags:
            if t == "--":
                no_more_flags = True
                continue
            if t.startswith("-") and t != "-":
                if t in value_flags:
                    skip_next = True
                continue
        out.append(w)
    return out


def _blocked(msg: str) -> str:
    return f"Command blocked: {msg}"


def _check_pip_install(args: list, allow_installs: bool) -> str | None:
    sub = None
    sub_idx = -1
    for idx, w in enumerate(args):
        if not w.text.startswith("-"):
            sub = w.text
            sub_idx = idx
            break
    if sub not in ("install", "uninstall", "download"):
        return None
    if not allow_installs:
        return _blocked(
            "package installation is disabled by default. Set "
            "ONIT_ALLOW_PACKAGE_INSTALL=1 (or use --container-allow-installs) "
            "to permit installs of pinned package versions.")
    if sub == "uninstall":
        return None
    rest = args[sub_idx + 1:]
    for w in rest:
        if w.text in _PIP_FORBIDDEN_FLAGS:
            return _blocked(
                f"'pip {sub} {w.text}' is not allowed — requirements files "
                "and editable installs cannot be pin-verified. Install pinned "
                "packages explicitly (name==version).")
    for w in _positionals(rest, _PIP_VALUE_FLAGS):
        t = w.text
        if w.has_expansion or w.has_substitution:
            return _blocked(
                "package names must be literal so version pins can be verified.")
        if "://" in t or t.startswith(("git+", ".", "/", "~")):
            return _blocked(
                f"'pip {sub}' from URLs or local paths is not allowed: {t}")
        if not _PIP_PIN_RE.match(t):
            return _blocked(
                f"package '{t}' is not version-pinned. Use name==version "
                f"(e.g. '{t.split('==')[0].split('[')[0]}==1.2.3').")
    return None


def _check_node_pkg(exe: str, args: list, allow_installs: bool) -> str | None:
    positionals = _positionals(args, set())
    sub = positionals[0].text if positionals else None
    dlx_like = exe == "npx" or (sub in ("dlx", "exec") and exe in ("pnpm", "yarn", "npm"))
    if exe == "npx":
        pkgs = positionals[:1]
    elif sub in _NPM_INSTALL_SUBCOMMANDS or sub in ("dlx", "exec"):
        pkgs = positionals[1:]
    else:
        return None
    if not allow_installs:
        return _blocked(
            "package installation is disabled by default. Set "
            "ONIT_ALLOW_PACKAGE_INSTALL=1 (or use --container-allow-installs) "
            "to permit installs of pinned package versions.")
    if sub in ("uninstall", "remove", "rm", "unlink"):
        return None
    if sub == "ci":
        return None  # lockfile-driven, inherently pinned
    if not pkgs and not dlx_like:
        return None  # bare `npm install` resolves from the lockfile
    for w in pkgs:
        if w.has_expansion or w.has_substitution:
            return _blocked(
                "package names must be literal so version pins can be verified.")
        if w.text.rfind("@") <= 0:
            return _blocked(
                f"package '{w.text}' is not version-pinned. Use name@version.")
    return None


def _check_package_policy(exe: str, args: list,
                          allow_installs: bool) -> str | None:
    """Gate package-manager install subcommands; require pins when allowed."""
    if exe in _SYSTEM_PKG_MANAGERS:
        # Only reachable when the user added the manager to the allowlist.
        positionals = _positionals(args, set())
        sub = positionals[0].text if positionals else None
        if sub in ("install", "add", "remove", "purge", "erase", "upgrade",
                   "dist-upgrade", "update", "-S", "-R", "-U"):
            if not allow_installs:
                return _blocked(
                    "package installation is disabled by default. Set "
                    "ONIT_ALLOW_PACKAGE_INSTALL=1 to permit pinned installs.")
            if sub in ("install", "add"):
                for w in positionals[1:]:
                    if not _EQ_PIN_RE.match(w.text):
                        return _blocked(
                            f"package '{w.text}' is not version-pinned "
                            "(use name=version).")
        return None

    if exe == "pip":
        return _check_pip_install(args, allow_installs)

    if exe == "python":
        texts = [w.text for w in args]
        if "-m" in texts:
            m_idx = texts.index("-m")
            if m_idx + 1 < len(texts) and _normalize_exe(texts[m_idx + 1]) == "pip":
                return _check_pip_install(args[m_idx + 2:], allow_installs)
        return None

    if exe == "uv":
        texts = [w.text for w in args]
        if texts[:1] == ["pip"]:
            return _check_pip_install(args[1:], allow_installs)
        positionals = _positionals(args, set())
        sub = positionals[0].text if positionals else None
        if sub in ("add", "tool", "sync"):
            if not allow_installs:
                return _blocked(
                    "package installation is disabled by default. Set "
                    "ONIT_ALLOW_PACKAGE_INSTALL=1 to permit pinned installs.")
            if sub == "add":
                for w in positionals[1:]:
                    if not _PIP_PIN_RE.match(w.text):
                        return _blocked(
                            f"package '{w.text}' is not version-pinned "
                            "(use name==version).")
        return None

    if exe == "pipx":
        positionals = _positionals(args, set())
        sub = positionals[0].text if positionals else None
        if sub in ("install", "run", "upgrade"):
            if not allow_installs:
                return _blocked(
                    "package installation is disabled by default. Set "
                    "ONIT_ALLOW_PACKAGE_INSTALL=1 to permit pinned installs.")
            for w in positionals[1:]:
                if not w.text.startswith("-") and not _PIP_PIN_RE.match(w.text):
                    return _blocked(
                        f"package '{w.text}' is not version-pinned "
                        "(use name==version).")
        return None

    if exe in ("npm", "npx", "pnpm", "yarn"):
        return _check_node_pkg(exe, args, allow_installs)

    if exe == "gem":
        positionals = _positionals(args, set())
        if positionals and positionals[0].text == "install":
            if not allow_installs:
                return _blocked(
                    "package installation is disabled by default. Set "
                    "ONIT_ALLOW_PACKAGE_INSTALL=1 to permit pinned installs.")
            texts = [w.text for w in args]
            if "-v" not in texts and "--version" not in texts \
                    and not any(":" in w.text for w in positionals[1:]):
                return _blocked(
                    "gem install requires a pinned version "
                    "(gem install NAME -v X.Y.Z).")
        return None

    if exe in ("cargo", "go"):
        positionals = _positionals(args, set())
        sub = positionals[0].text if positionals else None
        if (exe == "cargo" and sub == "install") or \
                (exe == "go" and sub in ("install", "get")):
            if not allow_installs:
                return _blocked(
                    "package installation is disabled by default. Set "
                    "ONIT_ALLOW_PACKAGE_INSTALL=1 to permit pinned installs.")
            texts = [w.text for w in positionals[1:]]
            if "--version" not in [w.text for w in args] and \
                    not all("@" in t for t in texts if not t.startswith("-")):
                return _blocked(
                    f"{exe} {sub} requires pinned versions (name@version).")
        return None

    if exe in ("conda", "mamba", "micromamba"):
        positionals = _positionals(args, set())
        sub = positionals[0].text if positionals else None
        if sub in ("install", "create", "update"):
            if not allow_installs:
                return _blocked(
                    "package installation is disabled by default. Set "
                    "ONIT_ALLOW_PACKAGE_INSTALL=1 to permit pinned installs.")
            if sub == "install":
                for w in positionals[1:]:
                    if not _EQ_PIN_RE.match(w.text):
                        return _blocked(
                            f"package '{w.text}' is not version-pinned "
                            "(use name=version).")
        return None

    return None


def _unwrap_command(words: list, allowed: frozenset) -> tuple:
    """Peel wrapper commands (env, nohup, timeout, xargs, ...) off the front.
    Returns (executable_name, remaining_args) or (None, None) when the
    command reduces to nothing (e.g. bare 'env')."""
    for _ in range(_MAX_DEPTH):
        if not words:
            return None, None
        head = words[0]
        if head.has_substitution or head.has_expansion:
            raise CommandParseError(
                "dynamic command names ($VAR, $(...)) cannot be allowlisted")
        exe = _normalize_exe(_basename(head.text))
        if exe not in allowed and _basename(head.text) not in allowed:
            return exe, words[1:]
        if exe in _SIMPLE_WRAPPERS:
            rest = words[1:]
            while rest and rest[0].text.startswith("-"):
                rest = rest[1:]
            if not rest:
                return None, None
            words = rest
            continue
        if exe in ("env", "nice", "stdbuf", "xargs", "timeout"):
            value_flags = _WRAPPER_VALUE_FLAGS.get(exe, set())
            rest = words[1:]
            out = []
            skip_next = False
            for w in rest:
                if skip_next:
                    skip_next = False
                    continue
                t = w.text
                if exe == "env" and _ASSIGNMENT_RE.match(t):
                    continue
                if t.startswith("-") and t != "-":
                    if t in value_flags:
                        skip_next = True
                    continue
                out.append(w)
            if exe == "timeout" and out:
                out = out[1:]  # first positional is the duration
            if not out:
                return None, None
            words = out
            continue
        return exe, words[1:]
    raise CommandParseError("too many wrapper commands")


def check_command(command: str, allowed: frozenset,
                  allow_installs: bool = False, _depth: int = 0) -> str | None:
    """Validate a command string against the allowlist and package policy.

    Returns an error string when blocked, else None. Fails closed: parse
    errors block the command.
    """
    if _depth > _MAX_DEPTH:
        return _blocked("shell re-invocation nesting too deep.")
    try:
        cmds = parse_commands(command)
    except CommandParseError as e:
        return _blocked(
            f"cannot statically analyze this command for allowlisting ({e}). "
            "Rewrite it using simpler shell constructs.")

    for cmd in cmds:
        try:
            exe, args = _unwrap_command(cmd.words, allowed)
        except CommandParseError as e:
            return _blocked(str(e))
        if exe is None:
            continue
        if exe not in allowed:
            return _blocked(
                f"'{exe}' is not in the command allowlist. Extend it via "
                "ONIT_ALLOWED_COMMANDS or permissions.allowedCommands in "
                "settings.json if this command should be permitted.")
        # Shell re-invocation: validate the -c payload recursively.
        if exe in ("bash", "sh", "zsh", "dash"):
            texts = [w.text for w in args]
            has_c = any(t.startswith("-") and "c" in t and not t.startswith("--")
                        for t in texts)
            if has_c:
                script = next((w for w in args
                               if not w.text.startswith("-")), None)
                if script is None:
                    continue
                if script.has_expansion or script.has_substitution:
                    return _blocked(
                        "dynamic shell -c payloads cannot be allowlisted.")
                if err := check_command(script.text, allowed,
                                        allow_installs, _depth + 1):
                    return err
                continue
        # find -exec/-execdir/-ok run an arbitrary program: check it too.
        if exe == "find":
            expect_cmd = False
            for w in args:
                if expect_cmd:
                    if w.has_expansion or w.has_substitution:
                        return _blocked(
                            "dynamic find -exec targets cannot be allowlisted.")
                    target = _normalize_exe(_basename(w.text))
                    if target not in allowed:
                        return _blocked(
                            f"'{target}' (via find -exec) is not in the "
                            "command allowlist.")
                    expect_cmd = False
                elif w.text in ("-exec", "-execdir", "-ok", "-okdir"):
                    expect_cmd = True
        if err := _check_package_policy(exe, args, allow_installs):
            return err
    return None
