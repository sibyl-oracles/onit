"""
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

Loop C, part two: the validation gate.  Five checks, all of which pass.

A skill is code the agent wrote, for the agent, to run with the agent's
privileges.  That is the sharpest edge in the whole self-improvement
track: everything else in the track changes what the agent *knows*, and
this changes what the agent *can do*.  So the gate is not a formality.
It is the thing standing between a mined fragment and a running tool.

The five checks are the RFC's, and they are conjunctive.  A skill that
fails any one of them is not admitted, and the failure says which one
and why, because a gate that only says "no" is a gate nobody can fix.

  1. **The manifest is well formed.**  A name that is a valid tool
     name, a description that says what the tool does, a version, and
     an input schema that parses.  A skill with no description is a
     skill the model will never choose, and a skill whose description
     lies is worse than no skill.

  2. **The source is clean.**  A static check: no new imports outside
     the standard library, no shelling out to something the command
     policy would refuse, no touching the network, no writing outside
     the skill's own directory, no reading the ruler.

  3. **The dependencies are permitted.**  A skill that needs a package
     that is not installed is a skill that fails at the worst moment.
     When installs are sealed -- the container, in web mode -- a new
     dependency is refused outright.

  4. **The tests pass.**  In the sandbox (I6).  A skill whose own
     tests fail does not get to run in the agent's place.

  5. **It does not regress.**  The benchmark delta on the holdout is
     >= 0.  A new tool that adds nothing still costs schema tokens on
     every request and dilutes the model's choice among every other
     tool.

**The gate does not run the code.**  It reads it, and it hands the
running to the sandbox, which is a separate process under the container
path with the command policy in force.  A validator that executes the
thing it is validating has already lost.
"""

import ast
import io
import os
import re

from .skills import manifest_errors

# The imports a skill may use.
#
# This is an *allowlist of pure computation*, and it is short on purpose.
# A skill is a transformation from its arguments to a value; the agent
# that calls it already has bash, read_file and write_file for anything
# else.  So the modules that touch the filesystem, the process, or the
# network are not here -- not because a skill could never want them, but
# because wanting them is a decision for an operator, and a list of
# modules the loop may import on its own is a list that has to be short.
#
# The previous version of this set contained ``os``, ``sys``,
# ``subprocess``, ``socket``, ``http``, ``urllib``, ``pickle``,
# ``shutil``, ``tempfile``, ``tarfile``, ``zipfile`` and ``sqlite3``.
# With ``sys`` importable, a skill could reach every loaded module --
# ``sys.modules['os'].system(...)`` -- and with ``os`` importable it did
# not even need that.  The denylist below could not catch either,
# because both are ordinary imports of allowed names.  An allowlist that
# admits the escape hatches is not an allowlist.
#
# ``io`` and ``types`` were on this list until October 28, 2026, and both
# were escape hatches of the same kind as ``os``: ``io.FileIO(path)``
# opens a file without calling ``open()``, so the call screen never sees
# it, and ``types.ModuleType`` / ``types.FunctionType`` build a callable
# out of parts.  A module is on this list only if *every* name it exports
# is pure computation, and those two do not qualify.
STDLIB_ALLOWED = frozenset({
    "argparse", "base64", "bisect", "collections", "colorsys", "copy",
    "csv", "dataclasses", "datetime", "decimal", "difflib", "enum",
    "fnmatch", "fractions", "functools", "hashlib", "heapq", "html",
    "itertools", "json", "math", "numbers", "operator", "pprint",
    "random", "re", "secrets", "statistics", "string", "textwrap",
    "time", "typing", "unicodedata", "uuid", "warnings",
})

# The standard library modules that reach outside the process or the
# filesystem, with the reason a skill may not have them.  Named
# explicitly so a refusal can say *what* the skill was reaching for
# rather than just "not allowed".
FORBIDDEN_MODULES = {
    # the process and its siblings
    "os": "reaches the filesystem and the process table",
    "sys": "reaches every loaded module, including os",
    "subprocess": "starts a process",
    "multiprocessing": "starts a process",
    "threading": "starts a thread",
    "asyncio": "starts an event loop",
    "signal": "reaches the signal table",
    "pty": "opens a terminal",
    "ctypes": "calls into native code",
    "mmap": "maps memory",
    "resource": "reaches process limits",
    "atexit": "runs code at interpreter exit",
    # the filesystem
    "pathlib": "reaches the filesystem",
    "shutil": "moves and removes files",
    "tempfile": "creates files",
    "glob": "walks the filesystem",
    "fileinput": "reads files",
    "linecache": "reads files",
    "pickle": "deserializes into arbitrary objects",
    "marshal": "deserializes into arbitrary objects",
    "shelve": "reads and writes a database file",
    "sqlite3": "reads and writes a database file",
    "tarfile": "reads and writes archives",
    "zipfile": "reads and writes archives",
    "gzip": "reads and writes files",
    "bz2": "reads and writes files",
    "lzma": "reads and writes files",
    "dbm": "reads and writes a database file",
    "io": "opens files without calling open()",
    # objects built out of parts
    "types": "builds a callable or a module out of parts",
    # the network
    "socket": "opens a socket",
    "socketserver": "opens a socket",
    "http": "makes an HTTP request",
    "urllib": "fetches a URL",
    "ftplib": "opens a connection",
    "smtplib": "sends mail",
    "telnetlib": "opens a connection",
    "webbrowser": "opens a browser",
    "ssl": "opens a connection",
    # the import machinery itself
    "importlib": "imports arbitrary code",
    "imp": "imports arbitrary code",
    "pkgutil": "imports arbitrary code",
    "runpy": "runs arbitrary code",
    "code": "runs arbitrary code",
    "codeop": "compiles arbitrary code",
    "builtins": "reaches the builtin namespace",
    "__builtin__": "reaches the builtin namespace",
    # introspection that turns an object into a capability
    "gc": "reaches every tracked object",
    "inspect": "reaches frames and source",
    "traceback": "reaches frames",
    "platform": "reaches the host",
}

# Calls a skill may not make, whatever module it imported them from.
#
# These are the builtins that turn *data* into *code* or into a
# capability: they are how a skill written as arithmetic becomes a
# program.  ``open`` is here for the same reason ``os`` is: a skill that
# reads a path can read the holdout, and the path screen below is a text
# screen, so it is the second line and not the first.
FORBIDDEN_CALLS = {
    "__import__": "imports a module by name",
    "breakpoint": "enters the debugger",
    "compile": "compiles code",
    "delattr": "removes an attribute",
    "eval": "evaluates code",
    "exec": "runs code",
    "exit": "exits the process",
    "globals": "reaches the module namespace",
    "input": "reads from the terminal",
    "locals": "reaches the local namespace",
    "open": "opens a file",
    "quit": "exits the process",
    "setattr": "sets an attribute",
    "vars": "reaches the object namespace",
}

# ``getattr`` is not forbidden -- it is how a skill reads an optional
# field -- but it is only allowed with a literal name, checked in
# :func:`_ast_problems`.  ``getattr(m, 'sys' + 'tem')`` is how the
# escape in the audit was written: every regex below sees two harmless
# string fragments, and the attribute that gets reached is ``system``.
DYNAMIC_GETATTR = "getattr"

# The names that *are* the escape, whatever module they hang off.
#
# The import screen checks a skill's imports; the attribute rule checks
# dunders and ``sys.modules``.  Neither looks at an ordinary attribute
# with an ordinary name, and that is the hole the third round closed:
# every module on :data:`STDLIB_ALLOWED` is a stdlib module, and stdlib
# modules import each other.  ``uuid`` imports ``os`` and keeps it as
# ``uuid.os``; ``random`` keeps it as ``random._os``; ``warnings``,
# ``typing``, ``enum``, ``fractions`` and ``statistics`` keep ``sys``;
# ``json`` keeps ``codecs``, which keeps ``builtins``.  So
#
#     import uuid
#     def run(): uuid.os.system("...")            # os.system, one dot away
#     import random
#     def run(): random._os.system("...")         # and again, spelled _os
#     import operator, uuid
#     def run(): operator.attrgetter("os")(uuid).system("...")
#
# all import an allowed module and call an allowed name, and the first
# two were confirmed to execute a command through ``load_skills``.  The
# third is the same reach through a callable.
#
# The rule is about the *attribute reached off an imported module*, not
# about the attribute name on its own.  ``uuid.os`` is a module reached
# off a module; ``items.remove`` is a method on the skill's own list and
# is none of this screen's business.  Scoping to the module root is what
# keeps the rule from refusing ordinary code -- ``json.loads`` and
# ``list.remove`` are both written every day, and a screen that refused
# them would be switched off by the first person who hit it.
FORBIDDEN_ATTRIBUTES = {
    # the escape itself, by the name it is reached under
    "system": "runs a shell command",
    "popen": "runs a shell command",
    "spawnv": "starts a process",
    "spawnl": "starts a process",
    "spawnve": "starts a process",
    "execv": "replaces the process image",
    "execve": "replaces the process image",
    "execl": "replaces the process image",
    "execvp": "replaces the process image",
    "fork": "starts a process",
    "forkpty": "starts a process",
    "kill": "signals a process",
    "killpg": "signals a process",
    "remove": "removes a file",
    "unlink": "removes a file",
    "rmdir": "removes a directory",
    "removedirs": "removes a directory",
    "rmtree": "removes a tree",
    "chmod": "changes permissions",
    "chown": "changes ownership",
    "symlink": "creates a link",
    "urlopen": "fetches a URL",
    "socket": "opens a socket",
    "getenv": "reads the environment",
    "environ": "reads the environment",
}

# Attribute names that are a *forbidden module* being reached as an
# attribute.  ``uuid.os`` and ``json.codecs.builtins`` are this shape,
# and ``random._os`` is the same shape with the private spelling.
_FORBIDDEN_MODULE_NAMES = frozenset(FORBIDDEN_MODULES) | {"_os", "_sys"}

# The dunder *names* a skill may write as a bare name or as a string.
# ``__name__`` is module metadata and is what the curator's own test
# template reads; ``__doc__`` and ``__all__`` are documentation.  Every
# other dunder is a way into a namespace: ``__builtins__`` is the
# builtin table, ``__globals__`` a module's, ``__class__`` an object's.
#
# This is a separate list from the *attribute* rule below because the
# two spellings are different attacks.  ``x.__class__`` is an attribute
# chain; ``__builtins__['open']`` is a bare name, and a screen that only
# looks at attributes never sees it.  The audit's payload was written as
# an attribute chain; the same reach written as a name walked past the
# first version of this screen.
ALLOWED_DUNDERS = frozenset({"__name__", "__main__", "__doc__", "__all__",
                             "__version__", "__init__"})

# How a dunder is recognised inside a *string literal*.  A skill can
# reach the same namespace without writing an attribute at all:
# ``'{0.__globals__}'.format(f)`` asks ``str.format`` to walk the chain,
# and ``operator.attrgetter('__globals__')(f)`` does the same through a
# callable.  Both spell the dunder as data, so the attribute rule cannot
# see it and the call rule has nothing to check.  Any dunder in a string
# literal is refused, other than the ones in :data:`ALLOWED_DUNDERS`.
_DUNDER_IN_TEXT_RE = re.compile(r"__[A-Za-z][A-Za-z0-9_]*__")

# The source patterns that mean the skill is doing something a skill has
# no business doing.  These are a *backstop*: the AST screen in
# :func:`_ast_problems` is the real check, because it sees structure
# rather than text.  A pattern here catches the case where the AST was
# built out of a string the screen never parsed.
FORBIDDEN_PATTERNS = (
    (re.compile(r"\b__import__\b"), "calls __import__"),
    (re.compile(r"\beval\s*\("), "calls eval()"),
    (re.compile(r"\bexec\s*\("), "calls exec()"),
    (re.compile(r"\bcompile\s*\("), "calls compile()"),
    (re.compile(r"\bglobals\s*\(\s*\)"), "mutates globals()"),
    (re.compile(r"\bvars\s*\(\s*\)"), "mutates vars()"),
    (re.compile(r"\blocals\s*\(\s*\)"), "reads locals()"),
    (re.compile(r"\bgetattr\s*\(\s*__"), "reaches for a dunder via getattr"),
    (re.compile(r"\bgetattr\s*\([^)]*\+"), "builds an attribute name at runtime"),
    (re.compile(r"\bpickle\.loads?\b"), "deserializes with pickle"),
    (re.compile(r"\bmarshal\.loads?\b"), "deserializes with marshal"),
    (re.compile(r"\bos\.system\b"), "calls os.system()"),
    (re.compile(r"\bos\.popen\b"), "calls os.popen()"),
    (re.compile(r"\bos\.exec[lv]p?\b"), "calls os.exec*()"),
    (re.compile(r"\bsubprocess\."), "uses subprocess"),
    (re.compile(r"\bsocket\."), "opens a socket"),
    (re.compile(r"\burlopen\b"), "fetches a URL"),
    (re.compile(r"\brequests\."), "makes an HTTP request"),
    (re.compile(r"\bshutil\.rmtree\b"), "removes a tree"),
    (re.compile(r"\bos\.remove\b"), "removes a file"),
    (re.compile(r"\bos\.unlink\b"), "unlinks a file"),
    (re.compile(r"\bPath\([^)]*\)\.unlink\b"), "unlinks a file"),
    (re.compile(r"\bholdout\b"), "mentions the holdout"),
    (re.compile(r"\beval_dir\b"), "mentions the eval directory"),
    (re.compile(r"\bbaselines?\b"), "mentions the baselines"),
    (re.compile(r"\bpinned\.yaml\b"), "mentions the pin file"),
)

# The paths a skill may not reach.  Compared against the *resolved*
# path, so a symlink or a ``..`` does not get around it.
FORBIDDEN_PATH_PARTS = (
    "eval", "baselines", "holdout",
)

# The check names, in the order the gate runs them.  The order matters
# only for the message a failure produces: the cheapest check that can
# explain the failure should run first.
CHECK_MANIFEST = "manifest"
CHECK_SOURCE = "source"
CHECK_DEPS = "dependencies"
CHECK_TESTS = "tests"
CHECK_REGRESSION = "regression"

ALL_CHECKS = (CHECK_MANIFEST, CHECK_SOURCE, CHECK_DEPS, CHECK_TESTS,
              CHECK_REGRESSION)


class GateResult:
    """The outcome of the gate: what passed, what failed, and why.

    A plain dict would do, but a result is read by a human, a CLI and a
    test, and the three want slightly different things.  This gives them
    one object with the obvious accessors.
    """

    def __init__(self, passed: bool = True, checks: dict | None = None):
        self.passed = bool(passed)
        # checks: {name: {"passed": bool, "reason": str, "detail": dict}}
        self.checks = checks or {}

    def add(self, name: str, passed: bool, reason: str = "",
            detail: dict | None = None) -> None:
        self.checks[name] = {"passed": bool(passed), "reason": reason,
                             "detail": detail or {}}
        if not passed:
            self.passed = False

    @property
    def failures(self) -> list:
        """The failed checks, in the order they ran."""
        return [name for name in ALL_CHECKS
                if name in self.checks and not self.checks[name]["passed"]]

    def reason(self) -> str:
        """Why the gate said no, in one line."""
        if self.passed:
            return ""
        parts = []
        for name in self.failures:
            entry = self.checks[name]
            parts.append("%s: %s" % (name, entry.get("reason") or "failed"))
        return "; ".join(parts)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks,
                "failures": self.failures, "reason": self.reason()}

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        return "<GateResult passed=%s failures=%s>" % (self.passed,
                                                       self.failures)


def check_manifest(manifest: dict | None) -> tuple:
    """Check 1: the manifest is well formed.

    Returns ``(passed, reason)``.
    """
    problems = manifest_errors(manifest)
    if problems:
        return False, "; ".join(problems)
    return True, ""


def _imports(tree: ast.AST) -> list:
    """Every module name the source imports, as dotted strings."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import
                names.append("." * node.level + (node.module or ""))
            elif node.module:
                names.append(node.module)
    return names


def _call_name(node: ast.AST) -> str:
    """The bare name a call goes through: ``eval`` in ``eval(x)``.

    Returns "" for anything that is not a plain name -- ``obj.method()``
    and ``f()()`` both return "".  The screen wants the builtin by name,
    and a call whose name is computed is caught by its own rule.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return ""


# The callables whose whole job is to fetch an attribute by name.  A
# literal handed to one of them is an attribute reference that no
# ``ast.Attribute`` node appears in, so the attribute rule cannot see it.
_ATTRIBUTE_GETTERS = frozenset({"getattr", "attrgetter", "itemgetter",
                                "methodcaller"})


def _reaches_from_module(node: ast.AST, modules: set) -> bool:
    """Whether ``node`` is (or starts from) a module the skill imported.

    ``uuid.os`` reaches from the imported ``uuid``; ``items.remove`` does
    not, and must not be refused by the rule that catches ``uuid.os``.
    A chain is walked through attributes and subscripts, because
    ``json.codecs.builtins`` is three levels deep and
    ``random._os.path`` is two.
    """
    while True:
        if isinstance(node, ast.Name):
            return node.id in modules
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return False


def _literal_str(node: ast.AST) -> str | None:
    """The value of a string literal node, or None when it is computed.

    Only the *literal* counts.  ``getattr(m, "system")`` names its
    attribute in the source; ``getattr(m, "sys" + "tem")`` does not, and
    the difference is exactly the difference between a skill reading an
    optional field and a skill assembling a capability out of pieces
    that no text screen can see through.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _attribute_chain(node: ast.AST) -> list:
    """The attribute names a value is reached through, outermost last.

    ``sys.modules['os'].system`` gives ``["system"]`` on a subscript,
    and ``a.b.c`` gives ``["b", "c"]``.  Used to notice a *dunder*
    reached through anything at all -- ``().__class__.__bases__`` is how
    a sandbox is escaped without importing a single forbidden module.
    """
    chain = []
    while True:
        if isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            break
    return chain


def _is_super_init(node: ast.Attribute) -> bool:
    """Whether this dunder attribute is ``super().__init__``.

    The one dunder a skill legitimately reaches: a skill that subclasses
    a ``dataclass`` or an exception has to call its parent's
    initialiser, and the parent is named in the skill's own source.
    """
    return (node.attr == "__init__"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "super")


def _ast_problems(tree: ast.AST) -> list:
    """Everything structurally wrong with the source, as messages.

    This is the screen the gate is actually built on, and the reason it
    is structural rather than textual is the audit's B1: a skill that
    writes ``sys.modules['o' + 's']`` imports only ``sys``, calls only
    ``getattr``, and every regex over its *text* sees two harmless
    string fragments.  The AST sees a ``getattr`` whose attribute name
    is not a literal, and a subscript on ``sys.modules``, and refuses
    both without needing to know what the fragments spell.

    The rules, in the order a failure reads best:

    * a forbidden call, by bare name -- ``eval``, ``exec``, ``open``,
      ``__import__``, ``globals``, ``setattr`` and the rest of
      :data:`FORBIDDEN_CALLS`;
    * a forbidden builtin used as a *value* rather than called --
      ``f = open``, ``def run(f=open)``, ``[open][0](path)``.  The call
      rule only sees ``open(``; a skill that passes ``open`` around and
      calls it through another name calls it without ever writing the
      name in front of a bracket;
    * a ``getattr`` whose attribute name is computed rather than
      written;
    * an attribute or subscript that reaches a dunder, anywhere;
    * a *bare name* that is a dunder -- ``__builtins__['open']`` needs
      no attribute chain and no import;
    * a dunder spelled inside a string literal -- ``'{0.__globals__}'``
      and ``attrgetter('__globals__')`` are the same reach written as
      data;
    * a bare name that is a forbidden module used as a value --
      ``sys.modules`` when ``sys`` was imported under another name.
    """
    problems = []
    aliases = {}  # local name -> the module it was imported from
    # The local names that hold a *module object*: the module itself
    # under ``import x``, or the alias under ``import x as y``.  The
    # attribute rule below needs this to tell ``uuid.os`` (a module off a
    # module) from ``items.remove`` (a method off a list), and the two
    # must not be refused by the same rule.
    modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                modules.add(alias.asname or root)
                if alias.asname:
                    aliases[alias.asname] = root
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if node.module:
                    modules.add(alias.asname or alias.name)
                if alias.asname and node.module:
                    aliases[alias.asname] = node.module.split(".")[0]

    # Every docstring node, so the string screen can tell prose from a
    # format string.  A docstring is the first statement of a module,
    # class or function and is a bare string expression.
    docstrings = set()
    for owner in ast.walk(tree):
        if isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef)):
            body = getattr(owner, "body", None) or []
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))

    # The names that are *called* are the call rule's business; a
    # forbidden builtin that is not called is a forbidden builtin being
    # passed around, which is the same capability one step removed.
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(id(node.func))

    for node in ast.walk(tree):
        # A forbidden builtin used as a *value* rather than called.
        # ``f = open``, ``def run(f=open)``, ``[open][0](path)``: the
        # call screen only sees a name in front of a bracket, and a
        # skill that passes ``open`` around calls it through another
        # name.  The capability is the same one.
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS \
                and id(node) not in called:
            problems.append("uses %s as a value rather than calling it, "
                            "which %s" % (node.id, FORBIDDEN_CALLS[node.id]))

        # A bare dunder *name*.  ``__builtins__['open']`` needs no
        # import and no attribute chain, so neither the import screen
        # nor the attribute rule sees it.  The allowed set is module
        # metadata only -- see :data:`ALLOWED_DUNDERS`.
        if isinstance(node, ast.Name) and node.id.startswith("__") \
                and node.id.endswith("__") \
                and node.id not in ALLOWED_DUNDERS:
            problems.append("reaches the dunder name %r" % node.id)

        # A dunder spelled inside a string literal.  ``'{0.__globals__}'
        # .format(f)`` asks ``str.format`` to walk the chain, and
        # ``attrgetter('__globals__')(f)`` does the same through a
        # callable; both write the dunder as data, where no attribute
        # rule can see it.
        #
        # A *docstring* is exempt.  Prose that says the word
        # ``__globals__`` is not a capability, and refusing it would
        # teach the curator's own prompt that a comment is dangerous --
        # the screen has to be about what runs, not about what is
        # written down.
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            for found in _DUNDER_IN_TEXT_RE.findall(node.value):
                if found not in ALLOWED_DUNDERS:
                    problems.append(
                        "spells the dunder %r inside a string, which "
                        "reaches the same attribute the attribute rule "
                        "refuses" % found)
                    break

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in FORBIDDEN_CALLS:
                problems.append("calls %s(), which %s"
                                % (name, FORBIDDEN_CALLS[name]))
            if name == DYNAMIC_GETATTR:
                # getattr(obj, "literal") is a field read; getattr(obj,
                # anything-else) is a capability assembled at runtime.
                if len(node.args) >= 2:
                    literal = _literal_str(node.args[1])
                    if literal is None:
                        problems.append(
                            "calls getattr() with a name that is not a "
                            "literal, so the attribute it reaches cannot "
                            "be read from the source")
                    elif literal.startswith("__") and literal.endswith("__"):
                        # A *written* dunder is no better than a computed
                        # one: getattr(m, "__globals__") reaches the same
                        # namespace that writing m.__globals__ does, and
                        # the attribute-chain rule above cannot see it
                        # because the name is a string, not an attribute.
                        problems.append(
                            "calls getattr() for the dunder %r" % literal)

        # A dunder reached through any chain: the sandbox escape that
        # needs no import at all.  ``().__class__.__bases__`` gets from
        # an object to ``object`` to every loaded subclass without an
        # import and without a call to anything on a denylist.
        #
        # ``super().__init__()`` is the one legitimate spelling, and it
        # is allowed by name: a skill that subclasses something needs it,
        # and it reaches a base class the skill itself declared.
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") \
                and node.attr.endswith("__"):
            if not _is_super_init(node):
                problems.append("reaches the dunder attribute %r" % node.attr)

        # sys.modules[...] -- the module table, by any spelling.
        if isinstance(node, ast.Subscript):
            chain = _attribute_chain(node.value)
            if chain and chain[0] == "modules":
                problems.append(
                    "subscripts .modules, which is the table of every "
                    "loaded module")

        # A forbidden module used as a value, under whatever local name.
        if isinstance(node, ast.Name) and node.id in aliases:
            root = aliases[node.id]
            if root in FORBIDDEN_MODULES:
                problems.append("uses %r, which is %s"
                                % (node.id, FORBIDDEN_MODULES[root]))

        # The third round: an escape reached as an *attribute* off an
        # allowed module.  See :data:`FORBIDDEN_ATTRIBUTES`.
        #
        # Every allowed module is stdlib, and stdlib modules import each
        # other, so ``uuid.os``, ``random._os`` and ``json.codecs`` are
        # all one dot from the process.  The import screen never saw them
        # because the import was ``import uuid``; the dunder rule never
        # saw them because ``os`` is not a dunder.  This rule reads the
        # attribute instead of the import, which is the level at which
        # the capability actually appears.
        if isinstance(node, ast.Attribute):
            bare = node.attr.lstrip("_")
            if node.attr in _FORBIDDEN_MODULE_NAMES \
                    or bare in _FORBIDDEN_MODULE_NAMES:
                if _reaches_from_module(node.value, modules):
                    problems.append(
                        "reaches %r off an imported module, which is the "
                        "%r module by another name" % (node.attr, bare))
            # The capability names are scoped to a module root for the
            # same reason: ``xs.remove(1)`` is a list method, and
            # ``uuid.os.remove(p)`` is a deleted file.  Only the second
            # is this screen's business, and the difference is the root.
            if bare in FORBIDDEN_ATTRIBUTES \
                    and _reaches_from_module(node.value, modules):
                problems.append("reaches the attribute %r, which %s"
                                % (node.attr, FORBIDDEN_ATTRIBUTES[bare]))

        # ``getattr(uuid, "os")`` and ``operator.attrgetter("os")(uuid)``
        # name the attribute as *data*, so the rule above cannot see it.
        # A literal naming a forbidden attribute or module, handed to a
        # callable whose whole job is to fetch an attribute by name, is
        # the same reach written in a form no Attribute node appears in.
        if isinstance(node, ast.Call):
            target = _call_name(node)
            if target in _ATTRIBUTE_GETTERS:
                for arg in node.args[1 if target == "getattr" else 0:]:
                    literal = _literal_str(arg)
                    if literal is None:
                        continue
                    bare = literal.lstrip("_")
                    if bare in FORBIDDEN_ATTRIBUTES \
                            or bare in _FORBIDDEN_MODULE_NAMES:
                        problems.append(
                            "reaches %r by name through %s(), which is "
                            "the attribute rule written as data"
                            % (literal, target))

    # Deduplicate, and keep the order the rules ran in: the first
    # message is the one an operator reads.
    seen = set()
    unique = []
    for problem in problems:
        if problem not in seen:
            seen.add(problem)
            unique.append(problem)
    return unique


# A path part is mentioned when it appears as a whole path component --
# bounded by a separator or the edge of the string -- and not when it is
# merely a substring of a longer word.
#
# The earlier version of this screen matched the part only when a quote
# came immediately before it, so `open('/x/eval/dev.jsonl')` and
# `open('../eval/dev.jsonl')` both slipped through: the separator, not a
# quote, is what precedes the directory name in a real path.  The screen
# has to see the *directory*, not the literal.
_PATH_PART_RE = (
    r"(?:^|[/\\\s'\"`(=]){part}(?:[/\\\s'\"`).,]|$)")


def _mentions_path_part(part: str, source: str) -> bool:
    """Whether ``source`` refers to ``part`` as a path component.

    ``eval`` matches ``'/x/eval/dev.jsonl'`` and ``'eval/x'`` and
    ``"open(eval_dir())"``; it does not match ``'myeval/x'`` or
    ``'evaluate'``, which are ordinary words.
    """
    if not part:
        return False
    pattern = _PATH_PART_RE.format(part=re.escape(part))
    return bool(re.search(pattern, source))


def check_source(source: str, *, forbidden_paths: list | None = None) -> tuple:
    """Check 2: the source is clean.

    Four screens, in order of what they catch:

    * it must parse -- a skill whose source does not parse is a skill
      that cannot be loaded;
    * its imports must be on the allowlist -- pure computation only,
      because a skill that needs the filesystem or a process is a skill
      that should have been a tool call, not a tool;
    * its *structure* must be clean -- no forbidden builtin called, no
      attribute name assembled at runtime, no dunder reached, no
      ``sys.modules`` subscript.  This is the screen that matters; the
      text patterns below are the backstop for the case where the
      source that gets compiled is not the source that was parsed;
    * its text must not name the ruler, or a path outside the agent's
      writable surface.

    Returns ``(passed, reason, detail)``.
    """
    detail = {}
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, "the source does not parse: %s" % e, detail

    imports = _imports(tree)
    detail["imports"] = imports

    outside = []
    for name in imports:
        root = name.lstrip(".").split(".")[0]
        if not root:
            continue
        if root in FORBIDDEN_MODULES:
            outside.append("%s (%s)" % (name, FORBIDDEN_MODULES[root]))
        elif root not in STDLIB_ALLOWED:
            outside.append("%s (not on the allowed list: a skill computes, "
                           "it does not reach outside its arguments)" % name)
    if outside:
        return False, ("the source imports %s" % "; ".join(outside)), detail

    problems = _ast_problems(tree)
    detail["structure"] = problems
    if problems:
        return False, ("the source %s" % problems[0]), detail

    for pattern, why in FORBIDDEN_PATTERNS:
        found = pattern.search(source)
        if found:
            return False, ("the source %s: %r" % (why, found.group(0)[:40])), detail

    # The paths a skill may not touch.  Checked against the *source
    # text* as well as the import list, because a skill that opens a
    # path built at runtime never mentions it in an import.
    paths = forbidden_paths or []
    for part in FORBIDDEN_PATH_PARTS:
        if _mentions_path_part(part, source):
            return False, ("the source refers to %r, which is outside the "
                           "agent's writable surface" % part), detail
    for path in paths:
        if path and str(path) in source:
            return False, ("the source refers to %r, which is outside the "
                           "agent's writable surface" % path), detail

    return True, "", detail


def ruler_paths(config_data: dict | None = None) -> list:
    """The paths a skill must not be able to reach.

    The ruler: the holdout, the dev split, the baseline, and the
    directory they live in.  I7 says the measuring instrument is
    outside the agent's writable surface; a skill that could read the
    holdout would be a system that grades its own exam, and a skill
    that could write it would be a system that edits it.
    """
    from .eval import baseline_path, dev_path, eval_dir, holdout_path

    return [eval_dir(config_data), holdout_path(config_data),
            dev_path(config_data), baseline_path(config_data)]


def check_dependencies(manifest: dict | None, *, sealed: bool = False) -> tuple:
    """Check 3: the dependencies are permitted.

    A skill that needs a package which is not installed fails at the
    worst possible moment -- in the middle of the agent's work, with
    the tool already selected.

    When installs are sealed, a new dependency is refused outright,
    because there is no way to satisfy it and pretending otherwise
    would ship a skill that cannot run.
    """
    deps = (manifest or {}).get("dependencies") or []
    if isinstance(deps, str):
        deps = [deps]
    deps = [str(d) for d in deps if str(d).strip()]
    detail = {"dependencies": deps, "sealed": bool(sealed)}

    if deps and sealed:
        return False, ("the skill needs %s, and package installs are "
                       "sealed in this deployment" % ", ".join(deps)), detail
    return True, "", detail


def check_tests(test_source: str | None) -> tuple:
    """Check 4: the tests are there and they parse.

    The *running* of the tests is the sandbox's job (I6); this check
    only confirms that a test file exists and parses, because a skill
    with no test file is a skill whose behaviour was never checked.

    Returns ``(passed, reason, detail)``.
    """
    detail = {}
    if not test_source or not test_source.strip():
        return False, "there is no test file", detail
    try:
        ast.parse(test_source)
    except SyntaxError as e:
        return False, "the test file does not parse: %s" % e, detail
    if not re.search(r"\bdef test_", test_source):
        return False, "the test file has no test functions", detail
    return True, "", detail


def check_regression(delta: float | None, *, required: bool = True) -> tuple:
    """Check 5: the skill does not regress the benchmark.

    ``delta`` is the change in the quality score on the holdout.  A
    negative delta means the skill made things worse, and a skill that
    makes things worse is not a skill.

    ``required=False`` is for the case where the benchmark has not been
    run: the check then passes with a note rather than pretending the
    skill was measured.  Pretending would be worse than not knowing.
    """
    detail = {"delta": delta, "required": bool(required)}
    if delta is None:
        if required:
            return False, ("no benchmark delta was measured; a skill is "
                           "not admitted on an unmeasured claim"), detail
        return True, "", detail
    if delta < 0:
        return False, ("the benchmark delta is %+.3f; a skill that makes "
                       "things worse is not a skill" % delta), detail
    return True, "", detail


def run_gate(*, manifest: dict | None = None, source: str | None = None,
             test_source: str | None = None, delta: float | None = None,
             sealed: bool = False, require_benchmark: bool = True,
             forbidden_paths: list | None = None) -> GateResult:
    """Run all five checks and return the result.

    The checks are conjunctive, and each one runs even if an earlier one
    failed: the operator wants to know everything that is wrong, not the
    first thing.
    """
    result = GateResult()

    ok, reason = check_manifest(manifest)
    result.add(CHECK_MANIFEST, ok, reason, {"problems": reason} if reason else {})

    ok, reason, detail = check_source(source or "", forbidden_paths=forbidden_paths)
    result.add(CHECK_SOURCE, ok, reason, detail)

    ok, reason, detail = check_dependencies(manifest, sealed=sealed)
    result.add(CHECK_DEPS, ok, reason, detail)

    ok, reason, detail = check_tests(test_source)
    result.add(CHECK_TESTS, ok, reason, detail)

    ok, reason, detail = check_regression(delta, required=require_benchmark)
    result.add(CHECK_REGRESSION, ok, reason, detail)

    return result


def read_skill_sources(directory: str) -> tuple:
    """Read a skill's source and test file.

    Returns ``(source, test_source, error)``.  The error is a message
    rather than an exception: a skill directory that is half-written is
    an expected state, not a crash.
    """
    source = None
    test_source = None
    error = ""
    try:
        with io.open(os.path.join(directory, "tool.py"), "r",
                     encoding="utf-8") as f:
            source = f.read()
    except OSError:
        error = "there is no tool.py"
    try:
        name = os.path.basename(directory.rstrip("/"))
        with io.open(os.path.join(directory, "test_%s.py" % name), "r",
                     encoding="utf-8") as f:
            test_source = f.read()
    except OSError:
        if not error:
            error = "there is no test file"
    return source, test_source, error
