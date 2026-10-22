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

# The imports a skill may use.  The standard library only: a skill that
# needs a package is a skill that needs an install, and an install is
# an operator decision, not a loop decision.
STDLIB_ALLOWED = frozenset({
    "argparse", "base64", "collections", "copy", "csv", "dataclasses",
    "datetime", "difflib", "enum", "fnmatch", "fractions", "functools",
    "glob", "gzip", "hashlib", "heapq", "html", "http", "inspect", "io",
    "itertools", "json", "math", "os", "pathlib", "pickle", "platform",
    "pprint", "random", "re", "shlex", "shutil", "socket", "sqlite3",
    "statistics", "string", "subprocess", "sys", "tarfile", "tempfile",
    "textwrap", "time", "types", "typing", "unicodedata", "unittest",
    "urllib", "uuid", "warnings", "xml", "zipfile",
})

# The standard library modules that reach outside the process.  A skill
# that imports one of these is asking for a capability the loop should
# not hand out on its own.
FORBIDDEN_MODULES = frozenset({
    "ctypes", "multiprocessing", "socketserver", "telnetlib", "webbrowser",
    "http.server", "ftplib", "smtplib", "asyncio", "signal", "pty",
})

# The source patterns that mean the skill is doing something a skill has
# no business doing.  These are a *backstop* -- the import and AST
# checks below are the real screen -- because a string built at runtime
# can hide anything.
FORBIDDEN_PATTERNS = (
    (re.compile(r"\b__import__\b"), "calls __import__"),
    (re.compile(r"\beval\s*\("), "calls eval()"),
    (re.compile(r"\bexec\s*\("), "calls exec()"),
    (re.compile(r"\bcompile\s*\("), "calls compile()"),
    (re.compile(r"\bglobals\s*\(\s*\)"), "mutates globals()"),
    (re.compile(r"\bvars\s*\(\s*\)"), "mutates vars()"),
    (re.compile(r"\bgetattr\s*\(\s*__"), "reaches for a dunder via getattr"),
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

    Three screens, in order of what they catch:

    * it must parse -- a skill whose source does not parse is a skill
      that cannot be loaded;
    * its imports must be standard library, and not one of the modules
      that reach outside the process;
    * its text must not contain the patterns that mean it is doing
      something it has no business doing.

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
            outside.append("%s (reaches outside the process)" % name)
        elif root not in STDLIB_ALLOWED:
            outside.append("%s (not in the standard library)" % name)
    if outside:
        return False, ("the source imports %s" % "; ".join(outside)), detail

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
