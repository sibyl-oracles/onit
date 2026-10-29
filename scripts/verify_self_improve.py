#!/usr/bin/env python3
"""Answer three questions about self-improvement, with evidence.

    python3 scripts/verify_self_improve.py

Three questions, and this script answers each one from the code rather
than from a docstring:

1. **Do I need to reinstall to see an effect?**  No, if the install is
   editable.  This prints where ``src`` is imported from and whether the
   tree on disk is the one being imported.

2. **Does self-improvement add anything to the codebase?**  No.  Every
   loop writes to the learned store, never to the package.  This asserts
   that by writing a skill and checking where it landed.

3. **How do I verify an improvement?**  This runs the three instruments
   that can answer it and prints what each one establishes, in order of
   how much they can carry:

   * ``onit learn --json``      -- was anything recorded at all
   * ``onit learn --playbook``  -- what would be injected
   * ``scripts/measure.py``     -- did an arm score differently

The script never writes outside ``tmp/`` and never touches the real
store.  Exit code is the number of checks that failed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FAILURES = []


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print("  " + text)
    print("=" * 72)


def check(name: str, ok: bool, detail: str = "") -> None:
    print("  [%s] %s" % ("ok  " if ok else "FAIL", name))
    if detail:
        for line in detail.strip().splitlines():
            print("        " + line)
    if not ok:
        FAILURES.append(name)


# ── 1. the install ───────────────────────────────────────────────────────────

def check_install() -> None:
    banner("1. Is the code on disk the code that runs?")

    import src
    import src.learn.config as cfg

    src_file = os.path.abspath(src.__file__)
    in_repo = src_file.startswith(REPO)
    check("src is imported from this working tree", in_repo,
          "imported from: %s\nrepo root:    %s" % (src_file, REPO))

    # The metadata is not a reliable test: this install writes a
    # ``__editable__*.pth`` finder and records the *mapping* in
    # ``dist.files``, so "does the distribution list src/__init__.py" is
    # true for an editable install too.  The behavioural checks above are
    # the evidence: the import resolves to this tree, and a symbol defined
    # only here is live.  A non-editable install fails both.

    # The strongest evidence: a symbol that only exists in the current tree.
    live = hasattr(cfg, "_MAX_IMPLEMENTED")
    check("a symbol from the current tree is live", live,
          "learn.config._MAX_IMPLEMENTED = %r"
          % getattr(cfg, "_MAX_IMPLEMENTED", None))

    print("""
  A non-editable install (``pip install onit``) would need
  ``pip install -e .`` again after a code change.  An editable one
  picks the change up on the next process, because the import resolves
  to this directory rather than to a copy in site-packages.""")


# ── 2. what self-improvement writes ──────────────────────────────────────────

def check_writes() -> None:
    banner("2. Does self-improvement add anything to the codebase?")

    from src.learn import skills_curator
    from src.learn.skills import skills_dir

    store = tempfile.mkdtemp(prefix="verify_store_", dir=os.path.join(REPO, "tmp"))
    cfg = {"learn": {"path": store, "autonomy": "extend"}}

    candidate = {"tool": "bash", "digest": "date -d X", "sessions": 4,
                 "calls": 7, "arg_keys": ["cmd"]}
    written = skills_curator.synthesize(candidate, name="verify_skill",
                                        config_data=cfg)

    inside_store = written["dir"].startswith(skills_dir(cfg))
    inside_pkg = os.path.abspath(os.path.join(REPO, "src")) in written["dir"]
    check("a generated skill lands in the store", inside_store,
          "written to: %s" % written["dir"])
    check("a generated skill does not land in the package", not inside_pkg,
          "package dir: %s" % os.path.join(REPO, "src"))

    # Nothing in src/learn writes relative to __file__.
    import re
    offenders = []
    learn_dir = os.path.join(REPO, "src", "learn")
    for name in sorted(os.listdir(learn_dir)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(learn_dir, name)
        with open(path, encoding="utf-8", errors="replace") as fh:
            for number, line in enumerate(fh, 1):
                if "__file__" in line and re.search(r"open\(.*['\"]w", line):
                    offenders.append("%s:%d" % (name, number))
    check("no learn module writes next to its own source", not offenders,
          "offenders: %s" % (offenders or "none"))

    from src.learn.config import EVOLVE, _MAX_IMPLEMENTED
    check("the rung that would edit source is not implemented",
          _MAX_IMPLEMENTED < EVOLVE,
          "highest implemented rung is %d; 'evolve' is %d"
          % (_MAX_IMPLEMENTED, EVOLVE))

    print("""
  Every loop is delta-only and lives outside the package:

    ~/.onit/learned/trajectories/   one JSONL per session
    ~/.onit/learned/playbook/       global.json, versioned
    ~/.onit/learned/skills/         <name>/{manifest.yaml,tool.py,test_*.py}
    ~/.onit/learned/archive/        what the lifecycle loop withheld

  ``learn.path`` (or ONIT_LEARN_PATH) moves all of it.  A skill is code,
  but it is code loaded *from the store at runtime*, gated on every
  serve -- it is not a file in this repository, and ``git status`` stays
  clean after any amount of learning.""")


# ── 3. the instruments ───────────────────────────────────────────────────────

def check_instruments() -> None:
    banner("3. What can each instrument tell you?")

    def run(args):
        result = subprocess.run([sys.executable, "-m", "src.cli"] + args,
                                capture_output=True, text=True, cwd=REPO)
        return result.returncode, result.stdout, result.stderr

    code, out, err = run(["learn", "--json"])
    ok = code == 0 and out.strip().startswith("{")
    detail = ""
    if ok:
        try:
            data = json.loads(out)
            detail = ("tasks recorded: %s across %s session(s)"
                      % (data.get("tasks"), data.get("sessions")))
        except Exception:
            detail = out[:200]
    check("the store can be read (claim 1: was it recorded)", ok,
          detail or (err or out)[:300])

    code, out, _ = run(["learn", "--playbook"])
    check("the playbook can be rendered (claim 2: what is injected)",
          code == 0 and "Playbook" in out, out.strip()[:300])

    print("""
  Three claims, and the instrument that can carry each:

    claim 1  the run was recorded      ``onit learn --json``
    claim 2  the next run saw it       the trajectory's learned_context
    claim 3  that changed the outcome  ``scripts/measure.py``, two arms

  Only claim 3 is evidence that the agent got better.  It needs a
  holdout the loop never saw, and a control arm:

    python3 scripts/measure.py --config CFG --tasks HOLDOUT.jsonl \\
        --autonomy off   --out baseline.json
    python3 scripts/measure.py --config CFG --tasks HOLDOUT.jsonl \\
        --autonomy adapt --store FROZEN --out treatment.json

  Read ``errors`` before ``accuracy`` in both files.  A run whose items
  errored did not measure the agent, and a delta between two error
  counts is not a result.""")


def main() -> int:
    check_install()
    check_writes()
    check_instruments()

    banner("Result")
    if FAILURES:
        print("  %d check(s) failed:" % len(FAILURES))
        for name in FAILURES:
            print("    - %s" % name)
        return len(FAILURES)
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
