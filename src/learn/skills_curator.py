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

Loop C, part three: the curator.  Turning a candidate into a skill.

The Reflector is :mod:`src.learn.skills_mining` -- it finds the
fragments.  The Curator is here: it takes a candidate, writes the
skill, runs the gate, and pins it if the gate says yes.

**The order is the safety property.**  A skill is written, gated, and
only then pinned -- in that order, every time.  A skill that fails the
gate exists on disk and is not in the pin, so it is not served; a skill
that is pinned without passing the gate is served without having been
checked, and that is the one state this code must never reach.  The
pin is written last, and only on a green gate.

**The pin is the allowlist.**  `pinned.yaml` names the skills that may
be served, and the MCP server serves exactly those.  Everything else in
the learned store -- drafts, candidates, skills that failed -- is
invisible to the agent, and there is no path from "exists on disk" to
"in the model's schema" except through the gate.

**Nothing is mutated in place (I3).**  The previous pin is copied aside
before the new one is written, so a promotion is undone by restoring a
file rather than by reconstructing what the file said.  That is what
makes an unattended promotion safe: the worst case is a rollback.
"""

import io
import os
import shutil
import time

import yaml

from .skills import (ACTIVE, PROBATION, apply_deltas, read_manifest,
                     read_pin, skill_dir, skills_dir, write_manifest,
                     write_pin)
from .skills_gate import read_skill_sources, run_gate

# The template a synthesised skill starts from.  It is a *starting
# point*, not the skill: the curator writes the body that does the
# thing.  What the template guarantees is the shape the loader expects
# -- a module with a `run` function and a `MANIFEST` -- so that a
# half-written skill is still loadable and still says what it is.
#
# The placeholders are @@-marked rather than brace-marked, because the
# body and the manifest are themselves full of braces, and a brace
# template would mangle them.
SKILL_TEMPLATE = '''"""
@@title

@@description

This file was written by the skill curator from a fragment the agent
kept repeating.  It is a tool the agent gave itself.
"""

MANIFEST = @@manifest


def run(**kwargs):
    """Run the skill."""
@@body
'''


def _fill(template: str, **values) -> str:
    """Fill a template, without the template's own syntax getting lost.

    ``str.format`` would choke on the braces in a Python body, and
    ``%`` would choke on the percent signs in one.  A placeholder that
    cannot appear in the code being inserted does neither.
    """
    out = template
    for key, value in values.items():
        out = out.replace("@@" + key, str(value))
    return out


def _now() -> float:
    return time.time()


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _title_for(name: str) -> str:
    """A human title from a tool name."""
    words = str(name or "skill").replace("_", " ").split()
    return " ".join(w[:1].upper() + w[1:] for w in words if w) or "Skill"


def default_manifest(candidate: dict, *, name: str | None = None) -> dict:
    """The manifest a candidate gets when nothing better is known.

    The description is the important field, and the honest version of it
    is the one that says what the fragment *was*: which tool was called,
    with which arguments, how often.  A description written by the
    curator that claims more than the record supports is a description
    that will send the model to the wrong thing.
    """
    name = name or candidate.get("name") or "skill"
    tool = candidate.get("tool") or "a tool"
    arg_keys = candidate.get("arg_keys") or []
    args = ", ".join(arg_keys) if arg_keys else "no arguments"
    return {
        "name": name,
        "description": (
            "Runs the %s call the agent kept repeating (%d sessions, %d "
            "calls), with %s. Synthesised from the trajectory record."
            % (tool, candidate.get("sessions", 0), candidate.get("calls", 0),
               args)),
        "version": 1,
        "input_schema": {
            "type": "object",
            "properties": {key: {"type": "string"} for key in arg_keys},
        },
        "dependencies": [],
        "mined_from": {
            "tool": tool,
            "digest": candidate.get("digest"),
            "sessions": candidate.get("sessions", 0),
            "calls": candidate.get("calls", 0),
        },
    }


def default_body(candidate: dict) -> str:
    """The body a synthesised skill starts with.

    The body cannot be the original fragment -- the trajectory store
    redacts argument values on purpose, so the original call is not
    available to copy.  What the body does is raise, clearly, so that
    the skill *fails loudly* rather than pretending to work.  A skill
    that is honest about being a draft is a skill an operator can
    finish; a skill that silently does nothing is one that gets pinned
    and then trusted.
    """
    tool = candidate.get("tool") or "the recorded tool"
    return (
        "    raise NotImplementedError(\n"
        "        'This skill is a mined draft. It has not been written yet, '\n"
        "        'so it raises rather than pretending to do the ' + {tool!r}\n"
        "        + ' call.')\n").format(tool=tool)


def write_skill(name: str, manifest: dict, body: str, *,
                config_data: dict | None = None) -> str:
    """Write a skill's directory.  Returns the directory.

    The test file is written alongside, because a skill without tests
    cannot pass the gate, and a skill that cannot pass the gate is
    never served -- so writing one without the other would create a
    draft that could never be promoted.
    """
    directory = skill_dir(name, config_data)
    os.makedirs(directory, exist_ok=True)

    title = _title_for(name)
    description = str(manifest.get("description") or "")
    source = _fill(SKILL_TEMPLATE, title=title, description=description,
                   manifest=repr(manifest), body=body)
    with io.open(os.path.join(directory, "tool.py"), "w",
                 encoding="utf-8") as f:
        f.write(source)

    write_manifest(os.path.join(directory, "manifest.yaml"), manifest)

    test_name = "test_%s.py" % name
    test_source = _fill(_TEST_TEMPLATE, name=name, title=title)
    with io.open(os.path.join(directory, test_name), "w",
                 encoding="utf-8") as f:
        f.write(test_source)

    return directory


_TEST_TEMPLATE = '''"""
Tests for the @@title skill.

The gate runs this file.  A skill that cannot pass its own tests is not
served, so these tests are the skill's contract with the agent that
will use it.
"""

import unittest

try:
    import tool
except ImportError:  # pragma: no cover - the loader puts the dir on the path
    import importlib.util
    import os

    _spec = importlib.util.spec_from_file_location(
        "skill_under_test", os.path.join(os.path.dirname(__file__), "tool.py"))
    tool = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(tool)


class TestSkill(unittest.TestCase):

    def test_the_manifest_is_present(self):
        self.assertIsInstance(getattr(tool, "MANIFEST", None), dict)

    def test_the_manifest_names_the_skill(self):
        self.assertEqual(getattr(tool, "MANIFEST", {}).get("name"), "@@name")

    def test_the_manifest_has_a_description(self):
        self.assertTrue(
            str(getattr(tool, "MANIFEST", {}).get("description") or "").strip())

    def test_run_is_callable(self):
        self.assertTrue(callable(getattr(tool, "run", None)))


if __name__ == "__main__":
    unittest.main()
'''


def synthesize(candidate: dict, *, name: str | None = None,
               config_data: dict | None = None,
               body: str | None = None) -> dict:
    """Write a skill from a candidate.  Returns the skill's record.

    This is the *drafting* step.  It writes the skill; it does not pin
    it.  The gate runs next, and only a green gate leads to a pin.
    """
    name = name or candidate.get("name") or "skill"
    manifest = default_manifest(candidate, name=name)
    body = body if body is not None else default_body(candidate)
    directory = write_skill(name, manifest, body, config_data=config_data)
    return {"name": name, "dir": directory, "manifest": manifest}


def gate_skill(name: str, *, config_data: dict | None = None,
               delta: float | None = None, require_benchmark: bool = True,
               sealed: bool | None = None) -> dict:
    """Run the gate on a written skill.

    ``sealed`` defaults to the deployment's own answer -- whether
    package installs are refused -- because that is a property of the
    deployment and not of the skill.

    Returns the gate's result as a dict.
    """
    directory = skill_dir(name, config_data)
    manifest = read_manifest(os.path.join(directory, "manifest.yaml"))
    source, test_source, _ = read_skill_sources(directory)

    if sealed is None:
        sealed = _installs_sealed()

    # The ruler goes in as a forbidden path: a skill that names the
    # holdout, the dev split or the baseline is reaching for the
    # measuring instrument, and I7 says the measuring instrument is
    # outside the agent's writable surface.
    try:
        from .skills_gate import ruler_paths
        forbidden = ruler_paths(config_data)
    except Exception:
        forbidden = []

    result = run_gate(manifest=manifest, source=source,
                      test_source=test_source, delta=delta,
                      sealed=sealed, require_benchmark=require_benchmark,
                      forbidden_paths=forbidden)
    return result.to_dict()


def _installs_sealed() -> bool:
    """Whether package installs are refused in this deployment.

    The command policy is the single source of truth for that rule, and
    the curator asks it rather than deciding for itself: two answers to
    "are installs allowed" is how a skill gets built against a rule that
    is not the one in force.
    """
    try:
        from src.mcp.servers.tasks.os.bash.command_policy import installs_sealed
        return bool(installs_sealed())
    except Exception:
        return False


def _keep_previous_pin(config_data: dict | None = None) -> None:
    """Copy the live pin aside before it is replaced.

    The copy is what makes a promotion reversible: a rollback is a file
    copy, not a reconstruction of what the file said.  It is written
    *before* the live pin is replaced, so the backup is always the pin
    that was actually in force.

    A pin that does not exist yet is still copied aside -- as an empty
    pin -- because "nothing was pinned" is a state worth returning to.
    Without this, the first promotion of a deployment's life would be
    the one promotion that could not be undone, which is the worst
    possible moment for that to be true.
    """
    from .skills import pinned_path, write_pin

    path = pinned_path(config_data)
    try:
        if os.path.isfile(path):
            with io.open(path, "r", encoding="utf-8") as f:
                data = f.read()
        else:
            data = yaml.safe_dump({"version": 0, "skills": {}},
                                  sort_keys=True)
        with io.open(path + ".prev", "w", encoding="utf-8") as f:
            f.write(data)
    except OSError:
        pass


def promote(name: str, *, config_data: dict | None = None,
            delta: float | None = None,
            require_benchmark: bool = True) -> dict:
    """Pin a skill, if the gate says it may be pinned.

    Returns ``{"pinned": bool, "gate": {...}, "reason": str}``.

    The gate runs first, and its verdict is the gate on the pin: a skill
    that failed any check is not pinned, whatever the operator asked
    for.  The operator's request is recorded in the reason so that the
    refusal is legible rather than mysterious.
    """
    gate = gate_skill(name, config_data=config_data, delta=delta,
                      require_benchmark=require_benchmark)
    if not gate["passed"]:
        return {"pinned": False, "gate": gate,
                "reason": "the gate said no: %s" % gate["reason"]}

    pin = read_pin(config_data)
    _keep_previous_pin(config_data)
    new_pin = apply_deltas(pin, [{"op": "insert", "name": name,
                                  "skill": {"name": name,
                                            "status": ACTIVE,
                                            "pinned_at": _today()}}])
    new_pin.pop("skipped", None)
    write_pin(new_pin, config_data)
    return {"pinned": True, "gate": gate, "reason": ""}


def rollback(config_data: dict | None = None) -> dict:
    """Restore the previous pin.

    A copy of a file, not a reconstruction: the previous pin is what was
    in force, byte for byte.
    """
    from .skills import pinned_path

    path = pinned_path(config_data)
    previous = path + ".prev"
    if not os.path.isfile(previous):
        return {"rolled_back": False,
                "reason": "there is no previous pin to roll back to"}
    try:
        shutil.copyfile(previous, path)
    except OSError as e:
        return {"rolled_back": False, "reason": "the rollback failed: %s" % e}
    return {"rolled_back": True, "reason": "",
            "pin": read_pin(config_data)}
