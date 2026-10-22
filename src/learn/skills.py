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

Loop C: the skill store.  Recurring fragments promoted to tools.

The playbook (Loop B) distils *tactics* -- prose the model reads and
applies.  This is the other thing: when the agent keeps solving the same
subtask with the same ad-hoc bash or Python fragment, the fragment itself
becomes a tool.  A tactic you have to remember is a tactic you will
forget; a tool is a capability.

**What a skill is.**  A directory under the learned store:

    skills/<name>/
      manifest.yaml     name, description, input schema, deps, version
      tool.py           the implementation, one `run` function
      test_<name>.py    the tests the gate runs

The manifest is the contract.  The description is not a nicety: tool
selection accuracy is mostly a function of description quality, and a
skill with a vague description is a skill the model never reaches for.
The description is therefore validated as strictly as the code.

**Delta-only (I2).**  Like the playbook, the skill library is edited by
operations on identified items -- insert, update, deprecate -- never
regenerated wholesale.  A wholesale rewrite of a skill library drops the
skills nobody was looking at, and the drop is invisible: nothing records
that a capability used to exist.  A delta names what it touched.

**Pinned and reversible (I3).**  Everything live is named in `pinned.yaml`,
next to the skills.  A skill that is not pinned is not served, however
good it looks.  `onit learn --skills --rollback` restores the previous
pin, which is what makes an automated promotion safe to run unattended:
the worst case is a rollback, not a broken agent.

**Sandboxed (I6).**  A skill is code.  Code the agent wrote, for the
agent, is the sharpest edge in the whole track.  The gate that admits a
skill runs its tests inside the container path and screens its source
against the command policy before it is ever pinned.  This module does
not execute anything -- it holds the *records*; `skills.validate` is
where the code runs.
"""

import hashlib
import io
import os
import re
import time

import yaml

# The three states a skill can be in, matching the playbook's.
ACTIVE = "active"
PROBATION = "probation"
DEPRECATED = "deprecated"

# A skill's own version, bumped by the curator when it changes.  The
# version is what a trajectory record carries, so a later run can ask
# "did the runs that saw skill v3 do better than the runs that saw v2?"
# -- the same question the playbook asks of itself.
SCHEMA_VERSION = 1

# The minimum a manifest must declare for the skill to be servable.  A
# skill without a description is a skill the model cannot choose, and a
# skill without a schema is a skill the model cannot call.
REQUIRED_MANIFEST_KEYS = ("name", "description", "version")

# The longest a description may be.  Every tool's description is paid
# for in the prompt on every request, and the descriptions compete with
# each other for the model's attention.  A wall of text in a tool
# description is worse than a thin one.
MAX_DESCRIPTION_CHARS = 400

# A skill name is a filename, a tool name, and a Python identifier all
# at once.  Keep it to something that can be all three.  One character
# is allowed: it is a legal identifier and a legal filename, and a rule
# that rejects it would reject a name the miner could legitimately
# produce.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,48}$")

# The longest a single input parameter's name may be.  Long names cost
# tokens in the schema on every request.
MAX_PARAM_NAME = 40


def _now() -> float:
    return time.time()


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def skills_dir(config_data: dict | None = None) -> str:
    """Where the skill library lives.

    A sibling of ``trajectories/``, not a child of it: the skills are
    not trajectories, and burying them under the trajectory directory
    would make ``trajectories/`` mean two things at once.
    """
    from .config import learn_root

    return os.path.join(learn_root(config_data), "skills")


def skill_dir(name: str, config_data: dict | None = None) -> str:
    """One skill's directory."""
    return os.path.join(skills_dir(config_data), name)


def pinned_path(config_data: dict | None = None) -> str:
    """The pin: the list of skills that are allowed to be served."""
    return os.path.join(skills_dir(config_data), "pinned.yaml")


def _safe_name(name: str) -> str:
    """A name that is safe to use as a directory name.

    Returns "" for a name that is not safe, and the caller treats that as
    "no such skill" rather than as an error -- a bad name in a pin file
    should not be able to make the loader write outside the store.
    """
    name = str(name or "").strip()
    if not _NAME_RE.match(name):
        return ""
    return name


def skill_id(name: str) -> str:
    """The stable id of a skill, from its name."""
    return "sk-" + hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:12]


def _atomic_write(path: str, text: str) -> bool:
    """Write a file without leaving a half-written one behind.

    A skill store that can be interrupted mid-write is a store that can
    serve a truncated manifest, and a truncated manifest is a tool with
    no description.  The write goes to a temporary file in the same
    directory and is renamed into place, which is atomic on every
    filesystem worth caring about.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with io.open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def read_manifest(path: str) -> dict | None:
    """Read a skill's manifest, or None if it is missing or unreadable.

    A manifest that does not parse is treated as a manifest that does not
    exist, because a skill the loader cannot read is a skill that cannot
    be served, and serving a half-understood skill is worse than not
    serving it.
    """
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def write_manifest(path: str, manifest: dict) -> bool:
    """Write a manifest."""
    return _atomic_write(path, yaml.safe_dump(manifest, sort_keys=True,
                                              default_flow_style=False))


def read_pin(config_data: dict | None = None) -> dict:
    """The pin file: which skills are live, and which version of each.

    Returns ``{"version": int, "skills": {name: {...}}}``.  A missing pin
    file is an empty pin -- nothing is served -- which is the safe
    default: a skill that has never been promoted is not served.
    """
    path = pinned_path(config_data)
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return {"version": 0, "skills": {}}
    if not isinstance(data, dict):
        return {"version": 0, "skills": {}}
    skills = data.get("skills")
    if not isinstance(skills, dict):
        skills = {}
    return {"version": int(data.get("version") or 0), "skills": skills}


def write_pin(pin: dict, config_data: dict | None = None) -> bool:
    """Write the pin file."""
    path = pinned_path(config_data)
    payload = {
        "version": int(pin.get("version") or 0),
        "updated": _today(),
        "skills": pin.get("skills") or {},
    }
    return _atomic_write(path, yaml.safe_dump(payload, sort_keys=True,
                                              default_flow_style=False))


def pinned_skills(config_data: dict | None = None) -> dict:
    """The pinned skills, by name.

    Only the names that are safe and whose directory still exists.  A pin
    that names a skill whose files were removed is a stale pin, and it is
    filtered here rather than at serve time so that every reader agrees
    on what is live.
    """
    pin = read_pin(config_data)
    out = {}
    for name, entry in (pin.get("skills") or {}).items():
        safe = _safe_name(name)
        if not safe:
            continue
        if not os.path.isfile(os.path.join(skill_dir(safe, config_data),
                                           "manifest.yaml")):
            continue
        out[safe] = entry if isinstance(entry, dict) else {}
    return out


def manifest_errors(manifest: dict | None) -> list:
    """Everything wrong with a manifest, as a list of messages.

    A list rather than a boolean, because the whole point of a gate is
    that it can say *why*.  A skill that failed validation and a skill
    that was never tried look identical otherwise, and the first one is
    a bug in the validator.
    """
    problems = []
    if not isinstance(manifest, dict):
        return ["the manifest is missing or is not a mapping"]

    name = manifest.get("name")
    if not name:
        problems.append("no name")
    elif not _NAME_RE.match(str(name)):
        problems.append(
            "the name %r is not a valid tool name (lower-case, letters "
            "and digits and underscores, starting with a letter)" % (name,))

    description = manifest.get("description")
    if not description or not str(description).strip():
        problems.append("no description -- the model cannot choose a tool "
                        "it cannot tell apart from the others")
    elif len(str(description)) > MAX_DESCRIPTION_CHARS:
        problems.append("the description is %d characters; the limit is %d"
                        % (len(str(description)), MAX_DESCRIPTION_CHARS))

    version = manifest.get("version")
    if version is None:
        problems.append("no version")
    elif not isinstance(version, int) or version < 1:
        problems.append("the version must be a positive integer")

    schema = manifest.get("input_schema")
    if schema is not None:
        if not isinstance(schema, dict):
            problems.append("the input schema is not a mapping")
        else:
            params = schema.get("properties")
            if params is not None and not isinstance(params, dict):
                problems.append("the input schema's properties are not a "
                                "mapping")
            elif isinstance(params, dict):
                for pname in params:
                    if len(str(pname)) > MAX_PARAM_NAME:
                        problems.append(
                            "the parameter %r is %d characters; the limit "
                            "is %d" % (pname, len(str(pname)),
                                       MAX_PARAM_NAME))
    return problems


def apply_deltas(pin: dict, deltas: list) -> dict:
    """Apply skill-library deltas to a pin.

    The operations are the playbook's: ``insert``, ``update``,
    ``deprecate``.  Each names the skill it touches, so a skill that
    nobody noticed is still there, with a counter saying so, rather than
    gone.

    A delta that names a skill that is not there is *not* an error: it
    is recorded in ``skipped``, because the interesting failure of a
    delta-only system is a delta aimed at something that moved, and
    silently ignoring that is how a library drifts away from what it
    says it contains.
    """
    pin = {"version": int(pin.get("version") or 0),
           "skills": dict(pin.get("skills") or {})}
    skills = pin["skills"]
    skipped = []

    for delta in deltas or []:
        if not isinstance(delta, dict):
            continue
        op = str(delta.get("op") or "").lower()
        name = _safe_name(delta.get("name") or "")
        if not name:
            skipped.append({"op": op, "name": delta.get("name"),
                            "reason": "not a valid skill name"})
            continue

        if op == "insert":
            entry = dict(delta.get("skill") or {})
            entry.setdefault("name", name)
            entry.setdefault("version", 1)
            entry.setdefault("status", PROBATION)
            entry.setdefault("pinned_at", _today())
            entry.setdefault("description", "")
            # An insert of a name that is already there is an update, not
            # a second copy.  Two entries for one name is a library that
            # cannot be reasoned about.
            skills[name] = entry
        elif op == "update":
            if name not in skills:
                skipped.append({"op": op, "name": name,
                                "reason": "not in the library"})
                continue
            entry = dict(skills[name])
            patch = delta.get("skill")
            if isinstance(patch, dict):
                entry.update(patch)
            entry["name"] = name
            entry["updated"] = _today()
            skills[name] = entry
        elif op == "deprecate":
            if name not in skills:
                skipped.append({"op": op, "name": name,
                                "reason": "not in the library"})
                continue
            entry = dict(skills[name])
            entry["status"] = DEPRECATED
            entry["deprecated_at"] = _today()
            entry["reason"] = delta.get("reason") or ""
            skills[name] = entry
        else:
            skipped.append({"op": op, "name": name,
                            "reason": "unknown operation"})

    pin["version"] = int(pin.get("version") or 0) + 1
    if skipped:
        pin["skipped"] = skipped
    return pin


def read_skill(name: str, config_data: dict | None = None) -> dict:
    """One skill's full record: its manifest and its pin entry.

    Returns ``{}`` for a skill that is not there, so that a caller can
    ask about a skill that has not been created yet without a special
    case.
    """
    safe = _safe_name(name)
    if not safe:
        return {}
    directory = skill_dir(safe, config_data)
    manifest = read_manifest(os.path.join(directory, "manifest.yaml"))
    if manifest is None:
        return {}
    pin = read_pin(config_data)
    entry = (pin.get("skills") or {}).get(safe) or {}
    return {"name": safe, "dir": directory, "manifest": manifest,
            "pin": entry}


def list_skills(config_data: dict | None = None) -> list:
    """Every skill in the library, pinned or not.

    Sorted by name, because a listing that changes order between two
    runs of the same command is a listing nobody can read twice.
    """
    root = skills_dir(config_data)
    out = []
    try:
        names = os.listdir(root)
    except OSError:
        return []
    pin = read_pin(config_data)
    for name in sorted(names):
        safe = _safe_name(name)
        if not safe:
            continue
        directory = os.path.join(root, safe)
        if not os.path.isdir(directory):
            continue
        manifest = read_manifest(os.path.join(directory, "manifest.yaml"))
        if manifest is None:
            continue
        out.append({
            "name": safe,
            "manifest": manifest,
            "pinned": safe in (pin.get("skills") or {}),
            "pin": (pin.get("skills") or {}).get(safe) or {},
        })
    return out


def summarize(config_data: dict | None = None) -> dict:
    """One screen of text about the skill library."""
    skills = list_skills(config_data)
    pin = read_pin(config_data)
    live = pinned_skills(config_data)
    by_status = {}
    for skill in skills:
        status = str((skill.get("pin") or {}).get("status") or "unpinned")
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "skills": len(skills),
        "pinned": len(live),
        "pin_version": int(pin.get("version") or 0),
        "by_status": by_status,
        "names": [s["name"] for s in skills],
    }
