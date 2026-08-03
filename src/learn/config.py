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

Where the learning loops are allowed to act, in one place.

Autonomy is a single ladder rather than a switch per feature, so "what is this
agent allowed to change about itself" has one answer that can be read off a
config file, printed in a benchmark row, and raised one rung at a time:

    0  off       nothing is recorded, nothing is injected
    1  observe   trajectories are recorded; nothing is injected   (default)
    2  adapt     recalled episodes and the playbook are injected
    3  extend    validated skills are served as tools
    4  evolve    scaffold variants may be proposed (never auto-merged)

Only levels 0 and 1 are implemented; the higher rungs are named here so that
the setting a deployment writes today keeps its meaning when they land.
``ONIT_LEARN`` overrides the config file, which is what benchmark baselines and
one-off reproductions use to pin a run to a known level.
"""

import os

OFF = 0
OBSERVE = 1
ADAPT = 2
EXTEND = 3
EVOLVE = 4

DEFAULT_AUTONOMY = OBSERVE

_NAMES = {"off": OFF, "observe": OBSERVE, "adapt": ADAPT,
          "extend": EXTEND, "evolve": EVOLVE}

# Level implemented today.  A config asking for more gets what exists, rather
# than a run that silently believes it is learning more than it is.
_MAX_IMPLEMENTED = OBSERVE

# Truthy/falsy spellings accepted from the environment, where everything is a
# string and "0" is not falsy on its own.
_FALSE = ("0", "false", "no", "off", "none", "")
_TRUE = ("1", "true", "yes", "on")


def _coerce(value) -> int | None:
    """Read an autonomy level from a name, a number, or a bare boolean."""
    if value is None:
        return None
    if isinstance(value, bool):
        return OBSERVE if value else OFF
    if isinstance(value, int):
        return max(OFF, min(EVOLVE, value))
    text = str(value).strip().lower()
    if text in _NAMES:
        return _NAMES[text]
    if text in _FALSE:
        return OFF
    if text in _TRUE:
        return OBSERVE
    try:
        return max(OFF, min(EVOLVE, int(text)))
    except ValueError:
        return None


def autonomy(config_data: dict | None = None) -> int:
    """Resolve the active autonomy level.

    ``ONIT_LEARN`` wins over the config file so a benchmark baseline can pin a
    run to level 0 without editing anyone's configuration.
    """
    level = _coerce(os.environ.get("ONIT_LEARN"))
    if level is None:
        learn = (config_data or {}).get("learn")
        if isinstance(learn, dict):
            level = _coerce(learn.get("autonomy"))
            if level is None and learn.get("enabled") is False:
                level = OFF
        else:
            level = _coerce(learn)
    if level is None:
        level = DEFAULT_AUTONOMY
    return min(level, _MAX_IMPLEMENTED)


def level_name(level: int) -> str:
    """Name of an autonomy level, for logs and benchmark rows."""
    for name, value in _NAMES.items():
        if value == level:
            return name
    return str(level)


def recording_enabled(config_data: dict | None = None) -> bool:
    """Whether trajectories are written for this run."""
    return autonomy(config_data) >= OBSERVE


def redact_tool_args(config_data: dict | None = None) -> bool:
    """Whether tool arguments are stored as digests rather than values.

    Defaults to on.  Tool arguments carry file paths, search queries and
    occasionally a credential that was pasted into a task, and a trajectory
    store is read back by later loops rather than by the person who typed it.
    """
    learn = (config_data or {}).get("learn")
    if isinstance(learn, dict) and learn.get("redact_tool_args") is not None:
        return bool(learn["redact_tool_args"])
    env = os.environ.get("ONIT_LEARN_REDACT")
    if env is not None:
        return str(env).strip().lower() not in _FALSE
    return True


def trajectory_dir(config_data: dict | None = None) -> str:
    """Directory that holds trajectory files.

    Unlike :func:`autonomy`, the config wins over the environment here: the
    level is a per-run choice an operator overrides to pin a baseline, while
    the path is deployment layout.  ``ONIT_LEARN_PATH`` supplies the default
    for a config that does not name one.
    """
    learn = (config_data or {}).get("learn")
    if isinstance(learn, dict) and learn.get("path"):
        base = str(learn["path"])
    else:
        base = os.environ.get("ONIT_LEARN_PATH", "~/.onit/learned")
    return os.path.join(os.path.expanduser(base), "trajectories")
