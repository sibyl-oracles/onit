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

Driving the A/B: two arms, one split, one number.

``runner.py`` knows how to replay a split and grade the answers.  This
module knows how to build the *real* agent, so that the replay is the
same agent a user would talk to, and how to put the two arms side by
side.

**The agent is built in a worker thread.**  ``OnIt.__init__`` discovers
its tools with ``asyncio.run``, which cannot run inside a running event
loop.  The benchmark provider solves this the same way: build the agent
in an executor, then use it from the loop.  Every arm gets its own
agent, because the arms differ in the agent's own configuration and a
shared agent would have the first arm's configuration for the second.

**The order of the arms is the control's advantage, and that is
deliberate.**  The control runs first, so it has no trajectories to
recall from -- and that is exactly the state a shipped agent is in.  The
treatment runs second, with the control's trajectories in the pool, so
it can learn from the run that preceded it.  The treatment is therefore
given the *better* chance, and a win under those conditions is a real
win rather than an artefact of ordering.
"""

import asyncio
import logging
import tempfile

logger = logging.getLogger(__name__)

# The default number of items to replay.  Small on purpose: the point
# of the first run is to find out whether the instrument works at all,
# and a 400-item replay that takes an hour tells you that slowly.
DEFAULT_LIMIT = 20

# The default split to replay.  The dev split, not the holdout: the
# holdout is read once per candidate, at promotion time, and a run that
# spent the holdout on a first attempt has nothing left to promote
# against.
DEFAULT_SPLIT = "dev"


def _agent_factory(config: dict):
    """Build an agent for one arm, off the event loop."""
    from src.cli import _ensure_mcp_servers
    from src.onit import OnIt

    _ensure_mcp_servers(config)
    return OnIt(config)


async def _make_agent(config: dict):
    """Build an agent in a worker thread and return it."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _agent_factory, config)


async def run_ab(items: list, *, root: str, base_config: dict | None = None,
                 arms=("off", "adapt"), limit: int | None = None,
                 with_grader: bool = True, make_agent=None) -> dict:
    """Replay a split through two arms and compare them.

    ``make_agent(config)`` builds the agent for one arm.  It defaults to
    the real one, which builds an ``OnIt`` in a worker thread; a test
    passes a stub instead.

    Returns the comparison dict from ``runner.compare``, with the
    per-arm results attached under ``"results"``.
    """
    from .llm import make_ask
    from .runner import (ARM_CONTROL, ARM_TREATMENT, compare, run_arm,
                         summarise)

    base_config = dict(base_config or {})
    ask = None
    if with_grader:
        try:
            ask = await make_ask(base_config)
        except Exception as e:
            logger.warning("no grader available (%s); answers will be "
                           "recorded but not graded", e)

    def factory(config: dict):
        if make_agent is not None:
            return make_agent(config)
        return _make_agent(config)

    per_arm = {}
    for arm in arms:
        results = await run_arm(items, arm=arm, root=root,
                                 base_config=base_config,
                                 make_agent=factory, ask=ask,
                                 limit=limit)
        per_arm[arm] = results

    control = per_arm.get(ARM_CONTROL, [])
    treatment = per_arm.get(ARM_TREATMENT, [])
    result = compare(control, treatment)
    result["results"] = per_arm
    result["summaries"] = {arm: summarise(r) for arm, r in per_arm.items()}
    return result


def default_root() -> str:
    """Where a run's scratch directory goes.

    A temp directory by default: a run's arm stores are throwaway
    working state, and leaving them in the operator's tree would put
    benchmark traffic next to their real corpus, which is the exact
    confusion this module exists to prevent.
    """
    return tempfile.mkdtemp(prefix="onit-evalrun-")


def base_config_for_run(config_data: dict | None = None) -> dict:
    """The agent configuration a replay run starts from.

    The operator's own configuration, with the parts that would make
    the run non-reproducible taken out: the terminal UI, streaming, and
    the loop's interactive mode.
    """
    config = dict(config_data or {})
    config["loop"] = True
    config["stream"] = False
    config["verbose"] = False
    config["show_logs"] = False
    return config
