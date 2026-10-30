"""
CLI entry point for the OnIt agent.

Usage:
    onit                                          # interactive terminal chat (continues the last session)
    onit --new-session                            # interactive terminal chat, fresh session
    onit --new-session --name-session "thesis ch3"   # ... named, so it is easy to find again
    onit --resume-session "thesis ch3"            # continue a named session
    onit setup                                    # interactive setup wizard
    onit setup --show                             # show current configuration
    onit setup --delete tavily                    # remove a stored key
    onit sessions                                 # list previous sessions
    onit sessions --rename 2 "chapter 3 draft"    # rename a session
    onit doctor [--deep]                          # run the live self-check battery
    onit resume                       # continue a previous session (same as --resume-session)
    onit serve web [--port 9000]                  # launch the web UI
    onit serve loop "task" [--period 60]          # repeat a task on a timer
    onit --config my.yaml                         # custom config file
    onit --container                              # run in a hardened Docker container
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import yaml

from .onit import OnIt
from .lib.tools import register_stdio_servers, apply_default_mcp_servers


def _find_default_config() -> str:
    """Locate the default config file, checking common locations."""
    candidates = [
        "configs/default.yaml",
        os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"),
        os.path.expanduser("~/.onit/config.yaml"),
        # Bundled config inside the installed package (pip install)
        os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return "configs/default.yaml"



def _is_external_server(server: dict) -> bool:
    """True for a server OnIt must not start or re-port: it lives elsewhere.

    Marked by ``external: true`` in the config (the old --mcp-sse /
    --mcp-server flags wrote the same thing as generated names).
    """
    if server.get('external'):
        return True
    name = server.get('name', '')
    return name.startswith('ExternalSSE_') or name.startswith('ExternalMCP_')


def _ensure_mcp_servers(config_data: dict, log_level='ERROR'):
    """Register this process's stdio MCP servers so the client can spawn them.

    Every server is a subprocess of this process: it runs as this user, dies
    with this process, and is reachable by no other account on the machine.
    The MCP client spawns a server the first time something connects to it, so
    there is nothing to start or wait for here.
    """
    # Propagate data_path to MCP servers via an environment variable.
    # Fall back to OnIt's own default (~/sandbox) when unset so the MCP servers write
    # to the same absolute local path the UI displays, instead of their /tmp fallback.
    data_path = config_data.get('data_path') or str(Path.home() / "sandbox")
    data_path = str(Path(data_path).expanduser().resolve())
    os.environ['ONIT_DATA_PATH'] = data_path

    # Fill in the default servers first. A config that names none still gets
    # them — from OnIt, further down startup — but a server the CLI never saw
    # would be registered with a data_path resolved differently, so the list
    # must be complete here, where the launch specs are written.
    config_data.setdefault('mcp', {}).setdefault('servers', [])
    servers = config_data['mcp']['servers']
    apply_default_mcp_servers(servers)

    register_stdio_servers(servers, data_path, log_level,
                           config_data=config_data)


# What the web UI serves with, over and above the shared ``serving:`` block.
# A browser turn is a chat: one iteration, no tool loop to amortise a reasoning
# pass over, and someone watching the composer while a 27B model deliberates —
# so the reasoning is the whole wait.  The terminal is where runs are long
# enough for thinking to earn its latency, and it keeps it.  Sampling moves
# with the switch because thinking and instruct mode want different numbers
# (see the table in docs/MODEL_SERVING.md); leaving the thinking-mode values on
# a model told not to think is the wrong half of the pair.
WEB_SERVING_DEFAULTS = {
    'think': False,
    'temperature': 0.7,
    'top_p': 0.8,
}


def _apply_web_serving_defaults(config_data: dict, is_web: bool,
                                force_think: bool = False):
    """Layer :data:`WEB_SERVING_DEFAULTS` over ``serving:`` for a web run.

    These beat a plain ``serving.think`` rather than filling in for it: one
    config serves both UIs, so a ``serving:`` block that turns reasoning on
    for the terminal would otherwise turn it on for the browser too, and the
    split would never take effect for anyone who had asked for thinking at
    all.  ``serving.web`` is the way back — any key set there wins over the
    default beside it, so ``serving.web.think: true`` restores reasoning in
    the browser and leaves the terminal alone.

    The defaults land as one trade rather than three independent settings:
    thinking off, and the sampling an instruct-mode turn wants.  Put the
    reasoning back — through ``serving.web`` or a ``--think`` on the command
    line, which *force_think* reports — and the sampling half goes with it,
    falling through to ``serving:``, which is already tuned for a model that
    thinks.  Sampling named explicitly under ``serving.web`` applies either
    way; it is a stated preference, not half of a default.

    The ``serving.web`` block is consumed here on every run, web or not, so
    that a terminal session never carries it into chat().
    """
    serving_cfg = config_data.get('serving')
    web_cfg = (serving_cfg.pop('web', None)
               if isinstance(serving_cfg, dict) else None)
    if not is_web:
        return
    web_cfg = dict(web_cfg) if isinstance(web_cfg, dict) else {}
    overrides = ({} if (force_think or web_cfg.get('think'))
                 else dict(WEB_SERVING_DEFAULTS))
    overrides.update(web_cfg)
    config_data.setdefault('serving', {}).update(overrides)


def _merge_base(override: dict, base: dict):
    """Recursively merge *override* into *base* (in-place).

    Values from *override* take precedence.  For nested dicts the merge
    is recursive so that e.g. ``serving.host`` from the override replaces
    only that key, not the entire ``serving`` block.
    """
    for key, value in override.items():
        if (key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)):
            _merge_base(value, base[key])
        else:
            base[key] = value


def _token_count(text: str) -> int:
    """Parse a token count written as a plain number or with a k/M suffix.

    ``32768``, ``128k`` and ``1M`` are all accepted (case-insensitive, with
    underscores or commas allowed as digit separators), because the numbers
    these flags take are the kind nobody wants to spell out in full.
    """
    cleaned = text.strip().replace(",", "").replace("_", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmM]?)", cleaned)
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid token count: {text!r} (use e.g. 32768, 128k, or 1M)")
    value = float(match.group(1)) * {"": 1, "k": 1_000, "m": 1_000_000}[
        match.group(2).lower()]
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"token count must be at least 1, got {text!r}")
    return int(value)


def _learn_eval_run(args, config_data: dict) -> None:
    """Replay a split through the real agent and grade the answers.

    This is the instrument.  Everything else in the plan changes the
    agent; this is what says whether the change helped.

    Without ``--ab`` it runs one arm and reports the numbers, which is
    how you find out whether the instrument works before spending a
    second arm on it.  With ``--ab`` it runs the control and the
    treatment and prints the difference with an error bar.

    The split is the dev split unless asked otherwise.  The holdout is
    read once per candidate, at promotion time: a first run that spends
    it has nothing left to promote against, and the whole point of a
    holdout is that it is not spent twice.
    """
    import asyncio
    import shutil

    from .learn.ab import (DEFAULT_LIMIT, DEFAULT_SPLIT, base_config_for_run,
                           default_root, run_ab)
    from .learn.runner import format_comparison, summarise

    split = getattr(args, "eval_split", None) or DEFAULT_SPLIT
    limit = getattr(args, "eval_limit", None) or DEFAULT_LIMIT

    from .learn.eval import read_dev, read_holdout
    items = read_holdout(config_data) if split == "holdout" else read_dev(config_data)
    if not items:
        print("The %s split is empty. Draw one with 'onit learn --holdout' "
              "first." % split)
        sys.exit(1)
    items = items[:limit]
    print("Replaying %d item(s) from the %s split." % (len(items), split))

    root = getattr(args, "eval_out", None) or default_root()
    owned = not getattr(args, "eval_out", None)
    base = base_config_for_run(config_data)

    try:
        if getattr(args, "ab", False):
            result = asyncio.run(run_ab(items, root=root, base_config=base,
                                        limit=limit))
            print()
            print(format_comparison(result,
                                    label="A/B on the %s split:" % split))
            out = os.path.join(root, "comparison.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in result.items()
                           if k != "results"}, f, indent=2, default=str)
            print()
            print("Results: %s" % out)
        else:
            from .learn.runner import run_arm
            from .learn.llm import make_ask
            from .learn.ab import _make_agent

            try:
                ask = asyncio.run(make_ask(base))
            except Exception as e:
                print("No grader available (%s); answers will be recorded "
                      "but not graded." % e)
                ask = None
            results = asyncio.run(run_arm(
                items, arm="off", root=root, base_config=base,
                make_agent=_make_agent, ask=ask, limit=limit))
            summary = summarise(results)
            print()
            print("  n=%d  correct=%d  wrong=%d  not-attempted=%d  "
                  "ungraded=%d  errors=%d  accuracy=%.3f"
                  % (summary["n"], summary["correct"], summary["incorrect"],
                     summary["not_attempted"], summary["ungraded"],
                     summary["errors"], summary["accuracy"]))
            out = os.path.join(root, "arms", "off", "results.json")
            print("Results: %s" % out)
    finally:
        if owned and not getattr(args, "eval_out", None):
            # The arm stores are throwaway working state.  Keeping them
            # would put benchmark traffic next to the operator's own
            # corpus, which is the confusion this command exists to
            # prevent.
            shutil.rmtree(root, ignore_errors=True)


def _learn_draw_holdout(args, config_data: dict) -> None:
    """Draw and freeze the holdout and the dev split.

    The one command that writes the ruler.  It is deliberately explicit and
    deliberately refuses to write inside the agent's own writable surface:
    a holdout the agent can edit is not a holdout, and the whole point of
    freezing it is that the thing being measured cannot reach it.

    Exit codes: 0 on success, 1 when there is nothing to draw or the draw
    would land inside the agent's surface.
    """
    from .learn.eval import (MIN_DEV_ITEMS, MIN_HOLDOUT_ITEMS, build_split,
                             eval_dir, is_outside_agent_surface, write_split)

    split = build_split(config_data=config_data,
                        model=getattr(args, "model", None),
                        holdout_ratio=getattr(args, "holdout_ratio", 0.4),
                        max_items=getattr(args, "max_items", 400))

    counts = split["counts"]
    if not counts["candidates"]:
        print("Nothing to draw from. The corpus holds no usable recorded "
              "tasks (a task needs text and an answer; a task owned by a "
              "tenant is never eligible for a shared split).", file=sys.stderr)
        sys.exit(1)

    if not is_outside_agent_surface(eval_dir(config_data), config_data):
        print("I7: the eval directory (%s) lies inside the agent's writable "
              "surface. The holdout cannot be frozen there -- the agent "
              "could edit its own exam. Point 'learn.eval.path' at a "
              "read-only mount outside learn.path, or unset it to use the "
              "repository's benchmarks/eval/." % eval_dir(config_data),
              file=sys.stderr)
        sys.exit(1)

    if getattr(args, "dry_run", False):
        print("Would draw %d holdout and %d dev items from %d candidates "
              "(model=%s)." % (counts["holdout"], counts["dev"],
                               counts["candidates"], split["model"] or "any"))
        print("Target: %s" % eval_dir(config_data))
        print("Nothing written (--dry-run).")
        return

    try:
        written = write_split(split, config_data)
    except PermissionError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print("Froze %d holdout items and %d dev items in %s"
          % (written["holdout_items"], written["dev_items"],
             written["holdout"]))
    if written["holdout_items"] < MIN_HOLDOUT_ITEMS:
        print("  Note: %d holdout items is below the %d needed for the "
              "holdout's stderr to mean anything -- a loop that cannot lose "
              "on it cannot win on it either."
              % (written["holdout_items"], MIN_HOLDOUT_ITEMS))
    if written["dev_items"] < MIN_DEV_ITEMS:
        print("  Note: %d dev items is below the %d needed to iterate "
              "against." % (written["dev_items"], MIN_DEV_ITEMS))


def _learn_playbook(args, config_data: dict) -> None:
    """Show, reflect over, or roll back the playbook (Loop B).

    Three jobs, one subcommand, because they are three views of the same
    store:

    * ``--playbook`` shows what is in it and what would be injected;
    * ``--reflect`` runs the offline pass over the recorded corpus and
      applies the deltas it finds (this is the Curator);
    * ``--rollback`` restores the previous version, which is what makes
      the change reversible (I3).

    Reflection is offline and explicit.  It never runs on the hot path
    of a task, and it never reads the holdout (I1): the holdout is the
    ruler, and a loop that read it would be grading its own homework.
    """
    from .learn import playbook as pb

    if getattr(args, "rollback", False):
        _learn_playbook_rollback(args, config_data)
        return

    if getattr(args, "reflect", False):
        _learn_playbook_reflect(args, config_data)
        return

    playbook = pb.load(config_data, owner=None)
    items = playbook["items"]
    if not items:
        print("The playbook is empty. It fills up when the reflect pass "
              "has contrasting runs to learn from -- 'onit learn --reflect'.")
        return
    print("Playbook v%d -- %d bullet(s)" % (playbook["version"], len(items)))
    print()
    for item in items:
        print("  [%s] %s" % (item["status"], item["text"]))
        if item.get("trigger"):
            print("      trigger: %s   used %d, helped %d, hurt %d"
                  % (item["trigger"], item.get("used", 0),
                     item.get("helped", 0), item.get("hurt", 0)))
    print()
    budget = pb.DEFAULT_TOKEN_BUDGET
    selected = pb.select(items, token_budget=budget)
    print("%d of %d fit the %d-token budget and would be injected."
          % (len(selected), len(items), budget))


def _reflect_with_model(args, config_data: dict, records: list) -> list:
    """The model-based Reflector: read two runs, name the difference.

    Best-effort throughout.  A missing endpoint, an unreachable model or
    a response that is not the shape that was asked for all yield no
    deltas rather than an error: the reflection is an enhancement, and
    the mechanical pass is still there to do what it can.
    """
    import asyncio

    from .learn.llm import make_ask
    from .learn.reflector import reflect_corpus

    try:
        ask = asyncio.run(make_ask(config_data))
    except Exception as e:
        print("Model reflection unavailable (%s). Falling back to the "
              "mechanical reflector." % e)
        return []

    try:
        return asyncio.run(reflect_corpus(ask, records,
                                          config_data=config_data))
    except Exception as e:
        print("Model reflection failed (%s). Falling back to the "
              "mechanical reflector." % e)
        return []


def _learn_playbook_reflect(args, config_data: dict) -> None:
    """The offline Reflector/Curator pass over the recorded corpus."""
    from .learn import playbook as pb
    from .learn.corpus import label_corpus, read_sessions
    from .learn.recall import ALL_OWNERS
    from .learn.trajectory import iter_records

    sessions = read_sessions(config_data)
    if not sessions:
        print("No trajectories on disk, so there is nothing to reflect on.")
        return

    # The labels come first: a contrast needs two sides, and without the
    # implicit signals every run is on the same side of it.
    label_corpus(config_data)

    records = [r for r in iter_records(config_data)
               if r.get("owner") is None]
    if not records:
        print("No owner-less records to reflect on. Scoped records belong "
              "to a tenant and are reflected into that tenant's own "
              "playbook.")
        return

    # Two reflectors, and the difference matters.  The mechanical one
    # names the difference between the tools that ran; the model-based
    # one reads the two runs and names the difference between what they
    # *did*.  The mechanical one alone produces tautologies -- "the same
    # tool, used badly" -- because a cluster is a tool signature, so its
    # two sides always ran the same tools.  With a model, the contrast
    # becomes about arguments, order and reasoning.
    #
    # The model is asked first, and the mechanical pass only fills in
    # what the model did not cover.  The mechanical bullets are weaker,
    # so they go last, and a run with no model available still reflects
    # -- just less well than it could have.
    deltas = []
    if not getattr(args, "mechanical_only", False):
        deltas.extend(_reflect_with_model(args, config_data, records))

    if not getattr(args, "model_only", False):
        # The holdout is excluded inside reflect() itself, so that every
        # caller gets the boundary and not just this one.
        deltas.extend(pb.reflect(records, scope=pb.SCOPE_GLOBAL,
                                  config_data=config_data))

    if not deltas:
        print("Reflected over %d record(s) and found no contrast worth a "
              "bullet: no cluster had both a run that went well and a run "
              "that went badly." % len(records))
        return

    if getattr(args, "dry_run", False):
        print("Would apply %d delta(s):" % len(deltas))
        for delta in deltas:
            print("  + %s" % delta["text"])
        return

    before = pb.load(config_data)
    after = pb.curate(before, deltas, config_data=config_data)
    added = len(after["items"]) - len(before["items"])
    print("Reflected over %d record(s); applied %d delta(s), %d new "
          "bullet(s). Playbook is now v%d."
          % (len(records), len(deltas), added, after["version"]))
    for delta in deltas:
        print("  + %s" % delta["text"])


def _learn_playbook_rollback(args, config_data: dict) -> None:
    """Restore the previous playbook version (I3).

    The rollback works by keeping the previous file beside the live one.
    Nothing is ever mutated in place, so "undo" is a file copy rather
    than a reconstruction, and the rolled-back file is still there to
    roll forward from.
    """
    from .learn import playbook as pb

    path = pb.global_path(config_data)
    backup = path + ".prev"
    if not os.path.isfile(backup):
        print("No previous version to roll back to. The playbook has only "
              "ever had one version.")
        return
    try:
        with open(backup, "r", encoding="utf-8") as f:
            previous = json.load(f)
    except (OSError, ValueError) as e:
        print("The previous version could not be read: %s" % e,
              file=sys.stderr)
        sys.exit(1)
    pb.write_playbook(path, previous)
    print("Rolled the playbook back to v%s." % previous.get("version", "?"))


def _learn_lifecycle(args, config_data: dict) -> None:
    """Report, apply, or undo the tool-lifecycle policy.

    The actuator that ``src/learn/lifecycle.py``'s policy needed and
    never had.  Three jobs:

    * report what the policy decides, whether or not it is enabled to
      act -- a deployment should be able to see the policy's mind
      before letting it change anything;
    * apply the decisions, which withholds a tool from the *registry*
      (it is not deleted, and the next session can have it back);
    * restore a withheld tool.

    Withholding is reversible by construction.  That is what makes
    automating it safe at all: the agent can never permanently destroy
    capability.
    """
    from .learn import actuator

    if getattr(args, "restore", None):
        ok = actuator.restore(args.restore, config_data)
        if not ok:
            print("Nothing to restore for %r." % args.restore,
                  file=sys.stderr)
            sys.exit(1)
        print("Restored %s. It will be offered again from the next session."
              % args.restore)
        return

    decisions = actuator.evaluate(config_data)
    if getattr(args, "apply", False):
        result = actuator.apply(config_data)
        withheld = result["withheld"]
        if not withheld:
            if not actuator.lifecycle_enabled(config_data):
                print("The lifecycle actuator is off. Enable it with "
                      "'learn.lifecycle.enabled: true' in the config, and "
                      "set the autonomy level to 'adapt' or higher.")
            else:
                print("Nothing was due.")
            return
        print("Withheld %d tool(s): %s" % (len(withheld), ", ".join(withheld)))
        print("Reversible: 'onit learn --lifecycle --restore TOOL'.")
        return

    print(actuator.format_report(decisions))
    if decisions and not actuator.lifecycle_enabled(config_data):
        print()
        print("The actuator is off, so none of this has been applied. Turn")
        print("it on with 'learn.lifecycle.enabled: true' and the autonomy")
        print("level at 'adapt' or higher, or apply it by hand with")
        print("'--apply'.")


def _learn_skills(args, config_data: dict) -> None:
    """List, mine, gate, pin, and roll back the skill library.

    The skill library is Loop C: the fragments the agent kept solving
    the same way, promoted to tools.  The order of the operations here
    is the order of the safety story -- a skill is written, gated, and
    only then pinned -- and each of them can be run on its own so that
    an operator can look before they leap.
    """
    from .learn import skills_curator, skills_mining

    action = getattr(args, "skills_action", None) or "list"

    if action == "list":
        summary = skills_curator  # noqa: F841  (import check)
        from .learn.skills import list_skills, summarize as summarize_skills
        items = list_skills(config_data)
        summary = summarize_skills(config_data)
        if not items:
            print("No skills yet.  A skill is a tool call the agent kept "
                  "repeating, promoted to a tool.")
            print("Find them with 'onit learn --skills mine'.")
            return
        print("Skills (%d, %d pinned, pin v%d)"
              % (summary["skills"], summary["pinned"],
                 summary["pin_version"]))
        for item in items:
            manifest = item["manifest"]
            mark = "*" if item["pinned"] else " "
            status = (item.get("pin") or {}).get("status") or "unpinned"
            print("  %s %-28s v%-3s %-12s %s"
                  % (mark, item["name"][:28], manifest.get("version", "?"),
                     status, str(manifest.get("description") or "")[:60]))
        if summary["pinned"]:
            print()
            print("  * = pinned, and therefore served to the model.")
        return

    if action == "mine":
        candidates = skills_mining.mine_store(config_data)
        print(skills_mining.format_candidates(candidates))
        return

    if action == "synthesize":
        name = getattr(args, "skill_name", None)
        if not name:
            print("Which skill?  --synthesize NAME")
            sys.exit(1)
        candidates = skills_mining.mine_store(config_data)
        match = None
        for candidate in candidates:
            if candidate["name"] == name or candidate["digest"] == name:
                match = candidate
                break
        if match is None:
            print("No mined candidate is called %r.  Run "
                  "'onit learn --skills mine' to see them." % name)
            sys.exit(1)
        skill = skills_curator.synthesize(match, name=name,
                                           config_data=config_data)
        print("Wrote the draft skill %s." % skill["name"])
        print("  %s" % skill["dir"])
        print("It is not served yet.  Gate it with "
              "'onit learn --skills gate %s'." % skill["name"])
        return

    if action == "gate":
        name = getattr(args, "skill_name", None)
        if not name:
            print("Which skill?  --gate NAME")
            sys.exit(1)
        delta = getattr(args, "delta", None)
        require = not getattr(args, "no_benchmark", False)
        result = skills_curator.gate_skill(name, config_data=config_data,
                                           delta=delta,
                                           require_benchmark=require)
        for check, entry in (result.get("checks") or {}).items():
            mark = "ok  " if entry["passed"] else "FAIL"
            print("  %s %-14s %s" % (mark, check, entry.get("reason") or ""))
        if result["passed"]:
            print("The gate is green.  Pin it with "
                  "'onit learn --skills pin %s'." % name)
        else:
            print()
            print("Not pinnable: %s" % result["reason"])
        return

    if action == "pin":
        name = getattr(args, "skill_name", None)
        if not name:
            print("Which skill?  --pin NAME")
            sys.exit(1)
        delta = getattr(args, "delta", None)
        require = not getattr(args, "no_benchmark", False)
        result = skills_curator.promote(name, config_data=config_data,
                                        delta=delta,
                                        require_benchmark=require)
        if result["pinned"]:
            print("Pinned %s.  It will be served to the model." % name)
            print("Reversible: 'onit learn --skills rollback'.")
        else:
            print("Not pinned.  %s" % result["reason"])
            sys.exit(1)
        return

    if action == "rollback":
        result = skills_curator.rollback(config_data)
        if result["rolled_back"]:
            print("Rolled the skill pin back.")
        else:
            print("Could not roll back: %s" % result["reason"])
            sys.exit(1)
        return

    print("Unknown --skills action %r." % action, file=sys.stderr)
    sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="onit",
        description="OnIt — an agent harness for task automation, in the terminal or on the web.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # setup
    setup_parser = subparsers.add_parser("setup", help="Interactive setup wizard.")
    setup_parser.add_argument("--delete", metavar="KEY", action="append",
                              default=[],
                              help="Delete a stored secret and exit "
                                   "(repeatable), e.g. --delete tavily. "
                                   "Names: ollama, tavily, weather, github, "
                                   "hf, openrouter, openai, vllm, google, "
                                   "or an endpoint URL.")
    setup_parser.add_argument("--show", action="store_true",
                              help="Display current configuration.")

    # sessions
    sessions_parser = subparsers.add_parser("sessions",
                                            help="List and manage previous sessions.")
    sessions_parser.add_argument("--limit", type=int, default=20,
                                 help="Maximum number of sessions to list (default: 20).")
    sessions_parser.add_argument("--rename", type=str, nargs=2,
                                 metavar=("SESSION", "NEW_NAME"),
                                 help="Rename a session: --rename <name|uuid|#> <new name>. "
                                      "The same thing as 'onit rename'.")
    # --tag was the only rename spelling for a while, and it said "tag" for a
    # thing every other surface calls a name.  It still parses, so an old
    # script or a habit is not broken; it is no longer what --help offers.
    sessions_parser.add_argument("--tag", type=str, nargs=2, dest="rename",
                                 default=argparse.SUPPRESS, metavar=("SESSION", "NEW_NAME"),
                                 help=argparse.SUPPRESS)
    sessions_parser.add_argument("--rebuild-index", action="store_true", dest="rebuild",
                                 help="Rebuild the session index from existing JSONL files.")
    sessions_parser.add_argument("--rebuild", action="store_true", default=argparse.SUPPRESS,
                                 help=argparse.SUPPRESS)
    sessions_parser.add_argument("--include-empty", action="store_true", dest="all",
                                 help="Also list sessions that were opened but never used "
                                      "(no turns, no name).")
    sessions_parser.add_argument("--all", action="store_true", default=argparse.SUPPRESS,
                                 help=argparse.SUPPRESS)
    sessions_parser.add_argument("--delete-all", action="store_true", dest="clear",
                                 help="Delete every session and the index. Asks first.")
    sessions_parser.add_argument("--clear", action="store_true", default=argparse.SUPPRESS,
                                 help=argparse.SUPPRESS)

    # learn: inspect what the agent has recorded about its own runs
    learn_parser = subparsers.add_parser(
        "learn",
        help="Show what OnIt has recorded about its own runs.")
    learn_parser.add_argument("--session", type=str, default=None,
                              metavar="ID",
                              help="Print one session's trajectory records as JSON.")
    learn_parser.add_argument("--json", action="store_true",
                              help="Print the summary as JSON instead of a table.")
    learn_parser.add_argument("--events", action="store_true",
                              help="Print the loop event log (tool lifecycle and "
                                   "the like) instead of the task summary.")
    learn_parser.add_argument("--eval", action="store_true",
                              help="Print the state of the ruler: the holdout, "
                                   "the dev split and the pinned baseline.")
    learn_parser.add_argument("--holdout", action="store_true",
                              help="Draw a holdout and a dev split from the "
                                   "recorded corpus and freeze them outside "
                                   "the agent's writable surface.")
    learn_parser.add_argument("--model", type=str, default=None,
                              metavar="NAME",
                              help="With --holdout, restrict the draw to one "
                                   "model. A split that mixes models mixes "
                                   "two agents into one measurement.")
    learn_parser.add_argument("--holdout-ratio", type=float, default=0.4,
                              metavar="R",
                              help="Fraction of the corpus held out (0-1).")
    learn_parser.add_argument("--max-items", type=int, default=400,
                              metavar="N",
                              help="Cap on how many tasks the draw considers.")
    learn_parser.add_argument("--dry-run", action="store_true",
                              help="With --holdout, report what would be "
                                   "drawn and write nothing.")
    learn_parser.add_argument("--signals", action="store_true",
                              help="Report the implicit outcome signals "
                                   "mined from the corpus: how many records "
                                   "carry a usable label and how many do "
                                   "not.")
    learn_parser.add_argument("--write-signals", action="store_true",
                              help="With --signals, materialise the derived "
                                   "labels to a file beside the store. The "
                                   "trajectories themselves are never "
                                   "rewritten.")
    learn_parser.add_argument("--playbook", action="store_true",
                              help="Show the playbook: what is in it, and "
                                   "what would be injected.")
    learn_parser.add_argument("--reflect", action="store_true",
                              help="Run the offline Reflector over the "
                                   "recorded corpus and apply the deltas "
                                   "it finds (the Curator).")
    learn_parser.add_argument("--mechanical-only", action="store_true",
                              help="With --reflect: use only the mechanical "
                                   "reflector, which names the difference "
                                   "between the tools that ran. No model "
                                   "call, so no model cost.")
    learn_parser.add_argument("--model-only", action="store_true",
                              help="With --reflect: use only the model-based "
                                   "reflector, which reads two contrasting "
                                   "runs and names what the failing one "
                                   "did differently.")
    learn_parser.add_argument("--eval-run", action="store_true",
                              help="Replay a split through the real agent "
                                   "and grade the answers: the ruler, "
                                   "used. Writes a per-item result file and "
                                   "prints the headline numbers.")
    learn_parser.add_argument("--eval-split", type=str, default="dev",
                              choices=["dev", "holdout"],
                              help="With --eval-run, which split to replay. "
                                   "Defaults to dev: the holdout is read "
                                   "once per candidate, at promotion time.")
    learn_parser.add_argument("--eval-limit", type=int, default=None,
                              help="With --eval-run, how many items to "
                                   "replay. Defaults to 20, which is "
                                   "enough to find out whether the "
                                   "instrument works.")
    learn_parser.add_argument("--ab", action="store_true",
                              help="With --eval-run, run both arms -- the "
                                   "control at autonomy off and the "
                                   "treatment at adapt -- and compare "
                                   "them with an error bar.")
    learn_parser.add_argument("--eval-out", type=str, default=None,
                              metavar="DIR",
                              help="With --eval-run, where to write the run's "
                                   "results. Defaults to a temporary "
                                   "directory.")
    learn_parser.add_argument("--rollback", action="store_true",
                              help="Restore the previous playbook version.")
    learn_parser.add_argument("--lifecycle", action="store_true",
                              help="Report the tool-lifecycle decisions: "
                                   "which tools have failed often enough to "
                                   "earn an update, or gone unused long "
                                   "enough to earn an archive.")
    learn_parser.add_argument("--apply", action="store_true",
                              help="With --lifecycle, act on the decisions: "
                                   "withhold the tools the policy says to "
                                   "withhold. Reversible.")
    learn_parser.add_argument("--restore", type=str, default=None,
                              metavar="TOOL",
                              help="With --lifecycle, bring a withheld tool "
                                   "back.")

    # Loop C: the skill library.  The actions are a chain -- mine,
    # synthesize, gate, pin, rollback -- and each can be run on its own
    # so that an operator can look at the gate's verdict before letting
    # anything be served.
    learn_parser.add_argument("--skills", action="store_true",
                              help="The skill library: the tool calls the "
                                   "agent kept repeating, promoted to "
                                   "tools. With --skills-action, runs one "
                                   "step of the chain.")
    learn_parser.add_argument("--skills-action", type=str, default=None,
                              choices=["list", "mine", "synthesize", "gate",
                                       "pin", "rollback"],
                              help="With --skills: what to do. 'list' (the "
                                   "default), 'mine' (find the recurring "
                                   "fragments), 'synthesize' (write a "
                                   "draft), 'gate' (run the five checks), "
                                   "'pin' (serve it), 'rollback' (unpin).")
    learn_parser.add_argument("--skill-name", type=str, default=None,
                              metavar="NAME",
                              help="With --skills-action: the skill to act "
                                   "on.")
    learn_parser.add_argument("--delta", type=float, default=None,
                              help="With --skills-action gate/pin: the "
                                   "benchmark delta measured for the "
                                   "skill. The gate refuses a negative one.")
    learn_parser.add_argument("--no-benchmark", action="store_true",
                              help="With --skills-action gate/pin: allow a "
                                   "skill that has not been measured. The "
                                   "gate says so rather than pretending "
                                   "it passed.")

    # doctor: run the live self-check battery from the shell, no session needed
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run the live self-check battery (same as \\doctor in the text UI).")
    doctor_parser.add_argument("--deep", action="store_true", default=False,
                               help="Also exercise a live model reply and a full "
                                    "tool-calling turn (costs tokens, adds up to a "
                                    "couple of minutes on a slow endpoint).")
    doctor_parser.add_argument("--json", action="store_true", default=False,
                               help="Print the report as JSON instead of text.")
    doctor_parser.add_argument("--keep-session", action="store_true", default=False,
                               help="Keep the throwaway session the check creates "
                                    "(visible in 'onit sessions').")
    doctor_parser.add_argument("--endpoints", action="store_true", default=False,
                               help="Also smoke-test every known endpoint "
                                    "(config, presets, stored keys) with a model "
                                    "listing. Exits 1 if any endpoint fails.")

    # rename
    rename_parser = subparsers.add_parser(
        "rename", help="Rename a session so it is easy to find later.")
    rename_parser.add_argument("session", metavar="SESSION",
                               help="Session to rename: a name, a UUID, or "
                                    "the number shown by 'onit sessions'.")
    rename_parser.add_argument("--session-limit", "--limit", type=int,
                               default=20, dest="limit",
                               help="How many rows 'onit sessions' showed, so "
                                    "a row number means the same session "
                                    "(default: 20).")
    rename_parser.add_argument("name", metavar="NAME",
                               help="The new name.")

    # resume
    resume_parser = subparsers.add_parser(
        "resume", help="Continue a previous session (same as --resume-session).")
    resume_parser.add_argument("session", nargs="?", default="last",
                               metavar="NAME_OR_ID",
                               help='Session name, UUID, or "last" (default: last).')

    # serve: run OnIt in a server or daemon mode
    serve_parser = subparsers.add_parser("serve",
                                         help="Run OnIt in a server or daemon mode.")
    serve_sub = serve_parser.add_subparsers(dest="serve_mode", metavar="MODE")
    serve_sub.required = True

    # serve web
    web_p = serve_sub.add_parser("web", help="Launch the web UI.")
    web_p.add_argument("--port", type=int, default=None,
                       help="Web UI port (default: 9000, or web_port in config).")
    web_p.add_argument("--no-login", action="store_true", dest="no_login",
                       help="Run the web UI without requiring Google login "
                            "(sessions are open to anyone who can reach the port).")
    web_p.add_argument("--voice", action="store_true", dest="voice",
                       help="Enable full-duplex speech-to-speech. Requires a "
                            "NemotronLabs VoiceChat container (see docs/VOICE.md).")

    # serve loop
    loop_p = serve_sub.add_parser("loop",
                                   help="Repeat a task on a configurable timer.")
    loop_p.add_argument("task", type=str,
                        help="Task to execute repeatedly.")
    loop_p.add_argument("--period", type=float, default=None,
                        help="Seconds between iterations (default: 10, or period in config).")

    # ── Session options ──────────────────────────────────────────────────────
    # One family, one shape: every flag about a session starts with the thing
    # it acts on and ends in -session, so the three sit together in --help and
    # read as a set.  The older spellings (--resume, --name,
    # --restart-session) still parse and are hidden, because a flag people
    # have in muscle memory is a flag that cannot simply be removed.
    session_opts = parser.add_argument_group(
        "session",
        "Which conversation this run is about. Without any of these, terminal "
        "chat continues the last session.")
    session_opts.add_argument("--new-session", "--restart-session", action="store_true",
                              default=False, dest="restart_session",
                              help="Start a fresh session instead of continuing the "
                                   "last one. Combine with --name-session to name it.")
    session_opts.add_argument("--resume-session", "--resume", type=str, default=None,
                              dest="resume", metavar="NAME_OR_ID",
                              help="Continue a particular session: its name, a UUID, "
                                   "or 'last' for the most recent.")
    session_opts.add_argument("--name-session", "--name", type=str, default=None,
                              dest="name", metavar="NAME",
                              help="Name a new session as it starts, so 'onit "
                                   "--resume-session NAME' finds it later. Implies "
                                   "--new-session; to rename a session that already "
                                   "exists, use 'onit sessions --rename'.")
    session_opts.add_argument("--session-limit", type=int, default=20,
                              dest="session_limit", metavar="N",
                              help="How many rows 'onit sessions' lists, so that a "
                                   "row number means the same session here "
                                   "(default: 20).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the configuration YAML file.")
    parser.add_argument("--host", type=str, default=None,
                        help="LLM serving host URL (e.g. http://localhost:8000/v1). "
                             "Overrides config and ONIT_HOST env var.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to use (e.g. Qwen/Qwen3-30B-A3B-Instruct-2507). "
                             "Skips auto-detection from endpoint.")
    parser.add_argument("--max-tokens", type=_token_count, default=None,
                        dest="max_tokens", metavar="N",
                        help="Max output tokens per model response, e.g. 32768, "
                             "128k or 1M. Each request is still clamped to what "
                             "is left of the context window. Overrides "
                             "serving.max_tokens in the config YAML "
                             "(default: 32768).")
    parser.add_argument("--max-context-tokens", type=_token_count, default=None,
                        dest="max_context_tokens", metavar="N",
                        help="Context window size in tokens, e.g. 128k or 1M. "
                             "Normally detected from the endpoint; set it when "
                             "the server does not report its own, or to hold the "
                             "agent to a smaller window than the model allows. "
                             "On Ollama this also sizes num_ctx unless "
                             "serving.num_ctx says otherwise. Overrides "
                             "serving.max_context_tokens in the config YAML "
                             "(default: whatever the endpoint reports, else "
                             "262144).")
    parser.add_argument("--verbose", action="store_true", default=None,
                        help="Enable verbose logging.")
    parser.add_argument("--think", action="store_true", default=None,
                        help="Enable thinking/reasoning mode (CoT).")
    parser.add_argument("--show-logs", action="store_true", default=None,
                        help="Show tool execution logs.")

    # ── Isolation ────────────────────────────────────────────────────────────
    parser.add_argument("--data-path", type=str, default=None, dest="data_path",
                        help="Working directory for agent files (default: ~/sandbox). "
                             "Overrides data_path in the config YAML.")
    parser.add_argument("--unrestricted", action="store_true", default=False,
                        help="Run with unrestricted host filesystem access. "
                             "The agent can read/write any path, use any working directory, "
                             "and install packages freely. Use only in trusted environments.")
    # Default on for a run that belongs to the person who started it, off for
    # one that serves other people (see the approval wiring further down), so
    # the flag is passed with a value only to override that.
    parser.add_argument("--auto", action="store_true", default=None, dest="auto",
                        help="Answer yes to every command approval prompt, so a run "
                             "never stops to ask. On by default for your own runs "
                             "(terminal chat, --loop, one-shot); pass it explicitly "
                             "to get the same for a web deployment, "
                             "which serves other people and so keeps asking. Only "
                             "questions that would have been put to a person are "
                             "answered: privilege escalation, container and "
                             "remote-shell tools, operator deny rules, and the "
                             "session path jail on a shared web deployment are still "
                             "refused.")
    parser.add_argument("--no-auto", "--ask", action="store_false", dest="auto",
                        help="Ask before running a command policy will not run on "
                             "its own, instead of approving it automatically.")
    parser.add_argument("--container", action="store_true", default=False,
                        help="Run the entire OnIt process inside a hardened Docker container "
                             "so a breach cannot reach the host OS.")
    parser.add_argument("--container-gpus", type=str, default=None, dest="container_gpus",
                        help='Pass GPUs into the container (e.g. "all" or "device=0,1"). '
                             "Requires the NVIDIA Container Toolkit on the host.")
    parser.add_argument("--container-mount", type=str, action="append", default=None,
                        dest="container_mount",
                        help="Extra bind mount for the container, e.g. "
                             "/host/path:/container/path:ro. Repeatable.")
    parser.add_argument("--container-memory", type=str, default=None,
                        dest="container_memory",
                        help="Hard memory cap for the container (e.g. 16g).")
    parser.add_argument("--container-shm-size", type=str, default=None,
                        dest="container_shm_size",
                        help="/dev/shm size inside the container (default: 4g).")
    parser.add_argument("--container-tmp-size", type=str, default=None,
                        dest="container_tmp_size",
                        help="/tmp tmpfs size inside the container (default: 16g).")
    parser.add_argument("--container-allow-installs", action="store_true",
                        default=False, dest="container_allow_installs",
                        help="Permit package installs inside the container. "
                             "Installs are still restricted to version-pinned "
                             "packages (e.g. pip install name==1.2.3).")

    # ── External MCP servers ─────────────────────────────────────────────────
    return parser


def _rename_session(identifier: str, new_name: str, sessions_dir: str,
                    limit: int = 20) -> int:
    """Rename one session; return the process exit code.

    Shared by ``onit rename``, ``onit sessions --rename`` and the ``\\rename``
    chat command's CLI twin, so the spellings of the same action cannot drift
    apart -- in what they accept as an identifier, in what they do with a bad
    name, or in what they tell the user afterwards.
    """
    from .sessions import _load_index, normalize_tag, resolve_session_choice, \
        tag_session

    name = normalize_tag(new_name)
    if not name:
        print("Error: a session name cannot be empty.", file=sys.stderr)
        return 1
    sid = resolve_session_choice(identifier, sessions_dir, limit=limit)
    if not sid:
        print(f"Error: no session matches '{identifier}'.", file=sys.stderr)
        print("Run 'onit sessions' to see what is there.", file=sys.stderr)
        return 1
    result = tag_session(sid, name, sessions_dir)
    if result is not True:
        print(f"Error: {result}", file=sys.stderr)
        return 1
    # Read the name back rather than echoing the argument: what is stored is
    # what 'onit --resume-session' will look up.
    stored = _load_index(sessions_dir).get(sid, {}).get("tag") or name
    print(f"Renamed {sid[:8]}... to '{stored}'.")
    print(f"Continue it with: onit --resume-session '{stored}'")
    return 0


def _parse_and_resolve_config(args: argparse.Namespace) -> dict:
    """Load the config file, merge setup defaults, and apply CLI overrides.

    Returns the fully-resolved config dict ready for use.
    """
    # resolve config file
    config_path = args.config or _find_default_config()
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    else:
        config_data = {}
        if args.config:
            print(f"Warning: config file '{args.config}' not found, using defaults.",
                  file=sys.stderr)

    # Merge ~/.onit/config.yaml (from 'onit setup') as a base layer.
    # Setup values fill in gaps but never override the project/user config.
    from .setup import CONFIG_PATH as _setup_config_path, resolve_credential
    _resolved_config = os.path.realpath(config_path) if os.path.isfile(config_path) else None
    _setup_resolved = os.path.realpath(_setup_config_path)
    if (_resolved_config != _setup_resolved
            and os.path.isfile(_setup_config_path)):
        with open(_setup_config_path, 'r') as f:
            setup_data = yaml.safe_load(f) or {}
        # Deep-merge: setup_data is the base, config_data overrides
        _merge_base(config_data, setup_data)
        config_data = setup_data

    # Map top-level CLI flags that still exist
    for arg_name, config_key in [
        ('verbose', 'verbose'),
        ('show_logs', 'show_logs'),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            config_data[config_key] = value

    # serve subcommand: inject mode-specific settings into config_data
    if getattr(args, 'command', None) == 'serve':
        serve_mode = getattr(args, 'serve_mode', None)
        if serve_mode == 'web':
            config_data['web'] = True
            if args.port is not None:
                config_data['web_port'] = args.port
            if getattr(args, 'no_login', False):
                config_data['web_require_auth'] = False
            if getattr(args, 'voice', False):
                voice_cfg = dict(config_data.get('voice') or {})
                voice_cfg['enabled'] = True
                config_data['voice'] = voice_cfg
        elif serve_mode == 'loop':
            config_data['loop'] = True
            config_data['task'] = args.task
            if getattr(args, 'period', None) is not None:
                config_data['period'] = args.period

    # Keyed off the resolved flag, not the subcommand, so a config file that
    # says ``web: true`` gets the same treatment as ``onit serve web``.  The
    # --think flag is only read here to keep the sampling consistent with it;
    # the flag itself is applied below, so it still wins outright.
    _apply_web_serving_defaults(config_data, bool(config_data.get('web')),
                                bool(getattr(args, 'think', False)))

    # --host, --model, --think override serving config
    if args.host:
        serving_cfg = config_data.setdefault('serving', {})
        serving_cfg['host'] = args.host
        # An explicit --host means a single endpoint: drop any second host
        # from config/env so the load balancer can't route requests to a
        # leftover server (e.g. a vLLM host2 shadowing an explicitly
        # requested Ollama host, which is fallback-only).
        for key in ('host2', 'model2', 'host2_key'):
            serving_cfg.pop(key, None)
        os.environ.pop('ONIT_HOST2', None)
        # Same reasoning for a configured endpoints list — it would otherwise
        # take precedence over the host the user just named on the CLI.
        serving_cfg.pop('endpoints', None)
    if args.model:
        config_data.setdefault('serving', {})['model'] = args.model
    if getattr(args, 'max_tokens', None) is not None:
        config_data.setdefault('serving', {})['max_tokens'] = args.max_tokens
    if getattr(args, 'max_context_tokens', None) is not None:
        serving_cfg = config_data.setdefault('serving', {})
        serving_cfg['max_context_tokens'] = args.max_context_tokens
        # Ollama allocates its KV cache from num_ctx and otherwise auto-sizes it
        # to a ceiling well below a 1M window, so an explicit context size has to
        # reach that knob too — unless the config names num_ctx itself, which is
        # the more specific setting and keeps precedence.
        serving_cfg.setdefault('num_ctx', args.max_context_tokens)
    if args.think:
        config_data.setdefault('serving', {})['think'] = True
    if args.data_path:
        config_data['data_path'] = args.data_path

    # Check that essential environment variables are set
    serving = config_data.get('serving', {})
    host = serving.get('host') or os.environ.get('ONIT_HOST')
    host_key = serving.get('host_key', '')
    # serving.endpoints supplies the hosts on its own; OnIt validates the list
    # and reports per-entry problems when it builds the load balancer.
    endpoints_configured = bool(serving.get('endpoints'))

    # Resolve host_key from keyring if not set via config/env. Only for
    # OpenRouter/OpenAI hosts — the stored legacy key is a key for that
    # provider, and injecting it into serving.host_key for a vLLM host would
    # shadow VLLM_API_KEY / the vllm_api_key keychain entry with the wrong key.
    #
    # A key stored for this endpoint's own URL is more specific than the
    # provider-wide one, and chat() will find it; injecting the legacy key
    # here would put it in front and authenticate as the wrong account.
    from src.setup import endpoint_key_source, legacy_key_for
    endpoint_key = bool(host) and endpoint_key_source(host) == 'endpoint'
    if (not host_key or host_key == 'EMPTY') and not endpoint_key and host:
        _keyring_key, _env_var, _label, _required = legacy_key_for(host)
        # Ollama and vLLM keys are not serving credentials in the same sense:
        # Ollama's doubles as the web-search key, and a vLLM host wants no
        # key at all — neither must land in serving.host_key.
        if _label not in ('Ollama cloud', 'vLLM'):
            kr_key = resolve_credential(None, _env_var, _keyring_key)
            if kr_key:
                host_key = kr_key
                config_data.setdefault('serving', {})['host_key'] = host_key

    missing = []
    if not host and not endpoints_configured:
        missing.append('ONIT_HOST (or set serving.host in config, or run: onit setup)')
    elif 'openrouter' in (host or '').lower():
        if not host_key and not endpoint_key:
            missing.append('OPENROUTER_API_KEY (or run: onit setup)')
    elif 'api.openai.com' in (host or '').lower():
        if not host_key and not endpoint_key:
            missing.append('OPENAI_API_KEY (or run: onit setup)')

    if missing:
        print("Error: missing required configuration:", file=sys.stderr)
        for var in missing:
            print(f"  - {var}", file=sys.stderr)
        print("\nSet via environment variable, CLI option (--host), config YAML, "
              "or run: onit setup", file=sys.stderr)
        sys.exit(1)

    # API keys: resolved from env vars and keyring only
    ollama_api_key = resolve_credential(None, 'OLLAMA_API_KEY', 'ollama_api_key')
    if ollama_api_key:
        os.environ['OLLAMA_API_KEY'] = ollama_api_key
    else:
        os.environ['ONIT_DISABLE_WEB_SEARCH'] = '1'

    # Optional first tier for web search. Without it the search tool still
    # works via Ollama / DuckDuckGo, so a missing key is not an error.
    tavily_api_key = resolve_credential(None, 'TAVILY_API_KEY', 'tavily_api_key')
    if tavily_api_key:
        os.environ['TAVILY_API_KEY'] = tavily_api_key

    weather_api_key = resolve_credential(None, 'OPENWEATHERMAP_API_KEY', 'openweathermap_api_key')
    if not weather_api_key:
        weather_api_key = resolve_credential(None, 'OPENWEATHER_API_KEY', 'openweathermap_api_key')
    if weather_api_key:
        os.environ['OPENWEATHERMAP_API_KEY'] = weather_api_key
    else:
        os.environ['ONIT_DISABLE_WEATHER'] = '1'

    # Web UI login: Google OAuth2 credentials live in the keyring (stored by
    # 'onit setup'); a value in the config file takes precedence.
    if config_data.get('web'):
        for key, env_var in [('web_google_client_id', 'GOOGLE_CLIENT_ID'),
                             ('web_google_client_secret', 'GOOGLE_CLIENT_SECRET')]:
            if not config_data.get(key):
                val = resolve_credential(None, env_var, key)
                if val:
                    config_data[key] = val

    return config_data


def _setup_servers(config_data: dict) -> None:
    """Start MCP servers and print tool-availability warnings."""
    _ensure_mcp_servers(
        config_data,
        log_level='DEBUG' if config_data.get('verbose') else 'ERROR',
    )

    # Print tool availability warnings before launching any mode
    if os.environ.get('ONIT_DISABLE_WEATHER'):
        print("Warning: OPENWEATHERMAP_API_KEY is not set. Weather tool is unavailable.",
              file=sys.stderr)
        print("  Set via env var or run: onit setup\n", file=sys.stderr)
    if os.environ.get('ONIT_DISABLE_WEB_SEARCH'):
        print("WARNING: OLLAMA_API_KEY is not set or invalid. "
              "Internet search is DISABLED.", file=sys.stderr)
        print("  OnIt will NOT be able to search the web in this session.",
              file=sys.stderr)
        print("  Set via env var or run: onit setup\n", file=sys.stderr)


def _dispatch_mode(config_data: dict) -> None:
    """Instantiate OnIt and launch the appropriate run mode."""
    onit = OnIt(config=config_data)
    asyncio.run(onit.run())


_ENDPOINT_PROBE_TIMEOUT = 20


def _probe_all_endpoints(args: argparse.Namespace, config_data: dict) -> int:
    """Smoke-test every known endpoint with a model listing; return the
    number that failed.

    The battery's own endpoint check only reaches the endpoint this session
    is assigned — the one that will serve.  This reaches the rest: the
    presets and every host with a stored key, so a dead key or an
    unreachable server is reported here, once, instead of the next time it
    is pointed at.  Each probe is bounded (a model listing is a small
    request; 20s covers a cold start), and rows print as they land, so one
    slow endpoint shows progress instead of a frozen terminal.

    A probe that cannot run at all — no config, no endpoints — is a skip,
    not a failure, and counts nothing.
    """
    from .setup import known_endpoints
    from .model.serving.chat import list_models

    # Both imported here, so both are patchable at their home modules — the
    # same reason _run_doctor imports run_checks inside the function.
    endpoints = known_endpoints(config_data or {})
    if not endpoints:
        print("  endpoints: none known — run 'onit setup' to configure one")
        return 0

    async def _probe(ep: dict) -> tuple:
        label = ep.get("preset") or ("config" if ep["configured"]
                                     else "stored key")
        try:
            names = await asyncio.wait_for(
                list_models(ep["host"], timeout=_ENDPOINT_PROBE_TIMEOUT),
                _ENDPOINT_PROBE_TIMEOUT)
            return (ep, label, f"ok — {len(names)} model(s)", None)
        except Exception as e:
            detail = (f"timed out after {_ENDPOINT_PROBE_TIMEOUT}s"
                      if isinstance(e, asyncio.TimeoutError)
                      else f"{type(e).__name__}: {e}"[:160])
            return (ep, label, None, detail)

    async def _run_all() -> list:
        results = []
        tasks = [asyncio.create_task(_probe(ep)) for ep in endpoints]
        for fut in asyncio.as_completed(tasks):
            ep, label, ok, err = await fut
            mark = "✓" if ok else "✗"
            print(f"  {mark} {ep['host']:<40} [{label}] "
                  f"{ok or err}")
            results.append((ep, ok, err))
        return results

    results = asyncio.run(_run_all())
    failed = sum(1 for _, ok, _ in results if not ok)
    return failed


def _run_doctor(args: argparse.Namespace, config_data: dict) -> int:
    """Run the self-check battery against a throwaway session; return exit code.

    The battery is written for a live session, so it needs a live agent: this
    builds one exactly the way a real session would — MCP servers started,
    tools discovered, a session registered — and deletes the session again
    afterwards.  The checks then read the same attributes (config_data,
    tool_registry, load_balancer, ...) they read in the text UI, which is
    the point: a pass here means the same thing a pass in the session means.
    """
    from .sessions import delete_session
    from .ui.doctor import render_report, run_checks

    # A doctor run is a diagnostic, not a deployment: strip the mode keys so
    # OnIt builds the plain terminal-chat shape (no web server, no loop) and
    # never auto-resumes someone's last real session. The a2a/gateway keys
    # are legacy config; they are stripped so old config files cannot turn a
    # diagnostic into a server.
    for key in ('web', 'a2a', 'gateway', 'loop', 'resume_session_id',
                'web_require_auth'):
        config_data.pop(key, None)
    # The battery's own probes are allowlisted, but the deep tool-calling turn
    # asks the model to run bash — give it the same standing a terminal chat
    # would, so the check fails on a broken loop rather than on a refused
    # approval nobody is there to answer.
    os.environ['ONIT_APPROVAL_CHANNEL'] = '1'
    os.environ['ONIT_AUTO_APPROVE'] = '1'

    try:
        # OnIt() prints the banner, the tool list and the balancer summary as
        # it starts.  In --json mode those prints would sit in front of the
        # JSON on stdout and make it unparseable, so they are captured and
        # dropped there; text mode keeps them as the run's startup trail.
        import contextlib
        import io
        sink = io.StringIO() if args.json else None
        with contextlib.redirect_stdout(sink if sink is not None
                                        else sys.__stdout__):
            agent = OnIt(config=config_data)
    except Exception as e:
        print(f"Error: could not start the agent for self-check: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    session_id = getattr(agent, "session_id", None)
    sessions_dir = os.path.expanduser(
        config_data.get('session_path', '~/.onit/sessions'))
    try:
        results = asyncio.run(run_checks(agent, deep=args.deep))
    finally:
        # The session existed only to give the battery something to check.
        # Kept only under --keep-session, for reading a failure's details in
        # the JSONL; the index entry goes with it either way.
        if session_id and not args.keep_session:
            delete_session(session_id, sessions_dir)

    if args.json:
        print(json.dumps({
            "deep": args.deep,
            "results": [vars(r) if not hasattr(r, "__dataclass_fields__")
                        else {"name": r.name, "state": r.state,
                              "detail": r.detail, "elapsed": r.elapsed}
                        for r in results],
        }, indent=2))
    else:
        print(render_report(results, deep=args.deep))

    failed = sum(1 for r in results if r.state == "fail")

    # --endpoints: the battery's endpoint check only reaches the endpoint
    # this session is assigned; this reaches every known one, so a dead
    # key or an unreachable server is reported here, once.
    if getattr(args, "endpoints", False):
        if not args.json:
            print()
            print("  Endpoint smoke test (model listing, "
                  f"{_ENDPOINT_PROBE_TIMEOUT}s each)")
            print("  " + "─" * 50)
        failed += _probe_all_endpoints(args, config_data)

    return 1 if failed else 0


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # --container: re-exec the whole process inside a hardened Docker container.
    # Must happen before any config load or server setup touches the host.
    if getattr(args, 'container', False):
        from .container_launcher import run as _container_run, strip_launcher_args
        sys.exit(_container_run(
            strip_launcher_args(sys.argv[1:]),
            gpus=getattr(args, 'container_gpus', None),
            mounts=getattr(args, 'container_mount', None) or [],
            memory=getattr(args, 'container_memory', None),
            shm_size=getattr(args, 'container_shm_size', None),
            tmp_size=getattr(args, 'container_tmp_size', None),
            allow_installs=getattr(args, 'container_allow_installs', False),
        ))

    # Setup wizard
    if args.command == "setup":
        if args.delete:
            from .setup import delete_named_secret
            failed = False
            for name in args.delete:
                ok, message = delete_named_secret(name)
                print(f"  {message}")
                failed = failed or not ok
            if failed:
                sys.exit(1)
            return
        from .setup import run_setup
        run_setup(show_only=args.show)
        return

    # Sessions management
    if args.command == "sessions":
        from .sessions import (list_sessions, count_sessions,
                               format_sessions_table, rebuild_index,
                               clear_sessions)
        sessions_dir = os.path.expanduser("~/.onit/sessions")
        if args.clear:
            answer = input("This will delete ALL session history. Are you sure? (yes/no): ")
            if answer.strip().lower() in ("yes", "y"):
                count = clear_sessions(sessions_dir)
                print(f"Deleted {count} session(s).")
            else:
                print("Cancelled.")
            return
        if args.rebuild:
            print("Rebuilding session index...")
            rebuild_index(sessions_dir)
            print("Done.")
        if args.rename:
            identifier, new_name = args.rename
            code = _rename_session(identifier, new_name, sessions_dir,
                                   limit=max(args.limit, 1))
            if code:
                sys.exit(code)
            return
        rows = list_sessions(sessions_dir, limit=args.limit,
                             include_empty=args.all)
        total = count_sessions(sessions_dir, include_empty=args.all)
        print(format_sessions_table(rows, total=total))
        if not args.all and not rows:
            print("No sessions with any turns yet. "
                  "Use --all to include the empty ones.")
        return

    # Trajectory store: read-only, and deliberately not gated on a config file
    # or a reachable model — "is anything being recorded" has to be answerable
    # when the rest of the stack is down.
    if args.command == "learn":
        from .learn import format_status, read_session, summarize
        # Only the learn block is needed, so the file is read directly rather
        # than run through the full resolver — that one reaches for the
        # keychain and the MCP hosts to answer a question about a directory.
        _learn_config_path = args.config or _find_default_config()
        config_data = {}
        if os.path.isfile(_learn_config_path):
            with open(_learn_config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        if args.session:
            records = read_session(args.session, config_data)
            if not records:
                print(f"No trajectory recorded for session '{args.session}'.",
                      file=sys.stderr)
                sys.exit(1)
            print(json.dumps(records, indent=2))
            return
        if args.json:
            summary = summarize(config_data)
            summary["models"] = dict(summary["models"])
            summary["ratings"] = dict(summary["ratings"])
            summary["totals"] = dict(summary["totals"])
            print(json.dumps(summary, indent=2))
            return
        if args.events:
            from .learn.events import summarize_events
            events = summarize_events(config_data)
            if not events["total"]:
                print("No loop events recorded yet. They appear when the "
                      "tool lifecycle starts emitting (registry loads, "
                      "tool loads/updates/archives).")
                return
            print(json.dumps(events, indent=2))
            return
        if args.holdout:
            _learn_draw_holdout(args, config_data)
            return
        if args.lifecycle:
            _learn_lifecycle(args, config_data)
            return
        if args.skills:
            _learn_skills(args, config_data)
            return
        if args.eval_run:
            _learn_eval_run(args, config_data)
            return
        if args.playbook or args.reflect or args.rollback:
            _learn_playbook(args, config_data)
            return
        if args.signals:
            from .learn.corpus import (format_summary, label_corpus,
                                       write_labels)
            summary = label_corpus(config_data)
            if getattr(args, "write_signals", False):
                path = write_labels(config_data)
                if path is None:
                    print("Could not write the derived labels — recording "
                          "is off, or the learned path is not writable.",
                          file=sys.stderr)
                    sys.exit(1)
                print("Wrote %s" % path)
            if getattr(args, "json", False):
                print(json.dumps(summary, indent=2))
            else:
                print(format_summary(summary))
            return
        if args.eval:
            from .learn.eval import summarize as eval_summary
            print(json.dumps(eval_summary(config_data), indent=2))
            return
        print(format_status(config_data))
        return

    # doctor: the live self-check battery, from the shell.  Runs the same
    # checks as \doctor in the text UI against a throwaway session, prints
    # the report and exits non-zero when anything failed, so it can gate a
    # deploy or sit at the end of an update script.  The battery is the
    # product here, not a conversation: nothing is dispatched to a mode.
    if args.command == "doctor":
        # A machine that has never been set up must still be diagnosable —
        # "your config is missing serving.host" is exactly the kind of answer
        # a self-check exists to give.  Resolution exits when the config is
        # incomplete; catching that leaves an empty dict for the battery to
        # report on instead of dying before it ran anything.
        try:
            config_data = _parse_and_resolve_config(args)
        except SystemExit:
            config_data = {}
        # The servers are the thing under test: start this process's own
        # before the battery polls their ports.
        _setup_servers(config_data)
        sys.exit(_run_doctor(args, config_data))

    # rename: a verb of its own, because renaming is the one session action a
    # person reaches for while a session is still in front of them.
    if args.command == "rename":
        sessions_dir = os.path.expanduser("~/.onit/sessions")
        code = _rename_session(args.session, args.name, sessions_dir,
                               limit=max(args.limit, 1))
        if code:
            sys.exit(code)
        return

    # resume subcommand: translate to --resume flag and continue normal startup
    if args.command == "resume":
        args.resume = args.session

    config_data = _parse_and_resolve_config(args)

    # Session selection.  The terminal chat continues where it left off, so an
    # explicit --resume-session is optional: without one we continue the last
    # session.  --new-session opts out and starts from scratch.  Server modes
    # (web/loop) manage their own sessions and never auto-resume.
    #
    # --name-session names a session that is being created, so it implies
    # --new-session.  Before, it was quietly ignored whenever a session was
    # being resumed -- which was the default -- so 'onit --name foo' started a
    # name you could not then find, having continued somebody else's
    # conversation under it.
    if args.name and args.resume:
        print("Error: --name-session names a new session, so it cannot be "
              "combined with --resume-session.", file=sys.stderr)
        print("To rename an existing session: onit sessions --rename "
              "<name|uuid|#> <new name>", file=sys.stderr)
        sys.exit(1)

    resume_target = args.resume
    auto_resume = False
    if not resume_target and not args.restart_session and not args.name and not (
            config_data.get('web') or config_data.get('loop')):
        resume_target = "last"
        auto_resume = True

    if resume_target:
        from .sessions import resolve_session_choice
        sessions_dir = os.path.expanduser(
            config_data.get('session_path', '~/.onit/sessions'))
        sid = resolve_session_choice(resume_target, sessions_dir,
                                     limit=max(getattr(args, 'session_limit', 20), 1))
        # An index entry can outlive its JSONL file; treat that as "no session".
        if sid and not os.path.exists(os.path.join(sessions_dir, f"{sid}.jsonl")):
            sid = None
        if not sid:
            # Nothing to resume is only an error when the user asked for a
            # specific session; the automatic path just starts a new one.
            if not auto_resume:
                print(f"Error: Session '{resume_target}' not found.", file=sys.stderr)
                print("Use 'onit sessions' to list available sessions.", file=sys.stderr)
                sys.exit(1)
        else:
            config_data['resume_session_id'] = sid
            if auto_resume:
                print(f"Resuming last session: {sid[:8]}... "
                      "(use --new-session to start a new one)")
            else:
                print(f"Resuming session: {sid[:8]}...")

    # A name given on the command line belongs to a session being created.  It
    # is carried in the config dict rather than passed as an argument because
    # OnIt._setup_session is what creates the session, and it reads the config.
    if args.name:
        config_data['session_name'] = args.name
        print(f"Naming this session '{args.name}'. "
              f"Return to it with: onit --resume-session '{args.name}'")


    # Must be set before MCP servers are spawned so child processes inherit it.
    if args.unrestricted:
        os.environ['ONIT_UNRESTRICTED'] = '1'
        print("Warning: running in unrestricted mode — agent has full host filesystem access.",
              file=sys.stderr)

    # Web UI mode: enforce ~/.onit/settings.json permission rules in the bash
    # MCP server. The text UI ignores the default settings file and runs
    # privileged. In the container this flag additionally seals off package
    # installs (see command_policy.installs_sealed). Must be set before MCP
    # servers are spawned.
    if config_data.get('web'):
        os.environ['ONIT_WEB_UI'] = '1'

    # Declare, before the tool servers are spawned and inherit it, whether
    # this run can answer at all about a command the policy will not run on
    # its own. Only the two interactive front ends have a person to ask: the
    # terminal chat and the web UI, both of which implement the prompt. A
    # --loop run has nobody at the other end, so unless something else is
    # answering it keeps refusing outright rather than minting approval
    # tickets nobody will ever answer. Set here rather than inferred inside
    # the servers, because "is anyone watching" is a fact about how OnIt was
    # started and nothing downstream can recover it.
    #
    # Automatic approval is a channel of its own: the answer comes from the
    # switch rather than from a person, so it works in the modes that have no
    # one attached — which is where an unattended run actually needs it.
    _interactive = not config_data.get('loop')

    # Whether the prompts are answered by the flag rather than by a person.
    # Unset means take the default, and the default turns on whose run this
    # is. A terminal chat, a --loop and a one-shot belong to whoever started
    # them: they already hold the shell OnIt is running commands from, so
    # stopping to ask them protects nobody, and the prompt is one more thing
    # to get stuck on. A web deployment answers to people who are not the
    # operator — on a shared host the approval prompt is part of what keeps
    # one session out of another's way — so it keeps asking unless --auto
    # says otherwise in as many words.
    _serves_others = bool(config_data.get('web'))
    _auto = (not _serves_others) if args.auto is None else args.auto
    _auto_by_default = _auto and args.auto is None

    if _interactive or _auto:
        os.environ['ONIT_APPROVAL_CHANNEL'] = '1'
    else:
        os.environ.pop('ONIT_APPROVAL_CHANNEL', None)

    if _auto:
        # Set for the harness, which is what answers the prompts; the tool
        # servers never read it. Approving is still answering a question the
        # policy chose to ask, so this cannot reach anything the policy would
        # have refused outright — see docs/ISOLATION.md, "Command Approvals".
        os.environ['ONIT_AUTO_APPROVE'] = '1'
        if _auto_by_default:
            # Short, because it prints on every ordinary run. Each approval
            # is still reported as it happens, so this only has to say which
            # way the switch is set and how to move it.
            print("Command approvals: approving automatically "
                  "(--no-auto to be asked instead).", file=sys.stderr)
        else:
            _where = ("every session of this deployment"
                      if config_data.get('web') else "this run")
            print(f"Warning: --auto approves command prompts automatically for "
                  f"{_where}. Commands outside the session directory, unlisted "
                  f"executables and unpinned installs will run without asking.",
                  file=sys.stderr)
        if os.environ.get('ONIT_ASK_APPROVAL') == '0':
            print("Warning: ONIT_ASK_APPROVAL=0 is also set, which refuses "
                  "those commands instead of asking. Nothing is left for "
                  "--auto to approve.", file=sys.stderr)
    else:
        os.environ.pop('ONIT_AUTO_APPROVE', None)

    _setup_servers(config_data)
    _dispatch_mode(config_data)


if __name__ == "__main__":
    main()
