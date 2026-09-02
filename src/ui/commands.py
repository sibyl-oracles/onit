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
"""Backslash commands for the text chat session.

``\\bye`` was the only one, and it is handled a level up as a stop command.
The rest answer the questions a session raises about itself — what am I
connected to, what is it serving, point me somewhere else — without dropping
out of the chat to edit YAML and start over.

The dispatcher is deliberately conservative about what it claims: a line is a
command only when its first word names one.  Anything else is the user's
message and goes to the model untouched, so prose, LaTeX (``\\frac{a}{b}``)
and Windows paths still work.  The one exception is a lone unrecognised
``\\word``, which is far more likely to be a typo than a question, and is
answered with the nearest command rather than sent off to be answered.
"""

import asyncio
import difflib
import os
import re
from dataclasses import dataclass

from ..model.serving.balancer import LoadBalancer, ServerEndpoint

# Aliases the session already ends on. They never reach dispatch() — OnIt
# checks its stop_commands first — but naming them here keeps \exit out of
# the "unknown command" path and puts them in the help listing.
STOP_ALIASES = ("bye", "exit", "quit", "goodbye")

# How many model ids a listing prints before it is cut short. An OpenRouter
# endpoint serves hundreds; the point of the list is to recognise a name, not
# to scroll the panel away.
MAX_LISTED_MODELS = 30


@dataclass(frozen=True)
class Command:
    name: str
    args: str
    summary: str

    @property
    def usage(self) -> str:
        return f"\\{self.name}" + (f" {self.args}" if self.args else "")


COMMANDS = (
    Command("help", "", "Show this list."),
    Command("doctor", "[deep]",
            "Run the live self-check: servers, tools, prompts, endpoint. "
            "'deep' adds a real model reply and a tool-calling turn "
            "(costs tokens)."),
    Command("setup", "", "Show the endpoints, keys and paths this session is using."),
    Command("model", "[name | -]",
            "Switch model; bare, list what the endpoint serves; '-' for "
            "auto-detect."),
    Command("host", "[<url> | add <url> | rm <n>]",
            "Use one endpoint, add one to the rotation, or drop one; bare, "
            "list them."),
    Command("key", "[<n | url> | rm <n | url>]",
            "Set an endpoint's API key, typed without echo; bare, the one "
            "serving this session."),
    Command("save", "",
            "Write the session's endpoints to the config file so the next "
            "one starts with them."),
    Command("bye", "", "End the session (also \\exit, \\quit, \\goodbye)."),
)

_BY_NAME = {c.name: c for c in COMMANDS}
_KNOWN = set(_BY_NAME) | set(STOP_ALIASES)


# ── parsing ─────────────────────────────────────────────────────────────────

def parse(text: str) -> tuple[str, str] | None:
    """Split a line into (command, argument), or None when it is not one.

    Returns ``("", word)`` for a lone unrecognised ``\\word`` so the caller can
    tell "typo" apart from "not a command at all".
    """
    if not text:
        return None
    line = text.strip()
    if not line.startswith("\\"):
        return None
    head, _, rest = line.partition(" ")
    name = head[1:].lower()
    if name in _KNOWN:
        return name, rest.strip()
    # A bare backslash word that names nothing: a typo, not a message.
    if not rest and name.isalpha():
        return "", name
    return None


def suggest(name: str) -> str:
    """The closest command to a mistyped one, or '' when nothing is close."""
    match = difflib.get_close_matches(name, sorted(_BY_NAME), n=1, cutoff=0.6)
    return match[0] if match else ""


def unknown(name: str) -> str:
    near = suggest(name)
    hint = f" Did you mean \\{near}?" if near else ""
    return f"\\{name} is not a command.{hint} Type \\help for the list."


# ── rendering helpers ───────────────────────────────────────────────────────

def render_help() -> str:
    width = max(len(c.usage) for c in COMMANDS)
    lines = ["Commands"]
    lines += [f"  {c.usage:<{width}}  {c.summary}" for c in COMMANDS]
    lines.append("")
    lines.append("Anything else is sent to the model, so a message may still "
                 "start with a backslash.")
    return "\n".join(lines)


def _key_label(endpoint) -> str:
    """Where this endpoint's API key comes from, in one word.

    Worth a column of its own: a 401 on the first task is otherwise the only
    place the answer shows up, and the fallback case — no key of the
    endpoint's own, a provider-named secret still covering for it — is exactly
    the one a user cannot guess.
    """
    in_config = bool(endpoint.host_key and endpoint.host_key != "EMPTY")
    try:
        from .. import setup as onit_setup
        return onit_setup.endpoint_key_label(endpoint.host, in_config)
    except Exception:
        return "in config" if in_config else "?"


def _endpoint_rows(agent) -> list[str]:
    """The endpoint table: preference order, with the active one marked."""
    lb = agent.load_balancer
    active = lb.assigned(getattr(agent, "session_id", None))
    order = sorted(range(len(lb.endpoints)),
                   key=lambda i: (lb.endpoints[i].priority, i))
    hosts = [ep.host for ep in lb.endpoints]
    hw = max(len("HOST"), max(len(h) for h in hosts))
    models = [ep.model or "auto-detect" for ep in lb.endpoints]
    mw = max(len("MODEL"), max(len(m) for m in models))
    keys = [_key_label(ep) for ep in lb.endpoints]
    kw = max(len("KEY"), max(len(k) for k in keys))
    rows = [f"  {'#':>2}  {'PRIO':<4}  {'HOST':<{hw}}  {'MODEL':<{mw}}  "
            f"{'KEY':<{kw}}  NAME"]
    held_back = False
    for i in order:
        ep = lb.endpoints[i]
        mark = " *" if ep is active else ""
        if not ep.is_healthy():
            state = "  (cooling down)"
        elif lb.is_fallback_only(ep):
            state = "  (fallback only)"
            held_back = True
        else:
            state = ""
        rows.append(f"  {i + 1:>2}  {ep.priority:<4}  {ep.host:<{hw}}  "
                    f"{models[i]:<{mw}}  {keys[i]:<{kw}}  "
                    f"{ep.name}{mark}{state}")
    rows.append(f"  * serves this session ({lb.algorithm})")
    if held_back:
        rows.append("  (fallback only) Ollama endpoints stay out of rotation "
                    "while any other endpoint is healthy.")
    return rows


def render_setup(agent) -> str:
    """What this session is actually running on, secrets masked."""
    # Imported here, not at module scope: the credential lookups reach the OS
    # keychain, and tests patch src.setup.get_secret to keep them out of it.
    from .. import setup as onit_setup

    lines = ["Endpoints"]
    lines += _endpoint_rows(agent)

    # Each endpoint's own key is already in the KEY column above; these are
    # only worth naming while one of them is still standing in for a missing
    # one, which the column reports by name when it happens.
    legacy = [(k, label, env) for k, label, env
              in onit_setup.LEGACY_SERVING_SECRETS if onit_setup.get_secret(k)]
    if legacy:
        lines += ["", "Legacy fallback keys"]
        width = max(len(label) for _, label, _ in legacy)
        for key, label, env_var in legacy:
            lines.append(f"  {label:.<{width + 2}} "
                         f"{onit_setup.secret_status(key, env_var)}")

    lines += ["", "Session"]
    ui = getattr(agent, "chat_ui", None)
    for label, value in (
        ("session id", getattr(agent, "session_id", "") or "-"),
        ("session log", getattr(agent, "session_path", "") or "-"),
        ("data path", getattr(agent, "data_path", "") or "-"),
        ("config file", onit_setup.CONFIG_PATH
         if os.path.isfile(onit_setup.CONFIG_PATH) else "not written yet"),
        ("token budgets", ui._fmt_ctx_label()
         if hasattr(ui, "_fmt_ctx_label") else "-"),
    ):
        lines.append(f"  {label:.<16} {value}")

    notes = _config_notes()
    if notes:
        lines += ["", "Notes"] + [f"  - {n}" for n in notes]
    return "\n".join(lines)


def _config_notes() -> list[str]:
    """Endpoint/key mismatches worth naming, from the saved config."""
    try:
        from .. import setup as onit_setup
        return onit_setup._provider_notes(onit_setup._load_config())
    except Exception:
        return []


# ── \model ──────────────────────────────────────────────────────────────────

async def _list_models(endpoint) -> tuple[list[str], str]:
    """Model ids the endpoint serves, or ([], reason) when it cannot say."""
    from ..model.serving.chat import list_models
    try:
        return await list_models(endpoint.host, endpoint.host_key), ""
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def _format_models(names: list[str]) -> list[str]:
    shown = names[:MAX_LISTED_MODELS]
    lines = [f"  {n}" for n in shown]
    if len(names) > len(shown):
        lines.append(f"  ... and {len(names) - len(shown)} more")
    return lines


async def cmd_model(agent, arg: str) -> str:
    lb = agent.load_balancer
    ep = lb.assigned(getattr(agent, "session_id", None))
    where = f"{ep.name or 'endpoint'} ({ep.host})"

    if not arg:
        current = ep.model or "auto-detect (first model the endpoint lists)"
        lines = [f"Model on {where}: {current}"]
        names, error = await _list_models(ep)
        if names:
            lines += ["", f"Serving {len(names)} model(s):"] + _format_models(names)
        elif error:
            lines += ["", f"Could not list models: {error}"]
        lines += ["", "Switch with \\model <name>, or \\model - for auto-detect."]
        return "\n".join(lines)

    if arg == "-":
        ep.model = None
        _forget_detected_model(ep.host)
        _mirror_model(agent, ep, None)
        return (f"Model on {where} is back to auto-detect; the next task uses "
                f"whatever it lists first.")

    name = arg.split()[0]
    names, error = await _list_models(ep)
    ep.model = name
    _mirror_model(agent, ep, name)
    msg = f"Model on {where} is now {name}. It takes effect on the next task."
    if names and name not in names:
        msg += (f"\n\nWarning: {ep.host} did not list that model. If it 404s, "
                f"run \\model to see what it serves.")
    elif error:
        msg += f"\n\nCould not verify it against the endpoint: {error}"
    return msg


def _forget_detected_model(host: str) -> None:
    """Drop a cached auto-detection so '-' really re-asks the endpoint."""
    try:
        from ..model.serving import chat as chat_mod
        chat_mod._MODEL_ID_CACHE.pop(host, None)
    except Exception:
        pass


def _mirror_model(agent, endpoint, model: str | None) -> None:
    """Keep the single-host mirrors in step with the endpoint we just edited.

    ``serving.model`` and ``chat_ui.model_name`` are read by code that expects
    one representative model, and both would otherwise keep naming the old one
    on screen until the next answer arrived.
    """
    if endpoint is agent.load_balancer.preferred:
        if model:
            agent.model_serving["model"] = model
        else:
            agent.model_serving.pop("model", None)
    ui = getattr(agent, "chat_ui", None)
    if ui is not None:
        ui.model_name = model or ""


# ── \host ───────────────────────────────────────────────────────────────────

_REMOVE_VERBS = ("rm", "remove", "delete", "del")

# Closing note on every change: the config file is not written, and it is
# worth saying so every time rather than leaving a user to find out at the
# next launch that the endpoint they added is gone.
_NOT_PERSISTED = ("The change lasts until you quit — the config file is "
                  "untouched. Run \\save to keep it, or 'onit setup' to edit "
                  "the file directly.")


def _valid_url(url: str) -> str | None:
    if url.startswith(("http://", "https://")):
        return None
    return (f"{url} is not an endpoint URL — it needs an http:// or "
            f"https:// scheme, e.g. \\host http://localhost:8000/v1.")


def _unique_name(lb, base: str = "manual") -> str:
    """A label no endpoint in the rotation is already using."""
    taken = {ep.name for ep in lb.endpoints}
    if base not in taken:
        return base
    n = 2
    while f"{base}{n}" in taken:
        n += 1
    return f"{base}{n}"


def _rebuild(agent, endpoints: list,
             ollama_fallback_only: bool | None = None) -> None:
    """Put a new endpoint list behind the session and resync the mirrors.

    A fresh LoadBalancer rather than a mutated one: sticky assignments are
    held as indices into the endpoint list, so inserting or removing an entry
    under them would silently re-point a session at a different server. The
    dropped assignment costs one re-pick on the next task, which is also what
    makes an added endpoint reachable at all under the sticky algorithm.

    ``ollama_fallback_only`` defaults to carrying the current rule forward.
    """
    lb = agent.load_balancer
    agent.load_balancer = LoadBalancer(
        endpoints, lb.algorithm,
        ollama_fallback_only=(lb.ollama_fallback_only
                              if ollama_fallback_only is None
                              else ollama_fallback_only))
    preferred = agent.load_balancer.preferred
    agent.model_serving["host"] = preferred.host
    if preferred.model:
        agent.model_serving["model"] = preferred.model
    else:
        agent.model_serving.pop("model", None)


def _resolve_endpoint(lb, token: str):
    """The endpoint a row number or URL names, or None."""
    if token.isdigit():
        row = int(token)
        if 1 <= row <= len(lb.endpoints):
            return lb.endpoints[row - 1]
        return None
    for ep in lb.endpoints:
        if ep.host == token or ep.name == token:
            return ep
    return None


def _host_add(agent, args: list[str]) -> str:
    lb = agent.load_balancer
    # An explicit opt out of the implicit "Ollama last" rule, for the session.
    # A flag rather than a positional word, which would be indistinguishable
    # from a model name.
    share = "--share" in args
    args = [a for a in args if a != "--share"]
    if not args:
        return "\\host add needs a URL, e.g. \\host add http://gpu-2:8000/v1."
    url, model = args[0], (args[1] if len(args) > 1 else None)
    bad = _valid_url(url)
    if bad:
        return bad
    existing = next((ep for ep in lb.endpoints if ep.host == url), None)
    if existing is not None:
        if not share:
            return (f"{url} is already in the rotation. Change its model with "
                    f"\\model, or drop it with \\host rm {url}.")
        # The message on a held-back add tells the user to type this exact
        # line; refusing it because the endpoint is already there would send
        # them back to remove-and-re-add for what is only a routing change.
        if not lb.is_fallback_only(existing):
            return (f"{url} is already in the rotation and already taking an "
                    f"equal share — nothing to change.")
        _rebuild(agent, list(lb.endpoints), ollama_fallback_only=False)
        lines = [f"{url} now takes an equal share of the rotation."]
        lines += [""] + _endpoint_rows(agent)
        lines += ["",
                  "--share turned off the Ollama fallback rule for this whole "
                  "session, so every Ollama endpoint in the list is affected.",
                  "", _NOT_PERSISTED]
        return "\n".join(lines)

    # Joins the endpoint that is currently preferred rather than starting a
    # tier of its own: "also use this one" means share the traffic, not sit
    # in reserve behind everything.
    endpoint = ServerEndpoint(host=url, model=model,
                              name=_unique_name(lb),
                              priority=lb.preferred.priority)
    _rebuild(agent, list(lb.endpoints) + [endpoint],
             ollama_fallback_only=False if share else None)

    others = len(lb.endpoints)
    lines = [f"Added {url} (model: {model or 'auto-detect'}) at priority "
             f"{endpoint.priority}, alongside "
             f"{others} other endpoint{'' if others == 1 else 's'}."]
    lines += [""] + _endpoint_rows(agent)

    new_lb = agent.load_balancer
    if new_lb.is_fallback_only(endpoint):
        # The priority in the message above is true and still not the whole
        # story: this endpoint will serve nothing until the others fail.
        lines += ["",
                  f"It will not serve traffic yet. Ollama endpoints are held "
                  f"back while any other endpoint is healthy, so this one is "
                  f"a standby. Add it with '\\host add {url} --share' to put "
                  f"it in rotation on equal terms for this session, or set "
                  f"serving.ollama_fallback_only: false to make that the "
                  f"default."]
    elif share:
        lines += ["",
                  "--share turned off the Ollama fallback rule for this "
                  "whole session, so every Ollama endpoint in the list now "
                  "takes an equal share."]
    elif new_lb.algorithm == "sticky":
        lines.append("")
        lines.append("Sticky routing pins a session to one endpoint, so this "
                     "session re-picks across the tier on the next task "
                     "rather than staying where it was.")
    lines += ["", _NOT_PERSISTED]
    return "\n".join(lines)


def _host_remove(agent, args: list[str]) -> str:
    lb = agent.load_balancer
    if not args:
        return ("\\host rm needs a row number or URL from the table — "
                "run \\host to see it.")
    if len(lb.endpoints) == 1:
        return ("That is the only endpoint left; there would be nothing to "
                "send the next task to. Point the session elsewhere with "
                "\\host <url> instead.")
    endpoint = _resolve_endpoint(lb, args[0])
    if endpoint is None:
        return (f"No endpoint matches {args[0]!r} — run \\host for the row "
                f"numbers and URLs.")

    _rebuild(agent, [ep for ep in lb.endpoints if ep is not endpoint])
    lines = [f"Dropped {endpoint.host} from the rotation."]
    lines += [""] + _endpoint_rows(agent)
    lines += ["", _NOT_PERSISTED]
    return "\n".join(lines)


def _host_switch(agent, args: list[str]) -> str:
    lb = agent.load_balancer
    url, model = args[0], (args[1] if len(args) > 1 else None)
    bad = _valid_url(url)
    if bad:
        return bad

    replaced = [ep.host for ep in lb.endpoints]
    _rebuild(agent, [ServerEndpoint(host=url, model=model, name="manual")])
    ui = getattr(agent, "chat_ui", None)
    if ui is not None:
        ui.model_name = model or ""

    lines = [f"Endpoint is now {url} "
             f"(model: {model or 'auto-detect'}). It takes effect on the "
             f"next task."]
    if len(replaced) > 1:
        lines.append("")
        lines.append(f"This replaces the {len(replaced)} endpoints that were "
                     f"in rotation ({', '.join(replaced)}), so nothing is "
                     f"load-balanced for the rest of the session. Use "
                     f"'\\host add <url>' to widen the rotation instead of "
                     f"replacing it.")
    lines += ["", _NOT_PERSISTED]
    return "\n".join(lines)


async def cmd_host(agent, arg: str) -> str:
    """List, switch, widen or narrow the endpoints this session talks to.

    A bare URL replaces the rotation, which is the "point me at that server
    instead" case; ``add`` and ``rm`` edit it in place, mirroring the [a]dd
    and [d]elete of the wizard's endpoint editor.
    """
    parts = arg.split()
    if not parts:
        lines = ["Endpoints"] + _endpoint_rows(agent)
        lines += ["",
                  "\\host <url> [model]      use this endpoint alone",
                  "\\host add <url> [model]  add it to the rotation",
                  "                         (--share overrides the Ollama "
                  "fallback rule)",
                  "\\host rm <n | url>       drop one"]
        return "\n".join(lines)

    verb = parts[0].lower()
    if verb == "add":
        return _host_add(agent, parts[1:])
    if verb in _REMOVE_VERBS:
        return _host_remove(agent, parts[1:])
    return _host_switch(agent, parts)


# ── \key ────────────────────────────────────────────────────────────────────

async def cmd_key(agent, arg: str) -> str:
    """Store an API key for one endpoint, read without echo.

    The key is never an argument on the command line: that line is drawn as
    it is typed, kept in the input history, and shown in the chat panel
    afterwards. It is asked for separately instead, and only ever stored in
    the OS keychain — never in the config file.
    """
    from .. import setup as onit_setup

    lb = agent.load_balancer
    parts = arg.split()
    forget = bool(parts) and parts[0].lower() in _REMOVE_VERBS
    if forget:
        parts = parts[1:]

    if len(parts) > 1:
        return ("\\key takes only the endpoint — the key itself is asked for "
                "separately so it is never echoed or kept in the input "
                "history. Run \\key " + parts[0] + " on its own.")

    if parts:
        endpoint = _resolve_endpoint(lb, parts[0])
        if endpoint is None:
            return (f"No endpoint matches {parts[0]!r} — run \\host for the "
                    f"row numbers and URLs.")
    else:
        endpoint = lb.assigned(getattr(agent, "session_id", None))
    where = f"{endpoint.name or 'endpoint'} ({endpoint.host})"

    if forget:
        if not onit_setup.get_endpoint_key(endpoint.host):
            return f"{where} has no stored key of its own to forget."
        onit_setup.delete_endpoint_key(endpoint.host)
        return (f"Forgot the stored key for {where}. It now falls back to "
                f"{onit_setup.endpoint_key_label(endpoint.host)}.")

    ui = getattr(agent, "chat_ui", None)
    if ui is None or not hasattr(ui, "read_secret"):
        return ("This interface cannot read a key without echoing it. Use "
                "'onit setup', which asks for it the same way.")

    loop = asyncio.get_running_loop()
    value = await loop.run_in_executor(
        None, ui.read_secret, f"    API key for {endpoint.host}: ")
    if not value:
        return f"No key entered — {where} is unchanged."

    onit_setup.store_endpoint_key(endpoint.host, value)
    lines = [f"Stored ••••{value[-4:]} for {where}. It takes effect on the "
             f"next task."]
    if endpoint.host_key and endpoint.host_key != "EMPTY":
        # Otherwise the key is stored, reported, and silently never used.
        lines += ["",
                  "Note: this endpoint also has a key written into the config "
                  "file, which is read first — remove it there for the stored "
                  "one to take effect."]
    lines += ["", "The key is in the OS keychain, not the config file, and "
                  "unlike the rest of this session it does persist."]
    if not _is_saved(endpoint.host):
        # The key outlives the session; the endpoint it belongs to does not.
        # Left there it is a keychain entry for a URL nothing mentions, which
        # 'onit setup --show' has no row to report it on.
        lines += ["",
                  f"Note: {endpoint.host} is not in the config file, so the "
                  f"next session will not know about it and 'onit setup "
                  f"--show' will not list this key. Run \\save to keep the "
                  f"endpoint too."]
    return "\n".join(lines)


# ── \save ───────────────────────────────────────────────────────────────────

# Labels the code hands out when the user did not choose one: 'serverN' from
# the config reader, 'manual' from \host. Writing them back would turn every
# save into a named endpoint list, and a plain one- or two-host config would
# never again fit the short serving.host / serving.host2 form it was written
# in. A name the user actually typed is read from the file and kept.
_GENERATED_NAME = re.compile(r"^(server\d+|manual\d*)$")


def _saved_entries(config: dict) -> list[dict]:
    """The endpoints the config file currently holds, in whichever shape."""
    from .. import setup as onit_setup
    return (onit_setup._endpoint_list(config)
            or onit_setup._entries_from_host_pair(config))


def _saved_by_host(config: dict) -> dict:
    """Saved entries keyed by normalized URL, for looking one up by host."""
    from .. import setup as onit_setup
    return {onit_setup.normalize_host(e["host"]): e
            for e in _saved_entries(config)}


def _is_saved(host: str) -> bool:
    """True when the config file already names this endpoint."""
    from .. import setup as onit_setup
    try:
        return (onit_setup.normalize_host(host)
                in _saved_by_host(onit_setup._load_config()))
    except Exception:
        return False


def _entry_to_save(endpoint, saved: dict) -> dict:
    """One endpoint as a config entry, carrying nothing secret.

    The API key is taken from the saved entry, never from
    ``ServerEndpoint.host_key``: that field may hold a key the launcher
    resolved out of the OS keychain, and writing it here would copy a secret
    into a file that is world-readable, gets pasted into issues, and is the
    one place 'onit setup' deliberately keeps keys out of.
    """
    from .. import setup as onit_setup

    previous = saved.get(onit_setup.normalize_host(endpoint.host), {})
    entry = {"host": endpoint.host, "priority": endpoint.priority}
    if endpoint.model:
        entry["model"] = endpoint.model
    name = previous.get("name") or endpoint.name
    if name and not _GENERATED_NAME.match(name):
        entry["name"] = name
    literal = previous.get("api_key") or previous.get("host_key")
    if literal and literal != "EMPTY":
        entry["api_key"] = literal
    return entry


def cmd_save(agent) -> str:
    """Write the session's endpoint list into the config file.

    Only what a command in this session can change is written — the endpoints
    and the Ollama fallback rule \\host --share turns off. Everything else in
    the file is left exactly as it was found, including settings this session
    was started with on the command line, which are an override for one run
    and not an edit the user asked to keep.
    """
    from .. import setup as onit_setup

    lb = agent.load_balancer
    config = onit_setup._load_config()
    saved = _saved_by_host(config)
    before = set(saved)

    entries = [_entry_to_save(ep, saved) for ep in lb.endpoints]
    onit_setup._write_endpoints(config, entries)

    serving = config.setdefault("serving", {})
    fallback_changed = (lb.ollama_fallback_only
                        != serving.get("ollama_fallback_only", True))
    if fallback_changed:
        serving["ollama_fallback_only"] = lb.ollama_fallback_only
    onit_setup._save_config(config)

    after = {onit_setup.normalize_host(ep.host) for ep in lb.endpoints}
    added, removed = sorted(after - before), sorted(before - after)

    n = len(entries)
    lines = [f"Saved {n} endpoint{'' if n == 1 else 's'} to "
             f"{onit_setup.CONFIG_PATH}."]
    if added:
        lines.append(f"  added    {', '.join(added)}")
    if removed:
        lines.append(f"  removed  {', '.join(removed)}")
    if not added and not removed:
        lines.append("  The same endpoints were already saved; their models, "
                     "priorities and labels now match this session.")
    if fallback_changed:
        lines.append(f"  serving.ollama_fallback_only is now "
                     f"{str(lb.ollama_fallback_only).lower()}")

    lines += [""] + _endpoint_rows(agent)
    lines += ["",
              "API keys are not written here — they stay in the OS keychain, "
              "where \\key puts them."]
    return "\n".join(lines)


# ── \doctor ─────────────────────────────────────────────────────────────────────────────

async def cmd_doctor(agent, arg: str) -> str:
    """Run the live self-check battery and return the report.

    Awaited, not spun onto its own loop: ``dispatch()`` runs inside the
    session's event loop, and ``asyncio.run()`` from inside a running loop
    is exactly the crash a self-check must not be.  The fast battery is a
    few seconds; ``deep`` adds up to a couple of minutes on a slow endpoint,
    which the command's help text warns about.
    """
    from .doctor import render_report, run_checks

    deep = arg.strip().lower() == "deep"
    if arg.strip() and not deep:
        return ("\\doctor takes no argument, or 'deep'. "
                "'deep' adds a live model reply and a tool-calling turn.")

    # Progress goes through the UI's log panel rather than a spinner: the
    # checks are individually quick, and a spinner that flashes one name per
    # second reads as noise.  add_log is the one channel every front end has.
    ui = getattr(agent, "chat_ui", None)

    def _on_start(name: str) -> None:
        if ui is not None and hasattr(ui, "add_log"):
            ui.add_log(f"self-check: running {name}…", level="info")

    results = await run_checks(agent, deep=deep, on_start=_on_start)
    return render_report(results, deep=deep)


# ── dispatch ────────────────────────────────────────────────────────────────

async def dispatch(agent, text: str) -> str | None:
    """Run the command in ``text``, returning what to show the user.

    Returns None when the line is not a command, which means it is a message
    for the model.  A handler that raises is reported rather than propagated:
    a bad endpoint URL should cost a line of red text, not the session.
    """
    parsed = parse(text)
    if parsed is None:
        return None
    name, arg = parsed
    if not name:
        return unknown(arg)
    if name in STOP_ALIASES:
        return None  # OnIt.stop_commands owns these
    try:
        if name == "help":
            return render_help()
        if name == "setup":
            return render_setup(agent)
        if name == "model":
            return await cmd_model(agent, arg)
        if name == "host":
            return await cmd_host(agent, arg)
        if name == "key":
            return await cmd_key(agent, arg)
        if name == "save":
            return cmd_save(agent)
        if name == "doctor":
            return await cmd_doctor(agent, arg)
    except Exception as e:
        return f"\\{name} failed: {type(e).__name__}: {e}"
    return None
