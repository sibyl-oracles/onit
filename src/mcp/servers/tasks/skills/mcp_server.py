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

Skills MCP Server

Serves the skills the agent has learned -- the fragments it kept solving
the same way, promoted to tools (Loop C, docs/SELF_IMPROVEMENT.md 4.4).

What is served is exactly what `pinned.yaml` names, and nothing else. A
skill that has not been through the validation gate is not in the pin, so
it is not in the schema, so the model cannot call it. That is the whole
safety story of this server, and it is a short one: the pin is the
allowlist, and the server reads the pin and nothing else.

The tool each skill contributes is named after the skill, takes the
skill's declared input schema, and calls the skill's `run` function.
The skill's code is loaded from the skill's own directory -- not
imported from a path the model can influence -- and it runs in this
process, under the same policy the rest of the server runs under.

Optional environment variables:
    ONIT_LEARN_PATH  - the learned store to read skills and the pin from
'''

import importlib.util
import inspect
import json
import logging
import os
import sys
from typing import Annotated, Any, Optional

from fastmcp import FastMCP
from pydantic import Field

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("Skills MCP Server")

# The learned store.  Read from the environment so the server finds the
# same store the agent writes to, without a config file of its own.
LEARN_PATH = os.path.expanduser(os.environ.get('ONIT_LEARN_PATH',
                                                '~/.onit/learned'))

# The loaded skills, by name.  Populated at startup from the pin.
_SKILLS: dict = {}


def _config_data() -> dict:
    """The config shape the learn package expects."""
    return {"learn": {"path": LEARN_PATH}}


def _load_module(path: str, name: str):
    """Import a skill's file as a module.

    The path is built from the skill's own directory, which comes from
    the pin, which is not writable by the model at runtime. The module
    is given a unique name so two skills that both define `run` do not
    collide in `sys.modules`.
    """
    unique = "onit_skill_%s" % name
    spec = importlib.util.spec_from_file_location(unique, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    spec.loader.exec_module(module)
    return module


def _schema_for(manifest: dict) -> dict:
    """The input schema a skill declares, in MCP's shape.

    A skill with no declared schema gets a schema with no properties --
    an empty object -- rather than a free-form anything. A tool with no
    parameters is still a tool, and declaring "anything" is how a tool
    ends up being called with a hundred arguments.
    """
    schema = manifest.get("input_schema")
    if not isinstance(schema, dict):
        schema = {}
    out = {
        "type": schema.get("type", "object"),
        "properties": schema.get("properties") or {},
    }
    if schema.get("required"):
        out["required"] = list(schema["required"])
    if schema.get("description"):
        out["description"] = schema["description"]
    return out


def load_skills(config_data: dict | None = None) -> dict:
    """Load every pinned skill, and return them by name.

    A skill that fails to load is skipped with a log line rather than
    taking the server down with it. One broken skill should cost the
    agent that skill, not every other skill it has.
    """
    from src.learn.skills import pinned_skills, skill_dir

    config_data = config_data or _config_data()
    loaded = {}
    for name in sorted(pinned_skills(config_data)):
        directory = skill_dir(name, config_data)
        try:
            module = _load_module(os.path.join(directory, "tool.py"), name)
            run = getattr(module, "run", None)
            if run is None or not callable(run):
                raise AttributeError("the skill has no run() function")
            manifest = getattr(module, "MANIFEST", None)
            if not isinstance(manifest, dict):
                from src.learn.skills import read_manifest
                manifest = read_manifest(os.path.join(directory,
                                                      "manifest.yaml")) or {}
            loaded[name] = {"module": module, "run": run, "manifest": manifest,
                            "dir": directory}
        except Exception as e:
            logger.error("Skill %r not loaded: %s", name, e)
    return loaded


# The JSON schema types a manifest may declare, and the Python types
# they become in the tool's signature.
_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _make_handler(skill_run, schema: dict):
    """Build a tool handler with the signature the skill declares.

    FastMCP reads a tool's parameters off the function's signature, and
    it will not accept ``**kwargs`` -- a handler that swallows
    everything is a handler whose schema the model cannot see, which
    means the model cannot call it correctly.

    So the signature is built from the skill's declared schema: one
    keyword parameter per declared property, with the declared type and
    the declared default.  A skill that declares nothing gets a
    zero-argument handler, which is exactly as callable as it is
    described to be.
    """
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    def handler(**kwargs) -> str:
        try:
            result = skill_run(**kwargs)
        except Exception as e:
            return "The skill failed: %s: %s" % (type(e).__name__, e)
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, default=str)
        except (TypeError, ValueError):
            return str(result)

    # The signature is built by hand, so the model sees the schema the
    # skill declared rather than a generic "anything goes".
    #
    # The function is *compiled* rather than given a `__signature__`:
    # pydantic resolves a function's annotations from its globals, and
    # a parameter name that only exists in the schema is not in the
    # globals, so the annotation lookup raises `KeyError` and the tool
    # is never registered.  Compiling the source puts the names and the
    # annotations in the same namespace, which is what pydantic reads.
    params = []
    for pname, spec in properties.items():
        if not isinstance(spec, dict):
            spec = {}
        ptype = _JSON_TO_PY.get(spec.get("type"), str)
        if pname in required:
            default = inspect.Parameter.empty
        else:
            default = spec.get("default")
            if default is None and ptype is str:
                default = ""
        params.append(inspect.Parameter(
            pname, inspect.Parameter.KEYWORD_ONLY,
            default=default, annotation=ptype))

    return _compile_handler(handler, params, skill_run)


def _compile_handler(handler, params, skill_run):
    """Build a callable whose real signature matches the schema.

    The generated function has the declared parameters, each annotated
    with its declared type, and forwards them to the skill.  The
    annotations are real annotations in the generated function's own
    namespace, which is what pydantic reads to build the tool schema.
    """
    names = [p.name for p in params]
    if not names:
        # No declared parameters: a plain zero-argument function.
        def wrapper():
            return handler()
        wrapper.__name__ = handler.__name__
        return wrapper

    def _default_for(p):
        """The literal default for a parameter, as source text.

        A parameter with no default is required, and a parameter with a
        default is not.  Getting this wrong changes what the *model*
        is told the tool accepts: a required parameter the caller does
        not have is a call that fails, and the failure is the model's
        to discover at run time.
        """
        if p.default is inspect.Parameter.empty:
            return None
        if isinstance(p.default, bool):
            return "True" if p.default else "False"
        if isinstance(p.default, (int, float)):
            return repr(p.default)
        if isinstance(p.default, str):
            return repr(p.default)
        return repr(p.default)

    pieces = []
    for p in params:
        annotation = getattr(p.annotation, "__name__", "str")
        default = _default_for(p)
        if default is None:
            pieces.append("%s: %s" % (p.name, annotation))
        else:
            pieces.append("%s: %s = %s" % (p.name, annotation, default))
    signature = ", ".join(pieces)
    calls = ", ".join("%s=%s" % (n, n) for n in names)
    source = (
        "def _skill_tool(%s):\n"
        "    return _handler(%s)\n" % (signature, calls))

    namespace = {"_handler": handler}
    exec(compile(source, "<skill:%s>" % handler.__name__, "exec"), namespace)
    wrapper = namespace["_skill_tool"]
    wrapper.__name__ = handler.__name__
    wrapper.__doc__ = handler.__doc__
    return wrapper


def register_skills(loaded: dict) -> list:
    """Register each loaded skill as a tool.

    The tool's name is the skill's name and its description is the
    skill's description, because the description is what the model
    reads when it decides which tool to reach for. A skill whose
    description is empty does not get registered at all: an
    undescribed tool is a tool the model cannot choose, and a tool the
    model cannot choose is schema tokens spent for nothing.
    """
    registered = []
    for name, entry in sorted(loaded.items()):
        manifest = entry["manifest"]
        description = str(manifest.get("description") or "").strip()
        if not description:
            logger.error("Skill %r has no description; not registered", name)
            continue
        run = entry["run"]
        schema = _schema_for(manifest)

        handler = _make_handler(run, schema)
        handler.__name__ = name
        try:
            mcp.tool(name=name, description=description)(handler)
            registered.append(name)
        except Exception as e:
            logger.error("Skill %r not registered: %s", name, e)
    return registered


def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18203,
    path: str = "/sse",
    options: dict = {}
) -> None:
    """Run the Skills MCP server."""
    global LEARN_PATH

    verbose = 'verbose' in options
    logger.setLevel(logging.INFO if verbose else logging.ERROR)

    if 'learn_path' in options:
        LEARN_PATH = options['learn_path']
    elif os.environ.get('ONIT_LEARN_PATH'):
        LEARN_PATH = os.environ['ONIT_LEARN_PATH']
    LEARN_PATH = os.path.abspath(os.path.expanduser(LEARN_PATH))

    loaded = load_skills(_config_data())
    registered = register_skills(loaded)
    _SKILLS.clear()
    _SKILLS.update(loaded)

    if registered:
        logger.info("Serving %d skill(s): %s", len(registered),
                    ", ".join(registered))
    else:
        logger.info("No pinned skills to serve. Pin one with "
                    "'onit learn --skills --pin NAME'.")

    if transport == 'stdio':
        logger.info("Starting Skills MCP Server on stdio")
    else:
        logger.info("Starting Skills MCP Server at %s:%s%s", host, port, path)

    run_kwargs = {'show_banner': False}
    if 'show_banner' not in inspect.signature(mcp.run).parameters:
        run_kwargs.pop('show_banner')
    if transport == 'stdio':
        mcp.run(transport='stdio', **run_kwargs)
    else:
        mcp.run(transport=transport, host=host, port=port, path=path,
                **run_kwargs)
