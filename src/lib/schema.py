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

Argument validation against the JSON Schema an MCP server declares.

A tool call arrives as whatever JSON the model produced.  Until now the only
checks were repairs — quote-swapping malformed JSON, unwrapping a value the
model nested inside its own key, refusing a required string left blank — so a
call that parsed but was *wrong* went to the server and came back as a stack
trace.  The model then has to read a traceback to learn that `depth` wanted a
number.  Naming the problem in the tool's own vocabulary is both cheaper (no
round trip) and easier to act on.

Deliberately not `jsonschema`: this validates the handful of keywords MCP
servers actually emit, and the dependency is unavailable here.  Deliberately
not pydantic either — pydantic validates *Python types*, and what arrives is a
JSON Schema dict, so every tool would need a model built for it at discovery
time.  That is more machinery than the keywords below justify.
'''

from typing import Any

# JSON Schema type name → the Python types that satisfy it.  bool is excluded
# from the numeric types on purpose: Python says True is an int, JSON Schema
# does not, and a model that passes true for a count has made a real mistake.
_TYPE_CHECKS: dict[str, Any] = {
    'string': lambda v: isinstance(v, str),
    'integer': lambda v: isinstance(v, int) and not isinstance(v, bool),
    'number': lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    'boolean': lambda v: isinstance(v, bool),
    'array': lambda v: isinstance(v, list),
    'object': lambda v: isinstance(v, dict),
    'null': lambda v: v is None,
}


def _type_names(prop: dict) -> list[str]:
    """The types a property accepts, flattening the ``anyOf`` MCP servers emit
    for optional parameters (``anyOf: [{type: string}, {type: null}]``)."""
    names: list[str] = []
    declared = prop.get('type')
    if isinstance(declared, str):
        names.append(declared)
    elif isinstance(declared, list):
        names.extend(t for t in declared if isinstance(t, str))
    for branch in prop.get('anyOf') or []:
        if isinstance(branch, dict):
            names.extend(_type_names(branch))
    return names


def _describe(value: Any) -> str:
    """What the model actually sent, in JSON's vocabulary rather than Python's —
    it wrote JSON, so `true` and `null` are what it will recognize."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        return repr(value) if len(value) <= 40 else repr(value[:37] + '...')
    if isinstance(value, (list, dict)):
        return {list: 'an array', dict: 'an object'}[type(value)]
    return repr(value)


def coerce_arguments(schema: dict, arguments: dict) -> tuple[dict, list[str]]:
    """Repair the mistakes worth repairing, and report what was changed.

    Small models routinely send `"3"` where a number was asked for and
    `"true"` where a boolean was.  Refusing those costs a whole round trip to
    fix something unambiguous, so they are coerced.  Anything ambiguous is left
    alone for ``validate_arguments`` to refuse: `"yes"` is not a boolean, and
    guessing that it is would be the harness inventing an argument.

    Returns the (possibly new) arguments and a list of human-readable notes.
    The input dict is never mutated — a caller that wants the original after a
    failed validation still has it.
    """
    if not isinstance(schema, dict) or not isinstance(arguments, dict):
        return arguments, []
    props = schema.get('properties')
    if not isinstance(props, dict):
        return arguments, []

    out = dict(arguments)
    notes: list[str] = []
    for name, value in arguments.items():
        prop = props.get(name)
        if not isinstance(prop, dict):
            continue
        types = _type_names(prop)
        if not types or _satisfies(types, value):
            continue
        if not isinstance(value, str):
            continue
        text = value.strip()
        coerced = _coerce_scalar(text, types)
        if coerced is not _NO_COERCION:
            out[name] = coerced
            notes.append(f"{name}: {_describe(value)} → {_describe(coerced)}")
    return out, notes


_NO_COERCION = object()


def _coerce_scalar(text: str, types: list[str]) -> Any:
    """A string onto the first declared type that unambiguously accepts it."""
    for type_name in types:
        if type_name == 'integer':
            try:
                return int(text)
            except ValueError:
                continue
        elif type_name == 'number':
            try:
                return float(text)
            except ValueError:
                continue
        elif type_name == 'boolean':
            low = text.lower()
            # Only JSON's own spellings.  "yes"/"1" are guesses, and a harness
            # that guesses an argument is worse than one that asks again.
            if low in ('true', 'false'):
                return low == 'true'
        elif type_name == 'array':
            # A single value where a list was asked for is the common shape of
            # this mistake, but it is only safe when the item type agrees;
            # leave it to the tool otherwise.
            continue
    return _NO_COERCION


def _satisfies(types: list[str], value: Any) -> bool:
    """True when the value matches any declared type, or none is recognized —
    an unknown type name is a schema we cannot judge, not a failed argument."""
    known = [t for t in types if t in _TYPE_CHECKS]
    if not known:
        return True
    return any(_TYPE_CHECKS[t](value) for t in known)


def validate_arguments(schema: dict, arguments: dict) -> list[str]:
    """Problems with ``arguments`` against ``schema``, each a readable sentence.

    An empty list means dispatch.  Checks the keywords MCP servers actually
    emit — ``required``, ``type``/``anyOf``, ``enum``, ``minimum``/``maximum``
    — and stays silent about everything else, because a schema this cannot
    judge must not become a refused tool call.

    Never raises: a malformed schema means the tool goes out unvalidated, which
    is exactly the behavior that preceded this function.
    """
    if not isinstance(schema, dict) or not isinstance(arguments, dict):
        return []
    problems: list[str] = []

    required = schema.get('required')
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in arguments:
                problems.append(f"{name}: required, but was not supplied")

    props = schema.get('properties')
    if not isinstance(props, dict):
        return problems

    # Parameters the tool does not define.  Refused only when the schema says
    # so — FastMCP emits ``additionalProperties: false``, and a server that
    # declares it will reject the call anyway, one network round trip later and
    # with a message the model has to reverse-engineer.
    #
    # This is the failure mode the recorded trajectories actually show: a model
    # reaching for ``read_file(offset=…, limit=…)`` because that is the read
    # tool it saw in training, on a tool whose parameters are ``path`` and
    # ``max_chars``.  Naming the real ones is what turns a guess into a fix, so
    # the message carries them.
    if schema.get('additionalProperties') is False:
        unknown = [n for n in arguments if n not in props]
        if unknown:
            accepted = ', '.join(props) or '(none)'
            for name in unknown:
                problems.append(
                    f"{name}: not a parameter of this tool; "
                    f"it accepts: {accepted}")

    for name, value in arguments.items():
        prop = props.get(name)
        if not isinstance(prop, dict):
            # Either the tool allows extras, or it does not and the block above
            # already said so.  Nothing further to check against.
            continue

        types = _type_names(prop)
        if types and not _satisfies(types, value):
            expected = ' or '.join(dict.fromkeys(t for t in types if t in _TYPE_CHECKS))
            problems.append(
                f"{name}: expected {expected}, got {_describe(value)}")
            # Range and enum are meaningless once the type is wrong, and a
            # second complaint about the same argument reads as two problems.
            continue

        choices = prop.get('enum')
        if isinstance(choices, list) and choices and value not in choices:
            shown = ', '.join(_describe(c) for c in choices[:8])
            if len(choices) > 8:
                shown += ', …'
            problems.append(
                f"{name}: expected one of [{shown}], got {_describe(value)}")
            continue

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            low, high = prop.get('minimum'), prop.get('maximum')
            if isinstance(low, (int, float)) and value < low:
                problems.append(f"{name}: must be >= {low}, got {_describe(value)}")
            elif isinstance(high, (int, float)) and value > high:
                problems.append(f"{name}: must be <= {high}, got {_describe(value)}")

    return problems
