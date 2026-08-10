"""Tests for src/lib/schema.py — argument validation against a tool's JSON Schema."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.schema import coerce_arguments, validate_arguments


def _schema(properties, required=None):
    out = {"type": "object", "properties": properties}
    if required is not None:
        out["required"] = required
    return out


# ── types ───────────────────────────────────────────────────────────────────

class TestTypes:
    def test_matching_types_pass(self):
        schema = _schema({"query": {"type": "string"}, "depth": {"type": "integer"}})
        assert validate_arguments(schema, {"query": "cats", "depth": 3}) == []

    def test_type_mismatch_names_the_parameter_and_both_types(self):
        schema = _schema({"depth": {"type": "integer"}})
        problems = validate_arguments(schema, {"depth": "three"})
        assert len(problems) == 1
        assert "depth" in problems[0]
        assert "integer" in problems[0]
        assert "'three'" in problems[0]

    def test_bool_is_not_an_integer(self):
        """Python says True is an int; JSON Schema does not, and a model that
        passes true for a count has made a real mistake."""
        schema = _schema({"depth": {"type": "integer"}})
        assert validate_arguments(schema, {"depth": True}) != []

    def test_int_satisfies_number(self):
        schema = _schema({"ratio": {"type": "number"}})
        assert validate_arguments(schema, {"ratio": 2}) == []

    def test_anyof_with_null_accepts_both(self):
        """The shape MCP servers emit for an optional parameter."""
        schema = _schema({"topic": {"anyOf": [{"type": "string"}, {"type": "null"}]}})
        assert validate_arguments(schema, {"topic": "physics"}) == []
        assert validate_arguments(schema, {"topic": None}) == []
        assert validate_arguments(schema, {"topic": 42}) != []

    def test_type_as_list_accepts_any_listed(self):
        schema = _schema({"limit": {"type": ["integer", "null"]}})
        assert validate_arguments(schema, {"limit": 5}) == []
        assert validate_arguments(schema, {"limit": None}) == []
        assert validate_arguments(schema, {"limit": "5"}) != []

    def test_array_and_object(self):
        schema = _schema({"paths": {"type": "array"}, "opts": {"type": "object"}})
        assert validate_arguments(schema, {"paths": ["a"], "opts": {"k": 1}}) == []
        assert "an object" in validate_arguments(schema, {"paths": {}})[0]

    def test_unknown_type_name_is_not_judged(self):
        """A schema we cannot judge must not become a refused tool call."""
        schema = _schema({"blob": {"type": "bytes"}})
        assert validate_arguments(schema, {"blob": 123}) == []


# ── required ────────────────────────────────────────────────────────────────

class TestRequired:
    def test_missing_required_is_reported(self):
        schema = _schema({"query": {"type": "string"}}, required=["query"])
        problems = validate_arguments(schema, {})
        assert len(problems) == 1
        assert "query" in problems[0]

    def test_present_required_passes(self):
        schema = _schema({"query": {"type": "string"}}, required=["query"])
        assert validate_arguments(schema, {"query": "x"}) == []

    def test_explicit_null_is_not_missing(self):
        """blank_required_args owns the blank/None case and has a better error
        for it; this check is only about absence."""
        schema = _schema({"query": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
                         required=["query"])
        assert validate_arguments(schema, {"query": None}) == []


# ── enum and range ──────────────────────────────────────────────────────────

class TestEnumAndRange:
    def test_bad_enum_lists_the_choices(self):
        schema = _schema({"mode": {"type": "string", "enum": ["fast", "deep"]}})
        problems = validate_arguments(schema, {"mode": "medium"})
        assert len(problems) == 1
        assert "'fast'" in problems[0] and "'deep'" in problems[0]

    def test_good_enum_passes(self):
        schema = _schema({"mode": {"type": "string", "enum": ["fast", "deep"]}})
        assert validate_arguments(schema, {"mode": "deep"}) == []

    def test_long_enum_is_elided(self):
        schema = _schema({"k": {"type": "string", "enum": [str(i) for i in range(20)]}})
        problems = validate_arguments(schema, {"k": "nope"})
        assert "…" in problems[0]

    def test_below_minimum(self):
        schema = _schema({"limit": {"type": "integer", "minimum": 1}})
        assert ">= 1" in validate_arguments(schema, {"limit": 0})[0]

    def test_above_maximum(self):
        schema = _schema({"limit": {"type": "integer", "maximum": 100}})
        assert "<= 100" in validate_arguments(schema, {"limit": 500})[0]

    def test_within_range_passes(self):
        schema = _schema({"limit": {"type": "integer", "minimum": 1, "maximum": 100}})
        assert validate_arguments(schema, {"limit": 50}) == []

    def test_wrong_type_reports_once_not_twice(self):
        """Range and enum are meaningless once the type is wrong, and two
        complaints about one argument read as two problems."""
        schema = _schema({"limit": {"type": "integer", "minimum": 1, "enum": [1, 2]}})
        assert len(validate_arguments(schema, {"limit": "many"})) == 1


# ── things that must never refuse a call ────────────────────────────────────

class TestUnknownParameters:
    """The failure the recorded trajectories actually show: a model reaching
    for parameters the tool does not define (read_file(offset=..., limit=...))."""

    def test_allowed_when_the_schema_is_silent(self):
        schema = _schema({"query": {"type": "string"}})
        assert validate_arguments(schema, {"query": "x", "extra": 1}) == []

    def test_refused_when_additional_properties_is_false(self):
        schema = _schema({"path": {"type": "string"}})
        schema["additionalProperties"] = False
        problems = validate_arguments(schema, {"path": "a.txt", "offset": 0})
        assert len(problems) == 1
        assert "offset" in problems[0]

    def test_the_error_names_the_real_parameters(self):
        """Naming them is what turns a guess into a fix."""
        schema = _schema({"path": {"type": "string"},
                          "max_chars": {"type": "integer"}})
        schema["additionalProperties"] = False
        problems = validate_arguments(schema, {"path": "a.txt", "limit": 10})
        assert "path" in problems[0] and "max_chars" in problems[0]

    def test_every_unknown_parameter_is_listed(self):
        schema = _schema({"path": {"type": "string"}})
        schema["additionalProperties"] = False
        problems = validate_arguments(schema, {"path": "a", "offset": 0, "limit": 5})
        assert len(problems) == 2

    def test_declared_parameters_still_pass(self):
        schema = _schema({"path": {"type": "string"},
                          "data_path": {"type": "string"}})
        schema["additionalProperties"] = False
        assert validate_arguments(schema, {"path": "a", "data_path": "/tmp"}) == []

    def test_harness_injected_params_are_declared_so_they_pass(self):
        """session_id/data_path are injected only when tool_accepts_param says
        the tool declares them, so they are never 'unknown' at this point."""
        schema = _schema({"q": {"type": "string"},
                          "session_id": {"type": "string"},
                          "data_path": {"type": "string"}})
        schema["additionalProperties"] = False
        assert validate_arguments(
            schema, {"q": "x", "session_id": "s", "data_path": "/d"}) == []


class TestNeverRefuses:
    def test_unknown_parameter_is_allowed_by_default(self):
        """Only a schema that explicitly forbids extras rejects them."""
        schema = _schema({"query": {"type": "string"}})
        assert validate_arguments(schema, {"query": "x", "session_id": "abc"}) == []

    def test_empty_schema_checks_nothing(self):
        assert validate_arguments({}, {"anything": object()}) == []

    def test_malformed_schema_does_not_raise(self):
        for bad in (None, [], "schema", {"properties": "not a dict"},
                    {"properties": {"q": "not a dict"}}, {"required": "q"}):
            assert validate_arguments(bad, {"q": 1}) == []

    def test_non_dict_arguments_do_not_raise(self):
        assert validate_arguments(_schema({"q": {"type": "string"}}), None) == []


# ── coercion ────────────────────────────────────────────────────────────────

class TestCoercion:
    def test_numeric_string_to_integer(self):
        schema = _schema({"depth": {"type": "integer"}})
        args, notes = coerce_arguments(schema, {"depth": "3"})
        assert args["depth"] == 3
        assert len(notes) == 1 and "depth" in notes[0]

    def test_numeric_string_to_number(self):
        schema = _schema({"ratio": {"type": "number"}})
        args, _ = coerce_arguments(schema, {"ratio": "0.5"})
        assert args["ratio"] == 0.5

    def test_json_boolean_spellings(self):
        schema = _schema({"deep": {"type": "boolean"}})
        assert coerce_arguments(schema, {"deep": "true"})[0]["deep"] is True
        assert coerce_arguments(schema, {"deep": "False"})[0]["deep"] is False

    def test_ambiguous_boolean_is_left_alone(self):
        """A harness that guesses an argument is worse than one that asks again."""
        schema = _schema({"deep": {"type": "boolean"}})
        args, notes = coerce_arguments(schema, {"deep": "yes"})
        assert args["deep"] == "yes"
        assert notes == []
        assert validate_arguments(schema, args) != []

    def test_non_numeric_string_is_left_alone(self):
        schema = _schema({"depth": {"type": "integer"}})
        args, notes = coerce_arguments(schema, {"depth": "three"})
        assert args["depth"] == "three"
        assert notes == []

    def test_already_valid_values_are_untouched(self):
        schema = _schema({"depth": {"type": "integer"}, "q": {"type": "string"}})
        args, notes = coerce_arguments(schema, {"depth": 3, "q": "5"})
        assert args == {"depth": 3, "q": "5"}
        assert notes == []

    def test_input_dict_is_not_mutated(self):
        """A caller that wants the original after a failed validation has it."""
        schema = _schema({"depth": {"type": "integer"}})
        original = {"depth": "3"}
        coerce_arguments(schema, original)
        assert original == {"depth": "3"}

    def test_coercion_then_validation_passes(self):
        schema = _schema({"depth": {"type": "integer", "maximum": 10}})
        args, _ = coerce_arguments(schema, {"depth": "7"})
        assert validate_arguments(schema, args) == []

    def test_coerced_value_still_range_checked(self):
        schema = _schema({"depth": {"type": "integer", "maximum": 10}})
        args, _ = coerce_arguments(schema, {"depth": "99"})
        assert validate_arguments(schema, args) != []

    def test_malformed_schema_does_not_raise(self):
        for bad in (None, [], {"properties": "nope"}):
            args, notes = coerce_arguments(bad, {"q": "1"})
            assert args == {"q": "1"} and notes == []
