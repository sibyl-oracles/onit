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

The lifecycle actuator: the half that makes the policy real.

``src/learn/lifecycle.py`` decided *whether*; this tests the part that
decides *what to do about it*, which is where the danger is.  An
actuator that can take a capability away is the one part of the
self-improvement track that can do irreversible harm, so the tests here
are mostly about the boundaries:

  * nothing is ever deleted, only withheld;
  * withholding is reversible;
  * the recovery tools are never withheld;
  * the actuator does nothing unless it was asked to.
"""

import os

import pytest

from src.learn import actuator
from src.learn import trajectory as traj


@pytest.fixture
def learned(tmp_path):
    """A learned store with recording on."""
    return {"learn": {"autonomy": "observe",
                      "path": str(tmp_path / "learned")}}


@pytest.fixture
def adaptive(tmp_path):
    """A learned store at the autonomy level the actuator needs."""
    return {"learn": {"autonomy": "adapt",
                      "path": str(tmp_path / "learned"),
                      "lifecycle": {"enabled": True}}}


def _record(store, session, task, tools, ok=True):
    """A recorded run that used ``tools``.

    The key is ``tool_runs``, not ``tools``: ``_turn_view`` reads the rich
    per-call records from ``tool_runs`` and only falls back to bare names in
    ``tools``.  Passing ``tools`` here would store a nested
    ``{"name": {"name": ...}}`` that no reader expects.
    """
    traj.record_task(session_id=session, turn=1, task=task, response="r",
                     config_data=store,
                     metrics={"turns": [{"n": 1, "tool_runs": [
                         {"name": name, "ok": ok} for name in tools]}]})


# ── the gates ─────────────────────────────────────────────────────────────

class TestGates:

    def test_the_actuator_is_off_by_default(self, learned):
        assert actuator.lifecycle_enabled(learned) is False

    def test_the_actuator_needs_the_adapt_level(self, tmp_path):
        config = {"learn": {"autonomy": "observe",
                            "path": str(tmp_path / "l"),
                            "lifecycle": {"enabled": True}}}
        assert actuator.lifecycle_enabled(config) is False

    def test_the_actuator_needs_to_be_enabled(self, tmp_path):
        config = {"learn": {"autonomy": "adapt",
                            "path": str(tmp_path / "l")}}
        assert actuator.lifecycle_enabled(config) is False

    def test_the_actuator_is_on_when_both_gates_open(self, adaptive):
        assert actuator.lifecycle_enabled(adaptive) is True

    def test_a_malformed_lifecycle_block_does_not_enable_it(self, tmp_path):
        config = {"learn": {"autonomy": "adapt",
                            "path": str(tmp_path / "l"),
                            "lifecycle": "yes"}}
        assert actuator.lifecycle_enabled(config) is False

    def test_nothing_is_applied_when_the_actuator_is_off(self, learned):
        _record(learned, "s1", "t", ["flaky_tool"], ok=False)
        result = actuator.apply(learned)
        assert result["applied"] is False
        assert result["withheld"] == []

    def test_the_decisions_are_still_reported_when_it_is_off(self, learned):
        # A deployment should be able to see the policy's mind before
        # letting it change anything.
        _record(learned, "s1", "t", ["flaky_tool"], ok=False)
        decisions = actuator.evaluate(learned)
        assert isinstance(decisions, list)


# ── the archive ───────────────────────────────────────────────────────────

class TestArchive:

    def test_a_missing_archive_reads_as_empty(self, learned):
        assert actuator.read_archive(learned) == {}
        assert actuator.withheld_tools(learned) == set()

    def test_a_broken_archive_reads_as_empty(self, learned):
        path = actuator.archive_path(learned)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        # The safe direction: an unreadable archive means nothing is
        # withheld, not everything.
        assert actuator.withheld_tools(learned) == set()

    def test_the_archive_round_trips(self, learned):
        actuator.write_archive({"tools": {"x": {"withheld": True}}}, learned)
        assert actuator.withheld_tools(learned) == {"x"}

    def test_a_restored_tool_is_not_withheld(self, learned):
        actuator.write_archive({"tools": {"x": {"withheld": True}}}, learned)
        actuator.restore("x", learned)
        assert actuator.withheld_tools(learned) == set()

    def test_restoring_a_tool_that_was_never_withheld_fails(self, learned):
        assert actuator.restore("nope", learned) is False


# ── the decisions ─────────────────────────────────────────────────────────

class TestDecisions:

    def test_nothing_is_due_for_an_empty_store(self, learned):
        assert actuator.evaluate(learned) == []

    def test_a_tool_that_always_fails_is_due_for_an_update(self, learned):
        for index in range(12):
            _record(learned, "s%d" % index, "t", ["flaky"], ok=False)
        actions = {d["tool"]: d["action"] for d in actuator.evaluate(learned)}
        assert actions.get("flaky") in ("update", "withhold")

    def test_a_tool_that_always_works_is_not_due(self, learned):
        for index in range(12):
            _record(learned, "s%d" % index, "t", ["solid"])
        actions = {d["tool"]: d["action"] for d in actuator.evaluate(learned)}
        assert "solid" not in actions

    def test_a_tool_called_twice_is_not_judged(self, learned):
        # Below the call floor the failure rate is noise, and a tool
        # called twice and failed once must not trigger a rewrite loop.
        _record(learned, "s1", "t", ["rare"], ok=False)
        _record(learned, "s2", "t", ["rare"], ok=False)
        actions = {d["tool"]: d["action"] for d in actuator.evaluate(learned)}
        assert "rare" not in actions

    def test_the_recovery_tools_are_never_due(self, learned):
        # Withholding the ability to write a note would make the agent
        # unable to record why it was wrong.
        for index in range(30):
            _record(learned, "s%d" % index, "t", ["write_file", "bash"],
                    ok=False)
        tools = {d["tool"] for d in actuator.evaluate(learned)}
        assert "write_file" not in tools
        assert "bash" not in tools

    def test_a_decision_says_why(self, learned):
        for index in range(12):
            _record(learned, "s%d" % index, "t", ["flaky"], ok=False)
        decisions = actuator.evaluate(learned)
        assert decisions
        assert all(d.get("reasons") for d in decisions)


# ── applying ──────────────────────────────────────────────────────────────

class TestApply:

    def _due(self, store):
        for index in range(12):
            _record(store, "s%d" % index, "t", ["flaky_tool"], ok=False)

    def test_a_due_tool_is_withheld_when_the_actuator_is_on(self, adaptive):
        self._due(adaptive)
        result = actuator.apply(adaptive)
        assert "flaky_tool" in result["withheld"]
        assert actuator.withheld_tools(adaptive) == {"flaky_tool"}

    def test_the_withholding_is_reversible(self, adaptive):
        self._due(adaptive)
        actuator.apply(adaptive)
        assert actuator.restore("flaky_tool", adaptive) is True
        assert actuator.withheld_tools(adaptive) == set()

    def test_a_tool_is_not_withheld_forever(self, adaptive):
        # A tool that keeps coming back was needed.  A policy that is
        # not learning is a policy that is wrong, and the fix is to stop
        # acting on it.
        self._due(adaptive)
        for _ in range(actuator.MAX_WITHHOLDINGS + 3):
            actuator.apply(adaptive)
        archive = actuator.read_archive(adaptive)
        entry = (archive.get("tools") or {}).get("flaky_tool") or {}
        assert entry.get("count", 0) <= actuator.MAX_WITHHOLDINGS

    def test_the_archive_records_why(self, adaptive):
        self._due(adaptive)
        actuator.apply(adaptive)
        entry = (actuator.read_archive(adaptive).get("tools")
                 or {}).get("flaky_tool") or {}
        assert entry.get("reasons")

    def test_applying_nothing_writes_nothing(self, adaptive):
        result = actuator.apply(adaptive)
        assert result["applied"] is False
        assert not os.path.isfile(actuator.archive_path(adaptive))


# ── the filter ────────────────────────────────────────────────────────────

class TestFilter:

    def test_nothing_is_filtered_when_nothing_is_withheld(self, learned):
        tools = ["bash", "read_file"]
        assert actuator.filter_tools(tools, learned) == tools

    def test_a_withheld_tool_is_filtered_out(self, learned):
        actuator.write_archive({"tools": {"flaky": {"withheld": True}}},
                                learned)
        assert actuator.filter_tools(["bash", "flaky"], learned) == ["bash"]

    def test_the_filter_does_not_mutate_its_argument(self, learned):
        actuator.write_archive({"tools": {"flaky": {"withheld": True}}},
                                learned)
        tools = ["bash", "flaky"]
        actuator.filter_tools(tools, learned)
        assert tools == ["bash", "flaky"]


# ── the registry ──────────────────────────────────────────────────────────

class TestRegistryWithholding:

    def _registry(self, names):
        from src.type.tools import ToolRegistry, ToolHandler
        registry = ToolRegistry()
        for name in names:
            handler = ToolHandler(
                url="stdio://test",
                tool_item={"type": "function", "function": {
                    "name": name, "description": "d",
                    "parameters": {"type": "object", "properties": {}}}})
            registry.register(handler)
        return registry

    def test_a_withheld_tool_leaves_the_registry(self):
        registry = self._registry(["bash", "flaky"])
        dropped = registry.withhold(["flaky"])
        assert dropped == ["flaky"]
        assert "flaky" not in registry.tools
        assert "flaky" not in registry.order

    def test_the_surviving_tools_keep_their_order(self):
        # The order is what the model sees; reordering it unsettles the
        # model and breaks the server's prefix cache.
        registry = self._registry(["a", "b", "c"])
        registry.withhold(["b"])
        assert registry.order == ["a", "c"]

    def test_withholding_a_tool_that_is_not_there_does_nothing(self):
        registry = self._registry(["a"])
        assert registry.withhold(["nope"]) == []
        assert registry.order == ["a"]

    def test_the_handler_survives_the_withholding(self):
        # Nothing is deleted: the tool stays reachable and restorable.
        registry = self._registry(["flaky"])
        registry.withhold(["flaky"])
        assert "flaky@stdio://test" in registry.handlers


# ── the CLI surface ───────────────────────────────────────────────────────

class TestLearnCLI:

    def test_the_flags_exist(self):
        from src.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["learn", "--lifecycle"])
        assert args.lifecycle is True
        args = parser.parse_args(["learn", "--lifecycle", "--apply"])
        assert args.apply is True
        args = parser.parse_args(["learn", "--lifecycle", "--restore", "bash"])
        assert args.restore == "bash"

    def test_the_report_says_when_nothing_is_due(self, capsys, tmp_path):
        import src.cli as cli
        import argparse
        config = {"learn": {"autonomy": "observe",
                             "path": str(tmp_path / "learned")}}
        args = argparse.Namespace(lifecycle=True, apply=False, restore=None)
        cli._learn_lifecycle(args, config)
        out = capsys.readouterr().out
        assert "Nothing is due" in out

    def test_the_report_says_the_actuator_is_off(self, capsys, tmp_path):
        import src.cli as cli
        import argparse
        config = {"learn": {"autonomy": "observe",
                             "path": str(tmp_path / "learned")}}
        for index in range(12):
            _record(config, "s%d" % index, "t", ["flaky_tool"], ok=False)
        args = argparse.Namespace(lifecycle=True, apply=False, restore=None)
        cli._learn_lifecycle(args, config)
        out = capsys.readouterr().out
        assert "off" in out

    def test_applying_with_the_actuator_off_says_so(self, capsys, tmp_path):
        import src.cli as cli
        import argparse
        config = {"learn": {"autonomy": "observe",
                             "path": str(tmp_path / "learned")}}
        for index in range(12):
            _record(config, "s%d" % index, "t", ["flaky_tool"], ok=False)
        args = argparse.Namespace(lifecycle=True, apply=True, restore=None)
        cli._learn_lifecycle(args, config)
        out = capsys.readouterr().out
        assert "off" in out

    def test_restoring_a_tool_that_is_not_withheld_fails(self, capsys,
                                                         tmp_path):
        import src.cli as cli
        import argparse
        config = {"learn": {"autonomy": "observe",
                             "path": str(tmp_path / "learned")}}
        args = argparse.Namespace(lifecycle=True, apply=False,
                                  restore="nope")
        with pytest.raises(SystemExit):
            cli._learn_lifecycle(args, config)
