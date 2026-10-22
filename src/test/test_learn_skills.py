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

Loop C: the skill library, the miner, the gate, and the curator.

These tests are mostly about the boundaries, because the boundaries are
where a self-improvement loop does its damage:

  * a skill that is not pinned is never served;
  * a skill that fails the gate is never pinned;
  * the gate's source screen refuses the things that would let a skill
    escape the sandbox;
  * the pin is reversible;
  * the miner never writes an argument value down, and never confuses
    two different fragments of the same tool.
"""

import os

import pytest

from src.learn import skills, skills_curator, skills_gate, skills_mining
from src.learn import trajectory as traj


@pytest.fixture
def store(tmp_path):
    """A learned store with recording on."""
    return {"learn": {"autonomy": "observe",
                      "path": str(tmp_path / "learned")}}


@pytest.fixture
def candidate():
    """A mined candidate, as the miner would produce one."""
    return {"tool": "bash", "digest": "abc123def456", "sessions": 5,
            "calls": 7, "errors": 0, "success_rate": 1.0,
            "arg_keys": ["command", "data_path"]}


# ── the store ───────────────────────────────────────────────────────────────

class TestStore:

    def test_the_skills_live_beside_the_trajectories(self, store):
        """Not under them.  `trajectories/` means one thing."""
        learn_root = os.path.dirname(skills.skills_dir(store))
        assert skills.skills_dir(store) == os.path.join(learn_root, "skills")
        assert skills.skills_dir(store) != os.path.join(
            learn_root, "trajectories", "skills")
        assert os.path.join(learn_root, "trajectories") == \
            traj.trajectory_dir(store)

    def test_nothing_is_pinned_to_begin_with(self, store):
        assert skills.pinned_skills(store) == {}

    def test_a_pin_naming_a_missing_skill_is_stale(self, store):
        """A pin that names a skill whose files are gone is filtered, so
        every reader agrees on what is live."""
        skills.write_pin({"version": 1, "skills": {"gone_skill": {}}},
                         store)
        assert skills.pinned_skills(store) == {}

    def test_a_pin_naming_an_unsafe_skill_is_ignored(self, store):
        skills.write_pin({"version": 1, "skills": {"../escape": {}}}, store)
        assert skills.pinned_skills(store) == {}

    def test_a_malformed_pin_reads_as_empty(self, store):
        path = skills.pinned_path(store)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("skills: [this is not a mapping\n")
        assert skills.read_pin(store)["skills"] == {}


# ── the manifest ────────────────────────────────────────────────────────────

class TestManifest:

    def test_a_good_manifest_has_no_problems(self):
        manifest = {"name": "my_skill", "description": "Does the thing.",
                    "version": 1}
        assert skills.manifest_errors(manifest) == []

    def test_a_missing_manifest_is_a_problem(self):
        assert skills.manifest_errors(None)

    def test_a_manifest_with_no_description_is_a_problem(self):
        """The model cannot choose a tool it cannot tell apart."""
        problems = skills.manifest_errors({"name": "my_skill",
                                          "description": "  ",
                                          "version": 1})
        assert any("description" in p for p in problems)

    def test_a_manifest_with_no_name_is_a_problem(self):
        problems = skills.manifest_errors({"description": "d", "version": 1})
        assert any("name" in p for p in problems)

    def test_a_name_that_is_not_a_tool_name_is_a_problem(self):
        problems = skills.manifest_errors({"name": "My Skill!",
                                          "description": "d", "version": 1})
        assert any("name" in p for p in problems)

    def test_a_manifest_with_no_version_is_a_problem(self):
        problems = skills.manifest_errors({"name": "my_skill",
                                          "description": "d"})
        assert any("version" in p for p in problems)

    def test_a_very_long_description_is_a_problem(self):
        """Every tool description is paid for on every request."""
        problems = skills.manifest_errors({"name": "my_skill",
                                          "description": "x" * 500,
                                          "version": 1})
        assert any("description" in p for p in problems)

    def test_a_manifest_that_is_not_a_mapping_is_a_problem(self):
        assert skills.manifest_errors("not a manifest")


# ── the deltas ──────────────────────────────────────────────────────────────

class TestDeltas:

    def test_an_insert_adds_the_skill(self):
        pin = skills.apply_deltas({"version": 0, "skills": {}},
                                  [{"op": "insert", "name": "my_skill"}])
        assert "my_skill" in pin["skills"]
        assert pin["version"] == 1

    def test_an_insert_is_a_probation_until_it_is_promoted(self):
        pin = skills.apply_deltas({"version": 0, "skills": {}},
                                  [{"op": "insert", "name": "my_skill"}])
        assert pin["skills"]["my_skill"]["status"] == skills.PROBATION

    def test_an_insert_of_an_existing_name_updates_it(self):
        """Two entries for one name is a library nobody can reason about."""
        pin = skills.apply_deltas({"version": 0, "skills": {}},
                                  [{"op": "insert", "name": "my_skill"}])
        pin = skills.apply_deltas(pin, [{"op": "insert", "name": "my_skill",
                                         "skill": {"description": "new"}}])
        assert len(pin["skills"]) == 1
        assert pin["skills"]["my_skill"]["description"] == "new"

    def test_an_update_changes_only_what_it_names(self):
        pin = skills.apply_deltas(
            {"version": 1, "skills": {"a": {"x": 1}, "b": {"x": 2}}},
            [{"op": "update", "name": "a", "skill": {"x": 9}}])
        assert pin["skills"]["a"]["x"] == 9
        assert pin["skills"]["b"]["x"] == 2

    def test_a_deprecate_keeps_the_skill_and_marks_it(self):
        """A dropped skill is a decision, not an oversight."""
        pin = skills.apply_deltas(
            {"version": 1, "skills": {"a": {"x": 1}}},
            [{"op": "deprecate", "name": "a", "reason": "it stopped working"}])
        assert pin["skills"]["a"]["status"] == skills.DEPRECATED
        assert pin["skills"]["a"]["reason"] == "it stopped working"

    def test_a_delta_aimed_at_nothing_is_recorded_not_ignored(self):
        """Silently ignoring a delta is how a library drifts from itself."""
        pin = skills.apply_deltas(
            {"version": 1, "skills": {}},
            [{"op": "update", "name": "ghost", "skill": {}}])
        assert pin["skipped"]
        assert pin["skipped"][0]["name"] == "ghost"

    def test_an_unknown_operation_is_recorded(self):
        pin = skills.apply_deltas(
            {"version": 1, "skills": {}},
            [{"op": "explode", "name": "x"}])
        assert pin["skipped"]


# ── the miner ───────────────────────────────────────────────────────────────

def _record(store, session, tools, ok=True):
    """A recorded run that used ``tools``."""
    traj.record_task(session_id=session, turn=1, task="t", response="r",
                     config_data=store,
                     metrics={"turns": [{"n": 1, "tool_runs": [
                         {"name": name, "ok": ok, "args_digest": "d" + name,
                          "arg_keys": ["command"]} for name in tools]}]})


class TestMining:

    def test_a_fragment_in_three_sessions_is_a_candidate(self, store):
        for i in range(3):
            _record(store, "s%d" % i, ["bash"])
        candidates = skills_mining.mine_store(store)
        assert any(c["tool"] == "bash" for c in candidates)

    def test_a_fragment_in_two_sessions_is_not(self, store):
        """Two is a coincidence; three is a pattern."""
        for i in range(2):
            _record(store, "s%d" % i, ["bash"])
        candidates = skills_mining.mine_store(store)
        assert not any(c["tool"] == "bash" for c in candidates)

    def test_a_fragment_that_keeps_failing_is_not_a_candidate(self, store):
        """A recurring failure is a defect, and promoting it promotes the
        bug."""
        for i in range(4):
            _record(store, "s%d" % i, ["bash"], ok=False)
        candidates = skills_mining.mine_store(store)
        assert not any(c["tool"] == "bash" for c in candidates)

    def test_the_bookkeeping_tools_are_not_mined(self, store):
        """A skill is a capability.  The agent talking to itself is not
        one, and promoting it would add schema tokens to save schema
        tokens."""
        for i in range(4):
            _record(store, "s%d" % i, ["context_status"])
        candidates = skills_mining.mine_store(store)
        assert not any(c["tool"] == "context_status" for c in candidates)

    def test_a_call_without_a_digest_is_not_mined(self, store):
        """Without a digest there is no way to know it is the same call."""
        traj.record_task(session_id="s", turn=1, task="t", response="r",
                         config_data=store,
                         metrics={"turns": [{"n": 1, "tool_runs": [
                             {"name": "bash", "ok": True,
                              "arg_keys": ["command"]}]}]})
        assert skills_mining.mine_store(store) == []

    def test_two_candidates_never_share_a_name(self, store):
        """They would fight over one directory.

        Two fragments, each in three sessions, distinguished only by
        their digest -- the case where the name has to carry the
        discriminator.
        """
        for i in range(3):
            for digest in ("digest_a", "digest_b"):
                traj.record_task(session_id="s%d" % i, turn=1, task="t",
                                 response="r", config_data=store,
                                 metrics={"turns": [{"n": 1, "tool_runs": [
                                     {"name": "bash", "ok": True,
                                      "args_digest": digest,
                                      "arg_keys": ["command"]}]}]})
        candidates = skills_mining.mine_store(store)
        names = [c["name"] for c in candidates]
        assert len(candidates) == 2
        assert len(names) == len(set(names))

    def test_the_miner_never_writes_an_argument_value_down(self, store):
        """The digest is the identity; the values stay where they were."""
        for i in range(4):
            traj.record_task(session_id="s%d" % i, turn=1, task="t",
                             response="r", config_data=store,
                             metrics={"turns": [{"n": 1, "tool_runs": [
                                 {"name": "bash", "ok": True,
                                  "args_digest": "d%d" % i,
                                  "arg_keys": ["command"]}]}]})
        candidates = skills_mining.mine_store(store)
        for c in candidates:
            assert "command" not in str(c.get("args"))
            assert set(c.keys()) == {"tool", "digest", "sessions", "calls",
                                     "errors", "success_rate", "arg_keys",
                                     "name"}

    def test_the_suggested_name_is_a_valid_tool_name(self, candidate):
        name = skills_mining.suggest_name(candidate)
        assert name == name.lower()
        assert name.replace("_", "").isalnum()
        assert name[0].isalpha()


# ── the gate ────────────────────────────────────────────────────────────────

GOOD_SOURCE = '''
"""A well-behaved skill."""

import json


def run(**kwargs):
    return json.dumps(kwargs)
'''

GOOD_TEST = '''
import unittest


def test_something():
    assert True
'''


class TestGateManifest:

    def test_a_good_manifest_passes(self):
        ok, reason = skills_gate.check_manifest(
            {"name": "my_skill", "description": "Does the thing.",
             "version": 1})
        assert ok, reason

    def test_a_bad_manifest_fails_with_a_reason(self):
        ok, reason = skills_gate.check_manifest({"name": "my_skill"})
        assert not ok
        assert reason


class TestGateSource:
    """The screen that keeps a skill inside the sandbox."""

    def test_a_clean_source_passes(self):
        ok, reason, _ = skills_gate.check_source(GOOD_SOURCE)
        assert ok, reason

    def test_a_source_that_does_not_parse_fails(self):
        ok, reason, _ = skills_gate.check_source("def broken(:")
        assert not ok
        assert "parse" in reason

    def test_a_third_party_import_fails(self):
        ok, reason, _ = skills_gate.check_source("import requests\n")
        assert not ok
        assert "standard library" in reason

    def test_a_module_that_reaches_outside_the_process_fails(self):
        ok, reason, _ = skills_gate.check_source("import ctypes\n")
        assert not ok

    def test_subprocess_fails(self):
        ok, reason, _ = skills_gate.check_source(
            "import subprocess\nsubprocess.run(['ls'])\n")
        assert not ok

    def test_os_system_fails(self):
        ok, reason, _ = skills_gate.check_source("import os\nos.system('ls')\n")
        assert not ok

    def test_eval_fails(self):
        ok, reason, _ = skills_gate.check_source("x = eval('1+1')\n")
        assert not ok

    def test_exec_fails(self):
        ok, reason, _ = skills_gate.check_source("exec('x=1')\n")
        assert not ok

    def test_dunder_import_fails(self):
        ok, reason, _ = skills_gate.check_source("m = __import__('os')\n")
        assert not ok

    def test_pickle_fails(self):
        ok, reason, _ = skills_gate.check_source(
            "import pickle\nx = pickle.loads(b'')\n")
        assert not ok

    def test_a_source_that_names_the_holdout_fails(self):
        """I7: the ruler is outside the agent's writable surface, and a
        skill that reads it would be a self-certifying system."""
        ok, reason, _ = skills_gate.check_source(
            "path = '/benchmarks/eval/holdout.jsonl'\n")
        assert not ok

    def test_a_source_that_names_the_baselines_fails(self):
        ok, reason, _ = skills_gate.check_source(
            "path = 'benchmarks/baselines/full.json'\n")
        assert not ok

    def test_a_source_that_names_the_pin_fails(self):
        ok, reason, _ = skills_gate.check_source(
            "path = 'pinned.yaml'\n")
        assert not ok

    def test_a_source_naming_a_forbidden_path_fails(self):
        ok, reason, _ = skills_gate.check_source(
            "path = '/somewhere/secret'\n",
            forbidden_paths=["/somewhere/secret"])
        assert not ok


class TestGateDependencies:

    def test_no_dependencies_passes(self):
        ok, reason, _ = skills_gate.check_dependencies(
            {"name": "s", "description": "d", "version": 1})
        assert ok, reason

    def test_a_dependency_passes_when_installs_are_open(self):
        ok, reason, _ = skills_gate.check_dependencies(
            {"name": "s", "description": "d", "version": 1,
             "dependencies": ["requests"]}, sealed=False)
        assert ok, reason

    def test_a_dependency_fails_when_installs_are_sealed(self):
        ok, reason, _ = skills_gate.check_dependencies(
            {"name": "s", "description": "d", "version": 1,
             "dependencies": ["requests"]}, sealed=True)
        assert not ok
        assert "sealed" in reason


class TestGateTests:

    def test_a_good_test_file_passes(self):
        ok, reason, _ = skills_gate.check_tests(GOOD_TEST)
        assert ok, reason

    def test_no_test_file_fails(self):
        ok, reason, _ = skills_gate.check_tests(None)
        assert not ok
        assert "no test file" in reason

    def test_a_test_file_that_does_not_parse_fails(self):
        ok, reason, _ = skills_gate.check_tests("def broken(:")
        assert not ok

    def test_a_test_file_with_no_tests_fails(self):
        ok, reason, _ = skills_gate.check_tests("x = 1\n")
        assert not ok


class TestGateRegression:

    def test_a_positive_delta_passes(self):
        ok, reason, _ = skills_gate.check_regression(0.2)
        assert ok, reason

    def test_a_zero_delta_passes(self):
        """A skill that adds nothing still costs schema tokens, but it
        does not make things worse."""
        ok, reason, _ = skills_gate.check_regression(0.0)
        assert ok, reason

    def test_a_negative_delta_fails(self):
        ok, reason, _ = skills_gate.check_regression(-0.1)
        assert not ok
        assert "worse" in reason

    def test_an_unmeasured_skill_fails_when_a_measurement_is_required(self):
        ok, reason, _ = skills_gate.check_regression(None, required=True)
        assert not ok
        assert "unmeasured" in reason

    def test_an_unmeasured_skill_may_pass_when_not_required(self):
        """But the gate says so rather than pretending it passed."""
        ok, reason, detail = skills_gate.check_regression(None,
                                                         required=False)
        assert ok
        assert detail["delta"] is None


class TestTheWholeGate:

    def _gate(self, **overrides):
        kwargs = {
            "manifest": {"name": "my_skill", "description": "Does it.",
                         "version": 1},
            "source": GOOD_SOURCE,
            "test_source": GOOD_TEST,
            "delta": 0.1,
        }
        kwargs.update(overrides)
        return skills_gate.run_gate(**kwargs)

    def test_a_good_skill_passes(self):
        result = self._gate()
        assert result.passed, result.reason()
        assert result.failures == []

    def test_the_gate_runs_every_check_even_after_a_failure(self):
        """The operator wants to know everything that is wrong."""
        result = self._gate(source="import requests\n")
        assert not result.passed
        assert set(result.checks) == set(skills_gate.ALL_CHECKS)

    def test_the_gate_reports_which_check_failed(self):
        result = self._gate(test_source=None)
        assert skills_gate.CHECK_TESTS in result.failures
        assert "tests" in result.reason()

    def test_the_result_serialises(self):
        result = self._gate()
        data = result.to_dict()
        assert data["passed"] is True
        assert isinstance(data["checks"], dict)


# ── the curator ─────────────────────────────────────────────────────────────

class TestCurator:

    def test_synthesising_writes_a_skill(self, store, candidate):
        skill = skills_curator.synthesize(candidate, name="my_skill",
                                          config_data=store)
        assert os.path.isfile(os.path.join(skill["dir"], "tool.py"))
        assert os.path.isfile(os.path.join(skill["dir"], "manifest.yaml"))
        assert os.path.isfile(os.path.join(skill["dir"],
                                           "test_my_skill.py"))

    def test_a_synthesised_skill_is_not_pinned(self, store, candidate):
        """Writing a skill is not serving it."""
        skills_curator.synthesize(candidate, name="my_skill",
                                  config_data=store)
        assert skills.pinned_skills(store) == {}

    def test_the_draft_body_raises_rather_than_pretending(self, store,
                                                         candidate):
        """A skill that silently does nothing gets pinned and trusted."""
        skills_curator.synthesize(candidate, name="my_skill",
                                  config_data=store)
        path = os.path.join(skills.skill_dir("my_skill", store), "tool.py")
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "NotImplementedError" in source

    def test_the_draft_manifest_describes_what_was_mined(self, store,
                                                         candidate):
        skill = skills_curator.synthesize(candidate, name="my_skill",
                                          config_data=store)
        description = skill["manifest"]["description"]
        assert "bash" in description

    def test_a_failing_gate_means_no_pin(self, store, candidate):
        """The pin is written last, and only on a green gate."""
        skills_curator.synthesize(candidate, name="my_skill",
                                  config_data=store)
        result = skills_curator.promote("my_skill", config_data=store,
                                        delta=None)
        assert result["pinned"] is False
        assert skills.pinned_skills(store) == {}

    def test_a_passing_gate_pins_the_skill(self, store, candidate):
        skills_curator.synthesize(candidate, name="my_skill",
                                  config_data=store)
        result = skills_curator.promote("my_skill", config_data=store,
                                        delta=0.1)
        assert result["pinned"] is True
        assert "my_skill" in skills.pinned_skills(store)

    def test_a_rollback_restores_the_previous_pin(self, store, candidate):
        """A promotion is undone by restoring a file."""
        skills_curator.synthesize(candidate, name="my_skill",
                                  config_data=store)
        skills_curator.promote("my_skill", config_data=store, delta=0.1)
        assert "my_skill" in skills.pinned_skills(store)
        result = skills_curator.rollback(store)
        assert result["rolled_back"] is True
        assert "my_skill" not in skills.pinned_skills(store)

    def test_a_rollback_with_nothing_to_roll_back_says_so(self, store):
        result = skills_curator.rollback(store)
        assert result["rolled_back"] is False
        assert "no previous pin" in result["reason"]

    def test_the_manifest_is_written_as_yaml(self, store, candidate):
        skill = skills_curator.synthesize(candidate, name="my_skill",
                                          config_data=store)
        with open(os.path.join(skill["dir"], "manifest.yaml"), "r",
                  encoding="utf-8") as f:
            text = f.read()
        assert "description:" in text


# ── the server ──────────────────────────────────────────────────────────────

class TestSkillsServer:

    def test_only_pinned_skills_are_loaded(self, store, candidate):
        from src.mcp.servers.tasks.skills import mcp_server

        skills_curator.synthesize(candidate, name="my_skill",
                                  config_data=store)
        config = {"learn": {"path": store["learn"]["path"]}}
        loaded = mcp_server.load_skills(config)
        assert "my_skill" not in loaded

        skills_curator.promote("my_skill", config_data=store, delta=0.1)
        loaded = mcp_server.load_skills(config)
        assert "my_skill" in loaded

    def test_a_skill_with_no_description_is_not_served(self, store,
                                                        candidate):
        """An undescribed tool is schema tokens spent for nothing."""
        from src.mcp.servers.tasks.skills import mcp_server

        skills_curator.synthesize(candidate, name="my_skill",
                                  config_data=store)
        manifest = skills.read_manifest(
            os.path.join(skills.skill_dir("my_skill", store),
                         "manifest.yaml"))
        manifest["description"] = ""
        skills.write_manifest(
            os.path.join(skills.skill_dir("my_skill", store),
                         "manifest.yaml"), manifest)
        skills_curator.promote("my_skill", config_data=store, delta=0.1)
        config = {"learn": {"path": store["learn"]["path"]}}
        loaded = mcp_server.load_skills(config)
        assert "my_skill" not in loaded

    def test_a_broken_skill_does_not_take_the_server_down(self, store,
                                                         candidate):
        """One broken skill should cost the agent that skill."""
        from src.mcp.servers.tasks.skills import mcp_server

        skills_curator.synthesize(candidate, name="my_skill",
                                  config_data=store)
        skills_curator.promote("my_skill", config_data=store, delta=0.1)
        with open(os.path.join(skills.skill_dir("my_skill", store),
                               "tool.py"), "w", encoding="utf-8") as f:
            f.write("this is not python(\n")
        config = {"learn": {"path": store["learn"]["path"]}}
        loaded = mcp_server.load_skills(config)
        assert loaded == {}

    def test_the_schema_is_built_from_the_manifest(self):
        from src.mcp.servers.tasks.skills import mcp_server

        schema = mcp_server._schema_for({
            "input_schema": {"type": "object",
                             "properties": {"a": {"type": "string"}}}})
        assert schema["properties"]["a"]["type"] == "string"

    def test_a_skill_with_no_schema_gets_an_empty_one(self):
        from src.mcp.servers.tasks.skills import mcp_server

        schema = mcp_server._schema_for({})
        assert schema["properties"] == {}

    def test_a_skill_with_no_declared_parameters_gets_none(self):
        """Not a free-form anything: a tool with no parameters is a tool
        with no parameters."""
        from src.mcp.servers.tasks.skills import mcp_server

        handler = mcp_server._make_handler(lambda **kw: "ok", {})
        import inspect
        assert list(inspect.signature(handler).parameters) == []
