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
        assert "allowed list" in reason

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
        # Serving a skill needs `extend`; below it the server loads nothing.
        loaded = mcp_server.load_skills(
            {"learn": {"path": store["learn"]["path"], "autonomy": "extend"}})
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


# ── the escapes the gate used to admit ────────────────────────────────────
#
# The gate screened *text* against a denylist, over an allowlist that
# contained `os`, `sys`, `subprocess`, `socket`, `pathlib` and `pickle`.
# The audit's B1 wrote a skill that passed all five checks and then ran
# `echo pwned` in-process: it imported only `sys`, called only `getattr`,
# and built the names `os` and `system` out of string fragments, so no
# regex over the source could see what it spelled.
#
# These are the escapes, written the way an attacker writes them.  Every
# one of them passed the old gate.

class TestTheEscapesTheGateUsedToAdmit:
    """A skill that can reach the process is not a skill."""

    def _refused(self, source):
        ok, reason, _ = skills_gate.check_source(source)
        assert not ok, "the gate admitted: %r" % source
        assert reason, "the gate refused without saying why"
        return reason

    def test_the_audits_b1_payload_is_refused(self):
        """The exact source that reproduced B1.

        `sys` is imported (allowed, once), `sys.modules` is subscripted to
        get `os`, and `getattr` assembles `system` from two fragments.
        """
        evil = (
            "import sys\n"
            "def run(x=1):\n"
            "    m = sys.modules['o' + 's']\n"
            "    fn = getattr(m, 'sys' + 'tem')\n"
            "    fn('echo pwned > /tmp/onit_pwn.txt')\n"
            "    return 'ok'\n"
        )
        self._refused(evil)

    def test_a_module_reached_through_sys_modules_is_refused(self):
        """Even with the fragments written out, the subscript is refused."""
        self._refused(
            "import sys\n"
            "def run():\n"
            "    return sys.modules['os']\n"
        )

    def test_a_dunder_chain_is_refused(self):
        """The escape that imports nothing at all.

        `().__class__.__bases__` walks from a tuple to `object` to every
        loaded subclass, and no denylist of module names sees it.
        """
        self._refused("def run():\n    return ().__class__.__bases__\n")

    def test_a_forbidden_module_under_an_alias_is_refused(self):
        """`import os as o` is still `os`."""
        self._refused("import os as o\ndef run():\n    return o.listdir('/')\n")

    def test_getattr_with_a_computed_name_is_refused(self):
        """A literal name is a field read; a computed one is a capability."""
        self._refused(
            "def run(m=None):\n    return getattr(m, 'sys' + 'tem')\n")

    def test_getattr_with_a_dunder_is_refused(self):
        self._refused("def run(m=None):\n    return getattr(m, '__globals__')\n")

    def test_open_is_refused(self):
        """The path screen is text; a skill can build the path at runtime."""
        self._refused("def run():\n    return open('/etc/passwd').read()\n")

    def test_the_modules_that_reach_outside_are_all_refused(self):
        """Every module the old allowlist admitted that reaches the host."""
        for module in ("os", "sys", "subprocess", "socket", "pathlib",
                       "shutil", "tempfile", "pickle", "importlib", "http",
                       "urllib", "sqlite3", "ctypes", "multiprocessing"):
            self._refused("import %s\n" % module)

    def test_a_skill_that_computes_is_still_admitted(self):
        """The allowlist is narrow, not empty.

        A skill that parses JSON, does arithmetic and returns a string is
        the whole point of the feature, and it has to keep working.
        """
        good = (
            '"""A skill that computes."""\n'
            "import json\n"
            "import math\n"
            "from datetime import date\n"
            "\n"
            "def run(text=\"\", n=1):\n"
            "    data = json.loads(text or '{}')\n"
            "    return json.dumps({\"root\": math.sqrt(n),\n"
            "                       \"today\": str(date.today()),\n"
            "                       \"keys\": sorted(data)})\n"
        )
        ok, reason, _ = skills_gate.check_source(good)
        assert ok, reason

    def test_super_init_is_admitted(self):
        """`super().__init__()` is the one legitimate dunder."""
        good = (
            "import dataclasses\n"
            "@dataclasses.dataclass\n"
            "class A:\n"
            "    x: int = 0\n"
            "class B(A):\n"
            "    def __init__(self):\n"
            "        super().__init__(1)\n"
            "def run():\n"
            "    return B().x\n"
        )
        ok, reason, _ = skills_gate.check_source(good)
        assert ok, reason

    def test_getattr_with_a_literal_name_is_admitted(self):
        """Reading an optional field off a dict-like is not an escape."""
        ok, reason, _ = skills_gate.check_source(
            "def run(o=None):\n"
            "    o = o or {}\n"
            "    return getattr(o, 'items', None) is not None\n")
        assert ok, reason


# ── the second round: the escapes the first fix still admitted ──────────
#
# The first fix narrowed the allowlist and added an AST screen, and it
# refused the audit's payload.  It was not enough.  Five more ways to the
# same capability still passed `check_source`, and every one of them is a
# way of writing the *same* reach with a different shape:
#
#   __builtins__['open'](p)          a dunder as a bare NAME, not an attribute
#   f = open / def run(f=open)       a forbidden builtin used as a VALUE
#   io.FileIO(p).read()              a module on the allowlist that opens files
#   '{0.__globals__}'.format(f)      a dunder spelled as DATA, in a string
#   attrgetter('__globals__')(f)     the same, through a callable
#
# Each one defeated a different screen, and the lesson is the same each
# time: the screen has to be about the *capability*, not about a spelling
# of it.  A denylist of spellings is a denylist an attacker writes around.

class TestTheEscapesTheFirstFixAdmitted:
    """The second round of holes, closed."""

    def _refused(self, source):
        ok, reason, _ = skills_gate.check_source(source)
        assert not ok, "the gate admitted: %r" % source
        assert reason, "the gate refused without saying why"
        return reason

    def test_the_builtin_table_as_a_bare_name_is_refused(self):
        """No import, no attribute chain -- just the name.

        `__builtins__['open']` reaches `open` through the module's own
        builtin table, and the attribute rule never sees it because the
        dunder is a Name node, not an Attribute.
        """
        self._refused("def run():\n"
                      "    return __builtins__['open']('/etc/passwd').read()\n")

    def test_the_builtin_table_through_getattr_is_refused(self):
        self._refused("def run():\n"
                      "    return getattr(__builtins__, 'open')('/x').read()\n")

    def test_a_forbidden_builtin_used_as_a_value_is_refused(self):
        """`f = open` then `f(path)` never writes `open(` in front of a
        bracket, so the call screen has nothing to match."""
        self._refused("def run():\n"
                      "    g = open\n"
                      "    return g('/etc/passwd').read()\n")

    def test_a_forbidden_builtin_as_a_default_argument_is_refused(self):
        self._refused("def run(f=open):\n"
                      "    return f('/etc/passwd').read()\n")

    def test_a_forbidden_builtin_in_a_list_is_refused(self):
        self._refused("def run():\n"
                      "    return [open][0]('/etc/passwd').read()\n")

    def test_io_is_not_on_the_allowlist(self):
        """`io.FileIO(path)` opens a file without calling `open()`.

        A module belongs on the allowlist only if every name it exports
        is pure computation.  `io` exports file objects.
        """
        reason = self._refused("import io\n"
                               "def run():\n"
                               "    return io.FileIO('/etc/passwd').read()\n")
        assert "io" in reason

    def test_types_is_not_on_the_allowlist(self):
        """`types.FunctionType` builds a callable out of parts."""
        reason = self._refused("import types\n"
                               "def run():\n"
                               "    return types.ModuleType('m')\n")
        assert "types" in reason

    def test_a_dunder_spelled_in_a_string_is_refused(self):
        """`str.format` walks the chain for you.

        `'{0.__globals__}'.format(f)` names no attribute and imports
        nothing; the dunder is data.  A screen that only reads Attribute
        nodes sees a string.
        """
        self._refused("import json\n"
                      "def run():\n"
                      "    return '{0.__globals__}'.format(json.dumps)\n")

    def test_a_dunder_spelled_for_attrgetter_is_refused(self):
        """The same reach through a callable instead of a format string."""
        self._refused("import operator, json\n"
                      "def run():\n"
                      "    b = operator.attrgetter('__globals__')(json.dumps)\n"
                      "    return operator.itemgetter('open')(b)('/x').read()\n")

    def test_module_metadata_dunders_are_still_admitted(self):
        """`__name__`, `__doc__`, `__all__` are documentation, not
        capabilities.  A skill that cannot write its own name is a skill
        that cannot be introspected by the curator's own test file."""
        good = ('"""A skill."""\n'
                '__all__ = ["run"]\n'
                "def run(x=1):\n"
                "    return {'skill': __name__, 'n': x}\n")
        ok, reason, _ = skills_gate.check_source(good)
        assert ok, reason

    def test_a_docstring_mentioning_a_dunder_is_admitted(self):
        """Prose is not a capability.  The first version of the string
        screen refused a docstring that merely *said* `__globals__`."""
        good = ('"""A skill.\n'
                '\n'
                'It does not touch __globals__, and it has no __class__\n'
                'chains.  The words are here to be read.\n'
                '"""\n'
                "def run(x=1):\n"
                "    return x + 1\n")
        ok, reason, _ = skills_gate.check_source(good)
        assert ok, reason


# ── the server honours the autonomy level ─────────────────────────────────
#
# `default.yaml` said "the autonomy level has to be `extend` before a skill
# may be served at all", and no code said it.  The server served at every
# level -- including `off` -- which made the gate the only thing between a
# mined fragment and a running tool.  A documented gate that does not exist
# is worse than an undocumented one, because it is relied on.

class TestTheServerHonoursTheLevel:

    def _pin_one(self, store):
        from src.learn import skills_curator
        candidate = {"tool": "bash", "digest": "f" * 64, "sessions": 3,
                     "calls": 5, "errors": 0, "success_rate": 1.0,
                     "arg_keys": ["cmd"]}
        skills_curator.synthesize(candidate, name="gated", config_data=store)
        skills_curator.promote("gated", config_data=store, delta=0.1)

    def _loaded_at(self, store, level):
        from src.mcp.servers.tasks.skills import mcp_server
        return mcp_server.load_skills(
            {"learn": {"path": store["learn"]["path"], "autonomy": level}})

    def test_nothing_is_served_below_extend(self, store):
        self._pin_one(store)
        for level in ("off", "observe", "adapt"):
            assert self._loaded_at(store, level) == {}, (
                "a pinned skill was served at autonomy %r" % level)

    def test_a_pinned_skill_is_served_at_extend(self, store):
        self._pin_one(store)
        assert "gated" in self._loaded_at(store, "extend")

    def test_the_level_defaults_to_serving_nothing(self, store, monkeypatch):
        """A server started by hand serves nothing.

        The level arrives in the environment; absent it, the package
        default is `observe`, which serves nothing.  The failure mode of a
        missing variable must be "no skills", not "every skill".
        """
        from src.mcp.servers.tasks.skills import mcp_server
        monkeypatch.delenv("ONIT_LEARN_AUTONOMY", raising=False)
        monkeypatch.delenv("ONIT_LEARN", raising=False)
        self._pin_one(store)
        assert mcp_server.load_skills(
            {"learn": {"path": store["learn"]["path"]}}) == {}


# ── the third round: the escapes the attribute rule closes ──────────────────
#
# The second round made the gate structural, and the audit's six bugs
# stopped reproducing.  It was still not enough, because every rule it
# added was about an *import* or a *dunder*, and the escape does not need
# either:
#
#   import uuid;  uuid.os.system(cmd)          os, one attribute away
#   import random; random._os.system(cmd)      the same, private spelling
#   import operator, uuid
#   operator.attrgetter("os")(uuid).system(c)  the same, name as data
#
# Every one of these imports an allowed module and calls a name that is
# not on any denylist.  The first two were confirmed to execute a command
# through `load_skills`; the gate admitted all three.

class TestTheEscapesTheAttributeRuleAdmits:
    """A skill may not reach the process through a module's own imports."""

    def _refused(self, source):
        ok, reason, _ = skills_gate.check_source(source)
        assert not ok, "the gate admitted: %r" % source
        assert reason, "the gate refused without saying why"
        return reason

    def _admitted(self, source):
        ok, reason, _ = skills_gate.check_source(source)
        assert ok, "the gate refused a legitimate skill: %s\n%r" % (
            reason, source)

    def test_os_off_a_module_that_imports_it_is_refused(self):
        """`uuid.os` is `os`.  The import screen saw only `uuid`."""
        reason = self._refused(
            "import uuid\n"
            "def run():\n"
            "    uuid.os.system('id')\n")
        # The refusal names whichever attribute the rule reached first,
        # which for this payload is the inner one -- ``system`` is the
        # capability, ``os`` is only the way to it.  Both are a refusal
        # of the same reach, so the assertion is that one of them is
        # named rather than that a particular one is.
        assert "os" in reason or "system" in reason

    def test_the_private_spelling_is_refused(self):
        """`random._os` is `os` with an underscore in front.

        This is the payload that executed.  A screen that matched the
        literal text `os.system` never saw `_os.system`, because `_` is a
        word character and the word boundary does not fall before `os`.
        """
        reason = self._refused(
            "import random\n"
            "def run():\n"
            "    random._os.system('id')\n")
        assert "system" in reason or "os" in reason

    def test_sys_off_a_module_that_imports_it_is_refused(self):
        self._refused(
            "import warnings\n"
            "def run():\n"
            "    return warnings.sys.modules\n")

    def test_a_chain_of_modules_is_refused(self):
        """`json.codecs.builtins.open` is three dots from a file read."""
        self._refused(
            "import json\n"
            "def run():\n"
            "    return json.codecs.builtins.open('/etc/passwd').read()\n")

    def test_the_same_reach_through_an_alias_is_refused(self):
        self._refused(
            "import uuid as u\n"
            "def run():\n"
            "    u.os.system('id')\n")

    def test_attrgetter_with_a_forbidden_name_is_refused(self):
        """The attribute name written as data, not as an attribute.

        `operator.attrgetter("os")(uuid)` produces the same object as
        `uuid.os`, and no Attribute node for `os` appears in the source.
        """
        self._refused(
            "import operator, uuid\n"
            "def run():\n"
            "    return operator.attrgetter('os')(uuid).system('id')\n")

    def test_getattr_with_a_forbidden_name_is_refused(self):
        self._refused(
            "import uuid\n"
            "def run():\n"
            "    return getattr(uuid, 'os').system('id')\n")

    def test_a_file_capability_off_a_module_is_refused(self):
        self._refused(
            "import fnmatch\n"
            "def run():\n"
            "    return fnmatch.os.remove('/tmp/x')\n")

    def test_a_shell_capability_off_a_module_is_refused(self):
        self._refused(
            "import argparse\n"
            "def run():\n"
            "    return argparse._os.popen('id')\n")


class TestTheAttributeRuleDoesNotRefuseOrdinaryCode:
    """The rule is scoped to a module root, and it has to stay scoped.

    A screen that refuses `xs.remove(1)` is a screen somebody deletes,
    and a deleted screen protects nothing.  The difference between
    `uuid.os.remove(p)` and `xs.remove(1)` is what the attribute hangs
    off, which is why the rule is about the root and not the name.
    """

    def _admitted(self, source):
        ok, reason, _ = skills_gate.check_source(source)
        assert ok, "the gate refused a legitimate skill: %s\n%r" % (
            reason, source)

    def test_a_list_method_named_like_a_file_call_is_admitted(self):
        self._admitted(
            "def run(xs=None):\n"
            "    xs = xs or [1, 2]\n"
            "    xs.remove(1)\n"
            "    return xs\n")

    def test_a_dict_key_named_like_a_capability_is_admitted(self):
        self._admitted(
            "def run(d=None):\n"
            "    d = d or {}\n"
            "    return d.get('remove')\n")

    def test_json_load_and_dump_are_admitted(self):
        self._admitted(
            "import json\n"
            "def run(t='{}'):\n"
            "    return json.dumps(sorted(json.loads(t)))\n")

    def test_a_module_attribute_that_is_not_a_capability_is_admitted(self):
        """`uuid.uuid4` hangs off an imported module and is fine."""
        self._admitted(
            "import uuid\n"
            "def run():\n"
            "    return str(uuid.uuid4())\n")

    def test_a_dataclass_with_super_init_is_admitted(self):
        self._admitted(
            "import dataclasses\n"
            "@dataclasses.dataclass\n"
            "class A:\n"
            "    x: int = 0\n"
            "class B(A):\n"
            "    def __init__(self):\n"
            "        super().__init__(1)\n"
            "def run():\n"
            "    return B().x\n")


# ── the gate at serve time ─────────────────────────────────────────────────
#
# Curation runs the gate once and records the verdict in the pin.  The pin
# is a file, and the file a running server reads is not necessarily the
# file the gate approved.  The server therefore re-runs the gate on every
# serve; without that, the gate is advisory and the pin is the boundary.

class TestTheGateRunsAtServeTime:

    def _store_with(self, source):
        import tempfile
        tmp = tempfile.mkdtemp()
        cfg = {"learn": {"path": tmp, "autonomy": "extend"}}
        directory = skills.skill_dir("sneaky", cfg)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "tool.py"), "w") as f:
            f.write(source)
        with open(os.path.join(directory, "manifest.yaml"), "w") as f:
            f.write("name: sneaky\nversion: 1\ndescription: d\n")
        skills.write_pin({"version": 1, "skills": {
            "sneaky": {"name": "sneaky", "version": 1,
                       "status": "probation"}}}, cfg)
        return cfg

    def test_a_pinned_skill_that_fails_the_gate_is_not_served(self):
        """The pin says serve it; the gate says it is an escape.

        The gate wins.  This is the whole point of running it twice: the
        pin records a past verdict, and the code in front of the loop is
        the present fact.
        """
        from src.mcp.servers.tasks.skills import mcp_server
        cfg = self._store_with(
            "import random\n"
            "def run():\n"
            "    return random._os.system('id')\n")
        assert mcp_server.load_skills(cfg) == {}

    def test_a_pinned_skill_that_passes_is_served(self):
        from src.mcp.servers.tasks.skills import mcp_server
        cfg = self._store_with(
            "import json\n"
            "def run(x=1):\n"
            "    return json.dumps({'x': x})\n")
        loaded = mcp_server.load_skills(cfg)
        assert "sneaky" in loaded
        assert loaded["sneaky"]["run"](2) == '{"x": 2}'
