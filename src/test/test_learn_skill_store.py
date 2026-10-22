"""
The skill library's store: the pin, the deltas, and the names.

``src/test/test_learn_skills.py`` covers the mining-to-promotion path.
This file covers the parts of the *store* that path leans on and does
not itself exercise: the manifest's validation rules, the delta
operations on the pin, and the name rules.

These are the rules that decide what the model is allowed to call.  The
pin is the allowlist, the name is a path component, and the manifest is
the description the model chooses the tool by -- each one is a
security-relevant decision, and each one is a place where a permissive
default would be a hole.
"""

import os

import pytest

from src.learn import skills


@pytest.fixture
def store(tmp_path):
    """An isolated skill library."""
    root = str(tmp_path / "learned")
    os.makedirs(root, exist_ok=True)
    return {"learn": {"path": root}}


# ── the name rules ────────────────────────────────────────────────────────

class TestNames:
    """A skill's name is a directory name and a tool name."""

    @pytest.mark.parametrize("name", [
        "my_skill", "a", "skill_2", "x" * 49, "read_file",
    ])
    def test_a_good_name_is_kept(self, name):
        assert skills._safe_name(name) == name

    @pytest.mark.parametrize("name", [
        "", "   ", None, "../escape", "a/b", "..", "/absolute",
        "has space", "has-dash", "UPPER", "with.dot", "x" * 50,
        "name;rm -rf", "$(whoami)", "naïve",
        # A name is also a Python identifier, so it starts with a letter.
        "_private", "1skill",
    ])
    def test_a_bad_name_is_refused(self, name):
        """A name that is not safe must not become a path.

        The name reaches ``skill_dir()`` and from there the filesystem,
        so a name like ``../escape`` would put the skill outside the
        store entirely.
        """
        assert skills._safe_name(name) == "", \
            "the name %r was accepted" % (name,)

    def test_a_name_is_trimmed(self):
        assert skills._safe_name("  my_skill  ") == "my_skill"

    def test_a_one_character_name_is_allowed(self):
        """One character is a short but legal name.

        The regex used to require two, which would have rejected a
        legitimate skill for no reason.
        """
        assert skills._safe_name("a") == "a"

    def test_a_long_name_is_refused(self):
        """The limit is 49 characters: one letter and up to 48 more."""
        assert skills._safe_name("x" * 49) != ""
        assert skills._safe_name("x" * 50) == ""

    def test_the_skill_id_is_stable(self):
        assert skills.skill_id("my_skill") == skills.skill_id("my_skill")

    def test_different_names_have_different_ids(self):
        assert skills.skill_id("a") != skills.skill_id("b")

    def test_the_id_is_a_prefix_and_a_hash(self):
        skill_id = skills.skill_id("my_skill")
        assert skill_id.startswith("sk-")
        assert len(skill_id) == 3 + 12


# ── the manifest rules ────────────────────────────────────────────────────

class TestManifestValidation:
    """A manifest that is wrong produces a tool the model cannot use."""

    def _manifest(self, **overrides):
        manifest = {
            "name": "my_skill",
            "description": "does a thing",
            "version": 1,
            "input_schema": {"type": "object", "properties": {}},
        }
        manifest.update(overrides)
        return manifest

    def test_a_good_manifest_has_no_problems(self):
        assert skills.manifest_errors(self._manifest()) == []

    def test_a_manifest_with_no_name_is_refused(self):
        problems = skills.manifest_errors(self._manifest(name=""))
        assert problems, "a manifest with no name was accepted"

    def test_a_manifest_with_no_description_is_refused(self):
        """No description, no tool.

        The model chooses a tool by its description.  A tool with no
        description is schema tokens spent on something the model
        cannot pick on purpose.
        """
        problems = skills.manifest_errors(self._manifest(description=""))
        assert problems

    def test_a_manifest_with_a_blank_description_is_refused(self):
        problems = skills.manifest_errors(self._manifest(description="   "))
        assert problems

    def test_an_overlong_description_is_refused(self):
        problems = skills.manifest_errors(
            self._manifest(description="x" * (skills.MAX_DESCRIPTION_CHARS + 1)))
        assert problems

    def test_a_manifest_with_no_version_is_refused(self):
        problems = skills.manifest_errors(self._manifest(version=None))
        assert problems

    def test_a_zero_version_is_refused(self):
        problems = skills.manifest_errors(self._manifest(version=0))
        assert problems

    def test_a_negative_version_is_refused(self):
        problems = skills.manifest_errors(self._manifest(version=-1))
        assert problems

    def test_a_non_integer_version_is_refused(self):
        problems = skills.manifest_errors(self._manifest(version="one"))
        assert problems

    def test_a_schema_that_is_not_a_mapping_is_refused(self):
        problems = skills.manifest_errors(
            self._manifest(input_schema="not a schema"))
        assert problems

    def test_properties_that_are_not_a_mapping_are_refused(self):
        problems = skills.manifest_errors(
            self._manifest(input_schema={"type": "object",
                                         "properties": "not a mapping"}))
        assert problems

    def test_an_overlong_parameter_name_is_refused(self):
        """A long parameter name is paid for on every request."""
        problems = skills.manifest_errors(
            self._manifest(input_schema={
                "type": "object",
                "properties": {"x" * (skills.MAX_PARAM_NAME + 1):
                               {"type": "string"}}}))
        assert problems

    def test_a_manifest_with_no_schema_is_accepted(self):
        """A skill with no parameters is a skill with no parameters."""
        manifest = self._manifest()
        del manifest["input_schema"]
        assert skills.manifest_errors(manifest) == []

    def test_an_empty_manifest_lists_every_problem(self):
        problems = skills.manifest_errors({})
        assert len(problems) >= 3, "the problems were not all reported"


# ── the pin ───────────────────────────────────────────────────────────────

class TestThePin:
    """The pin is the allowlist, so it has to be exact."""

    def test_a_missing_pin_pins_nothing(self, store):
        assert skills.pinned_skills(store) == {}

    def test_a_broken_pin_pins_nothing(self, store):
        path = skills.pinned_path(store)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("::: not yaml :::\n")
        assert skills.pinned_skills(store) == {}

    def test_a_pin_that_is_not_a_mapping_pins_nothing(self, store):
        path = skills.pinned_path(store)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("- a\n- b\n")
        assert skills.pinned_skills(store) == {}

    def test_a_pin_whose_skills_is_not_a_mapping_pins_nothing(self, store):
        path = skills.pinned_path(store)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("skills:\n  - a\n")
        assert skills.pinned_skills(store) == {}

    def test_a_written_pin_reads_back(self, store):
        skills.write_pin({"version": 3,
                          "skills": {"a": {"name": "a", "version": 1,
                                           "status": "active"}}}, store)
        pin = skills.read_pin(store)
        assert pin["version"] == 3
        assert "a" in pin["skills"]

    def test_the_pin_round_trips(self, store):
        original = {"version": 1,
                    "skills": {"a": {"name": "a", "version": 2,
                                     "status": "active",
                                     "description": "does a thing"}}}
        skills.write_pin(original, store)
        assert skills.read_pin(store) == original

    def test_a_pin_naming_a_skill_whose_files_are_gone_is_stale(self, store):
        """A pin that names a skill whose files were removed is stale.

        It is filtered at *read* time rather than at serve time, so
        that every reader agrees on what is live.  A reader that
        disagreed with the server about what is pinned would serve a
        skill the operator had removed.
        """
        skills.write_pin({"version": 1,
                          "skills": {"gone": {"name": "gone",
                                               "status": "active"}}}, store)
        assert skills.pinned_skills(store) == {}

    def test_a_pin_naming_an_existing_skill_serves_it(self, store):
        directory = skills.skill_dir("a", store)
        os.makedirs(directory, exist_ok=True)
        skills.write_manifest(os.path.join(directory, "manifest.yaml"),
                              {"name": "a", "description": "x", "version": 1})
        skills.write_pin({"version": 1,
                          "skills": {"a": {"name": "a", "status": "active"}}},
                         store)
        assert "a" in skills.pinned_skills(store)


# ── the deltas ────────────────────────────────────────────────────────────

class TestPinDeltas:
    """The pin changes by a delta, and the delta-only rules hold."""

    def test_an_insert_adds_a_skill(self):
        pin = skills.apply_deltas({"version": 0, "skills": {}},
                                  [{"op": "insert", "name": "a"}])
        assert "a" in pin["skills"]

    def test_an_insert_defaults_the_fields_it_needs(self):
        pin = skills.apply_deltas({"version": 0, "skills": {}},
                                  [{"op": "insert", "name": "a"}])
        entry = pin["skills"]["a"]
        assert entry["name"] == "a"
        assert entry["version"] == 1
        assert entry["status"]
        assert entry["pinned_at"]

    def test_inserting_the_same_name_twice_is_an_update(self):
        """Two entries for one name is a library nobody can reason
        about."""
        pin = skills.apply_deltas({"version": 0, "skills": {}},
                                  [{"op": "insert", "name": "a",
                                    "skill": {"description": "one"}},
                                   {"op": "insert", "name": "a",
                                    "skill": {"description": "two"}}])
        assert list(pin["skills"]) == ["a"]
        assert pin["skills"]["a"]["description"] == "two"

    def test_an_update_to_a_missing_skill_is_skipped_not_fatal(self):
        """The interesting failure of a delta-only system is a delta
        aimed at something that moved."""
        pin = skills.apply_deltas({"version": 0, "skills": {}},
                                  [{"op": "update", "name": "gone",
                                    "skill": {"description": "x"}}])
        assert pin["skills"] == {}
        assert pin.get("skipped"), "the skipped delta was not reported"

    def test_a_deprecate_to_a_missing_skill_is_skipped(self):
        pin = skills.apply_deltas({"version": 0, "skills": {}},
                                  [{"op": "deprecate", "name": "gone"}])
        assert pin["skills"] == {}
        assert pin.get("skipped")

    def test_a_deprecate_marks_the_skill(self):
        pin = skills.apply_deltas({"version": 0, "skills": {
            "a": {"name": "a", "status": "active"}}},
            [{"op": "deprecate", "name": "a", "reason": "it broke"}])
        assert pin["skills"]["a"]["status"] == "deprecated"
        assert pin["skills"]["a"]["reason"] == "it broke"

    def test_an_unknown_operation_is_skipped(self):
        pin = skills.apply_deltas({"version": 0, "skills": {}},
                                  [{"op": "explode", "name": "a"}])
        assert pin["skills"] == {}
        assert pin.get("skipped")

    def test_a_delta_with_an_unsafe_name_is_skipped(self):
        """A name that could escape the store must not be written."""
        pin = skills.apply_deltas({"version": 0, "skills": {}},
                                  [{"op": "insert", "name": "../escape"}])
        assert pin["skills"] == {}
        assert pin.get("skipped")

    def test_a_malformed_delta_is_ignored(self):
        for delta in ({}, {"op": "insert"}, "not a dict", None, 42):
            pin = skills.apply_deltas({"version": 0, "skills": {}}, [delta])
            assert pin["skills"] == {}

    def test_an_empty_delta_list_changes_nothing(self):
        before = {"version": 2, "skills": {"a": {"name": "a"}}}
        pin = skills.apply_deltas(before, [])
        assert pin["skills"] == before["skills"]

    def test_the_input_pin_is_not_mutated(self):
        """A delta-only system that mutates its input is not
        delta-only."""
        import copy
        before = {"version": 1, "skills": {"a": {"name": "a",
                                                 "status": "active"}}}
        snapshot = copy.deepcopy(before)
        skills.apply_deltas(before, [{"op": "deprecate", "name": "a"}])
        assert before == snapshot

    def test_a_change_bumps_the_version(self):
        """The version is the pin's own counter, and it moves when the
        pin does -- so a reader can tell two pins apart."""
        pin = skills.apply_deltas({"version": 7, "skills": {}},
                                  [{"op": "insert", "name": "a"}])
        assert pin["version"] == 8

    def test_the_version_counts_applications_not_changes(self):
        """The pin's version moves every time a delta is applied.

        Unlike the playbook's -- which bumps only when something
        actually changed -- the pin's version counts *applications*.
        The difference is deliberate and harmless: the pin's version
        is a monotonic counter of how many times the library has been
        written, and the rollback restores the file rather than
        reconstructing it, so the counter never has to be meaningful
        as a content hash.
        """
        pin = skills.apply_deltas({"version": 7, "skills": {}}, [])
        assert pin["version"] == 8
        again = skills.apply_deltas(pin, [])
        assert again["version"] == 9


# ── the listing ───────────────────────────────────────────────────────────

class TestTheListing:
    """A listing that changes order between two runs is a listing
    nobody can read twice."""

    def test_an_empty_library_lists_nothing(self, store):
        assert skills.list_skills(store) == []

    def test_the_listing_is_sorted(self, store):
        for name in ("zebra", "alpha", "middle"):
            directory = skills.skill_dir(name, store)
            os.makedirs(directory, exist_ok=True)
            skills.write_manifest(os.path.join(directory, "manifest.yaml"),
                                  {"name": name, "description": "x",
                                   "version": 1})
        names = [entry["name"] for entry in skills.list_skills(store)]
        assert names == sorted(names)

    def test_a_directory_without_a_manifest_is_not_listed(self, store):
        """A half-written skill is not a skill."""
        directory = skills.skill_dir("broken", store)
        os.makedirs(directory, exist_ok=True)
        assert "broken" not in [e["name"] for e in skills.list_skills(store)]

    def test_a_file_is_not_a_skill(self, store):
        path = skills.skill_dir("not_a_dir", store)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("hello")
        assert skills.list_skills(store) == []

    def test_the_summary_counts_the_pinned_and_the_unpinned(self, store):
        for name in ("pinned_one", "loose_one"):
            directory = skills.skill_dir(name, store)
            os.makedirs(directory, exist_ok=True)
            skills.write_manifest(os.path.join(directory, "manifest.yaml"),
                                  {"name": name, "description": "x",
                                   "version": 1})
        skills.write_pin({"version": 1,
                          "skills": {"pinned_one": {"name": "pinned_one",
                                                    "status": "active"}}},
                         store)
        summary = skills.summarize(store)
        assert summary["skills"] == 2
        assert summary["pinned"] == 1

    def test_reading_a_skill_that_is_not_there_is_empty(self, store):
        assert skills.read_skill("never_made", store) == {}

    def test_reading_a_skill_with_an_unsafe_name_is_empty(self, store):
        assert skills.read_skill("../escape", store) == {}
