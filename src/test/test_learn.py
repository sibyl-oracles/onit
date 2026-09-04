"""The experience substrate: what a run leaves behind, and what it must not.

Two things are being protected here. That a trajectory carries enough to
diagnose a run after the fact — which tools ran, which failed, how many turns
and retries it took — and that turning recording off leaves nothing on disk at
all, because a deployment that says no has to mean it.
"""

import json

import pytest

from learn import config as learn_config
from learn import trajectory as traj


@pytest.fixture
def learn_config_data(tmp_path):
    """Config with recording on, writing under tmp_path rather than ~/.onit."""
    return {"learn": {"autonomy": "observe", "path": str(tmp_path / "learned")}}


def _metrics(turns=None, **sink):
    base = {"turns": turns or [], "turn_count": len(turns or []),
            "tool_calls": 0, "compactions": 0, "api_retries": 0}
    base.update(sink)
    return base


class TestAutonomy:
    """One ladder, read the same way from a config file or the environment."""

    def test_defaults_to_observe(self):
        assert learn_config.autonomy({}) == learn_config.OBSERVE
        assert learn_config.recording_enabled({})

    def test_off_records_nothing(self):
        cfg = {"learn": {"autonomy": "off"}}
        assert learn_config.autonomy(cfg) == learn_config.OFF
        assert not learn_config.recording_enabled(cfg)

    def test_env_overrides_config(self, monkeypatch):
        """A baseline run pins the level without editing anyone's config."""
        cfg = {"learn": {"autonomy": "observe"}}
        monkeypatch.setenv("ONIT_LEARN", "off")
        assert learn_config.autonomy(cfg) == learn_config.OFF

    @pytest.mark.parametrize("value,expected", [
        ("observe", learn_config.OBSERVE), ("OFF", learn_config.OFF),
        ("0", learn_config.OFF), ("1", learn_config.OBSERVE),
        ("false", learn_config.OFF), ("true", learn_config.OBSERVE),
        ("", learn_config.OFF),
    ])
    def test_spellings(self, monkeypatch, value, expected):
        monkeypatch.setenv("ONIT_LEARN", value)
        assert learn_config.autonomy({}) == expected

    def test_unparseable_value_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("ONIT_LEARN", "sideways")
        assert learn_config.autonomy({}) == learn_config.DEFAULT_AUTONOMY

    def test_a_level_beyond_what_exists_is_capped(self):
        """Asking for `evolve` today must not produce a run that believes it is
        evolving.  It gets what is built, and the config keeps its meaning when
        the rest lands."""
        assert learn_config.autonomy({"learn": {"autonomy": "evolve"}}) == learn_config.OBSERVE

    def test_bare_boolean_config(self):
        assert learn_config.autonomy({"learn": False}) == learn_config.OFF
        assert learn_config.autonomy({"learn": {"enabled": False}}) == learn_config.OFF


class TestRedaction:
    """Argument names are kept; values are not, unless asked for."""

    def test_digest_ignores_key_order(self):
        assert (traj.args_digest({"a": 1, "b": 2})
                == traj.args_digest({"b": 2, "a": 1}))

    def test_digest_separates_different_values(self):
        assert traj.args_digest({"path": "/a"}) != traj.args_digest({"path": "/b"})

    def test_digest_survives_unserializable_arguments(self):
        assert traj.args_digest({"fn": object()})

    def test_redacted_call_keeps_names_and_drops_values(self):
        call = traj.describe_tool_call(
            "read_file", {"path": "/home/me/salary.pdf"},
            ok=True, ms=12, result_chars=99, redact=True)
        assert call["arg_keys"] == ["path"]
        assert "args" not in call
        assert "salary" not in json.dumps(call)

    def test_unredacted_call_keeps_values(self):
        call = traj.describe_tool_call(
            "read_file", {"path": "/tmp/x.pdf"},
            ok=True, ms=12, result_chars=99, redact=False)
        assert call["args"] == {"path": "/tmp/x.pdf"}

    def test_redaction_is_on_by_default(self):
        assert learn_config.redact_tool_args({}) is True

    def test_owner_is_hashed_not_stored(self):
        hashed = traj.owner_hash("someone@example.com")
        assert hashed and "example.com" not in hashed
        assert hashed == traj.owner_hash("someone@example.com")
        assert hashed != traj.owner_hash("other@example.com")

    def test_no_owner_stays_none(self):
        assert traj.owner_hash(None) is None


class TestSignals:
    """The cheap half of the reward, read off a finished run."""

    def test_counts_failed_tool_calls(self):
        metrics = _metrics(turns=[{"n": 1, "tool_runs": [
            {"name": "search", "ok": True}, {"name": "read_file", "ok": False}]}])
        assert traj.derive_signals(metrics)["tool_errors"] == 1

    def test_counts_truncations_and_retries(self):
        metrics = _metrics(
            turns=[{"n": 1, "finish_reason": "length"},
                   {"n": 2, "finish_reason": "stop"}],
            api_retries=2)
        signals = traj.derive_signals(metrics)
        assert signals["truncations"] == 1
        assert signals["retries"] == 2

    def test_a_clean_run_reads_as_clean(self):
        signals = traj.derive_signals(_metrics(turns=[{"n": 1, "finish_reason": "stop"}]))
        assert signals["tool_errors"] == 0
        assert signals["truncations"] == 0
        assert signals["user_rating"] is None

    def test_empty_metrics_do_not_raise(self):
        assert traj.derive_signals(None)["tool_errors"] == 0


class TestVerifierSignal:
    """§8 item 1: the fact-check's verdict, read off the metrics blob."""

    def test_a_check_that_ran_clean_reads_clean(self):
        metrics = _metrics(verify_s=1.2, verify_issues=0, verify_revisions=0)
        assert traj.derive_signals(metrics)["verifier"] == "clean"

    def test_corrected_claims_read_issues(self):
        metrics = _metrics(verify_s=18.0, verify_issues=2, verify_revisions=1)
        assert traj.derive_signals(metrics)["verifier"] == "issues"

    def test_a_check_that_never_ran_stays_none(self):
        """verify_s absent or zero means the checker was skipped — absence
        of a finding is not a finding, so this is not "clean"."""
        assert traj.derive_signals(_metrics())["verifier"] is None
        assert traj.derive_signals(_metrics(verify_s=0.0))["verifier"] is None
        assert traj.derive_signals(None)["verifier"] is None

    def test_helper_matches_the_derived_field(self):
        assert traj.verifier_signal(_metrics(verify_s=3.0, verify_issues=1)) == "issues"
        assert traj.verifier_signal(_metrics(verify_s=3.0, verify_issues=0)) == "clean"
        assert traj.verifier_signal({}) is None

    def test_a_garbage_issue_count_does_not_raise(self):
        assert traj.verifier_signal(_metrics(verify_s=1.0, verify_issues="many")) is None


class TestRecord:
    """Shape of a task record."""

    def test_carries_the_run_not_just_the_answer(self):
        record = traj.build_record(
            session_id="s1", turn=1, task="find the policy", response="here",
            tools_available=["search", "local_search"],
            metrics=_metrics(turns=[{
                "n": 1, "prompt_tokens": 500, "completion_tokens": 20,
                "finish_reason": "tool_calls",
                "tool_runs": [{"name": "local_search", "ok": True, "ms": 40}],
            }], tool_calls=1))
        assert record["schema"] == traj.SCHEMA_VERSION
        assert record["kind"] == traj.KIND_TASK
        assert record["tools_available"] == ["local_search", "search"]
        assert record["trajectory"][0]["tools"][0]["name"] == "local_search"
        assert record["trajectory"][0]["prompt_tokens"] == 500
        # The aggregate sink rides alongside, minus the turns it duplicates.
        assert record["metrics"]["tool_calls"] == 1
        assert "turns" not in record["metrics"]

    def test_names_survive_a_turn_recorded_without_rich_tool_data(self):
        record = traj.build_record(
            session_id="s1", turn=1, task="t", response="r",
            metrics=_metrics(turns=[{"n": 1, "tools": ["bash"]}]))
        assert record["trajectory"][0]["tools"] == [{"name": "bash"}]

    def test_learned_context_is_present_from_the_start(self):
        """A run recorded without this field cannot be compared against one
        recorded with it, and that comparison is the point of recording."""
        record = traj.build_record(session_id="s1", turn=1, task="t", response="r")
        assert record["learned_context"] == {"playbook_version": None,
                                             "episodes_used": []}


class TestWriter:
    def test_writes_one_line_per_task(self, learn_config_data):
        for turn in (1, 2):
            traj.record_task(session_id="s1", turn=turn, task=f"t{turn}",
                             response="r", config_data=learn_config_data)
        records = traj.read_session("s1", learn_config_data)
        assert [r["turn"] for r in records] == [1, 2]

    def test_sessions_are_separate_files(self, learn_config_data):
        traj.record_task(session_id="s1", turn=1, task="a", response="r",
                         config_data=learn_config_data)
        traj.record_task(session_id="s2", turn=1, task="b", response="r",
                         config_data=learn_config_data)
        assert len(traj.read_session("s1", learn_config_data)) == 1
        assert traj.read_session("s2", learn_config_data)[0]["task"] == "b"

    def test_off_writes_nothing_at_all(self, tmp_path):
        cfg = {"learn": {"autonomy": "off", "path": str(tmp_path / "learned")}}
        assert traj.record_task(session_id="s1", turn=1, task="t",
                                response="r", config_data=cfg) is None
        assert not (tmp_path / "learned").exists()

    def test_a_session_id_cannot_escape_the_directory(self, learn_config_data):
        import os
        directory = learn_config.trajectory_dir(learn_config_data)
        for hostile in ("../../etc/passwd", "/etc/passwd", "..", "a/b"):
            path = traj.session_file(hostile, learn_config_data)
            assert os.path.dirname(path) == directory
            assert os.path.normpath(path).startswith(os.path.normpath(directory))

    def test_an_unwritable_directory_does_not_raise(self, learn_config_data, monkeypatch):
        """A trajectory that fails to write must not turn a completed task into
        a failed one."""
        def _boom(*a, **k):
            raise OSError("read-only file system")
        monkeypatch.setattr(traj.os, "makedirs", _boom)
        assert traj.record_task(session_id="s1", turn=1, task="t", response="r",
                                config_data=learn_config_data) is None

    def test_a_torn_line_costs_one_record_not_the_file(self, learn_config_data):
        traj.record_task(session_id="s1", turn=1, task="good", response="r",
                         config_data=learn_config_data)
        with open(traj.session_file("s1", learn_config_data), "a") as f:
            f.write('{"kind": "task", "turn": 2, "tas\n')
        traj.record_task(session_id="s1", turn=3, task="also good", response="r",
                         config_data=learn_config_data)
        assert [r["turn"] for r in traj.read_session("s1", learn_config_data)] == [1, 3]

    def test_missing_session_reads_as_empty(self, learn_config_data):
        assert traj.read_session("never-existed", learn_config_data) == []

    def test_iter_records_walks_every_session(self, learn_config_data):
        traj.record_task(session_id="s1", turn=1, task="a", response="r",
                         config_data=learn_config_data)
        traj.record_task(session_id="s2", turn=1, task="b", response="r",
                         config_data=learn_config_data)
        tasks = {r["task"] for r in traj.iter_records(learn_config_data)}
        assert tasks == {"a", "b"}


class TestReport:
    """Recording nobody can read is indistinguishable from not recording."""

    def _run(self, cfg, sid, turn, tools, **metrics):
        traj.record_task(
            session_id=sid, turn=turn, task="t", response="r", config_data=cfg,
            metrics=_metrics(turns=[{"n": 1, "tool_runs": tools}], **metrics))

    def test_ranks_tools_by_failure_rate(self, learn_config_data):
        """The Phase 0 exit criterion: which tool fails most, and on what."""
        from learn import summarize
        cfg = learn_config_data
        self._run(cfg, "s1", 1, [{"name": "read_file", "ok": False, "ms": 40},
                                 {"name": "local_search", "ok": True, "ms": 800}])
        self._run(cfg, "s1", 2, [{"name": "read_file", "ok": True, "ms": 60},
                                 {"name": "local_search", "ok": True, "ms": 600}])
        s = summarize(cfg)
        assert s["tasks"] == 2 and s["sessions"] == 1
        assert s["tools"]["read_file"] == {"calls": 2, "errors": 1, "avg_ms": 50}
        assert s["tools"]["local_search"]["errors"] == 0

    def test_totals_roll_up_across_sessions(self, learn_config_data):
        from learn import summarize
        cfg = learn_config_data
        self._run(cfg, "s1", 1, [{"name": "bash", "ok": False}], api_retries=2)
        self._run(cfg, "s2", 1, [{"name": "bash", "ok": False}], api_retries=1)
        s = summarize(cfg)
        assert s["sessions"] == 2
        assert s["totals"]["tool_errors"] == 2
        assert s["totals"]["retries"] == 3

    def test_ratings_are_counted(self, learn_config_data):
        from learn import summarize
        cfg = learn_config_data
        for turn in (1, 2, 3):
            self._run(cfg, "s1", turn, [])
        traj.append_rating(session_id="s1", turn=1, rating="up", config_data=cfg)
        traj.append_rating(session_id="s1", turn=2, rating="down", config_data=cfg)
        s = summarize(cfg)
        assert s["ratings"] == {"up": 1, "down": 1}

    def test_verifier_labels_are_counted(self, learn_config_data):
        """The corpus is weakly labeled by the fact-check; the report shows
        how much of it is usable and how it splits."""
        from learn import summarize
        cfg = learn_config_data
        self._run(cfg, "s1", 1, [], verify_s=1.0, verify_issues=0)
        self._run(cfg, "s1", 2, [], verify_s=2.0, verify_issues=1)
        self._run(cfg, "s1", 3, [])  # check never ran
        s = summarize(cfg)
        assert s["verifier"] == {"clean": 1, "issues": 1, "unchecked": 1}

    def test_status_line_carries_the_verifier_counts(self, learn_config_data):
        from learn import format_status
        cfg = learn_config_data
        self._run(cfg, "s1", 1, [], verify_s=1.0, verify_issues=2)
        text = format_status(cfg)
        assert "Verifier" in text and "1 with issue(s)" in text

    def test_a_record_without_call_detail_invents_no_errors(self, learn_config_data):
        """Older records carry tool names only; counting those as failures
        would report errors that never happened."""
        from learn import summarize
        traj.record_task(session_id="s1", turn=1, task="t", response="r",
                         config_data=learn_config_data,
                         metrics=_metrics(turns=[{"n": 1, "tools": ["bash"]}]))
        assert summarize(learn_config_data)["tools"]["bash"] == {
            "calls": 1, "errors": 0, "avg_ms": 0}

    def test_status_says_so_when_nothing_is_recorded_yet(self, learn_config_data):
        from learn import format_status
        text = format_status(learn_config_data)
        assert "No trajectories recorded yet" in text
        assert "observe" in text

    def test_status_says_so_when_recording_is_off(self, tmp_path):
        from learn import format_status
        text = format_status({"learn": {"autonomy": "off",
                                        "path": str(tmp_path / "l")}})
        assert "Recording is off" in text
        assert "ONIT_LEARN=observe" in text

    def test_status_reports_what_is_there(self, learn_config_data):
        from learn import format_status
        self._run(learn_config_data, "s1", 1,
                  [{"name": "read_file", "ok": False, "ms": 40}])
        text = format_status(learn_config_data)
        assert "1 task(s) across 1 session(s)" in text
        assert "read_file" in text
        assert "100%" in text


class TestRatings:
    """A verdict arrives after the record it judges, from another process."""

    def test_rating_folds_into_the_task_it_judges(self, learn_config_data):
        traj.record_task(session_id="s1", turn=1, task="t", response="r",
                         config_data=learn_config_data)
        traj.append_rating(session_id="s1", turn=1, rating="up",
                           comment="exactly right", config_data=learn_config_data)
        record = traj.read_session("s1", learn_config_data)[0]
        assert record["signals"]["user_rating"] == 1
        assert record["signals"]["user_comment"] == "exactly right"

    def test_rating_does_not_rewrite_the_record(self, learn_config_data):
        """Appending is the whole storage strategy: the task line must be
        exactly as it was written."""
        traj.record_task(session_id="s1", turn=1, task="t", response="r",
                         config_data=learn_config_data)
        path = traj.session_file("s1", learn_config_data)
        first_line = open(path).readline()
        traj.append_rating(session_id="s1", turn=1, rating=-1,
                           config_data=learn_config_data)
        assert open(path).readline() == first_line
        assert len(open(path).readlines()) == 2

    def test_unrated_turns_stay_none(self, learn_config_data):
        traj.record_task(session_id="s1", turn=1, task="a", response="r",
                         config_data=learn_config_data)
        traj.record_task(session_id="s1", turn=2, task="b", response="r",
                         config_data=learn_config_data)
        traj.append_rating(session_id="s1", turn=2, rating="down",
                           config_data=learn_config_data)
        records = traj.read_session("s1", learn_config_data)
        assert records[0]["signals"]["user_rating"] is None
        assert records[1]["signals"]["user_rating"] == -1

    @pytest.mark.parametrize("value,expected", [
        ("up", 1), ("down", -1), (1, 1), (-1, -1), (True, 1), (False, -1),
        ("👍", 1), ("👎", -1), (0, None), ("shrug", None), (None, None),
    ])
    def test_spellings_normalize(self, value, expected):
        assert traj.normalize_rating(value) == expected

    def test_rating_with_recording_off_is_dropped(self, tmp_path):
        cfg = {"learn": {"autonomy": "off", "path": str(tmp_path / "learned")}}
        assert traj.append_rating(session_id="s1", turn=1, rating="up",
                                  config_data=cfg) is None
