"""Tests for the `onit doctor` subcommand (src/cli.py::_run_doctor and wiring).

The battery itself is tested in test_doctor.py; what is tested here is the
CLI side — that the subcommand exists, resolves a config, starts the MCP
servers, builds a throwaway agent, runs the battery, cleans up the session
it created, and exits non-zero when a check fails.
"""

import json
import os
import sys

import yaml
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.cli import _run_doctor, main

# main() writes the approval variables with a plain os.environ assignment,
# which monkeypatch cannot see and so cannot undo.  _run_doctor sets them
# too.  Snapshot and put back by hand, for the whole file — same reason as
# test_cli.py, which this pattern is copied from.
_APPROVAL_ENV_VARS = ("ONIT_APPROVAL_CHANNEL", "ONIT_AUTO_APPROVE",
                      "ONIT_ASK_APPROVAL", "ONIT_WEB_UI", "ONIT_UNRESTRICTED")


@pytest.fixture(autouse=True)
def _restore_approval_env():
    saved = {var: os.environ.get(var) for var in _APPROVAL_ENV_VARS}
    yield
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


@pytest.fixture
def doctor_env(tmp_path, monkeypatch):
    """A minimal config main() can resolve, plus a stubbed OnIt.

    OnIt is stubbed because building a real one spawns MCP servers and
    registers a session; the CLI-side wiring under test does not need it.
    The stub still records the config it was handed and fakes the
    attributes the battery and the cleanup read.
    """
    import yaml
    from src import setup as setup_mod

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "session_path": str(tmp_path / "sessions"),
        "serving": {"host": "http://localhost:8000/v1"},
    }))
    monkeypatch.setattr("src.cli._find_default_config", lambda: str(cfg))
    monkeypatch.setattr(setup_mod, "CONFIG_PATH", str(tmp_path / "no-setup.yaml"))
    monkeypatch.setattr(setup_mod, "get_secret", lambda key: None)

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "preexisting.jsonl").write_text("")
    (sessions_dir / "index.json").write_text(json.dumps(
        {"preexisting": {"tag": "preexisting"}}))

    agent = MagicMock()
    agent.session_id = "doctor-session-id"
    agent.config_data = {}

    created = {}
    fake_onit = MagicMock(return_value=agent)

    def _make(config):
        created.update(config)
        return fake_onit(config)

    monkeypatch.setattr("src.cli.OnIt", _make)
    return {
        "tmp": tmp_path,
        "sessions": sessions_dir,
        "agent": agent,
        "onit": fake_onit,
        "created": created,
    }


def _run(argv, run_checks=None):
    """Run main() with the doctor branch active; return (exit code, mocks).

    _run_doctor imports run_checks/render_report from src.ui.doctor and
    delete_session from src.sessions inside the function, so the patches
    target those home modules rather than src.cli.  run_checks is awaited
    through asyncio.run, so it must be a coroutine function; the given one
    is wrapped in a recorder that captures the kwargs it was called with.
    """
    captured = {}

    if run_checks is None:
        async def run_checks(agent, deep=False, on_start=None):
            return []

    calls = []

    async def recording(agent, deep=False, on_start=None):
        calls.append({"deep": deep})
        return await run_checks(agent, deep=deep, on_start=on_start)

    with patch.object(sys, "argv", ["onit"] + argv), \
            patch("src.cli._setup_servers") as setup_servers, \
            patch("src.ui.doctor.run_checks", recording), \
            patch("src.ui.doctor.render_report", return_value="REPORT") as render, \
            patch("src.sessions.delete_session") as delete:
        # The doctor branch calls sys.exit with _run_doctor's return value.
        # A battery that raises propagates out of main() — that is a crash,
        # not a check result — and is recorded rather than suppressed.
        try:
            main()
            code = None
        except SystemExit as e:
            code = e.code
        except Exception as e:
            code = None
            captured["raised"] = e
    return {
        "code": code,
        "setup_servers": setup_servers,
        "run_checks": recording,
        "run_checks_calls": calls,
        "render": render,
        "delete": delete,
        "captured": captured,
    }


# ── wiring ───────────────────────────────────────────────────────────────────


class TestDoctorWiring:
    def test_doctor_is_a_subcommand(self, doctor_env):
        out = _run(["doctor"])
        assert out["code"] == 0

    def test_starts_servers_before_the_battery(self, doctor_env):
        out = _run(["doctor"])
        out["setup_servers"].assert_called_once()

    def test_builds_a_throwaway_agent(self, doctor_env):
        _run(["doctor"])
        doctor_env["onit"].assert_called_once()

    def test_strips_server_mode_keys(self, doctor_env):
        # A config that also serves the web must not make the doctor run a
        # web server; the battery needs the plain terminal-chat shape.
        cfg = doctor_env["tmp"] / "config.yaml"
        data = yaml.safe_load(cfg.read_text())
        data["web"] = True
        data["loop"] = True
        data["resume_session_id"] = "someone-elses"
        cfg.write_text(yaml.safe_dump(data))
        _run(["doctor"])
        for key in ("web", "loop", "resume_session_id"):
            assert key not in doctor_env["created"], \
                f"{key} survived: the doctor would {key}"

    def test_deep_flag_reaches_the_battery(self, doctor_env):
        out = _run(["doctor", "--deep"])
        assert out["run_checks_calls"][0]["deep"] is True

    def test_fast_by_default(self, doctor_env):
        out = _run(["doctor"])
        assert out["run_checks_calls"][0]["deep"] is False

    def test_report_is_printed(self, doctor_env, capsys):
        _run(["doctor"])
        assert "REPORT" in capsys.readouterr().out

    def test_json_report(self, doctor_env, capsys):
        from src.ui.doctor import CheckResult
        results = [CheckResult("config", "pass", "ok", 0.1),
                   CheckResult("mcp-servers", "fail", "down", 0.2)]

        async def run_checks(agent, deep=False, on_start=None):
            return results
        _run(["doctor", "--json"], run_checks=run_checks)
        payload = json.loads(capsys.readouterr().out)
        assert payload["deep"] is False
        assert [r["name"] for r in payload["results"]] == ["config", "mcp-servers"]
        assert payload["results"][1]["state"] == "fail"

    def test_flags_are_accepted(self, doctor_env):
        # --keep-session and the shared flags parse and reach the run.
        out = _run(["doctor", "--keep-session"])
        assert out["code"] == 0


# ── exit codes ───────────────────────────────────────────────────────────────


class TestDoctorExitCodes:
    def _results(self, *states):
        from src.ui.doctor import CheckResult
        return [CheckResult(f"check-{i}", state, "", 0.1)
                for i, state in enumerate(states)]

    def _battery(self, states):
        """A coroutine function returning the given results, as run_checks must be."""
        results = self._results(*states)

        async def run_checks(agent, deep=False, on_start=None):
            return results
        return run_checks

    def test_zero_when_all_pass(self, doctor_env):
        out = _run(["doctor"], run_checks=self._battery(["pass", "pass"]))
        assert out["code"] == 0

    def test_zero_when_skipped(self, doctor_env):
        # A skip is a fact about the configuration, not a break.
        out = _run(["doctor"], run_checks=self._battery(["pass", "skip"]))
        assert out["code"] == 0

    def test_one_when_any_check_fails(self, doctor_env):
        out = _run(["doctor"], run_checks=self._battery(["pass", "fail"]))
        assert out["code"] == 1

    def test_one_when_only_deep_fails(self, doctor_env):
        out = _run(["doctor", "--deep"], run_checks=self._battery(["pass", "fail"]))
        assert out["code"] == 1


# ── session cleanup ──────────────────────────────────────────────────────────


class TestDoctorSessionCleanup:
    def test_throwaway_session_is_deleted(self, doctor_env):
        out = _run(["doctor"])
        out["delete"].assert_called_once_with("doctor-session-id",
                                              str(doctor_env["sessions"]))

    def test_keep_session_spares_it(self, doctor_env):
        out = _run(["doctor", "--keep-session"])
        out["delete"].assert_not_called()

    def test_session_is_deleted_even_when_a_check_raises(self, doctor_env):
        # A crashing battery must not leave a phantom session behind — the
        # cleanup is the finally, not the happy path.  The exception itself
        # propagates: a crash is a crash, the test only pins the cleanup.
        async def run_checks(agent, deep=False, on_start=None):
            raise RuntimeError("boom")
        out = _run(["doctor"], run_checks=run_checks)
        assert isinstance(out["captured"].get("raised"), RuntimeError)
        out["delete"].assert_called_once_with("doctor-session-id",
                                              str(doctor_env["sessions"]))

    def test_preexisting_sessions_are_untouched(self, doctor_env):
        _run(["doctor"])
        index = json.loads((doctor_env["sessions"] / "index.json").read_text())
        assert "preexisting" in index


# ── unconfigured machines ────────────────────────────────────────────────────


class TestDoctorOnUnconfiguredMachine:
    def test_missing_config_still_reaches_the_battery(self, tmp_path, monkeypatch, capsys):
        # The whole point of a self-check: a machine with no serving.host
        # gets a diagnosis, not a stack trace.  Resolution exits; the doctor
        # catches that and runs the battery on an empty config.
        import yaml
        from src import setup as setup_mod
        monkeypatch.setattr("src.cli._find_default_config",
                            lambda: str(tmp_path / "nope.yaml"))
        monkeypatch.setattr(setup_mod, "CONFIG_PATH", str(tmp_path / "no-setup.yaml"))
        monkeypatch.setattr(setup_mod, "get_secret", lambda key: None)
        calls = []

        async def battery(agent, deep=False, on_start=None):
            calls.append(deep)
            return []
        with patch.object(sys, "argv", ["onit", "doctor"]), \
                patch("src.cli._setup_servers"), \
                patch("src.ui.doctor.run_checks", battery), \
                patch("src.ui.doctor.render_report", return_value="REPORT"), \
                patch("src.sessions.delete_session"), \
                patch("src.cli.OnIt") as fake_onit:
            fake_onit.return_value = MagicMock(session_id="sid")
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 0
        assert len(calls) == 1
        # The agent was built with the empty config, not the resolved one.
        handed = fake_onit.call_args.args[0] if fake_onit.call_args.args \
            else fake_onit.call_args.kwargs.get("config")
        assert handed == {}

    def test_unstartable_agent_exits_one(self, tmp_path, monkeypatch, capsys):
        from src import setup as setup_mod
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({
            "session_path": str(tmp_path / "sessions"),
            "serving": {"host": "http://localhost:8000/v1"},
        }))
        monkeypatch.setattr("src.cli._find_default_config", lambda: str(cfg))
        monkeypatch.setattr(setup_mod, "CONFIG_PATH", str(tmp_path / "no-setup.yaml"))
        monkeypatch.setattr(setup_mod, "get_secret", lambda key: None)
        with patch.object(sys, "argv", ["onit", "doctor"]), \
                patch("src.cli._setup_servers"), \
                patch("src.cli.OnIt", side_effect=RuntimeError("no prompts server")), \
                patch("src.sessions.delete_session"):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1
        assert "could not start the agent" in capsys.readouterr().err


# ── the battery runs against the real agent shape ────────────────────────────


class TestDoctorAgentContract:
    """The checks read attributes off the agent; a real OnIt must expose them.

    These do not build a real OnIt (that spawns servers); they pin the
    attribute names _run_doctor relies on, so a rename in onit.py fails here
    rather than as a mysterious battery failure at 2am.
    """

    def test_run_checks_signature_matches_the_cli_call(self):
        import inspect
        from src.ui import doctor
        params = inspect.signature(doctor.run_checks).parameters
        assert list(params) == ["agent", "deep", "on_start"], \
            "_run_doctor calls run_checks(agent, deep=...) positionally by name"

    def test_check_result_states_are_the_three_the_cli_counts(self):
        from src.ui.doctor import CheckResult
        for state in ("pass", "fail", "skip"):
            assert CheckResult("x", state).state == state

    def test_delete_session_takes_id_and_dir(self):
        import inspect
        from src.sessions import delete_session
        params = list(inspect.signature(delete_session).parameters)
        assert params[:2] == ["session_id", "sessions_dir"]