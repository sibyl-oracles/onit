"""Tests for src/cli.py — _find_default_config, _send_task, _download_files, _upload_file."""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.cli import (
    _find_default_config, _send_task, _download_files, _upload_file,
    _mcp_servers_ready, _ensure_mcp_servers,
)


# ── _find_default_config ────────────────────────────────────────────────────

class TestFindDefaultConfig:
    def test_finds_config_in_configs_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_file = config_dir / "default.yaml"
        config_file.write_text("serving:\n  host: x\n")
        result = _find_default_config()
        assert "default.yaml" in result

    def test_returns_fallback_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("src.cli.os.path.isfile", return_value=False):
            result = _find_default_config()
        assert result == "configs/default.yaml"


# ── _download_files ─────────────────────────────────────────────────────────

class TestDownloadFiles:
    def test_downloads_referenced_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        mock_resp = MagicMock()
        mock_resp.content = b"file contents"
        mock_resp.raise_for_status = MagicMock()

        with patch("src.cli.requests.get", return_value=mock_resp):
            result = _download_files(
                "Here is the file: /uploads/report.pdf",
                "http://localhost:9001",
            )

        assert "Downloaded files:" in result
        assert (tmp_path / "report.pdf").exists()

    def test_handles_download_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        with patch("src.cli.requests.get", side_effect=Exception("timeout")):
            result = _download_files(
                "File: /uploads/missing.txt",
                "http://localhost:9001",
            )

        assert "Failed to download" in result

    def test_no_files_returns_unchanged(self):
        result = _download_files("No files here", "http://localhost:9001")
        assert result == "No files here"


# ── _upload_file ────────────────────────────────────────────────────────────

class TestUploadFile:
    def test_uploads_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("src.cli.requests.post", return_value=mock_resp):
            filename = _upload_file("http://localhost:9001", str(test_file))

        assert filename == "test.txt"

    def test_upload_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _upload_file("http://localhost:9001", "/nonexistent/file.txt")


# ── _send_task ──────────────────────────────────────────────────────────────

class TestSendTask:
    def test_sends_task_and_returns_text(self):
        response_data = {
            "result": {
                "status": {"state": "completed"},
                "artifacts": [
                    {"parts": [{"kind": "text", "text": "The answer is 42."}]}
                ],
            }
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("src.cli.requests.post", return_value=mock_resp):
            result = _send_task("http://localhost:9001", "What is 6*7?")

        assert "42" in result

    def test_handles_error_response(self):
        response_data = {"error": {"code": -32600, "message": "Invalid request"}}

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("src.cli.requests.post", return_value=mock_resp):
            result = _send_task("http://localhost:9001", "bad request")

        assert "Error" in result

    def test_sends_with_file_inline(self, tmp_path):
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2")

        response_data = {
            "result": {
                "status": {"state": "completed"},
                "artifacts": [{"parts": [{"kind": "text", "text": "Processed."}]}],
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("src.cli.requests.post", return_value=mock_resp) as mock_post:
            result = _send_task("http://localhost:9001", "analyze this", file=str(test_file))

        assert "Processed" in result
        # Verify file was embedded inline (single POST, not upload + send)
        assert mock_post.call_count == 1
        payload = mock_post.call_args[1].get("json") or mock_post.call_args[0][1] if len(mock_post.call_args[0]) > 1 else mock_post.call_args[1]["json"]
        parts = payload["params"]["message"]["parts"]
        assert len(parts) == 2
        assert parts[1]["kind"] == "file"
        assert parts[1]["file"]["name"] == "data.csv"

    def test_direct_message_response_format(self):
        """Handle A2A responses that have parts directly in result."""
        response_data = {
            "result": {
                "parts": [{"kind": "text", "text": "Direct answer."}]
            }
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("src.cli.requests.post", return_value=mock_resp):
            result = _send_task("http://localhost:9001", "question")

        assert "Direct answer" in result

    def test_fallback_to_json_dump(self):
        """When no text part found, falls back to JSON dump."""
        response_data = {"result": {"something": "unexpected"}}

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("src.cli.requests.post", return_value=mock_resp):
            result = _send_task("http://localhost:9001", "question")

        assert "unexpected" in result


# ── _mcp_servers_ready ─────────────────────────────────────────────────────

def _mock_mcp_client(raises=None):
    """Build a mock fastmcp.Client context manager for readiness probes."""
    client = AsyncMock()
    if raises:
        client.list_tools = AsyncMock(side_effect=raises)
    else:
        client.list_tools = AsyncMock(return_value=[MagicMock()])
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestMcpServersReady:
    def test_returns_true_when_all_servers_up(self):
        config = {
            "mcp": {
                "servers": [
                    {"name": "A", "url": "http://127.0.0.1:18200/sse", "enabled": True},
                    {"name": "B", "url": "http://127.0.0.1:18201/sse", "enabled": True},
                ]
            }
        }
        mock = _mock_mcp_client()
        with patch("src.cli.Client", return_value=mock):
            assert _mcp_servers_ready(config, timeout=1.0) is True

    def test_returns_true_when_no_servers(self):
        assert _mcp_servers_ready({}, timeout=1.0) is True

    def test_returns_false_when_server_unreachable(self):
        config = {
            "mcp": {
                "servers": [
                    {"name": "A", "url": "http://127.0.0.1:18200/sse", "enabled": True},
                ]
            }
        }
        mock = _mock_mcp_client(raises=ConnectionError("refused"))
        with patch("src.cli.Client", return_value=mock):
            assert _mcp_servers_ready(config, timeout=0.5) is False


# ── _ensure_mcp_servers ────────────────────────────────────────────────────

class TestEnsureMcpServers:
    @staticmethod
    def _http_config():
        return {"mcp": {"servers": [
            {"name": "A", "url": "http://127.0.0.1:18200/sse", "enabled": True},
        ]}}

    def test_always_starts_its_own_servers(self):
        """A port answering on 18200 belongs to somebody else's OnIt.

        This used to be read as "my servers are already up" and startup was
        skipped, which pointed this user's tools at another user's server
        process — running as them, jailed to their sandbox. Now every OnIt
        starts its own.
        """
        config = self._http_config()
        mock_thread_instance = MagicMock()
        with patch("src.cli.threading.Thread", return_value=mock_thread_instance), \
             patch("src.cli._mcp_servers_ready", return_value=True):
            _ensure_mcp_servers(config)
            mock_thread_instance.start.assert_called_once()

    def test_rewrites_urls_to_the_allocated_ports(self):
        config = self._http_config()
        with patch("src.cli.threading.Thread", return_value=MagicMock()), \
             patch("src.cli._mcp_servers_ready", return_value=True), \
             patch("src.mcp.servers.run.find_free_ports", return_value=[18999]):
            _ensure_mcp_servers(config)
        assert config["mcp"]["servers"][0]["url"] == "http://127.0.0.1:18999/sse"

    def test_fixed_ports_opt_out_keeps_the_configured_url(self):
        config = self._http_config()
        config["mcp"]["fixed_ports"] = True
        with patch("src.cli.threading.Thread", return_value=MagicMock()), \
             patch("src.cli._mcp_servers_ready", return_value=True):
            _ensure_mcp_servers(config)
        assert config["mcp"]["servers"][0]["url"] == "http://127.0.0.1:18200/sse"

    def test_stdio_server_gets_a_spec_and_no_pool(self):
        """A stdio-only config starts no pool: the client spawns the server."""
        from type.tools import _STDIO_SPECS
        config = {"mcp": {"servers": [
            {"name": "ToolsLocalMCPServer", "transport": "stdio",
             "module": "tasks.tools", "profile": "local", "enabled": True},
        ]}}
        with patch("src.cli.threading.Thread") as mock_thread:
            _ensure_mcp_servers(config)
            mock_thread.assert_not_called()

        server = config["mcp"]["servers"][0]
        assert server["url"] == "stdio://ToolsLocalMCPServer"
        spec = _STDIO_SPECS[server["url"]]
        assert "--profile" in spec["args"] and "local" in spec["args"]
        assert "tasks.tools" in spec["args"]
        # The session directory is pinned in the child's environment; that is
        # what confines its tools to this user's sandbox.
        assert spec["env"]["ONIT_DATA_PATH"] == os.environ["ONIT_DATA_PATH"]
        # An explicit env must not strip the rest: the MCP SDK hands a
        # subprocess only five variables unless given a full environment.
        assert "PATH" in spec["env"]

    def test_stdio_server_without_a_module_is_disabled(self):
        config = {"mcp": {"servers": [
            {"name": "Broken", "transport": "stdio", "enabled": True},
        ]}}
        with patch("src.cli.threading.Thread"):
            _ensure_mcp_servers(config)
        assert config["mcp"]["servers"][0]["enabled"] is False


# ── Web UI OAuth credential resolution ──────────────────────────────────────

class TestWebOAuthCredentialResolution:
    """Google OAuth2 credentials stored by 'onit setup' (keyring) or env vars
    must reach the resolved config when web mode is on — the web UI refuses
    to start without them."""

    def _resolve(self, tmp_path, monkeypatch, cfg, secrets=None):
        import yaml
        from src import setup as setup_mod
        from src.cli import _build_parser, _parse_and_resolve_config
        secrets = secrets or {}
        monkeypatch.setattr(setup_mod, "get_secret", lambda key: secrets.get(key))
        # Keep the user's real ~/.onit/config.yaml out of the test
        monkeypatch.setattr(setup_mod, "CONFIG_PATH", str(tmp_path / "no-setup.yaml"))
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg))
        args = _build_parser().parse_args(["--config", str(path)])
        return _parse_and_resolve_config(args)

    def test_keyring_credentials_reach_web_config(self, tmp_path, monkeypatch):
        cfg = {"web": True, "serving": {"host": "http://localhost:8000/v1"}}
        secrets = {"web_google_client_id": "kr-id.apps.googleusercontent.com",
                   "web_google_client_secret": "kr-secret"}
        resolved = self._resolve(tmp_path, monkeypatch, cfg, secrets)
        assert resolved["web_google_client_id"] == "kr-id.apps.googleusercontent.com"
        assert resolved["web_google_client_secret"] == "kr-secret"

    def test_env_credentials_reach_web_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-id.apps.googleusercontent.com")
        monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")
        cfg = {"web": True, "serving": {"host": "http://localhost:8000/v1"}}
        resolved = self._resolve(tmp_path, monkeypatch, cfg)
        assert resolved["web_google_client_id"] == "env-id.apps.googleusercontent.com"
        assert resolved["web_google_client_secret"] == "env-secret"

    def test_config_file_value_wins(self, tmp_path, monkeypatch):
        cfg = {"web": True, "serving": {"host": "http://localhost:8000/v1"},
               "web_google_client_id": "from-config"}
        secrets = {"web_google_client_id": "kr-id"}
        resolved = self._resolve(tmp_path, monkeypatch, cfg, secrets)
        assert resolved["web_google_client_id"] == "from-config"

    def test_not_resolved_outside_web_mode(self, tmp_path, monkeypatch):
        cfg = {"serving": {"host": "http://localhost:8000/v1"}}
        secrets = {"web_google_client_id": "kr-id"}
        resolved = self._resolve(tmp_path, monkeypatch, cfg, secrets)
        assert "web_google_client_id" not in resolved

    def test_no_login_flag_disables_required_auth(self, tmp_path, monkeypatch):
        import yaml
        from src import setup as setup_mod
        from src.cli import _build_parser, _parse_and_resolve_config
        monkeypatch.setattr(setup_mod, "get_secret", lambda key: None)
        monkeypatch.setattr(setup_mod, "CONFIG_PATH", str(tmp_path / "no-setup.yaml"))
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({"serving": {"host": "http://localhost:8000/v1"}}))
        args = _build_parser().parse_args(
            ["--config", str(path), "serve", "web", "--no-login"])
        resolved = _parse_and_resolve_config(args)
        assert resolved["web"] is True
        assert resolved["web_require_auth"] is False

    def test_serve_web_defaults_to_required_auth(self, tmp_path, monkeypatch):
        import yaml
        from src import setup as setup_mod
        from src.cli import _build_parser, _parse_and_resolve_config
        monkeypatch.setattr(setup_mod, "get_secret", lambda key: None)
        monkeypatch.setattr(setup_mod, "CONFIG_PATH", str(tmp_path / "no-setup.yaml"))
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({"serving": {"host": "http://localhost:8000/v1"}}))
        args = _build_parser().parse_args(["--config", str(path), "serve", "web"])
        resolved = _parse_and_resolve_config(args)
        assert "web_require_auth" not in resolved  # falls back to default True


# ── --host / --host2 serving overrides ──────────────────────────────────────

class TestHostOverrides:
    """An explicit --host without --host2 must yield a single endpoint: any
    host2 left over from config or env would keep the load balancer routing
    to the old server (Ollama endpoints are fallback-only, so a healthy vLLM
    host2 would shadow an explicitly requested Ollama --host entirely)."""

    _CFG = {"serving": {"host": "http://vllm1:8001/v1",
                        "host2": "http://vllm2:8000/v1",
                        "model2": "some/vllm-model",
                        "host2_key": "k2"}}

    def _resolve(self, tmp_path, monkeypatch, cfg, cli_args):
        import yaml
        from src import setup as setup_mod
        from src.cli import _build_parser, _parse_and_resolve_config
        monkeypatch.setattr(setup_mod, "get_secret", lambda key: None)
        monkeypatch.setattr(setup_mod, "CONFIG_PATH", str(tmp_path / "no-setup.yaml"))
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg))
        args = _build_parser().parse_args(["--config", str(path)] + cli_args)
        return _parse_and_resolve_config(args)

    def test_host_alone_drops_config_host2(self, tmp_path, monkeypatch):
        resolved = self._resolve(tmp_path, monkeypatch, self._CFG,
                                 ["--host", "https://api.ollama.com"])
        serving = resolved["serving"]
        assert serving["host"] == "https://api.ollama.com"
        assert "host2" not in serving
        assert "model2" not in serving
        assert "host2_key" not in serving

    def test_host_alone_clears_env_host2(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONIT_HOST2", "http://vllm2:8000/v1")
        self._resolve(tmp_path, monkeypatch, self._CFG,
                      ["--host", "https://api.ollama.com"])
        assert "ONIT_HOST2" not in os.environ

    def test_host_with_host2_keeps_both(self, tmp_path, monkeypatch):
        resolved = self._resolve(
            tmp_path, monkeypatch, self._CFG,
            ["--host", "https://api.ollama.com",
             "--host2", "http://other:8002/v1"])
        serving = resolved["serving"]
        assert serving["host"] == "https://api.ollama.com"
        assert serving["host2"] == "http://other:8002/v1"

    def test_no_host_flag_keeps_config_host2(self, tmp_path, monkeypatch):
        resolved = self._resolve(tmp_path, monkeypatch, self._CFG, [])
        serving = resolved["serving"]
        assert serving["host"] == "http://vllm1:8001/v1"
        assert serving["host2"] == "http://vllm2:8000/v1"

    def test_no_ollama_fallback_only_flag(self, tmp_path, monkeypatch):
        resolved = self._resolve(tmp_path, monkeypatch, self._CFG,
                                 ["--no-ollama-fallback-only"])
        assert resolved["serving"]["ollama_fallback_only"] is False

    def test_ollama_fallback_only_flag_overrides_config(self, tmp_path, monkeypatch):
        cfg = {"serving": {"host": "http://vllm1:8001/v1",
                           "ollama_fallback_only": False}}
        resolved = self._resolve(tmp_path, monkeypatch, cfg,
                                 ["--ollama-fallback-only"])
        assert resolved["serving"]["ollama_fallback_only"] is True

    def test_flag_absent_leaves_config_value(self, tmp_path, monkeypatch):
        cfg = {"serving": {"host": "http://vllm1:8001/v1",
                           "ollama_fallback_only": False}}
        resolved = self._resolve(tmp_path, monkeypatch, cfg, [])
        assert resolved["serving"]["ollama_fallback_only"] is False


# ── onit learn ──────────────────────────────────────────────────────────────

class TestLearnCommand:
    """Read-only, and deliberately usable when the rest of the stack is down:
    "is anything being recorded" must be answerable without a model."""

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONIT_LEARN_PATH", str(tmp_path / "learned"))
        # No config file to find, so the command falls back to env + defaults.
        monkeypatch.setattr("src.cli._find_default_config",
                            lambda: str(tmp_path / "missing.yaml"))
        from src.learn import record_task
        record_task(session_id="s1", turn=1, task="what scholarships exist?",
                    response="two", metrics={"turns": [
                        {"n": 1, "tool_runs": [
                            {"name": "read_file", "ok": False, "ms": 40},
                            {"name": "local_search", "ok": True, "ms": 800}]}],
                        "turn_count": 2, "completion_tokens": 50})
        return tmp_path

    def _run(self, argv):
        from src.cli import main
        with patch.object(sys, "argv", ["onit"] + argv):
            try:
                main()
            except SystemExit as e:
                if e.code:
                    raise

    def test_status_reports_the_worst_tool(self, store, capsys):
        self._run(["learn"])
        out = capsys.readouterr().out
        assert "1 task(s) across 1 session(s)" in out
        assert "read_file" in out and "100%" in out
        assert "observe" in out

    def test_json_summary_is_machine_readable(self, store, capsys):
        self._run(["learn", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["tasks"] == 1
        assert data["tools"]["read_file"]["errors"] == 1

    def test_one_session_can_be_dumped(self, store, capsys):
        self._run(["learn", "--session", "s1"])
        records = json.loads(capsys.readouterr().out)
        assert records[0]["task"] == "what scholarships exist?"

    def test_an_unknown_session_exits_nonzero(self, store, capsys):
        with pytest.raises(SystemExit) as excinfo:
            self._run(["learn", "--session", "nope"])
        assert excinfo.value.code == 1

    def test_recording_off_says_how_to_turn_it_on(self, store, capsys, monkeypatch):
        monkeypatch.setenv("ONIT_LEARN", "off")
        self._run(["learn"])
        out = capsys.readouterr().out
        assert "Recording is off" in out


# ── session selection on startup ────────────────────────────────────────────

class TestSessionSelection:
    """Terminal chat picks up where it left off: a bare `onit` resumes the
    last session, and --restart-session is the way to start over."""

    @pytest.fixture
    def sessions(self, tmp_path, monkeypatch):
        """A config whose session_path holds two sessions, 'old' then 'new'."""
        import yaml
        from src import setup as setup_mod
        from src.sessions import register_session

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        for sid, mtime in (("old-session-id", 1000), ("new-session-id", 2000)):
            path = sessions_dir / f"{sid}.jsonl"
            path.write_text("")
            register_session(sid, str(sessions_dir))
            os.utime(path, (mtime, mtime))

        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({
            "session_path": str(sessions_dir),
            "serving": {"host": "http://localhost:8000/v1"},
        }))
        monkeypatch.setattr("src.cli._find_default_config", lambda: str(cfg))
        monkeypatch.setattr(setup_mod, "CONFIG_PATH", str(tmp_path / "no-setup.yaml"))
        # Keep the real keychain out of credential resolution.
        monkeypatch.setattr(setup_mod, "get_secret", lambda key: None)
        return sessions_dir

    def _run(self, argv):
        """Run main() up to dispatch and return the resolved config."""
        from src.cli import main
        captured = {}
        with patch.object(sys, "argv", ["onit"] + argv), \
                patch("src.cli._setup_servers"), \
                patch("src.cli._dispatch_mode",
                      side_effect=lambda cfg: captured.update(cfg)):
            main()
        return captured

    def test_bare_invocation_resumes_the_last_session(self, sessions, capsys):
        config = self._run([])
        assert config["resume_session_id"] == "new-session-id"
        assert "Resuming last session" in capsys.readouterr().out

    def test_restart_session_starts_fresh(self, sessions):
        assert "resume_session_id" not in self._run(["--restart-session"])

    def test_new_session_is_an_alias_for_restart(self, sessions):
        assert "resume_session_id" not in self._run(["--new-session"])

    def test_explicit_resume_still_wins(self, sessions):
        config = self._run(["--resume", "old-session-id"])
        assert config["resume_session_id"] == "old-session-id"

    def test_resume_subcommand_still_works(self, sessions):
        config = self._run(["resume", "old-session-id"])
        assert config["resume_session_id"] == "old-session-id"

    def test_first_run_has_nothing_to_resume(self, sessions):
        for f in sessions.iterdir():
            f.unlink()
        assert "resume_session_id" not in self._run([])

    def test_stale_index_entry_is_ignored(self, sessions):
        # The index outlives a deleted JSONL file; resuming it would crash
        # later in _setup_session, so startup must fall back to a new session.
        for f in sessions.glob("*.jsonl"):
            f.unlink()
        assert "resume_session_id" not in self._run([])

    def test_unknown_session_still_errors(self, sessions):
        with pytest.raises(SystemExit) as excinfo:
            self._run(["--resume", "does-not-exist"])
        assert excinfo.value.code == 1

    def test_server_modes_do_not_auto_resume(self, sessions, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        assert "resume_session_id" not in self._run(["serve", "web"])
        assert "resume_session_id" not in self._run(["serve", "a2a"])
        assert "resume_session_id" not in self._run(["serve", "gateway", "telegram"])
        assert "resume_session_id" not in self._run(["serve", "loop", "check things"])
