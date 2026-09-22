"""Tests for src/cli.py — _find_default_config, MCP server readiness/startup."""

import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.cli import (
    _find_default_config,
    _ensure_mcp_servers,
)


# ── _find_default_config ────────────────────────────────────────────────────

# main() writes the approval variables with a plain os.environ assignment,
# which monkeypatch cannot see and so cannot undo — and monkeypatch.delenv on
# a variable that was not set records nothing to restore either. Every test in
# this file that reaches main() therefore leaks them, whatever it was actually
# testing: a leaked ONIT_AUTO_APPROVE turns a later module's refusal into an
# approval. Snapshot and put back by hand, for the whole file.
_APPROVAL_ENV_VARS = ("ONIT_APPROVAL_CHANNEL", "ONIT_AUTO_APPROVE",
                      "ONIT_ASK_APPROVAL", "ONIT_WEB_UI", "ONIT_UNRESTRICTED")

# The tool-availability switches are set the same way, in the same place, when
# credential resolution finds nothing: a config resolved without a key leaves
# ONIT_DISABLE_WEB_SEARCH / ONIT_DISABLE_WEATHER in the process environment,
# and every stdio server spawned afterwards inherits them — its web tools are
# never registered. The net-profile test in test_multiuser_isolation.py is
# exactly such a spawn, so a leak here fails a file this one has never heard of.
_TOOL_DISABLE_VARS = ("ONIT_DISABLE_WEB_SEARCH", "ONIT_DISABLE_WEATHER")


@pytest.fixture(autouse=True)
def _restore_approval_env():
    saved = {var: os.environ.get(var)
             for var in _APPROVAL_ENV_VARS + _TOOL_DISABLE_VARS}
    yield
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


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


# ── _ensure_mcp_servers ──────────────────────────────────────────────────────

class TestEnsureMcpServers:
    @staticmethod
    def _config():
        """One named server. _ensure_mcp_servers adds the defaults alongside."""
        return {"mcp": {"servers": [
            {"name": "A", "url": "http://127.0.0.1:9000/mcp",
             "external": True, "enabled": True},
        ]}}

    @staticmethod
    def _by_name(config, name):
        return next(s for s in config["mcp"]["servers"] if s["name"] == name)

    def test_an_external_server_is_left_alone(self):
        """A server that lives elsewhere is neither registered nor re-addressed:
        the client connects to its URL as written."""
        config = self._config()
        _ensure_mcp_servers(config)
        assert self._by_name(config, "A")["url"] == "http://127.0.0.1:9000/mcp"

    def test_the_defaults_are_added_beside_a_named_server(self):
        config = self._config()
        _ensure_mcp_servers(config)
        names = {s["name"] for s in config["mcp"]["servers"]}
        assert names == {"A", "PromptsMCPServer", "ToolsLocalMCPServer",
                         "ToolsNetMCPServer"}

    def test_stdio_server_gets_a_spec(self):
        from type.tools import _STDIO_SPECS
        config = {"mcp": {"servers": [
            {"name": "ToolsLocalMCPServer", "transport": "stdio",
             "module": "tasks.tools", "profile": "local", "enabled": True},
        ]}}
        _ensure_mcp_servers(config)

        server = self._by_name(config, "ToolsLocalMCPServer")
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
        _ensure_mcp_servers(config)
        assert self._by_name(config, "Broken")["enabled"] is False


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


# ── --host serving overrides ──────────────────────────────────────

class TestOpenRouterKeyResolution:
    """The legacy OPENROUTER_API_KEY must not be injected in front of a key
    stored for this endpoint's own URL — that authenticates as the wrong
    account with no sign anything went wrong but a 401."""

    def _resolve(self, tmp_path, monkeypatch, host, secrets):
        import yaml
        from src import setup as setup_mod
        from src.cli import _build_parser, _parse_and_resolve_config
        monkeypatch.setattr(setup_mod, "get_secret", lambda k: secrets.get(k))
        monkeypatch.setattr(setup_mod, "CONFIG_PATH",
                            str(tmp_path / "no-setup.yaml"))
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({"serving": {"host": host}}))
        args = _build_parser().parse_args(["--config", str(path)])
        return _parse_and_resolve_config(args)["serving"]

    def test_an_endpoint_key_is_left_for_chat_to_find(self, tmp_path,
                                                      monkeypatch):
        from src import setup as setup_mod
        host = "https://openrouter.ai/api/v1"
        serving = self._resolve(tmp_path, monkeypatch, host, {
            setup_mod.endpoint_secret_name(host): "sk-endpoint",
            "host_key": "sk-legacy",
        })
        assert "host_key" not in serving

    def test_the_legacy_key_is_still_injected_without_one(self, tmp_path,
                                                          monkeypatch):
        host = "https://openrouter.ai/api/v1"
        serving = self._resolve(tmp_path, monkeypatch, host,
                                {"host_key": "sk-legacy"})
        assert serving["host_key"] == "sk-legacy"

    def test_an_endpoint_key_satisfies_the_missing_key_check(self, tmp_path,
                                                             monkeypatch):
        """Without this the run exits before it can use the key it has."""
        from src import setup as setup_mod
        host = "https://openrouter.ai/api/v1"
        serving = self._resolve(tmp_path, monkeypatch, host, {
            setup_mod.endpoint_secret_name(host): "sk-endpoint"})
        assert serving["host"] == host


class TestOpenAIKeyResolution:
    """The OpenAI legacy key follows the same bargain as OpenRouter's: the
    endpoint's own key wins, the legacy one is injected only in its place,
    and a missing key is a startup error rather than a mid-task 401."""

    def _resolve(self, tmp_path, monkeypatch, host, secrets):
        import yaml
        from src import setup as setup_mod
        from src.cli import _build_parser, _parse_and_resolve_config
        monkeypatch.setattr(setup_mod, "get_secret", lambda k: secrets.get(k))
        monkeypatch.setattr(setup_mod, "CONFIG_PATH",
                            str(tmp_path / "no-setup.yaml"))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({"serving": {"host": host}}))
        args = _build_parser().parse_args(["--config", str(path)])
        return _parse_and_resolve_config(args)["serving"]

    def test_an_endpoint_key_is_left_for_chat_to_find(self, tmp_path,
                                                      monkeypatch):
        from src import setup as setup_mod
        host = "https://api.openai.com/v1"
        serving = self._resolve(tmp_path, monkeypatch, host, {
            setup_mod.endpoint_secret_name(host): "sk-endpoint",
            "openai_api_key": "sk-legacy",
        })
        assert "host_key" not in serving

    def test_the_legacy_key_is_still_injected_without_one(self, tmp_path,
                                                          monkeypatch):
        host = "https://api.openai.com/v1"
        serving = self._resolve(tmp_path, monkeypatch, host,
                                {"openai_api_key": "sk-legacy"})
        assert serving["host_key"] == "sk-legacy"

    def test_an_endpoint_key_satisfies_the_missing_key_check(self, tmp_path,
                                                             monkeypatch):
        from src import setup as setup_mod
        host = "https://api.openai.com/v1"
        serving = self._resolve(tmp_path, monkeypatch, host, {
            setup_mod.endpoint_secret_name(host): "sk-endpoint"})
        assert serving["host"] == host

    def test_a_vllm_host_never_receives_the_openai_key(self, tmp_path,
                                                       monkeypatch):
        """The generalized injection must not shadow VLLM_API_KEY."""
        host = "http://localhost:8000/v1"
        serving = self._resolve(tmp_path, monkeypatch, host,
                                {"openai_api_key": "sk-openai"})
        assert "host_key" not in serving


class TestHostOverrides:
    """An explicit --host must yield a single endpoint: any second host left
    over from config or env would keep the load balancer routing to the old
    server (Ollama endpoints are fallback-only, so a healthy vLLM second
    host would shadow an explicitly requested Ollama --host entirely)."""

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

    def test_no_host_flag_keeps_config_host2(self, tmp_path, monkeypatch):
        resolved = self._resolve(tmp_path, monkeypatch, self._CFG, [])
        serving = resolved["serving"]
        assert serving["host"] == "http://vllm1:8001/v1"
        assert serving["host2"] == "http://vllm2:8000/v1"

    def test_flag_absent_leaves_config_value(self, tmp_path, monkeypatch):
        cfg = {"serving": {"host": "http://vllm1:8001/v1",
                           "ollama_fallback_only": False}}
        resolved = self._resolve(tmp_path, monkeypatch, cfg, [])
        assert resolved["serving"]["ollama_fallback_only"] is False


# ── --max-tokens / --max-context-tokens ─────────────────────────────────────

class TestTokenLimitOverrides:
    """The two token budgets are reachable from the CLI, in the units people
    actually quote them in ("1M"), and they land on the serving keys chat()
    reads rather than anywhere else."""

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

    _CFG = {"serving": {"host": "http://localhost:8000/v1"}}

    @pytest.mark.parametrize("text,expected", [
        ("32768", 32768),
        ("128k", 128_000),
        ("1M", 1_000_000),
        ("1m", 1_000_000),
        ("1.5M", 1_500_000),
        ("1_048_576", 1_048_576),
        ("1,048,576", 1_048_576),
    ])
    def test_suffixes_and_separators(self, text, expected):
        from src.cli import _token_count
        assert _token_count(text) == expected

    @pytest.mark.parametrize("text", ["lots", "", "-5", "1G", "1.2.3", "0"])
    def test_rejects_nonsense(self, text):
        import argparse
        from src.cli import _token_count
        with pytest.raises(argparse.ArgumentTypeError):
            _token_count(text)

    def test_max_tokens_overrides_config(self, tmp_path, monkeypatch):
        cfg = {"serving": {"host": "http://localhost:8000/v1", "max_tokens": 32768}}
        resolved = self._resolve(tmp_path, monkeypatch, cfg, ["--max-tokens", "1M"])
        assert resolved["serving"]["max_tokens"] == 1_000_000

    def test_max_context_tokens_overrides_config(self, tmp_path, monkeypatch):
        cfg = {"serving": {"host": "http://localhost:8000/v1",
                           "max_context_tokens": 131072}}
        resolved = self._resolve(tmp_path, monkeypatch, cfg,
                                 ["--max-context-tokens", "1M"])
        assert resolved["serving"]["max_context_tokens"] == 1_000_000

    def test_max_context_tokens_sizes_ollama_num_ctx(self, tmp_path, monkeypatch):
        # Ollama auto-sizes num_ctx to a ceiling far below 1M, so the flag has
        # to reach that knob or the window the user asked for never exists.
        resolved = self._resolve(tmp_path, monkeypatch, self._CFG,
                                 ["--max-context-tokens", "1M"])
        assert resolved["serving"]["num_ctx"] == 1_000_000

    def test_configured_num_ctx_wins(self, tmp_path, monkeypatch):
        cfg = {"serving": {"host": "http://localhost:11434/v1", "num_ctx": 65536}}
        resolved = self._resolve(tmp_path, monkeypatch, cfg,
                                 ["--max-context-tokens", "1M"])
        assert resolved["serving"]["num_ctx"] == 65536
        assert resolved["serving"]["max_context_tokens"] == 1_000_000

    def test_flags_absent_leave_config_alone(self, tmp_path, monkeypatch):
        cfg = {"serving": {"host": "http://localhost:8000/v1", "max_tokens": 4096}}
        resolved = self._resolve(tmp_path, monkeypatch, cfg, [])
        assert resolved["serving"]["max_tokens"] == 4096
        assert "max_context_tokens" not in resolved["serving"]
        assert "num_ctx" not in resolved["serving"]


# ── web-only serving defaults ───────────────────────────────────────────────

class TestWebServingDefaults:
    """One config serves both UIs, and they want different latency trades.

    A browser turn is a single iteration with no tool loop behind it, so a
    reasoning pass is the entire wait; a terminal run is long enough for the
    deliberation to pay for itself. The split has to beat a plain
    ``serving.think`` rather than fill in for it — anyone who turned thinking
    on at all would otherwise never see it.
    """

    _CFG = {"serving": {"host": "http://vllm:8000/v1",
                        "think": True,
                        "temperature": 0.6,
                        "top_p": 0.95}}

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

    def test_web_run_drops_thinking_and_moves_sampling(self, tmp_path, monkeypatch):
        serving = self._resolve(tmp_path, monkeypatch, self._CFG,
                                ["serve", "web"])["serving"]
        assert serving["think"] is False
        assert serving["temperature"] == 0.7
        assert serving["top_p"] == 0.8

    def test_terminal_run_keeps_the_configured_values(self, tmp_path, monkeypatch):
        serving = self._resolve(tmp_path, monkeypatch, self._CFG, [])["serving"]
        assert serving["think"] is True
        assert serving["temperature"] == 0.6
        assert serving["top_p"] == 0.95

    def test_other_serve_modes_are_not_web(self, tmp_path, monkeypatch):
        """The loop mode is not someone watching a composer."""
        serving = self._resolve(tmp_path, monkeypatch, self._CFG,
                                ["serve", "loop", "check things"])["serving"]
        assert serving["think"] is True
        assert serving["temperature"] == 0.6

    def test_config_web_flag_gets_the_same_treatment(self, tmp_path, monkeypatch):
        """``web: true`` in the file launches the same UI as the subcommand."""
        cfg = dict(self._CFG, web=True)
        serving = self._resolve(tmp_path, monkeypatch, cfg, [])["serving"]
        assert serving["think"] is False
        assert serving["temperature"] == 0.7

    def test_serving_web_block_overrides_the_default(self, tmp_path, monkeypatch):
        cfg = {"serving": dict(self._CFG["serving"], web={"top_p": 0.9})}
        serving = self._resolve(tmp_path, monkeypatch, cfg,
                                ["serve", "web"])["serving"]
        assert serving["top_p"] == 0.9
        # Keys it does not name still take the built-in web value.
        assert serving["think"] is False
        assert serving["temperature"] == 0.7

    def test_reasoning_back_on_takes_its_sampling_with_it(self, tmp_path, monkeypatch):
        """The default is one trade, not three settings.

        A browser session told to think wants the thinking-mode sampling, and
        that is what ``serving:`` already holds — the instruct numbers are
        half of a default that no longer applies.
        """
        cfg = {"serving": dict(self._CFG["serving"], web={"think": True})}
        serving = self._resolve(tmp_path, monkeypatch, cfg,
                                ["serve", "web"])["serving"]
        assert serving["think"] is True
        assert serving["temperature"] == 0.6
        assert serving["top_p"] == 0.95

    def test_sampling_named_under_web_survives_thinking(self, tmp_path, monkeypatch):
        """Stated explicitly, it is a preference rather than half a default."""
        cfg = {"serving": dict(self._CFG["serving"],
                               web={"think": True, "temperature": 1.0})}
        serving = self._resolve(tmp_path, monkeypatch, cfg,
                                ["serve", "web"])["serving"]
        assert serving["think"] is True
        assert serving["temperature"] == 1.0
        assert serving["top_p"] == 0.95

    def test_serving_web_block_never_reaches_chat(self, tmp_path, monkeypatch):
        """It is a config layer, not a serving parameter."""
        cfg = {"serving": dict(self._CFG["serving"], web={"think": True})}
        for cli_args in ([], ["serve", "web"]):
            serving = self._resolve(tmp_path, monkeypatch, cfg, cli_args)["serving"]
            assert "web" not in serving

    def test_explicit_think_flag_wins_over_the_web_default(self, tmp_path, monkeypatch):
        """Asking for reasoning by hand is not a default to be overridden —
        and it brings the configured sampling back with it."""
        serving = self._resolve(tmp_path, monkeypatch, self._CFG,
                                ["--think", "serve", "web"])["serving"]
        assert serving["think"] is True
        assert serving["temperature"] == 0.6
        assert serving["top_p"] == 0.95

    def test_web_defaults_apply_to_a_bare_serving_block(self, tmp_path, monkeypatch):
        """Nothing to override — the web values are still the ones that apply."""
        cfg = {"serving": {"host": "http://vllm:8000/v1"}}
        serving = self._resolve(tmp_path, monkeypatch, cfg,
                                ["serve", "web"])["serving"]
        assert serving["think"] is False
        assert serving["top_p"] == 0.8


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
        assert "resume_session_id" not in self._run(["serve", "web"])
        assert "resume_session_id" not in self._run(["serve", "loop", "check things"])


class TestApprovalChannelWiring:
    """Which runs can ask a person, and which answer for themselves.

    The tool servers inherit these variables at spawn time and cannot work any
    of it out for themselves — "is anyone watching" is a fact about how OnIt
    was started. Getting it wrong in the permissive direction would mint
    approval tickets in a run with nobody to answer them.
    """

    @pytest.fixture
    def sessions(self, tmp_path, monkeypatch):
        """A minimal config main() can resolve without touching the host."""
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
        return cfg

    # These tests read what main() set, so they start from none of the
    # variables set; _restore_approval_env puts the outer values back after.
    @pytest.fixture(autouse=True)
    def _clean_env(self):
        for var in _APPROVAL_ENV_VARS:
            os.environ.pop(var, None)

    def _run(self, argv):
        """Run main() up to dispatch, so only the env wiring is exercised."""
        from src.cli import main
        with patch.object(sys, "argv", ["onit"] + argv), \
                patch("src.cli._setup_servers"), \
                patch("src.cli._dispatch_mode"):
            main()

    def test_your_own_run_answers_for_itself(self, sessions):
        """A terminal run belongs to the person who started it: they already
        hold the shell, so it approves rather than stopping to ask them."""
        self._run([])
        assert os.environ.get("ONIT_APPROVAL_CHANNEL") == "1"
        assert os.environ.get("ONIT_AUTO_APPROVE") == "1"

    def test_no_auto_puts_the_question_back(self, sessions):
        self._run(["--no-auto"])
        assert os.environ.get("ONIT_APPROVAL_CHANNEL") == "1"
        assert "ONIT_AUTO_APPROVE" not in os.environ

    def test_ask_is_the_same_switch(self, sessions):
        self._run(["--ask"])
        assert "ONIT_AUTO_APPROVE" not in os.environ

    def test_the_web_ui_can_ask(self, sessions):
        self._run(["serve", "web", "--no-login"])
        assert os.environ.get("ONIT_APPROVAL_CHANNEL") == "1"

    def test_the_web_ui_still_asks_by_default(self, sessions):
        """A deployment serves people who are not the operator, and on a
        shared host the prompt is part of what keeps sessions apart. Only an
        explicit --auto answers for them."""
        self._run(["serve", "web", "--no-login"])
        assert "ONIT_AUTO_APPROVE" not in os.environ

    def test_a_server_run_cannot(self, sessions):
        """A web deployment keeps the approval channel (a person at the
        browser can answer) but must not auto-approve without --auto."""
        self._run(["serve", "web"])
        assert os.environ.get("ONIT_APPROVAL_CHANNEL") == "1"
        assert "ONIT_AUTO_APPROVE" not in os.environ

    def test_a_loop_run_answers_for_itself(self, sessions):
        """A --loop is the operator's own unattended run, so it approves
        rather than refusing every gated command outright."""
        self._run(["serve", "loop", "check things"])
        assert os.environ.get("ONIT_APPROVAL_CHANNEL") == "1"
        assert os.environ.get("ONIT_AUTO_APPROVE") == "1"

    def test_a_loop_run_with_no_auto_cannot_ask(self, sessions):
        """Nobody is watching a loop, so turning the switch off leaves no
        channel at all rather than a prompt no one will answer."""
        self._run(["--no-auto", "serve", "loop", "check things"])
        assert "ONIT_APPROVAL_CHANNEL" not in os.environ
        assert "ONIT_AUTO_APPROVE" not in os.environ

    def test_auto_answers_for_itself(self, sessions):
        self._run(["--auto"])
        assert os.environ.get("ONIT_AUTO_APPROVE") == "1"

    def test_auto_answers_for_a_deployment_when_asked_to(self, sessions):
        self._run(["--auto", "serve", "web", "--no-login"])
        assert os.environ.get("ONIT_AUTO_APPROVE") == "1"

    def test_auto_gives_an_unattended_run_a_channel(self, sessions):
        """--auto is a channel of its own — the answer comes from the flag,
        which is exactly what a run with nobody watching needs."""
        self._run(["--auto", "serve", "loop", "check things"])
        assert os.environ.get("ONIT_APPROVAL_CHANNEL") == "1"
        assert os.environ.get("ONIT_AUTO_APPROVE") == "1"

    def test_auto_says_so(self, sessions, capsys):
        self._run(["--auto", "serve", "web", "--no-login"])
        assert "--auto approves command prompts automatically" in capsys.readouterr().err

    def test_the_default_says_so_more_quietly(self, sessions, capsys):
        """It prints on every ordinary run, so it says which way the switch
        is set and how to move it, and leaves the warning to the flag."""
        self._run([])
        err = capsys.readouterr().err
        assert "Command approvals: approving automatically" in err
        assert "--no-auto" in err

    def test_auto_warns_when_it_has_nothing_to_approve(self, sessions,
                                                       monkeypatch, capsys):
        monkeypatch.setenv("ONIT_ASK_APPROVAL", "0")
        self._run(["--auto"])
        assert "Nothing is left for --auto to approve" in capsys.readouterr().err
