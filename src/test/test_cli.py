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

    def test_the_spawned_servers_are_told_the_autonomy_level(self):
        """The skills server decides whether to serve anything by the
        autonomy level, and a subprocess has no config of its own -- so the
        level has to travel in the environment.  Without this, the server
        falls back to `observe` and serves nothing however the operator
        configured the run."""
        from type.tools import _STDIO_SPECS
        config = {"learn": {"autonomy": "extend"},
                  "mcp": {"servers": [
                      {"name": "SkillsMCPServer", "transport": "stdio",
                       "module": "src.mcp.servers.tasks.skills",
                       "enabled": True},
                  ]}}
        _ensure_mcp_servers(config)
        server = self._by_name(config, "SkillsMCPServer")
        spec = _STDIO_SPECS[server["url"]]
        assert spec["env"]["ONIT_LEARN_AUTONOMY"] == "3"

    def test_the_level_is_capped_before_it_is_passed_on(self):
        """A config asking for `evolve` is running at `extend`; the server
        must be told what the run *is*, not what it asked for."""
        from type.tools import _STDIO_SPECS
        config = {"learn": {"autonomy": "evolve"},
                  "mcp": {"servers": [
                      {"name": "SkillsMCPServer", "transport": "stdio",
                       "module": "src.mcp.servers.tasks.skills",
                       "enabled": True},
                  ]}}
        _ensure_mcp_servers(config)
        server = self._by_name(config, "SkillsMCPServer")
        assert _STDIO_SPECS[server["url"]]["env"]["ONIT_LEARN_AUTONOMY"] == "3"

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

    def test_the_resume_message_names_the_session(self, sessions, capsys):
        """Which session was resumed is the one thing the person needs to
        confirm before typing into it -- a bare `onit` continues the last one,
        and the id alone does not say which conversation that was."""
        from src.sessions import tag_session
        tag_session("new-session-id", "thesis ch3", str(sessions))
        self._run([])
        out = capsys.readouterr().out
        assert "thesis ch3" in out
        # The id stays too: it is what --resume-session accepts.
        assert "new-sess" in out

    def test_an_unnamed_resumed_session_is_named_by_its_short_id(self, sessions, capsys):
        self._run([])
        out = capsys.readouterr().out
        assert "new-sess" in out

    def test_the_explicit_resume_message_names_it_too(self, sessions, capsys):
        from src.sessions import tag_session
        tag_session("old-session-id", "old work", str(sessions))
        self._run(["--resume", "old-session-id"])
        out = capsys.readouterr().out
        assert "Resuming session: old work (old-sess" in out

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

    # ── the flag family ──────────────────────────────────────────────────────
    # Every flag about a session now ends in -session and they are declared
    # together, so --help reads as a set. The old spellings stay, hidden,
    # because a flag in muscle memory cannot simply be removed.

    def test_the_new_spellings_work(self, sessions):
        assert "resume_session_id" not in self._run(["--new-session"])
        config = self._run(["--resume-session", "old-session-id"])
        assert config["resume_session_id"] == "old-session-id"

    def test_naming_a_session_starts_a_new_one(self, sessions, capsys):
        """A name given to --new-session is the name the session gets.

        The flag carries the name itself, so naming and creating cannot
        disagree. Before, a separate --name was silently dropped whenever the
        run resumed instead -- which is the default -- so 'onit --name foo'
        continued the last conversation under a name the user could not then
        find. That is the bug this pins.
        """
        config = self._run(["--new-session", "thesis chapter 3"])
        assert "resume_session_id" not in config
        assert config["session_name"] == "thesis chapter 3"
        out = capsys.readouterr().out
        assert "thesis chapter 3" in out
        assert "--resume-session" in out

    def test_a_bare_new_session_asks_onit_to_name_it(self, sessions, capsys):
        """No name is a request for OnIt to pick one from the first message,
        not for a session called ''.

        The distinction is what makes the flag usable: an empty string in the
        config would reach register_session as a name that normalizes to
        nothing, and the session would stay nameless even after a real first
        message named it.
        """
        config = self._run(["--new-session"])
        assert "resume_session_id" not in config
        assert "session_name" not in config
        assert "Naming this session" not in capsys.readouterr().out

    def test_a_taken_name_is_announced_with_the_suffix_it_gets(self, sessions, capsys):
        """A name already in use is resolved with a suffix rather than
        refused, so the printed "return to it" line must carry the suffix.
        Announcing the name that was asked for would send the user to a
        different session."""
        import pathlib
        from src.sessions import register_session, resolve_session_choice
        register_session("older", str(pathlib.Path(sessions)), tag="chapter 3")
        config = self._run(["--new-session", "chapter 3"])
        assert config["session_name"] == "chapter 3-2"
        out = capsys.readouterr().out
        assert "chapter 3-2" in out
        assert "already in use" in out
        # the promise is kept: the announced name is stored verbatim when the
        # session is created, so it is not suffixed a second time
        stored = register_session("newer", str(pathlib.Path(sessions)),
                                  tag=config["session_name"])
        assert stored == "chapter 3-2"
        assert resolve_session_choice("chapter 3-2", str(pathlib.Path(sessions)))

    def test_the_name_flag_spellings_still_work(self, sessions):
        """--name and --name-session were the way to say this before the name
        moved onto --new-session. They are now the same flag, so an old script
        keeps working."""
        for argv in (["--name", "foo"], ["--name-session", "foo"]):
            config = self._run(argv)
            assert "resume_session_id" not in config
            assert config["session_name"] == "foo"

    def test_the_restart_session_spelling_still_starts_fresh(self, sessions):
        config = self._run(["--restart-session"])
        assert "resume_session_id" not in config

    def test_naming_and_resuming_together_is_refused(self, sessions, capsys):
        """One flag creates a session and the other continues an existing
        one; both at once has no meaning, and silently picking one would
        lose the name or the conversation."""
        with pytest.raises(SystemExit) as excinfo:
            self._run(["--new-session", "foo", "--resume-session", "old-session-id"])
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "cannot be combined" in err
        assert "onit sessions --rename" in err

    def test_help_offers_one_flag_per_job(self):
        """--new-session now carries the name, so --name-session is gone from
        --help. It still parses, hidden: a flag in muscle memory cannot simply
        be removed."""
        from src.cli import _build_parser
        text = _build_parser().format_help()
        assert "--new-session" in text
        assert "--resume-session" in text
        assert "--name-session" not in text
        assert "--restart-session" not in text
        assert "TAG_OR_ID" not in text

    def test_the_session_flags_are_one_group(self):
        """--help groups them under 'session'; that is what makes the family
        discoverable rather than three flags scattered among twenty."""
        from src.cli import _build_parser
        groups = {g.title: g for g in _build_parser()._action_groups}
        assert "session" in groups
        flags = {o for a in groups["session"]._group_actions for o in a.option_strings}
        assert {"--new-session", "--resume-session"} <= flags
        # and nowhere else, so there is one place to look
        others = {o for t, g in groups.items() if t != "session"
                  for a in g._group_actions for o in a.option_strings}
        assert not ({"--new-session", "--resume-session"} & others)

    def test_a_subcommand_after_the_bare_flag_is_not_eaten(self):
        """`onit --new-session serve web` means "fresh session, then serve".

        An optional-value flag reads the next word as its value, so without
        the recovery step argparse would name a session "serve" and then exit
        2 on the stray "web"."""
        from src.cli import _build_parser, _recover_subcommand
        parser = _build_parser()
        args = parser.parse_args(
            _recover_subcommand(["--new-session", "serve", "web"], parser))
        assert args.command == "serve"
        assert args.serve_mode == "web"
        assert args.new_session == ""

    def test_a_name_that_happens_to_be_a_subcommand_word_is_still_a_name(self):
        """The recovery only fires when the flag was given bare. A name is
        written `--new-session=serve` or in a position the subparser cannot
        claim, so a session may still be called "sessions"."""
        from src.cli import _build_parser, _recover_subcommand
        parser = _build_parser()
        args = parser.parse_args(
            _recover_subcommand(["--new-session=sessions"], parser))
        assert args.command is None
        assert args.new_session == "sessions"

    def test_session_limit_still_parses_but_no_longer_moves_a_row(self, sessions):
        """The listing shows every session now, so nothing truncates it and a
        row number needs no configuring.

        The flag keeps parsing because it was documented and a script may pass
        it -- but it must not change which session '3' means. It never really
        did: the list is sorted before it is sliced, so a limit only ever
        shortened the list from the bottom.
        """
        import pathlib
        from src.sessions import register_session, update_session
        store = pathlib.Path(sessions)
        for i in range(6):
            register_session(f"row-{i}", str(store))
            update_session(f"row-{i}", task=f"task {i}", sessions_dir=str(store))
            (store / f"row-{i}.jsonl").write_text("")

        plain = self._run(["--resume-session", "3"])
        with_flag = self._run(["--resume-session", "3", "--session-limit", "3"])
        assert plain["resume_session_id"] == with_flag["resume_session_id"]

    def test_the_listing_reads_the_configured_store(self, sessions, capsys):
        """'onit sessions' hardcoded ~/.onit/sessions while --resume-session
        honoured session_path. On a config that moved the store, the listing
        and the resume flag saw different sessions -- and --delete would have
        removed one from a store the agent never reads."""
        self._run(["sessions"])
        out = capsys.readouterr().out
        # the fixture's store holds old-session-id and new-session-id
        assert "old-session-id" in out or "(unnamed)" in out
        assert "SMOKE-OK" not in out      # that is the real ~/.onit store

    def test_the_listing_prints_every_row(self, sessions, capsys):
        """20 was the default limit and the listing printed 20 rows, so a
        21st session had no # to address it with -- it could not be resumed
        or deleted by row number at all."""
        import pathlib
        from src.sessions import register_session, update_session
        store = pathlib.Path(sessions)
        for i in range(25):
            register_session(f"extra-{i}", str(store))
            update_session(f"extra-{i}", task=f"task {i}", sessions_dir=str(store))

        self._run(["sessions"])
        out = capsys.readouterr().out
        # 27 sessions exist, so the last row number is 27
        assert "27 " in out
        assert "Showing" not in out    # nothing was held back


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


# ── session verbs ────────────────────────────────────────────────────────────
#
# `onit sessions` prints a # column and used to leave the user to copy a UUID
# out of the terminal to act on it.  The verbs below take that number, and
# `rename` exists as a verb at all instead of hiding inside --tag.

class TestSessionVerbs:

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        from src import sessions
        d = str(tmp_path / "sessions")
        os.makedirs(d, exist_ok=True)
        monkeypatch.setenv("HOME", str(tmp_path))
        for sid, task in (("aaaa-1111", "write the quarterly report"),
                          ("bbbb-2222", "debug the cuda kernel")):
            sessions.register_session(sid, d)
            sessions.update_session(sid, task=task, sessions_dir=d)
        return d

    def test_rename_takes_a_row_number(self, store, monkeypatch, capsys):
        from src import cli
        # the verb resolves ~/.onit/sessions, so point HOME at the fixture
        import pathlib
        home = pathlib.Path(store).parent
        (home / ".onit").mkdir(exist_ok=True)
        (home / ".onit" / "sessions").symlink_to(store)
        monkeypatch.setattr(sys, "argv",
                            ["onit", "rename", "1", "Report Work"])
        cli.main()
        out = capsys.readouterr().out
        assert "Report Work" in out

    def test_rename_reports_an_unknown_row(self, store, monkeypatch, capsys):
        from src import cli
        import pathlib
        home = pathlib.Path(store).parent
        (home / ".onit").mkdir(exist_ok=True)
        (home / ".onit" / "sessions").symlink_to(store)
        monkeypatch.setattr(sys, "argv", ["onit", "rename", "99", "x"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert "no session matches" in capsys.readouterr().err

    def test_rename_refuses_an_empty_name(self, store, monkeypatch, capsys):
        from src import cli
        import pathlib
        home = pathlib.Path(store).parent
        (home / ".onit").mkdir(exist_ok=True)
        (home / ".onit" / "sessions").symlink_to(store)
        monkeypatch.setattr(sys, "argv", ["onit", "rename", "1", "   "])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert "cannot be empty" in capsys.readouterr().err

    def test_sessions_rename_renames_a_row(self, store, monkeypatch, capsys):
        """Renaming belongs to the listing: you read a row, you fix its name.
        It used to be a separate verb reached by leaving the listing, and
        the only spelling inside the listing was --tag."""
        from src import cli, sessions
        import pathlib
        home = pathlib.Path(store).parent
        (home / ".onit").mkdir(exist_ok=True)
        (home / ".onit" / "sessions").symlink_to(store)
        monkeypatch.setattr(sys, "argv",
                            ["onit", "sessions", "--rename", "1", "Report Work"])
        cli.main()
        out = capsys.readouterr().out
        assert "Report Work" in out
        assert "--resume-session" in out
        assert sessions.find_session_by_tag("Report Work", store)

    def test_sessions_rename_refuses_an_empty_name(self, store, monkeypatch, capsys):
        from src import cli
        import pathlib
        home = pathlib.Path(store).parent
        (home / ".onit").mkdir(exist_ok=True)
        (home / ".onit" / "sessions").symlink_to(store)
        monkeypatch.setattr(sys, "argv",
                            ["onit", "sessions", "--rename", "1", "  "])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert "cannot be empty" in capsys.readouterr().err

    def test_sessions_rename_and_the_verb_agree(self, store, monkeypatch, capsys):
        """Two spellings, one implementation: what the verb accepts the flag
        accepts, down to the row number and the error text."""
        from src import cli
        import pathlib
        home = pathlib.Path(store).parent
        (home / ".onit").mkdir(exist_ok=True)
        (home / ".onit" / "sessions").symlink_to(store)
        monkeypatch.setattr(sys, "argv",
                            ["onit", "sessions", "--rename", "99", "x"])
        with pytest.raises(SystemExit):
            cli.main()
        via_flag = capsys.readouterr().err
        monkeypatch.setattr(sys, "argv", ["onit", "rename", "99", "x"])
        with pytest.raises(SystemExit):
            cli.main()
        assert capsys.readouterr().err == via_flag

    def test_the_old_tag_spelling_still_renames(self, store, monkeypatch, capsys):
        """--tag was the only spelling for a while. It still parses, so an
        old script keeps working, but it is not what --help offers."""
        from src import cli, sessions
        import pathlib
        home = pathlib.Path(store).parent
        (home / ".onit").mkdir(exist_ok=True)
        (home / ".onit" / "sessions").symlink_to(store)
        monkeypatch.setattr(sys, "argv",
                            ["onit", "sessions", "--tag", "1", "Old Spelling"])
        cli.main()
        assert "Old Spelling" in capsys.readouterr().out
        assert sessions.find_session_by_tag("Old Spelling", store)

    def test_sessions_help_names_rename_and_hides_tag(self):
        from src.cli import _build_parser
        # parse the subcommand's own help, so this reads what a user sees
        parser = _build_parser()
        sub = [a for a in parser._subparsers._group_actions[0].choices.values()
               if a.prog == "onit sessions"][0]
        text = sub.format_help()
        assert "--rename" in text
        assert "--tag" not in text
        assert "--rebuild-index" in text and "--delete-all" in text
        assert "--delete" in text and "--yes" in text
        # The listing shows everything, so there is nothing left to limit or
        # to filter -- and a flag that does nothing is worse than no flag.
        assert "--include-empty" not in text
        assert "--limit" not in text

    def test_sessions_shows_every_row(self, store, monkeypatch, capsys):
        """An empty session used to be hidden, with a flag to reveal it. That
        made the '#' column address a list the person could not see in full,
        and left no way to delete a row that was never used -- which is the
        one row a person most wants to delete."""
        from src import cli, sessions
        import pathlib
        sessions.register_session("cccc-3333", store)   # opened, never used
        home = pathlib.Path(store).parent
        (home / ".onit").mkdir(exist_ok=True)
        (home / ".onit" / "sessions").symlink_to(store)
        monkeypatch.setattr(sys, "argv", ["onit", "sessions"])
        cli.main()
        out = capsys.readouterr().out
        assert "quarterly" in out
        assert "(unnamed)" in out

    def test_the_listing_names_the_verbs(self, store, monkeypatch, capsys):
        from src import cli
        import pathlib
        home = pathlib.Path(store).parent
        (home / ".onit").mkdir(exist_ok=True)
        (home / ".onit" / "sessions").symlink_to(store)
        monkeypatch.setattr(sys, "argv", ["onit", "sessions"])
        cli.main()
        out = capsys.readouterr().out
        assert "onit sessions --rename" in out
        assert "onit sessions --delete" in out
        assert "onit --resume-session" in out
        assert "onit --new-session <name>" in out


class TestDeleteSessions:
    """`onit sessions --delete` takes the same identifiers as --rename.

    The '#' column addresses every verb on the listing, not just rename, and
    a session that was opened and never used is exactly the row someone wants
    to delete -- so it has to be visible and numbered.
    """

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        """Two used sessions in a store the CLI will find via HOME."""
        from src import sessions
        d = str(tmp_path / "sessions")
        os.makedirs(d, exist_ok=True)
        monkeypatch.setenv("HOME", str(tmp_path))
        for sid, task in (("aaaa-1111", "write the quarterly report"),
                          ("bbbb-2222", "debug the cuda kernel")):
            sessions.register_session(sid, d)
            sessions.update_session(sid, task=task, sessions_dir=d)
            open(os.path.join(d, sid + ".jsonl"), "w").write("{}\n")
        return d

    def _cli(self, store, monkeypatch, argv, answer="yes"):
        """Run the sessions command against *store*; return (exit, out, err)."""
        from src import cli
        import pathlib
        home = pathlib.Path(store).parent
        (home / ".onit").mkdir(exist_ok=True)
        link = home / ".onit" / "sessions"
        if not link.exists():
            link.symlink_to(store)
        monkeypatch.setattr(sys, "argv", ["onit"] + argv)
        monkeypatch.setattr("builtins.input", lambda prompt="": answer)
        code = 0
        try:
            cli.main()
        except SystemExit as exc:
            code = exc.code
        return code

    def test_delete_takes_a_row_number(self, store, monkeypatch, capsys):
        from src.sessions import list_sessions
        before = [s["session_id"] for s in list_sessions(store, limit=None)]
        target = before[0]
        code = self._cli(store, monkeypatch,
                         ["sessions", "--delete", "1", "--yes"])
        assert code == 0
        after = [s["session_id"] for s in list_sessions(store, limit=None)]
        assert target not in after
        assert len(after) == len(before) - 1
        assert "Deleted" in capsys.readouterr().out

    def test_delete_takes_a_name(self, store, monkeypatch):
        from src.sessions import find_session_by_tag, list_sessions
        sid = find_session_by_tag("write-quarterly-report", store)
        assert sid
        code = self._cli(store, monkeypatch,
                         ["sessions", "--delete", "write-quarterly-report", "--yes"])
        assert code == 0
        assert sid not in [s["session_id"] for s in list_sessions(store, limit=None)]

    def test_delete_takes_a_uuid_prefix(self, store, monkeypatch):
        from src.sessions import list_sessions
        code = self._cli(store, monkeypatch,
                         ["sessions", "--delete", "aaaa", "--yes"])
        assert code == 0
        assert "aaaa-1111" not in [s["session_id"]
                                   for s in list_sessions(store, limit=None)]

    def test_delete_takes_several_at_once(self, store, monkeypatch):
        from src.sessions import list_sessions
        code = self._cli(store, monkeypatch,
                         ["sessions", "--delete", "1", "2", "--yes"])
        assert code == 0
        assert len(list_sessions(store, limit=None)) == 0

    def test_delete_asks_before_deleting(self, store, monkeypatch, capsys):
        from src.sessions import list_sessions
        code = self._cli(store, monkeypatch,
                         ["sessions", "--delete", "1"], answer="no")
        assert code == 0
        out = capsys.readouterr().out
        assert "About to delete" in out and "Cancelled" in out
        assert len(list_sessions(store, limit=None)) == 2   # nothing went

    def test_a_typo_in_a_batch_deletes_nothing(self, store, monkeypatch, capsys):
        """Half a batch is the worst outcome: the person believes the whole
        thing failed and the sessions they meant to keep are gone."""
        from src.sessions import list_sessions
        before = [s["session_id"] for s in list_sessions(store, limit=None)]
        code = self._cli(store, monkeypatch,
                         ["sessions", "--delete", "aaaa", "nope-nothing", "--yes"])
        assert code == 1
        assert [s["session_id"] for s in list_sessions(store, limit=None)] == before
        assert "nope-nothing" in capsys.readouterr().err

    def test_deleting_the_same_row_twice_deletes_once(self, store, monkeypatch):
        from src.sessions import list_sessions
        code = self._cli(store, monkeypatch,
                         ["sessions", "--delete", "1", "1", "--yes"])
        assert code == 0
        assert len(list_sessions(store, limit=None)) == 1

    def test_an_unused_session_can_be_deleted(self, store, monkeypatch):
        """The whole point of listing them: the accidental session is the one
        that needs deleting, and it used to have no row to delete."""
        from src import sessions
        sessions.register_session("cccc-3333", store)
        code = self._cli(store, monkeypatch,
                         ["sessions", "--delete", "cccc-3333", "--yes"])
        assert code == 0
        assert "cccc-3333" not in [s["session_id"]
                                   for s in sessions.list_sessions(store, limit=None)]

    def test_delete_all_takes_yes_too(self, store, monkeypatch):
        """--yes belongs to the deletion family, not to --delete alone: a
        script that prunes everything has the same reason to skip the prompt."""
        from src.sessions import list_sessions
        def boom(prompt=""):
            raise AssertionError("prompted despite --yes")
        monkeypatch.setattr("builtins.input", boom)
        code = self._cli(store, monkeypatch, ["sessions", "--delete-all", "--yes"])
        assert code == 0
        assert list_sessions(store, limit=None) == []

    def test_delete_removes_the_history_file_too(self, store, monkeypatch):
        import pathlib
        from src.sessions import register_session, update_session
        register_session("dddd-4444", store)
        update_session("dddd-4444", task="a real conversation", sessions_dir=store)
        (pathlib.Path(store) / "dddd-4444.jsonl").write_text("{}\n")
        code = self._cli(store, monkeypatch,
                         ["sessions", "--delete", "dddd-4444", "--yes"])
        assert code == 0
        assert not (pathlib.Path(store) / "dddd-4444.jsonl").exists()



class TestDeleteByName:
    """`--delete <name>` must delete the session that has that name.

    Every case here was reported as "does not work", and each failed for a
    different reason -- a name cut short in the listing, a name the resolver
    could not reach at all, a name spread over several arguments.
    """

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        d = str(tmp_path / "sessions")
        os.makedirs(d, exist_ok=True)
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".onit").mkdir(exist_ok=True)
        (tmp_path / ".onit" / "sessions").symlink_to(d)
        return d

    def _add(self, store, sid, tag):
        from src import sessions
        sessions.register_session(sid, store, tag=tag)

    def _cli(self, monkeypatch, argv, answer="yes"):
        from src import cli
        monkeypatch.setattr(sys, "argv", ["onit"] + argv)
        monkeypatch.setattr("builtins.input", lambda prompt="": answer)
        code = 0
        try:
            cli.main()
        except SystemExit as exc:
            code = exc.code
        return code

    def _ids(self, store):
        from src.sessions import list_sessions
        return [s["session_id"] for s in list_sessions(store, limit=None)]

    def test_the_listing_shows_the_whole_name(self, store, monkeypatch, capsys):
        """A name cut to 32 characters is not the name.

        Seven real sessions named marikina-river-water-level-latest and -2
        through -6 all rendered as the same 32 characters, so the listing
        showed one name for six sessions and typing it back picked one of
        them -- the newest -- without a word.
        """
        from src.sessions import format_sessions_table, list_sessions
        long_name = "marikina-river-water-level-latest-6"
        self._add(store, "aaaa-1111", long_name)
        text = format_sessions_table(list_sessions(store, limit=None))
        assert long_name in text
        assert long_name[:32] + " " in text or long_name in text

    def test_a_name_that_fits_several_is_refused_not_guessed(self, store,
                                                             monkeypatch,
                                                             capsys):
        """Choosing by recency is what deleted the wrong session."""
        for i, sid in enumerate(["aaaa-1111", "bbbb-2222", "cccc-3333"]):
            self._add(store, sid, f"marikina-river-water-level-latest-{i}")
        code = self._cli(monkeypatch,
                         ["sessions", "--delete",
                          "marikina-river-water-level-latest", "--yes"])
        err = capsys.readouterr().err
        assert code == 1
        assert "matches 3 sessions" in err
        # every candidate is named, so the person can pick one
        for sid in ("aaaa-1111", "bbbb-2222", "cccc-3333"):
            assert sid[:8] in err
        # and nothing was deleted
        assert len(self._ids(store)) == 3

    def test_a_full_name_deletes_exactly_that_session(self, store, monkeypatch,
                                                      capsys):
        """The fix for the refusal: the name that fits one row still works."""
        self._add(store, "aaaa-1111", "marikina-river-water-level-latest")
        self._add(store, "bbbb-2222", "marikina-river-water-level-latest-2")
        code = self._cli(monkeypatch,
                         ["sessions", "--delete",
                          "marikina-river-water-level-latest-2", "--yes"])
        assert code == 0
        assert "bbbb-222" in capsys.readouterr().out
        assert self._ids(store) == ["aaaa-1111"]

    @pytest.mark.parametrize("tag", ["2026", "3", "last"])
    def test_a_session_can_be_named_like_a_row_number(self, store, monkeypatch,
                                                      capsys, tag):
        """A name is a name, even when it looks like a number.

        These three were unreachable: '2026' was a row past the end, '3' was
        row three, and 'last' was the most recent session -- each reporting
        or deleting something other than the session that had that name.
        """
        self._add(store, "aaaa-1111", tag)
        self._add(store, "bbbb-2222", "something-else")
        code = self._cli(monkeypatch, ["sessions", "--delete", tag, "--yes"])
        assert code == 0
        assert "aaaa-111" in capsys.readouterr().out
        assert self._ids(store) == ["bbbb-2222"]

    def test_a_row_number_still_works_when_no_name_matches(self, store,
                                                            monkeypatch):
        """The fix must not cost the # column: a bare number that is not a
        name is still the row it was printed next to."""
        from src.sessions import list_sessions
        self._add(store, "aaaa-1111", "first")
        self._add(store, "bbbb-2222", "second")
        target = list_sessions(store, limit=None)[0]["session_id"]
        code = self._cli(monkeypatch, ["sessions", "--delete", "1", "--yes"])
        assert code == 0
        assert target not in self._ids(store)

    def test_an_unquoted_name_with_spaces_is_one_name(self, store, monkeypatch,
                                                      capsys):
        """--delete takes a list, so a name with spaces arrived as several ids
        and each was reported as not found."""
        self._add(store, "aaaa-1111", "my thesis chapter 3")
        code = self._cli(monkeypatch,
                         ["sessions", "--delete", "my", "thesis", "chapter",
                          "3", "--yes"])
        assert code == 0
        assert "aaaa-111" in capsys.readouterr().out
        assert self._ids(store) == []

    def test_the_exact_spelling_beats_a_longer_name(self, store, monkeypatch,
                                                    capsys):
        """An exact name wins over a name that merely starts with it, and
        'Notes' wins over 'notes' when an old index holds both."""
        self._add(store, "aaaa-1111", "chapter")
        self._add(store, "bbbb-2222", "chapter-2")
        code = self._cli(monkeypatch, ["sessions", "--delete", "chapter",
                                       "--yes"])
        assert code == 0
        assert "aaaa-111" in capsys.readouterr().out
        assert self._ids(store) == ["bbbb-2222"]

    def test_a_case_differing_spelling_is_found(self, store, monkeypatch,
                                                capsys):
        """Typing a name in the wrong case is a typo, not a different
        session -- and it must not fall through to a longer name."""
        self._add(store, "aaaa-1111", "My Thesis Chapter 3")
        self._add(store, "bbbb-2222", "My Thesis Chapter 3-2")
        code = self._cli(monkeypatch,
                         ["sessions", "--delete", "MY THESIS CHAPTER 3",
                          "--yes"])
        assert code == 0
        assert "aaaa-111" in capsys.readouterr().out
        assert self._ids(store) == ["bbbb-2222"]

    def test_no_answer_deletes_nothing_and_says_so(self, store, monkeypatch,
                                                   capsys):
        """Piped input that runs out used to end in an EOFError traceback."""
        def no_input(prompt=""):
            raise EOFError()
        self._add(store, "aaaa-1111", "keep-me")
        monkeypatch.setattr(sys, "argv",
                            ["onit", "sessions", "--delete", "keep-me"])
        monkeypatch.setattr("builtins.input", no_input)
        from src import cli
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert self._ids(store) == ["aaaa-1111"]
        assert "--yes" in capsys.readouterr().err

    def test_rename_refuses_a_name_that_fits_several(self, store, monkeypatch,
                                                     capsys):
        """Renaming the wrong one of six is the same mistake as deleting it."""
        for i, sid in enumerate(["aaaa-1111", "bbbb-2222"]):
            self._add(store, sid, f"report-{i}")
        code = self._cli(monkeypatch,
                         ["sessions", "--rename", "report", "renamed"])
        assert code == 1
        assert "matches 2 sessions" in capsys.readouterr().err


class TestDeleteARangeOfRows:
    """`--delete 20-32` removes the rows the listing numbers 20 to 32.

    Pruning a block of rows one `#` at a time is the chore a range exists to
    remove, and it is also where a mistake is cheapest to make: the rows that
    matter here are that a range covers *exactly* the rows printed between its
    ends, that it is inclusive at both ends, and that nothing which is not a
    range -- a name with dashes, a reversed pair, a bare dash -- is silently
    read as one.
    """

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        d = str(tmp_path / "sessions")
        os.makedirs(d, exist_ok=True)
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".onit").mkdir(exist_ok=True)
        (tmp_path / ".onit" / "sessions").symlink_to(d)
        return d

    def _add(self, store, sid, tag):
        from src import sessions
        sessions.register_session(sid, store, tag=tag)
        open(os.path.join(store, sid + ".jsonl"), "w").write("{}\n")

    def _fill(self, store, n):
        """*n* named sessions, newest last so row 1 is the newest."""
        for i in range(1, n + 1):
            self._add(store, f"{i:04d}-aaaa", f"s{i}")

    def _cli(self, monkeypatch, argv, answer="yes"):
        from src import cli
        monkeypatch.setattr(sys, "argv", ["onit"] + argv)
        monkeypatch.setattr("builtins.input", lambda prompt="": answer)
        code = 0
        try:
            cli.main()
        except SystemExit as exc:
            code = exc.code
        return code

    def _tags(self, store):
        from src.sessions import list_sessions
        return [s["tag"] for s in list_sessions(store, limit=None)]

    def test_a_range_deletes_the_rows_between_its_ends(self, store, monkeypatch):
        """Inclusive at both ends: 20-32 is thirteen rows, not eleven."""
        self._fill(store, 40)
        rows = self._tags(store)
        doomed = rows[19:32]
        code = self._cli(monkeypatch, ["sessions", "--delete", "20-32", "--yes"])
        assert code == 0
        left = self._tags(store)
        assert len(left) == len(rows) - 13
        for tag in doomed:
            assert tag not in left
        # the neighbours on either side of the block are untouched
        assert rows[18] in left and rows[32] in left

    def test_an_open_end_runs_to_the_last_row(self, store, monkeypatch):
        self._fill(store, 10)
        rows = self._tags(store)
        code = self._cli(monkeypatch, ["sessions", "--delete", "8-", "--yes"])
        assert code == 0
        assert self._tags(store) == rows[:7]

    def test_an_open_start_runs_from_the_first_row(self, store, monkeypatch):
        self._fill(store, 10)
        rows = self._tags(store)
        code = self._cli(monkeypatch, ["sessions", "--delete", "-3", "--yes"])
        assert code == 0
        assert self._tags(store) == rows[3:]

    def test_an_end_past_the_last_row_is_clamped(self, store, monkeypatch,
                                                capsys):
        """-32 on a store of five means five. Refusing it would be pedantry
        about a range whose meaning is not in doubt."""
        self._fill(store, 5)
        code = self._cli(monkeypatch, ["sessions", "--delete", "-32", "--yes"])
        assert code == 0
        assert self._tags(store) == []

    def test_a_range_and_a_row_together(self, store, monkeypatch):
        """The flag takes a list, so a range is one entry in it, not the whole
        argument."""
        self._fill(store, 10)
        rows = self._tags(store)
        code = self._cli(monkeypatch, ["sessions", "--delete", "2-3", "9",
                                       "--yes"])
        assert code == 0
        left = self._tags(store)
        assert rows[1] not in left and rows[2] not in left
        assert rows[8] not in left
        assert len(left) == 7

    def test_a_reversed_range_deletes_nothing(self, store, monkeypatch, capsys):
        """Swapping the ends would delete rows nobody named. Say so instead."""
        self._fill(store, 10)
        before = self._tags(store)
        code = self._cli(monkeypatch, ["sessions", "--delete", "8-3", "--yes"])
        assert code == 1
        assert self._tags(store) == before
        assert "8-3" in capsys.readouterr().err

    def test_a_range_past_the_end_deletes_nothing(self, store, monkeypatch,
                                                  capsys):
        self._fill(store, 5)
        before = self._tags(store)
        code = self._cli(monkeypatch, ["sessions", "--delete", "20-32", "--yes"])
        assert code == 1
        assert self._tags(store) == before
        err = capsys.readouterr().err
        assert "no rows" in err and "20-32" in err

    def test_the_prompt_lists_every_row_of_the_range(self, store, monkeypatch,
                                                     capsys):
        """A block of thirteen is read before it goes, not summarised."""
        self._fill(store, 10)
        rows = self._tags(store)
        code = self._cli(monkeypatch, ["sessions", "--delete", "4-6"],
                         answer="no")
        out = capsys.readouterr().out
        assert code == 0
        for tag in rows[3:6]:
            assert tag in out
        # one line per row of the block, each naming the row it is, so nothing
        # in it goes unread and no line is a repeat of the one above
        assert out.count("turn(s)") == 3
        for n in (4, 5, 6):
            assert f"4-6 #{n}" in out
        assert "Cancelled" in out
        assert self._tags(store) == rows

    def test_a_name_with_dashes_is_still_a_name(self, store, monkeypatch):
        """Every range check has to survive the names it looks like."""
        self._fill(store, 4)
        self._add(store, "zzzz-9999", "marikina-river-water-level-latest")
        code = self._cli(monkeypatch,
                         ["sessions", "--delete",
                          "marikina-river-water-level-latest", "--yes"])
        assert code == 0
        assert "marikina-river-water-level-latest" not in self._tags(store)
        assert len(self._tags(store)) == 4

    def test_a_session_named_like_a_range_wins_over_the_range(self, store,
                                                              monkeypatch):
        """A chosen name must stay reachable, as a name beats a row number."""
        self._fill(store, 6)
        self._add(store, "zzzz-9999", "2-4")
        code = self._cli(monkeypatch, ["sessions", "--delete", "2-4", "--yes"])
        assert code == 0
        left = self._tags(store)
        assert "2-4" not in left
        assert len(left) == 6          # the rows the range would have taken stay

    def test_the_listing_advertises_the_range(self, store):
        from src.sessions import format_sessions_table, list_sessions
        self._fill(store, 2)
        text = format_sessions_table(list_sessions(store, limit=None))
        assert "<a>-<b>" in text


class TestSubcommandsDoNotPayForTheAgent:
    """A subcommand that never runs the agent must not import it.

    ``src.onit`` pulls in ``fastmcp.client``, ``openai`` and
    ``src.model.serving.chat`` -- 0.57s of the 0.70s ``onit sessions``
    took to start.  Measured, not assumed: the import graph is what the
    test reads, so a future import added back at module scope fails here
    rather than quietly costing every command half a second.
    """

    def test_importing_the_cli_does_not_import_the_agent(self):
        import subprocess
        code = ("import sys, src.cli; "
                "print(int('src.onit' in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(
                                 os.path.dirname(os.path.abspath(__file__)))))
        assert out.stdout.strip() == "0", (
            "importing src.cli dragged in src.onit: %s" % out.stderr[-400:])

    def test_the_name_is_still_reachable_and_still_patchable(self, monkeypatch):
        """Lazy must not mean unreachable -- ``patch("src.cli.OnIt")`` works."""
        from src import cli
        from unittest.mock import patch
        assert cli.OnIt.__name__ == "OnIt"
        with patch("src.cli.OnIt") as fake:
            fake.return_value = "an agent"
            assert cli.OnIt(config={}) == "an agent"

    def test_an_unknown_attribute_is_still_an_attribute_error(self):
        from src import cli
        with pytest.raises(AttributeError):
            cli.no_such_attribute


class TestTheRealDispatchBuildsTheAgent:
    """The two call sites, exercised for real rather than patched away.

    ``_dispatch_mode`` read a bare global ``OnIt``.  A module ``__getattr__``
    never serves a global lookup, so ``onit --resume`` raised ``NameError``
    on every run while the suite stayed green.

    Why it stayed green is the reason these tests are shaped as they are:
    ``patch("src.cli.OnIt")`` *creates* the global, so the bare lookup finds
    it and the bug disappears.  Patching it is what hid the defect, so these
    tests patch ``src.onit.OnIt`` -- the module ``__getattr__`` imports from
    -- and clear ``src.cli``'s namespace first, leaving no global to find.
    A first draft that patched ``src.cli.OnIt`` passed against the broken
    commit, which is how this shape was found.
    """

    @pytest.fixture
    def uncached(self, monkeypatch):
        """Leave ``src.cli`` with no ``OnIt`` global, and restore after."""
        from src import cli
        had = "OnIt" in vars(cli)
        saved = vars(cli).get("OnIt")
        monkeypatch.delitem(vars(cli), "OnIt", raising=False)
        yield
        if had:
            vars(cli)["OnIt"] = saved
        else:
            vars(cli).pop("OnIt", None)

    def test_dispatch_mode_builds_and_runs_the_agent(self, uncached, monkeypatch):
        from src import cli
        built, ran = [], []

        class FakeAgent:
            def __init__(self, config=None):
                built.append(config)

            async def run(self):
                ran.append(True)

        monkeypatch.setattr("src.onit.OnIt", FakeAgent)
        cli._dispatch_mode({"a": 1})
        assert built == [{"a": 1}]
        assert ran, "the agent was built but never run"

    def test_doctor_builds_the_agent_through_the_same_lookup(self, uncached,
                                                             monkeypatch):
        """The second site: a stubbed agent, or doctor returns 1 silently."""
        import argparse
        from src import cli
        built = []

        class FakeAgent:
            session_id = None

            def __init__(self, config=None):
                built.append(config)

        async def no_checks(agent, deep=False, on_start=None):
            return []

        monkeypatch.setattr("src.onit.OnIt", FakeAgent)
        args = argparse.Namespace(json=False, deep=False, keep_session=False,
                                  endpoints=False)
        with patch("src.ui.doctor.run_checks", no_checks), \
                patch("src.ui.doctor.render_report", lambda r, deep=False: ""):
            code = cli._run_doctor(args, {"session_path": "/tmp/none"})
        assert code == 0
        assert built == [{"session_path": "/tmp/none"}], (
            "doctor never reached the agent -- the lookup is broken again")

    def test_the_lookup_does_not_import_the_agent_until_called(self):
        """The fix must not undo the laziness it was added to keep."""
        import subprocess
        code = ("import sys, src.cli; "
                "print(int('src.onit' in sys.modules)); "
                "src.cli._agent_class(); "
                "print(int('src.onit' in sys.modules))")
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
        assert out.stdout.split() == ["0", "1"], out.stderr[-400:]


class TestVersionFlag:
    """``onit --version`` answers without loading anything that can fail.

    The flag exists because ``src.__version__`` did, and nothing in the CLI
    read it: the number was reachable from Python but not from the command
    line.  It is answered by argparse's ``version`` action during parsing,
    so these tests also pin that it stays that way -- a version flag that
    needs the config, the MCP servers or the agent is useless precisely when
    someone asks for it.
    """

    def test_it_prints_the_package_version(self):
        import subprocess
        from src import __version__
        out = subprocess.run([sys.executable, "-m", "src.cli", "--version"],
                             capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(
                                 os.path.dirname(os.path.abspath(__file__)))))
        assert out.returncode == 0
        assert out.stdout.strip() == f"onit {__version__}"

    def test_it_matches_the_packaging_metadata(self):
        """Two copies of the number must not drift."""
        import re
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        with open(os.path.join(root, "pyproject.toml")) as fh:
            declared = re.search(r'^version = "([^"]+)"', fh.read(),
                                 re.M).group(1)
        from src import __version__
        assert declared == __version__

    def test_it_does_not_import_the_agent_or_any_server(self):
        """The answer must not depend on the stack the flag is asked about.

        ``src.onit`` drags in ``fastmcp.client``, ``openai`` and model
        serving; importing any of them to print a version would make the
        flag fail on exactly the broken install where it is worth having.
        """
        import subprocess
        code = ("import sys, contextlib, src.cli\n"
                "with contextlib.suppress(SystemExit):\n"
                "    src.cli._build_parser().parse_args(['--version'])\n"
                "print(int('src.onit' in sys.modules),"
                " int('fastmcp' in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(
                                 os.path.dirname(os.path.abspath(__file__)))))
        # The version action prints to stdout first; the answer is the last
        # line, which the script above writes after parsing.
        assert out.stdout.strip().splitlines()[-1] == "0 0", out.stderr[-400:]

    def test_it_is_listed_in_help(self):
        import subprocess
        out = subprocess.run([sys.executable, "-m", "src.cli", "--help"],
                             capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(
                                 os.path.dirname(os.path.abspath(__file__)))))
        assert "--version" in out.stdout
        assert "Print the OnIt version and exit." in out.stdout


# ── deleting the sessions nobody ever used ──────────────────────────────────
#
# A test that built a real agent without naming a session store registered
# itself in the developer's own ~/.onit/sessions: unnamed, zero turns, newest
# entry, so also the one a bare `onit` resumed.  One laptop collected 360 of
# them.  They interleave with the real sessions by date, so a row range cannot
# name them -- `--delete 20-32` would take real conversations with it.

class TestDeleteEmptySessions:

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        from src import sessions
        d = str(tmp_path / "sessions")
        os.makedirs(d, exist_ok=True)
        # The verb resolves ~/.onit/sessions, so point HOME at the fixture.
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".onit").mkdir(exist_ok=True)
        (tmp_path / ".onit" / "sessions").symlink_to(d)
        for sid, task in (("real-1", "write the migration"),
                          ("real-2", "debug the parser")):
            sessions.register_session(sid, d)
            sessions.update_session(sid, task=task, sessions_dir=d)
        for sid in ("junk-1", "junk-2"):
            sessions.register_session(sid, d)
        return d

    def test_it_deletes_the_empty_ones_and_keeps_the_rest(self, store, monkeypatch,
                                                         capsys):
        from src import cli, sessions
        monkeypatch.setattr(sys, "argv",
                            ["onit", "sessions", "--delete-empty", "--yes"])
        cli.main()
        out = capsys.readouterr().out
        assert "Deleted 2 empty session(s)." in out
        assert [r["session_id"] for r in sessions.list_sessions(store)] == [
            "real-2", "real-1"]

    def test_it_says_how_many_before_asking(self, store, monkeypatch, capsys):
        from src import cli
        monkeypatch.setattr(sys, "argv", ["onit", "sessions", "--delete-empty"])
        monkeypatch.setattr("builtins.input", lambda _prompt: "no")
        cli.main()
        out = capsys.readouterr().out
        assert "2 of 4 sessions were opened and never used" in out
        assert "Cancelled." in out

    def test_a_refusal_deletes_nothing(self, store, monkeypatch):
        from src import cli, sessions
        monkeypatch.setattr(sys, "argv", ["onit", "sessions", "--delete-empty"])
        monkeypatch.setattr("builtins.input", lambda _prompt: "no")
        cli.main()
        assert len(sessions.list_sessions(store)) == 4

    def test_a_store_with_nothing_empty_says_so(self, tmp_path, monkeypatch,
                                                capsys):
        from src import cli, sessions
        d = str(tmp_path / "sessions")
        os.makedirs(d, exist_ok=True)
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".onit").mkdir(exist_ok=True)
        (tmp_path / ".onit" / "sessions").symlink_to(d)
        sessions.register_session("real", d)
        sessions.update_session("real", task="do the thing", sessions_dir=d)
        monkeypatch.setattr(sys, "argv", ["onit", "sessions", "--delete-empty"])
        cli.main()
        assert "No empty sessions" in capsys.readouterr().out
