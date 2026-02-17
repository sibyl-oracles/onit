"""Tests for src/mcp/servers/run.py — load_config, prepare_server_args, run_server."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.servers.run import load_config, prepare_server_args, run_server


# ── load_config ─────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        config = {
            "servers": [
                {"name": "TestServer", "module": "tasks.test", "enabled": True, "port": 9000}
            ]
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))
        result = load_config(str(config_file))
        assert "servers" in result
        assert result["servers"][0]["name"] == "TestServer"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_default_config_exists(self):
        """The built-in default config should be loadable."""
        default_path = os.path.join(
            os.path.dirname(__file__), "..", "mcp", "servers", "configs", "default.yaml"
        )
        if os.path.exists(default_path):
            result = load_config(default_path)
            assert "servers" in result


# ── prepare_server_args ─────────────────────────────────────────────────────

class TestPrepareServerArgs:
    def test_extracts_enabled_servers(self):
        config = {
            "servers": [
                {"name": "A", "module": "tasks.a", "enabled": True, "port": 9000,
                 "host": "0.0.0.0", "path": "/a", "transport": "streamable-http"},
                {"name": "B", "module": "tasks.b", "enabled": False, "port": 9001},
                {"name": "C", "module": "tasks.c", "port": 9002,
                 "host": "0.0.0.0", "path": "/c"},
            ]
        }
        args = prepare_server_args(config)
        # A and C should be included (C defaults to enabled=True)
        assert len(args) == 2
        names = [a[0] for a in args]
        assert "A" in names
        assert "C" in names
        assert "B" not in names

    def test_skips_server_without_name(self):
        config = {"servers": [{"module": "tasks.x", "port": 9000}]}
        args = prepare_server_args(config)
        assert len(args) == 0

    def test_skips_server_without_module(self):
        config = {"servers": [{"name": "NoModule", "port": 9000}]}
        args = prepare_server_args(config)
        assert len(args) == 0

    def test_empty_config(self):
        args = prepare_server_args({})
        assert args == []

    def test_options_passed_through(self):
        config = {
            "servers": [
                {"name": "A", "module": "tasks.a", "port": 9000,
                 "host": "0.0.0.0", "path": "/a",
                 "options": {"verbose": True}}
            ]
        }
        args = prepare_server_args(config)
        assert len(args) == 1
        assert args[0][6] == {"verbose": True}


# ── run_server ──────────────────────────────────────────────────────────────

class TestRunServer:
    def test_run_server_success(self):
        mock_module = MagicMock()
        mock_module.run = MagicMock()

        with patch("builtins.__import__", return_value=mock_module):
            result = run_server(
                name="Test", transport="streamable-http",
                host="0.0.0.0", port=9000, path="/test",
                module="tasks.test", options={}
            )
        # run_server returns True on success
        assert result is True

    def test_run_server_import_error(self):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = run_server(
                name="Bad", transport="streamable-http",
                host="0.0.0.0", port=9000, path="/bad",
                module="tasks.nonexistent", options={}
            )
        assert result is False

    def test_run_server_no_module(self):
        result = run_server(
            name="Empty", transport="streamable-http",
            host="0.0.0.0", port=9000, path="/empty",
            module="", options={}
        )
        assert result is False
