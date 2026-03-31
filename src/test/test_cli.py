"""Tests for src/cli.py — _find_default_config, _send_task, _download_files, _upload_file."""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.cli import (
    _find_default_config, _send_task, _download_files, _upload_file,
    _is_port_open, _mcp_servers_ready, _ensure_mcp_servers,
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
        result = _find_default_config()
        # Should return the default path string even if file doesn't exist
        assert "default.yaml" in result


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


# ── _is_port_open ──────────────────────────────────────────────────────────

class TestIsPortOpen:
    def test_returns_true_for_open_port(self):
        with patch("src.cli.socket.create_connection") as mock_conn:
            mock_sock = MagicMock()
            mock_conn.return_value = mock_sock
            assert _is_port_open("127.0.0.1", 8080) is True
            mock_sock.close.assert_called_once()

    def test_returns_false_for_closed_port(self):
        with patch("src.cli.socket.create_connection", side_effect=ConnectionRefusedError):
            assert _is_port_open("127.0.0.1", 9999) is False


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
    def test_skips_start_when_already_running(self):
        config = {
            "mcp": {
                "servers": [
                    {"name": "A", "url": "http://127.0.0.1:18200/sse", "enabled": True},
                ]
            }
        }
        with patch("src.cli._is_port_open", return_value=True), \
             patch("src.cli.threading.Thread") as mock_thread:
            _ensure_mcp_servers(config)
            mock_thread.assert_not_called()

    def test_starts_servers_when_not_running(self):
        config = {
            "mcp": {
                "servers": [
                    {"name": "A", "url": "http://127.0.0.1:18200/sse", "enabled": True},
                ]
            }
        }
        mock_thread_instance = MagicMock()
        with patch("src.cli._is_port_open", return_value=False), \
             patch("src.cli.threading.Thread", return_value=mock_thread_instance), \
             patch("src.cli._mcp_servers_ready", return_value=True):
            _ensure_mcp_servers(config)
            mock_thread_instance.start.assert_called_once()
