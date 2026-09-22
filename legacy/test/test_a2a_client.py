"""Tests for the A2A client (legacy/a2a_client.py) — the former ``onit ask``
client, moved beside the A2A server it speaks to. Run with: pytest legacy/test -v
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from legacy.a2a_client import (
    download_files, send_task, upload_file,
)


# ── download_files ─────────────────────────────────────────────────────────

class TestDownloadFiles:
    def test_downloads_referenced_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        mock_resp = MagicMock()
        mock_resp.content = b"file contents"
        mock_resp.raise_for_status = MagicMock()

        with patch("legacy.a2a_client.requests.get", return_value=mock_resp):
            result = download_files(
                "Here is the file: /uploads/report.pdf",
                "http://localhost:9001",
            )

        assert "Downloaded files:" in result
        assert (tmp_path / "report.pdf").exists()

    def test_handles_download_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        with patch("legacy.a2a_client.requests.get", side_effect=Exception("timeout")):
            result = download_files(
                "File: /uploads/missing.txt",
                "http://localhost:9001",
            )

        assert "Failed to download" in result

    def test_no_files_returns_unchanged(self):
        result = download_files("No files here", "http://localhost:9001")
        assert result == "No files here"


# ── upload_file ────────────────────────────────────────────────────────────

class TestUploadFile:
    def test_uploads_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("legacy.a2a_client.requests.post", return_value=mock_resp):
            filename = upload_file("http://localhost:9001", str(test_file))

        assert filename == "test.txt"

    def test_upload_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            upload_file("http://localhost:9001", "/nonexistent/file.txt")


# ── send_task ──────────────────────────────────────────────────────────────

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

        with patch("legacy.a2a_client.requests.post", return_value=mock_resp):
            result = send_task("http://localhost:9001", "What is 6*7?")

        assert "42" in result

    def test_handles_error_response(self):
        response_data = {"error": {"code": -32600, "message": "Invalid request"}}

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("legacy.a2a_client.requests.post", return_value=mock_resp):
            result = send_task("http://localhost:9001", "bad request")

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

        with patch("legacy.a2a_client.requests.post", return_value=mock_resp) as mock_post:
            result = send_task("http://localhost:9001", "analyze this", file=str(test_file))

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

        with patch("legacy.a2a_client.requests.post", return_value=mock_resp):
            result = send_task("http://localhost:9001", "question")

        assert "Direct answer" in result

    def test_fallback_to_json_dump(self):
        """When no text part found, falls back to JSON dump."""
        response_data = {"result": {"something": "unexpected"}}

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("legacy.a2a_client.requests.post", return_value=mock_resp):
            result = send_task("http://localhost:9001", "question")

        assert "unexpected" in result

