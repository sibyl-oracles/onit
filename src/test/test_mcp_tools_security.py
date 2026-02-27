"""Security tests for MCP tool functions — shell injection prevention."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import src.mcp.servers.tasks.os.bash.mcp_server as bash_mod
from src.mcp.servers.tasks.os.bash.mcp_server import (
    search_directory,
    find_files,
    transform_text,
)


@pytest.fixture(autouse=True)
def _set_data_path(tmp_path):
    """Set DATA_PATH to tmp_path so tools accept test directories."""
    old_data = bash_mod.DATA_PATH
    old_docs = bash_mod.DOCUMENTS_PATH
    old_env = bash_mod._SANDBOX_ENV
    bash_mod.DATA_PATH = str(tmp_path)
    bash_mod.DOCUMENTS_PATH = None
    bash_mod._SANDBOX_ENV = None
    yield
    bash_mod.DATA_PATH = old_data
    bash_mod.DOCUMENTS_PATH = old_docs
    bash_mod._SANDBOX_ENV = old_env


# ── search_directory ───────────────────────────────────────────────────────


class TestSearchDirectorySecurity:
    """Shell injection prevention in search_directory."""

    def test_normal_pattern(self, tmp_path):
        (tmp_path / "hello.txt").write_text("hello world\nfoo bar\nhello again\n")
        result = json.loads(
            search_directory(directory=str(tmp_path), pattern="hello", file_pattern="*.txt")
        )
        assert result["status"] == "success"
        assert result["total_matches"] >= 1

    def test_pattern_with_single_quote(self, tmp_path):
        (tmp_path / "test.txt").write_text("it's a test\n")
        result = json.loads(search_directory(directory=str(tmp_path), pattern="it's"))
        # Should succeed or fail gracefully — never execute injected commands
        assert result["status"] in ("success", "failed")

    def test_pattern_shell_injection(self, tmp_path):
        (tmp_path / "test.txt").write_text("safe content\n")
        marker = tmp_path / "pwned_search_pattern"
        malicious = f"'; touch '{marker}'; echo '"
        result = json.loads(search_directory(directory=str(tmp_path), pattern=malicious))
        assert not marker.exists(), "Shell injection via pattern succeeded"

    def test_file_pattern_shell_injection(self, tmp_path):
        (tmp_path / "test.txt").write_text("content\n")
        marker = tmp_path / "pwned_search_file_pattern"
        malicious = f"'; touch '{marker}'; echo '"
        result = json.loads(
            search_directory(directory=str(tmp_path), pattern="content", file_pattern=malicious)
        )
        assert not marker.exists(), "Shell injection via file_pattern succeeded"


# ── find_files ─────────────────────────────────────────────────────────────


class TestFindFilesSecurity:
    """Shell injection prevention and input validation in find_files."""

    def test_normal_find(self, tmp_path):
        (tmp_path / "test.py").write_text("pass\n")
        result = json.loads(find_files(directory=str(tmp_path), name_pattern="*.py"))
        assert result["status"] == "success"
        assert result["total_files"] >= 1

    def test_name_pattern_shell_injection(self, tmp_path):
        (tmp_path / "test.txt").write_text("x\n")
        marker = tmp_path / "pwned_find_name"
        malicious = f"'; touch '{marker}'; echo '"
        result = json.loads(find_files(directory=str(tmp_path), name_pattern=malicious))
        assert not marker.exists(), "Shell injection via name_pattern succeeded"

    def test_invalid_file_type_rejected(self, tmp_path):
        result = json.loads(find_files(directory=str(tmp_path), file_type="f; rm -rf /"))
        assert result["status"] == "error"
        assert "Invalid file_type" in result["error"]

    def test_valid_file_types_accepted(self, tmp_path):
        (tmp_path / "test.txt").write_text("x\n")
        for ft in ["f", "d", "l"]:
            result = json.loads(find_files(directory=str(tmp_path), file_type=ft))
            assert result["status"] == "success"

    def test_invalid_size_filter_rejected(self, tmp_path):
        result = json.loads(find_files(directory=str(tmp_path), size_filter="+1M; rm -rf /"))
        assert result["status"] == "error"
        assert "Invalid size_filter" in result["error"]

    def test_valid_size_filter_accepted(self, tmp_path):
        result = json.loads(find_files(directory=str(tmp_path), size_filter="+1M"))
        assert result["status"] == "success"


# ── transform_text ─────────────────────────────────────────────────────────


class TestTransformTextSecurity:
    """Shell injection prevention in transform_text."""

    def test_normal_sed(self):
        result = json.loads(
            transform_text(input_text="hello world", operation="sed", expression="s/hello/goodbye/g")
        )
        assert result["status"] == "success"
        assert "goodbye" in result["output"]

    def test_normal_awk(self):
        result = json.loads(
            transform_text(input_text="hello world", operation="awk", expression="{print $1}")
        )
        assert result["status"] == "success"
        assert "hello" in result["output"]

    def test_normal_tr(self):
        result = json.loads(
            transform_text(input_text="hello", operation="tr", expression="a-z A-Z")
        )
        assert result["status"] == "success"
        assert "HELLO" in result["output"]

    def test_sed_injection(self, tmp_path):
        marker = tmp_path / "pwned_sed"
        malicious = f"s/x/y/'; touch '{marker}'; echo '"
        result = json.loads(
            transform_text(input_text="test", operation="sed", expression=malicious)
        )
        assert not marker.exists(), "Shell injection via sed expression succeeded"

    def test_awk_injection(self, tmp_path):
        marker = tmp_path / "pwned_awk"
        malicious = f"{{print}}'; touch '{marker}'; echo '"
        result = json.loads(
            transform_text(input_text="test", operation="awk", expression=malicious)
        )
        assert not marker.exists(), "Shell injection via awk expression succeeded"

    def test_tr_injection(self, tmp_path):
        marker = tmp_path / "pwned_tr"
        malicious = f"a-z A-Z; touch {marker}"
        result = json.loads(
            transform_text(input_text="test", operation="tr", expression=malicious)
        )
        assert not marker.exists(), "Shell injection via tr expression succeeded"

    def test_file_path_with_spaces(self, tmp_path):
        test_file = tmp_path / "test file.txt"
        test_file.write_text("hello world\n")
        result = json.loads(
            transform_text(
                input_text=str(test_file), operation="sed",
                expression="s/hello/goodbye/g", is_file=True
            )
        )
        assert result["status"] == "success"
        assert "goodbye" in result["output"]
