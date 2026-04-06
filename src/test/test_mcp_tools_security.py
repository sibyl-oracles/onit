"""Security tests for MCP tool functions — shell injection prevention, path validation, sandbox."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import src.mcp.servers.tasks.os.bash.mcp_server as bash_mod
from src.mcp.servers.tasks.os.bash.mcp_server import (
    _validate_read_path,
    _validate_dir_path,
    _validate_bash_command,
    _get_sandbox_env,
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


# ── _validate_read_path ───────────────────────────────────────────────────


class TestValidateReadPath:
    """Path validation for file read operations."""

    def test_allows_path_within_data_path(self, tmp_path):
        test_file = tmp_path / "allowed.txt"
        test_file.write_text("ok")
        result = _validate_read_path(str(test_file))
        assert result == os.path.realpath(str(test_file))

    def test_allows_path_within_documents_path(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        doc_file = docs_dir / "readme.txt"
        doc_file.write_text("doc content")
        bash_mod.DOCUMENTS_PATH = str(docs_dir)
        result = _validate_read_path(str(doc_file))
        assert result == os.path.realpath(str(doc_file))

    def test_rejects_path_outside_allowed(self, tmp_path):
        with pytest.raises(ValueError, match="Read access denied"):
            _validate_read_path("/etc/passwd")

    def test_rejects_symlink_escape(self, tmp_path):
        """Symlink pointing outside DATA_PATH should be rejected."""
        link = tmp_path / "escape_link"
        link.symlink_to("/etc")
        target = str(link / "passwd")
        with pytest.raises(ValueError, match="Read access denied"):
            _validate_read_path(target)


# ── _validate_dir_path ────────────────────────────────────────────────────


class TestValidateDirPath:
    """Path validation for directory operations."""

    def test_allows_data_path_itself(self, tmp_path):
        result = _validate_dir_path(str(tmp_path))
        assert result == os.path.realpath(str(tmp_path))

    def test_allows_subdir_of_data_path(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        result = _validate_dir_path(str(sub))
        assert result == os.path.realpath(str(sub))

    def test_allows_documents_path(self, tmp_path):
        docs = tmp_path / "my_docs"
        docs.mkdir()
        bash_mod.DOCUMENTS_PATH = str(docs)
        result = _validate_dir_path(str(docs))
        assert result == os.path.realpath(str(docs))

    def test_rejects_outside_allowed(self):
        with pytest.raises(ValueError, match="Directory access denied"):
            _validate_dir_path("/tmp/random_other_dir")

    def test_rejects_parent_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="Directory access denied"):
            _validate_dir_path(str(tmp_path / ".."))


# ── _validate_bash_command ────────────────────────────────────────────────


class TestValidateBashCommand:
    """Bash command blocklist and path restriction."""

    def test_allows_safe_command(self, tmp_path):
        assert _validate_bash_command(f"ls {tmp_path}") is None

    def test_blocks_env_command(self):
        result = _validate_bash_command("env")
        assert result is not None
        assert "env command" in result

    def test_blocks_printenv(self):
        result = _validate_bash_command("printenv HOME")
        assert result is not None
        assert "printenv" in result

    def test_blocks_ps(self):
        result = _validate_bash_command("ps aux")
        assert result is not None
        assert "ps command" in result

    def test_blocks_proc_self_environ(self):
        result = _validate_bash_command("cat /proc/self/environ")
        assert result is not None
        assert "/proc/self/environ" in result

    def test_blocks_etc_passwd(self):
        result = _validate_bash_command("cat /etc/passwd")
        assert result is not None
        assert "/etc/passwd" in result

    def test_blocks_path_outside_allowed(self, tmp_path):
        result = _validate_bash_command("cat /home/user/secrets.txt")
        assert result is not None
        assert "outside allowed directories" in result

    def test_allows_path_within_data_path(self, tmp_path):
        assert _validate_bash_command(f"cat {tmp_path}/file.txt") is None

    def test_allows_standard_tools(self):
        assert _validate_bash_command("/usr/bin/wc -l") is None
        assert _validate_bash_command("/bin/echo hello") is None


# ── _get_sandbox_env ──────────────────────────────────────────────────────


class TestGetSandboxEnv:
    """Sandbox environment for subprocess execution."""

    def test_returns_minimal_env(self):
        bash_mod._SANDBOX_ENV = None  # reset cache
        env = _get_sandbox_env()
        assert "PATH" in env
        assert "LANG" in env
        assert "DATA_PATH" in env
        # HOME is intentionally set to the real home so credential helpers work
        assert "HOME" in env
        assert env["HOME"] == os.path.expanduser("~")

    def test_excludes_host_secrets(self, tmp_path):
        bash_mod._SANDBOX_ENV = None
        os.environ["SUPER_SECRET_KEY"] = "should-not-leak"
        env = _get_sandbox_env()
        assert "SUPER_SECRET_KEY" not in env
        del os.environ["SUPER_SECRET_KEY"]

    def test_includes_documents_path_when_set(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        bash_mod.DOCUMENTS_PATH = str(docs)
        bash_mod._SANDBOX_ENV = None
        env = _get_sandbox_env()
        assert "DOCUMENTS_PATH" in env
        assert env["DOCUMENTS_PATH"] == os.path.realpath(str(docs))

    def test_caches_result(self, tmp_path):
        bash_mod._SANDBOX_ENV = None
        env1 = _get_sandbox_env()
        env2 = _get_sandbox_env()
        assert env1 == env2
        assert bash_mod._SANDBOX_ENV is not None
