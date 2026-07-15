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
    """Set DATA_PATH to tmp_path so tools accept test directories.
    Also resets containment state so a violation recorded in one test can
    never lock down the rest of the suite."""
    old_data = bash_mod.DATA_PATH
    old_docs = bash_mod.DOCUMENTS_PATH
    old_env = bash_mod._SANDBOX_ENV
    bash_mod.DATA_PATH = str(tmp_path)
    bash_mod.DOCUMENTS_PATH = None
    bash_mod._SANDBOX_ENV = None
    bash_mod._VIOLATIONS = []
    bash_mod._CONTAINED = False
    yield
    bash_mod.DATA_PATH = old_data
    bash_mod.DOCUMENTS_PATH = old_docs
    bash_mod._SANDBOX_ENV = old_env
    bash_mod._VIOLATIONS = []
    bash_mod._CONTAINED = False


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


# ── settings-file permission rules ────────────────────────────────────────


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Point ONIT_SETTINGS at a writable settings file; returns a writer."""
    path = tmp_path / "settings.json"
    monkeypatch.setenv("ONIT_SETTINGS", str(path))
    monkeypatch.setattr(bash_mod, "SETTINGS_PATH", None)
    monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)

    def write(permissions: dict):
        path.write_text(json.dumps({"permissions": permissions}))
        bash_mod._PERMISSIONS_CACHE = None
        return path

    return write


class TestPermissionRules:
    """Claude Code-style allow/deny rules from ~/.onit/settings.json."""

    def test_no_settings_file_allows_everything(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONIT_SETTINGS", str(tmp_path / "missing.json"))
        monkeypatch.setattr(bash_mod, "SETTINGS_PATH", None)
        monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)
        assert _validate_bash_command("pip install requests") is None

    def test_deny_rule_blocks_command(self, settings_file):
        settings_file({"allow": ["Bash(*)"], "deny": ["Bash(pip install*)"]})
        result = _validate_bash_command("pip install requests")
        assert result is not None and "deny rule" in result

    def test_deny_rule_blocks_compound_command_segment(self, settings_file):
        settings_file({"allow": ["Bash(*)"], "deny": ["Bash(pip install*)"]})
        result = _validate_bash_command("echo hi && pip install requests")
        assert result is not None and "deny rule" in result

    def test_deny_rule_allows_unrelated_command(self, settings_file):
        settings_file({"allow": ["Bash(*)"], "deny": ["Bash(pip install*)"]})
        assert _validate_bash_command("pip list") is None

    def test_allowlist_blocks_unmatched_command(self, settings_file):
        settings_file({"allow": ["Bash(git *)"]})
        result = _validate_bash_command("pip list")
        assert result is not None and "allow rule" in result

    def test_allowlist_permits_matched_command(self, settings_file):
        settings_file({"allow": ["Bash(git *)"]})
        assert _validate_bash_command("git status") is None

    def test_allowlist_requires_every_segment(self, settings_file):
        settings_file({"allow": ["Bash(git *)"]})
        result = _validate_bash_command("git log | head -5")
        assert result is not None and "allow rule" in result

    def test_deny_applies_in_unrestricted_mode(self, settings_file, monkeypatch):
        settings_file({"deny": ["Bash(pip install*)"]})
        monkeypatch.setenv("ONIT_UNRESTRICTED", "1")
        result = _validate_bash_command("pip install requests")
        assert result is not None and "deny rule" in result

    def test_malformed_settings_ignored(self, tmp_path, monkeypatch):
        path = tmp_path / "settings.json"
        path.write_text("{not valid json")
        monkeypatch.setenv("ONIT_SETTINGS", str(path))
        monkeypatch.setattr(bash_mod, "SETTINGS_PATH", None)
        monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)
        assert _validate_bash_command("pip install requests") is None

    def test_non_bash_rules_ignored(self, settings_file):
        settings_file({"deny": ["WebFetch(domain:example.com)"]})
        assert _validate_bash_command("pip install requests") is None

    def test_edits_reload_without_restart(self, settings_file, tmp_path):
        path = settings_file({"deny": ["Bash(pip install*)"]})
        assert _validate_bash_command("pip install requests") is not None
        path.write_text(json.dumps({"permissions": {"deny": []}}))
        # Bump mtime past filesystem timestamp granularity
        stat = os.stat(path)
        os.utime(path, (stat.st_atime, stat.st_mtime + 2))
        assert _validate_bash_command("pip install requests") is None

    def test_builtin_blocklist_still_applies(self, settings_file):
        settings_file({"allow": ["Bash(*)"]})
        result = _validate_bash_command("sudo rm -rf /tmp/x")
        assert result is not None

    def test_env_prefix_does_not_bypass_deny(self, settings_file):
        settings_file({"deny": ["Bash(pip install*)"]})
        result = _validate_bash_command("PIP_TIMEOUT=60 pip install requests")
        assert result is not None and "deny rule" in result

    def test_wrapper_word_does_not_bypass_deny(self, settings_file):
        settings_file({"deny": ["Bash(pip install*)"]})
        for cmd in ("nohup pip install requests",
                    "command pip install requests",
                    "time pip install requests"):
            result = _validate_bash_command(cmd)
            assert result is not None and "deny rule" in result, cmd

    def test_python_m_pip_glob_rule(self, settings_file):
        settings_file({"deny": ["Bash(*-m pip install*)"]})
        for cmd in ("python -m pip install requests",
                    "python3 -m pip install requests",
                    "/opt/miniconda3/bin/python3.13 -m pip install requests",
                    "echo hi && python -m pip install requests"):
            result = _validate_bash_command(cmd)
            assert result is not None and "deny rule" in result, cmd
        assert _validate_bash_command("python -m pipdeptree") is None

    def test_env_prefix_allowed_by_effective_command(self, settings_file):
        settings_file({"allow": ["Bash(git *)"]})
        assert _validate_bash_command("GIT_PAGER=cat git log; git status") is None

    def test_serve_start_honors_deny_rules(self, settings_file, tmp_path):
        settings_file({"deny": ["Bash(pip install*)"]})
        result = json.loads(bash_mod.serve(action="start", command="pip install requests",
                                           name="pytest-denied", cwd=str(tmp_path)))
        assert result["status"] == "blocked"
        assert "deny rule" in result["error"]


class TestSettingsRulesUIGating:
    """The default ~/.onit/settings.json applies only to the web UI; the text
    UI runs privileged. Explicit ONIT_SETTINGS/settings_path enforces anywhere."""

    @pytest.fixture
    def default_settings(self, tmp_path, monkeypatch):
        """A deny rule living at the DEFAULT settings path (no explicit override)."""
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(
            {"permissions": {"deny": ["Bash(pip install*)"]}}))
        monkeypatch.delenv("ONIT_SETTINGS", raising=False)
        monkeypatch.delenv("ONIT_WEB_UI", raising=False)
        monkeypatch.setattr(bash_mod, "SETTINGS_PATH", None)
        monkeypatch.setattr(bash_mod, "DEFAULT_SETTINGS_PATH", str(path))
        monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)
        return path

    def test_text_ui_ignores_default_settings(self, default_settings):
        assert _validate_bash_command("pip install requests") is None

    def test_web_ui_enforces_default_settings(self, default_settings, monkeypatch):
        monkeypatch.setenv("ONIT_WEB_UI", "1")
        result = _validate_bash_command("pip install requests")
        assert result is not None and "deny rule" in result

    def test_explicit_onit_settings_enforces_in_text_ui(self, default_settings,
                                                        monkeypatch):
        monkeypatch.setenv("ONIT_SETTINGS", str(default_settings))
        result = _validate_bash_command("pip install requests")
        assert result is not None and "deny rule" in result

    def test_explicit_settings_path_enforces_in_text_ui(self, default_settings,
                                                        monkeypatch):
        monkeypatch.setattr(bash_mod, "SETTINGS_PATH", str(default_settings))
        result = _validate_bash_command("pip install requests")
        assert result is not None and "deny rule" in result

    def test_text_ui_ignores_default_allowed_commands(self, tmp_path, monkeypatch):
        """allowedCommands extras from the default file are also web-UI-only."""
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(
            {"permissions": {"allowedCommands": ["mytool"]}}))
        monkeypatch.delenv("ONIT_SETTINGS", raising=False)
        monkeypatch.setattr(bash_mod, "SETTINGS_PATH", None)
        monkeypatch.setattr(bash_mod, "DEFAULT_SETTINGS_PATH", str(path))
        monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)
        monkeypatch.delenv("ONIT_WEB_UI", raising=False)
        assert "mytool" not in bash_mod._allowed_commands()
        monkeypatch.setenv("ONIT_WEB_UI", "1")
        assert "mytool" in bash_mod._allowed_commands()


# ── AST command allowlist enforcement ─────────────────────────────────────


@pytest.fixture
def enforced(monkeypatch, tmp_path):
    """Enable allowlist enforcement with isolated settings."""
    monkeypatch.setenv("ONIT_COMMAND_ALLOWLIST", "1")
    monkeypatch.setenv("ONIT_SETTINGS", str(tmp_path / "missing-settings.json"))
    monkeypatch.setattr(bash_mod, "SETTINGS_PATH", None)
    monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)


class TestCommandAllowlist:
    """AST-based allowlist wired into _validate_bash_command."""

    def test_disabled_by_default_on_host(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ONIT_COMMAND_ALLOWLIST", raising=False)
        monkeypatch.delenv("ONIT_CONTAINER", raising=False)
        monkeypatch.setenv("ONIT_SETTINGS", str(tmp_path / "missing.json"))
        monkeypatch.setattr(bash_mod, "SETTINGS_PATH", None)
        monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)
        assert bash_mod._validate_bash_command("somerandomtool") is None

    def test_enforced_in_container_mode(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ONIT_CONTAINER", "1")
        monkeypatch.delenv("ONIT_COMMAND_ALLOWLIST", raising=False)
        monkeypatch.setenv("ONIT_SETTINGS", str(tmp_path / "missing.json"))
        monkeypatch.setattr(bash_mod, "SETTINGS_PATH", None)
        monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)
        err = bash_mod._validate_bash_command("evilbin --x")
        assert err is not None and "allowlist" in err

    def test_container_opt_out(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ONIT_CONTAINER", "1")
        monkeypatch.setenv("ONIT_COMMAND_ALLOWLIST", "0")
        monkeypatch.setenv("ONIT_SETTINGS", str(tmp_path / "missing.json"))
        monkeypatch.setattr(bash_mod, "SETTINGS_PATH", None)
        monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)
        assert bash_mod._validate_bash_command("evilbin --x") is None

    def test_allowed_command_passes(self, enforced, tmp_path):
        assert bash_mod._validate_bash_command(f"ls {tmp_path}") is None

    def test_unknown_command_blocked(self, enforced):
        err = bash_mod._validate_bash_command("evilbin --x")
        assert err is not None and "allowlist" in err

    def test_substitution_checked(self, enforced):
        err = bash_mod._validate_bash_command("echo $(evilbin)")
        assert err is not None and "allowlist" in err

    def test_package_install_gated(self, enforced):
        err = bash_mod._validate_bash_command("pip install requests")
        assert err is not None and "disabled by default" in err

    def test_pinned_install_with_optin(self, enforced, monkeypatch):
        monkeypatch.setenv("ONIT_ALLOW_PACKAGE_INSTALL", "1")
        assert bash_mod._validate_bash_command(
            "pip install requests==2.31.0") is None
        err = bash_mod._validate_bash_command("pip install requests")
        assert err is not None and "not version-pinned" in err

    def test_env_var_extends_allowlist(self, enforced, monkeypatch):
        monkeypatch.setenv("ONIT_ALLOWED_COMMANDS", "mytool,othertool")
        assert bash_mod._validate_bash_command("mytool --run") is None

    def test_settings_extend_allowlist(self, enforced, settings_file):
        settings_file({"allowedCommands": ["mytool"]})
        assert bash_mod._validate_bash_command("mytool --run") is None

    def test_onit_install_ml_gated(self, enforced, monkeypatch):
        err = bash_mod._validate_bash_command("onit-install-ml torch")
        assert err is not None and "allowlist" in err
        monkeypatch.setenv("ONIT_ALLOW_PACKAGE_INSTALL", "1")
        assert bash_mod._validate_bash_command("onit-install-ml torch") is None

    def test_serve_start_honors_allowlist(self, enforced, tmp_path):
        result = json.loads(bash_mod.serve(
            action="start", command="evilserver --port 8080",
            name="pytest-allowlist", cwd=str(tmp_path)))
        assert result["status"] == "blocked"
        assert "allowlist" in result["error"]

    def test_explicit_enforcement_overrides_unrestricted(self, enforced, monkeypatch):
        monkeypatch.setenv("ONIT_UNRESTRICTED", "1")
        err = bash_mod._validate_bash_command("evilbin --x")
        assert err is not None and "allowlist" in err


# ── auto-containment ───────────────────────────────────────────────────────


class TestAutoContainment:
    """Repeated policy violations lock down execution/write/send tools."""

    @pytest.fixture
    def bash_tool(self):
        tool = bash_mod.bash
        return tool.fn if hasattr(tool, "fn") else tool

    async def test_containment_after_threshold(self, enforced, monkeypatch,
                                               tmp_path, bash_tool):
        monkeypatch.setenv("ONIT_CONTAIN_THRESHOLD", "3")
        for _ in range(3):
            r = json.loads(await bash_tool(command="evilbin"))
            assert r["status"] == "blocked"
        # Now even an allowed command is refused.
        r = json.loads(await bash_tool(command=f"ls {tmp_path}"))
        assert r["status"] == "contained"
        assert os.path.isfile(bash_mod._containment_marker_path())

    async def test_below_threshold_not_contained(self, enforced, monkeypatch,
                                                 tmp_path, bash_tool):
        monkeypatch.setenv("ONIT_CONTAIN_THRESHOLD", "3")
        for _ in range(2):
            await bash_tool(command="evilbin")
        r = json.loads(await bash_tool(command=f"ls {tmp_path}"))
        assert r["status"] in ("success", "failed")

    async def test_threshold_zero_disables(self, enforced, monkeypatch,
                                           tmp_path, bash_tool):
        monkeypatch.setenv("ONIT_CONTAIN_THRESHOLD", "0")
        for _ in range(10):
            await bash_tool(command="evilbin")
        r = json.loads(await bash_tool(command=f"ls {tmp_path}"))
        assert r["status"] in ("success", "failed")

    def test_containment_gates_write_tools(self, tmp_path):
        bash_mod._CONTAINED = True
        write_fn = bash_mod.write_file.fn if hasattr(bash_mod.write_file, "fn") \
            else bash_mod.write_file
        edit_fn = bash_mod.edit_file.fn if hasattr(bash_mod.edit_file, "fn") \
            else bash_mod.edit_file
        send_fn = bash_mod.send_file.fn if hasattr(bash_mod.send_file, "fn") \
            else bash_mod.send_file
        assert json.loads(write_fn(path="x.txt", content="hi"))["status"] == "contained"
        assert json.loads(edit_fn(path="x.txt", old_string="a", new_string="b"))["status"] == "contained"
        assert json.loads(send_fn(path="x.txt"))["status"] == "contained"
        assert json.loads(transform_text(input_text="x", operation="sed",
                                         expression="s/x/y/"))["status"] == "contained"

    def test_containment_gates_serve_start(self, tmp_path):
        bash_mod._CONTAINED = True
        result = json.loads(bash_mod.serve(action="start", command="sleep 60",
                                           name="pytest-contained", cwd=str(tmp_path)))
        assert result["status"] == "contained"

    def test_marker_persists_containment(self, enforced, tmp_path):
        bash_mod._CONTAINED = True
        bash_mod._enter_containment()
        # Simulate a restart: in-memory flag cleared, marker still on disk.
        bash_mod._CONTAINED = False
        assert bash_mod._is_contained() is True

    def test_reads_still_work_when_contained(self, tmp_path):
        (tmp_path / "readable.txt").write_text("still visible\n")
        bash_mod._CONTAINED = True
        read_fn = bash_mod.read_file.fn if hasattr(bash_mod.read_file, "fn") \
            else bash_mod.read_file
        result = json.loads(read_fn(path=str(tmp_path / "readable.txt")))
        assert result["status"] == "success"
