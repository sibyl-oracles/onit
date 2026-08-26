"""Commands the policy will not decide alone go to a person.

The old policy had two answers: run it, or refuse it. Every executable nobody
had thought to list, and every install nobody had pre-approved, came back as a
refusal — which is what made an agent spend turns rephrasing a command that
was never going to run. Claude Code answers the same situation by asking, and
these tests pin the version of that OnIt can afford on a shared host.

Two things are being checked throughout, and they pull in opposite directions:

* the ask really is an ask — the command does not run until someone says yes,
  and the model cannot say yes on its own behalf;
* the ask is available only where a human's yes is worth something. On a
  multi-tenant deployment the person answering is one of several people the
  answer affects, so the boundaries between them stay machine-enforced.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import src.mcp.servers.tasks.os.bash.mcp_server as bash_mod  # noqa: E402
from src.mcp.servers.tasks.os.bash import approvals  # noqa: E402
from src.mcp.servers.tasks.os.bash.command_policy import (  # noqa: E402
    DEFAULT_ALLOWED_COMMANDS,
    check_command,
    evaluate_command,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(bash_mod, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(bash_mod, "DOCUMENTS_PATH", None)
    monkeypatch.setattr(bash_mod, "_SANDBOX_ENV", None)
    monkeypatch.setattr(bash_mod, "_VIOLATIONS", {})
    monkeypatch.setattr(bash_mod, "_CONTAINED", set())
    bash_mod._APPROVALS.reset()
    yield
    bash_mod._APPROVALS.reset()


@pytest.fixture
def enforced(monkeypatch, tmp_path):
    monkeypatch.setenv("ONIT_COMMAND_ALLOWLIST", "1")
    monkeypatch.setenv("ONIT_SETTINGS", str(tmp_path / "missing-settings.json"))
    monkeypatch.setattr(bash_mod, "SETTINGS_PATH", None)
    monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)


@pytest.fixture
def with_human(monkeypatch):
    """A deployment that declared it can reach a person."""
    monkeypatch.setenv("ONIT_APPROVAL_CHANNEL", "1")
    monkeypatch.delenv("ONIT_ASK_APPROVAL", raising=False)
    monkeypatch.delenv("ONIT_WEB_UI", raising=False)


@pytest.fixture
def bash_tool():
    tool = bash_mod.bash
    return tool.fn if hasattr(tool, "fn") else tool


def _session(tmp_path, name):
    path = tmp_path / name
    path.mkdir(exist_ok=True)
    return str(path)


# ── the policy layer ───────────────────────────────────────────────────────


class TestDecision:
    def test_unlisted_executable_is_a_question(self):
        d = evaluate_command("mytool run x.ts", DEFAULT_ALLOWED_COMMANDS,
                             ask_scope=frozenset({"command"}))
        assert d.verdict == "ask"
        assert d.subjects == ("mytool",)

    def test_without_a_human_it_is_the_old_refusal(self):
        """The message does not change with whether anyone was watching.

        A refusal the model can only interpret by knowing which deployment it
        is running in is a refusal it cannot learn anything from.
        """
        asked = evaluate_command("mytool run x.ts", DEFAULT_ALLOWED_COMMANDS,
                                 ask_scope=frozenset({"command"}))
        denied = evaluate_command("mytool run x.ts", DEFAULT_ALLOWED_COMMANDS)
        assert denied.verdict == "deny"
        assert denied.reason == asked.reason
        assert check_command("mytool run x.ts", DEFAULT_ALLOWED_COMMANDS) == denied.reason

    def test_a_parse_failure_is_never_a_question(self):
        """Nobody can approve what nobody can read.

        The parser fails closed precisely because it could not work out what
        would run; putting that string in front of a person and asking them to
        vouch for it would launder the uncertainty rather than resolve it.
        """
        d = evaluate_command("case $x in a) ls;; esac", DEFAULT_ALLOWED_COMMANDS,
                             ask_scope=frozenset({"command", "install"}))
        assert d.verdict == "deny"

    def test_a_deny_beats_an_ask_in_the_same_command(self):
        d = evaluate_command("mytool run x.ts; case $y in b) ls;; esac",
                             DEFAULT_ALLOWED_COMMANDS,
                             ask_scope=frozenset({"command"}))
        assert d.verdict == "deny"

    def test_one_prompt_covers_every_unlisted_executable(self):
        d = evaluate_command("mytool run x.ts | othertool run y.ts",
                             DEFAULT_ALLOWED_COMMANDS,
                             ask_scope=frozenset({"command"}))
        assert d.verdict == "ask"
        assert set(d.subjects) == {"mytool", "othertool"}

    def test_an_unpinned_install_is_a_question(self):
        d = evaluate_command("pip install requests", DEFAULT_ALLOWED_COMMANDS,
                             allow_installs=True,
                             ask_scope=frozenset({"install"}))
        assert d.verdict == "ask"

    def test_a_pinned_install_needs_no_one(self):
        d = evaluate_command("pip install requests==2.31.0",
                             DEFAULT_ALLOWED_COMMANDS, allow_installs=True,
                             ask_scope=frozenset({"install"}))
        assert d.verdict == "allow"


# ── the bash tool ──────────────────────────────────────────────────────────


class TestAskingThroughTheTool:
    async def test_no_channel_means_the_old_refusal(self, enforced, tmp_path,
                                                    bash_tool, monkeypatch):
        monkeypatch.delenv("ONIT_APPROVAL_CHANNEL", raising=False)
        r = json.loads(await bash_tool(command="mytool run x.ts",
                                       data_path=_session(tmp_path, "s")))
        assert r["status"] == "blocked"

    async def test_a_channel_turns_it_into_a_ticket(self, enforced, with_human,
                                                    tmp_path, bash_tool):
        r = json.loads(await bash_tool(command="mytool run x.ts",
                                       data_path=_session(tmp_path, "s")))
        assert r["status"] == "needs_approval"
        assert r["approval_id"]
        assert r["subjects"] == ["mytool"]

    async def test_the_command_did_not_run(self, enforced, with_human,
                                           tmp_path, bash_tool):
        base = _session(tmp_path, "s")
        await bash_tool(command=f"mytool run {base}/marker.txt", data_path=base)
        assert not os.path.exists(os.path.join(base, "marker.txt"))

    async def test_a_ticket_lets_it_through(self, enforced, with_human,
                                            tmp_path, bash_tool):
        base = _session(tmp_path, "s")
        ticket = json.loads(await bash_tool(command="echo hi; mytool --version",
                                            data_path=base))["approval_id"]
        r = json.loads(await bash_tool(command="echo hi; mytool --version",
                                       data_path=base,
                                       approval_token=ticket))
        # mytool does not exist here; what matters is that policy let go of it.
        assert r["status"] != "needs_approval"
        assert "allowlist" not in json.dumps(r)

    async def test_a_ticket_is_single_use(self, enforced, with_human,
                                          tmp_path, bash_tool):
        base = _session(tmp_path, "s")
        cmd = "mytool run x.ts"
        ticket = json.loads(await bash_tool(command=cmd, data_path=base))["approval_id"]
        await bash_tool(command=cmd, data_path=base, approval_token=ticket)
        again = json.loads(await bash_tool(command=cmd, data_path=base,
                                           approval_token=ticket))
        assert again["status"] == "needs_approval"

    async def test_a_ticket_is_bound_to_its_command(self, enforced, with_human,
                                                    tmp_path, bash_tool):
        """Approving one command must not approve the next one.

        This is the whole reason the ticket carries a digest: a person who
        said yes to `mytool --version` has not said yes to `mytool run
        untrusted.ts`, and an approval that generalised would be a capability
        rather than an answer.
        """
        base = _session(tmp_path, "s")
        ticket = json.loads(await bash_tool(command="mytool --version",
                                            data_path=base))["approval_id"]
        r = json.loads(await bash_tool(command="mytool run untrusted.ts",
                                       data_path=base,
                                       approval_token=ticket))
        assert r["status"] == "needs_approval"
        assert r["approval_id"] != ticket

    async def test_a_ticket_is_bound_to_its_session(self, enforced, with_human,
                                                    tmp_path, bash_tool):
        alice = _session(tmp_path, "alice")
        bob = _session(tmp_path, "bob")
        ticket = json.loads(await bash_tool(command="mytool run x.ts",
                                            data_path=alice))["approval_id"]
        r = json.loads(await bash_tool(command="mytool run x.ts", data_path=bob,
                                       approval_token=ticket))
        assert r["status"] == "needs_approval"

    async def test_an_invented_ticket_is_worth_nothing(self, enforced,
                                                       with_human, tmp_path,
                                                       bash_tool):
        """The harness strips this argument, but the server does not rely on
        that: a guessed token has to fail on its own merits too."""
        base = _session(tmp_path, "s")
        r = json.loads(await bash_tool(command="mytool run x.ts", data_path=base,
                                       approval_token="not-a-real-token"))
        assert r["status"] == "needs_approval"

    async def test_session_scope_stops_the_asking(self, enforced, with_human,
                                                  tmp_path, bash_tool):
        base = _session(tmp_path, "s")
        ticket = json.loads(await bash_tool(command="mytool --version",
                                            data_path=base))["approval_id"]
        await bash_tool(command="mytool --version", data_path=base,
                        approval_token=ticket, approval_scope="session")
        r = json.loads(await bash_tool(command="mytool run other.ts",
                                       data_path=base))
        assert r["status"] != "needs_approval"

    async def test_a_session_grant_stays_in_its_session(self, enforced,
                                                        with_human, tmp_path,
                                                        bash_tool):
        alice = _session(tmp_path, "alice")
        bob = _session(tmp_path, "bob")
        ticket = json.loads(await bash_tool(command="mytool --version",
                                            data_path=alice))["approval_id"]
        await bash_tool(command="mytool --version", data_path=alice,
                        approval_token=ticket, approval_scope="session")
        r = json.loads(await bash_tool(command="mytool --version", data_path=bob))
        assert r["status"] == "needs_approval"


class TestSayingNo:
    async def test_a_refusal_is_remembered(self, enforced, with_human,
                                           tmp_path, bash_tool):
        base = _session(tmp_path, "s")
        cmd = "mytool run x.ts"
        ticket = json.loads(await bash_tool(command=cmd, data_path=base))["approval_id"]
        refused = json.loads(await bash_tool(command=cmd, data_path=base,
                                             approval_token=ticket,
                                             approval_scope="deny"))
        assert refused["status"] == "blocked"
        # The same command must not raise a second prompt: asking again after
        # a no is how a model turns one decision into a war of attrition.
        again = json.loads(await bash_tool(command=cmd, data_path=base))
        assert again["status"] == "blocked"

    async def test_pressing_on_after_a_no_is_what_containment_is_for(
            self, enforced, with_human, tmp_path, bash_tool, monkeypatch):
        monkeypatch.setenv("ONIT_CONTAIN_THRESHOLD", "2")
        base = _session(tmp_path, "s")
        cmd = "mytool run x.ts"
        ticket = json.loads(await bash_tool(command=cmd, data_path=base))["approval_id"]
        await bash_tool(command=cmd, data_path=base, approval_token=ticket,
                        approval_scope="deny")
        for _ in range(2):
            await bash_tool(command=cmd, data_path=base)
        r = json.loads(await bash_tool(command="ls .", data_path=base))
        assert r["status"] == "contained"

    async def test_a_refusal_tells_the_model_to_stop(self, enforced, tmp_path,
                                                     bash_tool, monkeypatch):
        monkeypatch.delenv("ONIT_APPROVAL_CHANNEL", raising=False)
        r = json.loads(await bash_tool(command="mytool run x.ts",
                                       data_path=_session(tmp_path, "s")))
        assert r["retryable"] is False
        assert "different approach" in r["guidance"]


# ── what a human may not waive ─────────────────────────────────────────────


class TestNonNegotiable:
    async def test_privilege_escalation_is_never_asked(self, enforced,
                                                       with_human, tmp_path,
                                                       bash_tool):
        r = json.loads(await bash_tool(command="sudo ls /",
                                       data_path=_session(tmp_path, "s")))
        assert r["status"] == "blocked"
        assert "Do not look for another route" in r["guidance"]

    async def test_sealed_installs_are_never_asked(self, enforced, with_human,
                                                   tmp_path, bash_tool,
                                                   monkeypatch):
        """In the containerized web UI there is no override, so there is
        nothing to offer: a prompt here would be asking someone to authorize
        something that will not happen either way."""
        monkeypatch.setenv("ONIT_WEB_UI", "1")
        monkeypatch.setenv("ONIT_CONTAINER", "1")
        monkeypatch.setenv("ONIT_ALLOW_PACKAGE_INSTALL", "1")
        r = json.loads(await bash_tool(command="pip install requests==2.31.0",
                                       data_path=_session(tmp_path, "s")))
        assert r["status"] == "blocked"

    async def test_the_jail_holds_on_a_multi_tenant_deployment(
            self, with_human, tmp_path, bash_tool, monkeypatch):
        """One logged-in user cannot approve their way out of the jail.

        The web UI runs many people's sessions in one process under one OS
        account: the path check is all that separates them, and it is not any
        one session's to waive.
        """
        monkeypatch.setenv("ONIT_WEB_UI", "1")
        monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)
        r = json.loads(await bash_tool(command="cat /etc/hosts",
                                       data_path=_session(tmp_path, "s")))
        assert r["status"] == "blocked"

    async def test_the_same_slip_is_a_question_on_a_single_user_host(
            self, with_human, tmp_path, bash_tool, monkeypatch):
        monkeypatch.delenv("ONIT_WEB_UI", raising=False)
        monkeypatch.delenv("ONIT_CONTAINER", raising=False)
        r = json.loads(await bash_tool(command="cat /etc/hosts",
                                       data_path=_session(tmp_path, "s")))
        assert r["status"] == "needs_approval"
        assert r["subjects"] == ["path:/etc"]

    async def test_secrets_in_the_environment_stay_put_on_the_web(
            self, with_human, tmp_path, bash_tool, monkeypatch):
        """`env` prints the operator's GITHUB_TOKEN, not the asker's."""
        monkeypatch.setenv("ONIT_WEB_UI", "1")
        monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)
        r = json.loads(await bash_tool(command="printenv",
                                       data_path=_session(tmp_path, "s")))
        assert r["status"] == "blocked"

    async def test_an_operator_deny_rule_is_never_asked(self, with_human,
                                                        tmp_path, bash_tool,
                                                        monkeypatch):
        settings = tmp_path / "settings.json"
        settings.write_text(json.dumps(
            {"permissions": {"deny": ["Bash(curl *)"]}}))
        monkeypatch.setenv("ONIT_SETTINGS", str(settings))
        monkeypatch.setattr(bash_mod, "SETTINGS_PATH", None)
        monkeypatch.setattr(bash_mod, "_PERMISSIONS_CACHE", None)
        r = json.loads(await bash_tool(command="curl https://example.com",
                                       data_path=_session(tmp_path, "s")))
        assert r["status"] == "blocked"


class TestAskScope:
    def test_no_channel_no_scope(self, monkeypatch):
        monkeypatch.delenv("ONIT_APPROVAL_CHANNEL", raising=False)
        assert approvals.ask_scope() == frozenset()

    def test_operators_can_switch_asking_off(self, monkeypatch):
        monkeypatch.setenv("ONIT_APPROVAL_CHANNEL", "1")
        monkeypatch.setenv("ONIT_ASK_APPROVAL", "0")
        assert approvals.ask_scope() == frozenset()

    def test_multi_tenant_withholds_the_shared_boundaries(self, monkeypatch):
        monkeypatch.setenv("ONIT_APPROVAL_CHANNEL", "1")
        monkeypatch.delenv("ONIT_ASK_APPROVAL", raising=False)
        monkeypatch.setenv("ONIT_WEB_UI", "1")
        scope = approvals.ask_scope()
        assert "command" in scope
        assert "path" not in scope and "system" not in scope


# ── the harness half of the exchange ───────────────────────────────────────


class _GatedRegistry:
    """A tool that asks for approval the first time and runs the second.

    Stands in for the bash MCP server so the harness side can be tested on its
    own: what it does with a needs_approval result, and what it lets through
    to the tool.
    """

    def __init__(self, expect_token="ticket-1"):
        self.tools = {"bash": True}
        self.calls = []
        self.expect_token = expect_token

    def tool_accepts_param(self, tool_name, param_name):
        return param_name in ("data_path", "session_id", "approval_token",
                              "approval_scope")

    def __getitem__(self, name):
        async def handler(log_handler=None, **kwargs):
            self.calls.append(dict(kwargs))
            if kwargs.get("approval_token") == self.expect_token:
                if kwargs.get("approval_scope") == "deny":
                    return json.dumps({"status": "blocked",
                                       "error": "declined"})
                return json.dumps({"status": "success", "stdout": "ran"})
            return json.dumps({
                "status": "needs_approval",
                "approval_id": self.expect_token,
                "command": kwargs.get("command", ""),
                "reason": "'mytool' is not in the command allowlist.",
                "subjects": ["mytool"],
            })
        return handler


class _UI:
    """Enough of the chat UI for _execute_tool, plus a scripted answer."""

    def __init__(self, answer=None):
        self.answer = answer
        self.asked = []
        self.logs = []

    def add_tool_call(self, *a, **k): pass
    def show_tool_start(self, *a, **k): pass
    def start_tool_spinner(self, *a, **k): pass
    def stop_tool_spinner(self, *a, **k): pass
    def show_tool_done(self, *a, **k): pass
    def add_tool_result(self, *a, **k): pass
    def add_log(self, message, level="info"): self.logs.append(message)

    def ask_approval(self, request):
        self.asked.append(request)
        return self.answer


async def _run(registry, ui, args=None):
    from src.model.serving.chat import _execute_tool
    messages = []
    await _execute_tool(
        "bash", args if args is not None else {"command": "mytool run x.ts"},
        "call-1", registry, timeout=5, data_path="/tmp/session",
        chat_ui=ui, verbose=False, messages=messages,
        tool_call_history=[], max_repeated=5, session_id="session",
    )
    return messages


class TestHarnessApproval:
    async def test_the_model_cannot_supply_its_own_token(self):
        """The one property everything else rests on.

        approval_token is the harness's argument the way data_path is: it is
        dropped from whatever the model sent before the call goes out, so a
        model that has read a ticket id out of an earlier tool result — it
        can, the payload is right there in the transcript — still cannot use
        it.
        """
        registry = _GatedRegistry()
        await _run(registry, _UI(answer="deny"),
                   args={"command": "mytool run x.ts",
                         "approval_token": "ticket-1",
                         "approval_scope": "session"})
        assert registry.calls[0].get("approval_token") in (None, "")
        assert registry.calls[0].get("approval_scope") in (None, "")

    async def test_a_yes_re_issues_the_call_with_the_ticket(self):
        registry = _GatedRegistry()
        ui = _UI(answer="once")
        messages = await _run(registry, ui)
        assert len(ui.asked) == 1
        assert registry.calls[1]["approval_token"] == "ticket-1"
        assert registry.calls[1]["approval_scope"] == "once"
        assert "ran" in messages[-1]["content"]

    async def test_a_no_reaches_the_model_as_a_refusal(self):
        registry = _GatedRegistry()
        messages = await _run(registry, _UI(answer="deny"))
        result = json.loads(messages[-1]["content"])
        assert result["status"] == "blocked"
        assert result["retryable"] is False
        # The server still hears about it, so the next attempt at the same
        # command meets the earlier answer instead of a fresh prompt.
        assert registry.calls[1]["approval_scope"] == "deny"

    async def test_no_ui_means_no(self):
        """A run with nobody watching refuses rather than waits."""
        registry = _GatedRegistry()
        messages = await _run(registry, _NoAskUI())
        result = json.loads(messages[-1]["content"])
        assert result["status"] == "blocked"
        assert "No one is available" in result["error"]
        assert len(registry.calls) == 1

    async def test_an_unrecognised_answer_is_a_no(self):
        registry = _GatedRegistry()
        messages = await _run(registry, _UI(answer="maybe"))
        assert json.loads(messages[-1]["content"])["status"] == "blocked"

    async def test_a_ui_that_raises_is_a_no(self):
        class _Broken(_UI):
            def ask_approval(self, request):
                raise RuntimeError("no browser attached")

        registry = _GatedRegistry()
        messages = await _run(registry, _Broken())
        assert json.loads(messages[-1]["content"])["status"] == "blocked"


class _NoAskUI(_UI):
    """A UI with no way to ask — the A2A, gateway and scheduled-run case.

    Not a UI that answers no: one that has no method at all, which is how
    chat() tells "asked and refused" from "there was nobody to ask".
    """

    ask_approval = None


class TestNeverAskCommands:
    """Some executables are refused rather than asked about.

    The ask exists so that a person can take a risk that is theirs. These are
    the ones that are not: a container runtime, a remote shell, the account
    tools. Approving one through a chat window would be consenting on behalf
    of every other user of the host.
    """

    @pytest.mark.parametrize("command", [
        "docker run -v /:/host alpine sh",
        "kubectl get secrets -A",
        "ssh user@other-host",
        "rsync -a /data user@host:/tmp",
        "systemctl restart nginx",
        "crontab -e",
        "useradd backdoor",
    ])
    async def test_refused_even_with_a_human_present(self, enforced, with_human,
                                                     tmp_path, bash_tool,
                                                     command):
        r = json.loads(await bash_tool(command=command,
                                       data_path=_session(tmp_path, "s")))
        assert r["status"] == "blocked"
        assert "Do not look for another route" in r["guidance"]

    async def test_it_counts_toward_containment(self, enforced, with_human,
                                                tmp_path, bash_tool,
                                                monkeypatch):
        monkeypatch.setenv("ONIT_CONTAIN_THRESHOLD", "2")
        base = _session(tmp_path, "s")
        for _ in range(2):
            await bash_tool(command="docker ps", data_path=base)
        r = json.loads(await bash_tool(command="ls .", data_path=base))
        assert r["status"] == "contained"

    async def test_the_toolchain_additions_just_run(self, enforced, with_human,
                                                    tmp_path, bash_tool):
        """A linter nobody listed is not a security question.

        `deno`, `bun`, `gh` and the rest were added outright rather than left
        to the prompt: an allowlist that already holds python, node, perl and
        bash is not what stands between the agent and arbitrary code, so
        stopping to ask about a JavaScript runtime spends a person's attention
        on nothing.
        """
        base = _session(tmp_path, "s")
        for command in ("deno --version", "bun --version", "gh --version",
                        "tree .", "bc --version"):
            r = json.loads(await bash_tool(command=command, data_path=base))
            assert r["status"] != "needs_approval", command
            assert "allowlist" not in json.dumps(r), command


# ── both halves, joined ────────────────────────────────────────────────────


class _RealBashRegistry:
    """The actual bash MCP tool, reachable through the harness dispatcher.

    The two halves of this feature are written in different processes and
    speak only through a JSON payload; a test double on either side would
    happily agree with a mistake in the other. This one wires the real tool
    into the real dispatcher.
    """

    def __init__(self, data_path):
        self.tools = {"bash": True}
        self.data_path = data_path
        self.tool = bash_mod.bash.fn if hasattr(bash_mod.bash, "fn") else bash_mod.bash

    def tool_accepts_param(self, tool_name, param_name):
        return param_name in ("data_path", "session_id", "approval_token",
                              "approval_scope")

    def __getitem__(self, name):
        async def handler(log_handler=None, session_id=None, **kwargs):
            return await self.tool(**kwargs)
        return handler


class TestEndToEnd:
    async def test_a_yes_runs_the_command(self, enforced, with_human, tmp_path):
        from src.model.serving.chat import _execute_tool

        base = _session(tmp_path, "s")
        registry = _RealBashRegistry(base)
        ui = _UI(answer="once")
        messages = []
        await _execute_tool(
            "bash", {"command": "mytool --version"}, "call-1", registry,
            timeout=20, data_path=base, chat_ui=ui, verbose=False,
            messages=messages, tool_call_history=[], max_repeated=5,
            session_id="s",
        )
        assert len(ui.asked) == 1
        assert ui.asked[0]["subjects"] == ["mytool"]
        result = json.loads(messages[-1]["content"])
        # mytool does not exist, so the shell fails — but policy let it run,
        # which is the whole question here.
        assert result["status"] != "needs_approval"
        assert "allowlist" not in json.dumps(result)

    async def test_a_no_stops_at_the_policy(self, enforced, with_human,
                                            tmp_path):
        from src.model.serving.chat import _execute_tool

        base = _session(tmp_path, "s")
        registry = _RealBashRegistry(base)
        ui = _UI(answer="deny")
        messages = []
        marker = os.path.join(base, "ran.txt")
        await _execute_tool(
            "bash", {"command": f"mytool > {marker}"}, "call-1", registry,
            timeout=20, data_path=base, chat_ui=ui, verbose=False,
            messages=messages, tool_call_history=[], max_repeated=5,
            session_id="s",
        )
        assert not os.path.exists(marker)
        assert json.loads(messages[-1]["content"])["status"] == "blocked"
        # And the server remembers, so the model asking again gets the answer
        # rather than another prompt.
        assert bash_mod._APPROVALS.was_refused(f"mytool > {marker}",
                                               bash_mod._session_base(base))
