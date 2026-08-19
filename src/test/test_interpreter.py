"""Tests for src/model/serving/interpreter.py — code as action.

These start real child Python processes.  That is the point: the thing being
tested is a subprocess protocol, and a mocked one would test the mock.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.interpreter import (INJECTED_PARAMS, PythonInterpreter,
                                       _as_data, bindings_for, get_interpreter,
                                       shutdown_all, shutdown_session)


def _item(name, properties=None, required=None, description="d"):
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object",
                                        "properties": properties or {},
                                        "required": required or []}}}


SEARCH = _item("local_search",
               {"query": {"type": "string"}, "top_k": {"type": "integer"},
                "data_path": {"type": "string"}, "session_id": {"type": "string"}},
               ["query"])


@pytest.fixture
async def interp():
    """An interpreter with one tool, stopped however the test ends."""
    calls = []

    async def dispatch(name, kwargs):
        calls.append((name, dict(kwargs)))
        if name == "local_search":
            return '[{"title": "Q3 report", "path": "/d/q3.pdf"}, {"title": "Q2"}]'
        if name == "boom":
            raise RuntimeError("server said no")
        if name == "slow":
            await asyncio.sleep(30)
        if name not in ("local_search", "boom", "slow"):
            raise LookupError(f"no tool named {name!r}")
        return f"ran {name}"

    it = PythonInterpreter(session_id="s1", tool_items=[SEARCH, _item("boom"),
                                                        _item("slow")],
                           dispatch=dispatch)
    it.calls = calls
    try:
        yield it
    finally:
        await it.stop()


# ── generating the bindings ─────────────────────────────────────────────────

class TestBindings:
    def test_a_tool_becomes_a_signature(self):
        assert bindings_for([SEARCH]) == [{
            "name": "local_search",
            "params": [{"name": "query", "required": True},
                       {"name": "top_k", "required": False}],
            "doc": "d",
        }]

    def test_the_injected_parameters_are_dropped(self):
        """Not hidden, not ignored — absent. There is nothing to override."""
        names = [p["name"] for p in bindings_for([SEARCH])[0]["params"]]
        for injected in INJECTED_PARAMS:
            assert injected not in names

    def test_required_parameters_come_first(self):
        """Python will not accept a non-default parameter after a defaulted
        one, and a schema is under no obligation to order them."""
        item = _item("t", {"opt": {"type": "string"}, "req": {"type": "string"}},
                     ["req"])
        assert [p["name"] for p in bindings_for([item])[0]["params"]] == ["req", "opt"]

    def test_a_tool_that_is_not_an_identifier_is_skipped(self):
        assert bindings_for([_item("not-a-name")]) == []
        assert bindings_for([_item("class")]) == []
        assert bindings_for([_item("print")]) == []

    def test_a_parameter_that_is_not_an_identifier_is_skipped(self):
        item = _item("t", {"ok": {"type": "string"}, "not ok": {"type": "string"}})
        assert [p["name"] for p in bindings_for([item])[0]["params"]] == ["ok"]

    def test_junk_is_survived(self):
        assert bindings_for(None) == []
        assert bindings_for(["not a dict", {}, {"function": "not a dict"}]) == []

    def test_a_tool_with_no_schema_still_binds(self):
        assert bindings_for([{"type": "function", "function": {"name": "ping"}}]) == [
            {"name": "ping", "params": [], "doc": ""}]


# ── tool answers, as data ───────────────────────────────────────────────────

class TestAsData:
    def test_json_objects_and_arrays_are_parsed(self):
        assert _as_data('{"a": 1}') == {"a": 1}
        assert _as_data('[1, 2]') == [1, 2]

    def test_a_scalar_stays_a_string(self):
        """A tool answering "42" means the string; quietly making it an int is
        the kind of helpfulness that surfaces later as a type error."""
        assert _as_data("42") == "42"
        assert _as_data("true") == "true"

    def test_text_stays_text(self):
        assert _as_data("just some output") == "just some output"

    def test_broken_json_stays_a_string(self):
        assert _as_data('{"a": ') == '{"a": '

    def test_non_strings_pass_through(self):
        assert _as_data(None) is None
        assert _as_data(7) == 7


# ── the interpreter itself ──────────────────────────────────────────────────

class TestRunning:
    @pytest.mark.asyncio
    async def test_only_what_is_printed_comes_back(self, interp):
        out = await interp.run("x = 41\nprint('hello', x + 1)")
        assert out["ok"] is True
        assert out["stdout"] == "hello 42\n"

    @pytest.mark.asyncio
    async def test_the_namespace_persists_across_calls(self, interp):
        """The acceptance criterion, and the whole reason for a child process
        that stays alive rather than one per call."""
        await interp.run("total = 41")
        out = await interp.run("print(total + 1)")
        assert out["stdout"] == "42\n"

    @pytest.mark.asyncio
    async def test_a_trailing_expression_is_echoed(self, interp):
        """A model writing `local_search("x")` and seeing nothing is a wasted
        turn."""
        out = await interp.run("2 + 2")
        assert out["stdout"] == "4\n"

    @pytest.mark.asyncio
    async def test_a_trailing_none_is_not_echoed(self, interp):
        assert (await interp.run("x = 1"))["stdout"] == ""

    @pytest.mark.asyncio
    async def test_an_exception_is_a_result_not_a_failure(self, interp):
        out = await interp.run("1 / 0")
        assert out["ok"] is False
        assert "ZeroDivisionError" in out["error"]

    @pytest.mark.asyncio
    async def test_the_traceback_names_the_model_s_own_line(self, interp):
        out = await interp.run("def f():\n    raise ValueError('nope')\nf()")
        assert '"<run_code>"' in out["error"]
        # Our plumbing is noise the model cannot act on.
        assert "<bindings>" not in out["error"]
        assert "/ast.py" not in out["error"]

    @pytest.mark.asyncio
    async def test_a_syntax_error_points_at_the_line(self, interp):
        out = await interp.run("def (")
        assert out["ok"] is False
        assert "SyntaxError" in out["error"]
        assert "/ast.py" not in out["error"]

    @pytest.mark.asyncio
    async def test_output_before_an_exception_is_kept(self, interp):
        out = await interp.run("print('got this far')\n1 / 0")
        assert out["stdout"] == "got this far\n"
        assert "ZeroDivisionError" in out["error"]

    @pytest.mark.asyncio
    async def test_stdout_is_capped_in_the_child(self):
        it = PythonInterpreter(session_id="cap", max_stdout=2000)
        try:
            out = await it.run("print('x' * 50000)")
            assert len(out["stdout"]) < 2200
            assert "capped" in out["stdout"]
        finally:
            await it.stop()


# ── calling tools from inside ───────────────────────────────────────────────

class TestToolsFromCode:
    @pytest.mark.asyncio
    async def test_a_tool_is_a_function_of_the_same_name(self, interp):
        out = await interp.run('print(local_search("Q3")[0]["title"])')
        assert out["stdout"] == "Q3 report\n"
        assert interp.calls[0][0] == "local_search"

    @pytest.mark.asyncio
    async def test_json_answers_arrive_as_objects(self, interp):
        out = await interp.run('hits = local_search("Q3")\nprint([h.title for h in hits'
                               ' if "title" in h])')
        assert out["stdout"] == "['Q3 report', 'Q2']\n"

    @pytest.mark.asyncio
    async def test_several_steps_in_one_call(self, interp):
        """The point of the phase: this is one turn, not four."""
        out = await interp.run(
            'hits = local_search("Q3")\n'
            'titles = [h["title"] for h in hits]\n'
            'joined = ", ".join(titles)\n'
            'print(joined.upper())')
        assert out["stdout"] == "Q3 REPORT, Q2\n"
        assert len(interp.calls) == 1

    @pytest.mark.asyncio
    async def test_optional_arguments_left_out_are_not_sent(self, interp):
        """A server that distinguishes absent from null should see absent."""
        await interp.run('local_search("Q3")')
        assert interp.calls[-1][1] == {"query": "Q3"}

    @pytest.mark.asyncio
    async def test_optional_arguments_given_are_sent(self, interp):
        await interp.run('local_search("Q3", top_k=5)')
        assert interp.calls[-1][1] == {"query": "Q3", "top_k": 5}

    @pytest.mark.asyncio
    async def test_a_failing_tool_raises_a_catchable_error(self, interp):
        """A script looping over ten documents should survive one of them."""
        out = await interp.run(
            'try:\n    boom()\nexcept ToolError as e:\n    print("caught:", e)')
        assert out["ok"] is True
        assert "caught: boom: server said no" in out["stdout"]

    @pytest.mark.asyncio
    async def test_an_unknown_tool_is_an_error_inside_the_code(self, interp):
        out = await interp.run('call_tool("nope")')
        assert out["ok"] is False
        assert "ToolError" in out["error"]

    @pytest.mark.asyncio
    async def test_a_slow_tool_is_bounded_separately_from_the_block(self, interp):
        """Reported as a slow tool, not as the whole block dying with nothing
        to show for it."""
        interp.tool_timeout = 0.5
        out = await interp.run('try:\n    slow()\nexcept ToolError as e:\n'
                               '    print("caught:", e)', timeout=20)
        assert out["ok"] is True
        assert "timed out" in out["stdout"]

    @pytest.mark.asyncio
    async def test_call_tool_reaches_names_that_are_not_identifiers(self):
        seen = []

        async def dispatch(name, kwargs):
            seen.append(name)
            return "ok"

        it = PythonInterpreter(tool_items=[_item("not-a-name")], dispatch=dispatch)
        try:
            out = await it.run('print(call_tool("not-a-name", x=1))')
            assert out["stdout"] == "ok\n"
            assert seen == ["not-a-name"]
        finally:
            await it.stop()


# ── the trust boundary ──────────────────────────────────────────────────────

class TestInjectedParametersCannotBeOverridden:
    """data_path and session_id are the session isolation boundary.  Code
    inside the interpreter is model input, so it does not get a vote."""

    @pytest.mark.asyncio
    async def test_the_signature_has_no_place_to_put_them(self, interp):
        out = await interp.run("import inspect; print(inspect.signature(local_search))")
        assert out["stdout"].strip() == "(query, top_k=None)"

    @pytest.mark.asyncio
    async def test_passing_them_as_keywords_is_a_type_error(self, interp):
        out = await interp.run('local_search("Q3", data_path="/etc")')
        assert out["ok"] is False
        assert "unexpected keyword argument" in out["error"]

    @pytest.mark.asyncio
    async def test_call_tool_cannot_smuggle_them_either(self, interp):
        """The second check, for the same reason a jailed path is resolved
        after a regex has already made it safe."""
        await interp.run('call_tool("local_search", query="Q3", '
                         'data_path="/etc", session_id="other")')
        name, kwargs = interp.calls[-1]
        assert kwargs == {"query": "Q3"}


# ── recovering from a wedged interpreter ────────────────────────────────────

class TestTimeout:
    @pytest.mark.asyncio
    async def test_a_wedged_block_is_killed(self, interp):
        out = await interp.run("while True: pass", timeout=2)
        assert out["timed_out"] is True
        assert out["ok"] is False
        assert "did not finish" in out["error"]

    @pytest.mark.asyncio
    async def test_the_session_recovers_afterwards(self, interp):
        await interp.run("keep = 'me'", timeout=10)
        await interp.run("while True: pass", timeout=2)
        out = await interp.run("print('still here')")
        assert out["ok"] is True
        assert out["stdout"] == "still here\n"

    @pytest.mark.asyncio
    async def test_the_model_is_told_the_variables_are_gone(self, interp):
        """A restart loses the namespace. Saying so is the difference between
        a model that re-derives and one that reports a mystery."""
        await interp.run("keep = 'me'", timeout=10)
        out = await interp.run("while True: pass", timeout=2)
        assert "variables from earlier calls are gone" in out["error"]
        assert (await interp.run("print('keep' in dir())"))["stdout"] == "False\n"


# ── one interpreter per session ─────────────────────────────────────────────

class TestSessionScope:
    @pytest.mark.asyncio
    async def test_the_same_session_gets_the_same_interpreter(self):
        try:
            a = get_interpreter("sess-a", tool_items=[], dispatch=None)
            await a.run("shared = 1")
            b = get_interpreter("sess-a", tool_items=[], dispatch=None)
            assert b is a
            assert (await b.run("print(shared)"))["stdout"] == "1\n"
        finally:
            await shutdown_all()

    @pytest.mark.asyncio
    async def test_the_namespace_does_not_leak_across_sessions(self):
        """The acceptance criterion. Two sessions, two processes, no shared
        anything."""
        try:
            a = get_interpreter("sess-a", tool_items=[], dispatch=None)
            await a.run("secret = 'session a only'")
            b = get_interpreter("sess-b", tool_items=[], dispatch=None)
            assert b is not a
            out = await b.run("print('secret' in dir())")
            assert out["stdout"] == "False\n"
        finally:
            await shutdown_all()

    @pytest.mark.asyncio
    async def test_shutdown_forgets_the_session(self):
        try:
            a = get_interpreter("sess-a", tool_items=[], dispatch=None)
            await a.run("x = 1")
            assert a.alive
            await shutdown_session("sess-a")
            assert not a.alive
            assert get_interpreter("sess-a", tool_items=[], dispatch=None) is not a
        finally:
            await shutdown_all()

    @pytest.mark.asyncio
    async def test_the_least_recently_used_is_evicted_past_the_cap(self):
        from model.serving.interpreter import evict_stale
        try:
            first = get_interpreter("old", tool_items=[], dispatch=None)
            await first.run("x = 1")
            for n in range(3):
                get_interpreter(f"newer-{n}", tool_items=[], dispatch=None)
            await evict_stale(limit=2)
            assert not first.alive
        finally:
            await shutdown_all()

    @pytest.mark.asyncio
    async def test_a_later_task_keeps_the_namespace_but_refreshes_the_tools(self):
        """Tools can change between tasks; that is no reason to throw away what
        the session has computed."""
        try:
            a = get_interpreter("sess-a", tool_items=[], dispatch=None)
            await a.run("carried = 'over'")
            b = get_interpreter("sess-a", tool_items=[SEARCH], dispatch=None)
            assert b is a
            assert b.tool_items == [SEARCH]
            assert (await b.run("print(carried)"))["stdout"] == "over\n"
        finally:
            await shutdown_all()


# ── the process itself ──────────────────────────────────────────────────────

class TestProcess:
    @pytest.mark.asyncio
    async def test_it_runs_in_a_child_not_in_the_harness(self, interp):
        out = await interp.run("import os; print(os.getpid())")
        assert int(out["stdout"].strip()) != os.getpid()

    @pytest.mark.asyncio
    async def test_it_starts_in_the_session_directory(self, tmp_path):
        it = PythonInterpreter(session_id="s", data_path=str(tmp_path))
        try:
            out = await it.run("import os; print(os.getcwd())")
            assert os.path.realpath(out["stdout"].strip()) == os.path.realpath(tmp_path)
        finally:
            await it.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, interp):
        await interp.run("x = 1")
        await interp.stop()
        await interp.stop()
        assert not interp.alive

    @pytest.mark.asyncio
    async def test_stray_output_on_the_protocol_stream_is_survived(self, interp):
        """Code that spawns a subprocess inheriting fd 1 writes into the
        protocol. Losing the whole block to someone else's echo would be a poor
        trade."""
        out = await interp.run(
            "import os, subprocess, sys\n"
            "subprocess.run([sys.executable, '-c', 'print(\"noise\")'])\n"
            "print('mine')")
        assert out["ok"] is True
        assert "mine" in out["stdout"]
