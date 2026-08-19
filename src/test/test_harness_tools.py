"""Tests for src/model/serving/harness.py — the model-callable harness tools."""

import asyncio
import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.harness import (COMPACTION_NOTICE, MAX_NOTE_CHARS, MAX_NOTES,
                                   NOTES_SUBDIR, HarnessTools)
from model.serving.chat import _execute_tool, _parse_tool_call_from_content
from model.serving.interpreter import shutdown_session


def _harness(tmp_path, **kwargs):
    return HarnessTools(data_path=str(tmp_path), **kwargs)


# ── what the run offers ─────────────────────────────────────────────────────

class TestAvailability:
    def test_the_full_set_with_a_data_path(self, tmp_path):
        assert set(_harness(tmp_path).names) == {
            "context_status", "note_write", "note_read",
            "result_read", "result_grep"}

    def test_notes_withheld_without_a_data_path(self):
        """A tool that could only fail is worse than one that is not offered."""
        harness = HarnessTools(data_path="")
        assert harness.names == ("context_status",)
        assert not harness.handles("note_write")
        assert harness.dispatch("note_write", {"key": "k", "text": "t"}) is None

    def test_disabled_offers_nothing(self, tmp_path):
        harness = _harness(tmp_path, enabled=False)
        assert harness.names == ()
        assert harness.tool_items() == []
        assert harness.dispatch("context_status", {}) is None

    def test_tool_items_are_api_shaped(self, tmp_path):
        for item in _harness(tmp_path).tool_items():
            assert item["type"] == "function"
            fn = item["function"]
            assert {"name", "description", "parameters"} <= set(fn)
            # Phase 1's validator reads both, and the model needs them to not
            # invent a parameter in the first place.
            assert fn["parameters"]["additionalProperties"] is False
            assert isinstance(fn["parameters"]["required"], list)


# ── context_status ──────────────────────────────────────────────────────────

class TestContextStatus:
    def test_reports_the_loops_real_numbers(self, tmp_path):
        harness = _harness(tmp_path, max_context_tokens=100_000)
        harness.observe(prompt_tokens=42_000, turns=7, tools_called=12, compactions=1)
        status = json.loads(harness.context_status())
        assert status["used_tokens"] == 42_000
        assert status["max_tokens"] == 100_000
        assert status["pct_used"] == 42
        assert status["turns_taken"] == 7
        assert status["tool_calls_made"] == 12
        assert status["compactions"] == 1

    def test_observe_leaves_unmentioned_counters_alone(self, tmp_path):
        harness = _harness(tmp_path)
        harness.observe(turns=3, tools_called=5)
        harness.observe(turns=4)
        assert (harness.turns, harness.tools_called) == (4, 5)

    def test_no_measurement_yet_is_explained(self, tmp_path):
        """Bare nulls read as an error and cost a second call to rediscover."""
        status = json.loads(_harness(tmp_path, max_context_tokens=1000).context_status())
        assert status["used_tokens"] is None
        assert "next turn" in status["detail"]

    def test_unknown_window_size_is_explained(self, tmp_path):
        harness = _harness(tmp_path)
        harness.observe(prompt_tokens=500)
        status = json.loads(harness.context_status())
        assert status["used_tokens"] == 500
        assert status["max_tokens"] is None and status["pct_used"] is None
        assert "context window size" in status["detail"]

    def test_a_filling_context_says_to_write_things_down(self, tmp_path):
        harness = _harness(tmp_path, max_context_tokens=1000)
        harness.observe(prompt_tokens=850)
        status = json.loads(harness.context_status())
        assert status["pct_used"] == 85
        assert "note_write" in status["detail"]

    def test_lists_saved_notes(self, tmp_path):
        harness = _harness(tmp_path)
        harness.note_write("findings", "the port is 8001")
        harness.note_write("todo", "write the report")
        assert json.loads(harness.context_status())["notes_saved"] == ["findings", "todo"]


# ── notes ───────────────────────────────────────────────────────────────────

class TestNotes:
    def test_write_then_read_round_trips(self, tmp_path):
        harness = _harness(tmp_path)
        harness.note_write("findings", "the answer is 42")
        assert harness.note_read("findings") == "the answer is 42"

    def test_notes_land_under_data_path(self, tmp_path):
        """Not $HOME, and not loose in the working directory the task uses."""
        harness = _harness(tmp_path)
        harness.note_write("findings", "x")
        assert (tmp_path / NOTES_SUBDIR / "findings.md").read_text() == "x"

    def test_write_reports_what_it_did(self, tmp_path):
        harness = _harness(tmp_path)
        first = json.loads(harness.note_write("k", "one"))
        assert first["status"] == "saved" and first["chars"] == 3
        assert first["replaced"] is False
        assert json.loads(harness.note_write("k", "two"))["replaced"] is True
        assert harness.note_read("k") == "two"

    def test_missing_note_names_the_ones_that_exist(self, tmp_path):
        harness = _harness(tmp_path)
        harness.note_write("findings", "x")
        result = harness.note_read("plan")
        assert result.startswith("Error:")
        assert "findings" in result

    def test_missing_note_with_nothing_saved(self, tmp_path):
        assert "none saved yet" in _harness(tmp_path).note_read("plan")

    @pytest.mark.parametrize("key", [
        "../escape", "sub/dir", "..", "/etc/passwd", "", "-leading", "a" * 65,
        "spaced key",
    ])
    def test_unusable_keys_are_refused(self, tmp_path, key):
        harness = _harness(tmp_path)
        assert harness.note_write(key, "x").startswith("Error:")
        assert harness.note_read(key).startswith("Error:")

    def test_traversal_writes_nothing_outside_the_jail(self, tmp_path):
        """data_path is a session isolation boundary; a note cannot cross it."""
        outside = tmp_path.parent / "outside.md"
        harness = _harness(tmp_path / "session")
        harness.note_write("../../outside", "leaked")
        assert not outside.exists()

    def test_oversized_note_is_refused_not_truncated(self, tmp_path):
        harness = _harness(tmp_path)
        result = harness.note_write("dump", "x" * (MAX_NOTE_CHARS + 1))
        assert result.startswith("Error:")
        assert harness.note_keys() == []

    def test_note_count_is_bounded(self, tmp_path):
        harness = _harness(tmp_path)
        for i in range(MAX_NOTES):
            assert not harness.note_write(f"k{i}", "x").startswith("Error:")
        refused = harness.note_write("one_too_many", "x")
        assert refused.startswith("Error:")
        # An overwrite is not a new note and stays allowed at the limit.
        assert not harness.note_write("k0", "updated").startswith("Error:")

    def test_note_keys_survive_a_fresh_object(self, tmp_path):
        """The point of notes: they outlive whatever is holding them."""
        _harness(tmp_path).note_write("findings", "durable")
        assert _harness(tmp_path).note_read("findings") == "durable"


# ── dispatch ────────────────────────────────────────────────────────────────

class TestDispatch:
    def test_unknown_name_is_not_ours(self, tmp_path):
        assert _harness(tmp_path).dispatch("read_file", {"path": "x"}) is None

    def test_arguments_are_validated(self, tmp_path):
        result = _harness(tmp_path).dispatch("note_write", {"key": "k"})
        assert result.startswith("Error:")
        assert "text" in result

    def test_unknown_parameters_are_refused_by_name(self, tmp_path):
        result = _harness(tmp_path).dispatch(
            "note_read", {"key": "k", "offset": 10})
        assert "offset" in result and "key" in result

    def test_arguments_are_coerced_like_the_mcp_path(self, tmp_path):
        """A string where an object was asked for is a round trip not worth paying."""
        harness = _harness(tmp_path)
        assert not harness.dispatch("note_write", {"key": "n", "text": "5"}).startswith("Error:")
        assert harness.note_read("n") == "5"


# ── dispatch through _execute_tool ──────────────────────────────────────────

class TestExecuteToolIntegration:
    def _run(self, harness, name, args, registry=None, history=None):
        messages: list = []
        asyncio.run(_execute_tool(
            name, args, "call_1", registry, timeout=5, data_path=harness.data_path,
            chat_ui=None, verbose=False, messages=messages,
            tool_call_history=history if history is not None else [],
            max_repeated=30, harness=harness,
        ))
        return messages

    def test_harness_call_never_reaches_the_registry(self, tmp_path):
        registry = MagicMock()
        registry.tools = {"read_file"}
        messages = self._run(_harness(tmp_path), "context_status", {}, registry)
        registry.__getitem__.assert_not_called()
        assert json.loads(messages[-1]["content"])["turns_taken"] == 0

    def test_result_is_appended_as_a_tool_message(self, tmp_path):
        harness = _harness(tmp_path)
        messages = self._run(harness, "note_write", {"key": "k", "text": "saved"})
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["name"] == "note_write"
        assert messages[-1]["tool_call_id"] == "call_1"
        assert harness.note_read("k") == "saved"

    def test_works_without_a_registry_at_all(self, tmp_path):
        messages = self._run(_harness(tmp_path), "context_status", {}, registry=None)
        assert "turns_taken" in messages[-1]["content"]

    def test_repeated_calls_still_trip_the_guard(self, tmp_path):
        """A model looping on context_status is as stuck as one looping on bash."""
        harness = _harness(tmp_path)
        history: list = []
        for _ in range(29):
            self._run(harness, "context_status", {}, history=history)
        bail = asyncio.run(_execute_tool(
            "context_status", {}, "call_x", None, timeout=5,
            data_path=harness.data_path, chat_ui=None, verbose=False,
            messages=[], tool_call_history=history, max_repeated=30,
            harness=harness,
        ))
        assert bail is not None

    def test_a_harness_call_is_recorded_in_the_tool_log(self, tmp_path):
        """Trajectories must show these calls; they are turns the model spent."""
        harness = _harness(tmp_path)
        tool_log: list = []
        asyncio.run(_execute_tool(
            "note_write", {"key": "k", "text": "v"}, "call_1", None, timeout=5,
            data_path=harness.data_path, chat_ui=None, verbose=False, messages=[],
            tool_call_history=[], max_repeated=30, tool_log=tool_log, harness=harness,
        ))
        assert len(tool_log) == 1 and tool_log[0]["name"] == "note_write"

    def test_a_refusal_is_logged_as_a_failed_call(self, tmp_path):
        harness = _harness(tmp_path)
        tool_log: list = []
        asyncio.run(_execute_tool(
            "note_read", {"key": "missing"}, "call_1", None, timeout=5,
            data_path=harness.data_path, chat_ui=None, verbose=False, messages=[],
            tool_call_history=[], max_repeated=30, tool_log=tool_log, harness=harness,
        ))
        assert tool_log[0]["ok"] is False

    def test_ui_is_told_the_call_happened(self, tmp_path):
        chat_ui = MagicMock()
        messages: list = []
        asyncio.run(_execute_tool(
            "context_status", {}, "call_1", None, timeout=5,
            data_path=str(tmp_path), chat_ui=chat_ui, verbose=False,
            messages=messages, tool_call_history=[], max_repeated=30,
            is_structured=True, harness=_harness(tmp_path),
        ))
        chat_ui.add_tool_call.assert_called_once()
        chat_ui.add_tool_result.assert_called_once()
        chat_ui.stop_tool_spinner.assert_called_once()


# ── raw JSON tool calls ─────────────────────────────────────────────────────

class TestRawToolCallParsing:
    def test_a_harness_call_written_as_content_is_recognized(self, tmp_path):
        """Otherwise it is prose, and the JSON is handed to the user as an answer."""
        registry = MagicMock()
        registry.tools = {"read_file"}
        content = '{"name": "note_write", "arguments": {"key": "k", "text": "v"}}'
        parsed = _parse_tool_call_from_content(content, registry, _harness(tmp_path))
        assert parsed["name"] == "note_write"

    def test_still_unrecognized_when_the_harness_is_absent(self, tmp_path):
        registry = MagicMock()
        registry.tools = {"read_file"}
        content = '{"name": "note_write", "arguments": {"key": "k", "text": "v"}}'
        assert _parse_tool_call_from_content(content, registry) is None

    def test_withheld_note_tools_are_not_recognized(self):
        registry = MagicMock()
        registry.tools = {"read_file"}
        content = '{"name": "note_write", "arguments": {"key": "k", "text": "v"}}'
        assert _parse_tool_call_from_content(content, registry,
                                             HarnessTools(data_path="")) is None


# ── the result tools ────────────────────────────────────────────────────────

class TestResultToolsOnTheHarness:
    """Phase 4's two tools, dispatched in-process exactly like Phase 2's."""

    @staticmethod
    def _stored(tmp_path, text=None, tool="local_search"):
        harness = _harness(tmp_path)
        text = text or "".join(f"line {i} NEEDLE\n" if i == 500
                               else f"line {i}\n" for i in range(4000))
        preview = harness.results.put(tool, text)
        return harness, preview, text

    def test_withheld_when_the_store_is_off(self, tmp_path):
        """Nothing is ever stored, so a handle could never resolve — two
        schemas on every request buying nothing."""
        names = set(_harness(tmp_path, result_store=False).names)
        assert "result_read" not in names and "result_grep" not in names
        assert "note_write" in names

    def test_withheld_without_a_data_path(self):
        assert set(HarnessTools().names) == {"context_status"}

    def test_read_dispatches_to_the_store(self, tmp_path):
        harness, preview, text = self._stored(tmp_path)
        out = harness.dispatch("result_read", {"handle": "0001", "offset": 0,
                                               "limit": 100})
        assert not out.startswith("Error:")
        assert "line 0" in out

    def test_read_defaults_are_applied(self, tmp_path):
        harness, _, _ = self._stored(tmp_path)
        out = harness.dispatch("result_read", {"handle": "0001"})
        assert "showing 0–4,000" in out

    def test_grep_dispatches_to_the_store(self, tmp_path):
        harness, _, _ = self._stored(tmp_path)
        out = harness.dispatch("result_grep", {"handle": "0001", "pattern": "NEEDLE"})
        assert "NEEDLE" in out

    def test_arguments_are_validated_against_the_schema(self, tmp_path):
        """These bypass _execute_tool's checks by intercepting ahead of them."""
        harness, _, _ = self._stored(tmp_path)
        out = harness.dispatch("result_read", {})
        assert out.startswith("Error:")
        assert "handle" in out

    def test_a_string_offset_is_coerced_like_the_mcp_path(self, tmp_path):
        harness, _, _ = self._stored(tmp_path)
        out = harness.dispatch("result_read", {"handle": "0001", "offset": "100"})
        assert not out.startswith("Error:")
        assert "showing 100–" in out

    def test_a_traversal_handle_is_refused_through_dispatch(self, tmp_path):
        harness, _, _ = self._stored(tmp_path)
        out = harness.dispatch("result_read", {"handle": "../../etc/passwd"})
        assert out.startswith("Error:")

    def test_context_status_lists_the_handles(self, tmp_path):
        """Discovery without a sixth tool — the same call that reports how full
        the window is reports what can be read back."""
        harness, _, _ = self._stored(tmp_path)
        status = json.loads(harness.context_status())
        assert status["results_stored"] == [
            {"handle": "0001", "tool": "local_search",
             "chars": status["results_stored"][0]["chars"]}]

    def test_context_status_reports_nothing_stored_as_empty(self, tmp_path):
        assert json.loads(_harness(tmp_path).context_status())["results_stored"] == []

    def test_the_compaction_notice_mentions_stored_results(self):
        """A summary can drop the handle lines; the notice says where to get
        them back."""
        assert "result_read" in COMPACTION_NOTICE
        assert "context_status" in COMPACTION_NOTICE


# ── code as action ──────────────────────────────────────────────────────────

class TestRunCodeOnTheHarness:
    """Phase 5's tool. Off unless the deployment asked for it."""

    @staticmethod
    def _registry(answer='[{"title": "Q3"}]'):
        from unittest.mock import AsyncMock
        handler = AsyncMock(return_value=answer)
        registry = MagicMock()
        registry.tools = {"local_search"}
        registry.get_tool_items.return_value = [
            {"type": "function",
             "function": {"name": "local_search", "description": "search",
                          "parameters": {"type": "object",
                                         "properties": {"query": {"type": "string"},
                                                        "data_path": {"type": "string"},
                                                        "session_id": {"type": "string"}},
                                         "required": ["query"]}}}]
        registry.tool_accepts_param.side_effect = (
            lambda tool, param: param in ("data_path", "session_id"))
        registry.__getitem__ = MagicMock(return_value=handler)
        registry.handler = handler
        return registry

    def _harness_with_code(self, tmp_path, session_id="s1", **kwargs):
        return HarnessTools(data_path=str(tmp_path), code_execution=True,
                            session_id=session_id,
                            tool_registry=self._registry(), **kwargs)

    # ── availability ────────────────────────────────────────────────────────

    def test_off_by_default(self, tmp_path):
        assert "run_code" not in _harness(tmp_path).names

    def test_offered_when_switched_on(self, tmp_path):
        assert "run_code" in self._harness_with_code(tmp_path).names

    def test_offered_without_a_data_path(self):
        """An interpreter needs somewhere to run, not somewhere to write."""
        harness = HarnessTools(code_execution=True, session_id="s")
        assert "run_code" in harness.names
        assert "note_write" not in harness.names

    def test_withdrawn_with_the_whole_harness(self, tmp_path):
        harness = HarnessTools(data_path=str(tmp_path), enabled=False,
                               code_execution=True)
        assert harness.names == ()

    # ── running ─────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_it_returns_what_the_code_printed(self, tmp_path):
        harness = self._harness_with_code(tmp_path, session_id="run-1")
        try:
            out = await harness.adispatch("run_code", {"code": "print(6 * 7)"})
            assert "42" in out
            assert out.startswith("[run_code · ok ·")
        finally:
            await shutdown_session("run-1")

    @pytest.mark.asyncio
    async def test_tools_are_callable_from_inside(self, tmp_path):
        harness = self._harness_with_code(tmp_path, session_id="run-2")
        try:
            out = await harness.adispatch(
                "run_code", {"code": 'print(local_search("Q3")[0]["title"])'})
            assert "Q3" in out
        finally:
            await shutdown_session("run-2")

    @pytest.mark.asyncio
    async def test_the_harness_values_reach_the_handler(self, tmp_path):
        """The acceptance criterion: whatever the code passes, the handler is
        called with the harness's session_id and data_path."""
        registry = self._registry()
        harness = HarnessTools(data_path=str(tmp_path), code_execution=True,
                               session_id="the-real-session", tool_registry=registry)
        try:
            await harness.adispatch("run_code", {
                "code": 'call_tool("local_search", query="Q3", '
                        'data_path="/etc", session_id="somebody-else")'})
            kwargs = registry.handler.await_args.kwargs
            assert kwargs["session_id"] == "the-real-session"
            assert kwargs["data_path"] == str(tmp_path)
            assert kwargs["query"] == "Q3"
        finally:
            await shutdown_session("the-real-session")

    @pytest.mark.asyncio
    async def test_an_empty_print_is_explained(self, tmp_path):
        """An empty result reads as a failure, and the model re-runs the same
        block to find out what happened."""
        harness = self._harness_with_code(tmp_path, session_id="run-3")
        try:
            out = await harness.adispatch("run_code", {"code": "x = 1"})
            assert "printed nothing" in out
            assert "still there for the next call" in out
        finally:
            await shutdown_session("run-3")

    @pytest.mark.asyncio
    async def test_an_exception_comes_back_as_a_result(self, tmp_path):
        harness = self._harness_with_code(tmp_path, session_id="run-4")
        try:
            out = await harness.adispatch("run_code", {"code": "1 / 0"})
            assert "raised" in out
            assert "ZeroDivisionError" in out
        finally:
            await shutdown_session("run-4")

    @pytest.mark.asyncio
    async def test_a_huge_print_goes_through_the_result_store(self, tmp_path):
        """Why Phase 4 came first: a print that turns out to be a whole file is
        addressed, not pasted."""
        from model.serving.results import handle_of
        harness = self._harness_with_code(tmp_path, session_id="run-5")
        try:
            out = await harness.adispatch("run_code", {"code": "print('x' * 50000)"})
            assert handle_of(out) is not None
            assert len(out) < 7000
        finally:
            await shutdown_session("run-5")

    # ── validation and misuse ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_a_missing_code_argument_is_refused(self, tmp_path):
        harness = self._harness_with_code(tmp_path, session_id="run-6")
        out = await harness.adispatch("run_code", {})
        assert out.startswith("Error:")
        assert "code" in out

    @pytest.mark.asyncio
    async def test_blank_code_is_refused_without_starting_anything(self, tmp_path):
        harness = self._harness_with_code(tmp_path, session_id="run-7")
        out = await harness.adispatch("run_code", {"code": "   "})
        assert out.startswith("Error:")

    def test_the_sync_path_refuses_rather_than_pretending(self, tmp_path):
        """A caller that has not moved to adispatch gets told, not a
        half-answer."""
        harness = self._harness_with_code(tmp_path)
        assert harness.dispatch("run_code", {"code": "print(1)"}).startswith("Error:")

    @pytest.mark.asyncio
    async def test_adispatch_passes_the_sync_tools_straight_through(self, tmp_path):
        harness = self._harness_with_code(tmp_path)
        status = json.loads(await harness.adispatch("context_status", {}))
        assert status["turns_taken"] == 0
        assert await harness.adispatch("not_a_harness_tool", {}) is None
