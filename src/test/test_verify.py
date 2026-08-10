"""Tests for src/model/serving/verify.py — the fact-check that runs after an
answer is written, and its wiring into the chat loop."""

import asyncio
import json
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.verify import (CORRECTION_PREFIX, NOTE_PREFIX, append_note,
                                  build_revision_messages, build_verdict_messages,
                                  evidence_digest, needs_verification,
                                  parse_verdict, revision_note, verify_answer)
from model.serving.chat import _NO_TEMPLATE_KWARGS, chat, reset_endpoint_caches


# ── needs_verification ──────────────────────────────────────────────────────

class TestNeedsVerification:
    def test_numeric_claim_is_checked(self):
        assert needs_verification(
            "The library was released in 2019 and the current version is 4.2, "
            "which dropped support for the older runtime.")

    def test_attribution_is_checked(self):
        assert needs_verification(
            "According to the filing, the company is based in Delaware and was "
            "founded by the same team that authored the original specification.")

    def test_absolute_claim_is_checked(self):
        assert needs_verification(
            "It is the only implementation that never allocates on the hot path, "
            "which is why it remains the fastest option available for this use.")

    def test_url_is_checked(self):
        assert needs_verification(
            "The full changelog lives at https://example.com/changelog and covers "
            "every release since the project moved to its current home.")

    def test_claim_free_prose_is_skipped(self):
        """No assertion about the world, so there is nothing to check and the
        call is pure latency."""
        assert not needs_verification(
            "I've saved the file for you. Let me know if you'd like it formatted "
            "differently, or if you want me to walk through any of it with you.")

    def test_short_reply_is_skipped(self):
        assert not needs_verification("Done — 3 files updated.")

    def test_self_report_is_skipped(self):
        """An account of the run's own failure is true by construction; checking
        it means asking a model to fact-check the agent's own apology."""
        assert not needs_verification(
            "I couldn't find any file matching that pattern in the 4 directories "
            "I searched, so there is nothing to summarize here for you.")

    def test_empty_answer_is_skipped(self):
        assert not needs_verification("")
        assert not needs_verification(None)


# ── evidence_digest ─────────────────────────────────────────────────────────

class TestEvidenceDigest:
    def test_tool_results_and_user_turns_are_evidence(self):
        digest = evidence_digest([
            {"role": "user", "content": "What is the revenue?"},
            {"role": "tool", "name": "search", "content": "Revenue was 3.1M."},
        ])
        assert "Revenue was 3.1M." in digest
        assert "[search]" in digest
        assert "What is the revenue?" in digest

    def test_assistant_turns_are_not_evidence(self):
        """An answer that cites an earlier sentence of its own is the failure
        being looked for, not corroboration of it."""
        digest = evidence_digest([
            {"role": "assistant", "content": "Revenue was 4.2M, I believe."},
            {"role": "tool", "name": "search", "content": "Revenue was 3.1M."},
        ])
        assert "4.2M" not in digest

    def test_newest_evidence_survives_the_budget(self):
        messages = [{"role": "tool", "name": "search", "content": f"item {i} " + "x" * 500}
                    for i in range(40)]
        digest = evidence_digest(messages, max_chars=1200)
        assert "item 39" in digest
        assert "item 0 " not in digest
        assert len(digest) <= 1400  # budget plus the last item's own length

    def test_long_result_is_clipped_not_dropped(self):
        digest = evidence_digest(
            [{"role": "tool", "name": "read_file", "content": "y" * 5000}])
        assert "chars omitted" in digest
        assert digest.count("y") < 5000

    def test_multipart_content_is_flattened(self):
        digest = evidence_digest([
            {"role": "user", "content": [{"type": "text", "text": "check the 2019 filing"},
                                         {"type": "image_url", "image_url": {}}]},
        ])
        assert "check the 2019 filing" in digest

    def test_no_messages_is_empty(self):
        assert evidence_digest([]) == ""


# ── parse_verdict ───────────────────────────────────────────────────────────

class TestParseVerdict:
    def test_clean_verdict_is_an_empty_list(self):
        assert parse_verdict('{"issues": []}') == []

    def test_fenced_json_is_read(self):
        verdict = parse_verdict(
            'Checked them all.\n```json\n{"issues": [{"claim": "4.2M", '
            '"problem": "contradicted", "correction": "3.1M"}]}\n```')
        assert verdict == [{"claim": "4.2M", "problem": "contradicted",
                            "correction": "3.1M"}]

    def test_prose_after_the_json_does_not_break_parsing(self):
        """Scanning to the matching brace, not the last one in the string."""
        verdict = parse_verdict(
            '{"issues": [{"claim": "a", "correction": "b"}]}\nHope that helps {see above}')
        assert len(verdict) == 1

    def test_braces_inside_strings_are_not_counted(self):
        verdict = parse_verdict('{"issues": [{"claim": "the {placeholder} value", '
                                '"correction": "a literal"}]}')
        assert verdict[0]["claim"] == "the {placeholder} value"

    def test_inline_reasoning_is_stripped_before_parsing(self):
        """A thinking model's reasoning is prose about the claims, full of the
        braces and quoted draft fragments the object scanner looks for."""
        verdict = parse_verdict(
            '<think>The draft says {revenue: 4.2M} but the evidence says 3.1M. '
            'Let me write {"issues": []} — no wait, that is wrong.</think>\n'
            '{"issues": [{"claim": "4.2M", "correction": "3.1M"}]}')
        assert verdict == [{"claim": "4.2M", "problem": "contradicted",
                            "correction": "3.1M"}]

    def test_reasoning_that_never_reaches_a_verdict_is_unreadable(self):
        """Budget spent thinking, cut off mid-thought: there is no verdict
        after the block to find, and inventing one from it would be worse."""
        assert parse_verdict(
            '<think>Checking the first claim: {revenue: 4.2M} against') is None

    def test_verdict_ok_shorthand_is_a_pass(self):
        assert parse_verdict('{"verdict": "ok"}') == []

    def test_unreadable_verdict_is_none_not_a_pass(self):
        """None and [] are different answers: one is a clean bill of health,
        the other is a checker that failed to report one."""
        assert parse_verdict("I was unable to check these claims.") is None
        assert parse_verdict("") is None
        assert parse_verdict('{"something": "else"}') is None

    def test_malformed_entries_are_dropped(self):
        verdict = parse_verdict(
            '{"issues": ["a bare string", {"claim": "", "correction": ""}, '
            '{"claim": "real", "correction": "fixed"}]}')
        assert verdict == [{"claim": "real", "problem": "contradicted",
                            "correction": "fixed"}]


# ── notes ───────────────────────────────────────────────────────────────────

class TestRevisionNote:
    def test_corrections_are_joined(self):
        note = revision_note([{"claim": "a", "correction": "3.1M, not 4.2M"},
                              {"claim": "b", "correction": "founded in 1998"}])
        assert note == "3.1M, not 4.2M; founded in 1998"

    def test_claim_is_used_when_there_is_no_correction(self):
        assert revision_note([{"claim": "the second figure", "correction": ""}]) \
            == "the second figure"

    def test_long_list_is_counted_rather_than_quoted(self):
        note = revision_note([{"claim": "", "correction": "correction number %d "
                               "with quite a lot of detail attached" % i}
                              for i in range(8)])
        assert "more)" in note
        assert len(note) <= 260

    def test_no_usable_issues_is_no_note(self):
        assert revision_note([{"claim": "", "correction": ""}]) == ""

    def test_append_note_marks_the_answer(self):
        out = append_note("The revenue was 3.1M.", "3.1M, not 4.2M")
        assert out.startswith("The revenue was 3.1M.")
        assert NOTE_PREFIX in out
        assert out.endswith("3.1M, not 4.2M*")

    def test_no_note_leaves_the_answer_alone(self):
        assert append_note("Unchanged.", "") == "Unchanged."


# ── verify_answer ───────────────────────────────────────────────────────────

def _tool_call(name="search", arguments='{"query": "revenue"}'):
    return types.SimpleNamespace(
        id="call_1",
        function=types.SimpleNamespace(name=name, arguments=arguments))


class _Ask:
    """A scripted stand-in for one non-streaming completion.

    Records what it was asked so a test can assert on the number of calls and
    on whether tools were offered, which is the part of the contract the chat
    loop depends on.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def __call__(self, messages, tools=None, max_tokens=1024):
        self.calls.append({"messages": messages, "tools": tools,
                           "max_tokens": max_tokens})
        if not self.replies:
            raise AssertionError("verify_answer made more calls than the test scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


DRAFT = ("The company reported 4.2M in revenue for 2019, its best year since "
         "the restructuring, and has held that position ever since.")

# The check runs against what the run gathered, so a test that wants it to run
# at all has to have gathered something.  A run with an empty transcript has
# nothing to check the draft against and skips the call entirely.
EVIDENCE = [{"role": "tool", "name": "search",
             "content": "Revenue was 3.1M in 2019."}]
SEARCH_TOOL = [{"type": "function", "function": {"name": "search"}}]


class TestVerifyAnswer:
    @pytest.mark.asyncio
    async def test_clean_verdict_returns_the_draft_untouched(self):
        ask = _Ask(('{"issues": []}', None))
        answer, note, issues = await verify_answer(
            task="revenue?", answer=DRAFT, messages=EVIDENCE, ask=ask)
        assert answer == DRAFT
        assert note == ""
        assert issues == []
        assert len(ask.calls) == 1  # no revision call was made

    @pytest.mark.asyncio
    async def test_issues_produce_a_revision_and_a_note(self):
        revised = ("The company reported 3.1M in revenue for 2019, its best year "
                   "since the restructuring, and has held that position ever since.")
        ask = _Ask(('{"issues": [{"claim": "4.2M in revenue", "problem": '
                    '"contradicted", "correction": "3.1M, per the filing"}]}', None),
                   (revised, None))
        answer, note, issues = await verify_answer(
            task="revenue?", answer=DRAFT, messages=EVIDENCE, ask=ask)
        assert answer.startswith(revised)
        assert answer.endswith(f"*{NOTE_PREFIX}3.1M, per the filing*")
        assert note == "3.1M, per the filing"
        assert len(issues) == 1
        assert len(ask.calls) == 2
        assert ask.calls[1]["tools"] is None  # the revision never calls tools

    @pytest.mark.asyncio
    async def test_tool_calls_are_run_and_the_check_continues(self):
        ask = _Ask((None, [_tool_call()]), ('{"issues": []}', None))
        ran: list = []

        async def run_tools(tool_calls, messages):
            ran.append([tc.function.name for tc in tool_calls])
            messages.append({"role": "tool", "name": "search",
                             "content": "Revenue was 4.2M."})

        answer, note, _ = await verify_answer(
            task="revenue?", answer=DRAFT, messages=EVIDENCE, ask=ask,
            run_tools=run_tools, tools=SEARCH_TOOL, max_tool_turns=1)
        assert ran == [["search"]]
        assert answer == DRAFT
        assert note == ""

    @pytest.mark.asyncio
    async def test_tools_are_withdrawn_on_the_last_turn(self):
        """A checker that would keep looking things up forever is asked for a
        verdict instead once its budget is spent."""
        ask = _Ask((None, [_tool_call()]), ('{"issues": []}', None))

        async def run_tools(tool_calls, messages):
            messages.append({"role": "tool", "name": "search", "content": "..."})

        await verify_answer(task="t", answer=DRAFT, messages=EVIDENCE, ask=ask,
                            run_tools=run_tools, tools=SEARCH_TOOL,
                            max_tool_turns=1)
        assert ask.calls[0]["tools"] is not None
        assert ask.calls[1]["tools"] is None

    @pytest.mark.asyncio
    async def test_lookups_are_off_unless_the_caller_pays_for_them(self):
        """The default check is the transcript against the draft and nothing
        else: no tools offered, and no round trip for a call it did not ask
        for."""
        ask = _Ask((None, [_tool_call()]))
        ran: list = []

        async def run_tools(tool_calls, messages):
            ran.append(tool_calls)

        answer, note, _ = await verify_answer(
            task="t", answer=DRAFT, messages=EVIDENCE, ask=ask,
            run_tools=run_tools, tools=SEARCH_TOOL)
        assert ask.calls[0]["tools"] is None
        assert ran == []
        assert len(ask.calls) == 1
        assert answer == DRAFT
        assert note == ""

    @pytest.mark.asyncio
    async def test_a_run_with_no_evidence_is_not_checked_at_all(self):
        """Nothing gathered and no lookups allowed leaves only the weights that
        wrote the draft to check it against, which is not a check."""
        ask = _Ask(('{"issues": []}', None))
        answer, note, issues = await verify_answer(
            task="t", answer=DRAFT, messages=[], ask=ask)
        assert answer == DRAFT
        assert (note, issues) == ("", [])
        assert ask.calls == []

    @pytest.mark.asyncio
    async def test_no_evidence_still_checks_when_lookups_are_allowed(self):
        ask = _Ask(('{"issues": []}', None))

        async def run_tools(tool_calls, messages):
            pass

        await verify_answer(task="t", answer=DRAFT, messages=[], ask=ask,
                            run_tools=run_tools, tools=SEARCH_TOOL,
                            max_tool_turns=1)
        assert len(ask.calls) == 1

    @pytest.mark.asyncio
    async def test_uncovered_claims_are_dropped_when_nothing_can_settle_them(self):
        """Without a lookup, "unsupported" reports the absence of evidence, not
        an error — acting on it rewrites a good answer to hedge a true claim."""
        ask = _Ask(('{"issues": [{"claim": "best year", "problem": "unsupported",'
                    ' "correction": "the evidence does not say"}]}', None))
        answer, note, issues = await verify_answer(
            task="t", answer=DRAFT, messages=EVIDENCE, ask=ask)
        assert answer == DRAFT
        assert (note, issues) == ("", [])
        assert len(ask.calls) == 1  # no revision was attempted

    @pytest.mark.asyncio
    async def test_contradictions_are_still_acted_on_without_lookups(self):
        revised = DRAFT.replace("4.2M", "3.1M")
        ask = _Ask(('{"issues": [{"claim": "4.2M", "problem": "contradicted",'
                    ' "correction": "3.1M"}]}', None), (revised, None))
        answer, note, issues = await verify_answer(
            task="t", answer=DRAFT, messages=EVIDENCE, ask=ask)
        assert answer.startswith(revised)
        assert note == "3.1M"
        assert len(issues) == 1

    @pytest.mark.asyncio
    async def test_uncovered_claims_survive_when_lookups_were_available(self):
        ask = _Ask(('{"issues": [{"claim": "best year", "problem": "unsupported",'
                    ' "correction": "2021 was higher"}]}', None),
                   (DRAFT.replace("its best year", "a strong year"), None))

        async def run_tools(tool_calls, messages):
            pass

        _, note, issues = await verify_answer(
            task="t", answer=DRAFT, messages=EVIDENCE, ask=ask,
            run_tools=run_tools, tools=SEARCH_TOOL, max_tool_turns=1)
        assert note == "2021 was higher"
        assert len(issues) == 1

    @pytest.mark.asyncio
    async def test_no_tools_configured_means_no_tools_offered(self):
        ask = _Ask(('{"issues": []}', None))
        await verify_answer(task="t", answer=DRAFT, messages=EVIDENCE, ask=ask)
        assert ask.calls[0]["tools"] is None

    @pytest.mark.asyncio
    async def test_unreadable_verdict_keeps_the_draft(self):
        ask = _Ask(("I checked everything and it all looks fine to me.", None))
        answer, note, issues = await verify_answer(
            task="t", answer=DRAFT, messages=EVIDENCE, ask=ask)
        assert answer == DRAFT
        assert note == ""
        assert issues == []

    @pytest.mark.asyncio
    async def test_a_failed_check_never_costs_the_answer(self):
        ask = _Ask(RuntimeError("endpoint down"))
        answer, note, _ = await verify_answer(
            task="t", answer=DRAFT, messages=EVIDENCE, ask=ask)
        assert answer == DRAFT
        assert note == ""

    @pytest.mark.asyncio
    async def test_a_failed_lookup_never_costs_the_answer(self):
        ask = _Ask((None, [_tool_call()]))

        async def run_tools(tool_calls, messages):
            raise RuntimeError("stopped")

        answer, note, _ = await verify_answer(
            task="t", answer=DRAFT, messages=EVIDENCE, ask=ask, run_tools=run_tools,
            tools=SEARCH_TOOL, max_tool_turns=1)
        assert answer == DRAFT
        assert note == ""

    @pytest.mark.asyncio
    async def test_a_gutted_revision_is_discarded_but_the_finding_is_kept(self):
        """The model summarized or apologized instead of correcting. The draft
        with the error flagged beats a paragraph that lost the work — and beats
        the draft with the error left silent."""
        ask = _Ask(('{"issues": [{"claim": "4.2M", "correction": "3.1M"}]}', None),
                   ("Sorry, I cannot verify that.", None))
        answer, note, issues = await verify_answer(
            task="t", answer=DRAFT, messages=EVIDENCE, ask=ask)
        assert answer.startswith(DRAFT)
        assert CORRECTION_PREFIX + "3.1M" in answer
        assert note == "3.1M"
        assert len(issues) == 1

    @pytest.mark.asyncio
    async def test_a_failed_revision_flags_the_finding(self):
        ask = _Ask(('{"issues": [{"claim": "4.2M", "correction": "3.1M"}]}', None),
                   RuntimeError("endpoint down"))
        answer, note, _ = await verify_answer(
            task="t", answer=DRAFT, messages=EVIDENCE, ask=ask)
        assert answer.startswith(DRAFT)
        assert CORRECTION_PREFIX in answer
        assert note == "3.1M"

    @pytest.mark.asyncio
    async def test_an_answer_too_long_to_rewrite_is_flagged_not_replaced(self):
        """Rewriting means reproducing, and a revision call cannot reproduce
        what it was never shown. Losing the tail of a long answer to correct
        its opening is the wrong trade."""
        long_draft = DRAFT + "\n\n" + ("Supporting detail, 12 of them. " * 800).strip()
        ask = _Ask(('{"issues": [{"claim": "4.2M", "correction": "3.1M"}]}', None))
        answer, note, issues = await verify_answer(
            task="t", answer=long_draft, messages=EVIDENCE, ask=ask)
        assert answer.startswith(long_draft)
        assert CORRECTION_PREFIX + "3.1M" in answer
        assert len(ask.calls) == 1  # no revision was attempted
        assert len(issues) == 1

    @pytest.mark.asyncio
    async def test_the_revision_gets_the_callers_token_budget(self):
        ask = _Ask(('{"issues": [{"claim": "4.2M", "correction": "3.1M"}]}', None),
                   (DRAFT.replace("4.2M", "3.1M"), None))
        await verify_answer(task="t", answer=DRAFT, messages=EVIDENCE, ask=ask,
                            revision_max_tokens=32768)
        assert ask.calls[1]["max_tokens"] == 32768

    @pytest.mark.asyncio
    async def test_evidence_reaches_both_prompts(self):
        messages = [{"role": "tool", "name": "search", "content": "Revenue was 3.1M."}]
        ask = _Ask(('{"issues": [{"claim": "4.2M", "correction": "3.1M"}]}', None),
                   (DRAFT.replace("4.2M", "3.1M"), None))
        await verify_answer(task="revenue?", answer=DRAFT, messages=messages, ask=ask)
        for call in ask.calls:
            assert "Revenue was 3.1M." in call["messages"][1]["content"]


# ── prompts ─────────────────────────────────────────────────────────────────

class TestPrompts:
    def test_verdict_prompt_carries_task_answer_and_evidence(self):
        messages = build_verdict_messages("what is the revenue?", DRAFT,
                                          "[search]\nRevenue was 3.1M.")
        assert messages[0]["role"] == "system"
        body = messages[1]["content"]
        assert "what is the revenue?" in body
        assert DRAFT in body
        assert "Revenue was 3.1M." in body

    def test_the_default_prompt_asks_only_for_contradictions(self):
        """No lookups behind it, so it is not asked to raise claims it has no
        way to settle."""
        system = build_verdict_messages("t", DRAFT, "[search]\n3.1M")[0]["content"]
        assert "contradicts" in system
        assert "call the tool" not in system

    def test_the_tool_prompt_asks_for_uncovered_claims_too(self):
        system = build_verdict_messages("t", DRAFT, "[search]\n3.1M",
                                        with_tools=True)[0]["content"]
        assert "fails to support" in system
        assert "call the tool" in system

    def test_verdict_prompt_says_when_nothing_was_gathered(self):
        body = build_verdict_messages("t", DRAFT, "")[1]["content"]
        assert "nothing was gathered" in body

    def test_revision_prompt_lists_the_corrections(self):
        body = build_revision_messages(
            "t", DRAFT,
            [{"claim": "4.2M", "problem": "contradicted", "correction": "3.1M"}],
            "")[1]["content"]
        assert "4.2M" in body
        assert "contradicted" in body
        assert "3.1M" in body


# ── wiring into chat() ──────────────────────────────────────────────────────

def _completion(content=None, tool_calls=None, finish_reason="stop"):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = None
    return completion


ANSWER = ("The 2019 filing puts revenue at 4.2M, which is the figure quoted in "
          "the summary table on page 3 of the report you asked about.")
REVISED = ("The 2019 filing puts revenue at 3.1M, which is the figure quoted in "
           "the summary table on page 3 of the report you asked about.")


async def _run_chat(replies, **kwargs):
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=replies)
    with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
         patch("model.serving.chat._resolve_model_id",
               new_callable=AsyncMock, return_value="test-model"):
        result = await asyncio.wait_for(chat(
            host="http://localhost:8000/v1",
            instruction="What was the 2019 revenue?",
            safety_queue=asyncio.Queue(),
            **kwargs,
        ), timeout=30)
    return result, client


class TestChatWiring:
    @pytest.fixture(autouse=True)
    def _forget_endpoint_state(self):
        """What a host rejected is remembered for the process; a test that
        teaches it something must not teach the next one."""
        reset_endpoint_caches()
        yield
        reset_endpoint_caches()

    @pytest.mark.asyncio
    async def test_answer_is_checked_before_it_is_returned(self):
        result, client = await _run_chat([
            _completion(content=ANSWER),
            _completion(content='{"issues": []}'),
        ])
        assert result == ANSWER
        assert client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_a_corrected_answer_is_what_the_caller_gets(self):
        result, _ = await _run_chat([
            _completion(content=ANSWER),
            _completion(content='{"issues": [{"claim": "revenue at 4.2M", '
                                '"problem": "contradicted", "correction": '
                                '"3.1M in the filing"}]}'),
            _completion(content=REVISED),
        ])
        assert result.startswith(REVISED)
        assert f"{NOTE_PREFIX}3.1M in the filing" in result

    @pytest.mark.asyncio
    async def test_verification_can_be_turned_off(self):
        result, client = await _run_chat([_completion(content=ANSWER)],
                                         verify_answers=False)
        assert result == ANSWER
        assert client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_a_claim_free_answer_is_not_checked(self):
        plain = ("I've written that up for you. Let me know if you would like a "
                 "different structure, or a shorter version to share around.")
        result, client = await _run_chat([_completion(content=plain)])
        assert result == plain
        assert client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_a_loop_bailout_is_not_fact_checked(self):
        """The loop reporting on itself is not a claim about the world."""
        planning = "Let me start working on the analysis."
        registry = MagicMock()
        registry.get_tool_items.return_value = [
            {"type": "function", "function": {"name": "bash"}}]
        registry.tools = {"bash"}
        result, client = await _run_chat(
            [_completion(content=planning)] * 4, tool_registry=registry)
        assert "unable to complete" in result.lower()
        assert client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_the_ui_is_told_the_check_is_running(self):
        ui = MagicMock()
        ui.messages = []
        await _run_chat([
            _completion(content=ANSWER),
            _completion(content='{"issues": []}'),
        ], chat_ui=ui)
        ui.verification_start.assert_called_once()
        ui.verification_end.assert_called_once()
        assert ui.verification_end.call_args[0][1] == ""  # nothing was revised

    @pytest.mark.asyncio
    async def test_the_ui_is_handed_the_revision(self):
        ui = MagicMock()
        ui.messages = []
        await _run_chat([
            _completion(content=ANSWER),
            _completion(content='{"issues": [{"claim": "4.2M", "correction": '
                                '"3.1M in the filing"}]}'),
            _completion(content=REVISED),
        ], chat_ui=ui)
        answer, note = ui.verification_end.call_args[0]
        assert answer.startswith(REVISED)
        assert note == "3.1M in the filing"

    @pytest.mark.asyncio
    async def test_only_read_only_tools_are_offered_to_the_checker(self):
        """A checker able to write files or run commands would be a second
        agent acting after the user was told the work was done."""
        registry = MagicMock()
        registry.get_tool_items.return_value = [
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
            {"type": "function", "function": {"name": "search", "parameters": {}}},
        ]
        registry.tools = {"bash", "search"}
        _, client = await _run_chat([
            _completion(content=ANSWER),
            _completion(content='{"issues": []}'),
        ], tool_registry=registry, verify_max_tool_turns=1)
        verify_call = client.chat.completions.create.call_args_list[1]
        offered = [t["function"]["name"] for t in verify_call.kwargs["tools"]]
        assert offered == ["search"]

    @pytest.mark.asyncio
    async def test_the_check_makes_no_lookups_by_default(self):
        """One call against what the run already gathered. Every lookup past
        that is a round trip, a tool run, and another verdict call behind an
        answer the user is already reading."""
        registry = MagicMock()
        registry.get_tool_items.return_value = [
            {"type": "function", "function": {"name": "search", "parameters": {}}}]
        registry.tools = {"search"}
        _, client = await _run_chat([
            _completion(content=ANSWER),
            _completion(content='{"issues": []}'),
        ], tool_registry=registry)
        assert client.chat.completions.create.call_count == 2
        assert "tools" not in client.chat.completions.create.call_args_list[1].kwargs

    @pytest.mark.asyncio
    async def test_a_verdict_in_the_reasoning_field_is_still_read(self):
        """Servers split reasoning out of `content` inconsistently, and one
        that does can leave `content` empty with the JSON in the other field."""
        verdict = _completion(content="")
        verdict.choices[0].message.reasoning = '{"issues": []}'
        result, _ = await _run_chat([_completion(content=ANSWER), verdict])
        assert result == ANSWER

    @pytest.mark.asyncio
    async def test_the_verdict_gets_room_to_reason(self):
        """Measured at ~400 output tokens for three claims against one source
        on a thinking model. A budget it overruns yields no verdict at all."""
        _, client = await _run_chat([
            _completion(content=ANSWER),
            _completion(content='{"issues": []}'),
        ], max_tokens=32768)
        verify_call = client.chat.completions.create.call_args_list[1]
        assert verify_call.kwargs["max_tokens"] >= 4096

    @pytest.mark.asyncio
    async def test_the_check_does_not_pay_for_thinking(self):
        """Measured on Qwen3.6-27B: 18s and 1,277 tokens to report two wrong
        figures with the template's default thinking, 1.2s and 79 tokens for
        the same verdict with it off. Reading a claim off a source is
        recognition, not deliberation."""
        _, client = await _run_chat([
            _completion(content=ANSWER),
            _completion(content='{"issues": []}'),
        ])
        answer_call, verify_call = client.chat.completions.create.call_args_list
        assert (verify_call.kwargs["extra_body"]["chat_template_kwargs"]
                == {"enable_thinking": False})
        # The answer itself is untouched — it is the part worth deliberating.
        assert "enable_thinking" not in str(answer_call.kwargs.get("extra_body", {}))

    @pytest.mark.asyncio
    async def test_a_host_that_rejects_template_kwargs_is_retried_plainly(self):
        """A chat template without the switch ignores an unknown variable, but
        a server that validates them rejects the request. The check still runs;
        it just runs the way that host accepts."""
        from openai import OpenAIError

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            _completion(content=ANSWER),
            OpenAIError("unknown chat_template_kwargs"),
            _completion(content='{"issues": []}'),
        ])
        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await asyncio.wait_for(chat(
                host="http://localhost:8000/v1", instruction="q",
                safety_queue=asyncio.Queue()), timeout=30)
        assert result == ANSWER
        retried = client.chat.completions.create.call_args_list[2]
        assert retried.kwargs["extra_body"] == {}
        # And remembered, so the next answer does not pay for the same refusal.
        assert "http://localhost:8000/v1" in _NO_TEMPLATE_KWARGS

    @pytest.mark.asyncio
    async def test_a_remembered_rejection_skips_the_doomed_call(self):
        _NO_TEMPLATE_KWARGS.add("http://localhost:8000/v1")
        result, client = await _run_chat([
            _completion(content=ANSWER),
            _completion(content='{"issues": []}'),
        ])
        assert result == ANSWER
        assert client.chat.completions.create.call_count == 2
        assert client.chat.completions.create.call_args_list[1].kwargs["extra_body"] == {}

    @pytest.mark.asyncio
    async def test_ollama_is_asked_for_the_same_thing_its_own_way(self):
        """Ollama switches thinking with a top-level flag rather than through
        the chat template; the check must not pay for it there either."""
        message = MagicMock()
        message.content = '{"issues": []}'
        message.tool_calls = None
        verdict = MagicMock()
        verdict.message = message
        verdict.done_reason = "stop"
        verdict.eval_count = 12
        verdict.prompt_eval_count = 100

        answer = MagicMock()
        answer.message = MagicMock(content=ANSWER, tool_calls=None)
        answer.done_reason = "stop"
        answer.eval_count = 40
        answer.prompt_eval_count = 100

        ollama = AsyncMock()
        ollama.chat = AsyncMock(side_effect=[answer, verdict])
        with patch("model.serving.chat._create_ollama_client", return_value=ollama), \
             patch("model.serving.chat._ollama_resolve_model_id",
                   new_callable=AsyncMock, return_value="qwen3"), \
             patch("model.serving.chat._get_ollama_max_context",
                   new_callable=AsyncMock, return_value=32768):
            result = await asyncio.wait_for(chat(
                host="https://ollama.com", host_key="test-key", instruction="q",
                safety_queue=asyncio.Queue()), timeout=30)
        assert result == ANSWER
        assert ollama.chat.call_args_list[1].kwargs["think"] is False

    @pytest.mark.asyncio
    async def test_the_check_is_timed_separately(self):
        metrics: dict = {}
        await _run_chat([
            _completion(content=ANSWER),
            _completion(content='{"issues": [{"claim": "4.2M", "correction": '
                                '"3.1M"}]}'),
            _completion(content=REVISED),
        ], metrics=metrics)
        assert metrics["verify_issues"] == 1
        assert metrics["verify_revisions"] == 1
        assert "verify_s" in metrics

    @pytest.mark.asyncio
    async def test_a_stopped_run_keeps_the_draft(self):
        """Stop pressed while the check is in flight: the user keeps what they
        already read rather than losing the answer to a cancelled call."""
        queue: asyncio.Queue = asyncio.Queue()
        client = AsyncMock()
        calls = {"n": 0}

        async def _stop_during_the_check(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _completion(content=ANSWER)
            # The check is in flight when the user presses stop.  The call is
            # cancelled; the answer they already read is not.
            queue.put_nowait("stop")
            await asyncio.sleep(30)

        client.chat.completions.create = AsyncMock(side_effect=_stop_during_the_check)
        with patch("model.serving.chat.AsyncOpenAI", return_value=client), \
             patch("model.serving.chat._resolve_model_id",
                   new_callable=AsyncMock, return_value="test-model"):
            result = await asyncio.wait_for(chat(
                host="http://localhost:8000/v1",
                instruction="What was the 2019 revenue?",
                safety_queue=queue,
            ), timeout=30)
        assert result == ANSWER
        assert client.chat.completions.create.call_count == 2
