"""
The instrument, tested.

``runner.py`` is the thing that turns the ruler from a file into a
number.  These tests are about the parts that decide whether a number
means anything:

- the two arms must not contaminate each other, and neither may
  contaminate the operator's own corpus;
- the grade must be parsed strictly, because a mis-parsed grade is a
  measurement of the parse;
- the comparison must carry an error bar, and a delta that does not
  clear it must be reported as not significant;
- an unreachable grader must be a *missing* measurement, never a
  failure, because a failure would make the arms look different for a
  reason that has nothing to do with the agent.

The agent is a stub throughout.  The point is not that the agent works
-- that is the benchmark provider's job -- but that the instrument
measures the agent correctly.
"""

import json
import os

import pytest

from src.learn import runner
from src.learn import trajectory


# ── fixtures ────────────────────────────────────────────────────────────────

class StubAgent:
    """An agent that answers with a fixed string and records its calls."""

    def __init__(self, answer="the right answer", config=None):
        self.answer = answer
        self.config = config or {}
        self.calls = []

    async def process_task(self, task, **kwargs):
        self.calls.append({"task": task, "kwargs": kwargs})
        return self.answer


def _item(task="a question", response="the right answer", item_id="s1#1"):
    return {"id": item_id, "task": task, "response": response,
            "model": "m1", "owner": None, "session_id": "s1", "turn": 1}


async def _ask_ok(messages):
    """A grader that says everything is correct."""
    return "GRADE: C"


async def _ask_wrong(messages):
    return "GRADE: I"


async def _ask_not_attempted(messages):
    return "GRADE: N"


async def _ask_raises(messages):
    raise RuntimeError("the grader is down")


async def _ask_garbage(messages):
    return "I think it is probably correct, so GRADE: C"


# ── the grade ───────────────────────────────────────────────────────────────

class TestParseGrade:
    """The grade is parsed strictly."""

    def test_a_plain_grade_is_read(self):
        assert runner.parse_grade("GRADE: C") == "C"

    def test_the_grade_is_read_from_a_long_reply(self):
        assert runner.parse_grade("Let me think.\nGRADE: I\n") == "I"

    def test_the_grade_is_read_with_extra_space(self):
        assert runner.parse_grade("grade:   N") == "N"

    def test_a_reply_with_no_grade_is_no_grade(self):
        assert runner.parse_grade("the answer looks fine to me") == ""

    def test_an_empty_reply_is_no_grade(self):
        assert runner.parse_grade("") == ""

    def test_an_unknown_letter_is_no_grade(self):
        assert runner.parse_grade("GRADE: Z") == ""

    def test_the_first_grade_wins(self):
        assert runner.parse_grade("GRADE: C\nGRADE: I") == "C"


class TestGrade:
    """Grading one answer."""

    async def test_a_correct_answer_is_graded_correct(self):
        verdict = await runner.grade(_ask_ok, "q", "a", "an answer")
        assert verdict["grade"] == "C"

    async def test_an_empty_answer_is_not_attempted(self):
        """An empty answer is not a wrong answer.

        Counting it as wrong would make silence look like a failure and
        an abstention look like a mistake, when it is neither.
        """
        verdict = await runner.grade(_ask_ok, "q", "a", "")
        assert verdict["grade"] == "N"

    async def test_a_failed_grader_is_ungraded_not_wrong(self):
        """The measurement is missing, not negative.

        A grader that cannot be reached must not be counted as a
        failure: it would make the arms look different for a reason
        that has nothing to do with the agent.
        """
        verdict = await runner.grade(_ask_raises, "q", "a", "an answer")
        assert verdict["grade"] == ""

    async def test_a_garbled_reply_with_a_grade_in_it_is_read(self):
        verdict = await runner.grade(_ask_garbage, "q", "a", "an answer")
        assert verdict["grade"] == "C"

    async def test_the_grader_sees_the_question_and_the_reference(self):
        seen = {}

        async def spy(messages):
            seen["messages"] = messages
            return "GRADE: C"

        await runner.grade(spy, "the question", "the reference", "an answer")
        prompt = seen["messages"][-1]["content"]
        assert "the question" in prompt
        assert "the reference" in prompt
        assert "an answer" in prompt


# ── the summary ─────────────────────────────────────────────────────────────

class TestSummarise:
    """The headline numbers, and the ones that keep them honest."""

    def test_an_empty_run_is_zero(self):
        summary = runner.summarise([])
        assert summary["n"] == 0
        assert summary["accuracy"] == 0.0

    def test_the_accuracy_is_over_all_items_not_just_the_graded_ones(self):
        """An ungraded item counts against the accuracy.

        Otherwise a run that graded half its items and got all of them
        right would report 1.0, and a run that graded all of them and
        got two-thirds right would report 0.67 -- and the ungraded half
        would be invisible.
        """
        results = [
            {"grade": "C"}, {"grade": "C"},
            {"grade": ""}, {"grade": ""},
        ]
        summary = runner.summarise(results)
        assert summary["n"] == 4
        assert summary["correct"] == 2
        assert summary["ungraded"] == 2
        assert summary["accuracy"] == 0.5

    def test_the_attempted_rate_is_reported(self):
        results = [{"grade": "C"}, {"grade": "N"}, {"grade": "I"}]
        summary = runner.summarise(results)
        assert summary["not_attempted"] == 1
        assert summary["incorrect"] == 1
        assert summary["correct"] == 1

    def test_the_errors_are_counted(self):
        results = [{"grade": "", "error": "boom"}]
        assert runner.summarise(results)["errors"] == 1


# ── the comparison ──────────────────────────────────────────────────────────

class TestCompare:
    """Two arms, and whether the difference is real."""

    def test_a_bigger_delta_is_bigger(self):
        control = [{"grade": "C"}, {"grade": "I"}]
        treatment = [{"grade": "C"}, {"grade": "C"}]
        result = runner.compare(control, treatment)
        assert result["delta"] == 0.5

    def test_a_delta_carries_an_error_bar(self):
        control = [{"grade": "C"}, {"grade": "I"}]
        treatment = [{"grade": "C"}, {"grade": "C"}]
        result = runner.compare(control, treatment)
        assert result["stderr"] > 0

    def test_a_tiny_delta_is_not_significant(self):
        """A difference the data cannot support is not a result.

        Two arms that scored the same must be reported as tied, even
        when the numbers are not equal: 50 correct out of 100 against
        50 correct out of 100 is a delta of zero, and reporting that
        as a win would be reporting nothing.
        """
        control = ([{"grade": "C"} for _ in range(50)]
                   + [{"grade": "I"} for _ in range(50)])
        treatment = ([{"grade": "C"} for _ in range(50)]
                     + [{"grade": "I"} for _ in range(50)])
        result = runner.compare(control, treatment)
        assert result["delta"] == 0.0
        assert result["stderr"] > 0
        assert not result["significant"]

    def test_a_one_item_difference_against_a_wide_bar_is_not_significant(self):
        """A one-item difference in 200 is noise.

        The error bar on a 200-item run is wider than a single item's
        worth of accuracy, so a one-item difference cannot be called a
        result.  This is the case the plan's own gate is about: a
        delta that does not clear its error bar ships nothing.
        """
        control = ([{"grade": "C"} for _ in range(99)]
                   + [{"grade": "I"} for _ in range(101)])
        treatment = ([{"grade": "C"} for _ in range(100)]
                     + [{"grade": "I"} for _ in range(100)])
        result = runner.compare(control, treatment)
        assert abs(result["delta"]) < result["stderr"]
        assert not result["significant"]

    def test_a_large_delta_is_significant(self):
        control = [{"grade": "I"} for _ in range(50)]
        treatment = [{"grade": "C"} for _ in range(50)]
        result = runner.compare(control, treatment)
        assert result["significant"]

    def test_the_n_is_the_smaller_arm(self):
        """A comparison is only as big as the smaller arm.

        If one arm answered fewer items, the extra items in the other
        are not evidence about the pair.
        """
        control = [{"grade": "C"}] * 10
        treatment = [{"grade": "C"}] * 4
        result = runner.compare(control, treatment)
        assert result["n_items"] == 4

    def test_the_text_says_when_a_delta_is_not_significant(self):
        control = ([{"grade": "C"} for _ in range(50)]
                   + [{"grade": "I"} for _ in range(50)])
        treatment = ([{"grade": "C"} for _ in range(50)]
                     + [{"grade": "I"} for _ in range(50)])
        text = runner.format_comparison(runner.compare(control, treatment))
        assert "not significant" in text
        assert "Ship nothing" in text


# ── the run ─────────────────────────────────────────────────────────────────

class TestRunArm:
    """One arm, run over a list of items."""

    async def test_every_item_is_answered_and_graded(self, tmp_path):
        items = [_item("q1", "the reference", "s1#1"),
                 _item("q2", "the reference", "s1#2")]
        results = await runner.run_arm(
            items, arm="off", root=str(tmp_path),
            make_agent=lambda config: StubAgent(),
            ask=_ask_ok)
        assert len(results) == 2
        assert all(r["grade"] == "C" for r in results)
        assert all(r["arm"] == "off" for r in results)

    async def test_the_agent_is_asked_each_question(self, tmp_path):
        items = [_item("first question", "r", "s1#1"),
                 _item("second question", "r", "s1#2")]
        agent = StubAgent()
        await runner.run_arm(items, arm="off", root=str(tmp_path),
                             make_agent=lambda config: agent,
                             ask=_ask_ok)
        assert [c["task"] for c in agent.calls] == ["first question",
                                                    "second question"]

    async def test_a_failed_task_is_recorded_not_raised(self, tmp_path):
        """One bad task must not lose the whole run.

        The arm's results are a list; a task that raised has no
        answer, and the rest of the run is still worth measuring.
        """
        class Flaky:
            async def process_task(self, task, **kwargs):
                if task == "bad":
                    raise RuntimeError("the model is down")
                return "an answer"

        items = [_item("good", "r", "s1#1"), _item("bad", "r", "s1#2")]
        results = await runner.run_arm(items, arm="off", root=str(tmp_path),
                                       make_agent=lambda config: Flaky(),
                                       ask=_ask_ok)
        assert len(results) == 2
        assert results[0]["error"] == ""
        assert results[1]["error"]

    async def test_the_limit_caps_the_run(self, tmp_path):
        items = [_item("q%d" % i, "r", "s%d#1" % i) for i in range(10)]
        results = await runner.run_arm(items, arm="off", root=str(tmp_path),
                                       make_agent=lambda config: StubAgent(),
                                       ask=_ask_ok, limit=3)
        assert len(results) == 3

    async def test_the_results_are_written_to_disk(self, tmp_path):
        items = [_item("q", "r", "s1#1")]
        await runner.run_arm(items, arm="off", root=str(tmp_path),
                             make_agent=lambda config: StubAgent(),
                             ask=_ask_ok)
        path = os.path.join(str(tmp_path), "arms", "off", "results.json")
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved) == 1
        assert saved[0]["grade"] == "C"

    async def test_the_two_arms_get_different_stores(self, tmp_path):
        """The control's trajectories must not be the treatment's memory.

        This is the contamination that would make an A/B meaningless:
        the treatment arm recalling the control arm's runs would be
        measuring the harness, not the agent.
        """
        seen = {}

        def factory(config):
            seen.setdefault("configs", []).append(config)
            return StubAgent()

        items = [_item("q", "r", "s1#1")]
        await runner.run_arm(items, arm="off", root=str(tmp_path),
                             make_agent=factory, ask=_ask_ok)
        await runner.run_arm(items, arm="adapt", root=str(tmp_path),
                             make_agent=factory, ask=_ask_ok)
        control, treatment = seen["configs"]
        assert control["learn"]["autonomy"] == "off"
        assert treatment["learn"]["autonomy"] == "adapt"
        assert control["learn"]["path"] != treatment["learn"]["path"]

    async def test_the_arm_store_is_not_the_operators_corpus(self, tmp_path):
        """A run must not write into the store the loops read.

        The store the operator's own corpus lives in is the default
        ``learn.path``; a run that wrote into it would feed benchmark
        traffic back into the recall path under test.
        """
        seen = {}
        base = {"learn": {"path": "/somewhere/operator/learned"}}

        def factory(config):
            seen["config"] = config
            return StubAgent()

        await runner.run_arm([_item()], arm="adapt", root=str(tmp_path),
                             base_config=base, make_agent=factory,
                             ask=_ask_ok)
        assert seen["config"]["learn"]["path"] != "/somewhere/operator/learned"

    async def test_a_run_without_a_grader_records_but_does_not_grade(self,
                                                                      tmp_path):
        results = await runner.run_arm([_item()], arm="off",
                                       root=str(tmp_path),
                                       make_agent=lambda config: StubAgent(),
                                       ask=None)
        assert len(results) == 1
        assert results[0]["grade"] == ""
        assert results[0]["answer"] == "the right answer"


# ── the store ───────────────────────────────────────────────────────────────

class TestArmStore:
    def test_the_store_is_under_the_run_root(self, tmp_path):
        path = runner._arm_store(str(tmp_path), "adapt")
        assert path.startswith(str(tmp_path))
        assert os.path.isdir(path)

    def test_two_arms_get_two_stores(self, tmp_path):
        a = runner._arm_store(str(tmp_path), "off")
        b = runner._arm_store(str(tmp_path), "adapt")
        assert a != b
