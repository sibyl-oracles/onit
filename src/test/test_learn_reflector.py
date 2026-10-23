"""
The model-based Reflector, tested without a model.

The reflector's whole job is to turn a model's reply into a bullet
that will be injected into the agent's prompt.  Two things can go
wrong and both are bad:

- it accepts something that is not a bullet, and the agent is told
  "the bad run failed" as though it were advice;
- it rejects something that is a bullet, and the loop silently learns
  nothing.

So the parse is the unit under test.  The model is a stub throughout:
the question is never whether a model can write a tactic, it is
whether the code can tell one from a non-one.

The ask callable is injected, which is the same choice the fact-checker
makes, and it means these tests need no network and no endpoint.
"""

import asyncio
import json

import pytest

from src.learn import reflector


# ── fixtures ────────────────────────────────────────────────────────────────

def _record(session="s1", task="find the weather forecast",
            response="an answer", tools=None, verdict="clean"):
    return {
        "session_id": session,
        "turn": 1,
        "task": task,
        "response": response,
        "signals": {"verifier": verdict},
        "trajectory": [{"turn": 1,
                        "tools": tools or [{"name": "search", "ok": True}]}],
    }


def _asker(reply):
    """An ask callable that always says the same thing."""
    calls = []

    async def ask(messages):
        calls.append(messages)
        return reply

    ask.calls = calls
    return ask


# ── the parse ───────────────────────────────────────────────────────────────

class TestExtractJson:
    """The reply is JSON, or it is nothing."""

    def test_plain_json_is_read(self):
        reply = json.dumps({"bullets": [{"text": "do the thing",
                                         "trigger": "search"}]})
        bullets = reflector._extract_json(reply)
        assert len(bullets) == 1
        assert bullets[0]["text"] == "do the thing"

    def test_a_fenced_block_is_read(self):
        reply = "```json\n" + json.dumps({"bullets": [{"text": "x"}]}) + "\n```"
        assert len(reflector._extract_json(reply)) == 1

    def test_a_fenced_block_without_the_language_is_read(self):
        reply = "```\n" + json.dumps({"bullets": [{"text": "x"}]}) + "\n```"
        assert len(reflector._extract_json(reply)) == 1

    def test_json_inside_prose_is_read(self):
        reply = ("Here is what I found:\n"
                 + json.dumps({"bullets": [{"text": "x"}]})
                 + "\nHope that helps.")
        assert len(reflector._extract_json(reply)) == 1

    def test_a_thinking_block_is_stripped(self):
        reply = ("<think>the bad run searched first</think>\n"
                 + json.dumps({"bullets": [{"text": "x"}]}))
        assert len(reflector._extract_json(reply)) == 1

    def test_prose_is_not_json(self):
        assert reflector._extract_json("the bad run failed") == []

    def test_an_empty_reply_is_nothing(self):
        assert reflector._extract_json("") == []

    def test_a_list_at_the_top_level_is_nothing(self):
        """The shape was asked for; a different shape is not used."""
        assert reflector._extract_json(json.dumps([{"text": "x"}])) == []

    def test_a_missing_bullets_key_is_nothing(self):
        assert reflector._extract_json(json.dumps({"items": []})) == []

    def test_a_bullets_key_that_is_not_a_list_is_nothing(self):
        assert reflector._extract_json(json.dumps({"bullets": "x"})) == []

    def test_a_non_dict_bullet_is_dropped(self):
        reply = json.dumps({"bullets": [{"text": "keep"}, "drop", 42]})
        bullets = reflector._extract_json(reply)
        assert len(bullets) == 1


# ── the bullet ──────────────────────────────────────────────────────────────

class TestCleanBullet:
    """A bullet, normalised -- or nothing."""

    def test_a_good_bullet_survives(self):
        cleaned = reflector._clean_bullet({"text": "Open the file first",
                                           "trigger": "read_file"})
        assert cleaned["text"] == "Open the file first"
        assert cleaned["trigger"] == "read_file"

    def test_a_bullet_without_text_is_nothing(self):
        assert reflector._clean_bullet({"trigger": "search"}) is None

    def test_a_bullet_with_an_empty_text_is_nothing(self):
        assert reflector._clean_bullet({"text": "   "}) is None

    def test_a_meta_comment_is_nothing(self):
        """The most common failure: the model describes the contrast
        instead of giving advice."""
        assert reflector._clean_bullet({"text": "In the bad run, the search "
                                              "was called twice"}) is None
        assert reflector._clean_bullet({"text": "The failing run did not "
                                              "check the date"}) is None

    def test_a_bullet_without_a_trigger_is_kept(self):
        """The trigger is a convenience; the text can stand alone."""
        cleaned = reflector._clean_bullet({"text": "Check the date first"})
        assert cleaned is not None
        assert cleaned["trigger"] == ""

    def test_a_long_text_is_truncated(self):
        cleaned = reflector._clean_bullet({"text": "x" * 500})
        assert len(cleaned["text"]) <= reflector.MAX_BULLET_CHARS

    def test_a_long_trigger_is_truncated(self):
        cleaned = reflector._clean_bullet({"text": "x", "trigger": "y" * 200})
        assert len(cleaned["trigger"]) <= 60


# ── the contrast ────────────────────────────────────────────────────────────

class TestReflectContrast:
    """One contrast, one call to the model."""

    def test_a_good_reply_becomes_a_delta(self):
        reply = json.dumps({"bullets": [{"text": "Read the file before "
                                                "quoting it",
                                          "trigger": "read_file"}]})
        deltas = asyncio.run(reflector.reflect_contrast(
            _asker(reply), _record(), _record(session="s2", verdict="issues")))
        assert len(deltas) == 1
        assert deltas[0]["op"] == "insert"
        assert deltas[0]["text"] == "Read the file before quoting it"

    def test_the_delta_carries_the_provenance(self):
        reply = json.dumps({"bullets": [{"text": "x"}]})
        deltas = asyncio.run(reflector.reflect_contrast(
            _asker(reply), _record(session="s1"), _record(session="s2")))
        assert "s1" in deltas[0]["provenance"]
        assert "s2" in deltas[0]["provenance"]

    def test_the_model_sees_both_runs(self):
        reply = json.dumps({"bullets": [{"text": "x"}]})
        ask = _asker(reply)
        asyncio.run(reflector.reflect_contrast(
            ask, _record(task="the question"), _record(session="s2")))
        prompt = ask.calls[0][-1]["content"]
        assert "the question" in prompt
        assert "RUN THAT WENT WELL" in prompt
        assert "RUN THAT WENT BADLY" in prompt

    def test_a_failing_model_yields_no_deltas(self):
        """A reflection that cannot run is no reflection."""
        async def broken(messages):
            raise RuntimeError("the model is down")
        deltas = asyncio.run(reflector.reflect_contrast(
            broken, _record(), _record(session="s2")))
        assert deltas == []

    def test_a_garbled_reply_yields_no_deltas(self):
        deltas = asyncio.run(reflector.reflect_contrast(
            _asker("I think the bad run should have searched first"),
            _record(), _record(session="s2")))
        assert deltas == []

    def test_a_missing_side_yields_no_deltas(self):
        reply = json.dumps({"bullets": [{"text": "x"}]})
        deltas = asyncio.run(reflector.reflect_contrast(
            _asker(reply), _record(), {}))
        assert deltas == []

    def test_too_many_bullets_are_cut(self):
        bullets = [{"text": "b%d" % i} for i in range(10)]
        reply = json.dumps({"bullets": bullets})
        deltas = asyncio.run(reflector.reflect_contrast(
            _asker(reply), _record(), _record(session="s2")))
        assert len(deltas) <= reflector.MAX_BULLETS_PER_CONTRAST


# ── the corpus ──────────────────────────────────────────────────────────────

class TestReflectCorpus:
    """The whole corpus, one contrast at a time."""

    def test_a_contrast_is_found_and_reflected_on(self):
        good = _record(session="s1", verdict="clean")
        bad = _record(session="s2", verdict="issues",
                      tools=[{"name": "search", "ok": False}])
        reply = json.dumps({"bullets": [{"text": "Check the date first"}]})
        deltas = asyncio.run(reflector.reflect_corpus(
            _asker(reply), [good, bad]))
        assert deltas
        assert deltas[0]["text"] == "Check the date first"

    def test_a_corpus_with_no_contrast_yields_nothing(self):
        """Every run went well: there is nothing to contrast, and
        inventing a contrast would be inventing a lesson."""
        records = [_record(session="s%d" % i, verdict="clean") for i in range(5)]
        deltas = asyncio.run(reflector.reflect_corpus(
            _asker(json.dumps({"bullets": [{"text": "x"}]})), records))
        assert deltas == []

    def test_the_holdout_is_excluded(self):
        """A bullet written from the exam is a bullet written from the exam.

        The reflector must not see a holdout run, and -- because the
        corpus holds the same question asked many times -- it must not
        see a *sibling* of one either.
        """
        import os
        import shutil
        import tempfile

        root = tempfile.mkdtemp()
        ruler_root = os.path.join(root, "ruler")
        os.makedirs(ruler_root, exist_ok=True)
        config = {"learn": {"path": os.path.join(root, "learned"),
                            "eval": {"path": ruler_root}}}
        try:
            with open(os.path.join(ruler_root, "holdout.jsonl"), "w",
                      encoding="utf-8") as f:
                f.write(json.dumps({"id": "s1#1",
                                    "task": "find the weather forecast",
                                    "response": "held out"}) + "\n")
            good = _record(session="s1", verdict="clean")
            bad = _record(session="s2", verdict="issues",
                          tools=[{"name": "search", "ok": False}])
            reply = json.dumps({"bullets": [{"text": "x"}]})
            deltas = asyncio.run(reflector.reflect_corpus(
                _asker(reply), [good, bad], config_data=config))
            assert deltas == [], "the reflector used a holdout run"
        finally:
            shutil.rmtree(root, ignore_errors=True)
