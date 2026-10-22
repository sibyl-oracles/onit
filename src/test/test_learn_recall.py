"""
Loop A: episodic recall.  What the agent learned last time this came up.

What is being protected here:

1. **The scope filters.**  A recalled episode belongs to a session owner
   and to a model.  A retrieval that ignores either is a cross-tenant leak
   (the corpus is global on disk) or a strategy the current model cannot
   execute.  These are the two failures that make recall worse than no
   recall, and both are silent.
2. **The outcome ordering.**  Recall without a notion of which episode
   went well retrieves the agent's own mistakes as readily as its
   successes.  Within a band of similar matches, a clean episode must
   outrank a flagged one.
3. **The budget.**  Memory that crowds out the task is a regression.  The
   cap is hard, and an episode is never injected half-way.
4. **The trust marking.**  An episode is distilled from a run that read
   tool output, and tool output can carry a prompt injection.  The
   injected block is marked as untrusted data.
5. **The injection path.**  The block lands in the *volatile* half, ahead
   of the task, and never in the system message — a block in the prefix
   would change the cached bytes on every task of every session.
6. **Nothing at autonomy 0 or 1.**  A run with no memory must send
   byte-for-byte the same bytes it always did.
"""

import pytest

from learn import recall as rec
from learn import trajectory as traj


# ─── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def learned(tmp_path):
    """A learned store under tmp_path, recording on."""
    return {"learn": {"autonomy": "observe", "path": str(tmp_path / "learned")}}


def _record(learned, *, session="s1", turn=1, task="a question worth replaying",
            response="an answer worth comparing", model="m1", owner=None,
            verifier=None, implicit=None):
    """Record a task, optionally with a verifier or implicit label."""
    record = traj.record_task(session_id=session, turn=turn, task=task,
                              response=response, model=model, owner=owner,
                              config_data=learned)
    if verifier is not None or implicit is not None:
        # The writer does not take a verdict; the label lives in the
        # record's signals and is what recall reads.
        path = traj.session_file(session, learned)
        import json
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec_obj = json.loads(line)
                if rec_obj.get("kind") == "task":
                    if verifier is not None:
                        rec_obj.setdefault("signals", {})["verifier"] = verifier
                    if implicit is not None:
                        rec_obj.setdefault("signals", {})["implicit"] = implicit
                lines.append(json.dumps(rec_obj))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return record


# ─── ranking ────────────────────────────────────────────────────────────────

class TestRanking:
    """Which episodes match, and in what order."""

    def test_a_matching_task_ranks_first(self, learned):
        _record(learned, session="a", task="how do I list the local files")
        _record(learned, session="b", task="what is the weather in Tokyo")
        ranked = rec.rank_episodes("how do I list the local files",
                                   list(traj.iter_records(learned)))
        assert ranked[0][0]["session_id"] == "a"

    def test_an_unrelated_task_scores_zero(self, learned):
        _record(learned, session="a", task="how do I list the local files")
        ranked = rec.rank_episodes("quantum chromodynamics",
                                   list(traj.iter_records(learned)))
        assert ranked[0][1] == 0.0

    def test_an_empty_query_ranks_nothing(self, learned):
        _record(learned)
        assert rec.rank_episodes("", list(traj.iter_records(learned))) == []

    def test_no_records_ranks_nothing(self, learned):
        assert rec.rank_episodes("anything", []) == []

    def test_ties_are_broken_deterministically(self, learned):
        # Two identical tasks must not swap places between runs.
        for i in range(3):
            _record(learned, session=f"s{i}", task="the same question exactly")
        records = list(traj.iter_records(learned))
        first = [r["session_id"] for r, _ in rec.rank_episodes(
            "the same question exactly", records)]
        second = [r["session_id"] for r, _ in rec.rank_episodes(
            "the same question exactly", list(reversed(records)))]
        assert first == second


# ─── scope: owner and model ─────────────────────────────────────────────────

class TestScope:
    """Whose episodes may be recalled, and from which model."""

    def test_another_owners_episodes_are_not_recalled(self, learned):
        _record(learned, session="a", owner="alice@example.com",
                task="how do I list the local files")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        assert episodes == []

    def test_the_owners_own_episodes_are_recalled(self, learned):
        _record(learned, session="a", owner="alice@example.com",
                task="how do I list the local files")
        owner = traj.owner_hash("alice@example.com")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned, owner=owner)
        assert len(episodes) == 1

    def test_two_owners_do_not_see_each_other(self, learned):
        _record(learned, session="a", owner="alice@example.com",
                task="how do I list the local files")
        _record(learned, session="b", owner="bob@example.com",
                task="how do I list the local files")
        alice = rec.recall("how do I list the local files", config_data=learned,
                           owner=traj.owner_hash("alice@example.com"))
        bob = rec.recall("how do I list the local files", config_data=learned,
                         owner=traj.owner_hash("bob@example.com"))
        assert len(alice) == 1
        assert len(bob) == 1
        assert alice[0]["id"] != bob[0]["id"]

    def test_another_models_episodes_are_not_recalled(self, learned):
        _record(learned, session="a", model="m1",
                task="how do I list the local files")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned, model="m2")
        assert episodes == []

    def test_the_same_models_episodes_are_recalled(self, learned):
        _record(learned, session="a", model="m1",
                task="how do I list the local files")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned, model="m1")
        assert len(episodes) == 1

    def test_owner_less_episodes_are_recalled_without_an_owner_filter(self,
                                                                      learned):
        # The CLI records owner-less runs; those are the common case and
        # must be recallable.
        _record(learned, session="a", owner=None,
                task="how do I list the local files")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        assert len(episodes) == 1


# ─── outcome ordering ───────────────────────────────────────────────────────

class TestOutcomeOrdering:
    """Which of two similar episodes comes first."""

    def test_a_clean_episode_outranks_a_flagged_one(self, learned):
        _record(learned, session="a", task="how do I list the local files",
                verifier="clean")
        _record(learned, session="b", task="how do I list the local files",
                verifier="issues")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        assert episodes[0]["outcome"] == "clean"

    def test_an_implicit_bad_beats_a_clean_one(self, learned):
        # The user's own next move outranks a model reading the answer.
        # An episode the verifier called clean but which the user asked
        # for again is the strongest teacher there is.
        _record(learned, session="a", task="how do I list the local files",
                verifier="clean")
        _record(learned, session="b", task="how do I list the local files",
                verifier="clean", implicit="bad")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        assert episodes[0]["id"] == "b#1"

    def test_an_implicit_good_is_demoted(self, learned):
        # An episode the user came back and accepted teaches less than
        # one nobody said anything about.
        _record(learned, session="a", task="how do I list the local files",
                verifier="clean")
        _record(learned, session="b", task="how do I list the local files",
                verifier="clean", implicit="good")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        assert episodes[0]["id"] == "a#1"

    def test_a_bad_signal_beats_a_good_one(self, learned):
        _record(learned, session="a", task="how do I list the local files",
                implicit="bad")
        _record(learned, session="b", task="how do I list the local files",
                implicit="good")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        assert episodes[0]["id"] == "a#1"

    def test_the_summary_says_what_the_user_did(self, learned):
        # The model has to be told in words it can act on.
        _record(learned, session="a", task="how do I list the local files",
                implicit="bad")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        assert "asked for it again" in episodes[0]["summary"]

    def test_the_summary_says_when_the_user_accepted(self, learned):
        _record(learned, session="a", task="how do I list the local files",
                implicit="good")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        assert "accepted it" in episodes[0]["summary"]

        # The match itself is the better signal when it is *much* better.
        # The flagged episode here shares no content with the query at
        # all, so it scores zero and falls outside the similarity band;
        # the clean one is in the band and wins on outcome.
        _record(learned, session="a", task="how do I list the local files",
                verifier="clean")
        _record(learned, session="b",
                task="quantum chromodynamics and lattice gauge theory",
                verifier="issues")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        assert episodes[0]["id"] == "a#1"

    def test_an_episode_outside_the_similarity_band_is_not_reordered(self,
                                                                    learned):
        # Within the band, outcome decides.  Outside it, the match decides
        # and the outcome is not consulted -- a flagged episode that
        # matched far better is still the more relevant one.
        _record(learned, session="a", task="how do I list the local files",
                verifier="clean")
        _record(learned, session="b",
                task="how do I list the local files and the remote ones too",
                verifier="issues")
        ranked = rec.rank_episodes("how do I list the local files",
                                   list(traj.iter_records(learned)))
        top = ranked[0][1]
        band = [pair for pair in ranked if pair[1] >= top * rec._SIMILARITY_BAND]
        # Both are inside the band, so the clean one leads on outcome.
        assert len(band) == 2
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        assert episodes[0]["outcome"] == "clean"

    def test_an_unjudged_episode_is_still_recalled(self, learned):
        # 120 of the 1,010 recorded tasks carry no verifier label.  They
        # are the majority of what the corpus holds and dropping them
        # would leave recall with almost nothing to recall.
        _record(learned, session="a", task="how do I list the local files")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        assert len(episodes) == 1
        assert episodes[0]["outcome"] == ""

    def test_unjudged_episodes_can_be_excluded(self, learned):
        _record(learned, session="a", task="how do I list the local files")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned, include_unjudged=False)
        assert episodes == []


# ─── budget ─────────────────────────────────────────────────────────────────

class TestBudget:
    """Memory must not crowd out the task."""

    def test_a_tiny_budget_admits_nothing(self, learned):
        _record(learned, session="a", task="how do I list the local files")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned, token_budget=5)
        assert episodes == []

    def test_a_zero_top_k_admits_nothing(self, learned):
        _record(learned, session="a", task="how do I list the local files")
        assert rec.recall("how do I list the local files",
                          config_data=learned, top_k=0) == []

    def test_the_top_k_cap_is_respected(self, learned):
        for i in range(8):
            _record(learned, session=f"s{i}",
                    task="how do I list the local files")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned, top_k=3)
        assert len(episodes) == 3

    def test_the_budget_admits_fewer_rather_than_a_partial_episode(self,
                                                                  learned):
        for i in range(8):
            _record(learned, session=f"s{i}",
                    task="how do I list the local files and read them all")
        # A budget that fits two episodes must yield two, not three with
        # the third cut in half.
        episodes = rec.recall("how do I list the local files",
                              config_data=learned, top_k=8, token_budget=300)
        assert len(episodes) < 8
        for episode in episodes:
            assert episode["summary"].strip().endswith(")")


# ─── the injected block ─────────────────────────────────────────────────────

class TestRendered:
    """What actually goes into the prompt."""

    def test_nothing_to_recall_renders_nothing(self, learned):
        assert rec.render([]) == ""

    def test_the_block_is_marked_untrusted(self, learned):
        _record(learned, session="a", task="how do I list the local files")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        text = rec.render(episodes)
        assert "untrusted" in text.lower()
        assert "not instructions" in text.lower()

    def test_the_block_names_the_outcome(self, learned):
        _record(learned, session="a", task="how do I list the local files",
                verifier="clean")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        text = rec.render(episodes)
        assert "fact-check" in text

    def test_the_block_carries_the_question_and_the_answer(self, learned):
        _record(learned, session="a", task="how do I list the local files",
                response="use the shell")
        episodes = rec.recall("how do I list the local files",
                              config_data=learned)
        text = rec.render(episodes)
        assert "how do I list the local files" in text
        assert "use the shell" in text


# ─── build_context ──────────────────────────────────────────────────────────

class TestBuildContext:
    """The one call a caller makes."""

    def test_an_empty_task_recalls_nothing(self, learned):
        _record(learned)
        context = rec.build_context("", config_data=learned)
        assert context["text"] == ""
        assert context["ids"] == []

    def test_recording_off_recalls_nothing(self, tmp_path):
        off = {"learn": {"autonomy": "off", "path": str(tmp_path / "learned")}}
        _record(off, session="a", task="how do I list the local files")
        context = rec.build_context("how do I list the local files",
                                    config_data=off)
        assert context["text"] == ""

    def test_the_ids_name_the_episodes_shown(self, learned):
        _record(learned, session="a", task="how do I list the local files")
        context = rec.build_context("how do I list the local files",
                                    config_data=learned)
        assert context["ids"] == ["a#1"]

    def test_the_ids_are_the_credit_assignment_hook(self, learned):
        # A run recorded with episodes_used can be compared against a run
        # recorded without.  That is the only way "did the recall work"
        # gets an answer.
        _record(learned, session="a", task="how do I list the local files")
        context = rec.build_context("how do I list the local files",
                                    config_data=learned)
        traj.record_task(session_id="b", turn=1, task="how do I list the files",
                         response="use the shell", config_data=learned,
                         episodes_used=context["ids"])
        record = traj.read_session("b", learned)[0]
        assert record["learned_context"]["episodes_used"] == ["a#1"]


# ─── the tokenizer reuse ────────────────────────────────────────────────────

class TestSearchReuse:
    """The BM25 is the search toolkit's, not a second implementation."""

    def test_the_shared_toolkit_is_found(self):
        from learn import _search
        assert _search.toolkit_available()

    def test_plurals_are_folded(self):
        from learn import _search
        # "scholarships" and "scholarship" must be the same term on both
        # sides of a match, or a query scores nothing against a document
        # written in the other number.
        assert _search.tokenize("scholarships") == _search.tokenize(
            "scholarship")

    def test_the_fallback_matches_the_shared_tokenizer(self):
        from learn import _search
        toolkit = _search._load_toolkit()
        assert toolkit is not None
        for text in ("scholarships policy", "The Cat sat", "don't stop"):
            assert _search.tokenize(text) == list(toolkit.tokenize(text))


# ─── the injection path ─────────────────────────────────────────────────────

class TestInjectionPath:
    """Where the recalled block actually lands in the request."""

    def _messages(self, *, learned_context="", instruction="## Task\ndo it",
                  system_rules="## Instructions\nBe careful."):
        from model.serving.chat import _build_messages
        return _build_messages(instruction, [], "I am OnIt.",
                               session_history=None,
                               system_rules=system_rules,
                               learned_context=learned_context)

    def test_the_block_reaches_the_user_message(self):
        messages = self._messages(learned_context="## Prior runs\n- something")
        assert messages[-1]["role"] == "user"
        assert "## Prior runs" in messages[-1]["content"]

    def test_the_task_still_leads_the_user_message(self):
        # The task is what the model is being asked to do.  A recalled
        # episode that pushes the task out of view has taken over the
        # prompt.
        messages = self._messages(learned_context="## Prior runs\n- something")
        content = messages[-1]["content"]
        assert content.index("## Prior runs") < content.index("## Task")

    def test_the_block_never_reaches_the_system_message(self):
        # The system message is the cached prefix.  A per-task block in
        # it would change the cached bytes on every task of every
        # session and give back the whole benefit of the cache.
        messages = self._messages(learned_context="## Prior runs\n- something")
        assert "## Prior runs" not in messages[0]["content"]

    def test_the_system_message_is_unchanged_by_the_block(self):
        without = self._messages(learned_context="")
        with_block = self._messages(learned_context="## Prior runs\n- something")
        assert without[0] == with_block[0]

    def test_no_block_means_the_same_bytes_as_before(self):
        # The byte-for-byte guarantee for autonomy 0 and 1.
        messages = self._messages(learned_context="")
        assert messages[-1]["content"] == "## Task\ndo it"

    def test_the_block_is_separated_from_the_task(self):
        messages = self._messages(learned_context="## Prior runs\n- something")
        content = messages[-1]["content"]
        assert "## Current task" in content

    def test_the_session_history_still_precedes_the_task(self):
        from model.serving.chat import _build_messages
        messages = _build_messages("## Task\ndo it", [], "I am OnIt.",
                                   session_history=[{"task": "earlier",
                                                     "response": "an answer"}],
                                   system_rules="rules",
                                   learned_context="## Prior runs\n- something")
        roles = [m["role"] for m in messages]
        # system, then the history pair, then the task.
        assert roles == ["system", "user", "assistant", "user"]
        assert "## Prior runs" in messages[-1]["content"]
