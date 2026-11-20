"""
Hardening: the adversarial edges of the wiki and dreaming stack.

The happy paths are covered by test_learn_dream, test_learn_memory,
test_learn_gate and test_memory_chain.  What this file adds is the
edges those suites do not reach -- the inputs a real run produces and
a stubbed test does not:

1. **The extractor's reply, as models actually write it** -- fenced
   JSON, reasoning tags around it, a tuple instead of a string, prose
   instead of JSON.  Every one of these arrived in the 2026-11-17 and
   2026-11-18 gate runs.
2. **The operational filter, marker by marker** -- the 2026-11-20
   discordant pair came through one phrase; the filter has to hold on
   the rest of the family, and stay narrow enough to keep real
   knowledge.
3. **The store's worst-case I/O** -- a page that cannot be written, an
   index that cannot land, a staging file past its second chance.
4. **The grader's failure modes as a run hits them** -- the empty
   reply the retry exists for, the whitespace answer, the unknown
   letter.
5. **The gate's arithmetic under degenerate inputs** -- an arm with no
   items, a zero error bar, a coverage gap computed against an empty
   wiki.
6. **The per-jail stdio identity** -- the seventh placebo's fix, at
   the URL level the pooling actually keys on.
"""

import asyncio
import json
import os

import pytest

from onit.learn import ab, dream, memory as mem, runner, trajectory as traj


@pytest.fixture
def world(tmp_path):
    """A learned store with trajectories, autonomy adapt, memory on."""
    learned = tmp_path / "learned"
    (learned / "trajectories").mkdir(parents=True)
    return {
        "learn": {
            "autonomy": "adapt",
            "path": str(learned),
            "memory": {"path": str(learned / "memory"), "enabled": True},
        }
    }


def _session(world, session_id, tasks, ts="2026-11-16T10:00:00"):
    """Write a session of task records, the way the recorder does."""
    path = os.path.join(traj.trajectory_dir(world), "%s.jsonl" % session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for index, (task, response) in enumerate(tasks):
            record = {
                "kind": "task",
                "session_id": session_id,
                "turn": index,
                "task": task,
                "response": response,
                "model": "m1",
                "owner": None,
                "ts": ts,
                "trajectory": [],
                "signals": {},
            }
            f.write(json.dumps(record) + "\n")
    return path


def _read(world, session_id):
    return traj.read_session(session_id, world)


def _ask(items):
    async def ask(messages):
        return json.dumps({"items": items})

    return ask


def _item(task="a question", response="the right answer", item_id="s1#1"):
    return {
        "id": item_id,
        "task": task,
        "response": response,
        "model": "m1",
        "owner": None,
        "session_id": "s1",
        "turn": 1,
    }


def _page(slug, title, facts):
    page = mem.make_page(slug, title)
    deltas = [
        {"op": "add_fact", "text": t, "provenance": ["s1#%d" % (i + 1)]}
        for i, t in enumerate(facts)
    ]
    page, _ = mem.apply_deltas(page, deltas)
    return page


class StubAgent:
    """An agent that answers with a fixed string and records its calls."""

    def __init__(self, answer="the right answer", config=None):
        self.answer = answer
        self.config = config or {}
        self.calls = []

    async def process_task(self, task, **kwargs):
        self.calls.append({"task": task, "kwargs": kwargs})
        return self.answer


# ── the extractor's reply, as models actually write it ───────────────


class TestTheReplyAsModelsWriteIt:
    """_extract_json must survive the shapes an endpoint actually sends.

    A half-parsed candidate is injected into a wiki page as if it were
    established, so the parser is strict -- but strict must not mean
    brittle against the fence and the reasoning block that surround a
    well-formed reply.
    """

    def test_a_json_fence_is_unwrapped(self):
        text = 'Here is what the day established:\n```json\n{"items": [{"text": "the staging host is stg-7.internal", "kind": "fact", "cites": [1]}]}\n```'
        items = dream._extract_json(text)
        assert len(items) == 1
        assert items[0]["text"] == "the staging host is stg-7.internal"

    def test_a_bare_fence_without_the_word_json_is_unwrapped(self):
        text = '```\n{"items": []}\n```'
        assert dream._extract_json(text) == []

    def test_a_reasoning_block_around_the_json_is_stripped(self):
        text = '<think>let me consider the transcript</think>{"items": [{"text": "the broker is mq-2.internal", "kind": "fact"}]}'
        items = dream._extract_json(text)
        assert len(items) == 1

    def test_a_tuple_reply_is_unwrapped(self):
        # Some client paths return (text, usage) pairs.
        assert dream._extract_json(('{"items": []}', 12)) == []

    def test_a_top_level_list_is_not_items(self):
        assert dream._extract_json('[{"text": "a fact"}]') == []

    def test_a_dict_without_items_is_not_items(self):
        assert dream._extract_json('{"facts": [{"text": "a fact"}]}') == []

    def test_non_dict_entries_are_dropped_not_raised(self):
        text = '{"items": ["prose", 3, null, {"text": "a fact"}]}'
        items = dream._extract_json(text)
        assert len(items) == 1 and items[0]["text"] == "a fact"

    def test_junk_is_an_empty_list(self):
        assert dream._extract_json("the day established nothing at all") == []


class TestCleaningRejectsTheUnverifiable:
    """_clean_candidate's pre-cite gates: too short, or about itself."""

    def test_a_candidate_shorter_than_a_sentence_is_rejected(self, world):
        assert dream._clean_candidate({"text": "host is up", "cites": [1]}, "1: task") is None

    def test_a_candidate_about_the_session_is_rejected_before_the_cite_check(self, world):
        transcript = "1: deploy the service\n2: deployed to stg-7.internal"
        item = {"text": "The session was asked to deploy the service today", "cites": [1]}
        assert dream._clean_candidate(item, transcript) is None


class TestTheOperationalFilterHoldsTheWholeFamily:
    """Every marker class the 2026-11-20 fix installed must fire, and
    the filter must stay narrow -- a wiki that drops real knowledge is
    worth nothing either."""

    def _rejected(self, text):
        return dream._operational(text.lower())

    @pytest.mark.parametrize(
        "text",
        [
            "pushed to origin/main as commit 6da0713",  # the 2026-11-20 fact
            "the branch was merged into main this morning",
            "opened a pull request for the fix",
            "restarted service after the config change",
            "ran the migration on the production database",
            "the file was force-pushed to the remote",
        ],
    )
    def test_each_operational_family_is_rejected(self, text):
        assert self._rejected(text)

    @pytest.mark.parametrize(
        "text",
        [
            "the staging host is stg-7.internal",  # genuine knowledge, kept
            "deploys here are blue-green by design",  # a policy, not an action
            "the broker config lives in etc/mq/broker.conf",  # a location
            "the service restarts take about ninety seconds",  # a property
        ],
    )
    def test_knowledge_that_merely_touches_the_words_survives(self, text):
        assert not self._rejected(text)

    def test_an_operational_fact_is_dropped_end_to_end(self, world):
        _session(world, "s1", [("push the branch", "pushed to origin/main as commit 6da0713")])
        records = _read(world, "s1")
        candidates, _, _, _ = asyncio.run(
            dream.extract(
                _ask([{"text": "pushed to origin/main as commit 6da0713", "kind": "fact", "cites": [2]}]),
                records,
            )
        )
        assert candidates == []


# ── the store's worst-case I/O ────────────────────────────────────────


class TestTheStoreUnderFailure:
    """Best-effort writes: a failure must degrade to "did less", never
    to a page that reads as present but is not."""

    def test_a_page_that_cannot_be_written_reports_it(self, tmp_path):
        # A directory where the store file must land: every write fails.
        target = tmp_path / "wiki"
        target.mkdir()
        (target / "onit-repo.json").mkdir()  # a directory wearing a page's name
        page = _page("onit-repo", "OnIt repo", ["the main branch is main"])
        assert mem.write_page(str(target / "onit-repo.json"), page) is False

    def test_an_index_that_cannot_land_is_silent(self, tmp_path):
        (tmp_path / "INDEX.md").mkdir()  # a directory where the file must go
        assert mem.write_index([_page("s", "S", ["f"])], root=str(tmp_path)) is False

    def test_a_log_line_survives_a_missing_directory(self, tmp_path):
        # append_log creates its directory; a fresh root is not an error.
        mem.append_log(["== pass 1 =="], root=str(tmp_path / "fresh"))
        assert (tmp_path / "fresh" / "dream.log").exists()

    def test_an_empty_log_write_is_a_noop(self, tmp_path):
        mem.append_log([], root=str(tmp_path))
        assert not os.path.exists(str(tmp_path / "dream.log"))

    def test_a_page_path_is_always_slugified(self, tmp_path):
        path = mem.page_path("OnIt Repo!", root=str(tmp_path))
        assert path.endswith(".json") and " " not in os.path.basename(path)


class TestStagingExpiry:
    """A staging note gets two passes; the third sees it expire."""

    def test_a_note_staged_is_a_note_on_disk(self, world):
        assert dream.stage_session_note("s1", "the staging host is stg-7.internal", world)
        base = os.path.join(world["learn"]["memory"]["path"], mem.SESSIONS_SUBDIR)
        assert any(n.endswith(".md") for n in os.listdir(base))

    def test_an_empty_note_is_not_staged(self, world):
        assert dream.stage_session_note("s1", "   ", world) is False

    def test_a_note_past_two_passes_expires(self, world):
        dream.stage_session_note("s1", "a fact worth one chance", world)
        mem.save_state({"passes": 2}, world)
        log = []
        expired = dream._expire_staging(world, log_lines=log)
        assert expired == 1
        base = os.path.join(world["learn"]["memory"]["path"], mem.SESSIONS_SUBDIR)
        assert not os.listdir(base)

    def test_a_fresh_note_survives_its_first_passes(self, world):
        dream.stage_session_note("s1", "a fact with time", world)
        mem.save_state({"passes": 1}, world)
        assert dream._expire_staging(world) == 0

    def test_a_missing_staging_directory_expires_nothing(self, tmp_path):
        assert dream._expire_staging({"learn": {"path": str(tmp_path / "none")}}) == 0


# ── the pass's reconciliation edges ──────────────────────────────────


class TestReconciliationEdges:
    """Prune, contradictions resolved, and the title a page is founded
    under -- the parts of dream_once a one-fact stub run never reach."""

    def test_a_stale_page_is_pruned_by_a_pass(self, world):
        page = _page("staging", "Staging", ["fact %d about the staging host" % i for i in range(6)])
        page["updated"] = "2026-01-01"
        mem.save_page(page, world)
        result = asyncio.run(dream.run_dream(world, ask=_ask([]), force=True))
        assert result.get("ran") is True
        surviving = [p for p in mem.list_pages(world) if p["slug"] == "staging"][0]
        assert len(surviving["facts"]) <= mem.PRUNE_KEEP_FACTS

    def test_a_fresh_page_is_not_pruned(self, world):
        page = _page("staging", "Staging", ["fact %d about the staging host" % i for i in range(6)])
        mem.save_page(page, world)
        asyncio.run(dream.run_dream(world, ask=_ask([]), force=True))
        surviving = [p for p in mem.list_pages(world) if p["slug"] == "staging"][0]
        assert len(surviving["facts"]) == 6

    def test_an_open_contradiction_is_resolved_by_a_pass(self, world):
        page = _page("staging", "Staging", ["the staging host is stg-7.internal"])
        mem.save_page(page, world)
        assert mem.record_contradiction(
            "staging",
            claim="the staging host is stg-7.internal",
            correction="the staging host moved to stg-8.internal",
            config_data=world,
        )
        result = asyncio.run(dream.run_dream(world, ask=_ask([]), force=True))
        assert result.get("contradictions_cleared") == 1
        surviving = [p for p in mem.list_pages(world) if p["slug"] == "staging"][0]
        assert surviving["contradictions"] == []

    def test_a_page_title_is_the_most_common_topic(self):
        cands = [{"text": "a", "topic": "staging"}, {"text": "b", "topic": "staging"}, {"text": "c", "topic": "queue"}]
        assert "staging" in dream._title_from(cands).lower()

    def test_a_title_falls_back_to_the_first_text(self):
        assert dream._title_from([{"text": "the staging host is stg-7.internal"}])

    def test_a_candidate_finds_its_existing_page(self, world):
        page = _page("staging", "Staging", ["the staging host is stg-7.internal"])
        mem.save_page(page, world)
        pages = mem.list_pages(world)
        home = dream._find_home({"text": "the staging host is stg-7.internal, verified"}, pages)
        assert home is not None and home["slug"] == "staging"
        # BM25 scores every page against every query -- the threshold is
        # "above zero", deliberately loose.  A stranger must at least not
        # outrank the true home.
        stranger = dream._find_home({"text": "the quarterly revenue forecast doubled"}, pages)
        if stranger is not None:
            assert stranger["slug"] == "staging"

    def test_a_candidate_below_adapt_runs_nothing(self, tmp_path):
        frozen = {"learn": {"autonomy": "observe", "path": str(tmp_path / "l")}}
        assert asyncio.run(dream.dream_once(_ask([]), frozen))["ran"] is False


class TestTheLockUnderFailure:
    """A lock the store cannot create must fail the pass as busy, not
    run it unsynchronized."""

    def test_an_unwritable_store_is_busy_not_unsynchronized(self, tmp_path):
        # The memory directory exists but is not writable: os.open's
        # O_CREAT|O_EXCL raises OSError, not FileExistsError.
        base = tmp_path / "memory"
        base.mkdir()
        os.chmod(base, 0o500)
        try:
            frozen = {"learn": {"path": str(tmp_path / "l"), "memory": {"path": str(base)}}}
            with pytest.raises(dream.DreamBusy):
                with dream._Lock(frozen):
                    pass
        finally:
            os.chmod(base, 0o755)


# ── the grader, as a run hits it ─────────────────────────────────────


class TestTheGraderUnderFailure:
    """The 2026-11-17 run's grader spent its budget thinking and never
    wrote a verdict.  The retry and the visible failure are load-bearing."""

    def test_an_empty_reply_is_retried_and_the_second_wins(self):
        replies = ["", "GRADE: C"]

        async def flaky(messages):
            return replies.pop(0)

        verdict = asyncio.run(runner.grade(flaky, "task", "ref", "an answer"))
        assert verdict["grade"] == "C"
        assert not verdict.get("grading_error")

    def test_two_empty_replies_are_a_visible_grading_error(self):
        async def hollow(messages):
            return ""

        verdict = asyncio.run(runner.grade(hollow, "task", "ref", "an answer"))
        assert verdict["grade"] == ""
        assert "no text" in verdict["grading_error"]

    def test_a_whitespace_answer_is_not_attempted(self):
        async def ask(messages):
            return "GRADE: C"

        verdict = asyncio.run(runner.grade(ask, "task", "ref", "   "))
        assert verdict["grade"] == runner.GRADE_NOT_ATTEMPTED

    def test_a_grade_embedded_after_prose_is_read(self):
        assert runner.parse_grade("The rubric: GRADE: I because the push was unverified.") == "I"

    def test_a_grade_with_trailing_words_is_read(self):
        assert runner.parse_grade("GRADE: N — the agent declined.") == "N"

    def test_the_letter_must_lead_the_tail(self):
        assert runner.parse_grade("GRADE: because I think so") == ""


# ── the gate's arithmetic under degenerate inputs ────────────────────


class TestTheComparisonUnderDegenerateInputs:
    """An arm with no items, a zero error bar, and the paired stats a
    real (partially graded) run produces."""

    def test_an_empty_pair_of_arms_compares_to_zero(self):
        result = runner.compare([], [])
        assert result["delta"] == 0.0
        assert result["stderr"] == 0.0

    def test_perfect_separation_has_a_zero_error_bar(self):
        control = [dict(grade="I", error="") for _ in range(10)]
        treatment = [dict(grade="C", error="") for _ in range(10)]
        result = runner.compare(control, treatment)
        assert result["stderr"] == 0.0
        assert result["significant"] is True

    def test_a_summary_of_nothing_is_zeroed(self):
        summary = runner.summarise([])
        assert summary["n"] == 0 and summary["accuracy"] == 0.0

    def test_paired_stats_count_both_letter_churn_and_flips(self):
        control = [
            {"id": "a", "grade": "C", "grading_error": ""},
            {"id": "b", "grade": "I", "grading_error": ""},
            {"id": "c", "grade": "I", "grading_error": ""},
            {"id": "d", "grade": "", "grading_error": "endpoint down"},
        ]
        treatment = [
            {"id": "a", "grade": "I", "grading_error": ""},  # C -> I: a flip
            {"id": "b", "grade": "N", "grading_error": ""},  # I -> N: letter churn, both wrong
            {"id": "c", "grade": "I", "grading_error": ""},  # unchanged
            {"id": "d", "grade": "C", "grading_error": ""},  # ungraded control: no pair
        ]
        paired = runner._paired_stats(control, treatment)
        assert paired["pairs"] == 3
        assert paired["control_correct"] == 1
        # The treatment's only correct answer sits on the item the
        # control never had graded: it is not a pair, so it does not
        # count in the paired totals.
        assert paired["treatment_correct"] == 0
        assert paired["discordant"] == 1
        assert paired["grade_changes"] == 2

    def test_an_arm_missing_items_makes_no_pairs(self):
        control = [{"id": "a", "grade": "C", "grading_error": ""}]
        paired = runner._paired_stats(control, [])
        assert paired["pairs"] == 0

    def test_the_jail_refusal_count_reads_the_arm_s_own_words(self):
        results = [
            {"answer": "every tool call was refused: data-path mismatch"},
            {"answer": "the path must be within the designated data directory"},
            {"answer": "the task is done and pushed to origin"},
            {"answer": ""},
        ]
        assert runner._jail_refusal_count(results) == 2

    def test_a_coverage_gap_against_an_empty_wiki_names_every_item(self, tmp_path):
        items = [_item("bake the bread loaf", item_id="s1#%d" % i) for i in range(3)]
        gaps = runner._coverage_gaps(items, str(tmp_path))
        assert [g for g, _ in gaps] == ["s1#0", "s1#1", "s1#2"]

    def test_a_covered_item_is_not_a_gap(self, tmp_path):
        store = runner._arm_store(str(tmp_path), runner.WIKI_TREATMENT)
        config = runner._config_for_wiki_arm({}, True, store=store)
        mem.save_page(_page("bread", "Bread baking", ["the oven runs at 230c"]), config)
        gaps = runner._coverage_gaps([_item("bake the bread loaf")], str(tmp_path))
        assert gaps == []

    def test_a_page_count_of_an_empty_root_is_zero(self, tmp_path):
        assert runner._wiki_page_count(str(tmp_path)) == 0

    def test_a_page_count_reads_the_treatment_store(self, tmp_path):
        store = runner._arm_store(str(tmp_path), runner.WIKI_TREATMENT)
        config = runner._config_for_wiki_arm({}, True, store=store)
        mem.save_page(_page("bread", "Bread baking", ["the oven runs at 230c"]), config)
        assert runner._wiki_page_count(str(tmp_path)) == 1

    def test_the_inheritance_skips_the_store_s_own_files(self, tmp_path):
        src = tmp_path / "src" / "trajectories"
        src.mkdir(parents=True)
        (src / "s1.jsonl").write_text('{"kind": "task"}\n', encoding="utf-8")
        (src / "loop.jsonl").write_text("event\n", encoding="utf-8")  # derived, not a session
        (src / "notes.txt").write_text("not a session\n", encoding="utf-8")
        dst = str(tmp_path / "dst")
        assert runner._inherit_trajectories(str(tmp_path / "src"), dst) == 1
        assert os.path.exists(os.path.join(dst, "trajectories", "s1.jsonl"))
        assert not os.path.exists(os.path.join(dst, "trajectories", "loop.jsonl"))


# ── the wiki gate end to end, with the result's instruments ─────────


class TestTheGateResultCarriesItsInstruments:
    """Every tripwire the last three runs earned must be present in the
    comparison dict a run hands to whoever reads the verdict."""

    async def _run(self, tmp_path, items, seed_ask=None):
        return await runner.run_wiki_ab(
            items,
            root=str(tmp_path),
            base_config={"learn": {"memory": {"path": str(tmp_path / "mem")}}},
            make_agent=lambda config: StubAgent(config=config),
            ask=lambda messages: "GRADE: C",
            seed_ask=seed_ask or _ask([]),
        )

    async def test_the_result_names_the_jail_and_the_coverage(self, tmp_path):
        result = await self._run(tmp_path, [_item("bake the bread loaf")] * 2)
        assert result["jail_refusals"] == {runner.WIKI_CONTROL: 0, runner.WIKI_TREATMENT: 0}
        # The stub wiki is empty, so both items -- one id, listed per
        # item -- are gaps.  Duplicates are per-item rows, not a bug.
        assert result["coverage_gaps"] == ["s1#1", "s1#1"]
        assert "paired" in result and "wiki_pages" in result and "wiki_seed" in result

    async def test_a_treatment_without_a_seed_ask_is_not_seeded(self, tmp_path):
        result = await self._run(tmp_path, [_item()], seed_ask=None)
        # seed_ask=None inherits the grader's ask; the stub grader's
        # reply is not extraction JSON, so the pass keeps nothing --
        # and the gate must still say so, never silently.
        assert result["wiki_pages"] == 0
        assert (result["wiki_seed"] or {}).get("candidates", 0) == 0

    async def test_the_verdict_of_a_seeded_wiki_carries_the_numbers(self, tmp_path):
        result = await self._run(tmp_path, [_item()])
        text = runner.format_wiki_verdict(result)
        assert "wiki-off" in text and "wiki-on" in text


class TestTheVerdictNamesTheSeedCauses:
    """The four zero-page seed diagnoses, each from its own stats."""

    def _verdict(self, result):
        return runner.format_wiki_verdict(result)

    def _base(self):
        control = [dict(grade="I", error="") for _ in range(10)]
        treatment = [dict(grade="I", error="") for _ in range(10)]
        result = runner.compare(control, treatment)
        result["wiki_pages"] = 0
        result["jail_refusals"] = {}
        result["paired"] = {}
        result["coverage_gaps"] = []
        result["results"] = {}
        result["summaries"] = {}
        return result

    def test_a_seed_that_never_ran_is_named(self):
        result = self._base()
        result["wiki_seed"] = {"ran": False, "reason": "another pass holds the lock"}
        assert "did not run" in self._verdict(result)

    def test_items_dropped_by_the_cite_cleaning_are_named(self):
        result = self._base()
        result["wiki_seed"] = {"ran": True, "candidates": 0, "parsed_items": 4, "empty_replies": 0}
        assert "cite-cleaning" in self._verdict(result)

    def test_hollow_replies_are_named(self):
        result = self._base()
        result["wiki_seed"] = {"ran": True, "candidates": 0, "parsed_items": 0, "empty_replies": 3}
        assert "endpoint failed the extractor" in self._verdict(result)

    def test_a_day_that_established_nothing_is_named(self):
        result = self._base()
        result["wiki_seed"] = {"ran": True, "candidates": 0, "parsed_items": 0, "empty_replies": 0}
        assert "established no reusable facts" in self._verdict(result)

    def test_policy_starvation_is_named(self):
        result = self._base()
        result["wiki_seed"] = {"ran": True, "candidates": 5, "parsed_items": 6, "pages_created": 0}
        assert "page policy starved" in self._verdict(result)

    def test_a_jail_mismatch_dominates_the_numbers(self):
        result = self._base()
        result["jail_refusals"] = {runner.WIKI_CONTROL: 0, runner.WIKI_TREATMENT: 10}
        text = self._verdict(result)
        assert "JAIL MISMATCH" in text


# ── the A/B driver ───────────────────────────────────────────────────


class TestTheAbDriver:
    """run_ab: the control's advantage, the arm order, the scratch root."""

    async def test_the_control_runs_first_and_unaided(self, tmp_path):
        order = []

        def factory(config):
            order.append(config["learn"]["autonomy"])
            return StubAgent(config=config)

        await ab.run_ab(
            [_item()],
            root=str(tmp_path),
            base_config={},
            with_grader=False,
            make_agent=factory,
        )
        assert order == ["off", "adapt"]

    async def test_a_custom_arm_list_runs_those_arms(self, tmp_path):
        seen = []

        def factory(config):
            seen.append(config["learn"]["autonomy"])
            return StubAgent(config=config)

        await ab.run_ab(
            [_item()],
            arms=("adapt", "adapt"),
            root=str(tmp_path),
            base_config={},
            with_grader=False,
            make_agent=factory,
        )
        assert seen == ["adapt", "adapt"]

    def test_the_default_root_is_a_temp_directory(self):
        root = ab.default_root()
        assert os.path.isdir(root)
        assert "onit-evalrun-" in os.path.basename(root)

    def test_a_boolean_timeout_is_never_unbounded(self):
        # ``int(True)`` is 1 and the branch keeps it: a config bug, but
        # not a replay blocker.  The contract under test is that a
        # nonsense timeout never survives as "wait forever".
        timeout = ab.base_config_for_run({"timeout": True})["timeout"]
        assert timeout is not None and timeout != -1


# ── the per-jail stdio identity ──────────────────────────────────────


class TestThePerJailStdioIdentity:
    """The seventh placebo's fix, at the level the client pools on: the
    URL.  Two jails are two servers; one jail is one server."""

    def test_a_server_without_a_jail_keeps_the_bare_url(self):
        from onit.type.tools import STDIO_SCHEME, stdio_url

        assert stdio_url("ToolsLocalMCPServer") == "%sToolsLocalMCPServer" % STDIO_SCHEME

    def test_the_jail_hash_is_in_the_url(self, tmp_path):
        from onit.type.tools import stdio_url

        url = stdio_url("srv", str(tmp_path))
        assert url.startswith("stdio://srv@")
        assert url != "stdio://srv"

    def test_macos_realpath_spellings_are_one_identity(self, tmp_path):
        from onit.type.tools import stdio_url

        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        spelled_one = os.path.join(str(tmp_path), "a", "b", "c")
        os.makedirs(spelled_one, exist_ok=True)
        spelled_two = os.path.join(str(tmp_path), "a", "b", "..", "b", "c")
        assert stdio_url("srv", spelled_one) == stdio_url("srv", spelled_two)

    def test_a_trailing_slash_is_not_a_second_jail(self, tmp_path):
        from onit.type.tools import stdio_url

        assert stdio_url("srv", str(tmp_path)) == stdio_url("srv", str(tmp_path) + os.sep)

    def test_two_jails_are_two_urls(self, tmp_path):
        from onit.type.tools import stdio_url

        a = tmp_path / "one"
        b = tmp_path / "two"
        a.mkdir()
        b.mkdir()
        assert stdio_url("srv", str(a)) != stdio_url("srv", str(b))


# ── the fast path's budget ───────────────────────────────────────────


class TestTheInjectionBudget:
    """The block a session receives is fitted, on line boundaries."""

    def test_text_within_the_budget_is_returned_whole(self):
        text = "one line\n"
        assert mem._fit(text, 100) == text

    def test_an_over_budget_block_drops_whole_lines(self):
        lines = ["line %d of the block\n" % i for i in range(10)]
        text = "".join(lines)
        fitted = mem._fit(text, 30)  # roughly 7 tokens: a few lines
        assert fitted and not fitted.endswith("block 9\n")
        for line in fitted.splitlines(keepends=True):
            assert line.endswith("\n"), "a partial line is a pointer to a cut claim"

    def test_evidence_for_a_task_no_page_answers(self, world):
        out = mem.evidence_for("a task about nothing here", world)
        assert out == {"text": "", "slug": "", "page": None}

    def test_evidence_is_capped(self, world):
        page = _page("staging", "Staging", ["the staging host is stg-%d.internal" % i for i in range(40)])
        mem.save_page(page, world)
        out = mem.evidence_for("the staging host", world)
        assert out["page"] is not None
        assert len(out["text"]) <= mem.MAX_EVIDENCE_CHARS

    def test_a_contradiction_on_a_missing_page_is_not_recorded(self, tmp_path):
        assert (
            mem.record_contradiction(
                "no-such-page",
                claim="c",
                correction="x",
                config_data={"learn": {"path": str(tmp_path / "l"), "memory": {"path": str(tmp_path / "l" / "m")}}},
            )
            is False
        )
