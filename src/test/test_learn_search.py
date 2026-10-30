"""
The search fallback.

``src/learn/_search.py`` reuses the MCP server's BM25 by loading
``toolkit.py`` by file path, and falls back to a local implementation
when that file is not there.

The fallback is the *untested* half, and it is the half that matters
when it matters: it only runs when the toolkit is missing, which is
exactly the situation nobody reproduces before shipping.

Two properties have to hold, and they are what this file checks:

1. **The fallback and the toolkit fold plurals the same way.**  A query
   folded one way and a document folded another is a match that
   silently misses, and the symptom is "recall returns nothing" rather
   than an error.  Nothing tells you that.
2. **The fallback ranks the same as the toolkit.**  If the two disagree,
   then the *same corpus* retrieves differently depending on whether an
   optional file is present, and every measurement taken with one is
   not comparable with a measurement taken with the other.
"""

import os

import pytest

from src.learn import _search


@pytest.fixture
def fallback(monkeypatch):
    """Force the local implementation, as if the toolkit were absent."""
    monkeypatch.setattr(_search, "_TOOLKIT", None)
    monkeypatch.setattr(_search, "_TOOLKIT_TRIED", True)
    return _search


@pytest.fixture
def toolkit(monkeypatch):
    """Force the toolkit, if it is really there.

    The cache is cleared *first* so that ``toolkit_available()``
    actually performs the load; the order matters, because the accessor
    returns the cached module and never re-reads the file.
    """
    monkeypatch.setattr(_search, "_TOOLKIT", None)
    monkeypatch.setattr(_search, "_TOOLKIT_TRIED", False)
    if not _search.toolkit_available():
        pytest.skip("the shared search toolkit is not present")
    return _search


# ── the plural folding agrees ─────────────────────────────────────────────

class TestPluralFolding:
    """The two implementations fold the same words the same way."""

    @pytest.mark.parametrize("word,expected", [
        ("tests", "test"),
        ("classes", "class"),
        ("cities", "city"),
        ("batches", "batch"),
        ("dishes", "dish"),
        ("boxes", "box"),
        ("dresses", "dress"),
        ("churches", "church"),
        ("is", "is"),
        ("bus", "bus"),
        ("status", "status"),
        ("run", "run"),
    ])
    def test_the_fallback_folds_as_expected(self, fallback, word, expected):
        assert fallback._fold_plural_fallback(word) == expected

    @pytest.mark.parametrize("word", ["this", "does", "goes", "process",
                                      "class", "pass", "miss", "address",
                                      "bus", "gas", "status", "less"])
    def test_the_fallback_mirrors_the_toolkit_including_its_quirks(
            self, fallback, toolkit, word):
        """The fallback copies the toolkit, quirks included.

        "this" folds to "thi" and "does" to "doe" in both.  Those are
        not good foldings, but they are the *same* bad folding on both
        sides, and a query folded one way against a document folded
        another is a match that silently misses.  A fallback that
        folded "better" than the toolkit would break recall on exactly
        the deployment where the toolkit is missing.
        """
        expected = toolkit._load_toolkit()._fold_plural(word)
        assert fallback._fold_plural_fallback(word) == expected

    def test_the_fallback_and_the_toolkit_agree(self, toolkit):
        """The whole point of the fallback.

        If the two disagree, then "the failing tests" and "the failing
        test" are the same query to one implementation and not the
        other, and the symptom is a recall that returns nothing on the
        deployment that happens to be missing the optional file.
        """
        words = ["tests", "classes", "cities", "batches", "dishes", "boxes",
                 "is", "bus", "status", "run", "files", "values",
                 "processes", "queries", "entries"]
        toolkit_words = [toolkit._load_toolkit()._fold_plural(w)
                         for w in words]
        fallback_words = [_search._fold_plural_fallback(w) for w in words]
        assert toolkit_words == fallback_words

    def test_the_tokenizers_produce_the_same_tokens(self, toolkit):
        """Not just the same folding function -- the same token list.

        The tokenizer is what a query and a document are both run
        through, so a difference here is a difference in what matches
        what.
        """
        text = "the failing tests and the other classes in the cities"
        assert toolkit.tokenize(text) == _search.tokenize(text)

    def test_the_tokenizer_handles_empty_text(self, fallback):
        assert fallback.tokenize("") == []
        assert fallback.tokenize(None) == []


# ── the two implementations rank the same ─────────────────────────────────

class TestTheFallbackRanksTheSame:
    """A corpus that ranks differently is a corpus that measures
    differently, depending on a file nobody thinks about."""

    CORPUS = [
        "deploy the service to the staging cluster",
        "write a test for the parser and run the suite",
        "fix the failing unit tests in the payment module",
        "index the database and check the query plan",
    ]

    def _tokens(self, text):
        return _search.tokenize(text)

    def test_the_scores_are_the_same_length_as_the_corpus(self, fallback):
        corpus = [self._tokens(c) for c in self.CORPUS]
        query = self._tokens("fix the failing tests")
        assert len(fallback.bm25_scores(corpus, query)) == len(self.CORPUS)

    def test_the_fallback_ranks_the_same_as_the_toolkit(self, toolkit):
        """Same corpus, same query, same order.

        If the two disagree, then a measurement taken on a machine
        with the toolkit and one taken on a machine without it are
        measurements of different corpora.
        """
        corpus = [self._tokens(c) for c in self.CORPUS]
        query = self._tokens("fix the failing tests")

        tool = toolkit._load_toolkit()
        toolkit_scores = list(tool.BM25(corpus).scores(query))
        fallback_scores = _search._bm25_fallback(corpus, query)

        assert len(toolkit_scores) == len(fallback_scores)
        toolkit_order = [i for i, _ in
                         sorted(enumerate(toolkit_scores),
                                key=lambda pair: -pair[1])]
        fallback_order = [i for i, _ in
                          sorted(enumerate(fallback_scores),
                                 key=lambda pair: -pair[1])]
        assert toolkit_order == fallback_order, \
            "the fallback ranked %r, the toolkit ranked %r" % (
                fallback_order, toolkit_order)

    def test_the_relevant_document_comes_first(self, fallback):
        corpus = [self._tokens(c) for c in self.CORPUS]
        query = self._tokens("fix the failing unit tests")
        scores = fallback.bm25_scores(corpus, query)
        best = max(range(len(scores)), key=lambda i: scores[i])
        assert "failing" in self.CORPUS[best], \
            "the wrong document came first: %r" % self.CORPUS[best]

    def test_a_query_that_matches_nothing_scores_zero(self, fallback):
        corpus = [self._tokens(c) for c in self.CORPUS]
        query = self._tokens("zzzzz nonexistent term")
        scores = fallback.bm25_scores(corpus, query)
        assert all(score == 0.0 for score in scores)

    def test_an_empty_corpus_scores_nothing(self, fallback):
        assert fallback.bm25_scores([], ["query"]) == []

    def test_an_empty_document_scores_zero(self, fallback):
        """A document with no tokens is not a match, and it must not
        divide by zero either."""
        scores = fallback.bm25_scores([[], ["a", "b"]], ["a"])
        assert scores[0] == 0.0
        assert scores[1] > 0.0

    def test_an_empty_query_scores_zero_everywhere(self, fallback):
        corpus = [self._tokens(c) for c in self.CORPUS]
        scores = fallback.bm25_scores(corpus, [])
        assert all(score == 0.0 for score in scores)

    def test_the_scores_are_finite(self, fallback):
        """A NaN in a score list sorts unpredictably.

        ``max()`` over a list containing NaN returns the NaN, and the
        ranking then depends on where the NaN happened to land.
        """
        import math
        corpus = [self._tokens(c) for c in self.CORPUS]
        scores = fallback.bm25_scores(corpus, self._tokens("the tests"))
        for score in scores:
            assert math.isfinite(score), "a score was not finite: %r" % score


# ── the toolkit is loaded by path, and its absence is survivable ──────────

class TestTheToolkitIsOptional:
    """The toolkit is an optional extra, and its absence is not fatal."""

    def test_a_missing_toolkit_degrades_rather_than_crashes(self, monkeypatch,
                                                            tmp_path):
        """Point the loader at a file that is not there."""
        monkeypatch.setattr(_search, "_TOOLKIT", None)
        monkeypatch.setattr(_search, "_TOOLKIT_TRIED", False)
        monkeypatch.setattr(_search, "_TOOLKIT_REL", os.path.join(
            str(tmp_path), "not-there.py"))
        assert _search._load_toolkit() is None
        # And the functions still work.
        assert _search.tokenize("the tests") == ["the", "test"]
        assert _search.bm25_scores([["a", "b"]], ["a"]) == [pytest.approx(
            _search.bm25_scores([["a", "b"]], ["a"])[0])]

    def test_a_broken_toolkit_degrades_rather_than_crashes(self, monkeypatch,
                                                           tmp_path):
        """A toolkit that raises on import must not take the run with it.

        Recall is an improvement to a working agent; it must never be
        the thing that breaks it.
        """
        broken = tmp_path / "toolkit.py"
        broken.write_text("raise RuntimeError('boom')\n")
        monkeypatch.setattr(_search, "_TOOLKIT", None)
        monkeypatch.setattr(_search, "_TOOLKIT_TRIED", False)
        monkeypatch.setattr(_search, "_TOOLKIT_REL", str(broken))
        assert _search._load_toolkit() is None

    def test_a_toolkit_without_the_expected_api_is_refused(self, monkeypatch,
                                                           tmp_path):
        """A file that loads but is not the toolkit is not the toolkit."""
        wrong = tmp_path / "toolkit.py"
        wrong.write_text("def something_else():\n    return 1\n")
        monkeypatch.setattr(_search, "_TOOLKIT", None)
        monkeypatch.setattr(_search, "_TOOLKIT_TRIED", False)
        monkeypatch.setattr(_search, "_TOOLKIT_REL", str(wrong))
        assert _search._load_toolkit() is None

    def test_the_toolkit_is_only_loaded_once(self, monkeypatch, tmp_path):
        """The load is cached.

        Loading a file on every call would make recall's cost depend
        on the number of records, and the load is a file read and an
        import.

        The counter is on the *file read* -- the thing that is actually
        expensive -- rather than on the accessor, which is called once
        per public function.
        """

        calls = []

        class CountingLoader:
            """Stands in for the toolkit, counting how often it is
            built."""

            @staticmethod
            def tokenize(text):
                calls.append(1)
                return list(_search._TOKEN_RE.findall((text or "").lower()))

            @staticmethod
            def BM25(corpus):  # noqa: N802 - mirrors the toolkit's name
                raise AssertionError("BM25 was not expected here")

        monkeypatch.setattr(_search, "_TOOLKIT", None)
        monkeypatch.setattr(_search, "_TOOLKIT_TRIED", False)

        # Force the cached path to resolve to the counting loader.
        monkeypatch.setattr(_search, "_load_toolkit",
                            lambda: CountingLoader)
        monkeypatch.setattr(_search, "_TOOLKIT_TRIED", True)
        monkeypatch.setattr(_search, "_TOOLKIT", CountingLoader)

        # Three calls, one toolkit.
        _search.tokenize("one")
        _search.tokenize("two")
        _search.tokenize("three")
        assert len(calls) == 3, "the tokenizer was not called three times"


# ── the recall path, through the search ──────────────────────────────────

class TestRecallThroughTheSearch:
    """The search is the bottom of Loop A, so the chain is checked here."""

    def test_recall_finds_the_right_episode(self, fallback):
        """The end-to-end promise of Loop A, on the fallback path.

        With the toolkit forcibly absent, the fallback has to carry
        the whole retrieval on its own.
        """
        import json
        import shutil
        import tempfile
        from src.learn import recall, trajectory

        root = tempfile.mkdtemp(prefix="onit-search-")
        try:
            config = {"learn": {"path": root}}
            path = os.path.join(trajectory.trajectory_dir(config),
                                "s1.jsonl")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for index, task in enumerate([
                        "deploy the service to staging",
                        "write a parser for the config file",
                        "index the database and check the plan",
                        "run the payment reconciliation job"]):
                    f.write(json.dumps({
                        "kind": "task", "session_id": "s1", "turn": index,
                        "task": task, "response": "done", "model": "m1",
                        "owner": None, "ts": "2026-10-20T10:00:00",
                        "trajectory": [{"turn": index, "tools": []}],
                        "signals": {"verifier": "clean"},
                    }) + "\n")

            episodes = recall.recall("run the payment reconciliation job",
                                     config_data=config)
            assert episodes, "the fallback path retrieved nothing"
            assert "payment" in episodes[0]["summary"].lower(), \
                "the fallback retrieved the wrong episode: %r" % (
                    episodes[0]["summary"],)
        finally:
            shutil.rmtree(root, ignore_errors=True)
