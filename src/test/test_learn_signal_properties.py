"""
The implicit signals, as properties rather than as cases.

``src/test/test_learn_signals.py`` tests the signals one case at a time.
This file tests the *properties* that have to hold for every input, and
it does so with generated inputs rather than with a list of examples --
because the failure mode of a heuristic labeler is not a case someone
thought of, it is a case nobody thought of.

The properties here are the ones that decide whether a corpus is
trustworthy:

1. **A label is always one of three values.**  A fourth value would
   silently become "unknown" downstream.
2. **The same input always gives the same label.**  A heuristic that
   depends on dict ordering or set iteration is a heuristic that
   re-labels the corpus on every run, and every comparison against a
   previous measurement is then meaningless.
3. **A rephrase and an acceptance are mutually exclusive.**  A task
   cannot be both "you got it wrong, do it again" and "that worked".
4. **An explicit rating is never overwritten.**  A click is stronger
   evidence than a heuristic.
5. **A different subject is not a rephrase.**  "Benchmark mbpp" and
   "benchmark bigcodebench" are 93% similar and share no content; the
   corpus is full of that shape, and calling it a repeat would label
   every "now the same for the next dataset" as a failure.
6. **Changing a parameter is a new request.**  "3-layer" is not
   "2-layer".
"""

import random
import string

import pytest

from src.learn import signals as sig


# ── generators ────────────────────────────────────────────────────────────

_WORDS = ["run", "the", "tests", "fix", "bug", "write", "parser", "for",
          "benchmark", "dataset", "add", "feature", "refactor", "module",
          "build", "docker", "image", "deploy", "service", "query",
          "database", "index", "implement", "function", "that", "takes",
          "list", "and", "returns", "sum"]


def _sentence(rng, n=None):
    n = n or rng.randint(3, 12)
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def _garbage(rng, n=None):
    """A string with no words in it at all."""
    n = n or rng.randint(0, 20)
    alphabet = string.punctuation + string.digits
    return "".join(rng.choice(alphabet) for _ in range(n))


# ── property 1: a label is one of three values ────────────────────────────

class TestLabelDomain:
    """Every label is good, bad or unknown.  Nothing else."""

    @pytest.mark.parametrize("label", ["good", "bad", "unknown"])
    def test_the_three_labels_exist(self, label):
        assert label in ("good", "bad", "unknown")

    def test_label_pair_always_returns_a_known_label(self):
        rng = random.Random(20261023)
        for _ in range(400):
            previous = _sentence(rng)
            following = rng.choice([_sentence(rng), _garbage(rng), "",
                                     "no", "thanks", "yes, that worked"])
            label = sig.label_pair(previous, following)
            assert label in ("good", "bad", "unknown"), \
                "label_pair returned %r" % label

    def test_label_session_returns_one_label_per_record(self):
        rng = random.Random(20261023)
        for size in range(0, 12):
            records = [{"task": _sentence(rng)} for _ in range(size)]
            labels = sig.label_session(records)
            assert len(labels) == size
            for label in labels:
                assert label in ("good", "bad", "unknown")

    def test_label_records_gives_every_record_a_label(self):
        rng = random.Random(20261023)
        records = [{"task": _sentence(rng)} for _ in range(7)]
        sig.label_records(records)
        for record in records:
            assert record["signals"]["implicit"] in ("good", "bad", "unknown")


# ── property 2: the label is deterministic ────────────────────────────────

class TestDeterminism:
    """The same corpus must produce the same labels, every time.

    This is not a nicety.  The signals are used to *rank* the corpus and
    to decide what a playbook bullet is distilled from.  A labeler whose
    output depends on set iteration order re-labels the corpus on every
    run, and every comparison against a previous measurement is then
    measuring the re-labeling rather than the agent.
    """

    def test_label_pair_is_deterministic(self):
        rng = random.Random(20261023)
        for _ in range(200):
            previous = _sentence(rng)
            following = _sentence(rng)
            first = sig.label_pair(previous, following)
            for _ in range(5):
                assert sig.label_pair(previous, following) == first

    def test_label_session_is_deterministic(self):
        rng = random.Random(20261023)
        records = [{"task": _sentence(rng)} for _ in range(10)]
        first = sig.label_session(records)
        for _ in range(5):
            assert sig.label_session(records) == first

    def test_the_same_task_in_a_different_order_still_labels_the_same(self):
        """The label depends on the *pair*, not on the list.

        Two sessions with the same tasks in a different order must
        label the same *pair* the same way.  If it did not, the corpus
        would depend on how the records happened to be ordered.
        """
        a = "add a test for the parser"
        b = "now add a test for the parser"
        assert sig.label_pair(a, b) == sig.label_pair(a, b)

    def test_summarize_is_deterministic(self):
        rng = random.Random(20261023)
        records = [{"task": _sentence(rng)} for _ in range(10)]
        sig.label_records(records)
        first = sig.summarize(records)
        for _ in range(5):
            assert sig.summarize(records) == first


# ── property 3: rephrase and acceptance are exclusive ─────────────────────

class TestExclusivity:
    """A task cannot be both a rephrase and an acceptance."""

    @pytest.mark.parametrize("following", [
        "no", "no, do it again", "i said, use python", "again",
        "thanks, no, run it again", "still failing, try again",
        "no that is wrong", "that is not what i asked for",
    ])
    def test_a_rephrase_is_never_an_acceptance(self, following):
        previous = "write a script that sums a list"
        assert sig.is_rephrase(previous, following)
        assert not sig.is_acceptance(previous, following)

    @pytest.mark.parametrize("following", [
        "thanks", "great, now do the next one", "perfect", "that worked",
        "yes that is right", "ok, now the second part",
    ])
    def test_an_acceptance_is_never_a_rephrase(self, following):
        previous = "write a script that sums a list"
        assert sig.is_acceptance(previous, following)
        assert not sig.is_rephrase(previous, following)

    def test_the_label_follows_the_exclusivity(self):
        """``label_pair`` is decided by the rephrase test first.

        A task that opens with a courtesy and then contradicts the
        request -- "thanks, no, do it again" -- is a rephrase.  Checking
        acceptance first would label it "good", which is the worst
        possible answer: it would learn from a correction as though it
        were a success.
        """
        previous = "write a script that sums a list"
        assert sig.label_pair(previous, "thanks, no, do it again") == "bad"

    def test_no_input_is_both(self):
        rng = random.Random(20261023)
        for _ in range(400):
            previous = _sentence(rng)
            following = rng.choice([
                _sentence(rng), "no", "thanks", "no, again",
                "great, next one", _garbage(rng), "yes", "still wrong",
            ])
            assert not (sig.is_rephrase(previous, following)
                        and sig.is_acceptance(previous, following))


# ── property 4: an explicit rating is never overwritten ───────────────────

class TestExplicitRatingsWin:
    """A click is stronger evidence than a heuristic."""

    def test_an_explicit_rating_survives_labeling(self):
        records = [
            {"task": "write a script", "signals": {"user_rating": "up"}},
            {"task": "no, again", "signals": {}},
        ]
        sig.label_records(records)
        assert records[0]["signals"].get("implicit") is None, \
            "the heuristic overwrote an explicit rating"
        assert records[0]["signals"]["implicit_source"] == "explicit-rating"

    def test_a_negative_rating_survives_labeling(self):
        records = [
            {"task": "write a script", "signals": {"user_rating": "down"}},
            {"task": "thanks", "signals": {}},
        ]
        sig.label_records(records)
        assert records[0]["signals"].get("implicit") is None

    def test_the_source_records_which_evidence_was_used(self):
        records = [
            {"task": "write a script", "signals": {"user_rating": "up"}},
            {"task": "no, again", "signals": {}},
            {"task": "something else", "signals": {}},
        ]
        sig.label_records(records)
        assert records[0]["signals"]["implicit_source"] == "explicit-rating"
        assert records[1]["signals"]["implicit_source"] in ("follow-up", "none")
        assert records[2]["signals"]["implicit_source"] == "session-end"

    def test_the_summary_counts_explicit_ratings_separately(self):
        records = [
            {"task": "a", "signals": {"user_rating": "up"}},
            {"task": "b", "signals": {"user_rating": "down"}},
            {"task": "c", "signals": {}},
        ]
        sig.label_records(records)
        summary = sig.summarize(records)
        assert summary["explicit_ratings"] == 2


# ── property 5: a different subject is not a rephrase ─────────────────────

class TestSubstitutionIsNotRephrase:
    """Same shape, different subject: a new task, not a repeat.

    This is the pattern that dominates the corpus.  "Benchmark mbpp"
    followed by "benchmark bigcodebench" is the *next dataset*, not a
    retry, and a labeler that calls it a repeat turns every successful
    sequence of similar work into a string of failures.
    """

    @pytest.mark.parametrize("pair", [
        ("benchmark mbpp", "benchmark bigcodebench"),
        ("implement the parser for the json format",
         "implement the parser for the yaml format"),
        ("write a test for the http client",
         "write a test for the db client"),
        ("add a function that sums a list",
         "add a function that finds the maximum of a list"),
    ])
    def test_a_swapped_subject_is_not_a_rephrase(self, pair):
        previous, following = pair
        assert not sig.is_rephrase(previous, following)

    def test_the_swap_is_detected_as_a_substitution(self):
        assert sig.is_substitution("benchmark mbpp", "benchmark bigcodebench")

    def test_a_genuine_repetition_is_still_a_rephrase(self):
        """The screen must not be so eager that it stops catching
        genuine repeats -- that would silently disable the whole
        negative-signal path."""
        assert sig.is_rephrase("run the unit tests for the parser",
                               "run the unit tests for the parser again")

    def test_a_paraphrase_of_the_same_request_is_a_rephrase(self):
        """The same request, differently worded.

        The tokens differ by a filler word only, so the content overlap
        stays high and the similarity clears the bar.
        """
        assert sig.is_rephrase("please run the unit tests",
                               "please run the unit tests now")

    def test_a_shorter_request_that_drops_content_is_not_a_rephrase(self):
        """"Run the tests" after "run the unit tests" drops a content
        word.  It is a *narrower* request, not the same one, and
        treating it as a repeat would label every narrowing request as
        a failure."""
        assert not sig.is_rephrase("please run the unit tests",
                                   "run the tests please")


# ── property 6: a changed parameter is a new request ─────────────────────

class TestChangedParameterIsNew:
    """Numbers survive tokenization, and a changed number changes the
    request."""

    @pytest.mark.parametrize("pair", [
        ("build a 3-layer cnn", "build a 2-layer cnn"),
        ("run epoch 5", "run epoch 10"),
        ("use batch size 32", "use batch size 64"),
    ])
    def test_a_changed_number_is_not_a_rephrase(self, pair):
        previous, following = pair
        assert not sig.is_rephrase(previous, following)

    def test_numbers_survive_the_tokenizer(self):
        tokens = sig._tokens("build a 3-layer cnn")
        assert "3" in tokens, "the tokenizer dropped the number"

    def test_a_shared_number_is_not_a_difference(self):
        """The argument is the *difference*, not the whole token set.

        "run test 1" and "run test 2" disagree, because the numbers
        differ; a number that appears on both sides never reaches this
        function, so it is never counted as a difference.
        """
        assert sig._numbers_differ({"1"}, {"2"})
        assert not sig._numbers_differ({"1"}, set())
        assert not sig._numbers_differ(set(), {"2"})
        assert not sig._numbers_differ({"alpha"}, {"beta"})


# ── the degenerate inputs ─────────────────────────────────────────────────

class TestDegenerateInputs:
    """Empty, whitespace and garbage must not raise and must not label."""

    @pytest.mark.parametrize("previous", ["", "   ", "\n", None])
    @pytest.mark.parametrize("following", ["", "   ", "\n", None])
    def test_no_input_raises(self, previous, following):
        label = sig.label_pair(previous, following)
        assert label in ("good", "bad", "unknown")

    def test_garbage_is_not_a_rephrase(self):
        rng = random.Random(20261023)
        for _ in range(100):
            assert not sig.is_rephrase(_garbage(rng), _garbage(rng))

    def test_a_one_word_task_compares_to_nothing(self):
        """Two one-word tasks share nothing reliably, and pretending
        otherwise manufactures rephrases."""
        assert sig.similarity("fix", "fix") == 0.0
        assert not sig.is_rephrase("fix", "fix")

    def test_the_similarity_is_bounded(self):
        rng = random.Random(20261023)
        for _ in range(200):
            a, b = _sentence(rng), _sentence(rng)
            value = sig.similarity(a, b)
            assert 0.0 <= value <= 1.0

    def test_the_overlap_is_bounded(self):
        rng = random.Random(20261023)
        for _ in range(200):
            a, b = _sentence(rng), _sentence(rng)
            value = sig.content_overlap(a, b)
            assert 0.0 <= value <= 1.0


# ── the summary is honest ─────────────────────────────────────────────────

class TestSummaryIsHonest:
    """A report that hides the unknowns reads as a stronger corpus."""

    def test_the_unknowns_are_counted(self):
        records = [{"task": "a", "signals": {"implicit": "unknown"}},
                   {"task": "b", "signals": {"implicit": "good"}},
                   {"task": "c", "signals": {"implicit": "bad"}}]
        summary = sig.summarize(records)
        assert summary["unknown"] == 1
        assert summary["good"] == 1
        assert summary["bad"] == 1
        assert summary["total"] == 3

    def test_the_coverage_is_the_labeled_share(self):
        records = [{"task": "a", "signals": {"implicit": "good"}},
                   {"task": "b", "signals": {"implicit": "unknown"}}]
        summary = sig.summarize(records)
        assert summary["coverage"] == 0.5

    def test_an_empty_corpus_does_not_divide_by_zero(self):
        summary = sig.summarize([])
        assert summary["total"] == 0
        assert summary["coverage"] == 0.0
        assert summary["positive_rate"] == 0.0

    def test_the_positive_rate_is_over_the_labeled_part(self):
        """A corpus that is 90% labeled and 5% positive is a different
        thing from one that is 90% labeled and 95% positive, and the
        two rates are what tell them apart."""
        records = [{"task": "a", "signals": {"implicit": "good"}}] * 1
        records += [{"task": "b", "signals": {"implicit": "bad"}}] * 19
        records += [{"task": "c", "signals": {"implicit": "unknown"}}] * 80
        summary = sig.summarize(records)
        assert summary["labeled"] == 20
        assert summary["coverage"] == 0.2
        assert summary["positive_rate"] == 0.05


# ── the last task of a session ────────────────────────────────────────────

class TestSessionEnd:
    """The last task of a session is labelled from the session's end.

    It is the weakest of the three inferences -- a user who closed the
    tab because the answer was wrong looks exactly like one who closed
    it satisfied -- and it is reported as its own source so that a loop
    can weigh it differently from a follow-up.
    """

    def test_a_one_task_session_is_labelled_good(self):
        assert sig.label_session([{"task": "write a script"}]) == ["good"]

    def test_only_the_last_task_gets_the_session_end_label(self):
        records = [{"task": "a"}, {"task": "b"}, {"task": "c"}]
        labels = sig.label_session(records)
        assert labels[-1] == "good"
        assert labels[0] != "good" or len(records) == 1

    def test_a_middle_task_is_never_labelled_from_the_session_end(self):
        """A middle task has a following task, and the following task is
        the evidence.  The session end is not a substitute for it."""
        records = [{"task": "write a script"}, {"task": "no, again"},
                   {"task": "something else"}]
        labels = sig.label_session(records)
        assert labels[0] == "bad", "the middle task was labelled from the end"

    def test_an_empty_session_has_no_labels(self):
        assert sig.label_session([]) == []
