"""Tests for src/model/serving/results.py — the tool-result store."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.results import (CONTINUATION_PREFIX, MAX_GREP_MATCHES,
                                   MAX_READ_CHARS, MAX_STORED_RESULTS,
                                   RESULT_PREVIEW_CHARS, RESULT_STORE_THRESHOLD,
                                   ResultStore, handle_of, is_continued,
                                   is_decayed)


def _store(tmp_path, **kwargs):
    return ResultStore(str(tmp_path), **kwargs)


def _big(n=50000, marker="NEEDLE"):
    """A result with a findable line in the middle of it."""
    lines = [f"line {i} of the result" for i in range(n // 22)]
    lines[len(lines) // 2] = f"line with the {marker} in it"
    return "\n".join(lines)


# ── what gets stored ────────────────────────────────────────────────────────

class TestPut:
    def test_a_small_result_is_left_alone(self, tmp_path):
        assert _store(tmp_path).put("bash", "short output") is None

    def test_a_result_at_the_threshold_is_left_alone(self, tmp_path):
        assert _store(tmp_path).put("bash", "x" * RESULT_STORE_THRESHOLD) is None

    def test_a_large_result_comes_back_as_a_preview(self, tmp_path):
        store = _store(tmp_path)
        preview = store.put("local_search", "x" * 50000)
        assert preview is not None
        assert handle_of(preview) == "0001"
        assert "local_search" in preview
        assert "50,000 chars" in preview
        assert 'result_read("0001"' in preview

    def test_the_preview_is_bounded(self, tmp_path):
        preview = _store(tmp_path).put("bash", "x" * 200000)
        # Header and trailer are small next to the head they bracket.
        assert len(preview) < RESULT_PREVIEW_CHARS + 500

    def test_the_preview_is_the_head_of_the_result(self, tmp_path):
        text = "FIRST-LINE\n" + "x" * 50000
        preview = _store(tmp_path).put("bash", text)
        assert "FIRST-LINE" in preview

    def test_nothing_is_stored_without_a_data_path(self):
        store = ResultStore("")
        assert store.enabled is False
        assert store.put("bash", "x" * 50000) is None

    def test_nothing_is_stored_when_switched_off(self, tmp_path):
        store = _store(tmp_path, enabled=False)
        assert store.put("bash", "x" * 50000) is None

    def test_the_file_holds_the_result_untouched(self, tmp_path):
        text = _big()
        store = _store(tmp_path)
        store.put("local_search", text)
        written = (tmp_path / ".onit" / "results" / "0001-local_search.txt").read_text()
        assert written == text

    def test_results_live_under_the_session_directory(self, tmp_path):
        """Session isolation is what makes a global cache the wrong answer."""
        _store(tmp_path).put("bash", "x" * 50000)
        assert (tmp_path / ".onit" / "results").is_dir()

    def test_handles_do_not_repeat_within_a_session(self, tmp_path):
        store = _store(tmp_path)
        first = store.put("bash", "a" * 50000)
        second = store.put("bash", "b" * 50000)
        assert handle_of(first) == "0001"
        assert handle_of(second) == "0002"

    def test_a_second_store_continues_where_the_first_stopped(self, tmp_path):
        """A later task in the same session must not overwrite 0001 — every
        handle already in the conversation would start resolving elsewhere."""
        _store(tmp_path).put("bash", "a" * 50000)
        later = _store(tmp_path).put("bash", "b" * 50000)
        assert handle_of(later) == "0002"
        assert _store(tmp_path).read("0001", limit=10).count("a") > 0

    def test_a_tool_name_cannot_shape_the_filename(self, tmp_path):
        store = _store(tmp_path)
        preview = store.put("../../etc/passwd", "x" * 50000)
        handle = handle_of(preview)
        assert store.read(handle, limit=5).count("x") == 5
        assert not (tmp_path.parent / "etc").exists()
        names = [p.name for p in (tmp_path / ".onit" / "results").iterdir()]
        assert names == ["0001-.._.._etc_passwd.txt"]

    def test_a_write_that_fails_falls_back_rather_than_erroring(self, tmp_path):
        """A bookkeeping failure must never turn a tool result into an error."""
        blocker = tmp_path / "data"
        blocker.write_text("")  # a file where the results dir would go
        assert ResultStore(str(blocker)).put("bash", "x" * 50000) is None

    def test_the_oldest_are_pruned_past_the_cap(self, tmp_path):
        store = _store(tmp_path)
        for _ in range(MAX_STORED_RESULTS + 5):
            store.put("bash", "x" * 50000)
        stored = store.stored()
        assert len(stored) == MAX_STORED_RESULTS
        # The most recent survive; the first handles are what went.
        assert stored[-1]["handle"] == f"{MAX_STORED_RESULTS + 5:04d}"
        assert "0001" not in [r["handle"] for r in stored]


# ── read ────────────────────────────────────────────────────────────────────

class TestRead:
    def test_round_trip_recovers_the_whole_result(self, tmp_path):
        text = "".join(chr(65 + i % 26) for i in range(30000))
        store = _store(tmp_path)
        handle = handle_of(store.put("bash", text))

        recovered, offset = "", 0
        while True:
            out = store.read(handle, offset=offset, limit=MAX_READ_CHARS)
            body = out.split("\n", 1)[1].rsplit("\n", 1)[0]
            recovered += body
            offset += len(body)
            if out.rstrip().endswith(f"[end of result {handle}]"):
                break
        assert recovered == text

    def test_offset_and_limit_window_the_result(self, tmp_path):
        text = "0123456789" * 2000
        store = _store(tmp_path)
        handle = handle_of(store.put("bash", text))
        out = store.read(handle, offset=100, limit=10)
        assert "\n0123456789\n" in out
        assert "showing 100–110" in out

    def test_the_window_is_capped(self, tmp_path):
        store = _store(tmp_path)
        handle = handle_of(store.put("bash", "x" * 100000))
        out = store.read(handle, offset=0, limit=99999)
        assert out.count("x") == MAX_READ_CHARS

    def test_the_last_window_says_it_is_the_last(self, tmp_path):
        store = _store(tmp_path)
        handle = handle_of(store.put("bash", "x" * 9000))
        assert store.read(handle, offset=8000, limit=4000).rstrip().endswith(
            f"[end of result {handle}]")

    def test_a_window_with_more_after_it_says_where(self, tmp_path):
        store = _store(tmp_path)
        handle = handle_of(store.put("bash", "x" * 50000))
        out = store.read(handle, offset=0, limit=1000)
        assert 'result_read("0001", offset=1000)' in out

    def test_an_offset_past_the_end_is_explained(self, tmp_path):
        store = _store(tmp_path)
        handle = handle_of(store.put("bash", "x" * 9000))
        out = store.read(handle, offset=99999)
        assert out.startswith("Error:")
        assert "past the end" in out

    def test_an_unknown_handle_names_what_does_exist(self, tmp_path):
        store = _store(tmp_path)
        store.put("bash", "x" * 50000)
        out = store.read("0099")
        assert out.startswith("Error:")
        assert "0001" in out

    def test_reading_with_no_store_is_an_error_not_a_crash(self):
        assert ResultStore("").read("0001").startswith("Error:")


# ── grep ────────────────────────────────────────────────────────────────────

class TestGrep:
    def test_finds_a_line_and_its_context(self, tmp_path):
        store = _store(tmp_path)
        handle = handle_of(store.put("local_search", _big()))
        out = store.grep(handle, "NEEDLE", context=2)
        assert "NEEDLE" in out
        # Two lines either side, plus the match itself.
        assert len([l for l in out.splitlines() if l[:1].isdigit()]) == 5

    def test_context_zero_gives_the_matching_line_alone(self, tmp_path):
        store = _store(tmp_path)
        handle = handle_of(store.put("bash", _big()))
        out = store.grep(handle, "NEEDLE", context=0)
        assert len([l for l in out.splitlines() if l[:1].isdigit()]) == 1

    def test_a_pattern_that_matches_nothing_says_so(self, tmp_path):
        store = _store(tmp_path)
        handle = handle_of(store.put("bash", _big()))
        out = store.grep(handle, "not-in-there")
        assert "no line matches" in out

    def test_a_regex_works(self, tmp_path):
        store = _store(tmp_path)
        handle = handle_of(store.put("bash", _big()))
        assert "NEEDLE" in store.grep(handle, r"NEE.LE", context=0)

    def test_an_invalid_regex_is_taken_literally(self, tmp_path):
        """A model that meant a literal string gets what it meant, not a
        lecture about regex syntax."""
        store = _store(tmp_path)
        handle = handle_of(store.put("bash", "a(b" + "\nfiller" * 5000))
        out = store.grep(handle, "a(b", context=0)
        assert not out.startswith("Error:")
        assert "a(b" in out

    def test_matches_are_capped(self, tmp_path):
        store = _store(tmp_path)
        handle = handle_of(store.put("bash", "\n".join(["hit"] * 4000)))
        out = store.grep(handle, "hit", context=0)
        assert f"showing the first {MAX_GREP_MATCHES}" in out

    def test_overlapping_context_windows_are_merged(self, tmp_path):
        """A cluster of matches reads as one passage, not the same lines over."""
        store = _store(tmp_path)
        text = "\n".join(["hit" if i in (10, 11, 12) else f"line {i}"
                          for i in range(5000)])
        handle = handle_of(store.put("bash", text))
        out = store.grep(handle, "hit", context=3)
        # One contiguous block: three matches three lines apart share context,
        # so no line is printed twice and there is one "--" separator at most.
        assert out.count("10: line 9") == 1
        assert out.count("11: hit") == 1
        assert out.count("\n--\n") == 1

    def test_a_blank_pattern_is_refused(self, tmp_path):
        store = _store(tmp_path)
        handle = handle_of(store.put("bash", _big()))
        assert store.grep(handle, "  ").startswith("Error:")

    def test_an_unknown_handle_is_an_error(self, tmp_path):
        assert _store(tmp_path).grep("0099", "x").startswith("Error:")


# ── the path jail ───────────────────────────────────────────────────────────

class TestTraversalRejected:
    """data_path is a session isolation boundary; a handle is model input."""

    @pytest.fixture
    def store(self, tmp_path):
        s = _store(tmp_path)
        s.put("bash", "x" * 50000)
        return s

    @pytest.mark.parametrize("handle", [
        "../../../etc/passwd",
        "..",
        "../0001",
        "/etc/passwd",
        "0001/../../../etc/passwd",
        "0001.txt",
        "0001-bash.txt",
        "*",
        "",
        "   ",
        None,
        1,
    ])
    def test_rejected(self, store, handle):
        assert store._path_for(handle) is None
        if isinstance(handle, str):
            assert store.read(handle).startswith("Error:")

    def test_a_symlink_out_of_the_jail_is_not_followed(self, tmp_path):
        secret = tmp_path.parent / "secret.txt"
        secret.write_text("classified")
        store = _store(tmp_path)
        results = tmp_path / ".onit" / "results"
        results.mkdir(parents=True, exist_ok=True)
        (results / "0007-leak.txt").symlink_to(secret)

        assert store._path_for("0007") is None
        assert "classified" not in store.read("0007")

    def test_an_absolute_path_cannot_be_read(self, tmp_path):
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("classified")
        assert "classified" not in _store(tmp_path).read(str(outside))


# ── message helpers the loop uses ───────────────────────────────────────────

class TestMessageHelpers:
    def test_handle_of_reads_the_header(self):
        assert handle_of("[result:0042 · bash · 10 chars]\nbody") == "0042"

    def test_handle_of_ignores_anything_else(self):
        assert handle_of("ordinary tool output") is None
        assert handle_of("") is None
        assert handle_of(None) is None
        assert handle_of(["not", "a", "string"]) is None

    def test_a_preview_is_continued_but_not_decayed(self, tmp_path):
        preview = _store(tmp_path).put("bash", "x" * 50000)
        assert is_continued(preview) is True
        assert is_decayed(preview) is False

    def test_a_decay_trailer_is_both(self, tmp_path):
        store = _store(tmp_path)
        content = "[result:0001 · bash · 9 chars]\nhead\n\n" + store.decay_trailer("0001", 4)
        assert is_continued(content) is True
        assert is_decayed(content) is True

    def test_the_decay_trailer_points_at_a_local_read(self, tmp_path):
        """The entire point of the phase: recovery stops being a network round
        trip to re-run the tool."""
        trailer = _store(tmp_path).decay_trailer("0007", 1200)
        assert 'result_read("0007", offset=1200)' in trailer
        assert "call the tool again" not in trailer
        assert trailer.startswith(CONTINUATION_PREFIX)


# ── discovery ───────────────────────────────────────────────────────────────

class TestStored:
    def test_lists_handle_tool_and_size(self, tmp_path):
        store = _store(tmp_path)
        store.put("local_search", "x" * 50000)
        assert store.stored() == [
            {"handle": "0001", "tool": "local_search", "chars": 50000}]

    def test_empty_before_anything_is_stored(self, tmp_path):
        assert _store(tmp_path).stored() == []

    def test_empty_without_a_store(self):
        assert ResultStore("").stored() == []
