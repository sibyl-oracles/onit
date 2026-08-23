"""Tests for src/lib/text.py — remove_tags and text_between_tags."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.text import remove_tags, text_between_tags


# ── remove_tags ─────────────────────────────────────────────────────────────

class TestRemoveTags:
    def test_removes_simple_tags(self):
        assert remove_tags("<b>bold</b>") == "bold"

    def test_removes_nested_tags(self):
        assert remove_tags("<div><span>text</span></div>") == "text"

    def test_removes_self_closing_style_tags(self):
        assert remove_tags("hello <br/> world") == "hello  world"

    def test_no_tags_returns_unchanged(self):
        assert remove_tags("no tags here") == "no tags here"

    def test_empty_string(self):
        assert remove_tags("") == ""

    def test_none_returns_none(self):
        assert remove_tags(None) is None

    def test_mixed_content(self):
        assert remove_tags("before <tag>inside</tag> after") == "before inside after"

    def test_multiple_tags(self):
        result = remove_tags("<a>one</a> <b>two</b>")
        assert result == "one two"

    def test_bare_less_than_is_not_a_tag(self):
        """A comparison in prose is not an opening tag.

        The regression this guards: an answer containing "0.709 < 0.712" and a
        closing wrapper tag anywhere after it lost everything between the two.
        """
        text = ("mean **0.70931** < 0.71205 -> rejected\n\n"
                "## Round 2\ntext the user needs\n</answer>")
        out = remove_tags(text)
        assert "< 0.71205" in out
        assert "text the user needs" in out
        assert "</answer>" not in out

    def test_digit_after_less_than_is_not_a_tag(self):
        assert remove_tags("if x<3 then y>4") == "if x<3 then y>4"

    def test_tag_does_not_span_a_second_open_bracket(self):
        assert remove_tags("a < b and <i>c</i> < d") == "a < b and c < d"

    def test_keeps_markdown_autolinks(self):
        assert remove_tags("see <https://example.com> now") == "see <https://example.com> now"

    def test_removes_attributed_tags(self):
        assert remove_tags('<div class="x">y</div>') == "y"

    def test_removes_model_special_tokens(self):
        assert remove_tags("answer<|im_end|>") == "answer"

    def test_removes_comments_and_doctypes(self):
        assert remove_tags("<!-- note -->keep") == "keep"
        assert remove_tags("<!DOCTYPE html>keep") == "keep"


# ── text_between_tags ───────────────────────────────────────────────────────

class TestTextBetweenTags:
    def test_full_match(self):
        is_full, text = text_between_tags("<answer>42</answer>", "answer")
        assert is_full is True
        assert text == "42"

    def test_partial_match(self):
        is_full, text = text_between_tags("prefix <answer>42</answer> suffix", "answer")
        assert is_full is False
        assert text == "42"

    def test_no_start_tag(self):
        is_full, text = text_between_tags("no tags here", "answer")
        assert is_full is False
        assert text == "no tags here"

    def test_no_end_tag(self):
        is_full, text = text_between_tags("<answer>open only", "answer")
        assert is_full is False
        assert text == "<answer>open only"

    def test_empty_text(self):
        is_full, text = text_between_tags("", "tag")
        assert is_full is False
        assert text == ""

    def test_none_text(self):
        is_full, text = text_between_tags(None, "tag")
        assert is_full is False
        assert text is None

    def test_empty_tag(self):
        is_full, text = text_between_tags("<a>hello</a>", "")
        assert is_full is False

    def test_uses_last_occurrence(self):
        """rfind behaviour: extracts from the last matching pair."""
        content = "<t>first</t> middle <t>second</t>"
        is_full, text = text_between_tags(content, "t")
        assert text == "second"

    def test_whitespace_stripped(self):
        is_full, text = text_between_tags("<x>  spaced  </x>", "x")
        assert text == "spaced"
