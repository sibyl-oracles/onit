"""Tests for the web search provider chain.

``WebSearch`` had no tests (20% coverage) despite being on the path of every
web search the agent performs.  The behaviour that matters is the *chain*:
Tavily when a key is set, then Ollama, then DuckDuckGo -- and a failure at any
tier must fall through rather than surface as an exception or an empty answer.

No test here touches the network.  Each provider method is patched at the
``WebSearch`` boundary, and the provider implementations themselves are
exercised against mocked ``requests`` / ``ollama`` / ``DDGS`` objects.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from onit.mcp.servers.tasks.web.search.web_search import WebSearch


@pytest.fixture(autouse=True)
def _no_tavily_key(monkeypatch):
    """A developer's real TAVILY_API_KEY must not change which tier runs."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


# ── _clean_content ──────────────────────────────────────────────────────────


class TestCleanContent:
    def test_empty_is_empty(self):
        assert WebSearch()._clean_content("") == ""
        assert WebSearch()._clean_content(None) == ""

    def test_strips_html_tags(self):
        assert WebSearch()._clean_content("<p>hello <b>world</b></p>") == "hello world"

    def test_unescapes_entities(self):
        assert WebSearch()._clean_content("a &amp; b &quot;c&quot;") == 'a & b "c"'

    def test_escaped_tags_are_unescaped_then_stripped(self):
        """``&lt;b&gt;`` becomes a real tag and is then removed.

        Order matters: unescaping runs before tag stripping, so an escaped
        tag does not survive as visible text.  Pinned because the reverse
        order would leak ``<b>`` into the model's context.
        """
        assert WebSearch()._clean_content("a &lt;b&gt;bold&lt;/b&gt;") == "a bold"

    def test_collapses_whitespace(self):
        assert WebSearch()._clean_content("a\n\n\t  b") == "a b"

    def test_removes_broken_bracket_markup(self):
        """``[/cite]`` and ``[cite]`` are artefacts of scraped text."""
        assert WebSearch()._clean_content("text[/cite] more[cite]") == "text more"

    def test_truncates_on_a_word_boundary(self):
        ws = WebSearch(max_content_length=20)
        out = ws._clean_content("alpha beta gamma delta epsilon")
        assert out.endswith("...")
        assert len(out) <= 23
        # Must not cut mid-word: the character before "..." is not a letter
        # that was split from its word.
        assert not out[:-3].endswith("gam")

    def test_short_content_is_not_truncated(self):
        assert WebSearch(max_content_length=100)._clean_content("short") == "short"


# ── provider implementations ────────────────────────────────────────────────


class TestTavily:
    def test_no_key_means_no_call(self):
        ws = WebSearch()
        assert ws.tavily_api_key == ""
        assert ws._search_tavily("q") is None

    def test_success_maps_results(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        ws = WebSearch()
        resp = MagicMock()
        resp.json.return_value = {"results": [{"title": "T", "url": "u", "content": "c"}]}
        with patch("onit.mcp.servers.tasks.web.search.web_search.requests.post", return_value=resp):
            out = ws._search_tavily("q")
        assert out[0]["source"] == "tavily"
        assert out[0]["title"] == "T"

    def test_http_error_falls_through(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        ws = WebSearch()
        resp = MagicMock()
        resp.raise_for_status.side_effect = RuntimeError("500")
        with patch("onit.mcp.servers.tasks.web.search.web_search.requests.post", return_value=resp):
            assert ws._search_tavily("q") is None

    def test_empty_results_is_none(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        ws = WebSearch()
        resp = MagicMock()
        resp.json.return_value = {"results": []}
        with patch("onit.mcp.servers.tasks.web.search.web_search.requests.post", return_value=resp):
            assert ws._search_tavily("q") is None


class TestOllama:
    def test_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.OLLAMA_AVAILABLE", False)
        assert WebSearch()._search_ollama("q") is None

    def test_success_maps_results(self, monkeypatch):
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.OLLAMA_AVAILABLE", True)
        fake = MagicMock()
        fake.web_search.return_value = {"results": [{"title": "T", "url": "u", "content": "c"}]}
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.ollama", fake)
        out = WebSearch()._search_ollama("q")
        assert out[0]["source"] == "ollama"

    def test_empty_response_is_none(self, monkeypatch):
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.OLLAMA_AVAILABLE", True)
        fake = MagicMock()
        fake.web_search.return_value = None
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.ollama", fake)
        assert WebSearch()._search_ollama("q") is None

    def test_exception_is_swallowed(self, monkeypatch):
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.OLLAMA_AVAILABLE", True)
        fake = MagicMock()
        fake.web_search.side_effect = RuntimeError("down")
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.ollama", fake)
        assert WebSearch()._search_ollama("q") is None


class TestDuckDuckGo:
    def test_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.DDGS_AVAILABLE", False)
        assert WebSearch()._search_ddgs("q") is None

    def test_success_maps_href(self, monkeypatch):
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.DDGS_AVAILABLE", True)
        fake_cls = MagicMock()
        fake_cls.return_value.text.return_value = [{"title": "T", "href": "u", "body": "b"}]
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.DDGS", fake_cls)
        out = WebSearch()._search_ddgs("q")
        assert out[0]["url"] == "u"
        assert out[0]["source"] == "duckduckgo"

    def test_falls_back_to_link_key(self, monkeypatch):
        """Some ddgs versions return ``link`` rather than ``href``."""
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.DDGS_AVAILABLE", True)
        fake_cls = MagicMock()
        fake_cls.return_value.text.return_value = [{"title": "T", "link": "u2", "body": "b"}]
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.DDGS", fake_cls)
        assert WebSearch()._search_ddgs("q")[0]["url"] == "u2"

    def test_exception_is_swallowed(self, monkeypatch):
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.DDGS_AVAILABLE", True)
        fake_cls = MagicMock()
        fake_cls.return_value.text.side_effect = RuntimeError("blocked")
        monkeypatch.setattr("onit.mcp.servers.tasks.web.search.web_search.DDGS", fake_cls)
        assert WebSearch()._search_ddgs("q") is None


# ── the chain ───────────────────────────────────────────────────────────────


class TestSearchChain:
    def test_empty_query_is_refused(self):
        out = json.loads(WebSearch().search("   "))
        assert out["error"] == "Empty search query"

    def test_tavily_wins_when_it_answers(self):
        ws = WebSearch()
        with patch.object(ws, "_search_tavily", return_value=[{"title": "t"}]):
            with patch.object(ws, "_search_ollama") as ollama:
                out = json.loads(ws.search("q"))
        assert out[0]["title"] == "t"
        ollama.assert_not_called()

    def test_ollama_runs_when_tavily_is_empty(self):
        ws = WebSearch()
        with patch.object(ws, "_search_tavily", return_value=None):
            with patch.object(ws, "_search_ollama", return_value=[{"title": "o"}]):
                with patch.object(ws, "_search_ddgs") as ddg:
                    out = json.loads(ws.search("q"))
        assert out[0]["title"] == "o"
        ddg.assert_not_called()

    def test_duckduckgo_is_the_last_resort(self):
        ws = WebSearch()
        with patch.object(ws, "_search_tavily", return_value=None):
            with patch.object(ws, "_search_ollama", return_value=None):
                with patch.object(ws, "_search_ddgs", return_value=[{"title": "d"}]):
                    out = json.loads(ws.search("q"))
        assert out[0]["title"] == "d"

    def test_fallback_can_be_disabled(self):
        ws = WebSearch(use_fallback=False)
        with patch.object(ws, "_search_tavily", return_value=None):
            with patch.object(ws, "_search_ollama", return_value=None):
                with patch.object(ws, "_search_ddgs") as ddg:
                    out = json.loads(ws.search("q"))
        ddg.assert_not_called()
        assert "error" in out

    def test_all_providers_failing_names_what_was_tried(self):
        ws = WebSearch()
        with patch.object(ws, "_search_tavily", return_value=None):
            with patch.object(ws, "_search_ollama", return_value=None):
                with patch.object(ws, "_search_ddgs", return_value=None):
                    out = json.loads(ws.search("q"))
        assert out["query"] == "q"
        assert out["providers_tried"] == ["ollama", "duckduckgo"]

    def test_providers_tried_includes_tavily_when_configured(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        ws = WebSearch()
        with patch.object(ws, "_search_tavily", return_value=None):
            with patch.object(ws, "_search_ollama", return_value=None):
                with patch.object(ws, "_search_ddgs", return_value=None):
                    out = json.loads(ws.search("q"))
        assert out["providers_tried"][0] == "tavily"

    def test_query_is_stripped(self):
        ws = WebSearch()
        with patch.object(ws, "_search_tavily", return_value=[{"title": "t"}]) as t:
            ws.search("  padded  ")
        t.assert_called_once_with("padded")

    def test_non_ascii_survives_json(self):
        """``ensure_ascii=False`` keeps CJK readable rather than \\uXXXX."""
        ws = WebSearch()
        with patch.object(ws, "_search_tavily", return_value=[{"title": "日本語"}]):
            out = ws.search("q")
        assert "日本語" in out


# ── search_with_metadata ────────────────────────────────────────────────────


class TestSearchWithMetadata:
    def test_empty_query(self):
        out = WebSearch().search_with_metadata("")
        assert out["count"] == 0
        assert out["results"] == []

    def test_reports_which_source_answered(self):
        ws = WebSearch()
        with patch.object(ws, "_search_tavily", return_value=None):
            with patch.object(ws, "_search_ollama", return_value=[{"title": "o"}]):
                out = ws.search_with_metadata("q")
        assert out["source"] == "ollama"
        assert out["success"] is True
        assert out["count"] == 1

    def test_failure_is_success_false(self):
        ws = WebSearch()
        with patch.object(ws, "_search_tavily", return_value=None):
            with patch.object(ws, "_search_ollama", return_value=None):
                with patch.object(ws, "_search_ddgs", return_value=None):
                    out = ws.search_with_metadata("q")
        assert out["success"] is False
        assert out["count"] == 0
        assert out["source"] is None
