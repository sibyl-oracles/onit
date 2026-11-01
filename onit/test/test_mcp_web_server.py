"""Tests for the web search / fetch / weather MCP server.

This module was at 17% coverage -- the lowest in the tree -- and it is the one
that fetches arbitrary URLs on the model's behalf.  The tests below concentrate
on the parts where a mistake is a security bug rather than a cosmetic one:

* the per-session path jail (``_session_base`` / ``_validate_*_path``)
* API-key redaction in error strings (``_redact_secrets`` / ``_safe_error``)
* the search implementation's shape and its failure mode

Network access is patched out everywhere.  ``DATA_PATH`` is redirected to a
tmp dir per test so nothing touches the real data directory.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import onit.mcp.servers.tasks.web.search.mcp_server as web


@pytest.fixture(autouse=True)
def _isolated_data_path(tmp_path, monkeypatch):
    """Point the jail at a tmp dir and make sure we are not 'in a container'."""
    monkeypatch.setattr(web, "DATA_PATH", str(tmp_path))
    monkeypatch.delenv("ONIT_CONTAINER", raising=False)
    return tmp_path


# ── _validate_required ──────────────────────────────────────────────────────


class TestValidateRequired:
    def test_all_present_is_empty_string(self):
        assert web._validate_required(a=1, b="x") == ""

    def test_missing_is_json_error(self):
        out = json.loads(web._validate_required(a=None, b="x"))
        assert out["status"] == "error"
        assert "a" in out["error"]

    def test_lists_every_missing_name(self):
        out = json.loads(web._validate_required(a=None, b=None))
        assert "a" in out["error"] and "b" in out["error"]

    def test_falsey_but_present_is_not_missing(self):
        """``0`` and ``""`` are values, not absences."""
        assert web._validate_required(a=0, b="") == ""


# ── the path jail ───────────────────────────────────────────────────────────


class TestSessionBase:
    def test_no_data_path_returns_the_server_root(self, tmp_path):
        assert web._session_base() == os.path.realpath(str(tmp_path))

    def test_a_subdirectory_is_allowed(self, tmp_path):
        sub = tmp_path / "session1"
        sub.mkdir()
        assert web._session_base(str(sub)) == os.path.realpath(str(sub))

    def test_a_sibling_session_is_refused(self, tmp_path):
        """One session must not reach another session's folder."""
        outside = tmp_path.parent / "someone-else"
        outside.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="must be within"):
            web._session_base(str(outside))

    def test_traversal_out_of_the_root_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            web._session_base(str(tmp_path / ".." / "escape"))

    def test_container_mode_relaxes_the_jail(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONIT_CONTAINER", "1")
        outside = tmp_path.parent / "anywhere"
        outside.mkdir(exist_ok=True)
        assert web._session_base(str(outside)) == os.path.realpath(str(outside))


class TestValidateWritePath:
    def test_inside_the_root_is_allowed(self, tmp_path):
        p = tmp_path / "f.txt"
        assert web._validate_write_path(str(p)) == os.path.realpath(str(p))

    def test_outside_the_root_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="Write path must be within"):
            web._validate_write_path("/etc/passwd")

    def test_explicit_base_overrides_the_default(self, tmp_path):
        base = tmp_path / "session"
        base.mkdir()
        p = base / "f.txt"
        assert web._validate_write_path(str(p), base=str(base)) == os.path.realpath(str(p))

    def test_a_path_outside_the_explicit_base_is_refused(self, tmp_path):
        base = tmp_path / "session"
        base.mkdir()
        with pytest.raises(ValueError):
            web._validate_write_path(str(tmp_path / "other.txt"), base=str(base))

    def test_container_mode_skips_the_check(self, monkeypatch):
        monkeypatch.setenv("ONIT_CONTAINER", "1")
        assert web._validate_write_path("/etc/passwd") == os.path.realpath("/etc/passwd")


class TestValidateReadPath:
    def test_inside_the_root_is_allowed(self, tmp_path):
        p = tmp_path / "f.txt"
        assert web._validate_read_path(str(p)) == os.path.realpath(str(p))

    def test_outside_the_root_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="Read access denied"):
            web._validate_read_path("/etc/passwd")

    def test_container_mode_skips_the_check(self, monkeypatch):
        monkeypatch.setenv("ONIT_CONTAINER", "1")
        assert web._validate_read_path("/etc/passwd") == os.path.realpath("/etc/passwd")


class TestMediaDir:
    def test_is_under_the_root(self, tmp_path):
        assert web._get_media_dir() == os.path.join(str(tmp_path), "media")

    def test_explicit_base_is_honoured(self, tmp_path):
        base = tmp_path / "session"
        assert web._get_media_dir(str(base)) == os.path.join(str(base), "media")


# ── secret redaction ────────────────────────────────────────────────────────


class TestRedaction:
    def test_appid_value_is_masked(self):
        out = web._redact_secrets("https://api.openweathermap.org/x?appid=SECRET123&q=1")
        assert "SECRET123" not in out
        assert "appid=***" in out

    def test_case_insensitive(self):
        assert "abc" not in web._redact_secrets("APPID=abc")

    def test_none_is_safe(self):
        assert web._redact_secrets(None) == ""

    def test_text_without_a_key_is_unchanged(self):
        assert web._redact_secrets("nothing here") == "nothing here"

    def test_safe_error_hides_request_exception_detail(self):
        """A requests exception embeds the authenticated URL; keep only the class."""
        exc = web.requests.exceptions.ConnectionError("GET https://x?appid=SECRET")
        msg = web._safe_error(exc, "Weather failed")
        assert "SECRET" not in msg
        assert "ConnectionError" in msg
        assert "details withheld" in msg

    def test_safe_error_keeps_a_plain_exception_message(self):
        msg = web._safe_error(ValueError("bad place"), "Weather failed")
        assert "bad place" in msg

    def test_safe_error_truncates(self):
        msg = web._safe_error(ValueError("x" * 500), "Weather failed")
        assert len(msg) < 250


# ── _search_impl ────────────────────────────────────────────────────────────


class TestSearchImpl:
    def test_missing_query_is_refused(self):
        out = json.loads(web._search_impl())
        assert out["status"] == "error"

    def test_web_search_delegates_to_websearch(self):
        with patch.object(web, "WebSearch") as ws:
            ws.return_value.search.return_value = '[{"title": "t"}]'
            out = web._search_impl(query="q", type="web")
        assert json.loads(out)[0]["title"] == "t"

    def test_news_uses_ddgs_and_formats_dates(self):
        fake = MagicMock()
        fake.news.return_value = [
            {
                "title": "T",
                "body": "B",
                "date": "2026-11-01T10:00:00Z",
                "source": "S",
                "url": "u",
            }
        ]
        with patch.object(web, "DDGS", return_value=fake):
            out = json.loads(web._search_impl(query="q", type="news"))
        assert out[0]["date"] == "November 01, 2026"
        assert out[0]["source"] == "S"

    def test_news_snippet_is_capped(self):
        """An uncapped body would flood the model's context."""
        fake = MagicMock()
        fake.news.return_value = [{"title": "T", "body": "x" * 5000, "date": "", "url": "u"}]
        with patch.object(web, "DDGS", return_value=fake):
            out = json.loads(web._search_impl(query="q", type="news"))
        assert len(out[0]["snippet"]) == 300

    def test_news_with_unparseable_date_keeps_the_raw_string(self):
        fake = MagicMock()
        fake.news.return_value = [{"title": "T", "body": "b", "date": "not a date", "url": "u"}]
        with patch.object(web, "DDGS", return_value=fake):
            out = json.loads(web._search_impl(query="q", type="news"))
        assert out[0]["date"] == "not a date"

    def test_max_results_is_capped_at_ten(self):
        fake = MagicMock()
        fake.news.return_value = []
        with patch.object(web, "DDGS", return_value=fake) as ddgs:
            web._search_impl(query="q", type="news", max_results=500)
        assert ddgs.call_args[1]["timeout"] == 10
        assert fake.news.call_args[1]["max_results"] == 10

    def test_failure_returns_json_not_an_exception(self):
        with patch.object(web, "DDGS", side_effect=RuntimeError("network down")):
            out = json.loads(web._search_impl(query="q", type="news"))
        assert "error" in out
        assert "network down" in out["error"]

    def test_failure_redacts_a_key_in_the_message(self):
        with patch.object(web, "DDGS", side_effect=RuntimeError("bad appid=SECRET")):
            out = json.loads(web._search_impl(query="q", type="news"))
        assert "SECRET" not in out["error"]


# ── _get_location_from_ip / _get_coordinates ────────────────────────────────


class TestLocation:
    def test_ip_location_success(self):
        resp = MagicMock()
        resp.json.return_value = {
            "latitude": "1.5",
            "longitude": "2.5",
            "city": "C",
            "country": "X",
        }
        with patch.object(web.requests, "get", return_value=resp):
            lat, lon, city, country = web._get_location_from_ip()
        assert (lat, lon, city, country) == (1.5, 2.5, "C", "X")

    def test_ip_location_uses_https(self):
        """The user's IP must not be sent in cleartext."""
        resp = MagicMock()
        resp.json.return_value = {"latitude": "1", "longitude": "2"}
        with patch.object(web.requests, "get", return_value=resp) as g:
            web._get_location_from_ip()
        assert g.call_args[0][0].startswith("https://")

    def test_ip_location_failure_returns_nones(self):
        with patch.object(web.requests, "get", side_effect=RuntimeError("down")):
            assert web._get_location_from_ip() == (None, None, None, None)

    def test_ip_location_missing_fields_returns_nones(self):
        resp = MagicMock()
        resp.json.return_value = {}
        with patch.object(web.requests, "get", return_value=resp):
            assert web._get_location_from_ip() == (None, None, None, None)

    def test_coordinates_success(self):
        loc = MagicMock(latitude=1.0, longitude=2.0, address="Somewhere")
        with patch.object(web, "Nominatim") as n:
            n.return_value.geocode.return_value = loc
            assert web._get_coordinates("place") == (1.0, 2.0, "Somewhere")

    def test_coordinates_not_found(self):
        with patch.object(web, "Nominatim") as n:
            n.return_value.geocode.return_value = None
            assert web._get_coordinates("nowhere") == (None, None, None)

    def test_coordinates_failure(self):
        with patch.object(web, "Nominatim") as n:
            n.return_value.geocode.side_effect = RuntimeError("down")
            assert web._get_coordinates("x") == (None, None, None)


# ── _get_session ────────────────────────────────────────────────────────────


class TestSession:
    def test_mounts_retry_adapters(self):
        s = web._get_session()
        assert "https://" in s.adapters
        assert "http://" in s.adapters

    def test_sets_a_browser_user_agent(self):
        """Some sites 403 the default python-requests agent."""
        assert "Mozilla" in web._get_session().headers["User-Agent"]


# ── _extract_media_urls ─────────────────────────────────────────────────────


class TestExtractMediaUrls:
    def _soup(self, html):
        from bs4 import BeautifulSoup

        return BeautifulSoup(html, "html.parser")

    def test_relative_src_is_absolutised(self):
        out = web._extract_media_urls(self._soup('<img src="/a.png">'), "https://x.com/p/")
        assert out["images"] == ["https://x.com/a.png"]

    def test_lazy_attributes_are_honoured(self):
        out = web._extract_media_urls(self._soup('<img data-src="/b.png">'), "https://x.com/")
        assert "https://x.com/b.png" in out["images"]

    def test_tiny_images_are_dropped(self):
        out = web._extract_media_urls(
            self._soup('<img src="/t.png" width="10" height="10">'), "https://x.com/"
        )
        assert out["images"] == []

    def test_tracking_pixels_are_dropped(self):
        out = web._extract_media_urls(self._soup('<img src="/pixel.gif">'), "https://x.com/")
        assert out["images"] == []

    def test_srcset_is_expanded(self):
        out = web._extract_media_urls(
            self._soup('<img srcset="/a.png 1x, /b.png 2x">'), "https://x.com/"
        )
        assert "https://x.com/a.png" in out["images"]
        assert "https://x.com/b.png" in out["images"]

    def test_css_background_images(self):
        out = web._extract_media_urls(
            self._soup('<div style="background: url(/bg.jpg)"></div>'), "https://x.com/"
        )
        assert "https://x.com/bg.jpg" in out["images"]

    def test_video_and_source_tags(self):
        out = web._extract_media_urls(
            self._soup('<video src="/v.mp4"><source src="/s.mp4"></video>'), "https://x.com/"
        )
        assert "https://x.com/v.mp4" in out["videos"]
        assert "https://x.com/s.mp4" in out["videos"]

    def test_youtube_and_vimeo_embeds(self):
        out = web._extract_media_urls(
            self._soup(
                '<iframe src="https://youtube.com/embed/1"></iframe>'
                '<iframe src="https://vimeo.com/2"></iframe>'
            ),
            "https://x.com/",
        )
        assert len(out["videos"]) == 2

    def test_results_are_capped(self):
        html = "".join(f'<img src="/{i}.png">' for i in range(80))
        out = web._extract_media_urls(self._soup(html), "https://x.com/")
        assert len(out["images"]) == 50

    def test_duplicates_are_removed(self):
        out = web._extract_media_urls(
            self._soup('<img src="/a.png"><img src="/a.png">'), "https://x.com/"
        )
        assert out["images"].count("https://x.com/a.png") == 1


# ── fetch_content ───────────────────────────────────────────────────────────


class TestFetchContent:
    """``fetch_content`` goes through ``safe_fetch``, not a bare session.

    Patching ``safe_fetch`` is the only seam that keeps these tests off the
    network: the function screens the URL for SSRF and re-screens every
    redirect hop, so it is also the right thing to stub rather than bypass.
    """

    def _resp(
        self, body=b"<html><body><p>Hello</p></body></html>", headers=None, url="https://x.com"
    ):
        r = MagicMock()
        r.status_code = 200
        r.headers = headers or {}
        r.url = url
        r.iter_content.return_value = [body]
        r.raise_for_status.return_value = None
        r.encoding = "utf-8"
        return r

    def test_missing_url_is_refused(self):
        out = json.loads(web.fetch_content())
        assert out["status"] == "error"

    def test_unsupported_extension_is_refused(self):
        out = json.loads(web.fetch_content(url="https://x.com/a.zip"))
        assert "Unsupported file type" in out["error"]

    def test_scheme_is_added_when_absent(self):
        with patch.object(web, "safe_fetch", return_value=self._resp()) as f:
            web.fetch_content(url="x.com")
        assert f.call_args[0][1].startswith("https://")

    def test_html_body_is_extracted(self):
        with patch.object(web, "safe_fetch", return_value=self._resp()):
            out = json.loads(web.fetch_content(url="https://x.com"))
        assert "Hello" in out["content"]

    def test_a_non_public_url_is_refused_before_any_fetch(self):
        """The SSRF screen runs before the request is issued."""
        with patch.object(web, "safe_fetch") as f:
            out = json.loads(web.fetch_content(url="http://127.0.0.1/admin"))
        f.assert_not_called()
        assert "error" in out

    def test_timeout_is_reported(self):
        with patch.object(web, "safe_fetch", side_effect=web.requests.exceptions.Timeout("slow")):
            out = json.loads(web.fetch_content(url="https://x.com"))
        assert "timed out" in out["error"]

    def test_network_failure_returns_json(self):
        with patch.object(
            web, "safe_fetch", side_effect=web.requests.exceptions.ConnectionError("down")
        ):
            out = json.loads(web.fetch_content(url="https://x.com"))
        assert "error" in out

    def test_oversized_content_length_is_refused(self):
        with patch.object(
            web, "safe_fetch", return_value=self._resp(headers={"Content-Length": str(10**9)})
        ):
            out = json.loads(web.fetch_content(url="https://x.com"))
        assert "too large" in out["error"]

    def test_pdf_url_uses_the_pdf_reader(self):
        with patch.object(web, "_read_pdf", return_value="pdf text") as r:
            out = json.loads(web.fetch_content(url="https://x.com/a.pdf"))
        r.assert_called_once()
        assert out["content"] == "pdf text"
        assert out["content_type"] == "application/pdf"


# ── weather ─────────────────────────────────────────────────────────────────


class TestWeather:
    def test_no_api_key_is_a_clear_error(self, monkeypatch):
        monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
        monkeypatch.delenv("OPENWEATHERMAP_API_KEY", raising=False)
        monkeypatch.setattr(web, "openweather_api_key", None)
        out = json.loads(web._get_weather_impl())
        assert "API key not set" in out["error"]

    def test_key_is_read_from_env_on_demand(self, monkeypatch):
        """The CLI can set the key after module import."""
        monkeypatch.setattr(web, "openweather_api_key", None)
        monkeypatch.setenv("OPENWEATHER_API_KEY", "k")
        with patch.object(web, "_get_coordinates", return_value=(None, None, None)):
            with patch.object(web, "_get_location_from_ip", return_value=(None, None, None, None)):
                out = json.loads(web._get_weather_impl())
        assert "API key not set" not in out.get("error", "")

    def test_no_location_is_reported(self, monkeypatch):
        monkeypatch.setattr(web, "openweather_api_key", "k")
        with patch.object(web, "_get_coordinates", return_value=(None, None, None)):
            with patch.object(web, "_get_location_from_ip", return_value=(None, None, None, None)):
                out = json.loads(web._get_weather_impl())
        assert "location" in json.dumps(out).lower()


# ── _download_file ──────────────────────────────────────────────────────────


class TestDownloadFile:
    def _resp(self, body=b"DATA", headers=None):
        r = MagicMock()
        r.headers = headers if headers is not None else {"Content-Type": "image/png"}
        r.iter_content.return_value = [body]
        r.raise_for_status.return_value = None
        return r

    def test_writes_the_file_and_reports_its_size(self, tmp_path):
        with patch.object(web, "safe_fetch", return_value=self._resp()):
            out = web._download_file("https://x.com/a.png", str(tmp_path))
        assert out["size_bytes"] == 4
        assert os.path.exists(out["path"])

    def test_filename_gets_a_url_hash(self, tmp_path):
        """Two different URLs with the same basename must not collide."""
        with patch.object(web, "safe_fetch", return_value=self._resp()):
            a = web._download_file("https://x.com/1/a.png", str(tmp_path))
            b = web._download_file("https://x.com/2/a.png", str(tmp_path))
        assert a["path"] != b["path"]

    def test_extension_is_derived_from_content_type(self, tmp_path):
        with patch.object(web, "safe_fetch", return_value=self._resp()):
            out = web._download_file("https://x.com/image", str(tmp_path))
        assert out["path"].endswith(".png")

    def test_file_is_owner_only(self, tmp_path):
        with patch.object(web, "safe_fetch", return_value=self._resp()):
            out = web._download_file("https://x.com/a.png", str(tmp_path))
        assert (os.stat(out["path"]).st_mode & 0o777) == 0o600

    def test_oversized_is_refused(self, tmp_path):
        with patch.object(
            web, "safe_fetch", return_value=self._resp(headers={"Content-Length": str(10**9)})
        ):
            out = web._download_file("https://x.com/a.png", str(tmp_path))
        assert "too large" in out["error"]

    def test_incomplete_download_is_removed(self, tmp_path):
        """A short read must not leave a truncated file behind."""
        with patch.object(
            web, "safe_fetch", return_value=self._resp(headers={"Content-Length": "100"})
        ):
            out = web._download_file("https://x.com/a.png", str(tmp_path))
        assert "Incomplete" in out["error"]
        assert not os.path.exists(os.path.join(str(tmp_path), "a.png"))

    def test_non_public_url_is_refused(self, tmp_path):
        with patch.object(web, "safe_fetch") as f:
            out = web._download_file("http://169.254.169.254/latest", str(tmp_path))
        f.assert_not_called()
        assert "error" in out

    def test_request_error_is_withheld(self, tmp_path):
        with patch.object(
            web, "safe_fetch", side_effect=web.requests.exceptions.ConnectionError("x")
        ):
            out = web._download_file("https://x.com/a.png", str(tmp_path))
        assert "details withheld" in out["error"]


# ── weather success path ────────────────────────────────────────────────────


class TestWeatherSuccess:
    def _payload(self):
        return {
            "cod": 200,
            "id": 123,
            "sys": {"sunrise": 1700000000, "sunset": 1700040000},
            "weather": [{"description": "clear sky"}],
            "main": {"temp": 20.0, "feels_like": 19.0, "humidity": 50},
            "wind": {"speed": 3.5},
        }

    def _get(self, payload):
        r = MagicMock()
        r.json.return_value = payload
        return r

    def test_current_weather_is_formatted(self, monkeypatch):
        monkeypatch.setattr(web, "openweather_api_key", "k")
        with patch.object(web, "_get_coordinates", return_value=(1.0, 2.0, "Place")):
            with patch.object(web.requests, "get", return_value=self._get(self._payload())):
                out = json.loads(web._get_weather_impl(place="Place"))
        assert out["current"]["description"] == "clear sky"
        assert out["current"]["temperature_c"] == 20.0
        assert out["location"] == "Place"
        assert "openweathermap.org" in out["source"]

    def test_forecast_is_included_when_requested(self, monkeypatch):
        monkeypatch.setattr(web, "openweather_api_key", "k")
        fc = {
            "cod": "200",
            "list": [{"dt": 1700000000, "weather": [{"description": "rain"}], "main": {"temp": 5}}],
        }
        calls = [self._get(self._payload()), self._get(fc)]
        with patch.object(web, "_get_coordinates", return_value=(1.0, 2.0, "P")):
            with patch.object(web.requests, "get", side_effect=calls):
                out = json.loads(web._get_weather_impl(place="P", forecast=True))
        assert out["forecast_5day"][0]["description"] == "rain"

    def test_api_error_is_reported(self, monkeypatch):
        monkeypatch.setattr(web, "openweather_api_key", "k")
        with patch.object(web, "_get_coordinates", return_value=(1.0, 2.0, "P")):
            with patch.object(
                web.requests, "get", return_value=self._get({"cod": 401, "message": "bad key"})
            ):
                out = json.loads(web._get_weather_impl(place="P"))
        assert "bad key" in out["error"]

    def test_the_api_key_is_never_in_the_error(self, monkeypatch):
        """The key rides in the query string; a failure must not echo it."""
        monkeypatch.setattr(web, "openweather_api_key", "SECRETKEY")
        with patch.object(web, "_get_coordinates", return_value=(1.0, 2.0, "P")):
            with patch.object(
                web.requests,
                "get",
                side_effect=web.requests.exceptions.ConnectionError("appid=SECRETKEY"),
            ):
                out = json.loads(web._get_weather_impl(place="P"))
        assert "SECRETKEY" not in json.dumps(out)
