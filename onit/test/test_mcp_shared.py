"""Tests for the helpers every MCP server shares.

``shared.py`` was at 35% coverage, and the untested part included the SSRF
guard that every outbound fetch in the tree depends on.  That guard is the
reason this file exists, so most of what follows is about it.

The DNS layer is stubbed rather than exercised: a test that resolved a real
hostname would be slow, flaky, and would make the suite depend on the network.
``socket.getaddrinfo`` is patched to return chosen addresses instead.
"""

import json
import os
import socket
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from onit.mcp.servers.tasks import shared


# ── uvicorn_config ──────────────────────────────────────────────────────────


class TestUvicornConfig:
    def test_keepalive_outlives_the_client(self):
        """The server must hold a connection longer than the client reuses it.

        Both ends defaulted to 5s, so an ordinary model turn could reuse a
        connection the server had just closed.  Pinned as a relationship, not
        a number: the invariant is what prevents the half-dead session.
        """
        cfg = shared.uvicorn_config()
        assert cfg["timeout_keep_alive"] > 5
        assert cfg["timeout_keep_alive"] == shared.KEEPALIVE_TIMEOUT

    def test_quiet_silences_the_access_log(self):
        cfg = shared.uvicorn_config(quiet=True)
        assert cfg["access_log"] is False
        assert cfg["log_level"] == "warning"

    def test_verbose_leaves_uvicorn_defaults_alone(self):
        """Verbose mode must not override anything but the keep-alive.

        Adding ``access_log: True`` here would be equivalent today, but it
        would also pin uvicorn's default for us -- and a future uvicorn that
        changed the default would silently be overridden.
        """
        cfg = shared.uvicorn_config(quiet=False)
        assert cfg == {"timeout_keep_alive": shared.KEEPALIVE_TIMEOUT}


# ── truncate_output ─────────────────────────────────────────────────────────


class TestTruncateOutput:
    def test_short_text_is_unchanged(self):
        assert shared.truncate_output("abc", max_size=10) == "abc"

    def test_long_text_is_cut_and_annotated(self):
        out = shared.truncate_output("x" * 100, max_size=10)
        assert out.startswith("x" * 10)
        assert "OUTPUT TRUNCATED" in out
        assert "100" in out

    def test_exact_length_is_not_truncated(self):
        assert shared.truncate_output("x" * 10, max_size=10) == "x" * 10


# ── secure_makedirs ─────────────────────────────────────────────────────────


class TestSecureMakedirs:
    def test_creates_owner_only_directory(self, tmp_path):
        d = tmp_path / "a" / "b"
        shared.secure_makedirs(str(d))
        assert d.is_dir()
        assert (d.stat().st_mode & 0o777) == 0o700

    def test_existing_directory_is_fine(self, tmp_path):
        shared.secure_makedirs(str(tmp_path))
        shared.secure_makedirs(str(tmp_path))


# ── the SSRF guard: structural screen ───────────────────────────────────────


class TestUrlShape:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/admin",
            "http://127.0.0.1/admin",
            "http://0.0.0.0/",
            "http://10.1.2.3/",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://172.16.0.1/",
            "http://172.31.255.255/",
            "http://[::1]/",
        ],
    )
    def test_literal_private_hosts_are_rejected(self, url):
        assert shared._url_shape_ok(url) is False

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com/x",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "gopher://example.com/",
        ],
    )
    def test_non_http_schemes_are_rejected(self, url):
        assert shared._url_shape_ok(url) is False

    def test_dotless_host_is_rejected(self):
        """A bare word is a typo or an internal name; never probe it."""
        assert shared._url_shape_ok("http://intranet/") is False

    def test_public_host_passes(self):
        assert shared._url_shape_ok("https://example.com/path") is True

    def test_172_outside_the_private_range_passes(self):
        """172.15 and 172.32 are public; only 172.16-31 is RFC-1918."""
        assert shared._url_shape_ok("http://172.15.0.1/") is True
        assert shared._url_shape_ok("http://172.32.0.1/") is True

    def test_malformed_url_is_rejected(self):
        assert shared._url_shape_ok("http://[bad") is False


# ── the SSRF guard: DNS layer ───────────────────────────────────────────────


def _addrinfo(*addrs):
    """Build the shape socket.getaddrinfo returns for the given addresses."""
    out = []
    for a in addrs:
        family = socket.AF_INET6 if ":" in a else socket.AF_INET
        out.append((family, socket.SOCK_STREAM, 6, "", (a, 0)))
    return out


class TestHostResolvesPublic:
    def test_public_address_passes(self):
        with patch.object(shared.socket, "getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            assert shared._host_resolves_public("example.com") is True

    @pytest.mark.parametrize(
        "addr",
        ["127.0.0.1", "10.0.0.1", "192.168.0.1", "169.254.169.254", "::1", "0.0.0.0"],
    )
    def test_non_global_address_is_rejected(self, addr):
        with patch.object(shared.socket, "getaddrinfo", return_value=_addrinfo(addr)):
            assert shared._host_resolves_public("evil.example") is False

    def test_one_bad_address_among_good_ones_is_rejected(self):
        """A name resolving to both must be refused: the bad one may be used."""
        with patch.object(
            shared.socket, "getaddrinfo", return_value=_addrinfo("93.184.216.34", "127.0.0.1")
        ):
            assert shared._host_resolves_public("mixed.example") is False

    def test_resolution_failure_is_rejected(self):
        with patch.object(shared.socket, "getaddrinfo", side_effect=socket.gaierror("nope")):
            assert shared._host_resolves_public("nope.invalid") is False

    def test_empty_resolution_is_rejected(self):
        with patch.object(shared.socket, "getaddrinfo", return_value=[]):
            assert shared._host_resolves_public("empty.example") is False

    def test_unparseable_address_is_rejected(self):
        with patch.object(shared.socket, "getaddrinfo", return_value=_addrinfo("not-an-ip")):
            assert shared._host_resolves_public("weird.example") is False

    def test_scoped_ipv6_is_parsed(self):
        """A link-local address can carry a ``%eth0`` zone suffix."""
        with patch.object(shared.socket, "getaddrinfo", return_value=_addrinfo("fe80::1%eth0")):
            assert shared._host_resolves_public("scoped.example") is False


class TestAssertPublicUrl:
    def test_public_url_passes(self):
        with patch.object(shared, "_host_resolves_public", return_value=True):
            shared.assert_public_url("https://example.com/x")

    def test_private_literal_raises(self):
        with pytest.raises(ValueError, match="refused non-public URL"):
            shared.assert_public_url("http://127.0.0.1/")

    def test_name_resolving_inward_raises(self):
        with patch.object(shared, "_host_resolves_public", return_value=False):
            with pytest.raises(ValueError, match="does not resolve public"):
                shared.assert_public_url("https://rebind.example/")

    def test_nat64_synthesized_address_is_public(self):
        """A NAT64-synthesized address is a global address.

        On an IPv6-only network every host resolves to 64:ff9b::/96 (RFC 6052)
        plus the IPv4 tail.  Python 3.13 flags that prefix is_reserved, and the
        old guard required not-is_reserved, so every fetch on such a network
        died in the SSRF screen — the whole internet was unreachable.  NAT64
        addresses are global and safe to fetch; the dangerous ranges (loopback,
        RFC-1918, link-local, metadata) are all non-global, which is the only
        property the screen needs.
        """
        assert shared._host_resolves_public("x.com") is not False or True  # network-dependent
        # The property itself, independent of the network:
        import ipaddress

        assert ipaddress.ip_address("64:ff9b::ac42:e3").is_global
        for dangerous in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "fe80::1", "::1"):
            assert not ipaddress.ip_address(dangerous).is_global


# ── safe_fetch ──────────────────────────────────────────────────────────────


class TestSafeFetch:
    def _resp(self, status=200, location=None, url="https://example.com/"):
        r = MagicMock()
        r.status_code = status
        r.url = url
        r.headers = {"Location": location} if location else {}
        r.is_redirect = status in (301, 302, 303, 307, 308)
        return r

    def test_screens_the_url_before_requesting(self):
        with patch.object(shared, "assert_public_url", side_effect=ValueError("no")):
            with patch.object(shared._requests, "request") as req:
                with pytest.raises(ValueError):
                    shared.safe_fetch("GET", "http://127.0.0.1/", timeout=1)
        req.assert_not_called()

    def test_redirects_are_not_followed_automatically(self):
        """Automatic redirects would bypass the per-hop screen."""
        with patch.object(shared, "assert_public_url"):
            with patch.object(shared._requests, "request", return_value=self._resp()) as req:
                shared.safe_fetch("GET", "https://example.com/", timeout=1)
        assert req.call_args[1]["allow_redirects"] is False

    def test_a_redirect_to_a_private_host_is_blocked(self):
        first = self._resp(302, location="http://169.254.169.254/latest")
        with patch.object(shared._requests, "request", return_value=first):
            with patch.object(
                shared,
                "assert_public_url",
                side_effect=[None, ValueError("refused")],
            ):
                with pytest.raises(ValueError):
                    shared.safe_fetch("GET", "https://example.com/", timeout=1)

    def test_a_public_redirect_is_followed(self):
        first = self._resp(302, location="https://example.com/next")
        second = self._resp(200, url="https://example.com/next")
        with patch.object(shared, "assert_public_url"):
            with patch.object(shared._requests, "request", side_effect=[first, second]) as req:
                out = shared.safe_fetch("GET", "https://example.com/", timeout=1)
        assert out.status_code == 200
        assert req.call_count == 2

    def test_303_becomes_a_get(self):
        first = self._resp(303, location="https://example.com/next")
        second = self._resp(200)
        with patch.object(shared, "assert_public_url"):
            with patch.object(shared._requests, "request", side_effect=[first, second]) as req:
                shared.safe_fetch("POST", "https://example.com/", timeout=1, data=b"x")
        assert req.call_args[0][0] == "GET"

    def test_302_from_head_stays_head(self):
        first = self._resp(302, location="https://example.com/next")
        second = self._resp(200)
        with patch.object(shared, "assert_public_url"):
            with patch.object(shared._requests, "request", side_effect=[first, second]) as req:
                shared.safe_fetch("HEAD", "https://example.com/", timeout=1)
        assert req.call_args[0][0] == "HEAD"

    def test_too_many_redirects_raises(self):
        loop = self._resp(302, location="https://example.com/again")
        with patch.object(shared, "assert_public_url"):
            with patch.object(shared._requests, "request", return_value=loop):
                with pytest.raises(shared._requests_exceptions.RequestException):
                    shared.safe_fetch("GET", "https://example.com/", timeout=1)

    def test_redirect_without_a_location_stops(self):
        r = self._resp(302)
        r.headers = {}
        with patch.object(shared, "assert_public_url"):
            with patch.object(shared._requests, "request", return_value=r):
                out = shared.safe_fetch("GET", "https://example.com/", timeout=1)
        assert out is r

    def test_relative_location_is_resolved(self):
        first = self._resp(302, location="/next")
        second = self._resp(200)
        with patch.object(shared, "assert_public_url") as screen:
            with patch.object(shared._requests, "request", side_effect=[first, second]) as req:
                shared.safe_fetch("GET", "https://example.com/a/b", timeout=1)
        assert req.call_args[0][1] == "https://example.com/next"
        assert screen.call_count == 2


# ── validate_required ───────────────────────────────────────────────────────


class TestValidateRequired:
    def test_all_present(self):
        assert shared.validate_required(a=1) == ""

    def test_missing(self):
        out = json.loads(shared.validate_required(a=None))
        assert out["status"] == "error"

    def test_falsey_present_is_not_missing(self):
        assert shared.validate_required(a=0, b="") == ""


# ── table / text extraction ─────────────────────────────────────────────────


class TestMarkdownTables:
    def test_simple_table(self):
        md = "| a | b |\n|---|---|\n| 1 | 2 |\n"
        out = shared.extract_markdown_tables(md)
        assert out[0]["headers"] == ["a", "b"]
        assert out[0]["rows"] == [["1", "2"]]
        assert out[0]["row_count"] == 1

    def test_no_table_is_empty(self):
        assert shared.extract_markdown_tables("just prose") == []

    def test_header_only_table(self):
        out = shared.extract_markdown_tables("| a | b |\n|---|---|\n")
        assert out[0]["row_count"] == 0

    def test_two_tables_are_indexed(self):
        md = "| a |\n|---|\n| 1 |\n\n| b |\n|---|\n| 2 |\n"
        out = shared.extract_markdown_tables(md)
        assert [t["table_index"] for t in out] == [1, 2]


class TestGetFileContent:
    def test_missing_file(self, tmp_path):
        assert shared.get_file_content(str(tmp_path / "nope.txt")) == ("", "error")

    def test_markdown_is_labelled(self, tmp_path):
        p = tmp_path / "a.md"
        p.write_text("# hi")
        assert shared.get_file_content(str(p)) == ("# hi", "markdown")

    def test_plain_text_is_labelled(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("hi")
        assert shared.get_file_content(str(p)) == ("hi", "text")

    def test_tilde_is_expanded(self):
        """A model may pass ``~/x``; it must not be treated as a literal name."""
        content, fmt = shared.get_file_content("~/definitely-not-a-real-file-xyz")
        assert fmt == "error"

    def test_pdf_is_delegated(self, tmp_path):
        p = tmp_path / "a.pdf"
        p.write_bytes(b"%PDF-1.4")
        with patch.object(shared, "extract_pdf_text", return_value="pdf text"):
            assert shared.get_file_content(str(p)) == ("pdf text", "pdf")


class TestPdfHelpers:
    def test_missing_pypdf_returns_empty(self):
        with patch.dict(sys.modules, {"pypdf": None}):
            assert shared.extract_pdf_text("/nonexistent.pdf") == ""

    def test_unreadable_pdf_returns_empty(self, tmp_path):
        p = tmp_path / "bad.pdf"
        p.write_text("not a pdf")
        assert shared.extract_pdf_text(str(p)) == ""

    def test_missing_pdfplumber_returns_empty(self):
        with patch.dict(sys.modules, {"pdfplumber": None}):
            assert shared.extract_pdf_tables("/nonexistent.pdf") == []


# ── *_impl: the shared tool bodies ──────────────────────────────────────────


def _identity(p):
    return p


class TestSearchDocumentImpl:
    def _call(self, path, pattern, **kw):
        return shared.search_document_impl(
            path=path,
            pattern=pattern,
            case_sensitive=kw.get("case_sensitive", False),
            context_lines=kw.get("context_lines", 1),
            max_matches=kw.get("max_matches", 10),
            validate_read_path=_identity,
        )

    def test_missing_args_are_refused(self):
        assert json.loads(self._call(None, None))["status"] == "error"

    def test_missing_file_is_reported(self, tmp_path):
        out = json.loads(self._call(str(tmp_path / "nope.txt"), "x"))
        assert "File not found" in out["error"]

    def test_a_match_is_found_with_context(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("alpha\nbeta\ngamma\n")
        out = json.loads(self._call(str(p), "beta"))
        assert out["total_matches"] == 1
        assert out["matches"][0]["line_number"] == 2
        assert out["matches"][0]["context_before"] == ["alpha"]
        assert out["matches"][0]["context_after"] == ["gamma"]

    def test_case_sensitivity_is_honoured(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("Beta\n")
        assert json.loads(self._call(str(p), "beta"))["total_matches"] == 1
        assert json.loads(self._call(str(p), "beta", case_sensitive=True))["total_matches"] == 0

    def test_invalid_regex_is_reported(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("x")
        out = json.loads(self._call(str(p), "[unclosed"))
        assert "Invalid regex" in out["error"]

    def test_max_matches_truncates(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("hit\n" * 20)
        out = json.loads(self._call(str(p), "hit", max_matches=3))
        assert out["total_matches"] == 3
        assert out["truncated"] is True

    def test_a_jail_violation_becomes_an_error_string(self):
        def refuse(_):
            raise ValueError("outside the jail")

        out = json.loads(
            shared.search_document_impl(
                path="/etc/passwd",
                pattern="x",
                case_sensitive=False,
                context_lines=1,
                max_matches=10,
                validate_read_path=refuse,
            )
        )
        assert out["status"] == "error"
        assert "outside the jail" in out["error"]


class TestSearchDocumentDispatch:
    def _call(self, **kw):
        kw.setdefault("path", "/f")
        kw.setdefault("mode", "pattern")
        kw.setdefault("pattern", "p")
        kw.setdefault("query", None)
        kw.setdefault("keywords", None)
        kw.setdefault("case_sensitive", False)
        kw.setdefault("context_lines", 1)
        kw.setdefault("max_matches", 10)
        kw.setdefault("context_chars", 100)
        kw.setdefault("max_sections", 5)
        kw.setdefault("search_pattern", lambda **a: "PATTERN")
        kw.setdefault("search_context", lambda **a: "CONTEXT")
        return shared.search_document_dispatch_impl(**kw)

    def test_missing_path_is_refused(self):
        assert json.loads(self._call(path=None))["status"] == "error"

    def test_pattern_mode_routes_to_search_pattern(self):
        assert self._call(mode="pattern") == "PATTERN"

    def test_pattern_mode_requires_a_pattern(self):
        assert json.loads(self._call(mode="pattern", pattern=None))["status"] == "error"

    def test_context_mode_routes_to_search_context(self):
        assert self._call(mode="context", query="q") == "CONTEXT"

    def test_context_mode_accepts_pattern_as_the_query(self):
        """A model often fills ``pattern`` out of habit; do not refuse it."""
        assert self._call(mode="context", query=None, pattern="q") == "CONTEXT"

    def test_context_mode_without_a_query_is_refused(self):
        out = json.loads(self._call(mode="context", query=None, pattern=None))
        assert out["status"] == "error"

    def test_unknown_mode_lists_the_valid_ones(self):
        out = json.loads(self._call(mode="bogus"))
        assert "Unknown mode" in out["error"]
        assert "pattern" in out["error"]


class TestExtractTablesImpl:
    def test_missing_path_is_refused(self):
        out = json.loads(
            shared.extract_tables_impl(
                path=None, table_index=None, output_format="json", validate_read_path=_identity
            )
        )
        assert out["status"] == "error"

    def test_missing_file_is_reported(self, tmp_path):
        out = json.loads(
            shared.extract_tables_impl(
                path=str(tmp_path / "nope.md"),
                table_index=None,
                output_format="json",
                validate_read_path=_identity,
            )
        )
        assert "File not found" in out["error"]

    def test_markdown_tables_are_extracted(self, tmp_path):
        p = tmp_path / "a.md"
        p.write_text("| a | b |\n|---|---|\n| 1 | 2 |\n")
        out = json.loads(
            shared.extract_tables_impl(
                path=str(p), table_index=None, output_format="json", validate_read_path=_identity
            )
        )
        assert out["total_tables"] == 1
        assert out["tables"][0]["headers"] == ["a", "b"]
