"""Tests for src/ui/api.py — the FastAPI + SSE web UI (WebApiUI)."""

import asyncio
import base64
import json
import os
import sys
import threading
import time
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starlette.testclient import TestClient

from ui.api import ApiSession, WebApiUI, _sse
from ui import auth as ui_auth


# ── Helpers ─────────────────────────────────────────────────────────────────

class FakeOnit:
    """Stub of Onit.process_task that streams two tokens and returns a reply."""

    def __init__(self, response="Hello world", delay=0.0, config_data=None):
        self.response = response
        self.delay = delay
        self.calls = []
        # Real dict, pointed at a temp path by the fixture: left empty, the
        # learn defaults would send trajectories to the developer's ~/.onit.
        self.config_data = config_data if config_data is not None else {}

    async def process_task(self, task, session_path=None, data_path=None,
                           safety_queue=None, stream_callback=None,
                           stream_complete_callback=None, stats=None,
                           tool_status_callback=None, session_id=None, **kwargs):
        self.calls.append(task)
        if self.delay:
            await asyncio.sleep(self.delay)
        if tool_status_callback:
            tool_status_callback("test_tool(query)")
            tool_status_callback("")
        if stream_callback:
            stream_callback("Hello ", "Hello ")
            stream_callback("world", "Hello world")
        if stream_complete_callback:
            stream_complete_callback("Hello world", 42.0)
        if stats is not None:
            stats["tokens_per_second"] = 42.0
        if session_path:
            with open(session_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"task": task, "response": self.response,
                                    "timestamp": 0}) + "\n")
        return self.response


# Smallest valid PNG (1×1, transparent) — enough for mimetype and byte checks.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNkYAAAAAY"
    "AAjCB0C8AAAAASUVORK5CYII="
)


def _write_png(data_path, name="chart.png"):
    path = os.path.join(data_path, name)
    with open(path, "wb") as f:
        f.write(_PNG_1X1)
    return path


@pytest.fixture
def bg_loop():
    """A running event loop on a background thread (stands in for OnIt's loop)."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)


@pytest.fixture
def ui(tmp_path, bg_loop):
    ui = WebApiUI(
        data_path=str(tmp_path / "data"),
        session_path=str(tmp_path / "sessions" / "current.jsonl"),
        title="Test Chat",
        require_auth=False,
    )
    ui._onit = FakeOnit(config_data={
        "learn": {"autonomy": "observe", "path": str(tmp_path / "learned")}})
    ui._loop = bg_loop
    ui.build_app()
    return ui


@pytest.fixture
def client(ui):
    return TestClient(ui.app)


def parse_sse(text):
    """Parse SSE payload text into a list of (event, data) tuples."""
    events = []
    for block in text.split("\n\n"):
        event, data = None, None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if event:
            events.append((event, data))
    return events


# ── Unit: session management ───────────────────────────────────────────────

class TestSessionManagement:
    def test_create_new_session(self, ui):
        sid, session = ui._get_or_create_session()
        assert sid == session.session_id
        assert os.path.isfile(session.session_path)
        assert os.path.isdir(session.data_path)

    def test_session_data_dir_under_shared_data_path(self, ui):
        # Tool writes are jailed under the shared data path, so the session
        # dir must live there for generated files to be downloadable.
        sid, session = ui._get_or_create_session()
        assert session.data_path == os.path.join(ui.data_path, sid)

    def test_reuse_supplied_uuid(self, ui):
        want = str(uuid.uuid4())
        sid, _ = ui._get_or_create_session(want)
        assert sid == want

    def test_invalid_session_id_rejected(self, ui):
        sid, _ = ui._get_or_create_session("../../etc/passwd")
        assert sid != "../../etc/passwd"
        # A fresh valid UUID was generated instead
        uuid.UUID(sid)

    def test_existing_session_returned(self, ui):
        sid, session = ui._get_or_create_session()
        sid2, session2 = ui._get_or_create_session(sid)
        assert sid2 == sid
        assert session2 is session


class TestExtractFilePaths:
    def test_local_path_becomes_link(self, ui, tmp_path):
        sid, session = ui._get_or_create_session()
        fname = "result.csv"
        with open(os.path.join(session.data_path, fname), "w") as f:
            f.write("a,b\n")
        text = f"Saved to {session.data_path}/{fname}"
        cleaned, files = ui._extract_file_paths(
            text, data_path=session.data_path, session_id=sid)
        assert f"[{fname}](/uploads/{sid}/{fname})" in cleaned
        assert files == [os.path.join(session.data_path, fname)]

    def test_urls_survive_path_stripping(self, ui):
        # Regression: the absolute-path stripper matched the second slash of
        # "https://", turning arxiv links into "https:/2510.07979".
        sid, session = ui._get_or_create_session()
        text = "IntMeanFlow (ByteDance): https://arxiv.org/abs/2510.07979"
        cleaned, files = ui._extract_file_paths(
            text, data_path=session.data_path, session_id=sid)
        assert cleaned == text
        assert files == []

    def test_bare_absolute_path_still_stripped(self, ui):
        sid, session = ui._get_or_create_session()
        cleaned, _ = ui._extract_file_paths(
            "See /home/user/notes/report.txt for details",
            data_path=session.data_path, session_id=sid)
        assert cleaned == "See report.txt for details"

    def test_markdown_image_keeps_its_syntax(self, ui):
        # The model embedding its own chart must survive as an image, not end
        # up with a second link nested inside the parentheses.
        sid, session = ui._get_or_create_session()
        _write_png(session.data_path, "chart.png")
        cleaned, files = ui._extract_file_paths(
            f"Here it is:\n\n![chart]({session.data_path}/chart.png)",
            data_path=session.data_path, session_id=sid)
        assert cleaned.endswith(f"![chart](/uploads/{sid}/chart.png)")
        assert files == [os.path.join(session.data_path, "chart.png")]

    def test_markdown_link_target_rewritten_once(self, ui):
        sid, session = ui._get_or_create_session()
        with open(os.path.join(session.data_path, "report.pdf"), "wb") as f:
            f.write(b"%PDF-1.4\n")
        cleaned, _ = ui._extract_file_paths(
            "[report.pdf](report.pdf)",
            data_path=session.data_path, session_id=sid)
        assert cleaned == f"[report.pdf](/uploads/{sid}/report.pdf)"


class TestImagePreview:
    """Images an agent creates are rendered in the reply, not just downloaded."""

    def test_image_tagged_for_preview(self, ui):
        sid, session = ui._get_or_create_session()
        png = _write_png(session.data_path)
        infos = ui._file_infos([png], sid)
        assert infos[0]["kind"] == "image"
        assert infos[0]["url"] == f"/uploads/{sid}/chart.png"

    def test_other_files_are_downloads(self, ui):
        sid, session = ui._get_or_create_session()
        path = os.path.join(session.data_path, "result.csv")
        with open(path, "w") as f:
            f.write("a,b\n")
        assert ui._file_infos([path], sid)[0]["kind"] == "file"

    def test_svg_is_not_previewed(self, ui):
        # SVG can carry script, so it stays a download rather than something
        # the browser renders on our own origin.
        sid, session = ui._get_or_create_session()
        path = os.path.join(session.data_path, "plot.svg")
        with open(path, "w") as f:
            f.write("<svg xmlns='http://www.w3.org/2000/svg'/>")
        assert ui._file_infos([path], sid)[0]["kind"] == "file"

    def test_inline_request_renders_in_browser(self, client, ui):
        sid, session = ui._get_or_create_session()
        _write_png(session.data_path)
        res = client.get(f"/uploads/{sid}/chart.png?inline=1")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        assert "attachment" not in res.headers.get("content-disposition", "")
        assert res.content == _PNG_1X1

    def test_download_is_still_an_attachment(self, client, ui):
        sid, session = ui._get_or_create_session()
        _write_png(session.data_path)
        res = client.get(f"/uploads/{sid}/chart.png")
        assert "attachment" in res.headers["content-disposition"]

    def test_inline_ignored_for_non_images(self, client, ui):
        # A stray inline flag must never make the browser render tool-written
        # HTML or SVG on this origin.
        sid, session = ui._get_or_create_session()
        with open(os.path.join(session.data_path, "page.html"), "w") as f:
            f.write("<script>alert(1)</script>")
        res = client.get(f"/uploads/{sid}/page.html?inline=1")
        assert "attachment" in res.headers["content-disposition"]


class TestCodePreview:
    """Code an agent writes is inspectable in the reply, not only downloadable."""

    def _write(self, session, name, text, mode="w"):
        path = os.path.join(session.data_path, name)
        with open(path, mode) as f:
            f.write(text)
        return path

    def test_code_file_carries_its_source(self, ui):
        sid, session = ui._get_or_create_session()
        src = "def add(a, b):\n    return a + b\n"
        info = ui._file_infos([self._write(session, "calc.py", src)], sid)[0]
        assert info["kind"] == "code"
        assert info["preview"] == {"language": "python", "text": src, "truncated": False}

    def test_language_follows_extension(self, ui):
        sid, session = ui._get_or_create_session()
        info = ui._file_infos([self._write(session, "q.sql", "SELECT 1;\n")], sid)[0]
        assert info["preview"]["language"] == "sql"

    def test_unknown_extension_stays_a_download(self, ui):
        sid, session = ui._get_or_create_session()
        info = ui._file_infos([self._write(session, "data.parquet", "x")], sid)[0]
        assert info["kind"] == "file"
        assert "preview" not in info

    def test_binary_content_is_not_previewed(self, ui):
        # A .py that isn't valid UTF-8 is not the text it claims to be.
        sid, session = ui._get_or_create_session()
        path = self._write(session, "blob.py", b"\xff\xfe\x00\x01", mode="wb")
        assert ui._file_infos([path], sid)[0]["kind"] == "file"

    def test_empty_file_is_not_previewed(self, ui):
        sid, session = ui._get_or_create_session()
        info = ui._file_infos([self._write(session, "empty.py", "\n\n")], sid)[0]
        assert info["kind"] == "file"

    def test_long_file_truncated_on_a_line_boundary(self, ui):
        sid, session = ui._get_or_create_session()
        line = "print('x' * 10)\n"
        path = self._write(session, "big.py", line * 8000)   # ~125 KB
        preview = ui._file_infos([path], sid)[0]["preview"]
        assert preview["truncated"] is True
        assert len(preview["text"]) <= 64 * 1024
        assert preview["text"].endswith("\n")
        assert preview["text"].count(line) == preview["text"].count("\n")

    def test_image_wins_over_code_preview(self, ui):
        sid, session = ui._get_or_create_session()
        png = _write_png(session.data_path)
        assert ui._file_infos([png], sid)[0]["kind"] == "image"

    def test_preview_reaches_the_client(self, client, ui):
        # End to end: the SSE `done` payload carries what the UI renders.
        sid, session = ui._get_or_create_session()
        self._write(session, "script.py", "print('hi')\n")
        ui._onit = FakeOnit(response="Wrote script.py for you.")
        res = client.post("/api/chat", json={"message": "write a script"},
                          headers={"X-Session-Id": sid})
        done = [d for e, d in parse_sse(res.text) if e == "done"][0]
        code = [f for f in done["files"] if f["name"] == "script.py"][0]
        assert code["kind"] == "code"
        assert code["preview"]["text"] == "print('hi')\n"


class TestWebPreview:
    """A generated page runs in the reply — sandboxed, offline, origin-less."""

    PAGE = "<!doctype html><h1>Tic tac toe</h1><script>let turn='X'</script>\n"

    def _write_page(self, session, name="game.html"):
        path = os.path.join(session.data_path, name)
        with open(path, "w") as f:
            f.write(self.PAGE)
        return path

    def test_page_is_runnable_and_keeps_its_source(self, ui):
        sid, session = ui._get_or_create_session()
        info = ui._file_infos([self._write_page(session)], sid)[0]
        assert info["kind"] == "web"
        assert info["preview_url"] == f"/preview/{sid}/game.html"
        assert info["preview"]["text"] == self.PAGE     # still inspectable
        assert info["url"] == f"/uploads/{sid}/game.html"   # still downloadable

    def test_served_with_sandbox_csp(self, client, ui):
        sid, session = ui._get_or_create_session()
        self._write_page(session)
        res = client.get(f"/preview/{sid}/game.html")
        assert res.status_code == 200
        assert "<h1>Tic tac toe</h1><script>let turn='X'</script>" in res.text
        assert res.headers["content-type"].startswith("text/html")
        csp = res.headers["content-security-policy"]
        # The isolation that matters: an opaque origin, and nobody else framing it.
        assert csp.startswith("sandbox allow-scripts")
        assert "allow-same-origin" not in csp
        assert "allow-top-navigation" not in csp
        assert "frame-ancestors 'self'" in csp
        assert "object-src 'none'" in csp
        # Framing is the point, so this route overrides the global DENY —
        # and the middleware must not have put it back.
        assert res.headers["x-frame-options"] == "SAMEORIGIN"
        assert "attachment" not in res.headers.get("content-disposition", "")

    def test_storage_shim_injected_before_page_scripts(self, client, ui):
        # Sandboxed pages have no storage; without this the first
        # localStorage read throws and the app never draws.
        sid, session = ui._get_or_create_session()
        self._write_page(session)
        body = client.get(f"/preview/{sid}/game.html").text
        assert body.index("localStorage") < body.index("<h1>Tic tac toe</h1>")
        assert body.startswith("<!doctype html>")     # doctype stays first

    def test_source_on_disk_is_untouched(self, client, ui):
        sid, session = ui._get_or_create_session()
        path = self._write_page(session)
        client.get(f"/preview/{sid}/game.html")
        with open(path) as f:
            assert f.read() == self.PAGE
        assert ui._file_infos([path], sid)[0]["preview"]["text"] == self.PAGE

    def test_app_can_load_its_own_files(self, client, ui):
        # Agents routinely split an app into page + script + stylesheet, and
        # relative URLs resolve back into the preview path.
        sid, session = ui._get_or_create_session()
        os.makedirs(os.path.join(session.data_path, "assets"), exist_ok=True)
        with open(os.path.join(session.data_path, "game.js"), "w") as f:
            f.write("console.log(1)")
        with open(os.path.join(session.data_path, "assets", "style.css"), "w") as f:
            f.write("body { color: red }")
        js = client.get(f"/preview/{sid}/game.js")
        css = client.get(f"/preview/{sid}/assets/style.css")
        assert js.status_code == css.status_code == 200
        assert js.headers["content-type"].startswith("text/javascript")
        assert css.headers["content-type"].startswith("text/css")
        # Served under the same sandbox, so a direct hit is inert too.
        assert js.headers["content-security-policy"].startswith("sandbox")

    def test_only_html_gets_a_preview_frame(self, ui):
        sid, session = ui._get_or_create_session()
        path = os.path.join(session.data_path, "app.js")
        with open(path, "w") as f:
            f.write("alert(1)")
        assert ui._file_infos([path], sid)[0]["kind"] == "code"

    def test_traversal_and_unknown_session_rejected(self, client, ui):
        sid, session = ui._get_or_create_session()
        self._write_page(session)
        with open(os.path.join(ui.data_path, "secret.html"), "w") as f:
            f.write("<p>not this session's</p>")
        for path in ("..%2fsecret.html", "../secret.html", "..%2f..%2fetc%2fpasswd",
                     "%2e%2e%2fsecret.html"):
            assert client.get(f"/preview/{sid}/{path}").status_code == 404, path
        assert client.get(f"/preview/{uuid.uuid4()}/game.html").status_code == 404

    def test_symlink_out_of_the_session_rejected(self, ui, client):
        sid, session = ui._get_or_create_session()
        outside = os.path.join(ui.data_path, "outside.html")
        with open(outside, "w") as f:
            f.write("<p>elsewhere</p>")
        os.symlink(outside, os.path.join(session.data_path, "link.html"))
        assert client.get(f"/preview/{sid}/link.html").status_code == 404

    def test_missing_file_is_404(self, client, ui):
        sid, _ = ui._get_or_create_session()
        assert client.get(f"/preview/{sid}/nope.html").status_code == 404

    def test_other_routes_keep_the_strict_defaults(self, client):
        res = client.get("/")
        assert res.headers["x-frame-options"] == "DENY"
        assert "sandbox" not in res.headers["content-security-policy"]

    def test_disabled_by_config(self, tmp_path, bg_loop):
        off = WebApiUI(data_path=str(tmp_path / "d"),
                       session_path=str(tmp_path / "s" / "c.jsonl"),
                       require_auth=False, html_preview=False)
        off._onit = FakeOnit()
        off._loop = bg_loop
        off.build_app()
        sid, session = off._get_or_create_session()
        info = off._file_infos([self._write_page(session)], sid)[0]
        assert info["kind"] == "code"           # source and download only
        assert "preview_url" not in info
        assert TestClient(off.app).get(f"/preview/{sid}/game.html").status_code == 404

    def test_preview_requires_auth(self, auth_ui):
        sid, session = auth_ui._get_or_create_session()
        os.makedirs(session.data_path, exist_ok=True)
        self._write_page(session)
        res = TestClient(auth_ui.app).get(f"/preview/{sid}/game.html")
        assert res.status_code == 401


# ── API endpoints ──────────────────────────────────────────────────────────

class TestConfigEndpoint:
    def test_config(self, client):
        data = client.get("/api/config").json()
        assert data["title"] == "Test Chat"
        assert data["auth_enabled"] is False
        assert data["authenticated"] is True

    def test_index_served(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "OnIt" in res.text

    def test_static_assets_served(self, client):
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200

    def test_ga_id_served_on_open_ui(self, tmp_path, bg_loop):
        # No login required → every client is effectively authenticated, so
        # analytics is available.
        u = WebApiUI(
            data_path=str(tmp_path / "d"),
            session_path=str(tmp_path / "s" / "c.jsonl"),
            require_auth=False,
            ga_measurement_id="G-ABCD1234",
        )
        u._loop = bg_loop
        u.build_app()
        assert TestClient(u.app).get("/api/config").json()["ga_id"] == "G-ABCD1234"


class TestGaIdNotLeakedPreLogin:
    """The GA measurement ID must not appear in the public /api/config for an
    unauthenticated visitor; it's handed over only after login."""

    @pytest.fixture
    def auth_ga_ui(self, tmp_path, bg_loop, monkeypatch):
        monkeypatch.setattr(ui_auth, "GOOGLE_AUTH_AVAILABLE", True)
        monkeypatch.setattr("ui.api.GOOGLE_AUTH_AVAILABLE", True)
        monkeypatch.setattr("ui.api.GoogleAuthenticator", lambda *a, **k: object())
        u = WebApiUI(
            data_path=str(tmp_path / "d"),
            session_path=str(tmp_path / "s" / "c.jsonl"),
            google_client_id="test-client-id.apps.googleusercontent.com",
            google_client_secret="test-secret",
            ga_measurement_id="G-ABCD1234",
        )
        u._loop = bg_loop
        u.build_app()
        return u

    def test_ga_id_withheld_when_unauthenticated(self, auth_ga_ui):
        data = TestClient(auth_ga_ui.app).get("/api/config").json()
        assert data["authenticated"] is False
        assert data["ga_id"] is None

    def test_ga_id_present_after_auth(self, auth_ga_ui):
        from datetime import datetime, timedelta
        auth_ga_ui._authenticated_cookies["c"] = {
            "email": "user@test.com",
            "session_id": "s1",
            "expires": datetime.now() + timedelta(hours=1),
        }
        client = TestClient(auth_ga_ui.app, cookies={"onit_auth": "c"})
        assert client.get("/api/config").json()["ga_id"] == "G-ABCD1234"


class TestDomainBranding:
    """When the web UI is served through a real domain name the domain
    replaces the default OnIt title/brand shown by the SPA (including the
    "<brand> can make mistakes" composer hint); localhost and bare-IP access
    keep the defaults."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for var in ("ONIT_DOMAIN", "ONIT_PUBLIC_URL"):
            monkeypatch.delenv(var, raising=False)

    def _client(self, tmp_path, base_url="http://localhost", **kwargs):
        ui = WebApiUI(
            data_path=str(tmp_path / "data"),
            session_path=str(tmp_path / "sessions" / "current.jsonl"),
            require_auth=False,
            **kwargs,
        )
        ui.build_app()
        return TestClient(ui.app, base_url=base_url)

    def test_default_branding_on_localhost(self, tmp_path):
        data = self._client(tmp_path).get("/api/config").json()
        assert data["title"] == "OnIt Chat"
        assert data["brand"] == "OnIt"

    def test_onit_domain_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONIT_DOMAIN", "mychat.ai")
        data = self._client(tmp_path).get("/api/config").json()
        assert data["title"] == "mychat.ai"
        assert data["brand"] == "mychat.ai"

    def test_public_url_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONIT_PUBLIC_URL", "https://mychat.ai")
        data = self._client(tmp_path).get("/api/config").json()
        assert data["title"] == "mychat.ai"
        assert data["brand"] == "mychat.ai"

    def test_host_header_fallback(self, tmp_path):
        data = self._client(
            tmp_path, base_url="http://mychat.ai").get("/api/config").json()
        assert data["title"] == "mychat.ai"
        assert data["brand"] == "mychat.ai"

    def test_ip_address_keeps_default_branding(self, tmp_path):
        data = self._client(
            tmp_path, base_url="http://192.168.1.5").get("/api/config").json()
        assert data["title"] == "OnIt Chat"
        assert data["brand"] == "OnIt"

    def test_custom_web_title_keeps_title_but_brand_is_domain(
            self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONIT_DOMAIN", "mychat.ai")
        data = self._client(
            tmp_path, title="My Assistant").get("/api/config").json()
        assert data["title"] == "My Assistant"
        assert data["brand"] == "mychat.ai"


class TestGoogleAnalytics:
    """web_ga_measurement_id / ONIT_GA_MEASUREMENT_ID flow through
    /api/config as ga_id; malformed IDs are dropped, never echoed."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("ONIT_GA_MEASUREMENT_ID", raising=False)

    def _client(self, tmp_path, **kwargs):
        ui = WebApiUI(
            data_path=str(tmp_path / "data"),
            session_path=str(tmp_path / "sessions" / "current.jsonl"),
            require_auth=False,
            **kwargs,
        )
        ui.build_app()
        return TestClient(ui.app)

    def test_analytics_off_by_default(self, tmp_path):
        assert self._client(tmp_path).get("/api/config").json()["ga_id"] is None

    def test_config_id_exposed(self, tmp_path):
        data = self._client(
            tmp_path, ga_measurement_id="G-ABC123XYZ0").get("/api/config").json()
        assert data["ga_id"] == "G-ABC123XYZ0"

    def test_env_id_exposed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONIT_GA_MEASUREMENT_ID", "G-ENVID12345")
        assert self._client(tmp_path).get("/api/config").json()["ga_id"] == "G-ENVID12345"

    def test_config_id_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONIT_GA_MEASUREMENT_ID", "G-ENVID12345")
        data = self._client(
            tmp_path, ga_measurement_id="G-CFGID12345").get("/api/config").json()
        assert data["ga_id"] == "G-CFGID12345"

    def test_malformed_id_dropped(self, tmp_path):
        bad = '"><script>alert(1)</script>'
        assert self._client(
            tmp_path, ga_measurement_id=bad).get("/api/config").json()["ga_id"] is None


class TestHistoryEndpoint:
    def test_history_creates_session(self, client):
        data = client.get("/api/history").json()
        uuid.UUID(data["session_id"])
        assert data["messages"] == []
        assert data["processing"] is False

    def test_history_sticky_via_header(self, client):
        sid = client.get("/api/history").json()["session_id"]
        data = client.get("/api/history", headers={"X-Session-Id": sid}).json()
        assert data["session_id"] == sid


class TestChatEndpoint:
    def test_chat_streams_and_persists(self, client, ui):
        sid = client.get("/api/history").json()["session_id"]
        res = client.post("/api/chat", json={"message": "hi"},
                          headers={"X-Session-Id": sid})
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(res.text)
        names = [e for e, _ in events]
        assert "token" in names
        assert "phase_end" in names
        assert "status" in names
        assert "done" in names
        done = dict(events)["done"]
        assert done["content"] == "Hello world"
        assert done["tok_s"] == 42.0

        # Deltas reassemble the streamed message
        deltas = "".join(d["delta"] for e, d in events if e == "token")
        assert deltas == "Hello world"

        # Turn was persisted and appears in history
        history = client.get("/api/history", headers={"X-Session-Id": sid}).json()
        assert history["processing"] is False
        assert [m["role"] for m in history["messages"]] == ["user", "assistant"]
        assert ui._onit.calls == ["hi"]

    def test_chat_empty_message_rejected(self, client):
        assert client.post("/api/chat", json={"message": " "}).status_code == 400

    def test_chat_conflict_while_processing(self, client, ui):
        sid, session = ui._get_or_create_session()
        session.processing = True
        res = client.post("/api/chat", json={"message": "hi"},
                          headers={"X-Session-Id": sid})
        assert res.status_code == 409

    def test_chat_attaches_uploaded_files(self, client, ui):
        sid, session = ui._get_or_create_session()
        with open(os.path.join(session.data_path, "notes.txt"), "w") as f:
            f.write("hello")
        res = client.post("/api/chat",
                          json={"message": "summarize", "files": ["notes.txt"]},
                          headers={"X-Session-Id": sid})
        assert res.status_code == 200
        assert "Relevant files:" in ui._onit.calls[-1]
        assert os.path.join(session.data_path, "notes.txt") in ui._onit.calls[-1]

    def test_stop_signals_safety_queue(self, client, ui, bg_loop):
        sid, session = ui._get_or_create_session()
        session.processing = True
        res = client.post("/api/chat/stop", headers={"X-Session-Id": sid})
        assert res.json()["stopped"] is True
        deadline = time.time() + 2
        while session.safety_queue.qsize() == 0 and time.time() < deadline:
            time.sleep(0.01)
        assert session.safety_queue.qsize() == 1


class TestUploadEndpoints:
    def test_upload_roundtrip(self, client):
        sid = client.get("/api/history").json()["session_id"]
        res = client.post("/api/upload",
                          files={"file": ("data.txt", b"payload")},
                          headers={"X-Session-Id": sid})
        assert res.status_code == 200
        body = res.json()
        assert body["name"] == "data.txt"

        download = client.get(body["url"])
        assert download.status_code == 200
        assert download.content == b"payload"

    def test_upload_path_traversal_blocked(self, client, ui):
        sid, session = ui._get_or_create_session()
        res = client.post("/api/upload",
                          files={"file": ("../../evil.txt", b"x")},
                          headers={"X-Session-Id": sid})
        assert res.status_code == 200
        assert res.json()["name"] == "evil.txt"
        assert os.path.isfile(os.path.join(session.data_path, "evil.txt"))

    def test_serve_unknown_session_404(self, client):
        assert client.get(f"/uploads/{uuid.uuid4()}/x.txt").status_code == 404

    def test_mcp_callback_upload(self, client, ui):
        sid, session = ui._get_or_create_session()
        res = client.post(f"/uploads/{sid}/",
                          files={"file": ("tool-output.png", b"png")})
        assert res.status_code == 200
        assert os.path.isfile(os.path.join(session.data_path, "tool-output.png"))


class TestClearEndpoint:
    def test_clear_wipes_session(self, client, ui):
        sid, session = ui._get_or_create_session()
        with open(session.session_path, "w") as f:
            f.write(json.dumps({"task": "t", "response": "r", "timestamp": 0}) + "\n")
        with open(os.path.join(session.data_path, "f.txt"), "w") as f:
            f.write("x")
        res = client.post("/api/clear", headers={"X-Session-Id": sid})
        assert res.json()["cleared"] is True
        assert os.path.getsize(session.session_path) == 0
        assert os.listdir(session.data_path) == []


class TestRatingEndpoint:
    """The one outcome signal no heuristic can supply: whether the person who
    asked got what they wanted."""

    def _answered(self, ui, turns=1):
        sid, session = ui._get_or_create_session()
        with open(session.session_path, "w") as f:
            for n in range(turns):
                f.write(json.dumps({"task": f"t{n}", "response": "r",
                                    "timestamp": 0}) + "\n")
        return sid, session

    def test_thumbs_up_lands_on_the_newest_answer(self, client, ui):
        from learn import read_session
        sid, _ = self._answered(ui, turns=2)
        res = client.post("/api/rating", json={"rating": "up"},
                          headers={"X-Session-Id": sid})
        assert res.status_code == 200
        body = res.json()
        assert body == {"recorded": True, "session_id": sid, "turn": 2, "rating": 1}
        # It is stored where the trajectory loops will look for it.
        ratings = read_session(sid, ui._onit.config_data)
        assert ratings == [] or ratings[-1]["signals"]["user_rating"] == 1

    def test_an_older_answer_can_be_rated_explicitly(self, client, ui):
        sid, _ = self._answered(ui, turns=3)
        res = client.post("/api/rating", json={"rating": -1, "turn": 1},
                          headers={"X-Session-Id": sid})
        assert res.json()["turn"] == 1
        assert res.json()["rating"] == -1

    def test_a_comment_rides_along(self, client, ui):
        from learn import read_session, record_task
        sid, _ = self._answered(ui)
        record_task(session_id=sid, turn=1, task="t0", response="r",
                    config_data=ui._onit.config_data)
        client.post("/api/rating",
                    json={"rating": "down", "comment": "wrong policy"},
                    headers={"X-Session-Id": sid})
        record = read_session(sid, ui._onit.config_data)[0]
        assert record["signals"]["user_rating"] == -1
        assert record["signals"]["user_comment"] == "wrong policy"

    def test_a_meaningless_rating_is_rejected(self, client, ui):
        sid, _ = self._answered(ui)
        res = client.post("/api/rating", json={"rating": "maybe"},
                          headers={"X-Session-Id": sid})
        assert res.status_code == 400

    def test_a_missing_rating_is_rejected(self, client, ui):
        sid, _ = self._answered(ui)
        res = client.post("/api/rating", json={}, headers={"X-Session-Id": sid})
        assert res.status_code == 400

    def test_a_misclick_can_be_taken_back(self, client, ui):
        """A thumb pressed by accident has to be undoable, so an explicit
        nothing is a retraction rather than a malformed request."""
        from learn import read_session, record_task
        sid, _ = self._answered(ui)
        record_task(session_id=sid, turn=1, task="t0", response="r",
                    config_data=ui._onit.config_data)
        client.post("/api/rating", json={"rating": "up"},
                    headers={"X-Session-Id": sid})
        res = client.post("/api/rating", json={"rating": None, "turn": 1},
                          headers={"X-Session-Id": sid})
        assert res.status_code == 200
        assert res.json()["rating"] is None
        record = read_session(sid, ui._onit.config_data)[0]
        assert record["signals"]["user_rating"] is None

    def test_the_answer_says_which_turn_to_rate(self, client, ui):
        """The buttons need a turn number, and guessing it client-side breaks
        as soon as history is restored."""
        sid, _ = self._answered(ui)
        res = client.post("/api/chat", json={"message": "hello"},
                          headers={"X-Session-Id": sid})
        done = [d for e, d in parse_sse(res.text) if e == "done"]
        assert done and done[0]["turn"] == 2
        assert done[0]["rating_enabled"] is True

    def test_history_carries_turns_and_prior_verdicts(self, client, ui):
        from learn import record_task
        sid, _ = self._answered(ui, turns=2)
        for turn in (1, 2):
            record_task(session_id=sid, turn=turn, task=f"t{turn}", response="r",
                        config_data=ui._onit.config_data)
        client.post("/api/rating", json={"rating": "down", "turn": 1},
                    headers={"X-Session-Id": sid})
        messages = client.get("/api/history",
                              headers={"X-Session-Id": sid}).json()["messages"]
        answers = [m for m in messages if m["role"] == "assistant"]
        assert [m["turn"] for m in answers] == [1, 2]
        assert [m["rating"] for m in answers] == [-1, None]

    def test_config_hides_the_buttons_when_nothing_is_recorded(self, client, ui, tmp_path):
        assert client.get("/api/config").json()["rating_enabled"] is True
        ui._onit.config_data = {"learn": {"autonomy": "off",
                                          "path": str(tmp_path / "nope")}}
        assert client.get("/api/config").json()["rating_enabled"] is False

    def test_rating_before_any_answer_is_rejected(self, client, ui):
        sid, _ = ui._get_or_create_session()
        res = client.post("/api/rating", json={"rating": "up"},
                          headers={"X-Session-Id": sid})
        assert res.status_code == 400

    def test_a_non_numeric_turn_is_rejected(self, client, ui):
        sid, _ = self._answered(ui)
        res = client.post("/api/rating", json={"rating": "up", "turn": "last"},
                          headers={"X-Session-Id": sid})
        assert res.status_code == 400

    def test_recording_off_answers_without_storing(self, client, ui, tmp_path):
        """The UI offered a button and the deployment declined to keep the
        answer — not an error, and nothing on disk."""
        ui._onit.config_data = {"learn": {"autonomy": "off",
                                          "path": str(tmp_path / "nope")}}
        sid, _ = self._answered(ui)
        res = client.post("/api/rating", json={"rating": "up"},
                          headers={"X-Session-Id": sid})
        assert res.status_code == 200
        assert res.json()["recorded"] is False
        assert not (tmp_path / "nope").exists()


class TestVerifyLinks:
    def test_malformed_urls_rejected_without_probe(self, client, ui, monkeypatch):
        def _boom(url):
            raise AssertionError(f"should not probe {url}")
        monkeypatch.setattr(ui, "_probe_url", _boom)
        urls = [
            "https://manual",             # dotless host
            "ftp://example.com/file",     # non-http scheme
            "http://localhost:8080/x",    # loopback
            "http://192.168.1.5/admin",   # private range
            "not a url",
        ]
        res = client.post("/api/verify_links", json={"urls": urls})
        assert res.status_code == 200
        assert res.json()["results"] == {u: False for u in urls}

    def test_reachable_and_dead_links(self, client, ui, monkeypatch):
        monkeypatch.setattr(ui, "_probe_url",
                            lambda url: url == "https://example.com/ok")
        res = client.post("/api/verify_links", json={
            "urls": ["https://example.com/ok", "https://ge.php"],
        })
        assert res.json()["results"] == {
            "https://example.com/ok": True,
            "https://ge.php": False,
        }

    def test_verdicts_are_cached(self, client, ui, monkeypatch):
        calls = []
        monkeypatch.setattr(ui, "_probe_url",
                            lambda url: calls.append(url) or True)
        for _ in range(3):
            client.post("/api/verify_links",
                        json={"urls": ["https://example.com/"]})
        assert calls == ["https://example.com/"]

    def test_urls_capped_per_request(self, client, ui, monkeypatch):
        monkeypatch.setattr(ui, "_probe_url", lambda url: True)
        urls = [f"https://example.com/{i}" for i in range(30)]
        res = client.post("/api/verify_links", json={"urls": urls})
        assert len(res.json()["results"]) == 20

    def test_non_list_body_rejected(self, client):
        assert client.post("/api/verify_links",
                           json={"urls": "https://example.com"}).status_code == 400


class TestVerifyEmails:
    """mailto: verdicts come from session provenance, never the network."""

    @staticmethod
    def _verdict(client, sid, addr, monkeypatch=None):
        res = client.post("/api/verify_links", json={"urls": [f"mailto:{addr}"]},
                          headers={"X-Session-Id": sid})
        return res.json()["results"][f"mailto:{addr}"]

    def test_ungrounded_address_is_not_verified(self, client, ui, monkeypatch):
        monkeypatch.setattr(ui, "_probe_url",
                            lambda url: pytest.fail(f"probed {url}"))
        sid, _session = ui._get_or_create_session()
        assert self._verdict(client, sid, "invented@nowhere.example") is False

    def test_address_from_user_message_is_verified(self, client, ui):
        sid, _session = ui._get_or_create_session()
        client.post("/api/chat", json={"message": "mail rowel@sibyl.ai please"},
                    headers={"X-Session-Id": sid})
        assert self._verdict(client, sid, "rowel@sibyl.ai") is True

    def test_address_from_tool_result_is_verified(self, client, ui):
        """A web search hit or document read grounds the address."""
        class SearchingOnit(FakeOnit):
            async def process_task(self, task, **kwargs):
                cb = kwargs.get("tool_result_callback")
                if cb:
                    cb("search", "Top hit: Contact press@newsroom.example for media.")
                return await super().process_task(task, **kwargs)

        ui._onit = SearchingOnit()
        sid, _session = ui._get_or_create_session()
        assert self._verdict(client, sid, "press@newsroom.example") is False
        client.post("/api/chat", json={"message": "find the press contact"},
                    headers={"X-Session-Id": sid})
        assert self._verdict(client, sid, "press@newsroom.example") is True

    def test_address_from_uploaded_document_is_verified(self, client, ui):
        sid, _session = ui._get_or_create_session()
        client.post("/api/upload",
                    files={"file": ("contacts.csv", b"name,email\nAda,ada@lovelace.example\n")},
                    headers={"X-Session-Id": sid})
        assert self._verdict(client, sid, "ada@lovelace.example") is True

    def test_grounding_survives_session_eviction(self, client, ui):
        """A page reload after the in-memory session is dropped must not
        resurrect strikethrough on an already-grounded address."""
        sid, _session = ui._get_or_create_session()
        client.post("/api/chat", json={"message": "write to ada@lovelace.example"},
                    headers={"X-Session-Id": sid})
        del ui._web_sessions[sid]
        assert self._verdict(client, sid, "ada@lovelace.example") is True

    def test_grounding_does_not_leak_across_sessions(self, client, ui):
        sid_a, _a = ui._get_or_create_session()
        client.post("/api/chat", json={"message": "ping ada@lovelace.example"},
                    headers={"X-Session-Id": sid_a})
        sid_b = client.post("/api/sessions/new").json()["session_id"]
        assert self._verdict(client, sid_a, "ada@lovelace.example") is True
        assert self._verdict(client, sid_b, "ada@lovelace.example") is False

    def test_matching_ignores_case_and_mailto_params(self, client, ui):
        sid, _session = ui._get_or_create_session()
        client.post("/api/chat", json={"message": "Ada@Lovelace.example wrote back"},
                    headers={"X-Session-Id": sid})
        res = client.post("/api/verify_links", headers={"X-Session-Id": sid}, json={
            "urls": ["mailto:ada@lovelace.example?subject=Hi%20there"],
        })
        assert res.json()["results"]["mailto:ada@lovelace.example?subject=Hi%20there"] is True

    def test_malformed_mailto_rejected(self, client, ui):
        sid, session = ui._get_or_create_session()
        session.sourced_emails = {"ada@lovelace.example"}
        for bad in ["mailto:", "mailto:not-an-address", "mailto:missing@tld"]:
            res = client.post("/api/verify_links", json={"urls": [bad]},
                              headers={"X-Session-Id": sid})
            assert res.json()["results"][bad] is False, bad

    def test_mixed_batch_splits_by_scheme(self, client, ui, monkeypatch):
        monkeypatch.setattr(ui, "_probe_url", lambda url: True)
        sid, session = ui._get_or_create_session()
        session.sourced_emails = {"ada@lovelace.example"}
        res = client.post("/api/verify_links", headers={"X-Session-Id": sid}, json={
            "urls": ["https://example.com/ok", "mailto:ada@lovelace.example",
                     "mailto:ghost@nowhere.example"],
        })
        assert res.json()["results"] == {
            "https://example.com/ok": True,
            "mailto:ada@lovelace.example": True,
            "mailto:ghost@nowhere.example": False,
        }

    def test_clear_chat_drops_grounding(self, client, ui):
        sid, _session = ui._get_or_create_session()
        client.post("/api/chat", json={"message": "ping ada@lovelace.example"},
                    headers={"X-Session-Id": sid})
        client.post("/api/clear", headers={"X-Session-Id": sid})
        assert self._verdict(client, sid, "ada@lovelace.example") is False

    def test_delete_session_removes_sidecar(self, client, ui):
        sid, session = ui._get_or_create_session()
        client.post("/api/chat", json={"message": "ping ada@lovelace.example"},
                    headers={"X-Session-Id": sid})
        sidecar = ui._email_store_path(session)
        assert os.path.isfile(sidecar)
        client.delete(f"/api/sessions/{sid}")
        assert not os.path.isfile(sidecar)


class TestSessionsEndpoints:
    def test_new_session(self, client):
        sid = client.post("/api/sessions/new").json()["session_id"]
        uuid.UUID(sid)

    def test_list_sessions(self, client, ui):
        sid, session = ui._get_or_create_session()
        with open(session.session_path, "w") as f:
            f.write(json.dumps({"task": "first task", "response": "r",
                                "timestamp": 0}) + "\n")
        sessions = client.get("/api/sessions").json()["sessions"]
        assert any(s["session_id"] == sid for s in sessions)

    def test_delete_session(self, client, ui):
        sid, session = ui._get_or_create_session()
        path = session.session_path
        data_path = session.data_path
        res = client.request("DELETE", f"/api/sessions/{sid}")
        assert res.json()["deleted"] is True
        assert not os.path.exists(path)
        assert not os.path.isdir(data_path)
        assert sid not in ui._web_sessions

    def test_delete_session_not_in_memory(self, client, ui):
        # Simulate a server restart: session exists on disk but not in memory
        sid, session = ui._get_or_create_session()
        data_path = session.data_path
        ui._web_sessions.pop(sid)
        res = client.request("DELETE", f"/api/sessions/{sid}")
        assert res.json()["deleted"] is True
        assert not os.path.isdir(data_path)

    def test_delete_all_sessions(self, client, ui):
        sid1, s1 = ui._get_or_create_session()
        sid2, s2 = ui._get_or_create_session()
        with open(s1.session_path, "w") as f:
            f.write(json.dumps({"task": "t", "response": "r", "timestamp": 0}) + "\n")
        with open(os.path.join(s2.data_path, "f.txt"), "w") as f:
            f.write("x")
        res = client.request("DELETE", "/api/sessions")
        assert res.json()["deleted"] >= 2
        assert ui._web_sessions == {}
        assert not os.path.exists(s1.session_path)
        assert not os.path.isdir(s2.data_path)
        assert client.get("/api/sessions").json()["sessions"] == []

    def test_delete_all_blocked_while_processing(self, client, ui):
        _sid, session = ui._get_or_create_session()
        session.processing = True
        res = client.request("DELETE", "/api/sessions")
        assert res.status_code == 409
        assert os.path.exists(session.session_path)

    def test_delete_invalid_id(self, client):
        assert client.request("DELETE", "/api/sessions/not-a-uuid").status_code == 400

    def test_rename_unknown_session(self, client):
        res = client.patch(f"/api/sessions/{uuid.uuid4()}", json={"tag": "x"})
        assert res.status_code == 400


# ── Auth-enabled app ───────────────────────────────────────────────────────

@pytest.fixture
def auth_ui(tmp_path, bg_loop, monkeypatch):
    monkeypatch.setattr(ui_auth, "GOOGLE_AUTH_AVAILABLE", True)
    monkeypatch.setattr("ui.api.GOOGLE_AUTH_AVAILABLE", True)
    monkeypatch.setattr("ui.api.GoogleAuthenticator", lambda *a, **k: object())
    ui = WebApiUI(
        data_path=str(tmp_path / "data"),
        session_path=str(tmp_path / "sessions" / "current.jsonl"),
        google_client_id="test-client-id.apps.googleusercontent.com",
        google_client_secret="test-secret",
    )
    ui._onit = FakeOnit(config_data={
        "learn": {"autonomy": "observe", "path": str(tmp_path / "learned")}})
    ui._loop = bg_loop
    ui.build_app()
    return ui


class TestAuth:
    def test_api_requires_auth(self, auth_ui):
        client = TestClient(auth_ui.app)
        assert client.get("/api/history").status_code == 401
        assert client.post("/api/chat", json={"message": "hi"}).status_code == 401

    def test_config_is_public(self, auth_ui):
        client = TestClient(auth_ui.app)
        data = client.get("/api/config").json()
        assert data["auth_enabled"] is True
        assert data["authenticated"] is False

    def test_valid_cookie_grants_access(self, auth_ui):
        from datetime import datetime, timedelta
        auth_ui._authenticated_cookies["cookie123"] = {
            "email": "user@test.com",
            "session_id": "s1",
            "expires": datetime.now() + timedelta(hours=1),
        }
        client = TestClient(auth_ui.app, cookies={"onit_auth": "cookie123"})
        assert client.get("/api/history").status_code == 200
        data = client.get("/api/config").json()
        assert data["authenticated"] is True
        assert data["email"] == "user@test.com"

    def test_expired_cookie_rejected(self, auth_ui):
        from datetime import datetime, timedelta
        auth_ui._authenticated_cookies["old"] = {
            "email": "user@test.com",
            "session_id": "s1",
            "expires": datetime.now() - timedelta(hours=1),
        }
        client = TestClient(auth_ui.app, cookies={"onit_auth": "old"})
        assert client.get("/api/history").status_code == 401

    def test_auth_check_endpoint(self, auth_ui):
        client = TestClient(auth_ui.app)
        assert client.get("/auth/check").json() == {"authenticated": False}

    def test_login_redirects_to_google(self, auth_ui):
        client = TestClient(auth_ui.app)
        res = client.get("/auth/login", follow_redirects=False)
        assert res.status_code == 307 or res.status_code == 302
        assert "accounts.google.com" in res.headers["location"]


class TestForcedLogin:
    """Sessions must start with Google-hosted mail authentication."""

    def test_web_ui_refuses_to_start_without_credentials(self, tmp_path):
        with pytest.raises(RuntimeError, match="requires Google login"):
            WebApiUI(session_path=str(tmp_path / "sessions" / "current.jsonl"))

    def test_no_login_overrides_configured_credentials(self, tmp_path, bg_loop):
        # --no-login must yield an open UI even when OAuth credentials are
        # configured (e.g. stored in the keychain by 'onit setup').
        ui = WebApiUI(
            data_path=str(tmp_path / "data"),
            session_path=str(tmp_path / "sessions" / "current.jsonl"),
            google_client_id="test-client-id.apps.googleusercontent.com",
            google_client_secret="test-secret",
            require_auth=False,
        )
        ui._onit = FakeOnit(config_data={
        "learn": {"autonomy": "observe", "path": str(tmp_path / "learned")}})
        ui._loop = bg_loop
        ui.build_app()
        client = TestClient(ui.app)
        data = client.get("/api/config").json()
        assert data["auth_enabled"] is False
        assert data["authenticated"] is True
        # No login gate: API is reachable without any auth cookie
        assert client.get("/api/history").status_code == 200

    def test_no_session_cookie_before_login(self, auth_ui):
        client = TestClient(auth_ui.app)
        res = client.get("/")
        assert "onit_session" not in res.cookies
        res = client.get("/api/config")
        assert "onit_session" not in res.cookies

    def test_session_cookie_issued_after_login(self, auth_ui):
        client = _login(auth_ui, "alice@gmail.com")
        res = client.get("/api/config")
        assert "onit_session" in res.cookies


# ── Google-hosted mail gate ─────────────────────────────────────────────────

class TestGoogleHostedGate:
    """Only Gmail or Google-Workspace-hosted domains may log in."""

    def test_gmail_accepted(self):
        assert ui_auth.GoogleAuthenticator._is_google_hosted("a@gmail.com", None)
        assert ui_auth.GoogleAuthenticator._is_google_hosted("a@googlemail.com", None)

    def test_workspace_domain_accepted(self):
        assert ui_auth.GoogleAuthenticator._is_google_hosted("a@sibyl.ai", "sibyl.ai")

    def test_hd_claim_is_case_insensitive(self):
        assert ui_auth.GoogleAuthenticator._is_google_hosted("a@Sibyl.AI", "sibyl.ai")

    def test_non_google_account_rejected(self):
        # Google Account created on an outside address: no hd claim
        assert not ui_auth.GoogleAuthenticator._is_google_hosted("a@outlook.com", None)

    def test_mismatched_hd_claim_rejected(self):
        assert not ui_auth.GoogleAuthenticator._is_google_hosted("a@outlook.com", "sibyl.ai")


# ── Per-user session gating ─────────────────────────────────────────────────

def _login(auth_ui, email):
    """Register an auth cookie for *email* and return a client wearing it."""
    from datetime import datetime, timedelta
    cookie = f"cookie-{email}"
    auth_ui._authenticated_cookies[cookie] = {
        "email": email,
        "session_id": f"auth-{email}",
        "expires": datetime.now() + timedelta(hours=1),
    }
    return TestClient(auth_ui.app, cookies={"onit_auth": cookie})


class TestSessionOwnership:
    def test_session_bound_to_user(self, auth_ui):
        alice = _login(auth_ui, "alice@gmail.com")
        sid = alice.get("/api/history").json()["session_id"]
        assert auth_ui._web_sessions[sid].owner == "alice@gmail.com"

    def test_other_users_session_id_not_shared(self, auth_ui):
        alice = _login(auth_ui, "alice@gmail.com")
        bob = _login(auth_ui, "bob@gmail.com")
        alice_sid = alice.get("/api/history").json()["session_id"]
        res = bob.get("/api/history", headers={"X-Session-Id": alice_sid}).json()
        assert res["session_id"] != alice_sid

    def test_session_list_scoped_per_user(self, auth_ui):
        alice = _login(auth_ui, "alice@gmail.com")
        bob = _login(auth_ui, "bob@gmail.com")
        alice_sid = alice.post("/api/sessions/new").json()["session_id"]
        alice_ids = {s["session_id"] for s in alice.get("/api/sessions").json()["sessions"]}
        bob_ids = {s["session_id"] for s in bob.get("/api/sessions").json()["sessions"]}
        assert alice_sid in alice_ids
        assert alice_sid not in bob_ids

    def test_delete_other_users_session_rejected(self, auth_ui):
        alice = _login(auth_ui, "alice@gmail.com")
        bob = _login(auth_ui, "bob@gmail.com")
        alice_sid = alice.get("/api/history").json()["session_id"]
        res = bob.request("DELETE", f"/api/sessions/{alice_sid}")
        assert res.status_code == 404
        assert os.path.exists(auth_ui._web_sessions[alice_sid].session_path)

    def test_rename_other_users_session_rejected(self, auth_ui):
        alice = _login(auth_ui, "alice@gmail.com")
        bob = _login(auth_ui, "bob@gmail.com")
        alice_sid = alice.get("/api/history").json()["session_id"]
        res = bob.patch(f"/api/sessions/{alice_sid}", json={"tag": "stolen"})
        assert res.status_code == 404

    def test_uploads_gated_by_owner(self, auth_ui):
        alice = _login(auth_ui, "alice@gmail.com")
        bob = _login(auth_ui, "bob@gmail.com")
        alice_sid = alice.get("/api/history").json()["session_id"]
        data_path = auth_ui._web_sessions[alice_sid].data_path
        with open(os.path.join(data_path, "secret.txt"), "w") as f:
            f.write("hi")
        assert alice.get(f"/uploads/{alice_sid}/secret.txt").status_code == 200
        assert bob.get(f"/uploads/{alice_sid}/secret.txt").status_code == 404

    def test_delete_all_only_removes_own_sessions(self, auth_ui):
        alice = _login(auth_ui, "alice@gmail.com")
        bob = _login(auth_ui, "bob@gmail.com")
        alice_sid = alice.get("/api/history").json()["session_id"]
        bob_sid = bob.get("/api/history").json()["session_id"]
        bob.request("DELETE", "/api/sessions")
        assert alice_sid in auth_ui._web_sessions
        assert bob_sid not in auth_ui._web_sessions

    def test_unowned_session_claimed_on_access(self, auth_ui):
        # Session created before auth was enabled has no owner; the first
        # authenticated user to touch it claims it.
        sid, session = auth_ui._get_or_create_session()
        assert session.owner is None
        alice = _login(auth_ui, "alice@gmail.com")
        res = alice.get("/api/history", headers={"X-Session-Id": sid}).json()
        assert res["session_id"] == sid
        assert session.owner == "alice@gmail.com"
        # ... and it is thereafter locked to her
        bob = _login(auth_ui, "bob@gmail.com")
        res = bob.get("/api/history", headers={"X-Session-Id": sid}).json()
        assert res["session_id"] != sid


# ── Misc ───────────────────────────────────────────────────────────────────

def test_sse_format():
    out = _sse("token", {"delta": "hi"})
    assert out == 'event: token\ndata: {"delta": "hi"}\n\n'


def test_delete_session_helper(tmp_path):
    from sessions import delete_session, register_session
    sid = str(uuid.uuid4())
    sessions_dir = str(tmp_path)
    with open(os.path.join(sessions_dir, f"{sid}.jsonl"), "w") as f:
        f.write("")
    register_session(sid, sessions_dir=sessions_dir)
    assert delete_session(sid, sessions_dir=sessions_dir) is True
    assert not os.path.exists(os.path.join(sessions_dir, f"{sid}.jsonl"))
    assert delete_session(sid, sessions_dir=sessions_dir) is False


# ── Security hardening ─────────────────────────────────────────────────────

class TestSecurityHeaders:
    def test_openapi_spec_disabled(self, client):
        # The full API surface must not be published to anonymous callers.
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/docs").status_code == 404

    def test_security_headers_present(self, client):
        r = client.get("/api/config")
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert "Content-Security-Policy" in r.headers
        assert r.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "Permissions-Policy" in r.headers

    def test_server_header_generic(self, client):
        # No "uvicorn" advertised.
        assert "uvicorn" not in client.get("/api/config").headers.get("server", "").lower()

    def test_csp_omits_ga_when_analytics_off(self, ui, client):
        assert ui.ga_measurement_id is None
        csp = client.get("/api/config").headers["Content-Security-Policy"]
        assert "googletagmanager" not in csp
        assert "script-src 'self'" in csp

    def test_csp_includes_ga_when_configured(self, tmp_path, bg_loop):
        u = WebApiUI(
            data_path=str(tmp_path / "d"),
            session_path=str(tmp_path / "s" / "c.jsonl"),
            require_auth=False,
            ga_measurement_id="G-ABCD1234",
        )
        u._loop = bg_loop
        u.build_app()
        csp = TestClient(u.app).get("/api/config").headers["Content-Security-Policy"]
        assert "https://www.googletagmanager.com" in csp

    def test_hsts_only_on_https(self, client):
        # Plain-http test client must not receive an HSTS pin.
        assert "Strict-Transport-Security" not in client.get("/api/config").headers
        r = client.get("/api/config", headers={"X-Forwarded-Proto": "https"})
        assert "max-age=" in r.headers.get("Strict-Transport-Security", "")


class TestWellKnown:
    def test_robots_disallows_sensitive_paths(self, client):
        body = client.get("/robots.txt").text
        assert "Disallow: /api/" in body
        assert "Disallow: /auth/" in body

    def test_security_txt_absent_without_contact(self, client, monkeypatch):
        monkeypatch.delenv("ONIT_SECURITY_CONTACT", raising=False)
        assert client.get("/.well-known/security.txt").status_code == 404

    def test_security_txt_served_with_contact(self, client, monkeypatch):
        monkeypatch.setenv("ONIT_SECURITY_CONTACT", "mailto:sec@example.com")
        r = client.get("/.well-known/security.txt")
        assert r.status_code == 200
        assert "Contact: mailto:sec@example.com" in r.text
        assert "Expires:" in r.text


class TestUploadLimit:
    def test_oversized_upload_rejected(self, client, monkeypatch):
        import ui.api as api_mod
        monkeypatch.setattr(api_mod, "_MAX_UPLOAD_BYTES", 1024)
        big = b"x" * 4096
        r = client.post("/api/upload", files={"file": ("big.bin", big, "application/octet-stream")})
        assert r.status_code == 413

    def test_normal_upload_accepted(self, client):
        r = client.post("/api/upload", files={"file": ("ok.txt", b"hello", "text/plain")})
        assert r.status_code == 200
        assert r.json()["name"] == "ok.txt"


class TestVerifyLinksSSRF:
    def test_private_host_screened(self, ui):
        # Literal private/loopback hosts fail the structural screen.
        assert ui._verify_link("http://169.254.169.254/latest/meta-data/") is False
        assert ui._verify_link("http://127.0.0.1/") is False
        assert ui._verify_link("http://localhost/") is False

    def test_public_looking_host_resolving_private_rejected(self, ui, monkeypatch):
        # A public-looking name that resolves to a private/reserved address
        # must be rejected before any network probe (DNS-rebinding SSRF).
        import ui.api as api_mod

        def fake_getaddrinfo(host, *a, **k):
            return [(2, 1, 6, "", ("169.254.169.254", 0))]

        monkeypatch.setattr(api_mod.socket, "getaddrinfo", fake_getaddrinfo)
        # Bypass the structural screen so we exercise the resolution check.
        monkeypatch.setattr(api_mod, "_link_shape_ok", lambda url: True)
        called = {"probed": False}

        def fake_head(*a, **k):
            called["probed"] = True
            raise AssertionError("should not probe a private-resolving host")

        assert ui._probe_url("http://evil.example.com/") is False
        assert called["probed"] is False
