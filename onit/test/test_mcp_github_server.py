"""Tests for the GitHub MCP server.

The server had no tests at all (18% coverage).  Every function here is either
pure or talks to the GitHub REST API through ``requests``, so the whole tool
can be exercised without a token and without a network by patching
``requests.<verb>``.

The behaviour worth pinning is the shape of the answer, not the HTTP call:
a model reads these JSON strings, so a missing ``status`` or an unredacted
error is a user-visible defect even when the request itself succeeded.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import onit.mcp.servers.tasks.github.mcp_server as gh


# ── helpers ─────────────────────────────────────────────────────────────────


def _resp(status_code=200, payload=None, text=""):
    """A stand-in for requests.Response with the bits the server reads."""
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    r.json.return_value = payload if payload is not None else {}
    return r


@pytest.fixture(autouse=True)
def _no_real_token(request, monkeypatch):
    """No test may reach the network through a developer's real token.

    Skipped for ``TestGetToken``, which is the one place that exercises the
    real resolver -- patching it there would make those tests assert against
    their own stub.
    """
    if request.cls is TestGetToken:
        return
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(gh, "_get_token", lambda: "test-token")


# ── _get_token ──────────────────────────────────────────────────────────────


class TestGetToken:
    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "from-env")
        assert gh._get_token() == "from-env"

    def test_falls_back_to_keyring(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("onit.setup.get_secret", return_value="from-keyring"):
            assert gh._get_token() == "from-keyring"

    def test_returns_none_when_nothing_is_configured(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("onit.setup.get_secret", side_effect=RuntimeError("no keyring")):
            assert gh._get_token() is None

    def test_empty_env_var_is_not_a_token(self, monkeypatch):
        """An exported-but-empty GITHUB_TOKEN must not shadow the keyring."""
        monkeypatch.setenv("GITHUB_TOKEN", "")
        with patch("onit.setup.get_secret", return_value="from-keyring"):
            assert gh._get_token() == "from-keyring"


# ── _auth_headers / _repo_summary ───────────────────────────────────────────


class TestAuthHeaders:
    def test_bearer_and_api_version(self):
        h = gh._auth_headers("abc123")
        assert h["Authorization"] == "Bearer abc123"
        assert h["Accept"] == "application/vnd.github+json"
        assert h["X-GitHub-Api-Version"] == "2022-11-28"


class TestRepoSummary:
    def test_maps_github_fields(self):
        repo = {
            "name": "onit",
            "full_name": "sibyl/onit",
            "html_url": "https://github.com/sibyl/onit",
            "clone_url": "https://github.com/sibyl/onit.git",
            "ssh_url": "git@github.com:sibyl/onit.git",
            "description": "an agent",
            "private": True,
            "default_branch": "main",
            "created_at": "2025-01-01T00:00:00Z",
            "owner": {"login": "sibyl"},
        }
        s = gh._repo_summary(repo)
        assert s["name"] == "onit"
        assert s["full_name"] == "sibyl/onit"
        assert s["url"] == "https://github.com/sibyl/onit"
        assert s["owner"] == "sibyl"
        assert s["private"] is True

    def test_tolerates_a_missing_owner(self):
        """``repo["owner"]`` is absent on some error payloads; must not raise."""
        s = gh._repo_summary({"name": "x"})
        assert s["owner"] is None
        assert s["name"] == "x"

    def test_missing_fields_become_none(self):
        s = gh._repo_summary({})
        assert all(v is None for v in s.values())


# ── argument validation ─────────────────────────────────────────────────────


class TestValidation:
    def test_missing_action_is_refused(self):
        out = json.loads(gh.github_repo())
        assert out["status"] == "error"
        assert "action" in out["error"]

    def test_missing_name_on_create_is_refused(self):
        out = json.loads(gh.github_repo(action="create"))
        assert out["status"] == "error"
        assert "name" in out["error"]

    def test_no_token_is_a_clear_error(self, monkeypatch):
        monkeypatch.setattr(gh, "_get_token", lambda: None)
        out = json.loads(gh.github_repo(action="list"))
        assert out["status"] == "error"
        assert "GITHUB_TOKEN" in out["error"]


# ── create ──────────────────────────────────────────────────────────────────


class TestCreate:
    def test_create_success(self):
        with patch.object(gh.requests, "post", return_value=_resp(201, {"name": "new"})) as p:
            out = json.loads(gh.github_repo(action="create", name="new"))
        assert out["status"] == "created"
        assert out["repo"]["name"] == "new"
        assert p.call_args[0][0] == f"{gh.GITHUB_API}/user/repos"

    def test_create_in_org_uses_orgs_endpoint(self):
        with patch.object(gh.requests, "post", return_value=_resp(201, {"name": "n"})) as p:
            gh.github_repo(action="create", name="n", org="sibyl")
        assert p.call_args[0][0] == f"{gh.GITHUB_API}/orgs/sibyl/repos"

    def test_optional_fields_are_omitted_when_unset(self):
        with patch.object(gh.requests, "post", return_value=_resp(201, {})) as p:
            gh.github_repo(action="create", name="n")
        payload = p.call_args[1]["json"]
        assert "description" not in payload
        assert "gitignore_template" not in payload
        assert "license_template" not in payload

    def test_optional_fields_are_sent_when_set(self):
        with patch.object(gh.requests, "post", return_value=_resp(201, {})) as p:
            gh.github_repo(
                action="create",
                name="n",
                description="d",
                gitignore_template="Python",
                license_template="mit",
            )
        payload = p.call_args[1]["json"]
        assert payload["description"] == "d"
        assert payload["gitignore_template"] == "Python"
        assert payload["license_template"] == "mit"

    def test_create_failure_reports_github_message(self):
        with patch.object(gh.requests, "post", return_value=_resp(422, {"message": "name taken"})):
            out = json.loads(gh.github_repo(action="create", name="dup"))
        assert out["status"] == "error"
        assert out["error"] == "name taken"


# ── get / list ──────────────────────────────────────────────────────────────


class TestGet:
    def test_get_success(self):
        with patch.object(gh.requests, "get", return_value=_resp(200, {"name": "onit"})):
            out = json.loads(gh.github_repo(action="get", name="sibyl/onit"))
        assert out["repo"]["name"] == "onit"

    def test_get_requires_name(self):
        out = json.loads(gh.github_repo(action="get"))
        assert out["status"] == "error"

    def test_get_404_is_an_error(self):
        with patch.object(gh.requests, "get", return_value=_resp(404, {"message": "Not Found"})):
            out = json.loads(gh.github_repo(action="get", name="sibyl/nope"))
        assert out["status"] == "error"
        assert out["error"] == "Not Found"


class TestList:
    def test_list_returns_summaries(self):
        payload = [{"name": "a", "full_name": "o/a"}, {"name": "b", "full_name": "o/b"}]
        with patch.object(gh.requests, "get", return_value=_resp(200, payload)):
            out = json.loads(gh.github_repo(action="list"))
        assert [r["name"] for r in out["repos"]] == ["a", "b"]

    def test_list_for_org_uses_orgs_endpoint(self):
        with patch.object(gh.requests, "get", return_value=_resp(200, [])) as g:
            gh.github_repo(action="list", org="sibyl")
        assert g.call_args[0][0] == f"{gh.GITHUB_API}/orgs/sibyl/repos"

    def test_list_passes_per_page(self):
        with patch.object(gh.requests, "get", return_value=_resp(200, [])) as g:
            gh.github_repo(action="list", per_page=7)
        assert g.call_args[1]["params"]["per_page"] == 7


# ── fork / delete ───────────────────────────────────────────────────────────


class TestFork:
    def test_fork_success(self):
        with patch.object(gh.requests, "post", return_value=_resp(202, {"name": "onit"})):
            out = json.loads(gh.github_repo(action="fork", name="sibyl/onit"))
        assert out["repo"]["name"] == "onit"

    def test_fork_requires_name(self):
        out = json.loads(gh.github_repo(action="fork"))
        assert out["status"] == "error"


class TestDelete:
    def test_delete_success(self):
        with patch.object(gh.requests, "delete", return_value=_resp(204)):
            out = json.loads(gh.github_repo(action="delete", name="sibyl/onit"))
        assert out["status"] == "deleted"

    def test_delete_requires_name(self):
        out = json.loads(gh.github_repo(action="delete"))
        assert out["status"] == "error"

    def test_delete_failure_is_reported(self):
        with patch.object(gh.requests, "delete", return_value=_resp(403, {"message": "forbidden"})):
            out = json.loads(gh.github_repo(action="delete", name="sibyl/onit"))
        assert out["status"] == "error"
        assert out["error"] == "forbidden"


# ── unknown action / network failure ────────────────────────────────────────


class TestFailureModes:
    def test_unknown_action_is_refused(self):
        out = json.loads(gh.github_repo(action="explode", name="x"))
        assert out["status"] == "error"
        assert "explode" in out["error"]

    def test_network_error_is_caught(self):
        """A connection failure must return JSON, not raise into the MCP layer."""
        with patch.object(
            gh.requests, "get", side_effect=gh.requests.exceptions.ConnectionError("boom")
        ):
            out = json.loads(gh.github_repo(action="list"))
        assert out["status"] == "error"

    def test_token_is_never_echoed_in_a_request_error(self):
        """The error string reaches the model; the bearer token must not.

        ``requests`` embeds the request URL in its exception text, and the
        server returns ``str(e)`` verbatim for a ``RequestException``.  The
        token travels in a header, not the URL, so it stays out -- this pins
        that it is not added to the message anywhere along the way.
        """
        exc = gh.requests.exceptions.ConnectionError(
            "Failed to connect to https://api.github.com/user/repos"
        )
        with patch.object(gh.requests, "get", side_effect=exc):
            out = json.loads(gh.github_repo(action="list"))
        assert out["status"] == "error"
        assert "test-token" not in json.dumps(out)

    def test_non_request_exception_is_not_swallowed(self):
        """Only ``RequestException`` is caught; a programming error must surface.

        A bare ``except Exception`` here would turn a bug in the payload
        handling into a plausible-looking "error" string that nobody debugs.
        """
        with patch.object(gh.requests, "get", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                gh.github_repo(action="list")


# ── remaining error branches ────────────────────────────────────────────────


class TestErrorBranches:
    def test_list_failure_reports_http_status(self):
        with patch.object(
            gh.requests, "get", return_value=_resp(401, {"message": "Bad credentials"})
        ):
            out = json.loads(gh.github_repo(action="list"))
        assert out["status"] == "error"
        assert out["error"] == "Bad credentials"
        assert out["http_status"] == 401

    def test_fork_failure_is_reported(self):
        with patch.object(
            gh.requests, "post", return_value=_resp(422, {"message": "already forked"})
        ):
            out = json.loads(gh.github_repo(action="fork", name="sibyl/onit"))
        assert out["status"] == "error"
        assert out["error"] == "already forked"

    def test_fork_into_org_sends_organization(self):
        with patch.object(gh.requests, "post", return_value=_resp(202, {"name": "onit"})) as p:
            gh.github_repo(action="fork", name="sibyl/onit", org="other")
        assert p.call_args[1]["json"]["organization"] == "other"

    def test_delete_with_empty_body_does_not_crash(self):
        """A 403 with no body must not raise on ``resp.json()``."""
        r = _resp(403, {})
        r.content = b""
        r.text = ""
        with patch.object(gh.requests, "delete", return_value=r):
            out = json.loads(gh.github_repo(action="delete", name="sibyl/onit"))
        assert out["status"] == "error"


# ── run() ───────────────────────────────────────────────────────────────────


class TestRun:
    def test_stdio_uses_no_banner(self):
        with patch.object(gh.mcp, "run") as r:
            gh.run(transport="stdio")
        r.assert_called_once_with(transport="stdio", show_banner=False)

    def test_sse_passes_host_port_path(self):
        with patch.object(gh.mcp, "run") as r:
            gh.run(transport="sse", host="127.0.0.1", port=9999, path="/x")
        kw = r.call_args[1]
        assert kw["host"] == "127.0.0.1"
        assert kw["port"] == 9999
        assert kw["path"] == "/x"
        assert "uvicorn_config" in kw

    def test_verbose_keeps_info_logging(self):
        import logging

        with patch.object(gh.mcp, "run"):
            gh.run(transport="stdio", options={"verbose": True})
        assert gh.logger.level == logging.INFO

    def test_quiet_drops_to_error(self):
        import logging

        with patch.object(gh.mcp, "run"):
            gh.run(transport="stdio")
        assert gh.logger.level == logging.ERROR
