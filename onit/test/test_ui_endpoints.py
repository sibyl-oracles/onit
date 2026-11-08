"""The endpoint editor must not write the running config back to disk."""

import yaml

from onit import setup as setup_mod
from onit.ui import endpoints


def test_save_endpoints_writes_only_endpoints_to_setup_file(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"theme": "dark", "serving": {"host": "http://old:8000/v1"}}))
    monkeypatch.setattr(setup_mod, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(setup_mod, "CONFIG_PATH", str(path))
    monkeypatch.setattr(setup_mod, "get_secret", lambda k: None)
    monkeypatch.setattr(setup_mod, "store_secret", lambda k, v: None)
    monkeypatch.setattr(setup_mod, "delete_secret", lambda k: None)

    # What a `onit serve web --no-login` run holds in memory.
    live = {
        "web": True,
        "web_port": 56181,
        "web_require_auth": False,
        "web_google_client_secret": "GOCSPX-secret",
        "mcp": {"servers": [{"name": "ToolsNetMCPServer"}]},
        "serving": {"host": "http://old:8000/v1"},
    }
    endpoints.save_endpoints(live, [{"host": "https://ollama.com", "model": "m", "priority": 1}])

    saved = yaml.safe_load(path.read_text())
    assert saved["theme"] == "dark"
    for key in ("web", "web_port", "web_require_auth", "web_google_client_secret", "mcp"):
        assert key not in saved
    assert saved["serving"]["endpoints"][0]["host"] == "https://ollama.com"
    # The running harness sees the new list too.
    assert live["serving"]["endpoints"][0]["host"] == "https://ollama.com"
