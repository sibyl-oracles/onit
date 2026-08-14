"""Tests for src/setup.py — setup wizard settings and provider sanity notes."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import setup as setup_mod
from src.setup import SETTINGS, _provider_notes


def _patch_secrets(monkeypatch, present: set[str]):
    """Make get_secret return a value only for the given keyring keys."""
    monkeypatch.setattr(
        setup_mod, "get_secret",
        lambda key: "dummy" if key in present else None)


class TestSettingsSchema:
    def test_model_settings_present(self):
        paths = [dotpath for dotpath, _, _ in SETTINGS]
        for expected in ("serving.host", "serving.model",
                         "serving.host2", "serving.model2",
                         "serving.load_balancer"):
            assert expected in paths

    def test_model_settings_are_optional(self):
        defaults = {dotpath: default for dotpath, _, default in SETTINGS}
        assert defaults["serving.model"] == ""
        assert defaults["serving.model2"] == ""
        assert defaults["serving.host2"] == ""


class TestProviderNotes:
    def test_vllm_host_no_notes(self, monkeypatch):
        _patch_secrets(monkeypatch, set())
        config = {"serving": {"host": "http://localhost:8000/v1"}}
        assert _provider_notes(config) == []

    def test_ollama_host_missing_key_and_model(self, monkeypatch):
        _patch_secrets(monkeypatch, set())
        config = {"serving": {"host": "https://ollama.com"}}
        notes = _provider_notes(config)
        assert any("Ollama API key" in n for n in notes)
        assert any("first model available" in n for n in notes)

    def test_ollama_host_with_key_and_model_is_clean(self, monkeypatch):
        _patch_secrets(monkeypatch, {"ollama_api_key"})
        config = {"serving": {"host": "https://ollama.com",
                              "model": "glm-5.1:cloud"}}
        assert _provider_notes(config) == []

    def test_openrouter_missing_key_and_model(self, monkeypatch):
        _patch_secrets(monkeypatch, set())
        config = {"serving": {"host": "https://openrouter.ai/api/v1"}}
        notes = _provider_notes(config)
        assert any("OpenRouter endpoint but no" in n for n in notes)
        assert any("explicit model name" in n for n in notes)

    def test_openrouter_host2_falls_back_to_host_key(self, monkeypatch):
        _patch_secrets(monkeypatch, {"host_key"})
        config = {"serving": {"host": "http://localhost:8000/v1",
                              "host2": "https://openrouter.ai/api/v1",
                              "model2": "google/gemini-2.5-pro"}}
        assert _provider_notes(config) == []

    def test_ollama_host2_checked_too(self, monkeypatch):
        _patch_secrets(monkeypatch, {"ollama_api_key"})
        config = {"serving": {"host": "http://localhost:8000/v1",
                              "host2": "https://ollama.com"}}
        notes = _provider_notes(config)
        assert len(notes) == 1
        assert "serving.model2" in notes[0]

    def test_empty_config_no_notes(self, monkeypatch):
        _patch_secrets(monkeypatch, set())
        assert _provider_notes({}) == []

    def test_endpoints_list_entries_checked(self, monkeypatch):
        _patch_secrets(monkeypatch, set())
        config = {"serving": {"endpoints": [
            {"name": "gpu-a", "host": "http://gpu-a:8000/v1", "priority": 1},
            {"name": "cloud", "host": "https://ollama.com", "priority": 2},
        ]}}
        notes = _provider_notes(config)
        assert any("Ollama API key" in n and "cloud" in n for n in notes)
        assert not any("gpu-a" in n for n in notes)

    def test_endpoints_list_supersedes_host_pair(self, monkeypatch):
        """A stale serving.host is ignored once an endpoints list exists."""
        _patch_secrets(monkeypatch, set())
        config = {"serving": {"host": "https://openrouter.ai/api/v1",
                              "endpoints": [{"host": "http://gpu:8000/v1"}]}}
        assert _provider_notes(config) == []


class TestEndpointList:
    def test_returns_empty_when_unset(self):
        assert setup_mod._endpoint_list({}) == []
        assert setup_mod._endpoint_list({"serving": {"endpoints": "nope"}}) == []

    def test_url_strings_normalized_to_dicts(self):
        got = setup_mod._endpoint_list(
            {"serving": {"endpoints": ["http://gpu:8000/v1"]}})
        assert got == [{"host": "http://gpu:8000/v1"}]

    def test_entries_without_host_dropped(self):
        got = setup_mod._endpoint_list(
            {"serving": {"endpoints": [{"model": "x"}, 42,
                                       {"host": "http://gpu:8000/v1"}]}})
        assert [e["host"] for e in got] == ["http://gpu:8000/v1"]

    def test_priority_defaults_to_zero(self):
        assert setup_mod._entry_priority({}) == 0
        assert setup_mod._entry_priority({"priority": "high"}) == 0
        assert setup_mod._entry_priority({"priority": "2"}) == 2

    def test_host_settings_are_the_ones_superseded(self):
        assert setup_mod._HOST_SETTINGS <= {p for p, _, _ in SETTINGS}


class TestSectionGrouping:
    """Model-serving config is asked for together, apart from integrations."""

    def test_sections_partition_the_secrets(self):
        assert (setup_mod.SERVING_SECRETS + setup_mod.INTEGRATION_SECRETS
                == setup_mod.SECRETS)

    def test_model_serving_keys_are_in_the_serving_section(self):
        keys = {k for k, _, _ in setup_mod.SERVING_SECRETS}
        assert keys == {"host_key", "ollama_api_key", "vllm_api_key",
                        "host2_key"}

    def test_unrelated_keys_are_not_in_the_serving_section(self):
        keys = {k for k, _, _ in setup_mod.SERVING_SECRETS}
        for unrelated in ("openweathermap_api_key", "telegram_bot_token",
                          "github_token", "huggingface_token"):
            assert unrelated not in keys

    def test_sections_partition_the_settings(self):
        assert (setup_mod.HOST_SETTINGS + setup_mod.SERVING_SETTINGS
                + setup_mod.GENERAL_SETTINGS == SETTINGS)

    def test_host_settings_are_not_prompted_individually(self):
        """The endpoint editor owns them, so no section prompts for them."""
        prompted = {p for p, _, _ in
                    setup_mod.SERVING_SETTINGS + setup_mod.GENERAL_SETTINGS}
        assert prompted.isdisjoint(setup_mod._HOST_SETTINGS)


def _drive_editor(config, script):
    """Run _edit_endpoints against a scripted stdin, return the new config."""
    import builtins
    it = iter(script)
    original = builtins.input
    builtins.input = lambda prompt="": next(it, "")
    try:
        setup_mod._edit_endpoints(config)
    finally:
        builtins.input = original
    return config


class TestEndpointEditor:
    def test_adds_an_endpoint_with_priority(self, capsys):
        cfg = {"serving": {"host": "http://a:8000/v1"}}
        _drive_editor(cfg, ["a", "https://ollama.com", "glm-5.1:cloud",
                            "ollama", "2", ""])
        assert cfg["serving"]["endpoints"] == [
            {"host": "http://a:8000/v1", "priority": 0},
            {"name": "ollama", "host": "https://ollama.com",
             "model": "glm-5.1:cloud", "priority": 2},
        ]
        # Promoting to a list must retire the settings it supersedes.
        assert "host" not in cfg["serving"]

    def test_reads_the_legacy_host_pair(self, capsys):
        cfg = {"serving": {"host": "http://a:8000/v1",
                           "host2": "http://b:8000/v1", "model2": "qwen3"}}
        _drive_editor(cfg, [""])
        out = capsys.readouterr().out
        assert "http://a:8000/v1" in out and "qwen3" in out

    def test_untouched_legacy_config_keeps_its_shape(self):
        cfg = {"serving": {"host": "http://a:8000/v1",
                           "host2": "http://b:8000/v1"}}
        _drive_editor(cfg, [""])
        assert cfg["serving"]["host"] == "http://a:8000/v1"
        assert cfg["serving"]["host2"] == "http://b:8000/v1"
        assert "endpoints" not in cfg["serving"]

    def test_priority_promotes_to_the_list_shape(self):
        cfg = {"serving": {"host": "http://a:8000/v1",
                           "host2": "http://b:8000/v1"}}
        _drive_editor(cfg, ["p 2", "1", ""])
        assert cfg["serving"]["endpoints"] == [
            {"host": "http://a:8000/v1", "priority": 0},
            {"host": "http://b:8000/v1", "priority": 1},
        ]
        assert "host2" not in cfg["serving"]

    def test_delete_drops_the_second_host_and_its_model(self):
        cfg = {"serving": {"host": "http://a:8000/v1",
                           "host2": "http://b:8000/v1", "model2": "qwen3"}}
        _drive_editor(cfg, ["d 2", ""])
        assert cfg["serving"]["host"] == "http://a:8000/v1"
        for gone in ("host2", "model2"):
            assert gone not in cfg["serving"]

    def test_last_endpoint_cannot_be_deleted(self, capsys):
        cfg = {"serving": {"host": "http://a:8000/v1"}}
        _drive_editor(cfg, ["d 1", ""])
        assert "At least one endpoint" in capsys.readouterr().out
        assert cfg["serving"]["host"] == "http://a:8000/v1"

    def test_edit_changes_the_addressed_row(self):
        cfg = {"serving": {"endpoints": [
            {"host": "http://a:8000/v1", "priority": 1},
            {"host": "http://b:8000/v1", "priority": 2}]}}
        _drive_editor(cfg, ["e 2", "http://c:8000/v1", "", "", "", ""])
        assert [e["host"] for e in cfg["serving"]["endpoints"]] == [
            "http://a:8000/v1", "http://c:8000/v1"]

    def test_row_numbers_are_stable_across_reordering(self, capsys):
        """Row N addresses the same endpoint before and after a re-rank."""
        cfg = {"serving": {"endpoints": [
            {"host": "http://a:8000/v1", "priority": 0},
            {"host": "http://b:8000/v1", "priority": 0}]}}
        # Rank row 1 behind row 2, then edit row 1 — it must still be host a.
        _drive_editor(cfg, ["p 1", "5", "e 1", "http://a2:8000/v1",
                            "", "", "", ""])
        by_host = {e["host"]: e for e in cfg["serving"]["endpoints"]}
        assert by_host["http://a2:8000/v1"]["priority"] == 5

    def test_duplicate_host_rejected(self, capsys):
        cfg = {"serving": {"host": "http://a:8000/v1"}}
        _drive_editor(cfg, ["a", "http://a:8000/v1", "", "", "", ""])
        assert "already configured" in capsys.readouterr().out
        assert "endpoints" not in cfg["serving"]

    def test_add_with_no_url_is_cancelled(self, capsys):
        cfg = {"serving": {"host": "http://a:8000/v1"}}
        _drive_editor(cfg, ["a", "", "", "", "", ""])
        assert "nothing added" in capsys.readouterr().out
        assert cfg["serving"]["host"] == "http://a:8000/v1"

    def test_first_endpoint_offered_on_empty_config(self):
        cfg = {}
        _drive_editor(cfg, ["", "", "", "", ""])
        assert cfg["serving"]["host"] == setup_mod.DEFAULT_HOST

    def test_bad_row_number_is_reported_not_raised(self, capsys):
        cfg = {"serving": {"host": "http://a:8000/v1"}}
        _drive_editor(cfg, ["p 9", "e abc", ""])
        out = capsys.readouterr().out
        assert "No row 9" in out
        assert "not a row number" in out

    def test_unknown_command_is_reported(self, capsys):
        cfg = {"serving": {"host": "http://a:8000/v1"}}
        _drive_editor(cfg, ["frobnicate", ""])
        assert "Unknown command" in capsys.readouterr().out

    def test_non_numeric_priority_keeps_current(self, capsys):
        cfg = {"serving": {"endpoints": [
            {"host": "http://a:8000/v1", "priority": 3}]}}
        _drive_editor(cfg, ["p 1", "high", ""])
        assert cfg["serving"]["endpoints"][0]["priority"] == 3
