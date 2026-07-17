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
