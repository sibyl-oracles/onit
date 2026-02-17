"""Shared fixtures for OnIt test suite."""

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Minimal valid config dict (avoids real MCP / network calls)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config_dict(tmp_path):
    """Minimal config dict that satisfies OnIt.__init__ validation."""
    return {
        "serving": {
            "host": "http://localhost:8000/v1",
            "model": "test-model",
            "think": False,
            "max_tokens": 1024,
        },
        "mcp": {
            "servers": [
                {
                    "name": "PromptsMCPServer",
                    "url": "http://127.0.0.1:9002/prompts",
                    "enabled": True,
                },
                {
                    "name": "TestServer",
                    "url": "http://127.0.0.1:9003/test",
                    "enabled": True,
                },
            ],
        },
        "session_path": str(tmp_path / "sessions"),
        "theme": "white",
        "persona": "assistant",
        "verbose": False,
    }


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def make_tool_item():
    """Factory to build a tool_item dict."""
    def _make(name="test_tool", description="A test tool", url="http://127.0.0.1:9003/test"):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                "returns": {},
            },
        }
    return _make


@pytest.fixture
def mock_tool_registry(make_tool_item):
    """Pre-populated ToolRegistry with two fake tools."""
    from type.tools import ToolRegistry, ToolHandler

    registry = ToolRegistry()
    for name, url in [("search", "http://127.0.0.1:9003/search"),
                      ("bash", "http://127.0.0.1:9004/bash")]:
        handler = ToolHandler(url=url, tool_item=make_tool_item(name=name, url=url))
        registry.register(handler)
    return registry


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def safety_queue():
    return asyncio.Queue(maxsize=10)


# ---------------------------------------------------------------------------
# Temp session file
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_session_file(tmp_path):
    """Create a temp JSONL session file with sample entries."""
    path = tmp_path / "session.jsonl"
    entries = [
        {"task": "hello", "response": "Hi there!", "timestamp": 1.0},
        {"task": "weather", "response": "Sunny today.", "timestamp": 2.0},
    ]
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return str(path)
