"""Tests for src/type/tools.py — ToolRegistry, ToolHandler, RequestHandler."""

import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from type.tools import ToolRegistry, ToolHandler, RequestHandler


# ── RequestHandler (abstract) ───────────────────────────────────────────────

class TestRequestHandler:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            RequestHandler()


# ── ToolRegistry ────────────────────────────────────────────────────────────

def _make_handler(name="tool", url="http://localhost:9000/x"):
    item = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
            "returns": {},
        },
    }
    return ToolHandler(url=url, tool_item=item)


class TestToolRegistry:
    def test_register_and_len(self):
        reg = ToolRegistry()
        reg.register(_make_handler("a", "http://h1"))
        assert len(reg) == 1
        assert "a" in reg.tools

    def test_register_multiple_urls_for_same_tool(self):
        reg = ToolRegistry()
        reg.register(_make_handler("a", "http://h1"))
        reg.register(_make_handler("a", "http://h2"))
        # Still one unique tool name
        assert len(reg) == 1
        assert len(reg.urls["a"]) == 2
        # Two handler entries
        assert len(reg.handlers) == 2

    def test_get_url_returns_url(self):
        reg = ToolRegistry()
        reg.register(_make_handler("x", "http://only"))
        assert reg.get_url("x") == "http://only"

    def test_get_url_unknown_returns_none(self):
        reg = ToolRegistry()
        assert reg.get_url("nope") is None

    def test_get_tool_items(self):
        reg = ToolRegistry()
        reg.register(_make_handler("a"))
        reg.register(_make_handler("b"))
        items = reg.get_tool_items()
        assert len(items) == 2
        names = {i["function"]["name"] for i in items}
        assert names == {"a", "b"}

    def test_get_handler_by(self):
        reg = ToolRegistry()
        h = _make_handler("t", "http://u")
        reg.register(h)
        assert reg.get_handler_by("t", "http://u") is h
        assert reg.get_handler_by("t", "http://other") is None

    def test_getitem_returns_handler(self):
        reg = ToolRegistry()
        h = _make_handler("t", "http://u")
        reg.register(h)
        assert reg["t"] is h

    def test_getitem_unknown_returns_none(self):
        reg = ToolRegistry()
        assert reg["missing"] is None

    def test_iter(self):
        reg = ToolRegistry()
        reg.register(_make_handler("x"))
        reg.register(_make_handler("y"))
        assert set(reg) == {"x", "y"}


# ── ToolHandler ─────────────────────────────────────────────────────────────

class TestToolHandler:
    def test_get_tool(self):
        h = _make_handler("t")
        item = h.get_tool()
        assert item["function"]["name"] == "t"

    @pytest.mark.asyncio
    async def test_call_text_content(self):
        """Mocked MCP client returns TextContent."""
        from mcp.types import TextContent

        mock_response = MagicMock()
        mock_response.content = [TextContent(type="text", text="hello world")]

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        h = _make_handler("t")
        with patch("type.tools.Client", return_value=mock_client):
            result = await h(query="test")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_call_image_content(self):
        """Mocked MCP client returns ImageContent — should save a temp PNG."""
        import base64
        from mcp.types import ImageContent

        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        b64 = base64.b64encode(fake_png).decode()

        mock_response = MagicMock()
        mock_response.content = [ImageContent(type="image", data=b64, mimeType="image/png")]

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        h = _make_handler("t")
        with patch("type.tools.Client", return_value=mock_client):
            result = await h(query="img")

        assert result.endswith(".png")
        assert os.path.isfile(result)
        os.unlink(result)

    @pytest.mark.asyncio
    async def test_empty_images_list(self):
        h = _make_handler("t")
        result = await h(images=[])
        assert "No images provided" in result

    @pytest.mark.asyncio
    async def test_empty_audios_list(self):
        h = _make_handler("t")
        result = await h(audios=[])
        assert "No audios provided" in result

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        h = _make_handler("t")
        result = await h(images="/nonexistent/path.png")
        assert "File not found" in result
