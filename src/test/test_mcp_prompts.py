"""Tests for src/mcp/prompts/prompts.py — assistant_instruction."""

import os
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Import the module and get the assistant_instruction function.
# Depending on the fastmcp version, the @prompt() decorator may return
# the original function directly or wrap it in a FunctionPrompt with .fn.
import src.mcp.prompts.prompts as prompts_mod

_decorated = prompts_mod.assistant_instruction
_assistant_fn = getattr(_decorated, "fn", _decorated)


class TestAssistantInstruction:
    @pytest.mark.asyncio
    async def test_basic_instruction(self, tmp_path):
        dp = str(tmp_path / "data")
        result = await _assistant_fn(task="What is 2+2?", data_path=dp)
        assert "What is 2+2?" in result
        assert dp in result

    @pytest.mark.asyncio
    async def test_raises_if_data_path_none(self):
        with pytest.raises(ValueError, match="data_path is required"):
            await _assistant_fn(task="test task")

    @pytest.mark.asyncio
    async def test_raises_if_data_path_empty(self):
        with pytest.raises(ValueError, match="data_path is required"):
            await _assistant_fn(task="test task", data_path="")

    @pytest.mark.asyncio
    async def test_includes_data_path(self, tmp_path):
        dp = str(tmp_path / "data")
        result = await _assistant_fn(task="test", data_path=dp)
        assert dp in result

    @pytest.mark.asyncio
    async def test_custom_template(self, tmp_path):
        template_content = {
            "instruction_template": "Custom: {task} in {data_path}"
        }
        template_file = tmp_path / "custom.yaml"
        template_file.write_text(yaml.dump(template_content))

        dp = str(tmp_path / "data")
        result = await _assistant_fn(
            task="my task",
            data_path=dp,
            template_path=str(template_file),
        )
        assert "Custom: my task" in result

    @pytest.mark.asyncio
    async def test_invalid_template_uses_default(self, tmp_path):
        template_file = tmp_path / "empty.yaml"
        template_file.write_text(yaml.dump({"other_key": "value"}))

        dp = str(tmp_path / "data")
        result = await _assistant_fn(
            task="fallback test",
            data_path=dp,
            template_path=str(template_file),
        )
        assert "fallback test" in result
        assert "step by step" in result

    @pytest.mark.asyncio
    async def test_nonexistent_template_uses_default(self, tmp_path):
        dp = str(tmp_path / "data")
        result = await _assistant_fn(
            task="no template",
            data_path=dp,
            template_path="/nonexistent/template.yaml",
        )
        assert "no template" in result
        assert "step by step" in result

    @pytest.mark.asyncio
    async def test_file_server_url_appended(self, tmp_path):
        dp = str(tmp_path / "data")
        result = await _assistant_fn(
            task="create report",
            data_path=dp,
            file_server_url="http://192.168.1.100:9000",
        )
        # Upload URLs should be session-scoped using last path component of data_path
        assert "http://192.168.1.100:9000/uploads/data/" in result
        assert "callback_url" in result

    @pytest.mark.asyncio
    async def test_file_server_url_session_scoped(self, tmp_path):
        """Verify upload URLs use session-specific path derived from data_path."""
        dp = str(tmp_path / "abc-123")
        result = await _assistant_fn(
            task="test",
            data_path=dp,
            file_server_url="http://host:9000",
        )
        assert "http://host:9000/uploads/abc-123/" in result

    @pytest.mark.asyncio
    async def test_no_file_server_url(self, tmp_path):
        dp = str(tmp_path / "data")
        result = await _assistant_fn(
            task="simple task",
            data_path=dp,
            file_server_url=None,
        )
        assert "uploads" not in result
