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
    async def test_basic_instruction(self):
        result = await _assistant_fn(task="What is 2+2?", session_id="test-session")
        assert "What is 2+2?" in result
        assert "test-session" in result

    @pytest.mark.asyncio
    async def test_generates_session_id_if_none(self):
        result = await _assistant_fn(task="test task")
        assert "test task" in result
        assert "onit" in result

    @pytest.mark.asyncio
    async def test_includes_data_path(self):
        result = await _assistant_fn(task="test", session_id="sid")
        assert "data" in result

    @pytest.mark.asyncio
    async def test_custom_template(self, tmp_path):
        template_content = {
            "instruction_template": "Custom: {task} in {data_path} for {session_id}"
        }
        template_file = tmp_path / "custom.yaml"
        template_file.write_text(yaml.dump(template_content))

        result = await _assistant_fn(
            task="my task",
            session_id="s1",
            template_path=str(template_file),
        )
        assert "Custom: my task" in result

    @pytest.mark.asyncio
    async def test_invalid_template_uses_default(self, tmp_path):
        template_file = tmp_path / "empty.yaml"
        template_file.write_text(yaml.dump({"other_key": "value"}))

        result = await _assistant_fn(
            task="fallback test",
            session_id="s2",
            template_path=str(template_file),
        )
        assert "fallback test" in result
        assert "step by step" in result

    @pytest.mark.asyncio
    async def test_nonexistent_template_uses_default(self):
        result = await _assistant_fn(
            task="no template",
            session_id="s3",
            template_path="/nonexistent/template.yaml",
        )
        assert "no template" in result
        assert "step by step" in result

    @pytest.mark.asyncio
    async def test_file_server_url_appended(self):
        result = await _assistant_fn(
            task="create report",
            session_id="s4",
            file_server_url="http://192.168.1.100:9000",
        )
        assert "http://192.168.1.100:9000" in result
        assert "uploads" in result
        assert "callback_url" in result

    @pytest.mark.asyncio
    async def test_no_file_server_url(self):
        result = await _assistant_fn(
            task="simple task",
            session_id="s5",
            file_server_url=None,
        )
        assert "uploads" not in result
