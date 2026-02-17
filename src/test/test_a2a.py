"""Tests for A2A server protocol — agent card, JSON-RPC message/send, SDK client."""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ── Agent card ────────────────────────────────────────────────────────────────

class TestAgentCard:
    @pytest.mark.asyncio
    async def test_fetch_agent_card_success(self):
        card_data = {
            "name": "OnIt",
            "version": "1.0.0",
            "description": "AI assistant",
            "url": "http://localhost:5001",
            "capabilities": {},
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = card_data

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:5001/.well-known/agent.json")

        assert resp.status_code == 200
        card = resp.json()
        assert card["name"] == "OnIt"
        assert "version" in card

    @pytest.mark.asyncio
    async def test_fetch_agent_card_not_found(self):
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:5001/.well-known/agent.json")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_agent_card_url_construction(self):
        """Verify the agent card URL is properly formed."""
        base_urls = [
            "http://localhost:5001",
            "http://localhost:5001/",
        ]
        for base in base_urls:
            card_url = f"{base.rstrip('/')}/.well-known/agent.json"
            assert card_url == "http://localhost:5001/.well-known/agent.json"


# ── JSON-RPC message/send ─────────────────────────────────────────────────────

class TestSendTask:
    def _make_payload(self, task="What is 2 + 2?"):
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": task}],
                    "messageId": "test-msg-001",
                }
            },
        }

    @pytest.mark.asyncio
    async def test_send_task_completed(self):
        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "status": {"state": "completed"},
                "artifacts": [
                    {"parts": [{"kind": "text", "text": "The answer is 4."}]}
                ],
            },
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    "http://localhost:5001", json=self._make_payload()
                )

        assert resp.status_code == 200
        data = resp.json()
        result = data["result"]
        assert result["status"]["state"] == "completed"
        text_parts = [
            p["text"]
            for a in result["artifacts"]
            for p in a["parts"]
            if p.get("kind") == "text"
        ]
        assert any("4" in t for t in text_parts)

    @pytest.mark.asyncio
    async def test_send_task_direct_message_response(self):
        """Handle A2A responses that return parts directly in result."""
        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "parts": [{"kind": "text", "text": "Direct answer."}]
            },
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    "http://localhost:5001", json=self._make_payload()
                )

        data = resp.json()
        parts = data["result"]["parts"]
        assert any(p["text"] == "Direct answer." for p in parts if p["kind"] == "text")

    @pytest.mark.asyncio
    async def test_send_task_jsonrpc_error(self):
        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32600, "message": "Invalid request"},
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    "http://localhost:5001", json=self._make_payload()
                )

        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == -32600

    @pytest.mark.asyncio
    async def test_send_task_http_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    "http://localhost:5001", json=self._make_payload()
                )

        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_payload_structure(self):
        """Verify JSON-RPC payload is correctly formed."""
        payload = self._make_payload("Summarize this")
        assert payload["jsonrpc"] == "2.0"
        assert payload["method"] == "message/send"
        assert payload["params"]["message"]["role"] == "user"
        parts = payload["params"]["message"]["parts"]
        assert len(parts) == 1
        assert parts[0]["kind"] == "text"
        assert parts[0]["text"] == "Summarize this"

    @pytest.mark.asyncio
    async def test_send_task_with_nested_result(self):
        """Handle responses where result has a nested result field."""
        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "status": {"state": "completed"},
                "result": {
                    "status": "completed",
                    "parts": [{"kind": "text", "text": "Nested result."}],
                },
            },
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    "http://localhost:5001", json=self._make_payload()
                )

        data = resp.json()
        nested = data["result"]["result"]
        text_parts = [p["text"] for p in nested["parts"] if p["kind"] == "text"]
        assert "Nested result." in text_parts


# ── A2A SDK client ────────────────────────────────────────────────────────────

class TestSendWithSDK:
    @pytest.mark.asyncio
    async def test_sdk_send_message_task_response(self):
        """Test SDK client receiving a Task response with artifacts."""
        mock_artifact_part = MagicMock()
        mock_artifact_part.text = "SDK answer"

        mock_artifact = MagicMock()
        mock_artifact.parts = [mock_artifact_part]

        mock_task = MagicMock()
        mock_task.id = "task-123"
        mock_task.status = MagicMock(state="completed")
        mock_task.artifacts = [mock_artifact]

        mock_update = MagicMock()

        async def mock_send_message(message):
            yield (mock_task, mock_update)

        mock_client = MagicMock()
        mock_client.send_message = mock_send_message

        mock_connect = AsyncMock(return_value=mock_client)

        with patch.dict("sys.modules", {
            "a2a": MagicMock(),
            "a2a.client": MagicMock(
                ClientFactory=MagicMock(connect=mock_connect),
                create_text_message_object=MagicMock(return_value="mock_message"),
            ),
            "a2a.types": MagicMock(Role=MagicMock(user="user")),
        }):
            from a2a.client import ClientFactory, create_text_message_object
            from a2a.types import Role

            client = await ClientFactory.connect("http://localhost:5001")
            message = create_text_message_object(role=Role.user, content="test")

            results = []
            async for event in client.send_message(message):
                if isinstance(event, tuple):
                    task, update = event
                    results.append(task)

            assert len(results) == 1
            assert results[0].id == "task-123"
            assert results[0].artifacts[0].parts[0].text == "SDK answer"

    @pytest.mark.asyncio
    async def test_sdk_send_message_direct_response(self):
        """Test SDK client receiving a direct Message response."""
        mock_part_root = MagicMock()
        mock_part_root.text = "Direct SDK answer"

        mock_part = MagicMock()
        mock_part.root = mock_part_root

        mock_message = MagicMock()
        mock_message.parts = [mock_part]

        async def mock_send_message(message):
            yield mock_message

        mock_client = MagicMock()
        mock_client.send_message = mock_send_message

        mock_connect = AsyncMock(return_value=mock_client)

        with patch.dict("sys.modules", {
            "a2a": MagicMock(),
            "a2a.client": MagicMock(
                ClientFactory=MagicMock(connect=mock_connect),
                create_text_message_object=MagicMock(return_value="mock_message"),
            ),
            "a2a.types": MagicMock(Role=MagicMock(user="user")),
        }):
            from a2a.client import ClientFactory, create_text_message_object
            from a2a.types import Role

            client = await ClientFactory.connect("http://localhost:5001")
            message = create_text_message_object(role=Role.user, content="test")

            results = []
            async for event in client.send_message(message):
                if not isinstance(event, tuple):
                    for part in event.parts:
                        results.append(part.root.text)

            assert "Direct SDK answer" in results

    @pytest.mark.asyncio
    async def test_sdk_connection_error(self):
        """Test handling of connection errors from SDK client."""
        mock_connect = AsyncMock(side_effect=Exception("Connection refused"))

        with patch.dict("sys.modules", {
            "a2a": MagicMock(),
            "a2a.client": MagicMock(
                ClientFactory=MagicMock(connect=mock_connect),
            ),
            "a2a.types": MagicMock(Role=MagicMock(user="user")),
        }):
            from a2a.client import ClientFactory

            with pytest.raises(Exception, match="Connection refused"):
                await ClientFactory.connect("http://localhost:5001")
