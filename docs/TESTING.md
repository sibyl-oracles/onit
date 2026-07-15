# Testing

## Setup

Install test dependencies:

```bash
pip install -e ".[test]"
```

## Run the full test suite

```bash
pytest src/test/ -v
```

## Run specific test modules

```bash
pytest src/test/test_onit.py -v          # Core agent tests
pytest src/test/test_cli.py -v           # CLI tests
pytest src/test/test_a2a.py -v           # A2A protocol tests
pytest src/test/test_chat.py -v          # LLM chat tests
pytest src/test/test_viber.py -v         # Viber gateway tests
pytest src/test/test_chat_ui.py -v       # Terminal UI tests
pytest src/test/test_web_api.py -v       # Web UI tests
pytest src/test/test_mcp_prompts.py -v   # MCP prompt tests
pytest src/test/test_tool_discovery.py -v # Tool discovery tests
pytest src/test/test_mcp_tools_security.py -v # MCP tools security tests
```

## Test structure

All tests are in `src/test/`:

| File | Description |
|------|-------------|
| `test_onit.py` | Core agent, A2A executor, disconnect middleware, session isolation |
| `test_cli.py` | CLI argument parsing and client mode |
| `test_a2a.py` | A2A protocol integration tests |
| `test_chat.py` | LLM chat interface tests |
| `test_viber.py` | Viber gateway and session isolation tests |
| `test_chat_ui.py` | Terminal UI tests |
| `test_web_api.py` | Web UI: auth, WebSession, per-tab session isolation |
| `test_mcp_prompts.py` | Prompt template tests |
| `test_mcp_server_runner.py` | MCP server launcher tests |
| `test_mcp_tools_security.py` | Security tests for MCP tools |
| `test_text_utils.py` | Text utilities tests |
| `test_tool_discovery.py` | Tool discovery tests |
| `test_tool_registry.py` | Tool registry tests |

## Configuration

pytest is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["src/test"]
asyncio_mode = "auto"
```
