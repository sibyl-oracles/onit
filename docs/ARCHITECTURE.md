# Architecture

```
┌─────────────────────────────────────────────────────┐
│                       onit CLI                      │
│                  (argparse + YAML config)           │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                     OnIt (src/onit.py)              │
│                                                     │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────┐ │
│  │ ChatUI  │ │ WebApiUI │ │ Telegram │ │ Viber  │ │ A2A  │ │
│  │(terminal│ │(FastAPI) │ │ Gateway  │ │Gateway │ │Server│ │
│  └────┬────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ └──┬───┘ │
│       └─────────┬─┘            │             │       │
│                 ▼                 ▼                 │
│          client_to_agent()  /  process_task()       │
│                 │                                   │
│                 ▼                                   │
│        MCP Prompt Engineering (FastMCP)             │
│                 │                                   │
│                 ▼                                   │
│         chat() ◄──── Tool Registry                  │
│ (vLLM/Ollama/MLX/OpenRouter) (auto-discovered)      │
└─────────────────────────────────────────────────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
     ┌───────────┐ ┌──────────┐ ┌──────────┐
     │  Prompts  │ │  Tools   │ │ External │  ...
     │ MCP Server│ │MCP Server│ │MCP (SSE) │
     └───────────┘ └──────────┘ └──────────┘
```

## Project structure

```
onit/
├── configs/
│   └── default.yaml            # Agent configuration
├── pyproject.toml              # Package configuration
├── src/
│   ├── cli.py                  # CLI entry point
│   ├── setup.py                # Setup wizard (onit setup)
│   ├── onit.py                 # Core agent class
│   ├── lib/
│   │   ├── text.py             # Text utilities
│   │   └── tools.py            # MCP tool discovery
│   ├── mcp/
│   │   ├── prompts/            # Prompt engineering (FastMCP)
│   │   └── servers/            # MCP servers (tools, web, bash, filesystem)
│   ├── type/
│   │   └── tools.py            # Tool registry and schema utilities
│   ├── model/
│   │   └── serving/
│   │       └── chat.py         # LLM interface (vLLM, Ollama, MLX, OpenRouter)
│   ├── ui/
│   │   ├── text.py             # Rich terminal UI
│   │   ├── api.py              # FastAPI + SSE web UI
│   │   ├── static/             # Web UI assets (no build step)
│   │   ├── telegram.py         # Telegram bot gateway
│   │   └── viber.py            # Viber bot gateway
│   └── test/                   # Test suite (pytest)
```
