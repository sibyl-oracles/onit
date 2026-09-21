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
│  ┌─────────┐ ┌──────────┐ ┌───────┐ ┌───────────────┐ │
│  │ ChatUI  │ │ WebApiUI │ │ Loop  │ │ legacy/ (A2A, │ │
│  │(terminal│ │(FastAPI) │ │ runner│ │ Telegram,     │ │
│  └────┬────┘ └────┬─────┘ └───┬───┘ │ Viber)        │ │
└───────┼───────────┼───────────┼─────┴───┬───────────┘
│       └──────────┼──┘            │             │       │
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
│   │   └── static/             # Web UI assets (no build step)
│   └── test/                   # Test suite (pytest)
```

Legacy front ends (A2A server, Telegram and Viber gateways) live in
`legacy/` at the repository root.
