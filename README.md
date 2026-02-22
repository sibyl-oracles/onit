# OnIt

*OnIt* — the AI is working on the given task and will deliver the results shortly.

OnIt is an intelligent agent framework for task automation and assistance. It is built on [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) for tool integration and supports the [A2A](https://a2a-protocol.org/) (Agent-to-Agent) protocol for multi-agent communication. OnIt connects to LLMs via any OpenAI-compatible API (private [vLLM](https://github.com/vllm-project/vllm) servers or [OpenRouter.ai](https://openrouter.ai/)) and orchestrates tasks through modular MCP servers.

## Design Philosophy

OnIt is intended as lean AI agent framework. The design philosophy adheres to the following principles:

- **Portable** — Minimal dependencies beyond the core AI model library and MCP. Deployable from embedded devices to GPU servers.
- **Modular** — Clear separation of AI logic, tasks, and UIs. Easily extendable with new MCP servers and tools.
- **Scalable** — From a single tool to complex multi-server setups.
- **Redundant** — Multiple ways to solve a problem, sense the world, and execute actions. Let the AI decide the optimal path.
- **Configurable** — Edit a YAML file and you are good to go. Applies to both the agent and MCP servers.
- **Responsive** — Safety routines can interrupt running tasks at any time.

## Features

- **Interactive chat** — Rich terminal UI with input history, theming, and execution logs
- **Web UI** — Gradio-based browser interface with file upload, copy buttons, and real-time polling
- **MCP tool integration** — Automatic tool discovery from any number of MCP servers (web search, bash, office documents, Google Workspace)
- **A2A protocol** — Run OnIt as an A2A server so other agents can send tasks and receive responses
- **Loop mode** — Execute a fixed task on a configurable timer (useful for monitoring and periodic workflows)
- **Prompt templates** — Customizable YAML-based instruction templates per persona
- **Session logging** — All tasks and responses are saved as JSONL for audit and replay
- **Safety queue** — Press Enter or Ctrl+C to interrupt any running task

## Architecture

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
│  ┌─────────-┐  ┌──────────┐  ┌──────────┐           │
│  │ ChatUI   │  │ WebChatUI│  │ A2A      │           │
│  │(terminal)│  │ (Gradio) │  │ Server   │           │
│  └────┬────-┘  └────┬─────┘  └────┬─────┘           │
│       └─────────┬───┘             │                 │
│                 ▼                 ▼                 │
│          client_to_agent()  /  process_task()       │
│                 │                                   │
│                 ▼                                   │
│        MCP Prompt Engineering (FastMCP)             │
│                 │                                   │
│                 ▼                                   │
│         chat() ◄──── Tool Registry                  │
│    (vLLM / OpenRouter)  (auto-discovered)           │
└─────────────────────────────────────────────────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
     ┌───────────┐ ┌──────────┐ ┌──────────┐
     │ Web Search│ │  Bash    │ │  Office  │  ...
     │ MCP Server│ │MCP Server│ │MCP Server│
     └───────────┘ └──────────┘ └──────────┘
```

**Key components:**

| Component | Path | Description |
|-----------|------|-------------|
| `OnIt` | `src/onit.py` | Core agent class. Manages config, tool discovery, chat loop, A2A server, and session logging. |
| `ChatUI` | `src/ui/text.py` | Rich terminal UI with chat history, input history (arrow keys), execution logs panel, and theming. |
| `WebChatUI` | `src/ui/web.py` | Gradio web interface with file upload, async polling, and file download. |
| `Chat` | `src/model/serving/chat.py` | LLM interface via OpenAI-compatible API. Supports private vLLM and OpenRouter.ai models. Handles tool calling loops, thinking mode, retries, and safety interrupts. |
| `Tool discovery` | `src/lib/tools.py` | Connects to each MCP server URL, discovers available tools, and builds a unified tool registry. |
| `Prompts` | `src/mcp/prompts/prompts.py` | FastMCP-based prompt engineering. Supports custom YAML templates per persona. |
| `MCP servers` | `src/mcp/servers/` | Pre-built MCP servers for web search, bash, Microsoft Office, and Google Workspace. |

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/sibyl-oracles/onit.git
cd onit
pip install -e .
```

### With optional dependencies

```bash
# Web UI (Gradio)
pip install -e ".[web]"

# Everything
pip install -e ".[all]"
```

### From pip

```bash
pip install onit==0.1.1a
```

## Docker

### Build the image

```bash
docker build -t onit .
```

### Run the container

**Interactive terminal mode:**

```bash
docker run -it --rm --env-file .env onit
```

**Web UI (Gradio on port 9000):**

```bash
docker run -it --rm -p 9000:9000 --env-file .env onit --web --web-port 9000
```

**A2A server:**

```bash
docker run -it --rm -p 9001:9001 --env-file .env onit --a2a --a2a-port 9001
```

**With a custom config:**

```bash
docker run -it --rm -v $(pwd)/configs:/app/configs --env-file .env onit --config configs/default.yaml
```

### Docker Compose

Start the MCP servers, web UI, and A2A server together:

```bash
docker compose up --build
```

This launches three services defined in `docker-compose.yml`:
- **onit-mcp** — MCP servers on ports 18200-18204
- **onit-web** — Web UI on port 9000 (depends on MCP servers)
- **onit-a2a** — A2A server on port 9001 (depends on MCP servers)

The web and A2A services automatically use `--mcp-host onit-mcp` to route MCP requests to the MCP container via Docker networking.

> **Note:** Pass API keys via an `.env` file or individual `-e KEY=value` flags. Never bake secrets into the image.

## Quick Start

### 1. Set up environment variables

All environment variables must be set before running any OnIt component.

**LLM serving host** — OnIt works with any OpenAI-compatible API (auto-detected from URL):

```bash
# Private vLLM server
export ONIT_HOST=http://localhost:8000/v1

# Or OpenRouter.ai
export ONIT_HOST=https://openrouter.ai/api/v1
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

**MCP server API keys:**

| Variable | Description | Get Key |
|----------|-------------|---------|
| `OLLAMA_API_KEY` | API key for Ollama web search | [ollama.com](https://ollama.com/) |
| `OPENWEATHER_API_KEY` | API key for weather data | [openweathermap.org](https://openweathermap.org/api) |

```bash
export OLLAMA_API_KEY=your_ollama_api_key
export OPENWEATHER_API_KEY=your_openweather_api_key
```

Or configure everything in `configs/default.yaml`:

```yaml
serving:
  host: https://openrouter.ai/api/v1
  host_key: sk-or-v1-your-key-here   # or set OPENROUTER_API_KEY env var
  model: google/gemini-2.5-pro
```

> The provider is auto-detected: if the host URL contains `openrouter.ai`, the API key is read from `host_key` in the config or the `OPENROUTER_API_KEY` environment variable. All other hosts default to vLLM with no key required.

### 2. Start MCP servers

MCP servers must be running before launching the agent. Start all enabled servers with:

```bash
# Default config (src/mcp/servers/configs/default.yaml)
onit --mcp

# Custom config
onit --mcp --config path/to/mcp_servers.yaml

# With debug logging
onit --mcp --config path/to/mcp_servers.yaml --mcp-log-level DEBUG
```

This starts all enabled MCP servers defined in the config file. Each server runs in its own process. The MCP server config structure is:

```yaml
servers:
  - name: WebSearchMCPServer
    module: tasks.web.search
    description: "MCP server for web search"
    enabled: true
    host: 0.0.0.0
    port: 18201
    path: /search
    transport: 'streamable-http'
```

See `src/mcp/servers/configs/default.yaml` for the full default configuration.

### 3. Run the agent

With environment variables set and MCP servers running, launch the agent:

**Terminal chat:**

```bash
onit
```

**Web UI (Gradio):**

```bash
onit --web --web-port 9000
```

**Loop mode** (repeat a task on a timer):

```bash
onit --a2a-loop --a2a-task "Check the weather in Manila" --a2a-period 60
```

**A2A server** (accept tasks from other agents):

```bash
onit --a2a --a2a-port 9001
```

**Client mode** (send a task to a remote OnIt A2A server):

```bash
onit --a2a-client --a2a-host http://127.0.0.1:9001 --a2a-task "what is the weather"
```

## CLI Options

**General:**

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML configuration file | `configs/default.yaml` |
| `--host` | LLM serving host URL (overrides config and `ONIT_HOST` env var) | — |
| `--model` | Model name (overrides `serving.model` in config) | — |
| `--verbose` | Enable verbose logging | `false` |
| `--timeout` | Request timeout in seconds (`-1` = none) | `600` |
| `--template-path` | Path to custom prompt template YAML file | — |

**Text UI:**

| Flag | Description | Default |
|------|-------------|---------|
| `--text-theme` | Text UI theme (`white` or `dark`) | `dark` |
| `--text-show-logs` | Show execution logs panel | `false` |

**Web UI:**

| Flag | Description | Default |
|------|-------------|---------|
| `--web` | Launch Gradio web UI | `false` |
| `--web-port` | Gradio web UI port | `9000` |

**A2A (Agent-to-Agent):**

| Flag | Description | Default |
|------|-------------|---------|
| `--a2a` | Run as an A2A protocol server | `false` |
| `--a2a-port` | A2A server port | `9001` |
| `--a2a-client` | Client mode: send a task to a remote A2A server | `false` |
| `--a2a-host` | A2A server URL for client mode | `http://localhost:9001` |
| `--a2a-task` | Task string for A2A loop or client mode | — |
| `--a2a-file` | File to upload to the A2A server with the task | — |
| `--a2a-image` | Image file to send for vision processing | — |
| `--a2a-loop` | Enable A2A loop mode | `false` |
| `--a2a-period` | Seconds between A2A loop iterations | `10` |

**MCP (Model Context Protocol):**

| Flag | Description | Default |
|------|-------------|---------|
| `--mcp` | Run MCP servers | `false` |
| `--mcp-host` | Override the host/IP in all MCP server URLs (e.g. `192.168.1.100`) | — |
| `--mcp-log-level` | MCP server log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) | `INFO` |

## Configuration

All options can be set in the YAML config file and overridden via CLI flags. The full configuration structure:

```yaml
# LLM serving
serving:
  # host: resolved from --host flag > config value > ONIT_HOST env var
  model: Qwen/Qwen3-30B-A3B-Instruct-2507
  think: true
  max_tokens: 262144

# Agent behavior
persona: "assistant"
verbose: false
show_logs: false
theme: dark
timeout: 600

# Paths
session_path: "~/.onit/sessions"
template_path:                # optional path to custom prompt template YAML

# Web UI settings
web_title: "OnIt Chat"
web: false
web_port: 9000
web_share: false

# Google OAuth2 Authentication (optional)
web_google_client_id:
web_google_client_secret:

# MCP servers the agent connects to as a client
mcp:
  # mcp_host: 192.168.1.100    # override host/IP in all server URLs (or use --mcp-host)
  servers:
    - name: PromptsMCPServer
      description: Provides prompt templates for instruction generation
      url: http://127.0.0.1:18200/prompts
      enabled: true

    - name: WebSearchHandler
      description: Handles web, news and weather search queries
      url: http://127.0.0.1:18201/search
      enabled: true
```

## Custom Prompt Templates

Create a YAML file with an `instruction_template` field:

```yaml
# my_template.yaml
instruction_template: |
  You are a research assistant. Think step by step.

  <task>
  {task}
  </task>

  Save all results to `{data_path}`.
  Session ID: {session_id}
```

Then use it:

```bash
onit --config my_config.yaml
```

With `template_path: my_template.yaml` in the config, or set it directly in the config file.

See example templates in `src/mcp/prompts/prompt_templates/`.

## MCP Servers

### Pre-built servers

| Server | Module | Description |
|--------|--------|-------------|
| Web Search | `tasks.web.search` | Web, news, and weather search (uses Ollama Search API) |
| Bash | `tasks.os.bash` | Execute shell commands |
| Document Search | `tasks.os.filesystem` | Search patterns in documents (text, PDF, markdown) with table extraction |
| Microsoft Office | `tasks.office.microsoft` | Create Word, Excel, PowerPoint documents |
| Google Workspace | `tasks.office.google` | Create Google Docs, Sheets, Slides (disabled by default) |

> **Google Workspace:** The Google Workspace MCP server is disabled by default in the agent config. To enable it, first set up the Google Workspace API by following the [Google Workspace and OAuth guide](docs/GOOGLE_WORKSPACE_AND_OAUTH.md), then set `enabled: true` for `GoogleWorkspaceMCPServer` in your `configs/default.yaml`.

### Running MCP servers

All servers are configured via `src/mcp/servers/configs/default.yaml` and launched with:

```bash
onit --mcp

# Or with a custom config
onit --mcp --config path/to/mcp_servers.yaml
```

## A2A Protocol

OnIt can run as an [A2A](https://a2a-protocol.org/) server, allowing other agents to send tasks and receive responses.

### Start the A2A server

```bash
onit --a2a --a2a-port 9001
```

The agent card is available at `http://localhost:9001/.well-known/agent.json`.

### Send a task (Python)

```python
import httpx, asyncio

async def send_task():
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "What is 2 + 2?"}],
                "messageId": "test-001",
            }
        },
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post("http://localhost:9001", json=payload)
        print(resp.json())

asyncio.run(send_task())
```

**With an image (VLM):**

```python
import httpx, asyncio, base64, os

async def send_image_task():
    image_path = "assets/rambutan_calamansi.jpg"
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {"kind": "text", "text": "Are the rambutans ripe enough to be eaten?"},
                    {
                        "kind": "file",
                        "file": {
                            "bytes": image_data,
                            "mimeType": "image/jpeg",
                            "name": os.path.basename(image_path),
                        },
                    },
                ],
                "messageId": "vlm-001",
            }
        },
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post("http://localhost:9001", json=payload)
        print(resp.json())

asyncio.run(send_image_task())
```

### Send a task (A2A SDK)

```python
from a2a.client import ClientFactory, create_text_message_object
from a2a.types import Role

async def main():
    client = await ClientFactory.connect("http://localhost:9001")
    message = create_text_message_object(role=Role.user, content="What is the weather?")
    async for event in client.send_message(message):
        print(event)

asyncio.run(main())
```

**With an image (VLM):**

```python
import asyncio, base64, os, uuid
from a2a.client import ClientFactory
from a2a.types import FilePart, FileWithBytes, Message, Part, Role, TextPart

async def main():
    image_path = "assets/rambutan_calamansi.jpg"
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    message = Message(
        role=Role.user,
        message_id=str(uuid.uuid4()),
        parts=[
            Part(root=TextPart(text="Are the rambutans ripe enough to be eaten?")),
            Part(root=FilePart(file=FileWithBytes(
                bytes=image_data,
                mime_type="image/jpeg",
                name=os.path.basename(image_path),
            ))),
        ],
    )

    client = await ClientFactory.connect("http://localhost:9001")
    async for event in client.send_message(message):
        print(event)

asyncio.run(main())
```

A2A protocol tests are included in the test suite: `pytest src/test/test_a2a.py -v`.

### Send a task (OnIt client)

The simplest way to send a task to a remote OnIt A2A server:

```bash
# Basic usage
onit --a2a-client --a2a-host http://192.168.86.101:9001 --a2a-task "what is the weather"

# With a longer timeout (default is 120s)
onit --a2a-client --a2a-host http://192.168.86.101:9001 --a2a-task "summarize this report" --timeout 300
```

The command prints the response and exits. No config file, model serving, or UI is needed.

## Model Serving

OnIt works with any OpenAI-compatible API. The provider is auto-detected from the host URL.

### Private vLLM

Use [vLLM](https://github.com/vllm-project/vllm) to serve models locally:

```bash
# Serve Qwen3-30B with tool calling support

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --max-model-len 262144 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 --tensor-parallel-size 4 \
  --chat-template-content-format string

```

Then point the agent at your server:

```bash
export ONIT_HOST=http://localhost:8000/v1
onit
```

### Serving VLM

OnIt supports vision-language models (VLMs) for image understanding tasks over the A2A protocol.

**A2A server:**

```bash
onit --a2a --host <ONIT_HOST> --model Qwen/Qwen3-VL-8B-Instruct
```

**Client:**

```bash
onit --a2a-client --a2a-task "are the rambutans ripe enough to be eaten?" --a2a-image assets/rambutan_calamansi.jpg
```

### OpenRouter.ai

[OpenRouter](https://openrouter.ai/) gives access to models from OpenAI, Google, Meta, Anthropic, and others through a single API.

1. Create an account at [openrouter.ai](https://openrouter.ai/) and generate an API key.
2. Set the key and host:

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
export ONIT_HOST=https://openrouter.ai/api/v1
onit
```

Or configure in `configs/default.yaml`:

```yaml
serving:
  host: https://openrouter.ai/api/v1
  host_key: sk-or-v1-your-key-here
  model: google/gemini-2.5-pro
  think: true
  max_tokens: 262144
```

Browse available models at [openrouter.ai/models](https://openrouter.ai/models) and use the model ID (e.g. `google/gemini-2.5-pro`, `meta-llama/llama-4-maverick`, `openai/gpt-4.1`) as the `model` value.

## Project Structure

```
onit/
├── configs/
│   └── default.yaml            # Agent configuration
├── pyproject.toml              # Package configuration
├── src/
│   ├── __init__.py             # Exports OnIt
│   ├── cli.py                  # CLI entry point
│   ├── onit.py                 # Core agent class
│   ├── lib/
│   │   ├── text.py             # Text utilities
│   │   └── tools.py            # MCP tool discovery
│   ├── mcp/
│   │   ├── prompts/
│   │   │   ├── prompts.py      # Prompt engineering (FastMCP)
│   │   │   └── prompt_templates/  # YAML templates
│   │   └── servers/
│   │       ├── run.py          # Multi-process server launcher
│   │       ├── configs/        # Server config YAMLs
│   │       └── tasks/          # Task servers (web, bash, office)
│   ├── model/
│   │   └── serving/
│   │       └── chat.py          # LLM interface (vLLM + OpenRouter)
│   ├── type/
│   │   └── tools.py            # Type definitions
│   ├── ui/
│       ├── text.py             # Rich terminal UI
│       ├── utils.py            # UI utilities
│       └── web.py              # Gradio web UI
│   └── test/                   # Test suite (pytest)
│       ├── test_onit.py        # Core agent tests
│       ├── test_cli.py         # CLI tests
│       ├── test_a2a.py         # A2A protocol tests
│       ├── test_chat.py        # LLM chat tests
│       ├── test_chat_ui.py     # Terminal UI tests
│       ├── test_web_ui.py      # Web UI tests
│       └── ...                 # Additional test modules
```

## Testing

Run the full test suite:

```bash
pip install -e ".[test]"
pytest src/test/ -v
```

## Documentation

- [Google Workspace and OAuth for Gmail](docs/GOOGLE_WORKSPACE_AND_OAUTH.md) — Service account setup, domain-wide delegation, Gmail, and Web UI OAuth
- [OAuth2 Redirect Flow](docs/OAUTH_REDIRECT_FLOW.md) — Full OAuth2 redirect flow implementation details
- [OAuth Quick Start](docs/OAUTH_SETUP_QUICK_START.md) — Quick setup checklist for Google OAuth
- [Web Authentication](docs/WEB_AUTHENTICATION.md) — Web UI authentication reference
- [Web Deployment](docs/DEPLOYMENT_WEB.md) — Production deployment with HTTP/HTTPS via nginx or Caddy

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
