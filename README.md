# OnIt

*OnIt* — the AI is working on the given task and will deliver the results shortly.

OnIt is an intelligent agent framework for task automation and assistance. It is built on [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) for tool integration and supports the [A2A](https://a2a-protocol.org/) (Agent-to-Agent) protocol for multi-agent communication. OnIt connects to LLMs via any OpenAI-compatible API (private [vLLM](https://github.com/vllm-project/vllm) servers or [OpenRouter.ai](https://openrouter.ai/)) and orchestrates tasks through modular MCP servers.

## Quick Guide

### 1. Install

```bash
pip install onit==0.1.3c
```

Or from source:

```bash
git clone https://github.com/sibyl-oracles/onit.git
cd onit
pip install -e ".[all]" --upgrade
```

### 2. Configure

Set your LLM host and at least one API key:

```bash
# Option A: Private vLLM server
export ONIT_HOST=http://localhost:8000/v1

# Option B: OpenRouter.ai
export ONIT_HOST=https://openrouter.ai/api/v1
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Optional API keys for built-in tools:

```bash
export OLLAMA_API_KEY=your_key        # Web search. Best to enable this. Free rate limited.
export OPENWEATHER_API_KEY=your_key   # Weather data. Free.
```

Get your free API keys: [Ollama](https://ollama.com/) | [OpenWeatherMap](https://openweathermap.org/api)

### 3. Run

```bash
onit
```

That's it. MCP servers start automatically, and you get an interactive terminal chat with tool access.

**Other interfaces:**

```bash
onit --web                          # Gradio web UI on port 9000
onit --gateway                      # Telegram/Viber bot gateway
onit --a2a                          # A2A server on port 9001
onit --client --task "your task"    # Send a task to an A2A server and print the answer
```

## Configuration

All options can be set via CLI flags, environment variables, or a YAML config file:

```bash
onit --config configs/default.yaml
```

Example config (`configs/default.yaml`):

```yaml
serving:
  host: https://openrouter.ai/api/v1
  host_key: sk-or-v1-your-key-here   # or set OPENROUTER_API_KEY env var
  model: google/gemini-2.5-pro
  think: true
  max_tokens: 262144

verbose: false
timeout: 600
# prompt_intro: "I am a helpful AI assistant. My name is OnIt."

web: false
web_port: 9000

mcp:
  servers:
    - name: PromptsMCPServer
      url: http://127.0.0.1:18200/sse
      enabled: true
    - name: ToolsMCPServer
      url: http://127.0.0.1:18201/sse
      enabled: true
```

The LLM provider is auto-detected from the host URL. If it contains `openrouter.ai`, the API key is read from `host_key` or `OPENROUTER_API_KEY`. All other hosts default to vLLM with no key required.

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
| `--documents-path` | Path to local documents directory (model searches here before the web) | — |
| `--topic` | Default topic context (e.g. `"machine learning"`) | — |
| `--prompt-intro` | Custom system prompt intro for the model | `"I am a helpful AI assistant. My name is OnIt."` |

**Text UI:**

| Flag | Description | Default |
|------|-------------|---------|
| `--text-theme` | Text UI theme (`white` or `dark`) | `dark` |
| `--show-logs` | Show execution logs | `false` |

**Web UI:**

| Flag | Description | Default |
|------|-------------|---------|
| `--web` | Launch Gradio web UI | `false` |
| `--web-port` | Gradio web UI port | `9000` |

**Gateway (Telegram / Viber):**

| Flag | Description | Default |
|------|-------------|---------|
| `--gateway` | Auto-detect gateway (Telegram or Viber based on env vars) | — |
| `--gateway telegram` | Run as a Telegram bot gateway (requires `TELEGRAM_BOT_TOKEN`) | — |
| `--gateway viber` | Run as a Viber bot gateway (requires `VIBER_BOT_TOKEN`) | — |
| `--viber-webhook-url` | Public HTTPS URL for Viber webhook | — |
| `--viber-port` | Local port for Viber webhook server | `8443` |

**A2A (Agent-to-Agent):**

| Flag | Description | Default |
|------|-------------|---------|
| `--a2a` | Run as an A2A protocol server | `false` |
| `--a2a-port` | A2A server port | `9001` |
| `--client`, `--a2a-client` | Client mode: send a task to a remote A2A server | `false` |
| `--a2a-host` | A2A server URL for client mode | `http://localhost:9001` |
| `--task`, `--a2a-task` | Task string for A2A loop or client mode | — |
| `--file`, `--a2a-file` | File to upload to the A2A server with the task | — |
| `--image`, `--a2a-image` | Image file to send for vision processing | — |
| `--loop`, `--a2a-loop` | Enable A2A loop mode | `false` |
| `--period`, `--a2a-period` | Seconds between A2A loop iterations | `10` |

**MCP (Model Context Protocol):**

| Flag | Description | Default |
|------|-------------|---------|
| `--mcp-host` | Override the host/IP in all MCP server URLs (e.g. `192.168.1.100`) | — |
| `--mcp-sse` | URL of an external MCP tools server using SSE transport (can be repeated) | — |

## Features

### Interactive Chat

Rich terminal UI with input history, theming, and execution logs. Press Enter or Ctrl+C to interrupt any running task.

### Web UI

Gradio-based browser interface with file upload, copy buttons, and real-time polling:

```bash
onit --web --web-port 9000
```

Supports optional Google OAuth2 authentication — see [docs/WEB_AUTHENTICATION.md](docs/WEB_AUTHENTICATION.md).

### MCP Tool Integration

MCP servers are started automatically. Tools are auto-discovered and available to the agent.

| Server | Description |
|--------|-------------|
| PromptsMCPServer | Prompt templates for instruction generation |
| ToolsMCPServer | Web search, bash commands, file operations, and document tools |

Connect to additional external MCP servers:

```bash
onit --mcp-sse http://localhost:8080/sse --mcp-sse http://192.168.1.50:9090/sse
```

### Messaging Gateways (Telegram & Viber)

Chat with OnIt remotely from **Telegram** or **Viber** using a bot.

**Telegram:**

```bash
export TELEGRAM_BOT_TOKEN=your-bot-token-here
onit --gateway telegram
```

**Viber** (requires a public HTTPS webhook URL — see [Gateway Quick Start](docs/GATEWAY_QUICK_START.md)):

```bash
export VIBER_BOT_TOKEN=your-viber-token
export VIBER_WEBHOOK_URL=https://your-domain.com/viber
onit --gateway viber
```

Both gateways support text and photo messages. The photo caption is used as the prompt (defaults to "Describe this image." if no caption is provided). Use `--gateway` without a type to auto-detect based on which token is set.

**Install the gateway dependency separately if not using `[all]`:**

```bash
pip install "onit[gateway]"
```

### A2A Protocol

Run OnIt as an [A2A](https://a2a-protocol.org/) server so other agents can send tasks:

```bash
onit --a2a --a2a-port 9001
```

The agent card is available at `http://localhost:9001/.well-known/agent.json`.

**Send a task via CLI:**

```bash
onit --client --a2a-host http://192.168.86.101:9001 --task "what is the weather in Manila"
```

**Send a task via Python (A2A SDK):**

```python
from a2a.client import ClientFactory, create_text_message_object
from a2a.types import Role
import asyncio

async def main():
    client = await ClientFactory.connect("http://localhost:9001")
    message = create_text_message_object(role=Role.user, content="What is the weather?")
    async for event in client.send_message(message):
        print(event)

asyncio.run(main())
```

**Send a task with an image (VLM):**

```bash
# Server
onit --a2a --host <ONIT_HOST> --model Qwen/Qwen3-VL-8B-Instruct

# Client
onit --client --task "are the rambutans ripe?" --a2a-image assets/rambutan_calamansi.jpg
```

**Send an image task via Python (A2A SDK):**

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

### Loop Mode

Repeat a task on a configurable timer (useful for monitoring):

```bash
onit --a2a-loop --task "Check the weather in Manila" --a2a-period 60
```

### Custom Prompt Templates

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
onit --template-path my_template.yaml
```

See example templates in `src/mcp/prompts/prompt_templates/`.

## Model Serving

### Private vLLM

Serve models locally with [vLLM](https://github.com/vllm-project/vllm):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --max-model-len 262144 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 --tensor-parallel-size 4 \
  --chat-template-content-format string
```

```bash
export ONIT_HOST=http://localhost:8000/v1
onit
```

For vision-language models (VLM), serve on a separate port:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-35B-A3B \
  --port 8001 --max-model-len 262144 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 --tensor-parallel-size 4
```

```bash
export ONIT_HOST=http://localhost:8001/v1
onit
```

### OpenRouter.ai

[OpenRouter](https://openrouter.ai/) gives access to models from OpenAI, Google, Meta, Anthropic, and others through a single API.

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
export ONIT_HOST=https://openrouter.ai/api/v1
onit
```

Browse available models at [openrouter.ai/models](https://openrouter.ai/models) and use the model ID (e.g. `google/gemini-2.5-pro`, `meta-llama/llama-4-maverick`, `openai/gpt-4.1`).

## Design Philosophy

- **Portable** — Minimal dependencies. Deployable from embedded devices to GPU servers.
- **Modular** — Clear separation of AI logic, tasks, and UIs. Easily extendable with new MCP servers.
- **Scalable** — From a single tool to complex multi-server setups.
- **Redundant** — Multiple ways to solve a problem. Let the AI decide the optimal path.
- **Configurable** — Edit a YAML file and you are good to go.
- **Responsive** — Safety routines can interrupt running tasks at any time.

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
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────┐ │
│  │ ChatUI  │ │ WebChatUI│ │ Telegram │ │ Viber  │ │ A2A  │ │
│  │(terminal│ │ (Gradio) │ │ Gateway  │ │Gateway │ │Server│ │
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
│    (vLLM / OpenRouter)  (auto-discovered)           │
└─────────────────────────────────────────────────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
     ┌───────────┐ ┌──────────┐ ┌──────────┐
     │  Prompts  │ │  Tools   │ │ External │  ...
     │ MCP Server│ │MCP Server│ │MCP (SSE) │
     └───────────┘ └──────────┘ └──────────┘
```

## Project Structure

```
onit/
├── configs/
│   └── default.yaml            # Agent configuration
├── pyproject.toml              # Package configuration
├── src/
│   ├── cli.py                  # CLI entry point
│   ├── onit.py                 # Core agent class
│   ├── lib/
│   │   ├── text.py             # Text utilities
│   │   └── tools.py            # MCP tool discovery
│   ├── mcp/
│   │   ├── prompts/            # Prompt engineering (FastMCP)
│   │   └── servers/            # MCP servers (tools, web, bash, filesystem)
│   ├── model/
│   │   └── serving/
│   │       └── chat.py         # LLM interface (vLLM + OpenRouter)
│   ├── ui/
│   │   ├── text.py             # Rich terminal UI
│   │   ├── web.py              # Gradio web UI
│   │   ├── telegram.py         # Telegram bot gateway
│   │   └── viber.py            # Viber bot gateway
│   └── test/                   # Test suite (pytest)
```

## Documentation

- [Gateway Quick Start](docs/GATEWAY_QUICK_START.md) — Telegram and Viber bot setup
- [Testing](docs/TESTING.md) — Running the test suite
- [Docker](docs/DOCKER.md) — Docker and Docker Compose setup
- [Web Authentication](docs/WEB_AUTHENTICATION.md) — Web UI authentication reference
- [Web Deployment](docs/DEPLOYMENT_WEB.md) — Production deployment with HTTP/HTTPS via nginx or Caddy

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
