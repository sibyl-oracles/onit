# OnIt

*OnIt* — the AI is working on the given task and will deliver the results shortly.

OnIt is an intelligent agent for task automation and assistance. It connects to private [vLLM](https://github.com/vllm-project/vllm) servers, [OpenRouter.ai](https://openrouter.ai/), and [Ollama cloud](https://ollama.com) for hosted models — and uses [MCP](https://modelcontextprotocol.io/) tools for web search, file operations, and more. It also supports the [A2A](https://a2a-protocol.org/) protocol for multi-agent communication.

## Getting Started

### 1. Install

```bash
pip install onit
```

Or from source:

```bash
git clone https://github.com/sibyl-oracles/onit.git
cd onit
pip install -e ".[all]"
```

### 2. Setup

```bash
onit setup
```

The setup wizard walks you through configuring your LLM endpoint, API keys, and preferences. Secrets are stored securely in your OS keychain. Settings are saved to `~/.onit/config.yaml`.

To review your configuration at any time:

```bash
onit setup --show
```

### 3. Run

```bash
onit
```

That's it. MCP tools start automatically, and you get an interactive chat with tool access.

### Other interfaces

```bash
onit --web                          # Web UI on port 9000
onit --gateway                      # Telegram/Viber bot gateway
onit --a2a                          # A2A server on port 9001
onit --client --task "your task"    # Send a task to an A2A server
onit --sandbox                      # Delegate code execution to a sandbox MCP provider
onit --container                    # Run the whole process in a hardened Docker container
```

## Configuration

`onit setup` is the recommended way to configure OnIt. It stores:

- **Settings** in `~/.onit/config.yaml` (LLM endpoint, theme, ports, timeout)
- **Secrets** in your OS keychain (API keys, bot tokens)

You can also use environment variables or a project-level YAML config:

```bash
# Environment variables
export ONIT_HOST=https://openrouter.ai/api/v1
export OPENROUTER_API_KEY=sk-or-v1-...

# Or a custom config file
onit --config configs/default.yaml
```

Priority order: CLI flags > environment variables > `~/.onit/config.yaml` > project config file.

### Example config (`configs/default.yaml`)

```yaml
serving:
  host: https://openrouter.ai/api/v1
  host_key: sk-or-v1-your-key-here   # or set OPENROUTER_API_KEY env var
  # model: auto-detected from endpoint. Set explicitly for OpenRouter:
  # model: google/gemini-2.5-pro
  think: true
  max_tokens: 262144

verbose: false
timeout: 600
sandbox: false

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

### Tip: Qwen3.5 recommended parameters

[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-27B) works best with mode-specific sampling parameters:

| Mode | Use case | `temperature` | `top_p` | `top_k` | `presence_penalty` |
|------|----------|:---:|:---:|:---:|:---:|
| Thinking (`--think`) | General | `1.0` | `0.95` | `20` | `1.5` |
| Thinking (`--think`) | Precise coding | `0.6` | `0.95` | `20` | `0.0` |
| Instruct (no think) | General | `0.7` | `0.8` | `20` | `1.5` |
| Instruct (no think) | Reasoning | `1.0` | `1.0` | `40` | `2.0` |

Set `repetition_penalty: 1.0` in all cases (OnIt handles `think`/non-`think` automatically). Example for thinking + general tasks:

```bash
onit --think --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5
```

Or in `configs/default.yaml`:

```yaml
serving:
  host: http://localhost:8000/v1
  think: true
  temperature: 1.0
  top_p: 0.95
  top_k: 20
  presence_penalty: 1.5
  repetition_penalty: 1.0
```

## CLI Reference

**General:**

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML configuration file | `configs/default.yaml` |
| `--host` | LLM serving host URL | — |
| `--verbose` | Enable verbose logging | `false` |
| `--timeout` | Request timeout in seconds (`-1` = none) | `600` |
| `--template-path` | Path to custom prompt template YAML file | — |
| `--documents-path` | Path to local documents directory | — |
| `--topic` | Default topic context (e.g. `"machine learning"`) | — |
| `--prompt-intro` | Custom system prompt intro | — |
| `--no-stream` | Disable token streaming | `false` |
| `--think` | Enable thinking/reasoning mode (CoT) | `false` |
| `--temperature` | Sampling temperature | `0.6` |
| `--top-p` | Top-p nucleus sampling | `0.95` |
| `--top-k` | Top-k sampling | `20` |
| `--min-p` | Min-p sampling | `0.0` |
| `--presence-penalty` | Presence penalty | `0.0` |
| `--repetition-penalty` | Repetition penalty | `1.05` (or `1.0` with `--think`) |
| `--sandbox` | Delegate code execution to an external MCP sandbox provider | `false` |
| `--container` | Run the entire OnIt process inside a hardened Docker container (read-only rootfs, non-root user, caps dropped, no host mounts). Requires Docker. Composable with `--sandbox`. | `false` |
| `--container-gpus` | Pass GPUs into the container (e.g. `all`, `"device=0,1"`). Requires NVIDIA Container Toolkit. Default image ships CPU-only torch. | — |
| `--container-mount` | Extra bind mount for the container, e.g. `/host/path:/container/path:ro`. Repeatable. Prefer `:ro`. | — |

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
| `--gateway telegram` | Run as a Telegram bot | — |
| `--gateway viber` | Run as a Viber bot | — |
| `--viber-webhook-url` | Public HTTPS URL for Viber webhook | — |
| `--viber-port` | Local port for Viber webhook server | `8443` |

**A2A (Agent-to-Agent):**

| Flag | Description | Default |
|------|-------------|---------|
| `--a2a` | Run as an A2A protocol server | `false` |
| `--a2a-port` | A2A server port | `9001` |
| `--client` | Send a task to a remote A2A server | `false` |
| `--a2a-host` | A2A server URL for client mode | `http://localhost:9001` |
| `--task` | Task string for A2A client or loop mode | — |
| `--file` | File to upload with the task | — |
| `--image` | Image file for vision processing | — |
| `--loop` | Enable A2A loop mode | `false` |
| `--period` | Seconds between loop iterations | `10` |

**MCP (Model Context Protocol):**

| Flag | Description | Default |
|------|-------------|---------|
| `--mcp-host` | Override host/IP in all MCP server URLs | — |
| `--mcp-sse` | URL of an external MCP server (SSE transport, can be repeated) | — |
| `--mcp-server` | URL of an external MCP server (Streamable HTTP transport, can be repeated) | — |

## Features

### Interactive Chat

Rich terminal UI with input history, theming, and execution logs. Press Enter or Ctrl+C to interrupt any running task.

### Web UI

Gradio-based browser interface with file upload and real-time streaming:

```bash
onit --web
```

Supports optional Google OAuth2 authentication — see [docs/WEB_AUTHENTICATION.md](docs/WEB_AUTHENTICATION.md).

### Container Mode

Run the entire OnIt process inside a hardened Docker container so a breach in the agent, LLM output parser, or any MCP tool cannot reach the host OS:

```bash
onit --container                                         # interactive terminal in container
onit --container --web                                   # web UI, port 9000 published
onit --container --a2a --a2a-port 9100                   # non-default ports are honored (9100:9100)
onit --container --container-gpus all                    # NVIDIA GPU pass-through
onit --container --container-mount "$HOME/docs:/home/onit/documents:ro" \
  --documents-path /home/onit/documents                  # expose specific host paths read-only
onit --container --sandbox                               # combine with per-tool sandboxing
```

The first run auto-builds the `onit:local` image from the repo `Dockerfile`. Subsequent runs reuse the image — remove it (`docker rmi onit:local`) to force a rebuild after source changes.

#### Isolation posture

- Runs as non-root user `onit` (uid 1000).
- `--read-only` root filesystem; `/tmp` and `/home/onit/.cache` are ephemeral tmpfs.
- `--cap-drop=ALL`, `--security-opt=no-new-privileges`, `--pids-limit=256`, `--memory=2g`.
- No host filesystem mounts by default. Outbound network is allowed (needed for LLM APIs).

#### What crosses the boundary

| Resource | Default behavior |
|---|---|
| `~/.onit/config.yaml` | Bind-mounted read-only at `/home/onit/.onit/config.yaml` |
| `~/.onit/secrets.yaml` | Bind-mounted read-only if present |
| Host keychain secrets | Passed as ephemeral env vars (not baked into image): `OPENROUTER_API_KEY`, `OLLAMA_API_KEY`, `OPENWEATHERMAP_API_KEY`, `TELEGRAM_BOT_TOKEN`, `VIBER_BOT_TOKEN`, `GITHUB_TOKEN`, `HF_TOKEN` |
| Session data | Named volume `onit-data` mounted at `/home/onit/data` (writable, persistent) |
| Ports | Published **only** for the active mode — see the ports table below |
| GPUs | **Not** passed through unless `--container-gpus` is set |
| Host filesystem | **Nothing** beyond the config/secrets files above unless `--container-mount` is set |

#### Network ports

The launcher publishes the minimum set of host ports for the selected mode. Interactive terminal mode publishes **no ports at all**. Outbound network (from the container) is always allowed so the agent can reach LLM APIs.

| Mode flag | Default published port | Override flag | Example |
|---|---|---|---|
| (none / terminal) | — (no ports) | — | `onit --container` |
| `--web` | `9000:9000` | `--web-port` | `onit --container --web --web-port 9500` → `9500:9500` |
| `--a2a` | `9001:9001` | `--a2a-port` | `onit --container --a2a --a2a-port 9100` → `9100:9100` |
| `--gateway viber` | `8443:8443` | `--viber-port` | `onit --container --gateway viber --viber-port 9443` → `9443:9443` |

The host-side and container-side port numbers are kept identical so URLs such as webhooks work without translation. If you need to remap to a different host port, run the image with your own `docker run -p <host>:<container>` instead of `--container`.

#### Container-only flags (not forwarded inside the container)

| Flag | Purpose |
|---|---|
| `--container-gpus <spec>` | NVIDIA GPU pass-through. Values like `all` or `"device=0,1"`. Requires NVIDIA Container Toolkit on the host. The default image ships **CPU-only** torch — for GPU compute, rebuild the image with a CUDA torch wheel. |
| `--container-mount <host>:<container>[:ro]` | Extra bind mount. Repeatable. Prefer `:ro`. Each mount punches a hole in the sandbox — never mount a host path that contains secrets you don't want the agent to see. |

#### Preinstalled ML packages

The container image ships with a default ML stack so the agent can run code without `pip install` at execution time: `torch`, `torchvision`, `torchaudio` (CPU-only wheels), `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `einops`.

See [docs/DOCKER.md](docs/DOCKER.md) for full details.

### Sandbox Mode

Delegate individual code-execution tool calls to an external MCP sandbox provider. Complementary to `--container` (which isolates the whole OnIt process); combine them for defense in depth.

```bash
onit --sandbox                   # per-tool sandboxing via external MCP provider
onit --container --sandbox       # + whole-process isolation in a Docker container
```

Or in `config.yaml`:

```yaml
sandbox: true
```

Sandbox mode requires an MCP server that provides sandbox tools (e.g. `sandbox_run_code`, `sandbox_install_packages`, `sandbox_stop`). The agent auto-injects `session_id` and `data_path` into any tool whose schema declares those parameters, so the MCP server can maintain per-session containers and mount the correct data directory.

When the user interrupts a running task (Ctrl+C, Enter, or stop button), the agent automatically calls `sandbox_stop` to clean up the container.

### MCP Tool Integration

MCP servers start automatically. Tools are auto-discovered and available to the agent.

| Server | Description |
|--------|-------------|
| PromptsMCPServer | Prompt templates for instruction generation |
| ToolsMCPServer | Web search, bash commands, file operations, and document tools |

Connect to additional external MCP servers:

```bash
onit --mcp-sse http://localhost:8080/sse --mcp-sse http://192.168.1.50:9090/sse
```

### Messaging Gateways

Chat with OnIt from **Telegram** or **Viber**. Configure bot tokens via `onit setup` or environment variables, then:

```bash
onit --gateway telegram
onit --gateway viber --viber-webhook-url https://your-domain.com/viber
```

Install the gateway dependency separately if not using `[all]`:

```bash
pip install "onit[gateway]"
```

### A2A Protocol

Run OnIt as an [A2A](https://a2a-protocol.org/) server so other agents can send tasks:

```bash
onit --a2a
```

The agent card is available at `http://localhost:9001/.well-known/agent.json`.

**Send a task via CLI:**

```bash
onit --client --task "what is the weather in Manila"
onit --client --task "describe this" --image photo.jpg
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

### Loop Mode

Repeat a task on a configurable timer (useful for monitoring):

```bash
onit --loop --task "Check the weather in Manila" --period 60
```

### Custom Prompt Templates

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
onit --host http://localhost:8000/v1
```

### OpenRouter.ai

[OpenRouter](https://openrouter.ai/) gives access to models from OpenAI, Google, Meta, Anthropic, and others through a single API.

```bash
onit --host https://openrouter.ai/api/v1
```

Browse available models at [openrouter.ai/models](https://openrouter.ai/models).

### Ollama Cloud

[Ollama cloud](https://ollama.com) hosts models that are accessed via the native [Ollama Python SDK](https://github.com/ollama/ollama-python). Store your API key once:

```bash
onit setup   # enter your Ollama API key when prompted
```

Or set the environment variable:

```bash
export OLLAMA_API_KEY=your-ollama-key
```

Then point OnIt at the Ollama cloud host and specify a model:

```bash
onit --host https://api.ollama.com --model gemma4:31b-cloud
onit --host https://api.ollama.com --model llama4:scout-cloud
```

Model is auto-detected from the endpoint if `--model` is omitted. You can also set the host permanently in your config:

```yaml
serving:
  host: https://api.ollama.com
  model: gemma4:31b-cloud
```

> **Note:** Ollama cloud uses the `ollama_api_key` keyring entry (the same key used for the web search tool). Generation parameters (`temperature`, `top_p`, `top_k`, `--think`) are fully supported.

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
│  (vLLM / OpenRouter / Ollama cloud) (auto-discovered) │
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
│   │       └── chat.py         # LLM interface (vLLM, OpenRouter, Ollama cloud)
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
- [Web Deployment](docs/DEPLOYMENT_WEB.md) — Production deployment with HTTP/HTTPS

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
