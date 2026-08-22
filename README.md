# OnIt

*OnIt* — the AI is working on the given task and will deliver the results shortly.

OnIt is a terminal AI agent. It runs on a model server you control — [vLLM](https://github.com/vllm-project/vllm),
[Ollama](https://ollama.com), or [MLX](https://github.com/ml-explore/mlx-lm) on Apple silicon —
or on a hosted endpoint ([OpenRouter](https://openrouter.ai/), [Ollama cloud](https://ollama.com)),
and drives its work through [MCP](https://modelcontextprotocol.io/) tools: web search, weather,
shell, file editing, and search over your own documents.

This README gets the **text UI** running. Everything else — web UI, containers, bot
gateways, the full CLI and configuration reference — is in [docs/](docs/).

## Quick Start

Four steps: environment → model server → keys → run.

### 1. Install

OnIt needs **Python 3.10–3.12** (3.12 recommended). Install it into its own virtual
environment so dependencies stay isolated:

```bash
conda create -n onit python=3.12 -y && conda activate onit
# or: uv venv ~/.venvs/onit --python 3.12 && source ~/.venvs/onit/bin/activate
```

```bash
pip install onit
```

From source instead:

```bash
git clone https://github.com/sibyl-oracles/onit.git
cd onit && pip install -e ".[all]"
```

Activate the environment again in every new shell before running `onit`. To upgrade a
source install later: `pip install -e '.[all]' -U --upgrade-strategy eager` — and if
dependencies ever end up conflicting, recreate the environment from scratch.

### 2. Start a model server

Pick whichever matches your hardware. The model must support **tool calling** — OnIt
does its work through tools, so a model without it will talk but not act.

**Ollama** — simplest, runs anywhere:

```bash
OLLAMA_CONTEXT_LENGTH=131072 ollama serve   # raise the context; the 4096 default truncates agent turns
ollama pull qwen3:30b
```
→ host `http://localhost:11434/v1` (the `/v1` suffix is required)

**MLX** — Apple silicon:

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080
```
→ host `http://localhost:8080/v1`

**vLLM** — NVIDIA GPUs:

```bash
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000 \
  --max-model-len 262144 --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 --chat-template-content-format string
```
→ host `http://localhost:8000/v1`

No GPU at hand? Skip this step and use a hosted endpoint in step 3:
`https://openrouter.ai/api/v1` (OpenRouter) or `https://api.ollama.com` (Ollama cloud).

With local Ollama and MLX, **name the model explicitly** — auto-detection picks the
first entry the server lists, which is rarely the one you meant.

See [docs/MODEL_SERVING.md](docs/MODEL_SERVING.md) for API keys on vLLM, context-window
sizing, and running several servers with failover.

### 3. Keys

```bash
onit setup
```

The wizard asks for your model endpoint (host URL from step 2, plus the model name) and
the API keys below. Settings land in `~/.onit/config.yaml`; secrets go into your OS
keychain. Press Enter to skip anything you don't need — you can rerun it any time, and
`onit setup --show` prints what is currently set.

| Key | What it enables | Getting one |
|-----|-----------------|-------------|
| **Ollama API key** | **Web search.** The `search` tool uses the [Ollama web search API](https://ollama.com/blog/web-search); without a key it falls back to DuckDuckGo. The same key unlocks Ollama cloud models. | Free tier — sign in at [ollama.com](https://ollama.com) and create a key. Nothing needs to run locally: the key alone is enough for search. |
| **OpenWeatherMap** | The `get_weather` tool (current conditions and 5-day forecast). | Free — [openweathermap.org/api](https://openweathermap.org/api). |
| **vLLM API key** | Only if you started vLLM with `--api-key`. | Whatever you passed to `vllm serve`. |
| **OpenRouter** | Hosted models, if that is your endpoint. | [openrouter.ai](https://openrouter.ai/) (paid). |
| **GitHub token** | The `github_repo` tool (create/list/fork repos). | GitHub → Settings → Developer settings. |

Environment variables work too, if you'd rather not use the keychain:

```bash
export OLLAMA_API_KEY=...            # web search + Ollama cloud
export OPENWEATHERMAP_API_KEY=...    # weather
export VLLM_API_KEY=...              # vLLM with --api-key
export OPENROUTER_API_KEY=...        # OpenRouter
```

### 4. Run

```bash
onit
```

That's the text UI. MCP tools start automatically, the last session resumes, and the
agent works out of `~/sandbox` by default. Type `\bye` (or Ctrl+D) to leave.

```bash
onit --restart-session               # start fresh instead of resuming
onit --host http://localhost:11434/v1 --model qwen3:30b   # override the configured endpoint
onit --data-path ~/work              # work in a different directory
onit --think                         # reasoning mode, if the model supports it
onit --show-logs                     # show what the tools are doing
```

## Day to day

```bash
onit sessions                        # list saved sessions
onit sessions --tag abc123 "my-chat" # name one for easy recall
onit resume my-chat                  # continue a specific session
onit setup --show                    # review the current configuration
```

Files the agent reads and writes stay inside its working directory (`data_path`,
`~/sandbox` by default) — paths outside it are refused. Point it at a read-only folder
of your own documents to ask questions about them:

```bash
export ONIT_DOCUMENTS_PATH=~/company-docs
onit
> what is our vacation policy?
```

Full flag list: [docs/CLI.md](docs/CLI.md). Tool-by-tool reference: [docs/TOOLS.md](docs/TOOLS.md).

## Configuration

`onit setup` is enough for most setups. To edit by hand, `~/.onit/config.yaml`:

```yaml
serving:
  host: http://localhost:11434/v1
  model: qwen3:30b
  max_context_tokens: 131072   # set it when the server doesn't report its own
  think: true
  max_tokens: 32768

theme: white          # or "dark"
timeout: 600
data_path: ~          # working directory (default: ~/sandbox)
```

Priority order: CLI flags > environment variables > `~/.onit/config.yaml` > `--config FILE`.
Everything configurable — sampling parameters, multiple endpoints, answer fact-checking,
MCP server ports — is in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Beyond the terminal

| | |
|---|---|
| `onit serve web` | Browser chat UI with Google login ([docs/CLI.md](docs/CLI.md#onit-serve-web)) |
| `onit serve a2a` | [A2A protocol](https://a2a-protocol.org/) server; send tasks with `onit ask "…"` |
| `onit serve gateway` | Telegram or Viber bot ([docs/GATEWAY_QUICK_START.md](docs/GATEWAY_QUICK_START.md)) |
| `onit serve loop "task" --period 60` | Repeat a task on a timer |
| `onit --container` | Run the whole agent inside a hardened Docker container ([docs/DOCKER.md](docs/DOCKER.md)) |

## Documentation

- [Model Serving](docs/MODEL_SERVING.md) — vLLM, Ollama, MLX, OpenRouter, Ollama cloud, multi-endpoint failover
- [CLI Reference](docs/CLI.md) — every command and flag, including the `serve` modes
- [Configuration](docs/CONFIGURATION.md) — config file, environment variables, sampling, fact-checking
- [MCP Tools](docs/TOOLS.md) — the default tools, sandbox paths, notes, code execution
- [Local Search](docs/LOCAL_SEARCH.md) — indexing and searching in-house documents
- [Isolation Modes](docs/ISOLATION.md) — containers, command allowlisting, permission rules
- [Docker](docs/DOCKER.md) — `--container`, manual `docker run`, Compose stack, GPU pass-through
- [Web Authentication](docs/WEB_AUTHENTICATION.md) · [Web Deployment](docs/DEPLOYMENT_WEB.md) · [HTTPS](docs/HTTPS_DEPLOYMENT.md)
- [Gateway Quick Start](docs/GATEWAY_QUICK_START.md) — Telegram and Viber bots
- [Architecture](docs/ARCHITECTURE.md) · [Testing](docs/TESTING.md) · [Benchmarks](benchmarks/README.md)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
