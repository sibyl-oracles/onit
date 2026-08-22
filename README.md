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

Install from the latest source — not the PyPI wheel, which lags behind:

```bash
git clone https://github.com/sibyl-oracles/onit.git
cd onit && pip install -e ".[all]"
```

Activate the environment again in every new shell before running `onit`. To pick up
newer commits later: `git pull && pip install -e '.[all]' -U --upgrade-strategy eager` —
and if dependencies ever end up conflicting, recreate the environment from scratch.

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
| **GitHub token** | `git push` from the agent's shell, plus the `github_repo` tool — see [below](#github-and-hugging-face). | GitHub → Settings → Developer settings → Personal access tokens (`repo` scope). |
| **Hugging Face token** | Model and dataset downloads in `--container` runs — see [below](#github-and-hugging-face). | [huggingface.co](https://huggingface.co) → Settings → Access Tokens. |

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

That's the text UI. MCP tools start automatically, the agent works out of `~/sandbox`,
and `\bye` (or Ctrl+D) leaves. Running `onit` again picks the conversation back up where
you left it.

## Day to day

A few flags worth knowing:

```bash
onit --restart-session   # forget the previous conversation and start clean
onit --think             # reasoning mode, if the model supports it
onit --show-logs         # show what the tools are doing
onit --data-path ~/work  # work in a directory other than ~/sandbox
```

The agent reads and writes only inside its working directory — paths outside it are
refused. To ask about documents you keep elsewhere, point it at a folder:

```bash
export ONIT_DOCUMENTS_PATH=~/company-docs
onit
> what is our vacation policy?
```

Full flag list, plus juggling several named sessions: [docs/CLI.md](docs/CLI.md).
Tool-by-tool reference: [docs/TOOLS.md](docs/TOOLS.md). Hand-editing
`~/.onit/config.yaml` instead of rerunning `onit setup`:
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## GitHub and Hugging Face

Two integrations worth having for ML work: one lets the agent push code, the other lets
it pull models and datasets.

### GitHub

Store a personal access token with `repo` scope — `onit setup` (*GitHub personal access
token*), or `export GITHUB_TOKEN=...`. Two things switch on:

- **`git` in the agent's shell.** OnIt writes a `GIT_ASKPASS` helper into the bash tool's
  environment and mirrors the token to `GH_TOKEN`, so `git clone`, `pull`, and `push`
  over HTTPS reach private repos without an interactive prompt.
- **The `github_repo` tool** — create, get, list, fork, and delete repositories through
  the API, no shelling out.

The shell is jailed to the working directory, so clone inside it — or point OnIt at the
project you're working on:

```bash
onit --data-path ~/projects/my-model
```

### Hugging Face

On the host, log in once with the Hub CLI. The token lands in `~/.cache/huggingface`, and
the agent's shell runs with your real `HOME`, so `hf download`, `transformers`, and
`datasets` all find it:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login          # older huggingface_hub: huggingface-cli login
```

A `--container` run cannot see your home directory. Store the token with `onit setup`
(*HuggingFace access token*) instead and the launcher bridges it in as `HF_TOKEN`. The
image also ships without the heavy ML packages — `onit-install-ml [torch|hf|extras|all]`
installs CUDA-matched wheels onto the persistent volume
([docs/DOCKER.md](docs/DOCKER.md)).

One container-only gotcha: the command allowlist is enforced there and carries `git` and
`git-lfs` but not `gh` or `hf`. Add what you need with
`ONIT_ALLOWED_COMMANDS=gh,hf,huggingface-cli` ([docs/ISOLATION.md](docs/ISOLATION.md)).

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
