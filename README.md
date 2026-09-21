# OnIt

*OnIt* — the AI is working on the given task and will deliver the results shortly.

OnIt is an agent harness: it hands a language model a set of tools, a working
directory, and a memory of what it has already done, then runs the loop until the task
is finished. The agent is the same whichever way you reach it — a terminal session, a
browser chat UI, an [A2A](https://a2a-protocol.org/) endpoint, or a Telegram or Viber
bot. Only the front end changes.

It runs on a model server you control — [vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang), [Ollama](https://ollama.com), or
[MLX](https://github.com/ml-explore/mlx-lm) on Apple silicon — or on a hosted endpoint
([OpenRouter](https://openrouter.ai/), [Ollama cloud](https://ollama.com)), and drives
its work through [MCP](https://modelcontextprotocol.io/) tools: web search, weather,
shell, file editing, and search over your own documents.

This README gets you to a first session in the **text UI** in about ten minutes, then
shows how to grow the same setup into a **web UI** deployment. The other front ends,
containers, and the full CLI and configuration reference are in [docs/](docs/).

## Quick Start

Three steps: install, point OnIt at a model, run.

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
cd onit && pip install -U --upgrade-strategy eager -e '.[all]'
```

This step is most of the ten minutes — the rest is two commands. Activate the
environment again in every new shell before running `onit`. To pick up newer commits
later: `git pull && pip install -U --upgrade-strategy eager -e '.[all]'` — and if
dependencies ever end up conflicting, recreate the environment from scratch.

### 2. Point OnIt at a model

OnIt talks to any OpenAI-compatible endpoint. Two hosted options need no hardware and
no model-serving setup — pick one:

**Ollama cloud — free tier, nothing to install:**

```bash
export OLLAMA_API_KEY=...   # sign in at ollama.com and create a key
onit --host https://api.ollama.com --model glm-5.3:cloud
```

**OpenRouter — one API, many models (paid per token):**

```bash
export OPENROUTER_API_KEY=...   # create at openrouter.ai → Keys
onit --host https://openrouter.ai/api/v1 --model google/gemini-2.5-pro
```

Browse models at [openrouter.ai/models](https://openrouter.ai/models). Either way, the
model must support **tool calling**: OnIt does its work through tools, so a model
without it will talk but not act.

Prefer to run the model on your own hardware? Local Ollama needs no key at all —
`ollama pull qwen3:30b`, then `onit --host http://localhost:11434/v1 --model qwen3:30b`
(the `/v1` suffix is required). vLLM, SGLang, and MLX work the same way — see
[docs/RUN_A_MODEL_SERVER.md](docs/RUN_A_MODEL_SERVER.md).

### 3. Run

```bash
onit
```

That launches the text UI. MCP tools start automatically, the agent works out of
`~/sandbox`, and `\bye` (or Ctrl+D) leaves. Running `onit` again picks the conversation
back up where you left it.

A first task — this is the shape of a session (output abbreviated):

```text
> check the weather in Manila, then write manila.md comparing it to Tokyo's

◆ search   get_weather(place="Manila") → 29°C, thunderstorms
◆ search   get_weather(place="Tokyo")  → 24°C, clear
◆ bash     writing manila.md
Done — manila.md is in the working directory. Manila is 5°C warmer and the
UV is high; Tokyo is dry and clear this week.
```

The agent plans, calls tools, narrates each step, and asks before anything
destructive. Files it writes land in `~/sandbox` (or the directory you started it in).

To make the endpoint stick so you never retype it:

```bash
onit setup   # endpoint URL, API key, model name
onit
```

`onit setup` walks through your model endpoints one at a time, asking three things about
each: its **URL**, an **API key** (leave blank if the server needs none), and the
**model name** — for the Ollama cloud example above, that is
`https://api.ollama.com`, your Ollama key, and `glm-5.3:cloud`. The key prompt does not
echo what you type, and the key goes into your OS keychain — never into `config.yaml`.
Press Enter to skip anything; type `a` at the `endpoints>` prompt to add a second
server. Rerun it any time, and `onit setup --show` prints what is currently set.

On a hosted endpoint, and with local Ollama or MLX, **name the model explicitly** —
auto-detection picks the first entry the server lists, which is rarely the one you meant.

### Commands inside the session

Lines starting with a backslash are answered by OnIt itself, without a model round trip:

| Command | What it does |
| --- | --- |
| `\help` | List the commands. |
| `\setup` | The endpoints, keys and paths this session is using. |
| `\model [name]` | Switch model. Bare, it lists what the endpoint serves; `\model -` restores auto-detect. |
| `\host [add \| rm] [url]` | List the endpoints, switch to one, add another, or drop one. |
| `\key [n \| url]` | Set an endpoint's API key, typed at a prompt that does not echo it. |
| `\save` | Write the session's endpoints to the config file so the next session starts with them. |
| `\bye` | End the session (also `\exit`, `\quit`, `\goodbye`). |

A `\host` or `\model` change lasts until you quit; `\save` writes the endpoint list
into `~/.onit/config.yaml` so it survives. A key set with `\key` is stored in the OS
keychain the moment you type it, and never goes into that file — so an endpoint you
added and keyed in a session needs a `\save` for the key to have a row to belong to.
Two things worth knowing:

```
\host add http://gpu-2:8000/v1         spread this session across both servers
\host add https://ollama.com --share   Ollama endpoints sit out as a standby while
                                       another endpoint is healthy; --share puts
                                       them in rotation on equal terms
```

Any other line goes to the model, so a message may still start with a backslash.

## Keys worth adding

None of these are needed to start — the Quick Start above already gives you a working
agent. Add them with `onit setup`, or as environment variables if you would rather skip
the keychain.

| Key | What it enables | Getting one |
|-----|-----------------|-------------|
| **GitHub token** (`GITHUB_TOKEN`) | **Automated git workflows.** `git clone`, `pull`, and `push` from the agent's shell, plus the `github_repo` tool — see [below](#github-and-hugging-face). | GitHub → Settings → Developer settings → Personal access tokens (`repo` scope). |
| **Ollama API key** (`OLLAMA_API_KEY`) | **Web search.** The `search` tool uses the [Ollama web search API](https://ollama.com/blog/web-search); without a key it falls back to DuckDuckGo. | Free tier — sign in at [ollama.com](https://ollama.com) and create a key. Nothing needs to run locally. |
| **OpenWeatherMap** (`OPENWEATHERMAP_API_KEY`) | The `get_weather` tool. Optional — free, and the agent works fine without it. | [openweathermap.org/api](https://openweathermap.org/api). |
| **Hugging Face token** (`HF_TOKEN`) | Model and dataset downloads in `--container` runs — see [below](#github-and-hugging-face). | [huggingface.co](https://huggingface.co) → Settings → Access Tokens. |

Model-server keys are not in this table — `onit setup` asks for each endpoint's key
beside its URL, so two servers with different keys are simply two entries. See
[docs/MODEL_SERVING.md](docs/MODEL_SERVING.md) for the full resolution order, running
several servers with failover, and context-window sizing.

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

The GitHub token is the one key that turns OnIt from a chatbot into a coding agent —
with it, the agent can clone, branch, commit, and push without you touching the
terminal. Store a personal access token with `repo` scope — `onit setup` (*GitHub
personal access token*), or `export GITHUB_TOKEN=...`. Two things switch on:

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

A typical delegation, once the key is stored:

```text
> clone github.com/sibyl-oracles/onit, fix the typo in README line 12,
  commit as "docs: fix typo", and push

◆ bash     git clone https://github.com/sibyl-oracles/onit.git
◆ bash     edit + git commit -m "docs: fix typo"
◆ bash     git push origin main
Pushed to main.
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

## From text UI to web UI

The text UI and the web UI are the same agent with the same config — the endpoint and
keys you set up above are exactly what the browser version uses. Going from a working
`onit` session to a working `http://localhost:9000` takes about five more minutes.

### The short path (local machine)

```bash
onit serve web        # http://localhost:9000
```

By default this **requires Google login** — every browser session starts with a
"Sign in with Google" screen, and each session is private to the account that created
it. Without OAuth credentials the server refuses to start. Two ways through:

- **Trusted network (fastest, no Google Console visit):** `onit serve web --no-login`.
  Anyone who can reach the port can use the agent — fine for `localhost`, not for a
  shared server.
- **Google login (needed for anything reachable by others):** three steps, below.

### Google login, step by step (~5 minutes)

1. **Create the OAuth client.** At
   [console.cloud.google.com](https://console.cloud.google.com/) create a project (any
   Google account, no billing), open **APIs & Services → Credentials → Create
   credentials → OAuth client ID**, choose **Web application**, and configure the
   consent screen when prompted (app name + support email; **External** audience; add
   yourself under **Test users** while the app is in *Testing*).

2. **Add the redirect URI.** Under **Authorized redirect URIs** add exactly:

   ```
   http://localhost:9000/auth/callback
   ```

   Google rejects any callback not on this list, character for character — a mismatch
   here is the `Error 400: redirect_uri_mismatch` sign-in failure. Each extra host or
   port you browse from needs its own entry.

3. **Store the credentials and launch.** Copy the client ID and secret, then:

   ```bash
   onit setup    # paste them at the Google OAuth2 client ID / secret prompts
   onit serve web
   ```

   The startup banner shows `OAuth2 authentication enabled`. Open the URL, sign in,
   and you land in the same chat UI — streaming markdown, tool status, session
   sidebar, file attachments.

Optionally restrict who may log in beyond the built-in Gmail/Workspace gate:

```yaml
web_allowed_emails:
  - alice@gmail.com
  - "*@sibyl.ai"
```

### Putting it on a server

Three differences from the local run, all covered in the linked docs:

- **HTTPS is required** for non-localhost Google sign-in — put OnIt behind a TLS
  reverse proxy ([docs/HTTPS_DEPLOYMENT.md](docs/HTTPS_DEPLOYMENT.md)), and register
  the `https://YOUR_HOST/auth/callback` redirect URI.
- **The full deployment path** — Docker Compose, environment files, reverse proxy —
  is walked through in [docs/DEPLOYMENT_WEB.md](docs/DEPLOYMENT_WEB.md).
- **Session isolation and command approvals** matter once other people can reach the
  agent — see [docs/ISOLATION.md](docs/ISOLATION.md) and
  [docs/WEB_AUTHENTICATION.md](docs/WEB_AUTHENTICATION.md).

## Other front ends

The same agent, the same sessions, the same tools — reached a different way:

| | |
|---|---|
| `onit serve web` | Browser chat UI with Google login — see [above](#from-text-ui-to-web-ui) |
| `onit serve a2a` | [A2A protocol](https://a2a-protocol.org/) server; send tasks with `onit ask "…"` |
| `onit serve gateway` | Telegram or Viber bot ([docs/GATEWAY_QUICK_START.md](docs/GATEWAY_QUICK_START.md)) |
| `onit serve loop "task" --period 60` | Repeat a task on a timer |
| `onit --container` | Run the whole agent inside a hardened Docker container ([docs/DOCKER.md](docs/DOCKER.md)) |

## Documentation

- [Run a Model Server](docs/RUN_A_MODEL_SERVER.md) — serving your own model with vLLM, SGLang, MLX, or Ollama
- [Model Serving](docs/MODEL_SERVING.md) — connecting OnIt to endpoints: keys, hosted providers, multi-endpoint failover
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