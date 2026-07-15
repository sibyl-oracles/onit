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

To update an existing source install and upgrade all dependencies to their latest compatible versions:

```bash
pip install -e '.[all]' -U --upgrade-strategy eager
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

## CLI at a Glance

```
onit                                          # interactive terminal chat
onit setup                                    # configure LLM endpoint, API keys
onit resume [TAG_OR_ID]                       # continue a previous session
onit sessions                                 # list saved sessions

onit serve a2a                                # A2A protocol server (port 9001)
onit serve web                                # web UI (port 9000)
onit serve gateway [telegram|viber|auto]      # Telegram or Viber bot
onit serve loop "task" --period 60            # repeat a task on a timer

onit ask "what is the weather in Manila"      # send a task to a running A2A server

onit --container                              # run in a hardened Docker container
onit --sandbox                                # delegate code execution to a sandbox
onit --unrestricted                           # unrestricted host filesystem access
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
  max_tokens: 32768   # max output tokens per response (fits any single answer)
  # Sampling parameters (all optional — sensible defaults apply):
  # temperature: 1.0
  # top_p: 0.95
  # top_k: 20
  # presence_penalty: 1.5
  # repetition_penalty: 1.0

verbose: false
timeout: 600
sandbox: false

web_port: 9000
a2a_port: 9001

theme: white         # or "dark"
topic: ~             # default topic context, e.g. "machine learning"
template_path: ~     # custom prompt template YAML
data_path: ~         # working directory for file operations (default: system temp)

mcp:
  servers:
    - name: PromptsMCPServer
      url: http://127.0.0.1:18200/sse
      enabled: true
    - name: ToolsMCPServer
      url: http://127.0.0.1:18201/sse
      enabled: true
```

### Sampling parameters

Sampling parameters (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`) are set in `configs/default.yaml` under `serving:`. They are not exposed as CLI flags to keep the command line clean.

**Recommended parameters for Qwen3.5:**

| Mode | Use case | `temperature` | `top_p` | `top_k` | `presence_penalty` |
|------|----------|:---:|:---:|:---:|:---:|
| Thinking (`think: true`) | General | `1.0` | `0.95` | `20` | `1.5` |
| Thinking (`think: true`) | Precise coding | `0.6` | `0.95` | `20` | `0.0` |
| Instruct (no think) | General | `0.7` | `0.8` | `20` | `1.5` |
| Instruct (no think) | Reasoning | `1.0` | `1.0` | `40` | `2.0` |

Set `repetition_penalty: 1.0` in all cases.

## CLI Reference

### Interactive chat (default)

```bash
onit [OPTIONS]
```

Starts an interactive terminal chat with tool access. MCP servers start automatically.

| Flag | Description | Default |
|------|-------------|---------|
| `--config FILE` | Path to YAML configuration file | `configs/default.yaml` |
| `--host URL` | LLM serving host URL. Overrides config and `ONIT_HOST` | — |
| `--model NAME` | Model name. Skips auto-detection from endpoint | — |
| `--verbose` | Enable verbose logging | `false` |
| `--think` | Enable thinking/reasoning mode (CoT) | `false` |
| `--no-stream` | Disable token streaming | `false` |
| `--show-logs` | Show tool execution logs | `false` |
| `--resume TAG_OR_ID` | Resume a previous session by tag, UUID, or `last` | — |
| `--sandbox` | Delegate code execution to an external MCP sandbox provider | `false` |
| `--unrestricted` | Unrestricted host filesystem access (trusted environments only) | `false` |
| `--container` | Run the entire OnIt process inside a hardened Docker container | `false` |
| `--mcp-sse URL` | Add an external MCP server (SSE transport, repeatable) | — |
| `--mcp-server URL` | Add an external MCP server (Streamable HTTP transport, repeatable) | — |

### `onit setup`

Interactive setup wizard. Configures the LLM endpoint, API keys, and preferences. Stores settings in `~/.onit/config.yaml` and secrets in the OS keychain.

```bash
onit setup           # run the wizard
onit setup --show    # print current configuration
```

### `onit sessions`

List and manage saved sessions.

```bash
onit sessions                          # list recent sessions (default: 20)
onit sessions --limit 50               # list up to 50 sessions
onit sessions --tag abc123 "my-chat"   # tag a session for easy recall
onit sessions --rebuild                # rebuild session index from JSONL files
onit sessions --clear                  # delete all session history
```

### `onit resume`

Resume a previous session by tag or UUID.

```bash
onit resume              # resume the most recent session
onit resume my-chat      # resume by tag
onit resume abc123       # resume by session UUID prefix
```

Equivalent to `onit --resume TAG_OR_ID`.

### `onit ask`

Send a single task to a running OnIt A2A server and print the response. Useful for scripting, pipelines, or one-shot queries without starting a local agent.

```bash
onit ask "what is the weather in Manila"
onit ask "summarize this document" --file report.pdf
onit ask "describe this image" --image photo.jpg
onit ask "write a script" --server http://192.168.1.10:9001
```

| Argument / Flag | Description | Default |
|-----------------|-------------|---------|
| `task` (positional) | Task to send to the server | required |
| `--file PATH` | File to upload along with the task | — |
| `--image PATH` | Image file for vision processing (model must be a VLM) | — |
| `--server URL` | A2A server URL | `http://localhost:9001` |

### `onit serve`

Run OnIt in a persistent server or daemon mode. All serve modes run indefinitely until interrupted (Ctrl+C).

#### `onit serve a2a`

Run OnIt as an [A2A protocol](https://a2a-protocol.org/) server so other agents or clients can send tasks.

```bash
onit serve a2a                 # listen on port 9001 (default)
onit serve a2a --port 9100     # custom port
```

| Flag | Description | Default |
|------|-------------|---------|
| `--port PORT` | A2A server port | `9001` (or `a2a_port` in config) |

The agent card is available at `http://localhost:9001/.well-known/agent.json`.

**Send a task from another agent (Python A2A SDK):**

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

#### `onit serve web`

Launch the web chat UI — a FastAPI server that streams agent output over
Server-Sent Events into a modern chat interface (streaming markdown, tool
status, session sidebar, file attachments, light/dark theme).

```bash
onit serve web                 # open on port 9000 (default)
onit serve web --port 9500     # custom port
onit serve web --no-login      # skip Google login (open access — see below)
```

| Flag | Description | Default |
|------|-------------|---------|
| `--port PORT` | Web UI port | `9000` (or `web_port` in config) |
| `--no-login` | Run without requiring Google login | login required |

By default the web UI **requires Google login**: every session starts with a
Google OAuth2 sign-in, and only Google-hosted mail accounts are accepted —
Gmail (`@gmail.com` / `@googlemail.com`) or any Google Workspace domain
(i.e. any domain whose mail is hosted by Google). Each chat session is
private to the account that created it.

Without configured OAuth credentials, `onit serve web` refuses to start.
To run an open UI without login (e.g. local development on a trusted
network), pass `--no-login` or set `web_require_auth: false` in the config.
Anyone who can reach the port can then use the agent.

##### Setting up Google OAuth2 (step by step)

1. **Create a Google Cloud project.** Go to
   [console.cloud.google.com](https://console.cloud.google.com/), open the
   project selector (top-left) → **New Project**, give it a name (e.g.
   "OnIt Web"), and create it. Any Google account works; no billing needed.

2. **Configure the OAuth consent screen.** Navigate to **APIs & Services →
   OAuth consent screen** (newer consoles call this **Google Auth Platform →
   Branding**). Set the app name and support email, then choose the audience:
   - **External** — any Google account may attempt login (OnIt still rejects
     accounts that are not Gmail/Workspace-hosted). While the app's status is
     *Testing*, only accounts you add under **Audience → Test users** can log
     in; click **Publish app** to lift that limit.
   - **Internal** — available only on Google Workspace accounts; Google
     itself restricts login to your Workspace domain.

   No scope configuration is needed — OnIt only uses the basic
   `openid email profile` identity scopes.

3. **Create the OAuth client.** Navigate to **APIs & Services → Credentials →
   + Create credentials → OAuth client ID**. Choose application type
   **Web application** and name it (e.g. "OnIt Web UI").

4. **Add the authorized redirect URI.** Under **Authorized redirect URIs**,
   add one entry per host you will open the UI from, exactly matching:

   ```
   http://localhost:9000/auth/callback
   http://YOUR_SERVER_IP:9000/auth/callback
   ```

   Adjust the port if you use `--port`. Google rejects any callback not on
   this list, character for character. Non-localhost hosts require `https`
   URIs — put OnIt behind a TLS reverse proxy for public deployments.

5. **Copy the credentials.** After clicking **Create**, Google shows the
   **Client ID** (ends in `.apps.googleusercontent.com`) and the
   **Client secret** (starts with `GOCSPX-`). Copy both.

6. **Store them in OnIt.** Run `onit setup` and paste the values at the
   *Google OAuth2 client ID* and *client secret* prompts — they are stored
   in the OS keychain, not in a file. Alternatively set the
   `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` environment variables, or put
   `web_google_client_id` / `web_google_client_secret` in the config YAML.
   Verify with `onit setup --show`.

7. **(Optional) Restrict who may log in.** Beyond the built-in
   Gmail/Workspace gate, list exact addresses or whole domains in the config:

   ```yaml
   web_allowed_emails:
     - alice@gmail.com
     - "*@sibyl.ai"
   ```

8. **Launch and test.** Run `onit serve web` — the startup banner shows
   `OAuth2 authentication enabled`. Open `http://localhost:9000`, click
   **Sign in with Google**, and pick an account. You should land back in the
   chat, with your email and a Logout link shown in the UI.

More detail (session lifetime, troubleshooting): [docs/WEB_AUTHENTICATION.md](docs/WEB_AUTHENTICATION.md).

#### `onit serve gateway`

Run OnIt as a Telegram or Viber bot. Configure bot tokens via `onit setup` or environment variables.

```bash
onit serve gateway                                      # auto-detect from env vars
onit serve gateway telegram                             # Telegram bot
onit serve gateway viber --webhook-url https://...      # Viber bot
```

| Argument / Flag | Description | Default |
|-----------------|-------------|---------|
| `gateway_type` (positional) | `telegram`, `viber`, or `auto` | `auto` |
| `--webhook-url URL` | Public HTTPS URL for Viber webhook (or set `VIBER_WEBHOOK_URL`) | — |
| `--port PORT` | Local port for Viber webhook server | `8443` (or `viber_port` in config) |

Required environment variables (set via `onit setup` or export):
- Telegram: `TELEGRAM_BOT_TOKEN`
- Viber: `VIBER_BOT_TOKEN`, `VIBER_WEBHOOK_URL`

Install gateway dependencies if not using `[all]`:

```bash
pip install "onit[gateway]"
```

#### `onit serve loop`

Repeat a task on a configurable timer. Useful for monitoring, polling, or autonomous scheduled work.

```bash
onit serve loop "check the weather in Manila" --period 60
onit serve loop "summarize today's news" --period 3600
```

| Argument / Flag | Description | Default |
|-----------------|-------------|---------|
| `task` (positional) | Task to execute repeatedly | required |
| `--period SECONDS` | Seconds between iterations | `10` (or `period` in config) |

## Isolation Modes

OnIt offers three isolation levels. They can be combined (e.g. `--container --sandbox`).

### `--sandbox`

Delegates individual code-execution tool calls to an external MCP sandbox provider. Complementary to `--container`.

```bash
onit --sandbox
onit --container --sandbox   # defense in depth
```

Requires an MCP server that provides sandbox tools (`sandbox_run_code`, `sandbox_install_packages`, `sandbox_stop`). Set `sandbox: true` in `config.yaml` to enable by default.

### `--container`

Runs the entire OnIt process inside a hardened Docker container so a breach cannot reach the host OS.

```bash
onit --container                                          # interactive terminal in container
onit --container serve web                                # web UI, port 9000 published
onit --container serve a2a --port 9100                    # A2A server on custom port
onit --container --container-gpus all                     # NVIDIA GPU pass-through
onit --container --container-mount "$HOME/docs:/home/onit/documents:ro" \
  serve web                                               # expose host path read-only
onit --container --sandbox                                # combine with per-tool sandboxing
```

The first run auto-builds the `onit:local` image from the repo `Dockerfile`. Subsequent runs reuse the image.

**Container sub-flags:**

| Flag | Description |
|------|-------------|
| `--container-gpus SPEC` | NVIDIA GPU pass-through (e.g. `all`, `"device=0,1"`). Requires NVIDIA Container Toolkit. |
| `--container-mount HOST:CONTAINER[:ro]` | Extra bind mount. Repeatable. Prefer `:ro`. |
| `--container-memory SIZE` | Hard memory cap (e.g. `16g`). Default: unlimited. |
| `--container-shm-size SIZE` | `/dev/shm` size (default: `4g`). Raise for PyTorch DataLoader. |
| `--container-tmp-size SIZE` | `/tmp` tmpfs size (default: `16g`). Backed by host RAM. |
| `--container-allow-installs` | Permit package installs in-container. Installs must still be version-pinned (`pip install name==1.2.3`). |

**Isolation posture:** non-root user, read-only rootfs (`--read-only`), `--cap-drop=ALL`, `no-new-privileges` (no sudo/setuid escalation), RAM-backed tmpfs for all ephemeral writes (`/tmp`, `~/.cache`, `~/.onit`), no host mounts by default, outbound network allowed. Persistent state (pip installs via `PIP_TARGET`, Hugging Face caches, session artifacts) lives on the named `onit-data` volume — never the rootfs. The AST command allowlist (below) is enforced by default inside the container.

**What crosses the boundary:**

| Resource | Default behavior |
|---|---|
| `~/.onit/config.yaml` | Bind-mounted read-only |
| Host keychain secrets | Passed as ephemeral env vars |
| Session data | Named volume `onit-data` (writable, persistent) |
| Ports | Published only for the active mode |
| Host filesystem | Nothing beyond config/secrets unless `--container-mount` is set |

**Published ports by mode:**

| Mode | Default port | Override |
|---|---|---|
| (terminal) | — (no ports) | — |
| `serve web` | `9000:9000` | `--port` |
| `serve a2a` | `9001:9001` | `--port` |
| `serve gateway viber` | `8443:8443` | `--port` |

See [docs/DOCKER.md](docs/DOCKER.md) for full details.

### `--unrestricted`

Runs OnIt with lifted filesystem restrictions on the host — the agent can read/write any path, use any working directory, and install packages freely (pip, apt, brew, etc.). Use only in trusted, isolated environments.

```bash
onit --unrestricted
```

Catastrophic commands (disk wipe, reboot, kernel module loading) are always blocked regardless of this flag, and an explicit `ONIT_COMMAND_ALLOWLIST=1` still enforces the AST command allowlist.

### Command Permission Rules

The bash tool honors optional allow/deny rules from `~/.onit/settings.json` (override the path with the `ONIT_SETTINGS` env var). **These rules apply to the web UI only** (`onit serve web`) — web sessions may be reachable by other users, so the configured restrictions must hold there. The local text UI is a trusted terminal session and ignores the default settings file, running with full privileges under the built-in policy. To enforce the rules in the text UI too, point `ONIT_SETTINGS` at the file explicitly. Rules use glob patterns matched against the command; deny always wins, and compound commands (`&&`, `;`, `|`) are checked segment by segment:

```json
{
  "permissions": {
    "allow": ["Bash(*)"],
    "deny": [
      "Bash(sudo *)",
      "Bash(npm install*)",
      "Bash(pip install*)",
      "Bash(brew install*)"
    ]
  }
}
```

- **deny** — commands matching any rule are refused.
- **allow** — when non-empty, every command (and each segment of a compound command) must match an allow rule. Leave it as `["Bash(*)"]` (or omit it) to only use the deny list.

When active (web UI, or explicit `ONIT_SETTINGS`), rules apply in **all** modes, including `--container` and `--unrestricted`, and file edits take effect without a restart. Non-`Bash(...)` rules are ignored.

### Command Allowlisting (AST-based)

On top of the glob rules, the bash tool can enforce a **command allowlist backed by real shell parsing**: every command string is parsed into an AST (pipelines, `&&`/`||`/`;` lists, loops, subshells, `$(...)`/backtick substitutions, `bash -c` payloads, `find -exec` targets), and **every executable found anywhere in the tree** must be on the allowlist. Wrapper commands (`env`, `nohup`, `timeout`, `nice`, `stdbuf`, `xargs`) are peeled off so they can't hide a payload, and dynamic command names (`$CMD`, `$(which x)`) are rejected outright. The parser **fails closed**: anything it cannot statically analyze (`case` statements, function definitions, arithmetic commands) is blocked.

| Env var | Effect |
|---|---|
| `ONIT_COMMAND_ALLOWLIST` | `1` = enforce everywhere, `0` = disable. Unset: enforced inside `--container`, off on the host. |
| `ONIT_ALLOWED_COMMANDS` | Comma-separated extra executables to allow (e.g. `mytool,deno`). |
| `ONIT_ALLOW_PACKAGE_INSTALL` | `1` = permit package-manager installs (pinned versions only). Set by `--container-allow-installs`. |
| `ONIT_CONTAIN_THRESHOLD` | Blocked commands before auto-containment (default `5`, `0` disables). |

The allowlist can also be extended in `settings.json` (read in the web UI, or when `ONIT_SETTINGS` is set explicitly):

```json
{
  "permissions": {
    "allowedCommands": ["mytool", "deno"]
  }
}
```

**Package managers are blocked by default** under allowlist enforcement. System package managers (`apt`, `yum`, `dnf`, `pacman`, `brew`, `apk`, `snap`) are never allowlisted — in-container the rootfs is read-only anyway. Language package managers (`pip`, `npm`, `gem`, `cargo`, `go`, `uv`, `pipx`) may run non-mutating subcommands (`pip list`, `npm ls`), but `install` requires `ONIT_ALLOW_PACKAGE_INSTALL=1` **and pinned versions**:

```bash
pip install requests==2.31.0     # OK (with installs enabled)
pip install requests             # blocked: not pinned
pip install -r requirements.txt  # blocked: cannot pin-verify
npm install left-pad@1.3.0       # OK
npx cowsay@1.5.0                 # OK (pinned one-off execution)
```

Lockfile-driven installs (`npm ci`, bare `npm install`) are allowed since versions come from the lockfile. `onit-install-ml` (the curated CUDA-matched ML installer) is allowlisted only when installs are enabled.

### Auto-Containment

Policy violations (blocked commands) are counted per server process. When the count reaches `ONIT_CONTAIN_THRESHOLD` (default 5), the bash MCP server **auto-contains**:

- `bash`, `serve start`, `write_file`, `edit_file`, `transform_text`, and `send_file` refuse all further calls;
- every `serve`-managed background process is stopped;
- a marker file (`.onit-containment.json`, containing the violation log) is written to the data directory so containment **survives restarts**.

Read-only tools (`read_file`, `search_*`) keep working so the session can be diagnosed. To lift containment, delete the marker file and restart the MCP server.

## MCP Tool Integration

MCP servers start automatically. Tools are auto-discovered and available to the agent.

| Server | Description |
|--------|-------------|
| PromptsMCPServer | Prompt templates for instruction generation |
| ToolsMCPServer | Web search, local search, bash commands, file operations, and document tools |

Connect to additional external MCP servers:

```bash
onit --mcp-sse http://localhost:8080/sse
onit --mcp-server http://localhost:8080/mcp
```

## Local Search over In-House Data

OnIt includes a local search toolkit modeled on the [Mistral Search Toolkit](https://mistral.ai/news/search-toolkit/): a composable pipeline that unifies **ingestion** (parse → chunk → embed/index) and **retrieval** (BM25 sparse, dense embeddings, hybrid fusion) behind a single interface. Everything runs on your own infrastructure — documents, index, and embeddings never leave your machine, so the agent can answer questions from private company data that web search cannot see.

| Format | Extension | Parser |
|--------|-----------|--------|
| PDF | `.pdf` | pypdf (per page) |
| Markdown | `.md`, `.markdown` | built-in |
| Text / CSV | `.txt`, `.text`, `.csv` | built-in |
| Word | `.docx` | python-docx (paragraphs and tables) |
| Excel | `.xlsx`, `.xlsm` | openpyxl (per sheet) |

### Quick start

```bash
# 1. Install the optional parsers for Word and Excel (PDF/md/txt work out of the box)
pip install "onit[search]"

# 2. Point OnIt at your document folder
export ONIT_DOCUMENTS_PATH=~/company-docs

# 3. Run and ask questions about your data
onit
> what is our vacation policy?
```

The agent uses two MCP tools, registered automatically in the ToolsMCPServer:

| Tool | Description |
|------|-------------|
| `index_documents` | Ingest a directory: parse, chunk (default 1600 chars, 200 overlap), and index. Incremental — unchanged files are skipped, deleted files are dropped. Use `rebuild: true` to start fresh or `status_only: true` for index statistics. |
| `local_search` | Query the index and return ranked chunks with source file and location (page, sheet, table). Auto-ingests the default corpus on first use. |

### Retrieval methods

`local_search` supports three methods, selected with the `method` argument:

| Method | How it works | Requires |
|--------|--------------|----------|
| `bm25` | Okapi BM25 sparse lexical ranking (pure Python) | nothing |
| `dense` | Cosine similarity over chunk embeddings | an embedding endpoint |
| `hybrid` *(default)* | Reciprocal rank fusion of BM25 + dense rankings | falls back to `bm25` when no embedding endpoint is configured |

Dense and hybrid retrieval use any **OpenAI-compatible** `/embeddings` endpoint — a private vLLM or Ollama server keeps everything on-premises:

```bash
export ONIT_EMBEDDING_HOST=http://localhost:8000/v1   # vLLM, Ollama, etc.
export ONIT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
export ONIT_EMBEDDING_API_KEY=...                     # only if the endpoint needs one
```

When these are set, `index_documents` embeds chunks during ingestion and `local_search` embeds the query at search time. Without them, everything still works with BM25 — no network calls are made.

### How it works

```
Ingestion:  documents → parse (pdf/md/txt/csv/docx/xlsx) → chunk → [embed] → index
Retrieval:  query → BM25 ranking ─┐
            query → [dense ranking] ─┴→ reciprocal rank fusion → top-k chunks + sources
```

- The index is a single JSON file at `data_path/local_search/index.json` (owner-only permissions). Delete it or pass `rebuild: true` to re-ingest from scratch.
- Corpus directories must be inside `ONIT_DOCUMENTS_PATH` or `data_path` — the same filesystem sandbox that governs all OnIt file tools (relaxed inside `--container`).
- Set `ONIT_DISABLE_LOCAL_SEARCH=1` to unregister both tools.

### Adding a new document format

Parsers follow a small adapter interface: each returns a list of `(location, text)` blocks (e.g. `("page 3", ...)`, `("sheet Sales", ...)`). To support a new format, add a parser to `src/mcp/servers/tasks/local/search/toolkit.py`, register its extension in `SUPPORTED_EXTENSIONS`, and dispatch it from `parse_document()` — chunking, indexing, and retrieval pick it up automatically.

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

[Ollama cloud](https://ollama.com) hosts models accessed via the native [Ollama Python SDK](https://github.com/ollama/ollama-python). Store your API key once:

```bash
onit setup   # enter your Ollama API key when prompted
```

Or set the environment variable:

```bash
export OLLAMA_API_KEY=your-ollama-key
```

Then point OnIt at the Ollama cloud host and specify a model:

```bash
onit --host https://api.ollama.com --model glm-5.1:cloud
onit --host https://api.ollama.com --model gemma4:31b-cloud
onit --host https://api.ollama.com --model llama4:scout-cloud
```

Enable thinking mode (if supported by the model):

```bash
onit --think --host https://api.ollama.com --model glm-5.1:cloud
```

Model is auto-detected from the endpoint if `--model` is omitted. You can also set the host permanently in your config:

```yaml
serving:
  host: https://api.ollama.com
  model: glm-5.1:cloud
```

> **Note:** Ollama cloud uses the `ollama_api_key` keyring entry (the same key used for the web search tool).

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
│   │   ├── api.py              # FastAPI + SSE web UI
│   │   ├── static/             # Web UI assets (no build step)
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
- [Benchmarks](benchmarks/README.md) — Capability benchmark suite (Inspect AI) and results

## TODO

- [x] Integrate a [Mistral Search Toolkit](https://mistral.ai/news/search-toolkit/)–style pipeline for knowledge/RAG search over in-house data. See [Local Search over In-House Data](#local-search-over-in-house-data).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
