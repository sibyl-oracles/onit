# CLI Reference

Every `onit` command and flag. The terminal chat is the default mode; the
`serve` subcommands run OnIt as a server (web UI, A2A, chat gateways, timers),
and `onit doctor` self-checks the stack.


## Interactive chat (default)

```bash
onit [OPTIONS]
```

Starts an interactive terminal chat with tool access. MCP servers start automatically.

| Flag | Description | Default |
|------|-------------|---------|
| `--config FILE` | Path to YAML configuration file | `configs/default.yaml` |
| `--host URL` | LLM serving host URL. Overrides config and `ONIT_HOST` | — |
| `--model NAME` | Model name. Skips auto-detection from endpoint | — |
| `--max-tokens N` | Max output tokens per model response. Accepts `32768`, `128k`, `1M`. Each request is still clamped to what is left of the context window. Overrides `serving.max_tokens` | `131072` |
| `--max-context-tokens N` | Context window size in tokens (`128k`, `1M`). Set it when the server does not report its own, or to hold the agent to a smaller window. On Ollama it also sizes `num_ctx` unless `serving.num_ctx` is set. Overrides `serving.max_context_tokens` | detected from endpoint, else `262144` |
| `--verbose` | Enable verbose logging | `false` |
| `--think` | Enable thinking/reasoning mode (CoT) | `false` |
| `--no-stream` | Disable token streaming | `false` |
| `--show-logs` | Show tool execution logs | `false` |
| `--resume TAG_OR_ID` | Resume a previous session by tag, UUID, or `last` | last session |
| `--restart-session` | Start a new session instead of resuming the last one (alias: `--new-session`) | `false` |
| `--data-path PATH` | Working directory for agent files. Overrides `data_path` in the config YAML | `~/sandbox` |
| `--auto` | Answer every command approval prompt with yes, so the run never stops to ask. On for your own runs; pass it to get the same on a `serve web`/`a2a`/gateway deployment. Only questions the policy chose to ask are answered — see [Command Approvals](ISOLATION.md#command-approvals) | on, except for deployments |
| `--no-auto` | Be asked about a command policy will not run on its own, instead of approving it automatically (alias: `--ask`) | `false` |
| `--unrestricted` | Unrestricted host filesystem access (trusted environments only) | `false` |
| `--container` | Run the entire OnIt process inside a hardened Docker container | `false` |
| `--mcp-sse URL` | Add an external MCP server (SSE transport, repeatable) | — |
| `--mcp-server URL` | Add an external MCP server (Streamable HTTP transport, repeatable) | — |

## `onit setup`

Interactive setup wizard, in three sections:

- **Model serving** — the endpoint editor (add/edit/delete servers and rank them by priority), the load balancing algorithm, and the API keys those endpoints use (OpenRouter, Ollama, vLLM)
- **Preferences** — theme, web UI port, request timeout
- **Integrations** — OpenWeatherMap, Telegram, Viber, Google OAuth2, GitHub, HuggingFace

Settings go to `~/.onit/config.yaml`, secrets to the OS keychain.

Leave the model name blank to auto-detect it from the endpoint (first available model). Set it explicitly for Ollama cloud (e.g. `glm-5.3:cloud`), OpenRouter (e.g. `google/gemini-2.5-pro`), or a local Ollama/MLX server hosting several models, where auto-detection would pick an arbitrary model. Press Enter to keep a value, type `-` to clear it. The wizard warns when an Ollama cloud or OpenRouter endpoint is missing its API key or model name.

See [Multiple model endpoints](MODEL_SERVING.md#multiple-model-endpoints) for the endpoint editor and how priority routing works.

```bash
onit setup           # run the wizard
onit setup --show    # print current configuration
```

## `onit sessions`

List and manage saved sessions.

```bash
onit sessions                          # list recent sessions (default: 20)
onit sessions --limit 50               # list up to 50 sessions
onit sessions --tag abc123 "my-chat"   # tag a session for easy recall
onit sessions --rebuild                # rebuild session index from JSONL files
onit sessions --clear                  # delete all session history
```

## `onit resume`

Resume a previous session by tag or UUID. Terminal chat already resumes the most
recent session automatically, so this is for picking a *different* one.

```bash
onit resume my-chat      # resume by tag
onit resume abc123       # resume by session UUID prefix
onit resume              # resume the most recent session (same as bare `onit`)
```

Equivalent to `onit --resume TAG_OR_ID`.

To start from scratch instead, use `onit --restart-session`. Server modes
(`serve web`, `serve a2a`, `serve gateway`, `serve loop`) manage their own
sessions and never auto-resume.

A resumed session carries more than the conversation. Alongside each
`<session_id>.jsonl` is a `<session_id>.state.json` recording what the session
*did* — which tools it ran and how many times, how many turns it spent, and
whether the last attempt finished or stopped early at a limit. Resuming reads
it back and tells the agent, so a continued session builds on work that already
succeeded instead of starting it again. It is deleted with the session.

## `onit ask`

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

## `onit doctor`

Run the live self-check battery from the shell — the same checks `\doctor`
runs in the text UI, against a throwaway session that is deleted afterwards.
Use it after pulling changes, in update scripts, or in CI: it exits **0 when
every check passed and 1 when any check failed**, so it can gate a deploy.

```bash
onit doctor                 # 13 fast checks, a few seconds
onit doctor --deep          # + live model reply and a tool-calling turn (costs tokens)
onit doctor --json          # machine-readable report (stdout is pure JSON)
onit doctor --keep-session  # keep the throwaway session for inspecting a failure
```

The battery starts this process's own MCP servers on freshly allocated ports,
so it runs beside a live session without touching it, and it works on a
machine that has never been set up — reporting *what is missing* (no
`serving.host`, no MCP servers) instead of crashing. See
[TESTING.md](TESTING.md) for what each check covers.

```bash
# after an update: is this checkout's stack sound?
onit doctor || echo "self-check failed"

# in a script: refuse to continue on a broken stack
onit doctor --json > doctor.json || exit 1
```

| Argument / Flag | Description | Default |
|-----------------|-------------|---------|
| `--deep` | Also exercise a live model reply and a full tool-calling turn | off |
| `--json` | Print the report as JSON instead of text | off |
| `--keep-session` | Keep the throwaway session (visible in `onit sessions`) | off |
| `--config`, `--host`, `--model`, `--data-path` | Shared flags, same meaning as the interactive run | — |

## `onit serve`

Run OnIt in a persistent server or daemon mode. All serve modes run indefinitely until interrupted (Ctrl+C).

### `onit serve a2a`

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

### `onit serve web`

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

**Google Analytics (optional).** Set `web_ga_measurement_id: G-XXXXXXXXXX`
in the config (or the `ONIT_GA_MEASUREMENT_ID` env var — handy in the
docker-compose `.env`) and the web UI loads the GA4 gtag snippet for
authenticated users. The measurement ID is withheld from the public,
pre-login `/api/config` so it isn't exposed to anonymous visitors.
Analytics is off when unset.

#### Setting up Google OAuth2 (step by step)

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

More detail (session lifetime, troubleshooting): [WEB_AUTHENTICATION.md](WEB_AUTHENTICATION.md).

### `onit serve gateway`

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

### `onit serve loop`

Repeat a task on a configurable timer. Useful for monitoring, polling, or autonomous scheduled work.

```bash
onit serve loop "check the weather in Manila" --period 60
onit serve loop "summarize today's news" --period 3600
```

| Argument / Flag | Description | Default |
|-----------------|-------------|---------|
| `task` (positional) | Task to execute repeatedly | required |
| `--period SECONDS` | Seconds between iterations | `10` (or `period` in config) |

