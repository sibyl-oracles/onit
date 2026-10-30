# CLI Reference

Every `onit` command and flag. The terminal chat is the default mode; the
`serve` subcommands run OnIt as a server (web UI, timers),
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
| `--max-tokens N` | Max output tokens per model response. Accepts `32768`, `128k`, `1M`. Each request is still clamped to what is left of the context window. Overrides `serving.max_tokens` | `32768` |
| `--max-context-tokens N` | Context window size in tokens (`128k`, `1M`). Set it when the server does not report its own, or to hold the agent to a smaller window. On Ollama it also sizes `num_ctx` unless `serving.num_ctx` is set. Overrides `serving.max_context_tokens` | detected from endpoint, else `262144` |
| `--verbose` | Enable verbose logging | `false` |
| `--think` | Enable thinking/reasoning mode (CoT) | `false` |
| `--show-logs` | Show tool execution logs | `false` |
| `--new-session` | Start a fresh session instead of continuing the last one (alias: `--restart-session`) | `false` |
| `--resume-session NAME_OR_ID` | Continue a particular session: its name, a UUID, or `last` (alias: `--resume`) | last session |
| `--name-session NAME` | Name a new session as it starts, so `--resume-session NAME` finds it. Implies `--new-session` (alias: `--name`) | — |
| `--session-limit N` | How many rows `onit sessions` lists, so a `#` means the same session here | `20` |
| `--data-path PATH` | Working directory for agent files. Overrides `data_path` in the config YAML | `~/sandbox` |
| `--auto` | Answer every command approval prompt with yes, so the run never stops to ask. On for your own runs; pass it to get the same on a `serve web` deployment. Only questions the policy chose to ask are answered — see [Command Approvals](ISOLATION.md#command-approvals) | on, except for deployments |
| `--no-auto` | Be asked about a command policy will not run on its own, instead of approving it automatically (alias: `--ask`) | `false` |
| `--unrestricted` | Unrestricted host filesystem access (trusted environments only) | `false` |
| `--container` | Run the entire OnIt process inside a hardened Docker container | `false` |

## `onit setup`

Interactive setup wizard, in three sections:

- **Model serving** — the endpoint editor (add/edit/delete servers and rank them by priority), the load balancing algorithm, and the API keys those endpoints use (OpenRouter, Ollama, vLLM)
- **Preferences** — theme, web UI port, request timeout
- **Integrations** — OpenWeatherMap, Google OAuth2, GitHub, HuggingFace

Settings go to `~/.onit/config.yaml`, secrets to the OS keychain.

Leave the model name blank to auto-detect it from the endpoint (first available model). Set it explicitly for Ollama cloud (e.g. `glm-5.3:cloud`), OpenRouter (e.g. `google/gemini-2.5-pro`), or a local Ollama/MLX server hosting several models, where auto-detection would pick an arbitrary model. Press Enter to keep a value, type `-` to clear it. The wizard warns when an Ollama cloud or OpenRouter endpoint is missing its API key or model name.

See [Multiple model endpoints](MODEL_SERVING.md#multiple-model-endpoints) for the endpoint editor and how priority routing works.

```bash
onit setup           # run the wizard
onit setup --show    # print current configuration
```

## The session flags

Every flag about which conversation a run is about is declared together and
ends in `-session`, so they read as one family rather than three flags
scattered through `--help`:

```bash
onit --new-session                             # fresh session
onit --new-session --name-session "chapter 3"  # fresh session, named
onit --resume-session "chapter 3"              # continue that one
onit --resume-session 3                        # ... by row number from the listing
onit --resume-session last                     # ... or the most recent
```

The older spellings — `--restart-session`, `--resume`, `--name` — still parse.
They are hidden from `--help`, because a flag people have in muscle memory is
a flag that cannot simply be removed, but there is now one way to write each.

Giving `--name-session` implies `--new-session`: a name belongs to a session
that is being created, so naming one and resuming another in the same command
is refused with a pointer to `onit sessions --rename`.

## `onit sessions`

List and manage saved sessions.

```bash
onit sessions                                   # list recent sessions (default: 20)
onit sessions --limit 50                        # list up to 50 sessions
onit sessions --include-empty                   # also show sessions never used
onit sessions --rename 3 "my-chat"              # rename row 3 (same as `onit rename`)
onit sessions --rebuild-index                   # rebuild the index from JSONL files
onit sessions --delete-all                      # delete all session history
```

The `#` column is the session's handle: every session action takes it, so a
session can be acted on without copying a UUID out of the terminal. Sessions
that were opened and never used are left out (they have no name and nothing to
return to); `--include-empty` shows them.

A session is named automatically from its first message, but a message that
carries no topic -- "hi", "thanks" -- names nothing, and the next real message
does it instead. That keeps a session from being called `hi` forever.

## `onit rename`

Name a session so it can be found later.

```bash
onit sessions --rename 3 "thesis chapter 3"     # by row number from the listing
onit sessions --rename abc123 "thesis ch3"      # by UUID prefix
onit rename old-name "new name"                 # the same thing, as its own verb
```

Names may contain spaces and are matched case-insensitively. Two sessions
cannot share a name, because `--resume-session <name>` would otherwise be a
coin toss. The name is what `--resume-session` looks up.

Inside a chat, `\rename <name>` does the same thing to the session you are in,
and a bare `\rename` prints what it is currently called.

## `onit resume`

Resume a previous session by tag or UUID. Terminal chat already resumes the most
recent session automatically, so this is for picking a *different* one.

```bash
onit resume my-chat      # continue by name
onit resume 3            # continue row 3 of `onit sessions`
onit resume abc123       # continue by session UUID prefix
onit resume              # continue the most recent session (same as bare `onit`)
```

Equivalent to `onit --resume-session NAME_OR_ID`.

To start from scratch instead, use `onit --new-session`. Server modes
(`serve web`, `serve loop`) manage their own
sessions and never auto-resume.

A resumed session carries more than the conversation. Alongside each
`<session_id>.jsonl` is a `<session_id>.state.json` recording what the session
*did* — which tools it ran and how many times, how many turns it spent, and
whether the last attempt finished or stopped early at a limit. Resuming reads
it back and tells the agent, so a continued session builds on work that already
succeeded instead of starting it again. It is deleted with the session.

## The A2A client (was `onit ask`)

The `ask` subcommand was removed: its only bundled server, the A2A server,
already lives in [legacy/](../legacy/), and the client moved there with it. The
client is a plain JSON-RPC-over-HTTP implementation (no A2A SDK import) and
talks to any A2A server:

```bash
python -m legacy.a2a_client "what is the weather in Manila"     --server http://localhost:9001
python -m legacy.a2a_client "summarize this document" --file report.pdf
python -m legacy.a2a_client "describe this image" --image photo.jpg
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
onit doctor --endpoints     # + smoke-test every known endpoint (config, presets,
                            #   stored keys) with a model listing; 20s each
onit doctor --json          # machine-readable report (stdout is pure JSON)
onit doctor --keep-session  # keep the throwaway session for inspecting a failure
```

The battery's own endpoint check only reaches the endpoint this session is
assigned — the one that will serve. `--endpoints` reaches the rest: every
preset, every host with a stored key, so a dead key or an unreachable server
is reported once, here, instead of the next time it is pointed at. A failed
probe fails the run (exit 1); a machine with no known endpoints at all skips
the section rather than failing it.

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

