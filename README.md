# OnIt

*OnIt* — the AI is working on the given task and will deliver the results shortly.

OnIt is an agent harness: it hands a language model a set of tools — shell, file
editing, web search, weather, search over your own documents — a working directory,
and a memory of what it has already done, then runs the loop until the task is
finished. The same agent is reachable from a terminal or a browser chat UI.

## Quick Start (~10 minutes)

Install, point OnIt at a model, run. Steps 2 and 3 take two minutes; step 1 is the
rest of the clock.

### 1. Install

Python **3.10–3.12** (3.12 recommended), in its own environment:

```bash
git clone https://github.com/sibyl-oracles/onit.git
cd onit
uv venv --python 3.12
source .venv/bin/activate
uv pip install -U --upgrade-strategy eager -e '.[all]'
```

(`conda`/`pip` work the same way. Install from source, not the PyPI wheel, which lags.)

### 2. Point OnIt at a model

Any OpenAI-compatible endpoint. Two hosted options, no hardware needed — pick one:

```bash
# Ollama cloud — free tier
export OLLAMA_API_KEY=...                                  # key from ollama.com
onit --host https://api.ollama.com --model glm-5.3:cloud

# OpenRouter — one key, many models, paid per token
export OPENROUTER_API_KEY=...                              # key from openrouter.ai
onit --host https://openrouter.ai/api/v1 --model google/gemini-2.5-pro
```

The model must support **tool calling** — OnIt acts through tools, so a model
without them will talk but not act. Prefer your own hardware? Local Ollama needs no
key: `ollama pull qwen3:30b`, then `onit --host http://localhost:11434/v1 --model qwen3:30b`.
vLLM, SGLang, and MLX work the same way — [docs/RUN_A_MODEL_SERVER.md](docs/RUN_A_MODEL_SERVER.md).

To make it stick, `onit setup` walks through URL, key, and model once; keys go into
your OS keychain, never into a file.

### 3. Run

```bash
onit
```

That's the text UI. Tools start automatically, the agent works out of `~/sandbox`,
and `\bye` leaves. Run `onit` again and the conversation picks up where it left off.

## Three things to try

### 1. Ask questions over your local documents

Drop PDFs, notes, or reports into a folder and ask. OnIt indexes them (BM25 +
embeddings) and cites the file and section it answered from — no web round trip,
nothing leaves the machine.

```text
> What does our refund policy say about annual plans? Check ~/docs/policies/

◆ search    indexing 14 documents in ~/docs/policies/
◆ search    "refund policy annual plans" → 3 hits, top: policies/billing.pdf
◆ read      policies/billing.pdf, section 4.2
Annual plans are refundable pro rata within the first 60 days only
(section 4.2 of policies/billing.pdf). Monthly plans have no such window.
```

Works on PDF, DOCX, XLSX, CSV, and Markdown. See [docs/LOCAL_SEARCH.md](docs/LOCAL_SEARCH.md).

### 2. Build a web application from scratch

Describe the app. OnIt scaffolds it, writes and runs the tests, fixes what fails,
and commits — iterating until the test suite is green.

```text
> build a URL shortener in Flask: POST /shorten returns a code,
  GET /<code> redirects, SQLite storage, pytest tests, then commit

◆ write     app.py, templates/, tests/test_app.py
◆ bash      pytest -q → 1 failed: redirect returns 200, expected 302
◆ edit      app.py — redirect(..., 302)
◆ bash      pytest -q → 6 passed
◆ bash      git init && git commit -m "URL shortener: shorten + redirect"
Done. app.py + tests, all green, committed.
```

Add a GitHub token (below) and the same session can push, open PRs, and fix CI.

### 3. Run an automated security audit on a server

Point OnIt at a host you administer and let it sweep: exposed services, weak
configs, stale packages, world-writable paths — with every command it runs shown
and approval-gated.

```text
> security audit this server: open ports, outdated packages,
  weak SSH settings, world-writable files. Write findings to audit.md

◆ bash      ss -tlnp → 0.0.0.0:6379 (Redis, no auth)
◆ bash      apt list --upgradable → 11 packages, incl. openssl
◆ bash      grep -E 'PermitRootLogin|PasswordAuth' /etc/ssh/sshd_config
◆ write     audit.md — 3 critical, 4 high, 2 low, with fixes
Redis is reachable from all interfaces without a password — bind it to
127.0.0.1 or enable ACLs (audit.md, finding 1 of 9).
```

Run it on a schedule with `onit serve loop "re-audit and diff against audit.md" --period 86400`.

## API keys

| Key | Needed for | Priority |
| --- | --- | --- |
| `GITHUB_TOKEN` | Automated git workflows — clone, commit, push, PRs, CI fixes | Important |
| `OLLAMA_API_KEY` | Ollama cloud models + web search | Core |
| `OPENROUTER_API_KEY` | OpenRouter models | Core (alternative to Ollama) |
| `OPENWEATHER_API_KEY` | Weather tool | Optional — free anyway |

GitHub: create a token at [github.com/settings/tokens](https://github.com/settings/tokens)
(repo scope), then `onit setup` → *GitHub access token* (stored in the OS keychain).
A typical delegation once it's stored:

```text
> clone github.com/sibyl-oracles/onit, fix the typo in README line 12,
  commit as "docs: fix typo", and push

◆ bash     git clone https://github.com/sibyl-oracles/onit.git
◆ bash     edit + git commit -m "docs: fix typo"
◆ bash     git push origin main
Pushed to main.
```

## From text UI to web UI

Same agent, same sessions — now in the browser.

```bash
onit serve web --no-login     # trusted network (LAN, localhost)
```

For anything reachable beyond your machine, add Google login (~5 minutes):
create an OAuth client at [console.cloud.google.com](https://console.cloud.google.com/)
→ *APIs & Services* → *Credentials*, add `http://localhost:9000/auth/callback` as an
authorized redirect URI (a mismatch there is the classic
`Error 400: redirect_uri_mismatch`), then set `GOOGLE_CLIENT_ID` and
`GOOGLE_CLIENT_SECRET`. For a public server you also need HTTPS — the full path,
including Docker Compose and reverse proxy, is in
[docs/DEPLOYMENT_WEB.md](docs/DEPLOYMENT_WEB.md), with session isolation and
command approvals in [docs/ISOLATION.md](docs/ISOLATION.md) and
[docs/WEB_AUTHENTICATION.md](docs/WEB_AUTHENTICATION.md).

## Inside a session

Lines starting with `\` are answered by OnIt itself: `\help`, `\setup` (endpoints in
use), `\model [name]`, `\host add <url>` (spread across servers), `\key`, `\save`,
`\bye`. Everything else goes to the model, which works until the task is done —
each tool call is shown as it runs.

## Other front ends

| | |
|---|---|
| `onit serve web` | Browser chat UI — [above](#from-text-ui-to-web-ui) |
| `onit serve loop "task" --period 60` | Repeat a task on a timer |
| `onit --container` | Hardened Docker container ([docs/DOCKER.md](docs/DOCKER.md)) |

Telegram, Viber and A2A moved to [legacy/](legacy/).

## Documentation

[CLI](docs/CLI.md) · [Configuration](docs/CONFIGURATION.md) · [Tools](docs/TOOLS.md) ·
[Local document search](docs/LOCAL_SEARCH.md) · [Run a Model Server](docs/RUN_A_MODEL_SERVER.md) ·
[Model Serving](docs/MODEL_SERVING.md) · [Web deployment](docs/DEPLOYMENT_WEB.md) ·
[HTTPS](docs/HTTPS_DEPLOYMENT.md) · [Isolation](docs/ISOLATION.md) ·
[Web authentication](docs/WEB_AUTHENTICATION.md) · [Docker](docs/DOCKER.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Testing](docs/TESTING.md) · [Benchmarks](benchmarks/README.md)

## Size

| Category | Files | Lines |
|---|---|---|
| **Production code** (`src/`, excl. tests) | 61 `.py` | **34,602** (25,884 code, 4,414 comments, 4,305 blank) |
| Frontend (`src/ui/static/`, excl. vendor) | 6 | 3,274 |
| Prompt templates + configs (YAML) | 6 | 471 |
| **Test suites** (`src/test/`, `benchmarks/test_*`, `legacy/test/`) | 48 `.py` | 27,001 |
| Benchmarks (non-test) | 14 `.py` | 1,765 |
| Legacy (non-test) | 8 `.py` | 1,482 |
| Docs (`docs/`, README, RELEASE) | 26 `.md` | 7,172 |
| **Total Python** (all `.py`, excl. `__pycache__`) | 211 | **72,332** |

Production code by module: `mcp` 9,929 · `model` 9,327 · `ui` 7,324 · `onit.py` 2,191 · `setup.py` 936 · `learn` 1,127 · `lib` 882 · `cli.py` 857 · `type` 940 · `container_launcher.py` 574 · `sessions.py` 486 · `__init__.py` 29.

*Recompute:* `find src -name '*.py' -not -path '*/__pycache__/*' -not -path 'src/test/*' -exec cat {} + | wc -l`

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.