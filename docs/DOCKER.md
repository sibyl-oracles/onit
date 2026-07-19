# Docker

OnIt can run fully containerized in two ways:

1. **`onit --container`** *(recommended)* — the OnIt CLI on the host drives Docker for you: it builds the image on first use, applies all hardening flags, bridges your config and secrets in, and publishes only the ports the selected mode needs.
2. **Manual `docker build` / `docker run` / `docker compose`** — for servers, CI, or anywhere you don't want OnIt installed on the host.

## Prerequisites

- **Docker** — [Docker Desktop](https://docs.docker.com/get-docker/) on macOS/Windows, or Docker Engine on Linux. Verify the daemon is running:

  ```bash
  docker info
  ```

- **(GPU only) NVIDIA Container Toolkit** — required for `--container-gpus` / `--gpus`. The image is built on CUDA 12.6, so the host driver must support CUDA ≥ 12.6. Verify pass-through works:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
  ```

- **For the `onit --container` path**: OnIt installed on the host (`pip install onit`, or from source — see the README [Getting Started](../README.md#getting-started)) and, before the first containerized run, `onit setup` completed on the host so the launcher has a config and secrets to bridge in.

## Quick start: `onit --container`

```bash
onit --container                          # interactive terminal inside the container
onit --container serve web                # web UI, port 9000 published
onit --container serve a2a --port 9100    # A2A server on a custom port
onit --container serve gateway telegram   # Telegram bot (no ports published)
```

On first run, the launcher builds the `onit:local` image from the repository `Dockerfile` (this takes a few minutes — it installs the full Python stack). Subsequent runs reuse the image.

The launcher handles everything the manual examples below do by hand:

- verifies Docker is reachable, builds `onit:local` if missing;
- runs the container with the full hardening profile (see [What is isolated](#what-is-isolated));
- bind-mounts `~/.onit/config.yaml` (and `~/.onit/secrets.yaml`, if present) **read-only**;
- bridges secrets from the host OS keychain as ephemeral env vars — `OPENROUTER_API_KEY`, `OLLAMA_API_KEY`, `VLLM_API_KEY`, `OPENWEATHERMAP_API_KEY`, `TELEGRAM_BOT_TOKEN`, `VIBER_BOT_TOKEN`, `GITHUB_TOKEN`, `HF_TOKEN`. They are never baked into the image;
- mounts the named volume `onit-data` at `/home/onit/data` for persistent state;
- publishes only the ports needed by the selected mode;
- tears the container down cleanly on Ctrl-C.

> **Web UI note:** Google OAuth client credentials (`web_google_client_id` / `web_google_client_secret`) are **not** bridged from the keychain. For `onit --container serve web`, either export `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` on the host before launching, put them in `~/.onit/config.yaml`, or pass `--no-login`.

### Updating the image

The launcher rebuilds only when the image is missing. After upgrading OnIt or editing the source, force a rebuild:

```bash
docker rmi onit:local     # next `onit --container` run rebuilds
# or rebuild explicitly, bypassing the layer cache:
docker build --no-cache -t onit:local .
```

## What is isolated

The container is started with hardening flags:

- `--read-only` root filesystem. Writes land only on RAM-backed tmpfs mounts
  (`/tmp`, `~/.cache`, `~/.onit`) or the `onit-data` named volume
  (`/home/onit/data` — pip installs via `PIP_TARGET`, HF caches, sessions).
- `--cap-drop=ALL` (no Linux capabilities beyond what the Python process strictly needs)
- `--security-opt no-new-privileges:true` (no sudo/setuid escalation; the
  image ships without sudo or sudoers rules)
- `--pids-limit=4096`; memory unlimited by default (clamp with `--container-memory`)
- Runs as non-root `onit` user (uid 1000)
- No host filesystem mounts by default
- AST-based command allowlisting is enforced by default inside the container
  (`ONIT_COMMAND_ALLOWLIST`), package managers are blocked unless
  `--container-allow-installs` is passed, and installs must then be
  version-pinned (`pip install name==1.2.3`). Repeated policy violations
  auto-contain the bash MCP server (see README "Auto-Containment").

## What crosses the boundary

- `~/.onit/config.yaml` is bind-mounted **read-only** into the container.
- `~/.onit/secrets.yaml` is bind-mounted **read-only** if it exists.
- Secrets from the host OS keychain are passed as ephemeral env vars (not
  baked into the image) — see the list under [Quick start](#quick-start-onit---container).
- A named volume `onit-data` is mounted at `/home/onit/data` for session
  persistence.
- Outbound network is allowed (needed for LLM API calls).
- Ports are published **only** for the selected mode:

  | Mode | Default port | Override |
  |---|---|---|
  | (terminal) | — (no ports) | — |
  | `serve web` | `9000:9000` | `--port` |
  | `serve a2a` | `9001:9001` | `--port` |
  | `serve gateway viber` | `8443:8443` | `--port` |

## Exposing GPUs, extra host paths, and resources

By default the container has **no GPUs and no extra host paths**. Override as needed:

```bash
# NVIDIA GPU pass-through (requires NVIDIA Container Toolkit on host).
onit --container --container-gpus all

# Specific GPUs only.
onit --container --container-gpus '"device=0,1"'

# Mount a host documents directory read-only into the container.
onit --container --container-mount "$HOME/docs:/home/onit/documents:ro"

# Multiple mounts — flag is repeatable.
onit --container \
  --container-mount "$HOME/datasets:/data:ro" \
  --container-mount "$HOME/models:/models:ro"

# Resource tuning.
onit --container --container-memory 16g      # hard memory cap (default: unlimited)
onit --container --container-shm-size 8g     # /dev/shm (default 4g; raise for PyTorch DataLoader)
onit --container --container-tmp-size 32g    # /tmp tmpfs (default 16g; host-RAM backed)
```

Each `--container-mount` punches a hole in the sandbox. Use `:ro` whenever
possible, and never mount a host path that contains secrets you don't want
the agent to see.

Note: the default image ships **CPU-only torch** to keep the build small.
`--container-gpus` plumbs the device through, but for actual GPU compute
you need to rebuild the image with a CUDA torch wheel — see
[Preinstalled ML packages](#preinstalled-ml-packages).

## Preinstalled ML packages

The container image ships with a default machine-learning stack so the agent
can run ML code without `pip install` round-trips at execution time:

- `torch`, `torchvision`, `torchaudio` (CPU-only wheels)
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `einops`

CUDA is intentionally not included — containers have no GPU access by
default, and CUDA wheels would add roughly 3 GB to the image. For GPU
workloads, build a derivative image that replaces the CPU wheels with the
appropriate CUDA build.

## Combining with `--sandbox`

`--container` and `--sandbox` are complementary. `--container` isolates the
whole OnIt process from the host; `--sandbox` delegates individual code-
execution tool calls to an external MCP sandbox provider. They can be used
together for defense in depth:

```bash
onit --container --sandbox serve web
```

## Manual `docker run`

Use this path when OnIt is not installed on the host. The image's entrypoint
is the `onit` CLI itself (under `tini` for signal handling), so everything
after the image name is a regular `onit` command line — the same subcommands
and flags as on the host.

Build the image first:

```bash
docker build -t onit:local .
```

Configuration comes from env vars (`-e KEY=value` or `--env-file .env` — see
[Environment variables](#environment-variables)), or from a config file
mounted into the container. There is no keychain inside the container, so
every secret must be passed explicitly.

### Terminal mode (interactive chat)

```bash
docker run --rm -it \
  -e ONIT_HOST=https://openrouter.ai/api/v1 \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  onit:local --model google/gemini-2.5-pro
```

### Web mode

```bash
docker run --rm --name onit-web \
  -e ONIT_HOST=https://openrouter.ai/api/v1 \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -e GOOGLE_CLIENT_ID=....apps.googleusercontent.com \
  -e GOOGLE_CLIENT_SECRET=GOCSPX-... \
  -p 9000:9000 \
  onit:local serve web --port 9000
```

Open `http://localhost:9000`. Google login is required by default; for an
open UI on a trusted network, drop the `GOOGLE_*` vars and add `--no-login`.

### A2A mode (Agent-to-Agent server)

```bash
docker run --rm --name onit-a2a \
  -e ONIT_HOST=https://openrouter.ai/api/v1 \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -p 9001:9001 \
  onit:local serve a2a --port 9001
```

Send a task to it from another container (or use `onit ask` from any host):

```bash
docker run --rm onit:local ask "What is the weather in Manila?" \
  --server http://host.docker.internal:9001
```

### Gateway mode (Telegram / Viber)

```bash
docker run --rm --name onit-telegram \
  -e ONIT_HOST=https://openrouter.ai/api/v1 \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -e TELEGRAM_BOT_TOKEN=... \
  onit:local serve gateway telegram
```

```bash
docker run --rm --name onit-viber \
  -e ONIT_HOST=https://openrouter.ai/api/v1 \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -e VIBER_BOT_TOKEN=... \
  -e VIBER_WEBHOOK_URL=https://your-domain.com/viber \
  -p 8443:8443 \
  onit:local serve gateway viber
```

Port 8443 must be reachable from the internet for the Viber webhook.

### In-house documents

Mount a host directory read-only and point `ONIT_DOCUMENTS_PATH` at it:

```bash
docker run --rm -it \
  -e ONIT_HOST=... -e OPENROUTER_API_KEY=... \
  -e ONIT_DOCUMENTS_PATH=/docs \
  -v "$HOME/company-docs":/docs:ro \
  onit:local
```

### Hardening manual runs

`docker run` alone applies **none** of the launcher's isolation. To match it,
add the hardening flags and the persistent data volume yourself:

```bash
docker run --rm -it \
  --read-only \
  --cap-drop=ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 4096 \
  --tmpfs /tmp:size=16g \
  --tmpfs /home/onit/.cache:size=1g \
  --tmpfs /home/onit/.onit:size=64m \
  -v onit-data:/home/onit/data \
  -e ONIT_HOST=... -e OPENROUTER_API_KEY=... \
  onit:local
```

Or simply prefer `onit --container` / `docker compose`, which apply this
profile for you.

## Docker Compose

`docker-compose.yml` in the repo root defines a multi-service production
stack, all services sharing the same hardening profile (read-only rootfs,
`cap_drop: ALL`, `no-new-privileges`, tmpfs for ephemeral writes):

| Service | Description | Port |
|---------|-------------|------|
| `caddy` | TLS termination, automatic Let's Encrypt certificates | 80, 443 |
| `onit-mcp` | Shared MCP servers | 18200–18204 |
| `onit-web` | Web UI (loopback-only; public traffic goes through Caddy) | 127.0.0.1:9000 |
| `onit-a2a` | A2A server | 9001 |
| `onit-gateway` | Telegram bot gateway | — |
| `onit-viber` | Viber bot gateway | 8443 |
| `onit-terminal` | Interactive terminal (opt-in `terminal` profile) | — |

Configuration is driven by a `.env` file in the repo root (see
[Environment variables](#environment-variables)) plus two stack variables:

```bash
ONIT_DOMAIN=mychat.ai        # public domain for HTTPS (unset = localhost, self-signed)
ONIT_DATA_DIR=/data/sandbox  # host folder for the agent's working files
```

The data directory is a host bind mount and the containers run as uid 1000,
so create it writable by that user first:

```bash
sudo mkdir -p /data/sandbox
sudo chown 1000:1000 /data/sandbox
```

Start the stack:

```bash
docker compose up -d --build     # build + start web, a2a, gateways behind Caddy
docker compose logs -f onit-web  # follow a service's logs
docker compose --profile terminal run --rm onit-terminal   # one-off interactive chat
docker compose down              # stop (volumes are kept)
```

For the full HTTPS walkthrough — DNS, certificates, OAuth redirect URIs,
post-install smoke tests — see [HTTPS_DEPLOYMENT.md](HTTPS_DEPLOYMENT.md).

## Session persistence

State survives container restarts in named Docker volumes:

- `onit-data` (launcher) / `ONIT_DATA_DIR` bind mount (compose) — the agent's
  working directory: tool output, pip installs (`PIP_TARGET`), HF caches.
- `web-sessions`, `a2a-sessions`, `gateway-sessions`, `terminal-sessions`
  (compose) — per-mode chat history (JSONL) so sessions can be resumed.
- `caddy-data`, `caddy-config` (compose) — TLS certificates and ACME account
  keys. Keep these: recreating them repeatedly hits Let's Encrypt rate limits.

Each browser tab (web), A2A context, or messaging chat gets its own isolated
session with separate chat history and file storage. To start completely
fresh:

```bash
docker volume rm onit-data       # launcher state
docker compose down -v           # ALL compose volumes, including certificates
```

## Environment variables

Pass API keys via an `.env` file or individual `-e KEY=value` flags. Never
bake secrets into the image.

Example `.env` file:

```bash
# LLM endpoint (one of vLLM / OpenRouter / Ollama cloud)
ONIT_HOST=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=sk-or-v1-your-key-here
# VLLM_API_KEY=...                  # if your vLLM server was started with --api-key
# OLLAMA_API_KEY=...                # Ollama cloud + web search tool

# Optional tools
OPENWEATHERMAP_API_KEY=...          # weather tool
# GITHUB_TOKEN=...                  # github_repo tool + git credential helper
# HF_TOKEN=...                      # Hugging Face downloads

# Web UI login (required unless running with --no-login)
GOOGLE_CLIENT_ID=....apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-...

# Gateways
TELEGRAM_BOT_TOKEN=...              # serve gateway telegram
VIBER_BOT_TOKEN=...                 # serve gateway viber
VIBER_WEBHOOK_URL=https://your-domain.com/viber

# Compose stack
ONIT_DOMAIN=mychat.ai
ONIT_DATA_DIR=/data/sandbox
```

## Troubleshooting

- **`docker: command not found` / daemon not reachable** — install Docker
  Desktop (or start the `docker` service on Linux). `onit --container` checks
  this and prints the same guidance.
- **Stale image after upgrading OnIt** — the launcher only builds when the
  image is missing. `docker rmi onit:local`, then rerun.
- **GPU not visible in the container** — install the NVIDIA Container
  Toolkit, confirm `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi`
  works, and remember the host driver must support CUDA ≥ 12.6 (the image's
  CUDA version). Torch in the default image is CPU-only either way — rebuild
  with CUDA wheels for GPU compute.
- **`Permission denied` writing to the compose data dir** — the containers
  run as uid 1000; `sudo chown 1000:1000 $ONIT_DATA_DIR`.
- **Web UI refuses to start** — Google OAuth credentials are missing. Pass
  `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` (they are not bridged from the
  host keychain), or run with `--no-login`.
- **All write tools suddenly refuse calls** — the bash MCP server
  auto-contained after repeated policy violations. Delete
  `.onit-containment.json` from the data directory and restart (see README
  "Auto-Containment").
- **Let's Encrypt rate-limit errors** — you recreated the `caddy-data`
  volume too often. Keep it; certificates renew automatically.
