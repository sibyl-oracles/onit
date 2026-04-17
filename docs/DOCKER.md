# Docker

## Running OnIt in a container (`--container`)

The fastest way to run OnIt in an isolated container is:

```bash
onit --container                # interactive terminal inside container
onit --container --web          # web UI, port 9000 mapped to host
onit --container --a2a          # A2A server, port 9001 mapped to host
```

On first run, the launcher builds the `onit:local` image from the repository
`Dockerfile`. Subsequent runs reuse the image.

### What is isolated

The container is started with hardening flags:

- `--read-only` root filesystem (writes confined to `/home/onit/data` named volume and a `/tmp` tmpfs)
- `--cap-drop=ALL` (no Linux capabilities beyond what the Python process strictly needs)
- `--security-opt=no-new-privileges`
- `--pids-limit=256`, `--memory=2g`
- Runs as non-root `onit` user (uid 1000)
- No host filesystem mounts by default

### What crosses the boundary

- `~/.onit/config.yaml` is bind-mounted **read-only** into the container.
- `~/.onit/secrets.yaml` is bind-mounted **read-only** if it exists.
- Secrets from the host OS keychain are passed as ephemeral env vars (not
  baked into the image). Supported keys: `OPENROUTER_API_KEY`, `OLLAMA_API_KEY`,
  `OPENWEATHERMAP_API_KEY`, `TELEGRAM_BOT_TOKEN`, `VIBER_BOT_TOKEN`,
  `GITHUB_TOKEN`, `HF_TOKEN`.
- A named volume `onit-data` is mounted at `/home/onit/data` for session
  persistence.
- Outbound network is allowed (needed for LLM API calls).
- Ports are mapped **only** when the corresponding mode flag is used.

### Exposing GPUs, ports, and extra host paths

By default the container has **no GPUs, no extra host paths, and only the
ports needed for the mode you selected** (`--web` → 9000, `--a2a` → 9001,
`--gateway` viber → 8443). Override as needed:

```bash
# NVIDIA GPU pass-through (requires NVIDIA Container Toolkit on host).
onit --container --container-gpus all

# Specific GPUs only.
onit --container --container-gpus '"device=0,1"'

# Mount a host documents directory read-only into the container.
onit --container --container-mount "$HOME/docs:/home/onit/documents:ro" \
  --documents-path /home/onit/documents

# Multiple mounts — flag is repeatable.
onit --container \
  --container-mount "$HOME/datasets:/data:ro" \
  --container-mount "$HOME/models:/models:ro"

# Non-default ports are honored automatically.
onit --container --web --web-port 9500       # publishes 9500:9500
onit --container --a2a --a2a-port 9100       # publishes 9100:9100
```

Note: the default image ships **CPU-only torch** to keep the build small.
`--container-gpus` plumbs the device through, but for actual GPU compute
you need to rebuild the image with a CUDA torch wheel — see the section
below on preinstalled ML packages.

Each `--container-mount` punches a hole in the sandbox. Use `:ro` whenever
possible, and never mount a host path that contains secrets you don't want
the agent to see.

### Preinstalled ML packages

The container image ships with a default machine-learning stack so the agent
can run ML code without `pip install` round-trips at execution time:

- `torch`, `torchvision`, `torchaudio` (CPU-only wheels)
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `einops`

CUDA is intentionally not included — containers have no GPU access by
default, and CUDA wheels would add roughly 3 GB to the image. For GPU
workloads, build a derivative image that replaces the CPU wheels with the
appropriate CUDA build.

### Combining with `--sandbox`

`--container` and `--sandbox` are complementary. `--container` isolates the
whole OnIt process from the host; `--sandbox` delegates individual code-
execution tool calls to an external MCP sandbox provider. They can be used
together for defense in depth:

```bash
onit --container --sandbox --web
```

## Build the image

```bash
docker build --no-cache -t onit .
```

## Run the container

All examples below pass environment variables inline with `-e`. You can also use `--env-file .env` instead. Mount local directories with `-v host:container:ro` to make documents available inside the container.

### Text mode (interactive terminal)

Minimal:

```bash
docker run --name onit --rm \
  -e ONIT_HOST=<HOST> \
  -it onit \

```

With documents, MCP tools, and a custom prompt:

```bash
docker run --name onit --rm \
  -e ONIT_HOST=<HOST> \
  -e OLLAMA_API_KEY=<TOKEN> \
  -e OPENWEATHERMAP_API_KEY=<KEY> \
  -p 18200-18204:18200-18204 --ipc host \
  -it -v /path/to/docs:/docs:ro onit \
  --model <MODEL> \
  --documents-path /docs/my_topic \
  --show-logs \
  --prompt-intro "You are a helpful assistant" \
  --topic "my topic"
```

### Web mode (Gradio UI)

```bash
docker run --name onit --rm \
  -e ONIT_HOST=<HOST> \
  -e OLLAMA_API_KEY=<TOKEN> \
  -e OPENWEATHERMAP_API_KEY=<KEY> \
  -p 9000:9000 -p 18200-18204:18200-18204 --ipc host \
  -it -v /path/to/docs:/docs:ro onit \
  --model <MODEL> \
  --web --web-port 9000 \
  --documents-path /docs/my_topic \
  --show-logs \
  --prompt-intro "You are a helpful assistant" \
  --topic "my topic"
```

Open `http://localhost:9000` in your browser.

### A2A mode (Agent-to-Agent server)

```bash
docker run --name onit --rm \
  -e ONIT_HOST=<HOST> \
  -e OLLAMA_API_KEY=<TOKEN> \
  -e OPENWEATHERMAP_API_KEY=<KEY> \
  -p 9001:9001 -p 18200-18204:18200-18204 --ipc host \
  -it -v /path/to/docs:/docs:ro onit \
  --model <MODEL> \
  --a2a --a2a-port 9001 \
  --documents-path /docs/my_topic \
  --show-logs \
  --prompt-intro "You are a helpful assistant" \
  --topic "my topic"
```

Send tasks to the server from another terminal:

```bash
docker run --rm -it onit --client --a2a-host http://host.docker.internal:9001 --task "What is the weather?"
```

### Gateway mode (Telegram)

```bash
docker run --name onit-telegram --rm \
  -e ONIT_HOST=<HOST> \
  -e TELEGRAM_BOT_TOKEN=<TOKEN> \
  -e OLLAMA_API_KEY=<KEY> \
  -e OPENWEATHERMAP_API_KEY=<KEY> \
  --cap-drop ALL \
  -p 18200-18204:18200-18204 --ipc host \
  -it -v /path/to/docs:/docs:ro onit \
  --model <MODEL> \
  --documents-path /docs/my_topic \
  --show-logs \
  --prompt-intro "You are a helpful assistant" \
  --gateway telegram \
  --topic "my topic"
```

### Gateway mode (Viber)

```bash
docker run --name onit-viber --rm \
  -e ONIT_HOST=<HOST> \
  -e VIBER_BOT_TOKEN=<TOKEN> \
  -e VIBER_WEBHOOK_URL=<WEBHOOK_URL> \
  -e OLLAMA_API_KEY=<KEY> \
  -e OPENWEATHERMAP_API_KEY=<KEY> \
  --cap-drop ALL \
  -p 8443:8443 -p 18210-18214:18200-18204 --ipc host \
  -it -v /path/to/docs:/docs:ro onit \
  --model <MODEL> \
  --documents-path /docs/my_topic \
  --show-logs \
  --prompt-intro "You are a helpful assistant" \
  --gateway viber \
  --topic "my topic"
```

Port 8443 must be exposed for the Viber webhook server.

> **Running both gateways simultaneously:** Each container must use a unique `--name` and non-overlapping host ports. The Telegram container binds MCP ports `18200-18204` while the Viber container maps them to `18210-18214` on the host. Port 8443 is only needed by Viber (for the webhook server), so it is not included in the Telegram command.

### With a custom config

```bash
docker run --name onit --rm \
  -it -v $(pwd)/configs:/app/configs --env-file .env onit \
  --config configs/default.yaml
```

## Docker Compose

Start the web UI and A2A server together:

```bash
docker compose up --build
```

This launches services defined in `docker-compose.yml`:

| Service | Description | Port |
|---------|-------------|------|
| `onit-mcp` | MCP servers | 18200-18204 |
| `onit-web` | Web UI | 9000 |
| `onit-a2a` | A2A server | 9001 |
| `onit-gateway` | Telegram bot gateway | — |
| `onit-viber` | Viber bot gateway | 8443 |

The `onit-web` and `onit-a2a` services depend on `onit-mcp` and connect to it via the `--mcp-host` flag.

## Session persistence

Each service uses a named Docker volume to persist per-user session data (chat history JSONL files) across container restarts:

- `web-sessions` — Web UI sessions
- `a2a-sessions` — A2A server sessions
- `gateway-sessions` — Telegram/Viber gateway sessions

Each browser tab (web), A2A context, or messaging chat gets its own isolated session with separate chat history and file storage.

## Environment variables

Pass API keys via an `.env` file or individual `-e KEY=value` flags. Never bake secrets into the image.

Example `.env` file:

```bash
ONIT_HOST=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OLLAMA_API_KEY=your_ollama_key
OPENWEATHER_API_KEY=your_weather_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token  # required for --gateway telegram
VIBER_BOT_TOKEN=your_viber_bot_token        # required for --gateway viber
VIBER_WEBHOOK_URL=https://your-domain.com/viber  # public HTTPS URL for Viber webhook
```
