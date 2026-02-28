# Docker

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
  --model <MODEL>
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
