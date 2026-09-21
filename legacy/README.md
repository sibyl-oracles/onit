# OnIt Legacy Components

Support moved out of the active codebase to keep the core lean. Nothing here
is imported by `src/`; these components run standalone.

| Component | What it was | Entry point |
|---|---|---|
| `gateway/` | Telegram and Viber chat-bot gateways | `python -m legacy.gateway` |
| `a2a/` | A2A protocol server (agent-to-agent) | `python -m legacy.a2a_server` |

## Running a legacy component

From the repository root, with the legacy dependencies installed:

```bash
pip install -r legacy/requirements.txt

# Telegram gateway
TELEGRAM_BOT_TOKEN=... python -m legacy.gateway telegram

# Viber gateway (needs a public HTTPS webhook)
VIBER_BOT_TOKEN=... VIBER_WEBHOOK_URL=https://... python -m legacy.gateway viber \
    --webhook-url https://... --port 8443

# A2A protocol server
python -m legacy.a2a_server --port 9001
```

The gateways and the A2A server construct the same `OnIt` agent the active
front ends use, so serving, tools and sessions behave as they did before
extraction.

## The `onit ask` client stays in the active CLI

`onit ask` is a plain JSON-RPC-over-HTTP client (no A2A SDK import), so it
remains part of the active CLI. It can talk to any A2A server, including
`python -m legacy.a2a_server`:

```bash
python -m legacy.a2a_server --port 9001 &
onit ask "what is the weather in Manila" --server http://localhost:9001
```

## Tests

Legacy tests live beside the code and are excluded from the default pytest
run (`testpaths = src/test`). Run them on demand:

```bash
pytest legacy/test -v
```

## Why these moved

- The A2A SDK (`a2a-sdk[all]`) pulls in gRPC and protobuf — the heaviest
  dependencies in the project — for a transport most deployments never use.
- The chat gateways serve a narrow use case and pinned
  `python-telegram-bot`.
- Removing both shortens install time and shrinks the supply chain, in line
  with the goal of a sub-10-minute setup.

See `legacy_extraction_proposal.md` in the repository root for the full
feasibility study and migration plan.