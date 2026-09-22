# Legacy Extraction Proposal: Viber, Telegram, and A2A

**Status:** Proposal — awaiting approval
**Date:** September 21, 2026
**Goal:** Simplify the active codebase by moving Viber, Telegram, and A2A support into a self-contained `legacy/` directory. The core remains: terminal chat, web UI, loop mode, MCP servers, sandbox.

---

## 1. Feasibility verdict

**Feasible, low risk.** The three features are well-isolated:

- **Viber + Telegram** are leaf modules: `src/ui/viber.py` (315 lines) and `src/ui/telegram.py` (265 lines) are imported from exactly one place each (`run_gateway_sync`, `src/onit.py:2072/2091`). No other production code touches them.
- **A2A** is the only entangled one: `OnItA2AExecutor` (src/onit.py:654) subclasses `a2a.server.agent_execution.AgentExecutor`, and `ClientDisconnectMiddleware` (798) wraps it — these **must move out of `onit.py`** because top-level imports of `a2a.*` (lines 58–61) would otherwise crash every startup once the SDK is removed. `run_a2a` (1963–2060) does its own lazy imports, so it moves cleanly.
- The `onit ask` client (`_send_task`, cli.py:267) talks raw JSON-RPC over `requests` — **no a2a-sdk dependency** — so it can move to legacy without touching core dependencies.

Nothing in `src/mcp/`, `src/model/`, `src/lib/`, or the web UI depends on any of the three.

## 2. What moves where

```
legacy/
├── README.md                      # what this is, why it was frozen, how to run
├── requirements.txt               # a2a-sdk[all]>=1.0.0, python-telegram-bot
├── a2a/
│   ├── executor.py                # OnItA2AExecutor + ClientDisconnectMiddleware (from src/onit.py)
│   ├── server.py                  # run_a2a() body (from src/onit.py:1963-2060)
│   └── client.py                  # _build_a2a_parts, _extract_a2a_text, _handle_sse_events,
│                                  #   _format_output, _send_task (from src/cli.py)
├── gateway/
│   ├── telegram.py                # from src/ui/telegram.py
│   ├── viber.py                   # from src/ui/viber.py
│   └── run.py                     # run_gateway_sync() body (from src/onit.py:2061-2101)
└── test/
    ├── test_a2a.py                # from src/test/test_a2a.py
    └── test_viber.py              # from src/test/test_viber.py
```

Legacy code will need small shims: import `OnIt` via `sys.path` insertion or `from src.onit import OnIt`, and read config the same way it does today.

## 3. Removal checklist (active codebase)

**`src/onit.py`**
- Delete top-level `a2a.*` imports (58–61)
- Delete `OnItA2AExecutor` (654–798) and `ClientDisconnectMiddleware` (798–~856)
- Delete `run_a2a()` (1963–2060) and `run_gateway_sync()` (2061–2101)
- Delete dispatch branches in `run()` (a2a/gateway), banner branches (1015–1018), `file_server_url` a2a branch (1328–1342)
- Delete config fields: `a2a`, `a2a_port`, `a2a_name`, `a2a_description` (917–920), `gateway`, `gateway_token`, `viber_webhook_url`, `viber_port` (921–924) and their `_setup_config_fields` reads (1381–1386)
- Keep `_call_sandbox_stop` — used by web/text/loop paths

**`src/cli.py`**
- Delete `ask` subcommand + parser (701–713) and dispatch (1365–1382)
- Delete `serve a2a` parser (720–723) and `serve gateway` parser (738–748)
- Delete a2a client helpers (82–360) and serve-mode config blocks (930–933, 949–951, 1097–1131)
- Update usage text (lines 12–15) and the a2a/gateway entries in mode gates (1183, 1390, 1449, 1461)

**`src/setup.py`** — drop `telegram_bot_token` / `viber_bot_token` prompts (64–67)

**`src/container_launcher.py`** — drop telegram/viber secret passthroughs (45–46), `--viber-port` mapping (402–404), and `--a2a`/`--gateway` from the serving-modes check (410)

**`pyproject.toml`** — remove `a2a-sdk[all]>=1.0.0` (line 34) and `python-telegram-bot` (55); drop `a2a` from keywords (15)

**`docker-compose.yml`** — remove `onit-a2a` (131–143) and `onit-gateway` (145–155) services + their volumes (173–174)

**Tests** — move `test_a2a.py`, `test_viber.py` to `legacy/test/`; update `test_cli.py` (5 refs), `test_container_launcher.py` (6 refs), `test_setup.py` (1 ref)

**Docs** — move `docs/GATEWAY_QUICK_START.md` → `legacy/`; update `README.md`, `docs/CLI.md`, `docs/ARCHITECTURE.md`, `RELEASE.md` (14 a2a refs), `docs/DOCKER.md`, `docs/ISOLATION.md`, `docs/CONFIGURATION.md`, etc.

## 4. Dependency payoff

| Dependency | Status | Why it matters |
|---|---|---|
| `a2a-sdk[all]` | **dropped** | Pulls grpcio + protobuf — the heaviest deps in the tree |
| `python-telegram-bot` | **dropped** | Only used by the Telegram gateway |
| `aiohttp` | kept | Also used by `ui/voice.py` and `ui/api.py` |
| `fastapi`, `uvicorn`, `requests` | kept | Web UI and MCP servers use them |

Install size and supply-chain surface shrink accordingly; `pip install` gets faster for every new user (aligns with the <10-min setup goal).

## 5. Risks and mitigations

1. **Hidden importers.** Verified by grep: only `onit.py:2072/2091` import the gateways; only `onit.py` imports `a2a.*`. The `.venv` here is incomplete (can't import `src.onit` at all), so the real check is the CI suite + a fresh install.
2. **Users relying on gateway/A2A.** Mitigated by keeping the code runnable from `legacy/` with its own `requirements.txt` and a README explaining the two-line invocation. Nothing is deleted from git history.
3. **`ask` removal breaks scripts.** `onit ask` moves with the A2A client; the legacy README documents `python -m legacy.a2a.client` as the replacement. If you'd rather keep `ask` in the active CLI (it has no SDK dependency), say so — it's a one-line change to the plan.
4. **Test coverage loss.** `pyproject.toml` sets `testpaths = ["src/test"]`, so legacy tests won't run in CI — intentional. They stay executable via `pytest legacy/test` on demand.

## 6. Recommended execution order

1. Create `legacy/` tree and copy files (no active-code changes yet) — commit
2. Move the two executor/middleware classes + `run_a2a`/`run_gateway_sync` out of `onit.py`; strip a2a/gateway config fields and dispatch — commit
3. Strip CLI subcommands, setup prompts, container-launcher passthroughs — commit
4. Prune `pyproject.toml` deps + docker-compose services — commit
5. Move/adjust tests; run full suite — commit
6. Docs pass (README, CLI, ARCHITECTURE, RELEASE, DOCKER) — commit
7. Fresh-install canary: `pip install -e .` in a clean venv, `onit --help`, `onit serve web --no-login`, run pytest

Each step leaves the tree green. Steps 2–4 are mechanical; step 7 is the real gate.

## 7. Open questions

1. Keep `onit ask` in the active CLI (SDK-free client) or move it to legacy with the rest of A2A?
2. Keep `docs/GATEWAY_QUICK_START.md` inside `legacy/` (my recommendation) or delete outright?
3. Version the change as a minor (0.x+1) with a "Breaking: A2A/gateway moved to legacy" note in RELEASE.md?