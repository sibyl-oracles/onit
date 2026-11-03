# Native UI Plan — OnIt for macOS, Linux and Windows

**Status:** Proposal
**Author:** Rowel Atienza
**Date:** November 2026
**Target:** macOS 14+ first, then Linux, then Windows

---

## 1. The decision that shapes everything

There are two ways to build a native OnIt, and only one of them is affordable.

| Approach | What it means | Cost | Verdict |
|---|---|---|---|
| **A. Rewrite in Swift** | Port `onit.py`, `chat.py`, the MCP servers, the loop policy, the tool registry to Swift | 12–18 months; every future Python feature must be written twice | **Rejected** |
| **B. Native shell + Python sidecar** | SwiftUI app owns the window; the existing Python agent runs as a bundled child process; they speak the **HTTP/SSE contract that already exists** | 6–8 weeks to a shipping macOS app | **Chosen** |

Approach B is not a compromise, because the backend was already designed for it. `WebApiUI` (`onit/ui/api.py`) is a FastAPI server that streams agent output over Server-Sent Events, and `ApiSession` (`api.py:525`) is *already* a per-session record holding `session_id`, `session_path`, `data_path`, `processing`, `pending_approvals` and `open_stream`. Multi-session, per-session memory and per-session approval routing are **existing backend properties**, not things this plan has to invent. The native app is a second client of an API the browser already uses.

The plan therefore spends its effort on the four things that are genuinely new:

1. **Isolation (§3)** — the harness and the UIs must be able to change without breaking each other. This is a prerequisite: today the harness imports five UI modules, so a third client would be built on a boundary that moves under it.
2. **Process lifecycle** — starting, supervising and stopping a Python sidecar from a sandboxed-ish macOS app.
3. **A native multi-session UI** — SwiftUI, with one window per session and a real sidebar.
4. **Keychain integration** — keys never touch a file the app writes.

Everything else is a client for endpoints that already work.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  OnIt.app  (SwiftUI, macOS)                                  │
│                                                              │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ SessionList│  │ ChatView     │  │ SettingsView         │  │
│  │ (sidebar)  │  │ (per window) │  │ (model/host/keys)    │  │
│  └─────┬──────┘  └──────┬───────┘  └──────────┬───────────┘  │
│        └────────────────┼─────────────────────┘              │
│                  ┌──────▼───────┐   ┌──────────────────────┐ │
│                  │ OnItClient   │   │ KeychainStore        │ │
│                  │ (HTTP + SSE) │   │ (Security.framework) │ │
│                  └──────┬───────┘   └──────────────────────┘ │
│                  ┌──────▼───────┐                            │
│                  │ SidecarMgr   │  spawn / health / restart  │
│                  └──────┬───────┘                            │
└─────────────────────────┼────────────────────────────────────┘
                          │ 127.0.0.1:<ephemeral>  HTTP + SSE
                 ┌────────▼─────────────────────────────────┐
                 │  onit-sidecar  (PyInstaller one-folder)  │
                 │  `onit serve web --port N --no-login`    │
                 │  FastAPI + uvicorn + the whole agent     │
                 └──────────────────────────────────────────┘
```

Inside the sidecar there are two layers, and the arrow between them is the one §3 makes real:

```
   onit/ui/api.py  ─────────────────────────────────┐
   (the HTTP/SSE server)                                 │
        │ imports only onit.contract.* and              │
        │ onit.harness.session_io                       │
        ▼                                               │
   onit/contract/   ─────────────────────────────────┤
   (ChatUIProtocol, UISettings, event names)            │
        ▲                                               │
        │ imports only onit.contract.*                  │
   onit/harness/onit.py                                  │
   (the agent — never imports onit.ui)                   │
                                                         ┘
```

Neither side may import the other. Both may import the contract. That is the whole rule, and §3.8 is the test that keeps it.

### Why a sidecar and not an embedded interpreter

- **Crash isolation.** A model call that segfaults a native extension kills the sidecar, not the app. The window stays open and shows an error.
- **Clean restarts.** Changing the model or host in Settings restarts the sidecar in ~1 s without relaunching the app.
- **The same binary works on all three platforms.** Only the shell is rewritten per OS.
- **It is the contract that already exists.** No new protocol to design, document and keep in sync.

### Why not embed Python via `PythonKit` / `libpython`

Tempting for startup speed, but it welds the app to one Python ABI, makes crash isolation impossible, and turns every dependency upgrade into a rebuild of the app. Rejected.
---

## 3. Isolation: the harness and the UIs must be able to move independently

The goal, stated as a test: **a change to the agent harness must not be able to break a UI, and a change to a UI must not be able to break the harness.** Today neither holds. This section is the work that makes both true, and it is a prerequisite for §4, not an afterthought — the native app is the third client, and a third client built on a contract that is not enforced is a third thing that breaks.

### 3.1 What "isolated" has to mean here

Four properties, in increasing strength. The plan targets the first three and treats the fourth as a non-goal.

| # | Property | Meaning | Status today |
|---|---|---|---|
| **I1** | **No import cycles** | The harness never imports a UI module; a UI imports the harness only through one named surface | **Broken** — see 3.2 |
| **I2** | **One contract, enforced** | The harness↔UI boundary is a versioned, typed interface with a conformance test, not a set of duck-typed attribute reads | **Partial** — `ChatUIProtocol` exists but is not enforced |
| **I3** | **Independent test suites** | A harness change is caught by harness tests; a UI change by UI tests; neither suite needs the other's fixtures | **Partial** — 535 UI tests exist, but they reach into harness internals |
| **I4** | **Independent release** | The harness can ship without rebuilding the app | **Non-goal for v1** — the sidecar is bundled; see 3.9 |

### 3.2 The coupling that exists now, measured

Every arrow below is a real import in the current tree. The ones marked **✗** violate I1.

**Harness → UI (the direction that must be cut):**

| Site | Import | What it is | Verdict |
|---|---|---|---|
| `onit.py:49` | `from .ui import ChatUI` | Constructs the terminal UI inside `OnIt.__init__` | **✗** |
| `onit.py:50` | `from .ui.protocol import ChatUIProtocol` | Type annotation only | **✗** (cheap to move) |
| `onit.py:51` | `from .ui.commands import dispatch` | Backslash-command dispatcher | **✗** |
| `onit.py:994` | `from .ui.api import WebApiUI` | Constructs the web UI | **✗** |
| `onit.py:2528` | `from onit.ui.text import Message` | Rebuilds a message object to dedupe a streamed answer | **✗** (worst of the five) |
| `cli.py:1993` | `from .ui.doctor import render_report, run_checks` | `onit doctor` | **✗** |
| `model/serving/chat.py` | *(none)* | — | **✓ already clean** |

`chat.py` is the important one. The 8,900-line agent loop already takes its UI as a parameter and never imports one. That is the pattern the rest of the harness should follow, and it is the reason this refactor is tractable rather than a rewrite.

**UI → harness (the direction that must be narrowed):**

| Site | Import | Verdict |
|---|---|---|
| `ui/api.py:53-67` | `onit.learn` (4 names), `onit.model.serving.chat` (2), `onit.sessions` (7) | **Narrow** — 13 names, all data-layer, none of them the agent |
| `ui/api.py:340` | `onit.mcp.servers.tasks.shared` (2 URL-screen helpers) | **Move** — a UI should not import an MCP server |
| `ui/commands.py:35` | `..model.serving.balancer` | **Narrow** — 2 names |
| `ui/doctor.py:70,72` | `..type.tools`, `..lib.tools` | **Narrow** — 4 private names |
| `ui/voice.py:62` | `onit.lib.text.remove_tags` | Acceptable |
| `ui/text.py`, `ui/auth.py`, `ui/rating.py` | *(none)* | **✓ already clean** |

Two facts make this smaller than it looks:

1. **`WebApiUI` touches exactly one method on the agent.** `grep '_onit\.'` in `api.py` returns a single hit: `self._onit.process_task(...)` at line 1188. Everything else goes through `getattr(self._onit, "config_data", None)` — four call sites, one attribute. The web UI is *already* almost decoupled; it is coupled to the **session store** and the **learn store**, not to the agent.
2. **The terminal UI's coupling is one-directional and shallow.** `onit.py` reads 16 attributes off `chat_ui` (listed in 3.4) and writes 4. `chat.py` reads 26 methods off it. That is the whole surface.

### 3.3 Target structure

```
onit/
  contract/                 ← NEW. The only module both sides may import.
    __init__.py
    ui.py                   ChatUIProtocol, StreamingSink, ApprovalRequest
    events.py               the 10 SSE event names + payload dataclasses
    config.py               UISettings: the config subset a UI may read
    version.py              CONTRACT_VERSION = 1
  harness/                  ← the agent, moved behind the contract
    onit.py                 OnIt (no UI imports)
    runner.py               the three whole-run paths, unified (3.6)
    session_io.py           session JSONL + index + run state (3.5)
    ...
  ui/
    text.py                 imports contract + harness.session_io only
    api.py                  imports contract + harness.session_io only
    commands.py             imports contract + harness.serving only
    doctor.py               imports contract + harness.doctor_api only
  static/                   ← MOVED OUT of onit/ui/ (3.7)
```

`onit/contract/` has **no imports from `onit.harness` or `onit.ui`** — only stdlib and `pydantic`. That is checkable in one line and is the load-bearing rule of the whole section.

### 3.4 The contract, concretely

Three types. Everything a UI needs is one of them.

```python
# onit/contract/ui.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class ChatUIProtocol(Protocol):
    """What the agent loop may call on a UI.  Version 1."""
    # streaming
    def stream_start(self) -> None: ...
    def stream_token(self, token: str) -> None: ...
    def stream_think_token(self, token: str) -> None: ...
    def stream_end(self, elapsed: str = "") -> None: ...
    def stream_fold(self, summary: str = "") -> None: ...
    # tools
    def add_tool_call(self, name: str, arguments: dict) -> None: ...
    def add_tool_result(self, name: str, result: str, truncate: int = 300) -> None: ...
    def show_tool_start(self, name: str, arguments: dict) -> None: ...
    def show_tool_done(self, name: str, result: str, success: bool = True) -> None: ...
    def start_tool_spinner(self, name: str, arguments: dict) -> None: ...
    def stop_tool_spinner(self) -> None: ...
    def start_tool_batch(self, calls: list) -> None: ...
    def end_tool_batch(self) -> None: ...
    def tool_log(self, name: str, data, level: str = "info") -> None: ...
    def tool_progress(self, name: str, elapsed_seconds: int) -> None: ...
    # run events
    def add_log(self, message: str, level: str = "info") -> None: ...
    def show_context_compaction(self, orig_msg_count: int, summary_chars: int) -> None: ...
    def show_turn_limit(self, limit: int) -> None: ...
    def verification_start(self) -> None: ...
    def verification_end(self, answer: str, note: str) -> None: ...
    def verification_correction(self, answer: str, note: str) -> None: ...
    # context / budgets
    def set_context_usage(self, pct: float, max_tokens: int = 0) -> None: ...
    def set_token_budgets(self, max_output_tokens: int = 0, max_context_tokens: int = 0) -> None: ...
    def set_turn_context(self, tools_run: int = 0) -> None: ...
    def set_metrics(self, sink: dict) -> None: ...
    # attributes the loop writes
    model_name: str
    provider_name: str
```

This is the existing `onit/ui/protocol.py` verbatim, moved. It is already accurate — `chat.py` calls exactly these 26 methods. The move is mechanical; the *enforcement* is the new part:

```python
# onit/test/test_contract_conformance.py
def test_every_ui_implements_the_protocol():
    from onit.contract.ui import ChatUIProtocol
    from onit.ui.text import ChatUI
    from onit.harness.runner import StreamingAdapter
    for impl in (ChatUI, StreamingAdapter):
        assert isinstance(impl, ChatUIProtocol)   # runtime_checkable
```

`StreamingAdapter` moves out of `onit.py` into `harness/runner.py` and becomes the **only** UI the web and native clients ever see. It already is — `WebApiUI` never reaches `chat()`.

**`UISettings`** replaces the four `getattr(self._onit, "config_data", None)` reads in `api.py`. It is a frozen dataclass built once at startup:

```python
@dataclass(frozen=True)
class UISettings:
    title: str
    brand: str
    rating_enabled: bool
    voice_enabled: bool
    voice_sample_rate: int
    show_logs: bool
    html_preview: bool
    agent_cursor: str
    contract_version: int = CONTRACT_VERSION
```

The web UI's `/api/config` returns it. The native app decodes it. Neither reads `config_data`, so a harness config key can be renamed without touching a UI — which is exactly the failure this section exists to prevent.

### 3.5 The session store becomes a module, not a habit

`api.py` imports seven functions from `onit.sessions` and, separately, **opens the session JSONL itself** (`_load_history`, line 934 onward) and parses `{"task", "response"}` out of it. That is a file format the harness writes in three different places (`onit.py:1951`, `:2327`, `:2553`) and the UI reads in one. It is the single most likely place for a harness change to silently break a UI.

Fix: one module owns the format.

```python
# onit/harness/session_io.py
@dataclass(frozen=True)
class Turn:
    index: int          # 1-based, matches the trajectory store's turn number
    task: str
    response: str
    timestamp: float

def read_turns(session_path: str) -> list[Turn]: ...
def append_turn(session_path: str, task: str, response: str) -> Turn: ...
def turn_count(session_path: str) -> int: ...
```

All three harness write sites call `append_turn`. `api.py::_load_history` calls `read_turns`. The format is then a private detail of one file, and a test asserts the three write paths produce identical records.

### 3.6 Three run paths become one

The harness currently has **three** near-duplicate whole-run implementations, and they have already drifted:

| Path | Entry | Writes session | Records trajectory | Persists run state |
|---|---|---|---|---|
| Web / native | `process_task` (`onit.py:1768`) | yes (1951) | yes | yes |
| Terminal | `agent_session` + `_handle_successful_response` (`:2741`, `:2487`) | yes (2553) | yes | yes |
| Loop | `run_loop` (`:2255`) | yes (2327) | yes | yes |

The drift is visible: `run_loop` passes `adapter=None` to `_record_trajectory`, so loop-mode runs never record which model answered; `agent_session` passes `self.chat_ui` directly rather than an adapter. A harness change made in `process_task` does not reach the other two.

Fix: one `run_turn()` in `harness/runner.py` that takes a `ChatUIProtocol | None` and does the eight steps in one place — build instruction, call `_chat_with_failover`, stop sandbox on interrupt, append the turn, update the index, persist run state, record trajectory, schedule the deep check. The three callers become thin: they differ only in *where the task comes from* and *where the answer goes*.

This is the change that most reduces the cost of future harness work, and it is the one that makes the UI boundary real: after it, a UI is a `ChatUIProtocol` and a task source, nothing else.

### 3.7 `onit/ui/static/` moves out of the Python package

The browser SPA (9 files, `app.js`, `style.css`, `vendor/`) lives at `onit/ui/static/` and is shipped via `pyproject.toml` `package-data`. It is **not** part of the Python UI — it is the web client, and it changes on a completely different cadence.

Move to `web/` at the repo root. `api.py` serves it from a path resolved at startup (`ONIT_STATIC_DIR`, defaulting to the packaged location for wheels). Benefits: the SPA can be rebuilt and deployed without touching the Python package; a Python-only change does not invalidate the SPA's cache; and the native app's Tauri shell (§10) can point at the same directory.

### 3.8 Enforcement: three tests that keep it true

Isolation that is not tested decays. Three tests, all cheap:

1. **`test_no_harness_imports_ui`** — parse every module under `onit/harness/` (and `onit/model/`, `onit/mcp/`, `onit/learn/`) with `ast`, assert no import of `onit.ui.*`. This is the I1 gate. It fails today on five sites; it must pass before §4 starts.
2. **`test_contract_has_no_harness_imports`** — same, for `onit/contract/`. Guards the load-bearing rule of 3.3.
3. **`test_ui_imports_only_contract`** — every module under `onit/ui/` may import `onit.contract.*` and the named data-layer modules (`onit.harness.session_io`, `onit.harness.serving`, `onit.harness.doctor_api`), and nothing else from the harness. This is the I2 gate.

A fourth, optional and recommended: **`test_ui_suite_runs_without_agent`** — run `onit/test/test_web_api.py` with `onit.harness.onit` blocked from import, proving the UI suite does not need the agent to exist. It will fail today; it is the I3 gate.

### 3.9 What is deliberately *not* isolated

- **The sidecar is bundled, not separate.** The native app ships a frozen Python runtime (§5.4). A harness change therefore requires rebuilding the app — I4 is not achieved. Making it true would mean shipping the harness as a separately-versioned download, which trades a build step for a compatibility matrix. Not worth it for v1; revisit if the harness ships more often than monthly.
- **The CLI stays in the harness package.** `onit/cli.py` is the harness's own entry point, not a UI in the sense this section means. It imports `ui.doctor` today (one site) and will import `harness.doctor_api` after 3.2 — a harness→harness edge, which is fine.
- **`onit/ui/voice.py` keeps its `remove_tags` import.** `lib/text` is a leaf utility with no harness dependencies; routing it through the contract would add a hop for no isolation gain.

### 3.10 Sequencing

**Status: the thin half has landed.** The original plan called for four commits over 4–5 days.
Measurement cut it roughly in half, to ~2 days, by dropping the parts that are not isolation
problems. What shipped:

| Commit | Change | Gate |
|---|---|---|
| 1 | Create `onit/contract/`; move `protocol.py` into it; update the three importers and the docstrings that named the old path | `test_harness_does_not_import_a_ui_at_module_level` passes |
| 2 | Add `UISettings` + `OnIt.ui_settings`; replace the six `getattr(self._onit, ...)` reads in `api.py` with `self._settings()` | UI suite green with `onit.ui` blocked |
| 3 | Cut `onit.py`'s module-level UI imports: `ChatUI` and `dispatch` become function-body imports, `Message` becomes a duck-typed rebuild | `import onit.onit` succeeds with `onit.ui` blocked |

The measured result: `import onit.onit` now loads **zero** `onit.ui` modules, and
`onit/test/test_ui_isolation.py` (12 tests) fails the build if that changes.

**Deferred, deliberately.** These were in the original §3 and are *not* isolation problems —
they are harness-internal cleanups that were riding along:

| Deferred | Why it can wait |
|---|---|
| `run_turn()` unification of the three run paths | A harness concern, not a boundary one. Nothing about the native UI depends on it. |
| `harness/session_io.py` | Do it *with* the native UI, when the shape of what the client needs to read is known rather than guessed. |
| Full `ast` enforcement suite (three tests) | One test covers the failure that actually recurs: a module-level UI import. Add the rest when a second failure mode appears. |
| `dispatch` relocation out of `onit/ui/commands.py` | The import is now function-body and gated on `if not self.web`, so it costs nothing at import time. Moving the module is churn. |

**One correction to the original plan.** It listed five harness→UI imports and called all five
high severity. Three were module-level (`onit.py:49,50,51`); two were already function-body
(`onit.py:994`, `:2528`). Only the module-level ones actually coupled the harness to a front end
at import time. The `ChatUIProtocol` import (`:50`) was never a problem — it only needed to move
to `onit/contract/` so the *name* lives on the shared side.

**A second correction.** The plan said `api.py` reads `config_data` four times. It reads it six
times, plus `background_checks` and `_cancel_background_check` — eight `getattr` calls in total,
all now behind `UISettings`.

**Regression note.** The full suite has 9 failures and a large error cascade that are
**pre-existing and unrelated**: `test_mcp_server_runner.py::test_run_server_import_error` patches
`builtins.__import__` to raise, which breaks `contextlib`'s generator teardown and corrupts file
descriptors for every later test. Verified by running the suite at the parent commit — the
failure set is byte-identical. Worth fixing separately.


---

## 4. The backend contract (already implemented — do not change it)

This section is the specification the Swift client codes against. Every item is verified against the current source.

### 4.1 Launch

```
onit serve web --port <N> --no-login
```

`cli.py:1161` defines the subcommand; it sets `config_data["web"] = True`, `web_port`, and `web_require_auth`. `onit/onit.py:996` then constructs `WebApiUI`, and `WebApiUI.launch()` (`api.py:2124`) starts uvicorn on a daemon thread.

> **Two changes are required for the sidecar.** See §12. `launch()` currently binds `0.0.0.0` and prints to stdout; a desktop app needs `127.0.0.1` and a machine-readable ready signal.

### 4.2 Session model

| Concern | Endpoint | Notes |
|---|---|---|
| List | `GET /api/sessions` | Returns `{sessions:[{session_id, tag, preview, timestamp, processing}]}` |
| Create | `POST /api/sessions/new` | Returns `{session_id}` |
| Rename | `PATCH /api/sessions/{id}` | Body `{tag}` |
| Delete | `DELETE /api/sessions/{id}` | 404 for unknown/foreign ids |
| Delete all | `DELETE /api/sessions` | 409 if any session is processing |
| History | `GET /api/history` | Rehydrates a window on open |

Session identity travels in the **`X-Session-Id` header** (`_resolve_session`, `api.py:751`). The browser uses a cookie; the native app uses the header, which is cleaner and avoids cookie-jar state entirely.

### 4.3 Chat stream

`POST /api/chat` with `{message, files}` returns `text/event-stream`. A `409` means a task is already running in that session.

Event types, from the emit sites at `api.py:1098–1277`:

| Event | Payload | UI treatment |
|---|---|---|
| `token` | `{delta}` | Append to the answer bubble |
| `think` | `{delta}` | Append to a collapsed "working" disclosure |
| `phase_end` | `{content}` | Close the current phase |
| `status` | `{text}` | Transient status line (tool running) |
| `answer_start` | `{}` | Stop the spinner; answer is arriving |
| `approval` | `{approval_id, command, reason, subjects, expires_in}` | **Blocking modal** — see §7 |
| `approval_closed` | `{approval_id}` | Dismiss the modal |
| `correction` | answer + note | Post-`done` fact-check correction |
| `done` | `{content, files, elapsed, tok_s, timing, turn, rating_enabled}` | Finalize bubble, unlock composer |
| `error` | `{message}` | Error banner |

`think` is deliberately a separate channel from `token` — the source comment at `api.py:1104` says the browser folds reasoning away when the answer lands, "which it can only do if the two never share a channel." The native UI must preserve that separation.

### 4.4 Other endpoints

`POST /api/chat/stop`, `POST /api/approval`, `POST /api/clear`, `POST /api/rating`, `POST /api/upload`, `GET /api/config`, `GET /api/logs`, `GET /uploads/{sid}/{file}`, `GET /preview/{sid}/{path}`.

---

## 5. macOS build (phase 1)

### 5.1 Toolchain

Verified on the target machine: **macOS 26.6.2 (arm64)**, **Xcode 27.0**, **Swift 6.4**, **Python 3.13.12**.

- **SwiftUI** for all UI. No AppKit views except where SwiftUI has no equivalent (`NSSavePanel`).
- **Swift 6 strict concurrency.** The SSE reader is an `actor`; UI state is `@MainActor`.
- **No third-party Swift dependencies.** `URLSession` handles SSE fine with a byte-stream delegate; adding a package for it is not worth the supply-chain cost.
- **Minimum deployment target: macOS 14.** Below that, `@Observable` and the newer `NavigationSplitView` behaviours are unavailable, and the compatibility shims cost more than they save.

### 5.2 Project layout

```
apps/macos/
  OnIt.xcodeproj
  OnIt/
    OnItApp.swift              app entry, WindowGroup + Settings scene
    Model/
      Session.swift            @Observable session view-model
      SessionStore.swift       list, create, rename, delete
      ChatMessage.swift        role, content, reasoning, files
      AppSettings.swift        host, model, temperature, think
    Net/
      OnItClient.swift         URLSession, base URL, X-Session-Id
      SSEReader.swift          actor: byte stream -> AsyncStream<Event>
      Event.swift              Codable enum mirroring §4.3
    Sidecar/
      SidecarManager.swift     spawn, health-poll, restart, terminate
      PortAllocator.swift      bind :0, read port, close, hand to child
      SidecarPaths.swift       locate bundled binary; Application Support dir
    Security/
      KeychainStore.swift      SecItemAdd/CopyMatching/Delete
    UI/
      SidebarView.swift        session list, new/rename/delete
      ChatView.swift           transcript + composer
      MessageBubble.swift      markdown, code blocks, file chips
      ReasoningView.swift      collapsed disclosure for `think`
      ApprovalSheet.swift      blocking approval modal
      SettingsView.swift       model, host, keys, advanced
      StatusBar.swift          model · tok/s · elapsed
    Resources/
      Assets.xcassets
  Sidecar/                     PyInstaller build inputs
    onit-sidecar.spec
    build.sh
```

### 5.3 Multi-session UI

The requirement is "multiple sessions at the same time, each with its own memory." The backend already gives each session its own JSONL transcript and its own data directory (`_get_or_create_session`, `api.py:688`), so the UI's job is to make that visible and concurrent.

**Layout:** `NavigationSplitView`. Sidebar lists sessions with tag, preview and a live spinner for `processing`. Detail pane is the selected session's transcript.

**Concurrency:** each session gets its own `Session` view-model holding its own `AsyncStream<Event>` and its own in-flight `Task`. Switching sessions does **not** cancel the other one — a run in session A keeps streaming while you read session B. This is the whole point of the feature and it falls out of the backend for free, because `ApiSession.processing` is per-session.

**Optional second window:** `WindowGroup(for: Session.ID.self)` lets a session be torn off into its own window. Cheap to add once the view-model is per-session; defer to phase 1.5.

**Memory indicator:** show the session's turn count from `done.turn` and the context usage from `done.timing`. When the agent compacts context, surface it as a visible event rather than silent truncation.

### 5.4 Sidecar lifecycle

```
app launch
  ├─ locate sidecar binary in Contents/Resources/
  ├─ allocate ephemeral port (bind :0, read, close)
  ├─ spawn Process with:
  │     ONIT_DATA_PATH, ONIT_SESSIONS_PATH -> ~/Library/Application Support/OnIt/
  │     ONIT_HOST / ONIT_MODEL from settings (keys via env, see §6)
  ├─ poll GET /api/config until 200 (timeout 30 s)
  ├─ show main window
  └─ on quit: SIGTERM, wait 5 s, SIGKILL
```

**Port allocation** is the subtle part. Binding `:0`, reading the port and closing it before spawning the child is a TOCTOU race. Two mitigations, both cheap: retry the spawn up to 3 times on bind failure, and have the sidecar print its actual bound port to stdout so the app can confirm rather than assume.

**Restart on settings change.** Editing host or model writes config, terminates the sidecar, respawns, and re-opens the current session by id. Sessions survive because they are on disk.

**Crash recovery.** If the sidecar exits non-zero, show a banner with the last 50 lines of its stderr and a Retry button. Do not silently respawn in a loop.

### 5.5 Distribution

| Item | Choice |
|---|---|
| Signing | Developer ID, hardened runtime |
| Notarization | `notarytool`, stapled |
| Entitlements | `com.apple.security.cs.allow-unsigned-executable-memory` **not** needed; the sidecar is signed with the same identity |
| Sandbox | **Off for v1.** A sandboxed app cannot spawn an arbitrary helper that writes to `~/.onit`. Revisit in v2 with an XPC helper. |
| Update | Sparkle is the obvious choice; defer to phase 2 |

> **Honest note on the sandbox.** Turning it off is a real security decision, not a formality. The app runs agent-generated shell commands; the sandbox is one of the few things that would contain them. The plan accepts this for v1 because the agent already runs arbitrary commands by design, and a sandbox that permits that is not much of a sandbox. Document it in the README rather than let a reviewer discover it.

---

## 6. Keychain integration

The requirement is "keys stored in the keychain." The Python side already does this via `keyring` (`setup.py:209`), with `SERVICE_NAME = "onit"` and a per-endpoint scheme `endpoint_key:<url>`.

**The native app must use the same service name and the same key names**, or the CLI and the app will disagree about where a key lives.

### 6.1 Two options, and the recommendation

| Option | How | Problem |
|---|---|---|
| **A. Swift writes, Python reads** | App calls `SecItemAdd` with service `onit`; `keyring` reads it back | `keyring`'s macOS backend stores items with a specific attribute layout; hand-rolled `SecItemAdd` may not match it |
| **B. Python writes, Swift never touches secrets** | App sends the key to the sidecar over loopback; the sidecar calls `store_secret()` | Key crosses a socket (loopback only, but still) |
| **C. Shared helper** | App shells out to `security add-generic-password -s onit -a <key> -w <value>` | Matches what `keyring` does on macOS; no socket, no layout guessing |

**Recommendation: C for macOS, with B as the fallback on Linux/Windows.**

`keyring`'s macOS backend is a thin wrapper over the `security` CLI's keychain semantics, so `security add-generic-password -s onit -a <key> -w <value>` produces an item `keyring.get_password("onit", key)` will find. This keeps the secret out of the app's process memory after the write, out of any socket, and out of every file the app writes.

### 6.2 Rules the app must follow

1. **Never write a key to `~/.onit/config.yaml`.** The existing `commands.py:863` already warns about this; the native app must not regress it.
2. **Never log a key.** Redact in the stderr banner.
3. **Read keys only at sidecar spawn.** Pass via environment (`ONIT_HOST_KEY` etc.), which `get_secret` checks first (`setup.py:236`). The env var dies with the child.
4. **Show a masked field** (`••••••••`) with a Reveal button that requires Touch ID (`LAContext`).
5. **Deleting a key** calls `security delete-generic-password -s onit -a <key>` and also removes any `endpoint_key:` entry for that URL.

### 6.3 Settings screen

| Field | Storage | Notes |
|---|---|---|
| Host URL | `config.yaml` (`serving.host`) | Not secret |
| Model | `config.yaml` (`serving.model`) | Not secret |
| API key | **Keychain** | Per host; masked |
| Temperature / max tokens / think | `config.yaml` | Not secret |
| Second host (load balancing) | `config.yaml` (`serving.host2`) | Key in keychain |

A "Test connection" button calls `GET /api/config` against a temporary sidecar and reports the resolved model — this catches a wrong host before the user starts a session.

---

## 7. The approval flow (do not get this wrong)

The agent can run shell commands. The web UI gates them through a blocking modal: the `approval` event carries `approval_id`, `command`, `reason` and `subjects`, and the answer goes back via `POST /api/approval`.

Three properties the native UI must preserve, each of which the backend already enforces:

- **Session-scoped.** `pending_approvals` lives on the `ApiSession` (`api.py:548`), so a ticket raised in one session can only be answered from that session. The Swift client must therefore send `X-Session-Id` on the approval POST — not a global id.
- **Fail-closed.** Timeout, closed window, or no answer is a **deny** (`api.py:1141`). The native UI must not default to allow, and must not let a modal be dismissed into an implicit yes.
- **Expiring.** `expires_in` is in the payload. Show a countdown; let it run out.

**UI:** a sheet, not a notification. It names the exact command, the reason, and the subjects touched. Approve / Deny, with Deny as the default button. If the app is backgrounded, bring it forward — a gated command must never expire because the user did not see the prompt.

---

## 8. Speed and leanness

The brief says prioritize speed and lean code. Concretely:

| Target | Budget | How |
|---|---|---|
| App launch → window visible | < 400 ms | Window appears before the sidecar is ready; sessions list loads from disk cache |
| Sidecar ready | < 2 s | PyInstaller one-folder (not one-file — one-file unpacks to `/tmp` on every launch) |
| First token latency | unchanged | The sidecar is the same Python code; no added hop beyond loopback |
| App binary size | < 8 MB | No third-party Swift packages; SwiftUI only |
| Sidecar size | ~120–180 MB | Python + PyMuPDF + FastAPI; exclude `bench/`, `test/`, `datasets` |
| Idle CPU | ~0% | SSE stream is idle-blocked; no polling |

**Lean-code rules for the Swift side:**

- One view-model per session, no global mutable state.
- `AsyncStream<Event>` end to end; no Combine.
- No view model holds a `URLSession` task it does not cancel on deinit.
- Markdown rendering: use `AttributedString(markdown:)` for prose and a small custom renderer for fenced code. Do not pull in a Markdown package.

---

## 9. Linux (phase 2)

The sidecar is unchanged. Only the shell differs.

| Concern | Choice | Rationale |
|---|---|---|
| Toolkit | **GTK4 + libadwaita** via Swift (`SwiftGtk`) **or** a Rust shell | Swift on Linux has no first-party GUI; see below |
| Recommended | **Rust + `gtk4-rs`** | Mature, matches GNOME HIG, one binary, no runtime |
| Alternative | **Tauri** (Rust + system WebView) | Reuses the existing web UI almost verbatim; least new code |

> **Recommendation: Tauri for Linux.** The web UI already exists (`onit/ui/static/`). Wrapping it in Tauri gives a native window, a native menu, and a tray icon for roughly two weeks of work, versus two months for a GTK rewrite that will drift from the macOS UI. The macOS app stays SwiftUI because it must feel native on macOS; Linux users are better served by a fast, familiar webview shell than by a half-finished GTK clone.

**Keychain on Linux:** `libsecret` (`secret-tool`), which is what `keyring` uses. Same service name `onit`, same key names. If no secret service is running, fall back to the existing `~/.onit/secrets.yaml` with `0600` — and say so in the UI rather than silently downgrading.

**Packaging:** Flatpak (sandboxed, portal for filesystem access) and a `.deb`. Flatpak's sandbox interacts badly with "spawn a Python sidecar that writes to `~/.onit`" — plan for `--filesystem=home` in v1 and tighten later.

---

## 10. Windows (phase 3)

| Concern | Choice |
|---|---|
| Shell | **WinUI 3** (C#) or **Tauri** |
| Recommended | **Tauri**, for the same reason as Linux — one shell codebase covers both |
| Keychain | **Windows Credential Manager** via `wincred`; `keyring` uses this backend, same service name |
| Packaging | MSIX, or an Inno Setup installer for v1 |
| Sidecar | Same PyInstaller output, `.exe` |

**Windows-specific hazards:**

- **Path length.** `~/.onit/data/<uuid>/` under a deep user profile can exceed 260 chars. Enable long paths in the manifest and keep the data root short (`%LOCALAPPDATA%\OnIt`).
- **Console window.** A spawned `onit.exe` opens a console unless built with `--noconsole` or spawned with `CREATE_NO_WINDOW`. Use the latter so stderr is still capturable.
- **Antivirus.** A PyInstaller binary that spawns shells will be flagged. Sign it; submit for SmartScreen reputation early.
- **Line endings.** Session JSONL is written by Python with `\n`; do not let the shell rewrite it.

---

## 11. Required backend changes

The native app is a new client, but four small changes make it a *good* client. All are additive and none break the browser.

| # | Change | File | Why |
|---|---|---|---|
| 1 | Bind `127.0.0.1` when a `--host`/`ONIT_BIND` env says so | `ui/api.py:2147` | A desktop app must not expose the agent to the LAN |
| 2 | Print a ready line to stdout: `ONIT_READY port=<N>` | `ui/api.py:2147` | Removes the port race in §5.4 |
| 3 | `GET /api/health` returning `{status, model, provider}` | `ui/api.py` | Cheaper than `/api/config` for polling |
| 4 | Accept `X-Session-Id` on `POST /api/approval` | `ui/api.py:1569` | Already session-scoped server-side; make the client contract explicit |

Each is a few lines plus a test. They are worth doing **before** the Swift work starts, so the client is built against the final contract.

---

## 12. Milestones

| Phase | Deliverable | Duration | Exit criterion |
|---|---|---|---|
| **0a** | **Isolation §3 (done)**: `onit/contract/`, the three module-level harness→UI imports cut, `UISettings`, one enforcement test | 2 days | `import onit.onit` loads zero `onit.ui` modules |
| **0b** | Backend changes §12 + a `docs/NATIVE_API.md` freezing the contract | 3 days | Contract documented; tests green |
| **1a** | Sidecar build: PyInstaller spec, signed, launches, `/api/config` responds | 1 week | `onit-sidecar` runs standalone on a clean Mac |
| **1b** | Swift shell: window, sidebar, one session, SSE streaming | 2 weeks | A prompt streams into a native window |
| **1c** | Multi-session, per-session concurrency, history rehydration | 1 week | Two sessions stream at once |
| **1d** | Settings + Keychain + approval sheet | 1 week | Key written to Keychain, read by sidecar; gated command approves |
| **1e** | Signing, notarization, DMG, crash banner | 1 week | A colleague installs and runs it |
| **2** | Linux (Tauri + libsecret + Flatpak) | 3 weeks | AppImage/Flatpak runs on Ubuntu 24.04 |
| **3** | Windows (Tauri + Credential Manager + MSIX) | 3 weeks | Installer runs on Windows 11 |

**Total to a shippable macOS app: ~7 weeks** (6 + the isolation work, now done). Linux and Windows add ~6 weeks and share one shell codebase. Phase 0a was not optional: without it the Swift client is built against a boundary that moves under it, and every later harness change is a potential app break.

---

## 13. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PyInstaller misses a hidden import (FastMCP, PyMuPDF) | High | Sidecar fails at runtime | Smoke test in CI: build, launch, hit `/api/config` |
| Keychain layout mismatch between `security` CLI and `keyring` | Medium | Keys written by app invisible to CLI | Test both directions in phase 1d before building UI |
| Port race on spawn | Medium | Intermittent launch failure | Ready-line on stdout (§12.2) |
| Sidecar size bloats past 200 MB | Medium | Slow download, bad first impression | Exclude `bench/`, `test/`, `datasets`; measure in 1a |
| Approval modal missed while backgrounded | Low | Gated command silently denied | Bring app forward on `approval`; countdown visible |
| macOS sandbox expectations | Low | App Store rejection | Ship outside the App Store for v1; state it plainly |
| UI drift between macOS (SwiftUI) and Linux/Windows (Tauri) | High | Two UIs to maintain | Accept it: macOS gets the native UI, the others get the web UI in a native window. Document the divergence. |
| **Isolation refactor breaks the terminal UI** | Medium | The CLI is the daily driver; a regression is felt immediately | The cut imports are constructor-time only, so `OnIt(config)` behaves exactly as before. `test_chat_ui.py` and `test_onit.py` are the gate; both green. |
| **`session_io` changes the on-disk format** | Low | Existing sessions unreadable | It must not: the extraction is a pure move. A test reads a checked-in fixture session written by the current code and asserts `read_turns` returns the same records. |
| **Contract drift after phase 0a** | Medium | The boundary silently widens again | `test_ui_isolation.py` runs in CI on every PR; it fails the build, not a reviewer. |

---

## 14. What this plan deliberately does not do

- **It does not rewrite the agent in Swift.** The Python core is 500k+ lines of working logic; a rewrite is a different project.
- **It does not add a new protocol.** SSE over loopback is already implemented, tested, and used by the browser.
- **It does not put the app in the App Store.** The sandbox and the agent's need to run shell commands are incompatible in v1.
- **It does not unify the three UIs.** macOS is native; Linux and Windows are a webview shell. That is a deliberate trade of purity for shipping.

---

## 15. First three commits

1. `refactor(contract): extract onit/contract/ and enforce it` — §3.10 commit 1. `protocol.py` moves; `test_contract_has_no_harness_imports` lands with it.
2. `refactor(harness): stop importing the UIs` — §3.10 commit 2. The five sites in §3.2; `test_no_harness_imports_ui` lands with it.
3. `refactor(session): one module owns the session format` — §3.10 commit 3, plus the fixture test.
4. `feat(api): bind loopback and emit a ready line for desktop clients` — §12.1, §12.2, with tests.
5. `docs: freeze the native client contract` — `docs/NATIVE_API.md`, generated from the event table in §4.3.
6. `build(macos): PyInstaller spec for the onit sidecar` — `apps/macos/Sidecar/`, plus a CI smoke test that launches it and polls `/api/config`.

Commits 1–3 are the isolation work and land before any Swift is written. Everything after that is the Swift app, built against a contract that no longer moves.
