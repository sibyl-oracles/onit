# Building the OnIt macOS app

A step-by-step guide to building and running the native macOS client. Written
for someone who has never built it before and wants to test it.

The app is a SwiftUI shell that talks to the OnIt harness over the same HTTP
contract the browser UI uses. It does not embed Python: it spawns
`python -m onit.cli serve web` on a free loopback port and drives it with
`URLSession`.

---

## 1. What you need

| Requirement | Version | Check |
|---|---|---|
| macOS | 14 (Sonoma) or newer | `sw_vers` |
| Xcode | 15 or newer (for the Swift 6 toolchain) | `xcodebuild -version` |
| Swift | 6.x | `swift --version` |
| Python | 3.10+ with OnIt's dependencies installed | `python3 -c "import onit"` |

The app is built with `swift build`, so **you do not need an Xcode project**.
Xcode only has to be installed for its toolchain.

> **Note on the `onit` command.** On some machines the `onit` console script is
> stale and fails with `ModuleNotFoundError: No module named 'src'`. The app
> does not use it — it runs `python -m onit.cli` — so this does not block you.
> If you want the CLI itself working, reinstall with `pip install -e .` from the
> repo root.

---

## 2. Build the app

From the repository root:

```bash
cd apps/macos
./Scripts/build.sh
```

That is the whole build. It takes about 10 seconds and produces:

```
apps/macos/build/OnIt.app
```

To build and launch in one step:

```bash
./Scripts/build.sh --run
```

For a faster, larger debug build:

```bash
./Scripts/build.sh --debug
```

### What the script does

1. Runs `swift build -c release`.
2. Creates the `.app` bundle structure (`Contents/MacOS`, `Contents/Resources`).
3. Writes `Info.plist` with the version read from `pyproject.toml`.
4. Ad-hoc signs the bundle so it launches on the machine that built it.

It does not bundle Python. See §6.

---

## 3. Run it

```bash
open apps/macos/build/OnIt.app
```

On first launch the app:

1. Finds a Python that can `import onit`.
2. Starts the harness on a free port (chosen by the OS, not fixed).
3. Polls `GET /api/config` until it answers.
4. Loads the session list.

You should see a three-column window: a narrow connection rail on the left,
the session list, and the transcript. The window appears **before** the harness
is ready — the harness takes 15–30 seconds to boot because it discovers MCP
tools — and shows a "Starting the harness…" pane with the sidecar's log tail
while it waits.

### If it does not start

The app writes a trace to stderr. To see it, run the binary directly:

```bash
apps/macos/build/OnIt.app/Contents/MacOS/OnIt
```

You will see lines like:

```
[onit-app] start: python=/opt/miniconda3/bin/python3 root=/Users/you/github/sibyl/onit
[onit-app] start: spawned pid 45164 on port 60754
[onit-app] start: ready on 60754
```

If the first line says `python=nil`, no interpreter was found. If it says
`root=nil`, the app could not locate the `onit` package. Both are fixable with
the environment variables in §4.

The app also shows the failure reason and the sidecar log in the window, and
in **Settings → Diagnostics**.

---

## 4. Pointing the app at a different harness

The app resolves two things at startup, and both can be overridden.

| Variable | Meaning | Default |
|---|---|---|
| `ONIT_SIDECAR_PYTHON` | Interpreter that runs the harness | First of `/opt/homebrew/bin/python3`, `/opt/miniconda3/bin/python3`, `/usr/local/bin/python3`, `/usr/bin/python3` that can `import onit` |
| `ONIT_SIDECAR_ROOT` | Directory containing the `onit/` package | Walks up from the app binary looking for `onit/__init__.py` |

To launch with an override:

```bash
ONIT_SIDECAR_ROOT=/path/to/onit \
ONIT_SIDECAR_PYTHON=/opt/miniconda3/bin/python3 \
  apps/macos/build/OnIt.app/Contents/MacOS/OnIt
```

`open` does not pass your shell environment to the app. If you need the
overrides with `open`, use `launchctl setenv` or run the binary directly as
above.

---

## 5. Connecting to a remote OnIt

The rail's **+** button adds a second connection. Give it the address of an
OnIt running elsewhere:

```bash
# on the other machine
onit serve web --port 9000 --no-login
```

Then in the app: **+** → address `http://192.168.1.20:9000` → **Connect**.

The app checks `GET /api/config` before adding the connection, so a wrong
address fails immediately rather than producing an empty window. Sessions on
the remote host are independent of the local ones; switching between them does
not stop a run in progress on either.

---

## 6. How the harness is found (and why it is not bundled)

This build runs the harness from your working tree. That is a deliberate
trade for the first testable version:

- **What you get:** the app runs in minutes, and it always talks to the code
  you are editing. Change a Python file, restart the app, and the change is
  live.
- **What you do not get:** a self-contained `.app` you can hand to someone who
  does not have the repo.

Bundling the harness with PyInstaller is phase 1a of the plan
(`docs/NATIVE_UI_PLAN.md` §12) and is not done yet. When it is, the app will
find `Contents/Resources/onit-src/` first — the lookup order in
`SidecarPaths.packageRoot()` already prefers it.

---

## 7. What works today

Verified against a live harness:

| Feature | Status |
|---|---|
| Launch, spawn sidecar, health-poll | ✅ |
| Session list, grouped by recency | ✅ |
| Create, rename, delete session | ✅ |
| Streaming answers (token + think channels kept separate) | ✅ |
| Folded reasoning disclosure | ✅ |
| Folded tool ledger | ✅ |
| Markdown with fenced code blocks and copy button | ✅ |
| File attachments (upload + download chips) | ✅ |
| Inline approval card with countdown | ✅ |
| Stop a running turn | ✅ |
| Rating (thumbs up/down) | ✅ |
| Multiple connections in the rail | ✅ |
| Settings → Diagnostics with sidecar log | ✅ |

### Not built yet

| Feature | Why | Plan reference |
|---|---|---|
| Permission mode in composer | Needs a backend session flag | §11.5 |
| Keychain | Sidecar uses the existing config | §6 |
| Signing and notarization | Ad-hoc only | §5.5 |
| Bundled Python | Phase 1a | §12 |
| `/api/health`, `ONIT_READY` line | Backend changes not made | §11 |

The composer deliberately has **no** permission-mode menu. A menu that says
"Allow All" while the agent keeps asking would be worse than no menu.

---

## 8. Testing the approval card

The approval card only appears when the harness's command policy decides to
**ask**. On a default install it often does not, for two reasons worth knowing:

**Your settings may allow everything.** If `~/.onit/settings.json` contains
`"allow": ["Bash(*)"]`, every command matches and nothing is ever asked.

**The command allowlist is off outside a container.** `_allowlist_enforced()`
returns false on bare metal, so `permissions.allowedCommands` is not consulted.

To exercise the card, start the harness with the allowlist forced on:

```bash
ONIT_COMMAND_ALLOWLIST=1 \
  python -m onit.cli serve web --port 9000 --no-login
```

Then ask the app to run something not in the allowlist:

> Use your shell tool to run exactly: cowsay hello

You should see an inline card in the transcript naming the command, the reason,
and a countdown from 900 seconds, with **Deny** as the default button.

Rules in `settings.json` must be written as `Bash(...)` patterns. A plain
string like `"cowsay"` is silently ignored by the loader.

> **A note on what "ask" means on the web UI.** A miss against a non-empty
> `permissions.allow` list is a `system`-kind refusal, and `system` is withheld
> on a multi-tenant deployment — so it is **denied**, not asked. The askable
> path is the command allowlist (`command` kind). This is why the card appears
> with `ONIT_COMMAND_ALLOWLIST=1` and not with a restrictive `allow` list.

---

## 9. Where the code lives

```
apps/macos/
  Package.swift               Swift package manifest
  Scripts/build.sh            the build
  OnIt/
    OnItApp.swift             app entry, WindowGroup + Settings scene
    Model/
      AppModel.swift          sidecar + connections + sessions
      Session.swift           one conversation, its own stream and task
      Connection.swift        one OnIt: local sidecar or remote host
      ChatMessage.swift       one turn, reasoning kept apart from answer
    Net/
      OnItClient.swift        the whole HTTP contract, one file
      SSEReader.swift         actor: bytes -> events
      Event.swift             Codable mirror of the SSE event set
    Sidecar/
      SidecarManager.swift    spawn, health-poll, terminate; port allocation
    UI/
      RootView.swift          the three-level split
      ConnectionRail.swift    the rail, and the add-host sheet
      SessionListView.swift   grouped list, rename, delete
      ChatView.swift          transcript + composer
      MessageBubble.swift     markdown, reasoning, files, rating
      ToolLedger.swift        folded steps
      ApprovalCard.swift      inline approval
      MarkdownText.swift      fenced code blocks
      SettingsScene.swift     General / Connections / Diagnostics
    Theme/
      Typography.swift        13 pt body, 1.45 line height
      Palette.swift           system colours, Color.accentColor
```

The network layer is three files and has no dependency on SwiftUI. That is
intentional: it is the part that mirrors the backend contract, and it is the
part that should be testable without a window server.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "The harness did not start" | No Python found, or `onit` not importable | Set `ONIT_SIDECAR_PYTHON` and `ONIT_SIDECAR_ROOT` |
| Window opens, stays on "Starting…" | Harness still booting (15–30 s is normal) | Wait; check Diagnostics for the log tail |
| "No OnIt answered at …" when adding a host | Wrong address, or that host is not running `serve web` | Check `curl http://host:port/api/config` |
| Answers never stream | Port blocked, or the harness died mid-run | Diagnostics tab shows the sidecar log |
| Approval card never appears | See §8 | Force `ONIT_COMMAND_ALLOWLIST=1` |
| `swift build` fails with a toolchain error | Xcode too old | Install Xcode 15+ |

---

## 11. Uninstalling

```bash
rm -rf apps/macos/build
```

The app stores nothing of its own outside the bundle. Sessions live in
`~/.onit/sessions`, shared with the terminal and browser UIs.
