# Isolation Modes


OnIt offers three isolation levels, plus optional sandbox delegation. They can be combined.

## MCP sandbox delegation

Delegates individual code-execution tool calls to an external MCP sandbox
provider. Complementary to `--container`, and useful when code should run on a
different machine than the agent.

There is no flag: configure an MCP server that provides the sandbox tools
(`sandbox_run_code`, `sandbox_install_packages`, `sandbox_stop`) under `mcp.servers`,
and OnIt routes code execution there automatically. Registering the provider *is*
the opt-in — a separate switch could only ever disagree with it.

## `--container`

Runs the entire OnIt process inside a hardened Docker container so a breach cannot reach the host OS.

```bash
onit --container                                          # interactive terminal in container
onit --container serve web                                # web UI, port 9000 published
onit --container serve a2a --port 9100                    # A2A server on custom port
onit --container --container-gpus all                     # NVIDIA GPU pass-through
onit --container --container-mount "$HOME/docs:/home/onit/documents:ro" \
  serve web                                               # expose host path read-only
```

The first run auto-builds the `onit:local` image from the repo `Dockerfile`. Subsequent runs reuse the image.

**Container sub-flags:**

| Flag | Description |
|------|-------------|
| `--container-gpus SPEC` | NVIDIA GPU pass-through (e.g. `all`, `"device=0,1"`). Requires NVIDIA Container Toolkit. |
| `--container-mount HOST:CONTAINER[:ro]` | Extra bind mount. Repeatable. Prefer `:ro`. |
| `--container-memory SIZE` | Hard memory cap (e.g. `16g`). Default: unlimited. |
| `--container-shm-size SIZE` | `/dev/shm` size (default: `4g`). Raise for PyTorch DataLoader. |
| `--container-tmp-size SIZE` | `/tmp` tmpfs size (default: `16g`). Backed by host RAM. |
| `--container-allow-installs` | Permit package installs in-container. Installs must still be version-pinned (`pip install name==1.2.3`). |

**Isolation posture:** non-root user, read-only rootfs (`--read-only`), `--cap-drop=ALL`, `no-new-privileges` (no sudo/setuid escalation), RAM-backed tmpfs for all ephemeral writes (`/tmp`, `~/.cache`, `~/.onit`), no host mounts by default, outbound network allowed. Persistent state (pip installs via `PIP_TARGET`, Hugging Face caches, session artifacts) lives on the named `onit-data` volume — never the rootfs. The AST command allowlist (below) is enforced by default inside the container.

**What crosses the boundary:**

| Resource | Default behavior |
|---|---|
| `~/.onit/config.yaml` | Bind-mounted read-only |
| Host keychain secrets | Passed as ephemeral env vars |
| Session data | Named volume `onit-data` (writable, persistent) |
| Ports | Published only for the active mode |
| Host filesystem | Nothing beyond config/secrets unless `--container-mount` is set |

**Published ports by mode:**

| Mode | Default port | Override |
|---|---|---|
| (terminal) | — (no ports) | — |
| `serve web` | `9000:9000` | `--port` |
| `serve a2a` | `9001:9001` | `--port` |
| `serve gateway viber` | `8443:8443` | `--port` |

See [DOCKER.md](DOCKER.md) for full details.

## `--unrestricted`

Runs OnIt with lifted filesystem restrictions on the host — the agent can read/write any path, use any working directory, and install packages freely (pip, apt, brew, etc.). Use only in trusted, isolated environments.

```bash
onit --unrestricted
```

Catastrophic commands (disk wipe, reboot, kernel module loading) are always blocked regardless of this flag, and an explicit `ONIT_COMMAND_ALLOWLIST=1` still enforces the AST command allowlist.

## Command Permission Rules

The bash tool honors optional allow/deny rules from `~/.onit/settings.json` (override the path with the `ONIT_SETTINGS` env var). **These rules apply to the web UI only** (`onit serve web`) — web sessions may be reachable by other users, so the configured restrictions must hold there. The local text UI is a trusted terminal session and ignores the default settings file, running with full privileges under the built-in policy. To enforce the rules in the text UI too, point `ONIT_SETTINGS` at the file explicitly. Rules use glob patterns matched against the command; deny always wins, and compound commands (`&&`, `;`, `|`) are checked segment by segment:

```json
{
  "permissions": {
    "allow": ["Bash(*)"],
    "deny": [
      "Bash(sudo *)",
      "Bash(npm install*)",
      "Bash(pip install*)",
      "Bash(brew install*)"
    ]
  }
}
```

- **deny** — commands matching any rule are refused.
- **allow** — when non-empty, every command (and each segment of a compound command) must match an allow rule. Leave it as `["Bash(*)"]` (or omit it) to only use the deny list.

When active (web UI, or explicit `ONIT_SETTINGS`), rules apply in **all** modes, including `--container` and `--unrestricted`, and file edits take effect without a restart. Non-`Bash(...)` rules are ignored.

## Command Allowlisting (AST-based)

On top of the glob rules, the bash tool can enforce a **command allowlist backed by real shell parsing**: every command string is parsed into an AST (pipelines, `&&`/`||`/`;` lists, loops, subshells, `$(...)`/backtick substitutions, `bash -c` payloads, `find -exec` targets), and **every executable found anywhere in the tree** must be on the allowlist. Wrapper commands (`env`, `nohup`, `timeout`, `nice`, `stdbuf`, `xargs`) are peeled off so they can't hide a payload, and dynamic command names (`$CMD`, `$(which x)`) are rejected outright. The parser **fails closed**: anything it cannot statically analyze (`case` statements, function definitions, arithmetic commands) is blocked.

| Env var | Effect |
|---|---|
| `ONIT_COMMAND_ALLOWLIST` | `1` = enforce everywhere, `0` = disable. Unset: enforced inside `--container`, off on the host. |
| `ONIT_ALLOWED_COMMANDS` | Comma-separated extra executables to allow (e.g. `mytool,deno`). |
| `ONIT_ALLOW_PACKAGE_INSTALL` | `1` = permit package-manager installs (pinned versions only). Set by `--container-allow-installs`. |
| `ONIT_CONTAIN_THRESHOLD` | Blocked commands before auto-containment. Default `0` (disabled); set a positive number to enable. |

The allowlist can also be extended in `settings.json` (read in the web UI, or when `ONIT_SETTINGS` is set explicitly):

```json
{
  "permissions": {
    "allowedCommands": ["mytool", "deno"]
  }
}
```

**Package managers are blocked by default** under allowlist enforcement. System package managers (`apt`, `yum`, `dnf`, `pacman`, `brew`, `apk`, `snap`) are never allowlisted — in-container the rootfs is read-only anyway. Language package managers (`pip`, `npm`, `gem`, `cargo`, `go`, `uv`, `pipx`) may run non-mutating subcommands (`pip list`, `npm ls`), but `install` requires `ONIT_ALLOW_PACKAGE_INSTALL=1` **and pinned versions**:

```bash
pip install requests==2.31.0     # OK (with installs enabled)
pip install requests             # blocked: not pinned
pip install -r requirements.txt  # blocked: cannot pin-verify
npm install left-pad@1.3.0       # OK
npx cowsay@1.5.0                 # OK (pinned one-off execution)
```

Lockfile-driven installs (`npm ci`, bare `npm install`) are allowed since versions come from the lockfile. `onit-install-ml` (the curated CUDA-matched ML installer) is allowlisted only when installs are enabled.

## Auto-Containment

**Auto-containment is off by default** (`ONIT_CONTAIN_THRESHOLD=0`) and must be opted into. Blocked commands are always blocked and logged regardless; the threshold only controls whether repeated violations escalate to a persistent server-wide lockdown.

When `ONIT_CONTAIN_THRESHOLD` is set to a positive number, policy violations (blocked commands) are counted per server process, and on reaching the threshold the bash MCP server **auto-contains**:

- `bash`, `serve start`, `write_file`, `edit_file`, `transform_text`, and `send_file` refuse all further calls;
- `serve`-managed background processes registered at the data-directory root are stopped;
- a marker file (`.onit-containment.json`, containing the violation log) is written to the data directory so containment **survives restarts**.

Read-only tools (`read_file`, `search_*`) keep working so the session can be diagnosed. To lift containment, unset `ONIT_CONTAIN_THRESHOLD` (the check short-circuits on `0`, so a stale marker is ignored without a restart), or delete the marker file and restart the MCP server.

Two properties to weigh before enabling it. The counter is a **process-lifetime total with no decay**, so violations accumulate across an entire session rather than measuring a rate. And on the host the most common violation is a benign path slip — an absolute path outside the session jail, e.g. `/etc/`, `/opt/homebrew/bin/`, or another session's data directory — not an adversarial command. A low threshold therefore tends to strand long-lived sessions over accumulated typos. Inside `--container` the container itself is already the filesystem boundary, and the path allowlist is skipped there.

The marker lives at the **data-directory root, not the session jail**, so containment is deliberately server-wide: one session's violations contain every later session on that host until the marker is removed.

