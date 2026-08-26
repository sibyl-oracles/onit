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
| `ONIT_CONTAIN_THRESHOLD` | Critical refusals before auto-containment. Default `0` (disabled); set a positive number to enable. |
| `ONIT_CONTAIN_WINDOW` | Seconds the containment counter looks back over. Default `600`. |
| `ONIT_APPROVAL_CHANNEL` | `1` when this run has a person who can approve commands. Set automatically by `onit` for the terminal and web UIs; never set for `serve a2a`, gateways or `--loop`. |
| `ONIT_ASK_APPROVAL` | `0` disables approval prompts, so every unlisted command is refused outright. |

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

## Command Approvals

An allowlist has three possible answers, not two, and OnIt now uses all three:
**allow**, **ask a person**, and **refuse**. A command that matches nothing —
an executable nobody thought to list, an install nobody pre-approved — used to
come back as a refusal, which is why an agent would spend turns rewording a
command that was never going to run. Those now pause and ask.

```
mytool --version
  ⚠  OnIt wants to run a command that policy does not allow on its own
     mytool --version
     Command blocked: 'mytool' is not in the command allowlist.
     [y] run once   [a] allow for this session   [n] refuse (default)
```

In the web UI the same question appears as a card in the transcript with three
buttons. The run is paused until it is answered.

- **Run once** — this exact command, this one time. The ticket is bound to the
  command text and the session directory, is single-use, and expires after 5
  minutes.
- **Allow for this chat** — adds what was asked about (the executable, the
  directory, installs for that tool) to the session's own allowlist. It lives
  in memory, in that session only, and is gone when the process exits.
- **Refuse** — the command is not run, and the same command is refused rather
  than re-asked for the rest of the session.

One command is one question, however many things it needs approving for.
`ls ~/project; mytool ~/notes` names two directories outside the jail and an
unlisted executable, and all three appear in a single prompt — approving them
one at a time would mean answering, watching the command fail on the next one,
and answering again.

### `--auto`

`onit --auto` answers every prompt with yes, so an unattended run never stops:

```bash
onit --auto serve a2a          # a server with no one watching
onit --auto --loop 30m "check the build"
onit --container --auto        # a sandboxed run that should not block
```

The flag substitutes for the *person*, not for the policy. It answers tickets,
and a refusal is not a ticket — so `sudo`, `docker`, `ssh`, `systemctl`,
operator `deny` rules and everything else in the "never" rows above stay
refused with `--auto` exactly as without it. What it does grant, silently, is
the askable set: unlisted executables, unpinned installs, and (on a
single-user host) paths outside the session directory. Each one is written to
the run log as `--auto approved: <command>`, so what was allowed on your
behalf can be read back afterwards.

On `serve web` the flag applies to **every** session of the deployment. The
askable set there already excludes the path jail and the environment, so it
cannot hand one logged-in user another's data — but it does mean any of them
can run any executable. `ONIT_ASK_APPROVAL=0` refuses instead of asking, and
wins over `--auto`: with both set, nothing is asked and nothing is approved.

**Where there is nobody to ask, the answer is no.** `ONIT_APPROVAL_CHANNEL` is
set only for the terminal and web UIs; an A2A server, a gateway bot and a
`--loop` run keep the old fail-closed behaviour with the identical refusal
message.

### What a person may not approve

Approval can only lift restrictions that protect *the person answering*.
Anything protecting someone else stays machine-enforced:

| Refusal | Askable? | Why |
|---|---|---|
| Unlisted executable | yes | The allowlist already permits `python`, `node`, `perl` and `bash`; a linter it missed is not the boundary. |
| Unpinned install (`pip install requests`) | yes | Pinning is a reproducibility rule, and the pinned form is already allowed. |
| Path outside the session jail | terminal only | On the web UI the jail is the only thing separating logged-in users, and no session may waive it. |
| `env`, `ps`, `/etc/passwd`, `curl \| sh` | terminal only | The tool environment carries the *operator's* `GITHUB_TOKEN` on a shared deployment. |
| Settings-file `allow` rule miss | terminal only | On the web UI the operator's rules are the boundary around users. |
| Settings-file `deny` rule | **no** | Someone wrote it down explicitly. |
| `sudo`, `mkfs`, `shutdown`, kernel and firewall changes | **no** | Host-wide; not one user's to authorize. |
| `docker`, `kubectl`, `ssh`, `rsync`, `systemctl`, `crontab`, `useradd`, … | **no** | Access to other accounts, other hosts, or the machine itself — see `NEVER_ASK_COMMANDS`. |
| Installs in the containerized web UI | **no** | Sealed; there is no yes to give. |
| Anything the parser could not analyze | **no** | Nobody can approve what nobody can read. |

Approvals are per session and never persisted: a grant that outlived the
session that asked for it would be a capability nobody remembers giving.

## Auto-Containment

**Auto-containment is off by default** (`ONIT_CONTAIN_THRESHOLD=0`) and must be
opted into. Blocked commands are always blocked and logged regardless; the
threshold only controls whether repeated violations escalate to a lockdown.

When `ONIT_CONTAIN_THRESHOLD` is set to a positive number and that many
**critical** refusals happen within `ONIT_CONTAIN_WINDOW` seconds (default
600), the session auto-contains:

- `bash`, `serve start`, `write_file`, `edit_file`, `transform_text` and
  `send_file` refuse all further calls **for that session**;
- `serve`-managed background processes registered under that session are
  stopped;
- a marker file (`.onit-containment.json`, containing the violation log) is
  written to the session directory so containment **survives restarts**.

Read-only tools (`read_file`, `search_*`) keep working so the session can be
diagnosed. To lift containment, unset `ONIT_CONTAIN_THRESHOLD` (the check
short-circuits on `0`, so a stale marker is ignored without a restart), or
delete the marker file and restart the MCP server.

### What counts as a violation

Only **critical** refusals — the ones that mean something went wrong rather
than something was not anticipated:

- catastrophic system operations (`sudo`, `mkfs`, `shutdown`, kernel modules);
- an executable on the never-ask list (`docker`, `ssh`, `crontab`, …);
- a rule an operator wrote in `settings.json` under `deny`;
- **re-submitting a command a person already declined.**

An unlisted executable, an unpinned install and a path outside the jail do
*not* count. They are the ordinary friction of an agent finding the edges of
the policy — and they are now questions rather than refusals anyway.

Three properties, each of which used to be the opposite:

- The counter **decays**. It is a rate over `ONIT_CONTAIN_WINDOW`, not a
  process-lifetime total, so an agent that trips once an hour is not treated
  like one trying ten things in ten seconds.
- Containment is **scoped to the session** that earned it. The marker lives in
  that session's directory; only a refusal with no session context contains
  the server as a whole. One user's bad turn stranding every later session on
  a shared host was a denial of service wearing a safety feature's clothes.
- Benign path slips no longer accumulate, because they are no longer counted.
