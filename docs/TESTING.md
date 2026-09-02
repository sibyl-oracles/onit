# Testing

Three layers, answering different questions:

- **The pytest suite** (`src/test/`) proves the code is correct in the abstract —
  mocks stand in for servers, no model is called, nothing listens on a port.
  Run it after every change; it is the fast, deterministic half.
- **The live self-check** (`onit doctor` from the shell, `\doctor` in the text
  UI) proves that *this machine's* stack is actually wired — the MCP servers
  this process starts answer, the tools execute a real round trip, the model
  endpoint replies. Run it after pulling changes and before trusting a
  session with real work.

## Setup

Install test dependencies:

```bash
pip install -e ".[test]"
```

## Run the full test suite

```bash
pytest src/test/ -v
```

## Run specific test modules

```bash
pytest src/test/test_onit.py -v          # Core agent tests
pytest src/test/test_cli.py -v           # CLI tests
pytest src/test/test_cli_doctor.py -v    # `onit doctor` CLI wiring tests
pytest src/test/test_a2a.py -v           # A2A protocol tests
pytest src/test/test_chat.py -v          # LLM chat tests
pytest src/test/test_viber.py -v         # Viber gateway tests
pytest src/test/test_chat_ui.py -v       # Terminal UI tests
pytest src/test/test_web_api.py -v       # Web UI tests
pytest src/test/test_mcp_prompts.py -v   # MCP prompt tests
pytest src/test/test_tool_discovery.py -v # Tool discovery tests
pytest src/test/test_mcp_tools_security.py -v # MCP tools security tests
pytest src/test/test_doctor.py -v        # Live self-check battery tests
```

## The live self-check

One battery, two front ends. `src/ui/doctor.py` holds the checks; the CLI
subcommand and the text-UI command are thin wrappers that build (or reuse) a
live agent and print the same report. Both run 22 fast checks against the
live stack — config, backslash commands, load balancer, MCP server
reachability, tool discovery, the shipped toolset, then **one round trip
per shipped tool** (bash echo; `write_file`→`read_file`→`edit_file`; grep
over a planted marker; `search_document` pattern mode; `index_documents` +
`local_search` over a dedicated probe corpus; `send_file` base64; a full
`serve` start/status/stop cycle; a web search; a `fetch_content` of
example.com; `get_weather` and `github_repo` when their credentials are
present), plus the harness's note tools, prompt assembly, the model
endpoint's model listing, session history, and trajectory recording — each
time-boxed, each cleaning up after itself:

```
Self-check (fast) — 22 passed in 4.4s

  ✓ config                3 endpoint(s) from serving.endpoints, 3 MCP server(s) (~/.onit/config.yaml)
  ✓ commands              8 commands parse and list
  ✓ load-balancer         3 endpoint(s), serving via server3
  ✓ mcp-servers           ToolsNetMCPServer:18203 up · ToolsLocalMCPServer: stdio spec ok · PromptsMCPServer:18204 up
  ✓ tool-registry         14 tools
  ✓ default-tools         all 14 default tools discovered
  ✓ tool-bash             echo round trip ok (onit-doctor-1788338977)
  ✓ tool-files            write/read/edit round trip ok (doctor-probe-8034-1788338978.txt)
  ✓ tool-grep             grep found the planted marker (1 match)
  ✓ tool-search-document  pattern search found the planted marker (1 match)
  ✓ tool-local-search     index + local_search round trip ok (631 doc indexed, retrieved)
  ✓ tool-send-file        base64 round trip ok (doctor-send-8034.txt)
  ✓ tool-serve            start/status/stop cycle ok (pid 8046)
  ✓ tool-search           web search returned 3 result(s)
  ✓ tool-fetch-content    fetched example.com and found its known text
  ✓ tool-weather          conditions for 東京都, 日本 came back
  ✓ tool-github           listed 1 repo(s) for the token
  ...
```

A check that cannot run here **skips** (`–`) rather than failing — no
`data_path` means the file-tool probe has nowhere to run; no
`OPENWEATHER_API_KEY` or `GITHUB_TOKEN` means the weather/GitHub probes
cannot exercise their tools. Those are facts about the configuration, not
evidence that the code broke.

### From the shell: `onit doctor`

```bash
onit doctor            # the 22 fast checks, a few seconds
onit doctor --deep     # + three live model turns (costs tokens)
onit doctor --json     # machine-readable report (for scripts and CI)
onit doctor --keep-session   # keep the throwaway session for reading a failure's details
```

The command builds a throwaway session exactly the way a real one starts —
MCP servers spawned, tools discovered, a session registered — runs the
battery against it, deletes the session again, and exits **0 when nothing
failed, 1 when any check failed** (skips do not fail the run). That exit
code makes it usable as a gate:

```bash
# after pulling changes: is this checkout's stack sound?
onit doctor || echo "self-check failed — fix before working"

# in an update script: refuse to continue on a broken stack
onit doctor --json > /tmp/doctor.json || exit 1
```

Because it runs its own MCP servers on freshly allocated ports, it can run
alongside a live session without touching it, and it never resumes or
modifies an existing session. On a machine that has never been set up, the
battery still runs and reports *what is missing* (no `serving.host`, no MCP
servers) rather than crashing — that diagnosis is the point.

### In the text UI: `\doctor`

Type `\doctor` in a running session. It runs the same 22 checks against that
session's own live stack — its MCP servers, its tool registry, its endpoint —
so it answers "is *this* session healthy", including things the CLI run
cannot see (a server that died mid-session, tools lost after a config
reload).

```
\doctor          # the fast battery
\doctor deep     # + three live model turns (costs tokens)
```

`deep` adds three checks that cost tokens: a live model reply through
`chat()` (no tools, one plain turn), a full tool-calling turn (the model
must call `bash` and report its output), and a file-reading turn (the
harness plants a probe file; the model must `read_file` it and echo its
contents — the read half of the tool loop, the path every document task
rides). Use it when the fast battery passes but tasks still misbehave —
that combination points at the model/loop layer rather than the wiring.

### Which one, when

| | `onit doctor` | `\doctor` |
|---|---|---|
| When | after pulling changes, in scripts/CI, before starting work | when a running session misbehaves |
| Stack checked | this machine's, freshly started | the live session's, as it stands |
| Session | throwaway, deleted after | the session you are in |
| Exit code | 0/1 — scriptable | report only |

Both report the same checks in the same words, so a failure found in one is
reproducible in the other. The checks live in `src/ui/doctor.py`; the CLI
wiring in `src/cli.py` (`_run_doctor`), the text-UI command in
`src/ui/commands.py` (`cmd_doctor`); their tests in `src/test/test_doctor.py`
and `src/test/test_cli_doctor.py`.

## Test structure

All tests are in `src/test/`:

| File | Description |
|------|-------------|
| `test_onit.py` | Core agent, A2A executor, disconnect middleware, session isolation |
| `test_cli.py` | CLI argument parsing and client mode |
| `test_cli_doctor.py` | `onit doctor`: subcommand wiring, exit codes, session cleanup |
| `test_a2a.py` | A2A protocol integration tests |
| `test_chat.py` | LLM chat interface tests |
| `test_viber.py` | Viber gateway and session isolation tests |
| `test_chat_ui.py` | Terminal UI tests |
| `test_web_api.py` | Web UI: auth, WebSession, per-tab session isolation |
| `test_mcp_prompts.py` | Prompt template tests |
| `test_mcp_server_runner.py` | MCP server launcher tests |
| `test_mcp_tools_security.py` | Security tests for MCP tools |
| `test_text_utils.py` | Text utilities tests |
| `test_tool_discovery.py` | Tool discovery tests |
| `test_tool_registry.py` | Tool registry tests |
| `test_doctor.py` | The live self-check battery (`\doctor`) |

## Configuration

pytest is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["src/test"]
asyncio_mode = "auto"
```
