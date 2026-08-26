# MCP Tools


MCP servers start automatically. Tools are auto-discovered and available to the agent.

| Server | Transport | Description |
|--------|-----------|-------------|
| PromptsMCPServer | loopback socket | Prompt templates for instruction generation |
| ToolsLocalMCPServer | stdio (per user) | Bash, file operations, document and local search, GitHub |
| ToolsNetMCPServer | loopback socket | Web search and weather |

The split follows one rule: a tool that touches this session's files — or acts
under this user's credentials, as `github_repo` does — is served over a pipe
that belongs to one OnIt process. What is left is stateless lookups.

## Default tools

The tools servers register these by default (required parameters in **bold**; defaults in parentheses):

| Tool | Parameters | Purpose |
|------|------------|---------|
| `search` | **`query`**, `type` (`web`\|`news`, `web`), `max_results` (5) | Search the web or recent news. Web search uses the [Ollama web search API](https://ollama.com/blog/web-search) (`OLLAMA_API_KEY`) with automatic DuckDuckGo fallback; news search uses DuckDuckGo. |
| `fetch_content` | **`url`**, `extract_media` (true), `download_media` (false), `output_dir` (`data_path/media`), `media_limit` (10) | Fetch a URL and extract text, image, and video links. Handles PDFs. Optionally downloads media locally. |
| `get_weather` | `place` (auto-detect from IP), `forecast` (false) | Current weather and optional 5-day forecast. Requires `OPENWEATHER_API_KEY`. |
| `bash` | **`command`**, `cwd` (`data_path`), `timeout` (300) | Execute a shell command and capture stdout, stderr, and return code. |
| `read_file` | **`path`**, `mode` (`text`\|`tables`\|`images`, `text`), `encoding` (utf-8), `max_chars` (100000), `table_index`, `output_format` (`json`), `output_dir`, `min_size` (100) | Read a file, or extract structured tables (PDF/markdown) or embedded images (PDF). |
| `write_file` | **`path`**, **`content`**, `mode` (`write`\|`append`, `write`), `encoding` (utf-8) | Write content to a file, creating directories as needed. Files get owner-only access. |
| `edit_file` | **`path`**, **`old_string`**, **`new_string`**, `replace_all` (false), `encoding` (utf-8) | Edit a file by replacing an exact string with new content. |
| `serve` | **`action`** (`start`\|`stop`\|`status`\|`logs`\|`list`\|`restart`), `command`, `name`, `pid`, `cwd`, `lines` (50) | Run anything slower than `bash`'s 300s cap in the background — builds, installs, test suites, training runs — plus web servers and daemons. No time limit; poll with `status` and `logs`. |
| `grep` | **`path`**, **`pattern`**, `file_pattern` (`*`), `case_sensitive` (false), `include_hidden` (false), `max_results` (100) | Recursive regex search across files in a directory. Returns file, line number, and matching content. |
| `send_file` | **`path`**, `callback_url` | Send a file to a remote client — via HTTP POST when `callback_url` is given, otherwise as base64 (max 10MB). |
| `github_repo` | **`action`** (`create`\|`get`\|`list`\|`fork`\|`delete`), `name`, `description`, `private` (false), `auto_init` (true), `gitignore_template`, `license_template`, `org`, `per_page` (30) | Create, inspect, list, fork, or delete GitHub repositories. Requires `GITHUB_TOKEN`. |
| `search_document` | **`path`**, `mode` (`pattern`\|`context`, `pattern`), `pattern`, `query`, `keywords`, `case_sensitive` (false), `context_lines` (3), `max_matches` (50), `context_chars` (500), `max_sections` (5) | Search within a single document (text, PDF, markdown) by regex or by keyword/query relevance. |
| `index_documents` | `path` (`documents_path`, else `data_path`), `recursive` (true), `rebuild` (false), `chunk_size` (1600), `chunk_overlap` (200), `status_only` (false) | Ingest in-house documents (pdf, md, txt, csv, docx, xlsx) into the local search index. Incremental. |
| `local_search` | **`query`**, `top_k` (5), `method` (`hybrid`\|`bm25`\|`dense`, `hybrid`), `path` | Search indexed in-house documents. Auto-ingests the default corpus on first use. |

### Which paths can the tools touch?

All `path`, `directory`, and `cwd` parameters are validated against two sandbox roots:

- **`data_path`** — the read/write working directory. Defaults to `~/sandbox`. Precedence: `--data-path` CLI flag > `data_path` in the config YAML > `~/sandbox`. The CLI exports the resolved value as `ONIT_DATA_PATH` before starting the MCP servers, so agent and tools always agree on the same directory. Relative paths always resolve against `data_path`, never the process working directory.
- **`ONIT_DOCUMENTS_PATH`** — an optional read-only documents root for in-house data (also settable as `documents_path`).

A2A server sessions each work in their own subdirectory `<data_path>/<session_id>`, created automatically per session.

| Tools | Allowed roots |
|-------|---------------|
| `write_file`, `edit_file` | `data_path` only |
| `read_file`, `send_file`, `search_document` | `data_path` or `ONIT_DOCUMENTS_PATH` |
| `grep` (`path`), `bash` (`cwd`) | `data_path` or `ONIT_DOCUMENTS_PATH` |
| `index_documents`, `local_search` (`path`) | `data_path` or `ONIT_DOCUMENTS_PATH`; when `path` is omitted the corpus defaults to `ONIT_DOCUMENTS_PATH` if set, else `data_path` |

Paths outside the allowed roots are rejected. The checks are relaxed in `--container` mode (the container is the isolation boundary) and in `--unrestricted` mode.

On a single-user terminal run, a path outside the roots is put to the user as an
approval prompt rather than refused outright — the most common refusal there is
a benign one, and the person at the prompt already owns the files. On the web UI
it stays a refusal: the jail is what separates one logged-in user's session from
another's. See [Command Approvals](ISOLATION.md#command-approvals).

### Disabling tools

Some tools can be switched off via environment variables: `ONIT_DISABLE_WEB_SEARCH` (removes `search`), `ONIT_DISABLE_WEATHER` (removes `get_weather`), and `ONIT_DISABLE_LOCAL_SEARCH` (removes `index_documents` and `local_search`).

Connect to additional external MCP servers:

```bash
onit --mcp-sse http://localhost:8080/sse
onit --mcp-server http://localhost:8080/mcp
```

## Harness tools

Three more tools reach the model without an MCP server behind them: their
subject is the run itself, so they are answered in process.

| Tool | Parameters | Purpose |
|------|------------|---------|
| `context_status` | — | How full the context window is, how many turns and tool calls the run has taken, how many times it has been summarized, and which notes are saved. |
| `note_write` | **`key`**, **`text`** | Save a short note under `<data_path>/.onit/notes/`. It survives context summarization; writing the same key again replaces it. |
| `note_read` | **`key`** | Read a saved note back. |

The reason they exist: when the context fills, the conversation is summarized
and the detail in it is lost — and until now only the terminal was told. The
model could not see it coming and had nowhere to put a finding it wanted to
keep. Now it can check, write things down first, and is told when a
summarization has happened.

Notes are session state. They live under the session's `data_path` and go when
it does; nothing crosses between sessions. They are offered only to a run that
has tools of its own, and `serving.harness_tools: false` withdraws them along
with the prompt block that describes them.

## Large tool results

A tool result over ~8,000 characters is written whole to
`<data_path>/.onit/results/` and enters the conversation as its first 6,000
characters under a handle:

```
[result:0007 · local_search · 48,320 chars · showing the first 6,000]
<the opening of the result>
… [rest of this result: result_read("0007", offset=6000) or result_grep("0007", "pattern")]
```

| Tool | Parameters | Purpose |
|------|------------|---------|
| `result_read` | **`handle`**, `offset`, `limit` | A window of a stored result, up to 8,000 characters at a time. |
| `result_grep` | **`handle`**, **`pattern`**, `context` | Matching lines with surrounding context — faster than paging when you know what you are looking for. |

Before this, a large result was cut to 16,000 characters with the middle
discarded permanently, and the only way to recover any of it was running the
tool again — a network round trip for bytes the harness had already been given.
Now nothing is lost and recovery is a local file read, so a result the
conversation has moved past can be trimmed to ~1,200 characters instead of
6,000. On a six-tool research loop that takes peak prompt size down about 65%
and per-turn growth about 77%.

Results are session state on the same terms as notes: under the session's
`data_path`, gone when it goes, never shared between sessions. The oldest are
pruned past 200. `serving.result_store: false` restores the old hard cut.

## Running code (`run_code`) — off by default

`serving.code_execution: true` adds one more tool: a Python interpreter, one per
session, that keeps its variables between calls. Every registered tool is
available inside it as a function of the same name, so a task that is several
dependent steps runs as one block instead of one turn each:

```python
hits = local_search("Q3 revenue")[:3]
totals = {h["title"]: extract_tables(h["path"]) for h in hits}
print(totals)
```

Only what the code `print`s comes back; everything else stays as a live variable
for the next call. A failing tool raises `ToolError`, which the code can catch.
Six dependent steps measured at 7 model turns / 4.6 s as individual tool calls
run as 2 turns / 1.5 s this way — the saving is the prefill and decode of the
turns that no longer happen, so it grows with how slow the model is.

**Read this before enabling it.** The code runs in a child process with the same
privileges as OnIt — no AST allowlist and no path jail, unlike the `bash` tool.
Enable it where the deployment is already isolated (`onit --container`), and be
deliberate about a web deployment other people can reach. What is enforced: the
interpreter's working directory is the session's `data_path`, `session_id` and
`data_path` cannot be set from inside the code (they are not in the generated
signatures and are re-bound by the harness on every call), and each session gets
its own process. A block that overruns `serving.code_timeout` (120 s) is killed
and the interpreter restarts — the model is told its variables are gone.

It is also **unbenchmarked**: small models write worse Python than they write
JSON tool calls, so this can cost accuracy even as it cuts turns. Measure your
own model class with `benchmarks/` before relying on it.

