# Release Notes

## v0.1.4b

### New Features

- **Native Web UI** — The web UI no longer runs on Gradio. It is now a FastAPI app with a hand-written front end (`src/ui/api.py` + `src/ui/static/`) served over SSE, with streaming tokens, inline tool-call detail during the thinking phase, code and image rendering, copy buttons, and a login screen (`--no-login` to disable). Gradio is gone from the dependency list.
- **Voice Mode** (`onit --voice`) — Full-duplex speech-to-speech. Audio is handled end to end by NVIDIA NemotronLabs VoiceChat 11B over an OpenAI-Realtime-compatible WebSocket (`--voice-url`); OnIt supplies the tools and the work. Install with `onit[voice]`. See [docs/VOICE.md](docs/VOICE.md).
- **Subcommand CLI** — Modes are now subcommands instead of flags: `onit setup`, `onit sessions`, `onit learn`, `onit resume`, `onit ask`, and `onit serve {a2a,web,gateway,loop}`. Plain `onit` still opens terminal chat.
- **Persistent Sessions** — Terminal chat resumes the last session by default. `onit sessions` lists, tags, rebuilds, and clears them; `--resume TAG_OR_ID` (or `last`) reopens one; `--restart-session` starts fresh.
- **Two-Host Load Balancing** — Serve from two endpoints at once with `--host2`/`--model2` and pick a policy with `--load-balancer {sticky,round_robin,random,least_busy}`. Endpoints are health-ranked, and Ollama endpoints stay in reserve unless `--no-ollama-fallback-only` puts them in normal rotation.
- **Agent Harness Capabilities** — All six harness phases from the NOOA framework landed: run-state budgeting, a result store that keeps large tool results readable instead of truncating them (`result_read` / `result_grep`), harness tools, early stopping, and answer verification. See [docs/HARNESS_CAPABILITIES.md](docs/HARNESS_CAPABILITIES.md).
- **Answer Verification** — Answers stream to the user and are then checked against the evidence the run gathered, in a fast stage the user waits for and a thorough stage that runs behind the delivered answer. Set `verify_answers: false` to hand back the answer unchecked.
- **Code as Action** (`code_execution`, off by default) — A per-session Python interpreter that keeps variables between calls and exposes every registered tool as a function, collapsing multi-turn tool chains into one block. Off by default: the code runs with OnIt's privileges and no path jail, so enable it only where the deployment is already isolated (`onit --container`).
- **Self-Improvement Trajectories** — OnIt records what its own runs did (`src/learn/`); `onit learn` shows the summary as a table or `--json`, and `onit learn --session ID` prints one trajectory. See [docs/SELF_IMPROVEMENT.md](docs/SELF_IMPROVEMENT.md).
- **Local Search over In-House Data** — A local search MCP server modeled on the Mistral Search Toolkit: parse → chunk → embed/index, then BM25, dense, or hybrid retrieval over PDF, Markdown, text/CSV, DOCX, and XLSX. Documents and index never leave the machine. Install with `onit[search]`. See [docs/LOCAL_SEARCH.md](docs/LOCAL_SEARCH.md).
- **Per-User MCP Isolation** — MCP tool servers are started per user, and MCP ports are chosen after the default servers exist, so concurrent users on one host no longer share tool-server state or collide on ports. See [docs/ISOLATION.md](docs/ISOLATION.md).
- **Bash Command Policy** — Command classification (`src/mcp/servers/tasks/os/bash/command_policy.py`) routes long-running commands to `serve` rather than the bash tool and enforces the blocklist in one place.
- **Container Mode** (`--container`) — Run the whole OnIt process inside a hardened Docker container, with GPU pass-through (`--container-gpus`), extra bind mounts (`--container-mount`), a memory cap (`--container-memory`), custom `/dev/shm` and `/tmp` sizes, and opt-in package installs (`--container-allow-installs`).
- **Target Environment** (`--target-env`) — Point the agent's bash tool at a specific conda or virtual environment's Python, pip, and binaries without leaving the host shell.
- **Unrestricted Mode** (`--unrestricted`) — Opt-in flag granting full host filesystem access, lifting the default sandbox path restrictions for power users and CI pipelines.
- **Streaming Output** — Chat responses stream token-by-token in the terminal UI, web UI, and A2A client, with a live tok/s indicator. `--no-stream` turns it off.
- **Auto Model Detection** — The model is auto-detected from the LLM endpoint, so vLLM hosts no longer need it set explicitly. `--model` still overrides.
- **Context Compaction** — Automatic context-window compaction summarises older messages as the limit approaches, with a streaming inline notice when it happens.
- **MLX and Ollama Cloud** — Added local MLX serving on Apple silicon and hosted Ollama cloud endpoints alongside vLLM, local Ollama, and OpenRouter. See [docs/MODEL_SERVING.md](docs/MODEL_SERVING.md).
- **VLM Web Image Fetch** — New MCP server (`VLMToolsMCPServer`) for fetching and processing images from web URLs in vision-language workflows.
- **Windows Support** — Platform-specific fixes for signal handling and the terminal UI.
- **SWE-bench Harness** — `onit[swe_bench]` extra and benchmark plumbing for running OnIt against SWE-bench through inspect-ai.

### Improvements

- **Docs Rewrite** — New guides for architecture, CLI, configuration, tools, isolation, model serving, local search, voice, HTTPS deployment, harness capabilities, and self-improvement; README simplified to a quick start.
- **Removed `--plan` and `--sandbox`** — `--plan FILE` and the `--sandbox` flag are gone. Sandbox path restrictions are now the default behaviour, lifted with `--unrestricted`; MCP sandbox delegation is configured through an MCP server rather than a flag.
- **Token Accounting** — Reworked token-length calculation and `max_tokens` handling across models with different context windows, plus a CLI override.
- **DeepSeek v4 Compatibility** — Continuation token budget raised (64 → 512) so models that prepend thinking tokens before tool-call JSON are not truncated mid-call.
- **Repetition Penalty** — Applied automatically for Ollama models (higher default with thinking mode off) to reduce output looping.
- **Show Logs Everywhere** — `--show-logs` applies to web UI and A2A server modes, not just terminal and gateway.
- **4 Default MCP Servers** — The unified Tools server is split into local and net profiles (`ToolsLocalMCPServer`, `ToolsNetMCPServer`), joining `PromptsMCPServer` and the new `VLMToolsMCPServer`.
- **a2a-sdk 1.0** — Upgraded to `a2a-sdk>=1.0.0` for improved A2A protocol compatibility.
- **Prompt Engineering** — Reworked prompt templates: more concise instructions, date awareness, and better plan generation.

### Bug Fixes

- Fixed token length calculation errors causing premature context truncation.
- Fixed context compaction triggering incorrectly in some conversation flows.
- Fixed planning continuation count not resetting after a successful tool call, which could make the agent give up early on multi-tool tasks.
- Fixed early stopping firing before the task was actually finished, and unfinished answers being returned as final.
- Fixed httpx-related MCP client errors and request timeouts.
- Fixed tok/s reporting, truncated output, and several terminal and web UI rendering bugs.
- Fixed git credential handling in container mode.
- Fixed `--show` not applying correctly in some invocation paths.

## v0.1.3c

### New Features

- **Session Persistence** — Browser sessions now survive server restarts. A session cookie (`onit_session`) ties each browser tab to its session, and sessions are restored from disk (JSONL files) on reconnect.
- **Inline File Upload (A2A)** — The `--file` flag now embeds files as base64 `FilePart` in the A2A JSON-RPC payload instead of uploading via a separate HTTP request. The A2A executor saves non-image files to the session data folder and appends file references to the task prompt.
- **Chat Clear** — The "Clear" button in the web UI now also clears the session working directory (uploaded/generated files) and resets the session JSONL history file.

### Improvements

- **Scroll Lock (Web UI)** — Added a MutationObserver-based scroll controller that prevents auto-scroll when the user has scrolled up (2px threshold). Spinner and response updates only re-render the chatbot component when there are actual changes, reducing unnecessary DOM thrashing.
- **Reduced Chatbot Re-renders** — The polling loop now returns `gr.skip()` for the chatbot and stop button when nothing has changed, avoiding re-renders that fight user scrolling.
- **Spinner Efficiency** — Spinner text only updates the DOM when the message actually changes (tick threshold), not on every poll cycle.
- **Session ID Validation** — Session IDs are validated against a UUID regex to prevent path traversal attacks via crafted cookies.
- **Tool Discovery** — `PromptsMCPServer` is now excluded from tool discovery (`discover_tools`) since prompts are not callable tools. Removed `list_prompts()` from the tool discovery pipeline.
- **Cleaner Footer** — Removed Gradio branding/footer links; replaced with a minimal "OnIt" attribution link.
- **File Path Display** — Chat history now shows friendly filenames (e.g. `📎 report.pdf`) instead of raw absolute paths for uploaded files.
- **Download Path Fix** — Files downloaded from A2A servers are now saved with `os.path.basename()` to avoid creating nested directories from session-scoped upload paths.
- **Prompt Wording** — Clarified "provide a download link to the file" in assistant prompt templates; simplified "Working directory or data_path" label.

### Bug Fixes

- Fixed regex pattern for file URL extraction to also exclude backticks and asterisks, preventing malformed download filenames from Markdown-formatted responses.

## v0.1.3b

### Security

- **Sandboxed Shell Execution** — The bash MCP tool now runs commands in a minimal, isolated environment that strips all inherited environment variables (API keys, tokens, etc.). Only essential variables (`PATH`, `LANG`, `HOME`, `DATA_PATH`, `DOCUMENTS_PATH`) are exposed.
- **Command Blocklist** — Blocked dangerous commands (`env`, `printenv`, `ps`, `top`) and access to sensitive system files (`/etc/passwd`, `/etc/shadow`, `/proc/self/environ`).
- **Path Restriction Enforcement** — All file read, write, directory listing, and bash operations are now validated against allowed directories (`DATA_PATH` and `DOCUMENTS_PATH`). Symlink traversal attacks are prevented using `os.path.realpath()`.
- **Read Path Validation** — Added `_validate_read_path()` and `_validate_dir_path()` to enforce read access boundaries on all file tools (`read_file`, `send_file`, `search_document`, `search_directory`, `extract_tables`, `find_files`, `transform_text`, `get_document_context`).

### Improvements

- **Unified `--show-logs` Flag** — Consolidated `--text-show-logs` and `--gateway-show-logs` into a single `--show-logs` flag that works across all modes (text, web, gateway).
- **Documents Path Propagation** — `--documents-path` is now propagated to MCP servers via the `ONIT_DOCUMENTS_PATH` environment variable, making mounted documents accessible in Docker deployments.
- **Telegram Concurrency** — Increased Telegram gateway concurrent updates limit from `True` (unbounded) to `256` for better resource control.
- **Docker Documentation** — Comprehensive Docker run examples for all modes (text, web, A2A, Telegram gateway, Viber gateway) with inline environment variables, volume mounts, and MCP port mappings.

### Bug Fixes

- Removed unused `gateway_show_logs` field from `OnIt` model — gateway now uses the unified `show_logs` setting.
- Removed stale `configs/default.yaml` and `docs/GOOGLE_WORKSPACE_AND_OAUTH.md`.

## v0.1.3a

### New Features

- **Per-Session Isolation (Web UI)** — Each browser tab now gets its own independent session with isolated chat history, file storage, and response routing. Multiple users can chat concurrently without seeing each other's messages or files. Sessions auto-cleanup after 24 hours.
- **Per-Session Isolation (A2A Server)** — Each A2A context (client conversation) gets its own isolated session with separate chat history, data directory, and safety queue. Different A2A clients no longer share state.
- **Concurrent Request Processing (Web UI)** — Web UI requests are now processed concurrently via `process_task()` (matching the Telegram/Viber gateway pattern), instead of sequentially through a single queue.
- **Viber Gateway** — Chat with OnIt remotely via a Viber bot. Supports text and photo messages with vision processing. Requires a public HTTPS webhook URL (see [Gateway Quick Start](docs/GATEWAY_QUICK_START.md)).
- **Gateway Auto-Detection** — `onit --gateway` now auto-detects Telegram or Viber based on which environment variable is set (`TELEGRAM_BOT_TOKEN` or `VIBER_BOT_TOKEN`).
- **Tunnel Documentation** — Comprehensive guide for tunneling options: Cloudflare Tunnel, ngrok, localtunnel, Tailscale Funnel, and SSH reverse tunnel.


### Improvements

- **Session-Scoped File Routes** — File uploads and downloads are now scoped per session (`/uploads/{session_id}/{filename}`), preventing file conflicts between users. Legacy `/uploads/{filename}` route preserved for backward compatibility.
- **Per-Session Stop** — The Stop button in the web UI now only cancels the current browser tab's task, not all users' tasks. A2A client disconnects similarly only cancel that client's in-flight request.
- **Friendly Error Messages** — All user-facing interfaces (terminal, web, Telegram, Viber, A2A) now return friendly messages instead of exposing internal error details. Server errors are logged via `logger.error()` for debugging.
- **Webhook Registration Timing** — Fixed race condition where Viber webhook was registered before uvicorn started accepting connections.
- **Logging** — Added `logger` to `chat.py` and `onit.py` so API errors (timeouts, connection failures) are always logged regardless of `verbose` setting.

### Bug Fixes

- Fixed `chat()` returning raw error strings to users on `APITimeoutError` and `OpenAIError` — now returns `None` so callers handle it consistently.
- Fixed `agent_session()` sending raw exception text to the output queue — now sends `None` to trigger the retry prompt.
- Fixed Telegram and Viber gateways exposing `f"Error: {e}"` to users on exceptions.

## v0.1.2

### New Features

- **Telegram Gateway** — Chat with OnIt remotely via a Telegram bot. Supports text and photo messages with vision processing.
- **VLM Integration** — Send images to A2A servers for vision-language model processing (`--a2a-image` flag).
- **Remote MCP Servers** — Connect to external MCP servers using `--mcp-sse` and `--mcp-host` flags.
- **Unified Tools MCP Server** — Consolidated web search, bash, filesystem, and document tools into a single `ToolsMCPServer`.

### Improvements

- **Standalone App Refactor** — OnIt is now a fully self-contained package installable via `pip install onit==0.1.2`.
- **Simplified CLI** — Streamlined command-line options and argument parsing.
- **Docker Compose** — Multi-service orchestration with `onit-mcp`, `onit-web`, `onit-a2a`, and `onit-gateway` services.
- **Prompt Engineering** — Date-aware prompts and improved prompt template handling.
- **Error Handling** — Better error recovery and user-facing error messages across all interfaces.

### Bug Fixes

- Fixed vLLM kwargs handling for tool calls.
- Fixed message formatting across terminal, web, and Telegram UIs.
- Resolved test failures and improved test coverage.
- Security fixes for bash and filesystem MCP servers.

### Dependencies

- Added: `a2a-sdk[all]`, `beautifulsoup4`, `python-dateutil`, `geopy`, `ddgs`, `pypdf`, `ollama`, `urllib3`, `PyMuPDF`
- Optional: `python-telegram-bot` (gateway), `google-auth` (web)
- Removed: `requirements.txt` (dependencies now managed entirely via `pyproject.toml`)
- Removed: Google Workspace and Microsoft Office MCP servers (moved to separate packages)

## v0.1.1

- Initial public release with MCP tool integration, web UI, and A2A protocol support.
