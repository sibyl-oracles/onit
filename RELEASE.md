# Release Notes

## v0.1.4a

### New Features

- **Streaming Output** — Chat responses now stream token-by-token in the terminal UI, web UI, and A2A client for a more responsive experience.
- **Tokens/sec Indicator** — Real-time tok/s display during streaming output in both terminal and web interfaces.
- **Auto Model Detection** — The model is now auto-detected from the LLM endpoint, removing the need to set it explicitly for vLLM hosts. OpenRouter users can still override with `--model`.
- **VLM Web Image Fetch** — New MCP server (`VlmWebMCPServer`) for fetching and processing images from web URLs in vision-language model workflows.
- **Web UI Tool Calls** — Tool call details are now shown inline during the spinner/thinking phase in the web UI.
- **Windows Support** — OnIt now works on Windows with platform-specific fixes for signal handling and terminal UI.

### Improvements

- **Show Logs for Web & A2A** — The `--show-logs` flag now applies to web UI and A2A server modes, not just terminal and gateway.
- **3 Default MCP Servers** — Added a third default MCP server for VLM web image URL fetching alongside the existing Prompts and Tools servers.
- **Prompt Engineering** — Improved prompt templates for better instruction generation.

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
