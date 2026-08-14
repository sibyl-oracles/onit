'''
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Tools MCP Server - Consolidated Web Search + Bash/Document Operations + GitHub

Combines web search, content fetching, weather, bash commands, file I/O,
document search, and GitHub repository management into a single MCP server.

14 Core Tools:
 1. search          - Web/news search via Ollama web search API (DuckDuckGo fallback)
 2. fetch_content   - Extract text, images, videos from URLs
 3. get_weather     - Weather with auto location detection
 4. bash            - Execute shell commands
 5. read_file       - Read files; mode="tables" or "images" for structured extraction
 6. write_file      - Write content to files (write/append modes)
 7. edit_file       - Edit a file by replacing an exact string
 8. serve           - Launch/stop/monitor background processes (web servers)
 9. grep            - Recursive pattern search across files in a directory
10. search_document - Search within a single document; mode="context" for semantic search
11. send_file       - Send files via callback URL or base64
12. github_repo     - Create, get, list, fork, or delete GitHub repositories
13. index_documents - Ingest in-house documents (pdf/md/txt/csv/docx/xlsx) into the local search index
14. local_search    - BM25/dense/hybrid retrieval over indexed in-house documents
'''

import json
import os
import tempfile
from typing import Annotated, Optional

from fastmcp import FastMCP, Context
from pydantic import Field

import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("Tools MCP Server")

# Data path for file creation (set via options['data_path'] in run())
DATA_PATH = os.path.join(tempfile.gettempdir(), "onit", "data")

# Optional read-only documents directory (set via options or env var in run())
DOCUMENTS_PATH = None


from src.mcp.servers.tasks.shared import (
    secure_makedirs as _secure_makedirs,
    validate_required as _validate_required,
    READ_FILE_DESCRIPTION,
    SEARCH_DOCUMENT_DESCRIPTION,
)


def _init_submodules(data_path: str, documents_path: str = None, verbose: bool = False):
    """Initialize DATA_PATH, DOCUMENTS_PATH, and logging in the sub-modules."""
    import src.mcp.servers.tasks.os.bash.mcp_server as bash_mod
    import src.mcp.servers.tasks.web.search.mcp_server as search_mod
    import src.mcp.servers.tasks.local.search.mcp_server as local_mod

    bash_mod.DATA_PATH = data_path
    bash_mod.DOCUMENTS_PATH = documents_path
    bash_mod._SANDBOX_ENV = None  # Reset sandbox env cache
    search_mod.DATA_PATH = data_path
    search_mod.DEFAULT_MEDIA_DIR = os.path.join(
        os.path.abspath(os.path.expanduser(data_path)), "media"
    )
    local_mod.DATA_PATH = data_path
    local_mod.DOCUMENTS_PATH = documents_path

    level = logging.INFO if verbose else logging.ERROR
    bash_mod.logger.setLevel(level)
    search_mod.logger.setLevel(level)
    local_mod.logger.setLevel(level)

    # Re-ingest the corpus in the background so this server never serves a
    # local_search index left stale by changes made while it was down
    local_mod.rebuild_indexes()


# ---------------------------------------------------------------------------
# Import and re-register all tool functions from both sub-modules.
# The @mcp.tool() decorator returns the original function, so these are
# plain callables that we can re-decorate on our unified mcp instance.
# ---------------------------------------------------------------------------

# -- Web Search tools (4) --------------------------------------------------

from src.mcp.servers.tasks.web.search.mcp_server import (
    SEARCH_TOOL_DESCRIPTION,
    search as _search,
    fetch_content as _fetch_content,
    get_weather as _get_weather,
    extract_pdf_images as _extract_pdf_images,
)

# Conditionally register tools based on API key availability.
# When the env var is set, the tool is not registered at all (not just disabled at runtime).

if not os.environ.get('ONIT_DISABLE_WEB_SEARCH'):
    @mcp.tool(
        title="Search the Web",
        description=SEARCH_TOOL_DESCRIPTION
    )
    def search(query: Optional[str] = None, type: str = "web", max_results: int = 5) -> str:
        if err := _validate_required(query=query):
            return err
        return _search(query=query, type=type, max_results=max_results)


@mcp.tool(
    title="Fetch Web Content",
    description="""Fetch content from a URL. Extracts text, images, and video links. Handles PDFs.

Args:
- url: Webpage URL to fetch (e.g., "https://example.com/article")
- extract_media: Extract image/video URLs (default: True)
- download_media: Download media files locally (default: False)
- output_dir: Save location for downloads within data_path folder (default: data_path/media)
- media_limit: Max files to download (default: 10)
- data_path: Session working directory — set automatically by the harness; leave unset.

Returns JSON: {title, url, content, images, videos, downloaded}"""
)
def fetch_content(
    url: Optional[str] = None,
    extract_media: bool = True,
    download_media: bool = False,
    output_dir: str = "",
    media_limit: int = 10,
    data_path: str = "",
) -> str:
    if err := _validate_required(url=url):
        return err
    return _fetch_content(
        url=url,
        extract_media=extract_media,
        download_media=download_media,
        output_dir=output_dir,
        media_limit=media_limit,
        data_path=data_path,
    )


if not os.environ.get('ONIT_DISABLE_WEATHER'):
    @mcp.tool(
        title="Get Weather",
        description="""Get current weather and optional 5-day forecast. Auto-detects location if not specified.

    Args:
    - place: City or location (e.g., "Tokyo, Japan"). Auto-detects from IP if omitted
    - forecast: Include 5-day forecast (default: False)

    Returns JSON: {location, current: {description, temperature_c, humidity_percent, wind_speed_ms, sunrise, sunset}, forecast_5day}

    Requires: OPENWEATHER_API_KEY environment variable."""
    )
    def get_weather(place: Optional[str] = None, forecast: bool = False) -> str:
        return _get_weather(place=place, forecast=forecast)


# -- Bash/Document tools -----------------------------------------------

from src.mcp.servers.tasks.os.bash.mcp_server import (
    bash as _bash,
    read_file as _read_file,
    write_file as _write_file,
    edit_file as _edit_file,
    serve as _serve,
    send_file as _send_file,
    search_document as _search_document,
    get_document_context as _get_document_context,
    extract_tables as _extract_tables,
    search_directory as _search_directory,
)

from src.mcp.servers.tasks.web.search.mcp_server import (
    extract_pdf_images as _extract_pdf_images,
)


@mcp.tool(
    title="Run Shell Command",
    description="""Execute a bash/shell command. Captures stdout, stderr, and return code.

Args:
- command: Shell command to run (e.g., "ls -la", "python script.py", "grep -r 'TODO' .")
- cwd: Working directory — must be within data_path (default: data_path)
- timeout: Max seconds to wait (default: 300, max: 1800). Raise it for slow work — installs, builds, full test suites — rather than letting the default kill them. A timed-out command is killed outright, along with anything it spawned.
- data_path: Session working directory — set automatically by the harness; leave unset.

Returns JSON: {stdout, stderr, returncode, cwd, command, status}

Long-running servers: never run one in the foreground — it will burn the whole timeout and be killed. Start it in the background (`cmd > log 2>&1 &`) and poll the log instead."""
)
async def bash(command: Optional[str] = None, cwd: str = ".", timeout: Optional[int] = None,
               data_path: str = "", ctx: Context = None) -> str:
    """Pass-through to the bash server's bash, which owns timeout defaulting
    and the ceiling.  The signature must stay identical to that one: the tool
    registry keys replica-vs-collision off the parameter schema, so a stray
    default here would split one tool into two."""
    if err := _validate_required(command=command):
        return err
    return await _bash(command=command, cwd=cwd, timeout=timeout, data_path=data_path, ctx=ctx)


@mcp.tool(
    title="Read File",
    description=READ_FILE_DESCRIPTION,
)
def read_file(
    path: Optional[str] = None,
    mode: str = "text",
    encoding: str = "utf-8",
    max_chars: int = 100000,
    table_index: Optional[int] = None,
    output_format: str = "json",
    output_dir: str = "",
    min_size: int = 100,
    data_path: str = "",
) -> str:
    """Pass-through to the bash server's read_file, which now routes all three
    modes itself.  Signature and description are shared verbatim so the two
    registrations are replicas of one tool rather than a name collision."""
    return _read_file(
        path=path, mode=mode, encoding=encoding, max_chars=max_chars,
        table_index=table_index, output_format=output_format,
        output_dir=output_dir, min_size=min_size, data_path=data_path,
    )



@mcp.tool(
    title="Write File",
    description="""Write content to a file. Creates directories if needed.
Files are created within the working directory with owner-only access.

Args:
- path: FULL absolute file path (e.g., "data_path/output.txt"). Always use the complete working directory path — never use relative paths.
- content: Text content to write (required)
- mode: "write" (overwrite) or "append" (add to end) (default: "write")
- encoding: Text encoding (default: utf-8)

Returns JSON: {path, size_bytes, mode, status}"""
)
def write_file(
    path: Optional[str] = None,
    content: Optional[str] = None,
    mode: str = "write",
    encoding: str = "utf-8",
    data_path: str = "",
) -> str:
    if err := _validate_required(path=path, content=content):
        return err
    return _write_file(path=path, content=content, mode=mode, encoding=encoding,
                       data_path=data_path)


@mcp.tool(
    title="Edit File",
    description="""Edit a file by replacing an exact string with new content. The file must exist.

Args:
- path: FULL absolute file path (e.g., "data_path/main.py"). Always use the complete working directory path — never use relative paths.
- old_string: The exact string to find and replace (must exist in the file)
- new_string: The replacement string
- replace_all: Replace every occurrence of old_string (default: false, replaces first only)
- encoding: Text encoding (default: utf-8)

Returns JSON: {path, replacements, status}"""
)
def edit_file(
    path: Optional[str] = None,
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    replace_all: bool = False,
    encoding: str = "utf-8",
    data_path: str = "",
) -> str:
    if err := _validate_required(path=path, old_string=old_string, new_string=new_string):
        return err
    return _edit_file(path=path, old_string=old_string, new_string=new_string, replace_all=replace_all, encoding=encoding, data_path=data_path)


@mcp.tool(
    title="Serve",
    description="""Manage long-running background processes such as web servers.

Actions:
- start   : Launch a command as a background process. Returns name, pid, and log paths.
- stop    : Stop a running process by name or pid.
- status  : Check if a process is running (name or pid).
- logs    : Tail stdout/stderr logs for a process (name or pid).
- list    : List all managed processes with running/stopped status.
- restart : Stop then re-start a named process using its saved command.

Args:
- action  : One of "start", "stop", "status", "logs", "list", "restart" (required)
- command : Shell command to run — required for "start" (e.g., "uvicorn main:app --port 8080")
- name    : Human-readable label for the process (default: auto-generated)
- pid     : Process ID — alternative to name for stop/status/logs
- cwd     : Working directory for the process. Can be any accessible directory.
- lines   : Number of log lines to return for "logs" action (default: 50)

Returns JSON with process details and, for "logs", stdout/stderr tail."""
)
def serve(
    action: Optional[str] = None,
    command: Optional[str] = None,
    name: Optional[str] = None,
    pid: Optional[int] = None,
    cwd: Optional[str] = None,
    lines: int = 50,
    data_path: str = "",
) -> str:
    if err := _validate_required(action=action):
        return err
    return _serve(action=action, command=command, name=name, pid=pid, cwd=cwd, lines=lines,
                  data_path=data_path)



@mcp.tool(
    title="Grep",
    description="""Search for a pattern across all files in a directory (recursive grep).
Returns structured results with file path, line number, and matching content.

Args:
- path: Directory to search in (required)
- pattern: Regex search pattern (required, e.g., "def train", "TODO", "import.*torch")
- file_pattern: Glob to filter files (default: "*" for all, e.g., "*.py", "*.md")
- case_sensitive: Case-sensitive matching (default: false)
- include_hidden: Include hidden files/directories (default: false)
- max_results: Maximum matches to return (default: 100)

Returns JSON: {results, total_matches, pattern, directory, file_pattern, status}
Each result includes: {file, line_number, content}"""
)
def grep(
    path: Optional[str] = None,
    pattern: Optional[str] = None,
    file_pattern: str = "*",
    case_sensitive: bool = False,
    include_hidden: bool = False,
    max_results: int = 100,
    data_path: str = "",
) -> str:
    if err := _validate_required(path=path, pattern=pattern):
        return err
    return _search_directory(
        directory=path, pattern=pattern, file_pattern=file_pattern,
        case_sensitive=case_sensitive, include_hidden=include_hidden,
        max_results=max_results, data_path=data_path,
    )


@mcp.tool(
    title="Send File",
    description="""Send a file from this host to a remote client.

If callback_url is provided, uploads the file via HTTP POST and returns the download URL.
Otherwise, returns the file content as base64-encoded data (max 10MB).

Args:
- path: Path to the file within data_path folder (required)
- callback_url: Full upload URL prefix (e.g., "http://host:9000/uploads/session_id"). File is POSTed to {callback_url}/ (optional)

Returns JSON: {path, filename, size_bytes, download_url, status} or {path, filename, size_bytes, file_data_base64, status}"""
)
def send_file(path: Optional[str] = None, callback_url: Optional[str] = None,
              data_path: str = "") -> str:
    if err := _validate_required(path=path):
        return err
    return _send_file(path=path, callback_url=callback_url, data_path=data_path)


# -- GitHub tools --------------------------------------------------------------

from src.mcp.servers.tasks.github.mcp_server import (
    github_repo as _github_repo,
)


@mcp.tool(
    title="GitHub Repository",
    description="""Create, get, list, fork, or delete GitHub repositories via the GitHub API.

Requires GITHUB_TOKEN environment variable (personal access token with repo scope).

Actions:
- create : Create a new repository (user or org). Returns repo details.
- get    : Get info about an existing repository.
- list   : List repositories for the authenticated user or an org.
- fork   : Fork an existing repository into the authenticated user's account or an org.
- delete : Delete a repository (requires admin access).

Args:
- action      : One of "create", "get", "list", "fork", "delete" (required)
- name        : Repository name — required for create, get (owner/repo), fork (owner/repo), delete (owner/repo)
- description : Repository description (create only, optional)
- private     : Make repo private (create only, default: false)
- auto_init   : Initialize with a README (create only, default: true)
- gitignore_template : e.g. "Python", "Node" (create only, optional)
- license_template   : e.g. "mit", "apache-2.0" (create only, optional)
- org         : Organization name — if set for create/list, targets the org instead of the user
- per_page    : Results per page for list (default: 30, max: 100)

Returns JSON: repo details for create/get/fork; list of repos for list; status for delete."""
)
def github_repo(
    action: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    private: bool = False,
    auto_init: bool = True,
    gitignore_template: Optional[str] = None,
    license_template: Optional[str] = None,
    org: Optional[str] = None,
    per_page: int = 30,
) -> str:
    if err := _validate_required(action=action):
        return err
    return _github_repo(
        action=action,
        name=name,
        description=description,
        private=private,
        auto_init=auto_init,
        gitignore_template=gitignore_template,
        license_template=license_template,
        org=org,
        per_page=per_page,
    )


@mcp.tool(
    title="Search Document",
    description=SEARCH_DOCUMENT_DESCRIPTION,
)
def search_document(
    path: Optional[str] = None,
    mode: str = "pattern",
    pattern: Optional[str] = None,
    query: Optional[str] = None,
    keywords: Optional[str] = None,
    case_sensitive: bool = False,
    context_lines: int = 3,
    max_matches: int = 50,
    context_chars: int = 500,
    max_sections: int = 5,
    data_path: str = "",
) -> str:
    """Pass-through to the bash server's search_document, which now routes both
    modes itself.  Signature and description are shared verbatim so the two
    registrations are replicas of one tool rather than a name collision."""
    return _search_document(
        path=path, mode=mode, pattern=pattern, query=query, keywords=keywords,
        case_sensitive=case_sensitive, context_lines=context_lines,
        max_matches=max_matches, context_chars=context_chars,
        max_sections=max_sections, data_path=data_path,
    )


# -- Local Search tools (in-house data) ----------------------------------------

from src.mcp.servers.tasks.local.search.mcp_server import (
    INDEX_DOCUMENTS_DESCRIPTION,
    LOCAL_SEARCH_DESCRIPTION,
    index_documents as _index_documents,
    local_search as _local_search,
)

if not os.environ.get('ONIT_DISABLE_LOCAL_SEARCH'):
    @mcp.tool(
        title="Index Local Documents",
        description=INDEX_DOCUMENTS_DESCRIPTION,
    )
    def index_documents(
        path: Optional[str] = None,
        recursive: bool = True,
        rebuild: bool = False,
        chunk_size: int = 1600,
        chunk_overlap: int = 200,
        status_only: bool = False,
        data_path: str = "",
    ) -> str:
        return _index_documents(
            path=path, recursive=recursive, rebuild=rebuild,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            status_only=status_only, data_path=data_path,
        )

    @mcp.tool(
        title="Search Local Documents",
        description=LOCAL_SEARCH_DESCRIPTION,
    )
    def local_search(
        query: Optional[str] = None,
        top_k: int = 5,
        method: str = "hybrid",
        path: Optional[str] = None,
        data_path: str = "",
    ) -> str:
        if err := _validate_required(query=query):
            return err
        return _local_search(query=query, top_k=top_k, method=method, path=path,
                             data_path=data_path)


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18201,
    path: str = "/sse",
    options: dict = {}
) -> None:
    """Run the consolidated Tools MCP server."""
    global DATA_PATH, DOCUMENTS_PATH

    verbose = 'verbose' in options
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    if 'data_path' in options:
        DATA_PATH = options['data_path']
    elif os.environ.get('ONIT_DATA_PATH'):
        DATA_PATH = os.environ['ONIT_DATA_PATH']
    abs_data = os.path.abspath(os.path.expanduser(DATA_PATH))
    _secure_makedirs(abs_data)

    if 'documents_path' in options:
        DOCUMENTS_PATH = options['documents_path']
    elif os.environ.get('ONIT_DOCUMENTS_PATH'):
        DOCUMENTS_PATH = os.environ['ONIT_DOCUMENTS_PATH']

    # Propagate DATA_PATH, DOCUMENTS_PATH, and log level to sub-modules
    _init_submodules(DATA_PATH, documents_path=DOCUMENTS_PATH, verbose=verbose)

    logger.info(f"Starting Tools MCP Server at {host}:{port}{path}")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info("14 Core Tools: search, fetch_content, get_weather, extract_pdf_images, "
                 "bash, read_file, send_file, search_document, "
                 "extract_tables, get_document_context, github_repo, "
                 "index_documents, local_search")

    if not verbose:
        import uvicorn.config
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    mcp.run(transport=transport, host=host, port=port, path=path,
            uvicorn_config={"access_log": False, "log_level": "warning"} if not verbose else {})


if __name__ == "__main__":
    run()
