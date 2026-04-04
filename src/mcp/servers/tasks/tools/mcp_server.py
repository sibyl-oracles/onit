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

Tools MCP Server - Consolidated Web Search + Bash/Document Operations

Combines web search, content fetching, weather, bash commands, file I/O,
and document search into a single MCP server.

10 Core Tools:
 1. search            - Web/news search via DuckDuckGo
 2. fetch_content     - Extract text, images, videos from URLs
 3. get_weather       - Weather with auto location detection
 4. extract_pdf_images- Extract images from PDF files
 5. bash              - Execute shell commands
 6. read_file         - Read files (text, PDF, binary metadata)
 7. send_file         - Send files via callback URL or base64
 8. search_document   - Search patterns in documents (incl. PDF)
 9. extract_tables    - Extract tables from PDF/markdown
10. get_document_context - Extract relevant context from documents
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


def _secure_makedirs(dir_path: str) -> None:
    """Create directory with owner-only permissions (0o700)."""
    os.makedirs(dir_path, mode=0o700, exist_ok=True)


def _validate_required(**kwargs) -> str:
    """Check for missing required arguments. Returns JSON error string or empty string."""
    missing = [name for name, value in kwargs.items() if value is None]
    if missing:
        return json.dumps({
            "error": f"Missing required argument(s): {', '.join(missing)}.",
            "status": "error"
        })
    return ""


def _init_submodules(data_path: str, documents_path: str = None, verbose: bool = False):
    """Initialize DATA_PATH, DOCUMENTS_PATH, and logging in both sub-modules."""
    import src.mcp.servers.tasks.os.bash.mcp_server as bash_mod
    import src.mcp.servers.tasks.web.search.mcp_server as search_mod

    bash_mod.DATA_PATH = data_path
    bash_mod.DOCUMENTS_PATH = documents_path
    bash_mod._SANDBOX_ENV = None  # Reset sandbox env cache
    search_mod.DATA_PATH = data_path
    search_mod.DEFAULT_MEDIA_DIR = os.path.join(
        os.path.abspath(os.path.expanduser(data_path)), "media"
    )

    level = logging.INFO if verbose else logging.ERROR
    bash_mod.logger.setLevel(level)
    search_mod.logger.setLevel(level)


# ---------------------------------------------------------------------------
# Import and re-register all tool functions from both sub-modules.
# The @mcp.tool() decorator returns the original function, so these are
# plain callables that we can re-decorate on our unified mcp instance.
# ---------------------------------------------------------------------------

# -- Web Search tools (4) --------------------------------------------------

from src.mcp.servers.tasks.web.search.mcp_server import (
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
        description="""Search the web for news or general information using DuckDuckGo.

    Args:
    - query: Search terms (e.g., "AI regulations 2024", "how to bake bread")
    - type: "news" for recent news, "web" for general search (default: "web")
    - max_results: Number of results (default: 5, max: 10)

    Returns JSON: [{title, snippet, url, source, date}]"""
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

Returns JSON: {title, url, content, images, videos, downloaded}"""
)
def fetch_content(
    url: Optional[str] = None,
    extract_media: bool = True,
    download_media: bool = False,
    output_dir: str = "",
    media_limit: int = 10,
) -> str:
    if err := _validate_required(url=url):
        return err
    return _fetch_content(
        url=url,
        extract_media=extract_media,
        download_media=download_media,
        output_dir=output_dir,
        media_limit=media_limit,
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
    def get_weather(place: str = None, forecast: bool = False) -> str:
        return _get_weather(place=place, forecast=forecast)


@mcp.tool(
    title="Extract PDF Images",
    description="""Extract all images from a PDF file and save them locally.

Args:
- pdf_path: Path to PDF file within data_path folder or URL (required)
- output_dir: Directory within data_path folder to save extracted images (default: data_path/pdf_images)
- min_size: Minimum image dimension in pixels to extract (default: 100)

Returns JSON: {pdf_path, output_dir, images: [{path, width, height, format}], image_count, status}"""
)
def extract_pdf_images(pdf_path: Optional[str] = None, output_dir: str = "", min_size: int = 100) -> str:
    if err := _validate_required(pdf_path=pdf_path):
        return err
    return _extract_pdf_images(pdf_path=pdf_path, output_dir=output_dir, min_size=min_size)


# -- Bash/Document tools (6) -----------------------------------------------

from src.mcp.servers.tasks.os.bash.mcp_server import (
    bash as _bash,
    read_file as _read_file,
    send_file as _send_file,
    search_document as _search_document,
    extract_tables as _extract_tables,
    get_document_context as _get_document_context,
)


@mcp.tool(
    title="Run Shell Command",
    description="""Execute a bash/shell command. Captures stdout, stderr, and return code.

Args:
- command: Shell command to run (e.g., "ls -la", "python script.py", "grep -r 'TODO' .")
- cwd: Working directory — must be within data_path (default: data_path)
- timeout: Max seconds to wait (default: 300)

Returns JSON: {stdout, stderr, returncode, cwd, command, status}"""
)
async def bash(command: Optional[str] = None, cwd: str = ".", timeout: int = 300, ctx: Context = None) -> str:
    if err := _validate_required(command=command):
        return err
    return await _bash(command=command, cwd=cwd, timeout=timeout, ctx=ctx)


@mcp.tool(
    title="Read File",
    description="""Read file contents. Supports text files and PDFs. Binary files (images, audio, video) return metadata only.

Args:
- path: File path within data_path folder (e.g., "data_path/report.pdf", "data_path/output.txt")
- encoding: Text encoding (default: utf-8)
- max_chars: Max characters to read (default: 100000)

Returns JSON: {content, path, size_bytes, format, status} or {path, size_bytes, format, type} for binary files"""
)
def read_file(path: Optional[str] = None, encoding: str = "utf-8", max_chars: int = 100000) -> str:
    if err := _validate_required(path=path):
        return err
    return _read_file(path=path, encoding=encoding, max_chars=max_chars)



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
def send_file(path: Optional[str] = None, callback_url: Optional[str] = None) -> str:
    if err := _validate_required(path=path):
        return err
    return _send_file(path=path, callback_url=callback_url)


@mcp.tool(
    title="Search Document",
    description="""Search for a regex pattern in a single document file. Supports text, PDF, and markdown files.
Uses grep-like regex pattern matching and returns matching lines with surrounding context.

IMPORTANT - Required parameters:
- path: File path within data_path folder to search (e.g., "data_path/report.pdf", "data_path/README.md")
- pattern: Regex search pattern to find in the document (e.g., "error.*timeout", "subjects")
  Do NOT use 'query' - the parameter name is 'pattern'.

Optional parameters:
- case_sensitive: Whether search is case-sensitive (default: false)
- context_lines: Number of lines of context before/after each match (default: 3).
  Do NOT use 'context_chars' - the parameter name is 'context_lines'.
- max_matches: Maximum number of matches to return (default: 50).
  Do NOT use 'max_sections' - the parameter name is 'max_matches'.

Example: search_document(path="data_path/report.pdf", pattern="conclusion")

Returns JSON: {matches, total_matches, file, format, status}
Each match includes: {line_number, match, context_before, context_after}"""
)
def search_document(
    path: Annotated[Optional[str], Field(description="File path within data_path folder to search")] = None,
    pattern: Annotated[Optional[str], Field(description="Regex search pattern to find in the document (e.g., 'error.*timeout', 'subjects')")] = None,
    case_sensitive: Annotated[bool, Field(description="Whether search is case-sensitive")] = False,
    context_lines: Annotated[int, Field(description="Number of lines of context before/after each match")] = 3,
    max_matches: Annotated[int, Field(description="Maximum number of matches to return")] = 50,
) -> str:
    if err := _validate_required(path=path, pattern=pattern):
        return err
    return _search_document(
        path=path, pattern=pattern, case_sensitive=case_sensitive,
        context_lines=context_lines, max_matches=max_matches,
    )



@mcp.tool(
    title="Extract Tables",
    description="""Extract tables from documents. Supports PDF and markdown files.
Tables are returned in a structured format with headers and rows.

Args:
- path: File path within data_path folder (e.g., "data_path/report.pdf", "data_path/README.md")
- table_index: Specific table index to extract (1-based, default: all)
- output_format: Output format - "json" or "markdown" (default: "json")

Returns JSON: {tables, total_tables, file, format, status}
Each table includes: {headers, rows, row_count, page (for PDF)}"""
)
def extract_tables(
    path: Optional[str] = None, table_index: Optional[int] = None, output_format: str = "json"
) -> str:
    if err := _validate_required(path=path):
        return err
    return _extract_tables(path=path, table_index=table_index, output_format=output_format)



@mcp.tool(
    title="Get Document Context",
    description="""Extract relevant context from a document for answering questions.
Searches for keywords and returns surrounding context that can support answers.

Args:
- path: Document path within data_path folder (text, PDF, or markdown)
- query: The question or topic to find context for
- keywords: Additional keywords to search (comma-separated)
- context_chars: Characters of context around matches (default: 500)
- max_sections: Maximum context sections to return (default: 5)

Returns JSON: {sections, query, file, status}
Each section includes: {content, relevance_keywords, position}"""
)
def get_document_context(
    path: Optional[str] = None,
    query: Optional[str] = None,
    keywords: Optional[str] = None,
    context_chars: int = 500,
    max_sections: int = 5,
) -> str:
    if err := _validate_required(path=path, query=query):
        return err
    return _get_document_context(
        path=path, query=query, keywords=keywords,
        context_chars=context_chars, max_sections=max_sections,
    )


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
    logger.info("10 Core Tools: search, fetch_content, get_weather, extract_pdf_images, "
                 "bash, read_file, send_file, search_document, "
                 "extract_tables, get_document_context")

    if not verbose:
        import uvicorn.config
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    mcp.run(transport=transport, host=host, port=port, path=path,
            uvicorn_config={"access_log": False, "log_level": "warning"} if not verbose else {})


if __name__ == "__main__":
    run()
