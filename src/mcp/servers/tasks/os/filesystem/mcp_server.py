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

Document Search MCP Server

Search patterns in documents using command-line tools (grep, awk, sed, find, tr).
Supports plain text, PDF, markdown files with table understanding.
Provides context for LLM question answering and task completion.

Requires:
    pip install fastmcp pypdf pdfplumber

Core Tools:
1. search_document - Search for patterns in a document (text, PDF, markdown)
2. search_directory - Search for patterns across files in a directory
3. extract_tables - Extract tables from documents (PDF, markdown)
4. find_files - Find files matching patterns
5. transform_text - Transform text using sed/awk/tr operations
'''

import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Optional, List, Dict, Any
from pydantic import Field

from fastmcp import FastMCP

import logging

from requests import options
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("Document Search MCP Server")

# Constants
DEFAULT_TIMEOUT = 60
MAX_OUTPUT_SIZE = 100000  # 100KB max output
MAX_CONTEXT_LINES = 5  # Lines of context around matches

# Data path for temporary file creation (set via options['data_path'] in run())
# All temp files are confined to this directory. Never use home folder.
DATA_PATH = os.path.join(tempfile.gettempdir(), "onit", "data")


def _truncate_output(text: str, max_size: int = MAX_OUTPUT_SIZE) -> str:
    """Truncate output if it exceeds max size."""
    if len(text) > max_size:
        return text[:max_size] + f"\n\n... [OUTPUT TRUNCATED - {len(text)} bytes total]"
    return text


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


def _validate_read_path(file_path: str) -> str:
    """Validate that the path is within DATA_PATH. Returns resolved absolute path.
    Raises ValueError if outside allowed directory."""
    abs_path = os.path.realpath(os.path.expanduser(file_path))
    abs_data = os.path.realpath(os.path.expanduser(DATA_PATH))
    if abs_path.startswith(abs_data + os.sep) or abs_path == abs_data:
        return abs_path
    raise ValueError(
        f"Access denied. Path must be within: {abs_data}. Got: {abs_path}"
    )


def _validate_dir_path(dir_path: str) -> str:
    """Validate a directory path is within DATA_PATH. Returns resolved absolute path.
    Raises ValueError if outside allowed directory."""
    abs_path = os.path.realpath(os.path.expanduser(dir_path))
    abs_data = os.path.realpath(os.path.expanduser(DATA_PATH))
    if abs_path.startswith(abs_data + os.sep) or abs_path == abs_data:
        return abs_path
    raise ValueError(
        f"Directory access denied. Path must be within: {abs_data}. Got: {abs_path}"
    )


def _run_command(command: str, cwd: str = ".", timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Execute a shell command and return results."""
    try:
        work_dir = os.path.abspath(os.path.expanduser(cwd))
        if not os.path.isdir(work_dir):
            return {"error": f"Directory does not exist: {work_dir}", "status": "error"}
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=timeout,
            env={**os.environ, "TERM": "dumb"}
        )
        
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "status": "success" if result.returncode == 0 else "failed"
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout} seconds", "status": "timeout"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


def _extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("pypdf not installed")
        return ""
    except Exception as e:
        logger.error(f"Failed to read PDF: {e}")
        return ""


def _extract_pdf_tables(file_path: str) -> List[Dict[str, Any]]:
    """Extract tables from PDF using pdfplumber."""
    tables = []
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                for table_idx, table in enumerate(page_tables, 1):
                    if table and len(table) > 0:
                        # Convert to structured format
                        headers = table[0] if table else []
                        rows = table[1:] if len(table) > 1 else []
                        tables.append({
                            "page": page_num,
                            "table_index": table_idx,
                            "headers": headers,
                            "rows": rows,
                            "row_count": len(rows)
                        })
        return tables
    except ImportError:
        logger.warning("pdfplumber not installed. Run: pip install pdfplumber")
        return []
    except Exception as e:
        logger.error(f"Failed to extract tables from PDF: {e}")
        return []


def _extract_markdown_tables(content: str) -> List[Dict[str, Any]]:
    """Extract tables from markdown content."""
    tables = []
    # Regex to match markdown tables
    table_pattern = r'(\|[^\n]+\|\n\|[-:\| ]+\|\n(?:\|[^\n]+\|\n)*)'
    
    matches = re.finditer(table_pattern, content)
    for idx, match in enumerate(matches, 1):
        table_text = match.group(1)
        lines = table_text.strip().split('\n')
        
        if len(lines) >= 2:
            # Parse headers
            headers = [cell.strip() for cell in lines[0].split('|')[1:-1]]
            # Skip separator line (lines[1])
            # Parse rows
            rows = []
            for line in lines[2:]:
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if cells:
                    rows.append(cells)
            
            tables.append({
                "table_index": idx,
                "headers": headers,
                "rows": rows,
                "row_count": len(rows),
                "raw": table_text
            })
    
    return tables


def _get_file_content(file_path: str) -> tuple[str, str]:
    """Get file content and format. Returns (content, format)."""
    file_path = os.path.abspath(os.path.expanduser(file_path))
    
    if not os.path.isfile(file_path):
        return "", "error"
    
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        return _extract_pdf_text(file_path), "pdf"
    else:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            if ext == '.md':
                return content, "markdown"
            else:
                return content, "text"
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return "", "error"


# =============================================================================
# TOOL 1: SEARCH DOCUMENT
# =============================================================================

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
    max_matches: Annotated[int, Field(description="Maximum number of matches to return")] = 50
) -> str:
    if err := _validate_required(path=path, pattern=pattern):
        return err
    try:
        file_path = _validate_read_path(path)

        if not os.path.isfile(file_path):
            return json.dumps({
                "error": f"File not found: {file_path}",
                "path": path,
                "status": "error"
            })

        content, file_format = _get_file_content(file_path)
        
        if file_format == "error":
            return json.dumps({
                "error": "Failed to read file",
                "path": path,
                "status": "error"
            })
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return json.dumps({
                "error": f"Invalid regex pattern: {e}",
                "pattern": pattern,
                "status": "error"
            })
        
        lines = content.split('\n')
        matches = []
        
        for i, line in enumerate(lines):
            if regex.search(line):
                # Get context
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                
                matches.append({
                    "line_number": i + 1,
                    "match": line.strip(),
                    "context_before": [l.strip() for l in lines[start:i]],
                    "context_after": [l.strip() for l in lines[i+1:end]]
                })
                
                if len(matches) >= max_matches:
                    break
        
        return json.dumps({
            "matches": matches,
            "total_matches": len(matches),
            "pattern": pattern,
            "file": file_path,
            "format": file_format,
            "truncated": len(matches) >= max_matches,
            "status": "success"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "path": path,
            "status": "error"
        })


# =============================================================================
# TOOL 2: SEARCH DIRECTORY
# =============================================================================

@mcp.tool(
    title="Search Directory",
    description="""Search for patterns across files in a directory using grep.
Recursively searches text files matching the file pattern.

Args:
- directory: Directory within data_path folder to search (e.g., "data_path", "data_path/subdir")
- pattern: Search pattern (regex with -E flag)
- file_pattern: File glob pattern (default: "*" for all files)
- case_sensitive: Case-sensitive search (default: false)
- include_hidden: Include hidden files (default: false)
- max_results: Maximum results to return (default: 100)

Returns JSON: {results, total_files, total_matches, status}
Each result includes: {file, line_number, content}"""
)
def search_directory(
    directory: Optional[str] = None,
    pattern: Optional[str] = None,
    file_pattern: str = "*",
    case_sensitive: bool = False,
    include_hidden: bool = False,
    max_results: int = 100
) -> str:
    if err := _validate_required(directory=directory, pattern=pattern):
        return err
    try:
        dir_path = _validate_dir_path(directory)

        if not os.path.isdir(dir_path):
            return json.dumps({
                "error": f"Directory not found: {dir_path}",
                "directory": directory,
                "status": "error"
            })
        
        # Build grep command
        grep_flags = "-rn"  # recursive, line numbers
        if not case_sensitive:
            grep_flags += "i"
        grep_flags += "E"  # extended regex
        
        # Exclude hidden files unless requested
        exclude = "" if include_hidden else "--exclude-dir='.[!.]*' --exclude='.[!.]*'"
        
        # Build command
        cmd = f"grep {grep_flags} {exclude} --include={shlex.quote(file_pattern)} {shlex.quote(pattern)} . 2>/dev/null | head -n {int(max_results)}"
        
        result = _run_command(cmd, cwd=dir_path)
        
        if result.get("status") == "error":
            return json.dumps(result)
        
        # Parse grep output
        results = []
        output = result.get("stdout", "")
        
        if output:
            for line in output.split('\n'):
                if ':' in line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        results.append({
                            "file": parts[0],
                            "line_number": int(parts[1]) if parts[1].isdigit() else parts[1],
                            "content": parts[2].strip()
                        })
        
        return json.dumps({
            "results": results,
            "total_matches": len(results),
            "pattern": pattern,
            "directory": dir_path,
            "file_pattern": file_pattern,
            "truncated": len(results) >= max_results,
            "status": "success"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "directory": directory,
            "status": "error"
        })


# =============================================================================
# TOOL 3: EXTRACT TABLES
# =============================================================================

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
    path: Optional[str] = None,
    table_index: Optional[int] = None,
    output_format: str = "json"
) -> str:
    if err := _validate_required(path=path):
        return err
    try:
        file_path = _validate_read_path(path)

        if not os.path.isfile(file_path):
            return json.dumps({
                "error": f"File not found: {file_path}",
                "path": path,
                "status": "error"
            })
        
        ext = Path(file_path).suffix.lower()
        tables = []
        
        if ext == '.pdf':
            tables = _extract_pdf_tables(file_path)
        elif ext in ['.md', '.markdown']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tables = _extract_markdown_tables(content)
        else:
            # Try to parse as markdown anyway
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                tables = _extract_markdown_tables(content)
            except Exception:
                return json.dumps({
                    "error": "File format not supported for table extraction",
                    "path": path,
                    "supported_formats": ["pdf", "md", "markdown"],
                    "status": "error"
                })
        
        # Filter by table index if specified
        if table_index is not None:
            if 1 <= table_index <= len(tables):
                tables = [tables[table_index - 1]]
            else:
                return json.dumps({
                    "error": f"Table index {table_index} out of range (1-{len(tables)})",
                    "total_tables": len(tables),
                    "status": "error"
                })
        
        # Convert to markdown format if requested
        if output_format == "markdown":
            md_tables = []
            for table in tables:
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                
                if headers:
                    md = "| " + " | ".join(str(h) for h in headers) + " |\n"
                    md += "| " + " | ".join("---" for _ in headers) + " |\n"
                    for row in rows:
                        md += "| " + " | ".join(str(c) for c in row) + " |\n"
                    md_tables.append({
                        "table_index": table.get("table_index"),
                        "page": table.get("page"),
                        "markdown": md
                    })
            tables = md_tables
        
        return json.dumps({
            "tables": tables,
            "total_tables": len(tables),
            "file": file_path,
            "format": ext.lstrip('.'),
            "status": "success"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "path": path,
            "status": "error"
        })


# =============================================================================
# TOOL 4: FIND FILES
# =============================================================================

@mcp.tool(
    title="Find Files",
    description="""Find files matching patterns using the find command.
Searches recursively from the specified directory within data_path folder.

Args:
- directory: Directory within data_path folder to search (default: data_path)
- name_pattern: File name pattern (glob, e.g., "*.py", "test_*")
- file_type: Type filter - "f" (file), "d" (directory), or None (all)
- max_depth: Maximum directory depth (default: unlimited)
- size_filter: Size filter (e.g., "+1M", "-100k", "50k")
- modified_days: Modified within N days (e.g., 7 for last week)
- max_results: Maximum results (default: 100)

Returns JSON: {files, total_files, directory, status}"""
)
def find_files(
    directory: str = ".",
    name_pattern: Optional[str] = None,
    file_type: Optional[str] = None,
    max_depth: Optional[int] = None,
    size_filter: Optional[str] = None,
    modified_days: Optional[int] = None,
    max_results: int = 100
) -> str:
    try:
        dir_path = _validate_dir_path(directory)

        if not os.path.isdir(dir_path):
            return json.dumps({
                "error": f"Directory not found: {dir_path}",
                "directory": directory,
                "status": "error"
            })

        # Validate numeric parameters
        max_results = int(max_results)
        if max_results <= 0:
            max_results = 100

        if max_depth is not None:
            max_depth = int(max_depth)
            if max_depth < 0:
                return json.dumps({"error": "max_depth must be non-negative", "status": "error"})

        if modified_days is not None:
            modified_days = int(modified_days)
            if modified_days < 0:
                return json.dumps({"error": "modified_days must be non-negative", "status": "error"})

        # Validate file_type against allowlist
        allowed_file_types = {"f", "d", "l", "b", "c", "p", "s"}
        if file_type and file_type not in allowed_file_types:
            return json.dumps({
                "error": f"Invalid file_type: {file_type}. Must be one of: {', '.join(sorted(allowed_file_types))}",
                "status": "error"
            })

        # Validate size_filter format
        if size_filter and not re.match(r'^[+-]?\d+[bcwkMG]?$', size_filter):
            return json.dumps({
                "error": f"Invalid size_filter: {size_filter}. Expected format: [+-]N[bcwkMG]",
                "status": "error"
            })

        cmd_parts = ["find", shlex.quote(dir_path)]

        if max_depth is not None:
            cmd_parts.append(f"-maxdepth {max_depth}")

        if file_type:
            cmd_parts.append(f"-type {file_type}")

        if name_pattern:
            cmd_parts.append(f"-name {shlex.quote(name_pattern)}")

        if size_filter:
            cmd_parts.append(f"-size {size_filter}")

        if modified_days is not None:
            cmd_parts.append(f"-mtime -{modified_days}")

        cmd = " ".join(cmd_parts) + f" 2>/dev/null | head -n {max_results}"
        
        result = _run_command(cmd, cwd="/")
        
        if result.get("status") == "error":
            return json.dumps(result)
        
        output = result.get("stdout", "")
        files = [f.strip() for f in output.split('\n') if f.strip()]
        
        # Get file info
        file_info = []
        for f in files:
            try:
                stat = os.stat(f)
                file_info.append({
                    "path": f,
                    "name": os.path.basename(f),
                    "size_bytes": stat.st_size,
                    "is_dir": os.path.isdir(f)
                })
            except Exception:
                file_info.append({"path": f, "name": os.path.basename(f)})
        
        return json.dumps({
            "files": file_info,
            "total_files": len(file_info),
            "directory": dir_path,
            "pattern": name_pattern,
            "truncated": len(files) >= max_results,
            "status": "success"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "directory": directory,
            "status": "error"
        })


# =============================================================================
# TOOL 5: TRANSFORM TEXT
# =============================================================================

@mcp.tool(
    title="Transform Text",
    description="""Transform text using sed, awk, or tr commands.
Useful for extracting, replacing, or reformatting text content.

Args:
- input_text: Text to transform (or path to file if is_file=true)
- is_file: If true, input_text is treated as a file path (default: false)
- operation: Transformation type - "sed", "awk", or "tr"
- expression: The sed/awk/tr expression to apply
  - sed: e.g., "s/old/new/g", "/pattern/d"
  - awk: e.g., "{print $1}", "NR==1", "/pattern/{print}"
  - tr: e.g., "a-z A-Z" (translate), "-d '\\n'" (delete)

Returns JSON: {output, operation, expression, status}"""
)
def transform_text(
    input_text: Optional[str] = None,
    operation: Optional[str] = None,
    expression: Optional[str] = None,
    is_file: bool = False
) -> str:
    if err := _validate_required(input_text=input_text, operation=operation, expression=expression):
        return err
    try:
        if operation not in ["sed", "awk", "tr"]:
            return json.dumps({
                "error": f"Invalid operation: {operation}. Use 'sed', 'awk', or 'tr'",
                "status": "error"
            })

        # Get input content
        if is_file:
            file_path = _validate_read_path(input_text)
            if not os.path.isfile(file_path):
                return json.dumps({
                    "error": f"File not found: {file_path}",
                    "status": "error"
                })
            input_source = f"cat {shlex.quote(file_path)}"
        else:
            # Use a temp file within DATA_PATH (not system temp or home folder)
            tmp_dir = os.path.join(os.path.abspath(os.path.expanduser(DATA_PATH)), "tmp")
            _secure_makedirs(tmp_dir)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir=tmp_dir) as f:
                f.write(input_text)
                temp_path = f.name
            os.chmod(temp_path, 0o600)
            input_source = f"cat {shlex.quote(temp_path)}"

        if operation == "sed":
            cmd = f"{input_source} | sed {shlex.quote(expression)}"
        elif operation == "awk":
            cmd = f"{input_source} | awk {shlex.quote(expression)}"
        elif operation == "tr":
            try:
                tr_args = shlex.split(expression)
                quoted_tr_args = " ".join(shlex.quote(arg) for arg in tr_args)
                cmd = f"{input_source} | tr {quoted_tr_args}"
            except ValueError as e:
                return json.dumps({
                    "error": f"Invalid tr expression: {e}",
                    "status": "error"
                })
        
        result = _run_command(cmd)
        
        # Clean up temp file
        if not is_file:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        
        if result.get("status") == "error":
            return json.dumps(result)
        
        output = _truncate_output(result.get("stdout", ""))
        
        return json.dumps({
            "output": output,
            "operation": operation,
            "expression": expression,
            "status": "success"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "operation": operation,
            "status": "error"
        })


# =============================================================================
# TOOL 6: GET DOCUMENT CONTEXT
# =============================================================================

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
    max_sections: int = 5
) -> str:
    if err := _validate_required(path=path, query=query):
        return err
    try:
        file_path = _validate_read_path(path)

        if not os.path.isfile(file_path):
            return json.dumps({
                "error": f"File not found: {file_path}",
                "path": path,
                "status": "error"
            })

        content, file_format = _get_file_content(file_path)

        if file_format == "error" or not content:
            return json.dumps({
                "error": "Failed to read file",
                "path": path,
                "status": "error"
            })
        
        # Extract keywords from query and additional keywords
        search_terms = set()
        
        # Simple keyword extraction from query (words > 3 chars, not stopwords)
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 
                     'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
                     'what', 'when', 'where', 'which', 'who', 'how', 'this', 'that',
                     'with', 'from', 'they', 'will', 'would', 'there', 'their'}
        
        for word in re.findall(r'\b\w+\b', query.lower()):
            if len(word) > 3 and word not in stopwords:
                search_terms.add(word)
        
        # Add explicit keywords
        if keywords:
            for kw in keywords.split(','):
                kw = kw.strip().lower()
                if kw:
                    search_terms.add(kw)
        
        if not search_terms:
            # Fallback to all significant words in query
            search_terms = {w.lower() for w in query.split() if len(w) > 2}
        
        # Find matching positions
        matches = []
        content_lower = content.lower()
        
        for term in search_terms:
            for match in re.finditer(re.escape(term), content_lower):
                matches.append({
                    "position": match.start(),
                    "term": term
                })
        
        # Sort by position and deduplicate overlapping regions
        matches.sort(key=lambda x: x["position"])
        
        sections = []
        used_ranges = []
        
        for match in matches:
            pos = match["position"]
            
            # Check if this position is already covered
            is_covered = any(start <= pos <= end for start, end in used_ranges)
            if is_covered:
                continue
            
            # Extract context
            start = max(0, pos - context_chars // 2)
            end = min(len(content), pos + context_chars // 2)
            
            # Try to start/end at sentence boundaries
            if start > 0:
                sentence_start = content.rfind('.', start - 100, start)
                if sentence_start > start - 100:
                    start = sentence_start + 1
            
            if end < len(content):
                sentence_end = content.find('.', end, end + 100)
                if sentence_end != -1:
                    end = sentence_end + 1
            
            section_content = content[start:end].strip()
            
            # Find all keywords in this section
            section_keywords = [t for t in search_terms if t in section_content.lower()]
            
            sections.append({
                "content": section_content,
                "relevance_keywords": section_keywords,
                "position": pos,
                "char_range": [start, end]
            })
            
            used_ranges.append((start, end))
            
            if len(sections) >= max_sections:
                break
        
        return json.dumps({
            "sections": sections,
            "total_sections": len(sections),
            "query": query,
            "search_terms": list(search_terms),
            "file": file_path,
            "format": file_format,
            "status": "success"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "path": path,
            "status": "error"
        })


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18202,
    path: str = "/sse",
    options: dict = {}
) -> None:
    """Run the MCP server."""
    global DATA_PATH

    if 'verbose' in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    if 'data_path' in options:
        DATA_PATH = options['data_path']
    _secure_makedirs(os.path.abspath(os.path.expanduser(DATA_PATH)))

    logger.info(f"Starting Document Search MCP Server at {host}:{port}{path}")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info("Available tools: search_document, search_directory, extract_tables, find_files, transform_text, get_document_context")

    quiet = 'verbose' not in options
    if quiet:
        import uvicorn.config
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    mcp.run(transport=transport, host=host, port=port, path=path,
            uvicorn_config={"access_log": False, "log_level": "warning"} if quiet else {})

# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    run()
