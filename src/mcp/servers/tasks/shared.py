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

Shared utilities for MCP task servers.

Common helper functions and tool logic shared between the bash and filesystem
MCP servers. Server-specific behavior (path validation, command execution) is
injected via callable parameters so each server retains its own security model.
'''

import json
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import logging

logger = logging.getLogger(__name__)

# Default constants (servers may override via their own module-level values)
MAX_OUTPUT_SIZE = 100000  # 100KB max output


# =============================================================================
# SHARED HELPER FUNCTIONS
# =============================================================================


def truncate_output(text: str, max_size: int = MAX_OUTPUT_SIZE) -> str:
    """Truncate output if it exceeds max size."""
    if len(text) > max_size:
        return text[:max_size] + f"\n\n... [OUTPUT TRUNCATED - {len(text)} bytes total]"
    return text


def secure_makedirs(dir_path: str) -> None:
    """Create directory with owner-only permissions (0o700)."""
    os.makedirs(dir_path, mode=0o700, exist_ok=True)


def validate_required(**kwargs) -> str:
    """Check for missing required arguments. Returns JSON error string or empty string."""
    missing = [name for name, value in kwargs.items() if value is None]
    if missing:
        return json.dumps({
            "error": f"Missing required argument(s): {', '.join(missing)}.",
            "status": "error"
        })
    return ""


def extract_pdf_text(file_path: str) -> str:
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


def extract_pdf_tables(file_path: str) -> List[Dict[str, Any]]:
    """Extract tables from PDF using pdfplumber."""
    tables = []
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                for table_idx, table in enumerate(page_tables, 1):
                    if table and len(table) > 0:
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


def extract_markdown_tables(content: str) -> List[Dict[str, Any]]:
    """Extract tables from markdown content."""
    tables = []
    table_pattern = r'(\|[^\n]+\|\n\|[-:\| ]+\|\n(?:\|[^\n]+\|\n)*)'

    matches = re.finditer(table_pattern, content)
    for idx, match in enumerate(matches, 1):
        table_text = match.group(1)
        lines = table_text.strip().split('\n')

        if len(lines) >= 2:
            headers = [cell.strip() for cell in lines[0].split('|')[1:-1]]
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


def get_file_content(file_path: str) -> tuple[str, str]:
    """Get file content and format. Returns (content, format)."""
    file_path = os.path.abspath(os.path.expanduser(file_path))

    if not os.path.isfile(file_path):
        return "", "error"

    ext = Path(file_path).suffix.lower()

    if ext == '.pdf':
        return extract_pdf_text(file_path), "pdf"
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
# SHARED TOOL LOGIC
# =============================================================================

# Each function below implements the core logic of an MCP tool. The caller
# (bash or filesystem server) passes in its own _validate_read_path,
# _validate_dir_path, _run_command, etc. so security behaviour stays
# server-specific.


def search_document_impl(
    path: Optional[str],
    pattern: Optional[str],
    case_sensitive: bool,
    context_lines: int,
    max_matches: int,
    validate_read_path: Callable[[str], str],
) -> str:
    """Core logic for search_document tool."""
    if err := validate_required(path=path, pattern=pattern):
        return err
    try:
        file_path = validate_read_path(path)

        if not os.path.isfile(file_path):
            return json.dumps({
                "error": f"File not found: {file_path}",
                "path": path,
                "status": "error"
            })

        content, file_format = get_file_content(file_path)

        if file_format == "error":
            return json.dumps({
                "error": "Failed to read file",
                "path": path,
                "status": "error"
            })

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


def search_directory_impl(
    directory: Optional[str],
    pattern: Optional[str],
    file_pattern: str,
    case_sensitive: bool,
    include_hidden: bool,
    max_results: int,
    validate_dir_path: Callable[[str], str],
    run_command: Callable[..., Dict[str, Any]],
) -> str:
    """Core logic for search_directory tool."""
    if err := validate_required(directory=directory, pattern=pattern):
        return err
    try:
        dir_path = validate_dir_path(directory)

        if not os.path.isdir(dir_path):
            return json.dumps({
                "error": f"Directory not found: {dir_path}",
                "directory": directory,
                "status": "error"
            })

        grep_flags = "-rn"
        if not case_sensitive:
            grep_flags += "i"
        grep_flags += "E"

        exclude = "" if include_hidden else "--exclude-dir='.[!.]*' --exclude='.[!.]*'"

        cmd = f"grep {grep_flags} {exclude} --include={shlex.quote(file_pattern)} {shlex.quote(pattern)} . 2>/dev/null | head -n {int(max_results)}"

        result = run_command(cmd, cwd=dir_path)

        if result.get("status") == "error":
            return json.dumps(result)

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


def extract_tables_impl(
    path: Optional[str],
    table_index: Optional[int],
    output_format: str,
    validate_read_path: Callable[[str], str],
) -> str:
    """Core logic for extract_tables tool."""
    if err := validate_required(path=path):
        return err
    try:
        file_path = validate_read_path(path)

        if not os.path.isfile(file_path):
            return json.dumps({
                "error": f"File not found: {file_path}",
                "path": path,
                "status": "error"
            })

        ext = Path(file_path).suffix.lower()
        tables = []

        if ext == '.pdf':
            tables = extract_pdf_tables(file_path)
        elif ext in ['.md', '.markdown']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tables = extract_markdown_tables(content)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                tables = extract_markdown_tables(content)
            except Exception:
                return json.dumps({
                    "error": "File format not supported for table extraction",
                    "path": path,
                    "supported_formats": ["pdf", "md", "markdown"],
                    "status": "error"
                })

        if table_index is not None:
            if 1 <= table_index <= len(tables):
                tables = [tables[table_index - 1]]
            else:
                return json.dumps({
                    "error": f"Table index {table_index} out of range (1-{len(tables)})",
                    "total_tables": len(tables),
                    "status": "error"
                })

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


def find_files_impl(
    directory: str,
    name_pattern: Optional[str],
    file_type: Optional[str],
    max_depth: Optional[int],
    size_filter: Optional[str],
    modified_days: Optional[int],
    max_results: int,
    validate_dir_path: Callable[[str], str],
    run_command: Callable[..., Dict[str, Any]],
) -> str:
    """Core logic for find_files tool."""
    try:
        dir_path = validate_dir_path(directory)

        if not os.path.isdir(dir_path):
            return json.dumps({
                "error": f"Directory not found: {dir_path}",
                "directory": directory,
                "status": "error"
            })

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

        allowed_file_types = {"f", "d", "l", "b", "c", "p", "s"}
        if file_type and file_type not in allowed_file_types:
            return json.dumps({
                "error": f"Invalid file_type: {file_type}. Must be one of: {', '.join(sorted(allowed_file_types))}",
                "status": "error"
            })

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

        result = run_command(cmd, cwd=dir_path)

        if result.get("status") == "error":
            return json.dumps(result)

        output = result.get("stdout", "")
        files = [f.strip() for f in output.split('\n') if f.strip()]

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


def transform_text_impl(
    input_text: Optional[str],
    operation: Optional[str],
    expression: Optional[str],
    is_file: bool,
    data_path: str,
    validate_read_path: Callable[[str], str],
    run_command: Callable[..., Dict[str, Any]],
) -> str:
    """Core logic for transform_text tool."""
    if err := validate_required(input_text=input_text, operation=operation, expression=expression):
        return err
    try:
        if operation not in ["sed", "awk", "tr"]:
            return json.dumps({
                "error": f"Invalid operation: {operation}. Use 'sed', 'awk', or 'tr'",
                "status": "error"
            })

        temp_path = None
        if is_file:
            file_path = validate_read_path(input_text)
            if not os.path.isfile(file_path):
                return json.dumps({
                    "error": f"File not found: {file_path}",
                    "status": "error"
                })
            input_source = f"cat {shlex.quote(file_path)}"
        else:
            tmp_dir = os.path.join(os.path.abspath(os.path.expanduser(data_path)), "tmp")
            secure_makedirs(tmp_dir)
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

        result = run_command(cmd)

        if not is_file and temp_path:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        if result.get("status") == "error":
            return json.dumps(result)

        output = truncate_output(result.get("stdout", ""))

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


def get_document_context_impl(
    path: Optional[str],
    query: Optional[str],
    keywords: Optional[str],
    context_chars: int,
    max_sections: int,
    validate_read_path: Callable[[str], str],
) -> str:
    """Core logic for get_document_context tool."""
    if err := validate_required(path=path, query=query):
        return err
    try:
        file_path = validate_read_path(path)

        if not os.path.isfile(file_path):
            return json.dumps({
                "error": f"File not found: {file_path}",
                "path": path,
                "status": "error"
            })

        content, file_format = get_file_content(file_path)

        if file_format == "error" or not content:
            return json.dumps({
                "error": "Failed to read file",
                "path": path,
                "status": "error"
            })

        search_terms = set()

        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
                     'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
                     'what', 'when', 'where', 'which', 'who', 'how', 'this', 'that',
                     'with', 'from', 'they', 'will', 'would', 'there', 'their'}

        for word in re.findall(r'\b\w+\b', query.lower()):
            if len(word) > 3 and word not in stopwords:
                search_terms.add(word)

        if keywords:
            for kw in keywords.split(','):
                kw = kw.strip().lower()
                if kw:
                    search_terms.add(kw)

        if not search_terms:
            search_terms = {w.lower() for w in query.split() if len(w) > 2}

        matches = []
        content_lower = content.lower()

        for term in search_terms:
            for match in re.finditer(re.escape(term), content_lower):
                matches.append({
                    "position": match.start(),
                    "term": term
                })

        matches.sort(key=lambda x: x["position"])

        sections = []
        used_ranges = []

        for match in matches:
            pos = match["position"]

            is_covered = any(start <= pos <= end for start, end in used_ranges)
            if is_covered:
                continue

            start = max(0, pos - context_chars // 2)
            end = min(len(content), pos + context_chars // 2)

            if start > 0:
                sentence_start = content.rfind('.', start - 100, start)
                if sentence_start > start - 100:
                    start = sentence_start + 1

            if end < len(content):
                sentence_end = content.find('.', end, end + 100)
                if sentence_end != -1:
                    end = sentence_end + 1

            section_content = content[start:end].strip()

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
# EXTRACT PDF IMAGES
# =============================================================================


def extract_pdf_images_impl(
    pdf_path: Optional[str],
    output_dir: str,
    min_size: int,
    validate_read_path: Callable[[str], str],
    validate_write_path: Callable[[str], str],
    default_output_dir: str,
) -> str:
    """Core logic for extracting embedded images from a PDF.

    Lived in the web-search server, which meant the only tool that could reach
    it was the one that server exposed — so the consolidated ``read_file``
    could offer ``mode="images"`` and the bash server's could not, and the two
    read_file tools disagreed about what they accepted.  It is a document
    operation, not a web one; here both servers can offer the same contract.
    """
    if err := validate_required(pdf_path=pdf_path):
        return err
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return json.dumps({
            "error": "PyMuPDF not installed. Run: pip install PyMuPDF",
            "pdf_path": pdf_path
        })

    try:
        output_path = (validate_write_path(output_dir) if output_dir
                       else default_output_dir)
        secure_makedirs(output_path)

        # Handle URL or local path
        if pdf_path.startswith(('http://', 'https://')):
            # Imported here rather than at module scope: this module is also
            # the sandbox filesystem server's dependency, and it should not
            # acquire an HTTP stack just by being imported.
            import requests
            from urllib.parse import urlparse
            response = requests.get(pdf_path, timeout=30)
            response.raise_for_status()
            expected = response.headers.get('Content-Length')
            if expected and len(response.content) < int(expected):
                return json.dumps({
                    "error": f"Incomplete PDF download "
                             f"({len(response.content)}/{expected} bytes)",
                    "pdf_path": pdf_path})
            doc = fitz.open(stream=response.content, filetype="pdf")
            pdf_name = os.path.basename(urlparse(pdf_path).path) or "document"
        else:
            local_path = validate_read_path(pdf_path)
            if not os.path.exists(local_path):
                return json.dumps({"error": f"PDF not found: {local_path}"})
            doc = fitz.open(local_path)
            pdf_name = os.path.splitext(os.path.basename(local_path))[0]

        extracted_images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]

                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image["width"]
                    height = base_image["height"]

                    # Skip small images (likely icons/artifacts)
                    if width < min_size or height < min_size:
                        continue

                    image_filename = (f"{pdf_name}_p{page_num + 1}"
                                      f"_img{img_index + 1}.{image_ext}")
                    image_path = os.path.join(output_path, image_filename)

                    # Owner-only permissions
                    fd = os.open(image_path,
                                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, "wb") as f:
                        f.write(image_bytes)

                    extracted_images.append({
                        "path": image_path,
                        "width": width,
                        "height": height,
                        "format": image_ext,
                        "page": page_num + 1
                    })

                except Exception:
                    # Skip images that can't be extracted
                    continue

        doc.close()

        return json.dumps({
            "pdf_path": pdf_path,
            "output_dir": output_path,
            "images": extracted_images,
            "image_count": len(extracted_images),
            "status": "success" if extracted_images else "no images found"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "pdf_path": pdf_path,
            "status": "failed"
        })


# =============================================================================
# READ FILE  (one definition, registered by every server that offers it)
# =============================================================================

# The single description for the read_file tool.  Shared rather than written
# out per server on purpose: two servers offering the same name with different
# text is a tool-name collision to ToolRegistry, and a model shown two
# conflicting descriptions of one tool learns to guess at its parameters.
READ_FILE_DESCRIPTION = """Read a file or extract structured content from it.

Args:
- path: FULL absolute file path within the session working directory (required).
  Always use the complete path — never a relative one.
- mode: What to extract — "text" (default), "tables", or "images"
  - "text"   : Return file content. Supports text files and PDFs; binary files return metadata.
  - "tables" : Extract tables from PDF or markdown. Returns structured rows/headers.
  - "images" : Extract embedded images from a PDF and save them locally.
- encoding: Text encoding for "text" mode (default: utf-8)
- max_chars: Max characters for "text" mode (default: 100000)
- table_index: For "tables" — specific table to return (1-based, default: all)
- output_format: For "tables" — "json" or "markdown" (default: "json")
- output_dir: For "images" — directory to save extracted images (default: data_path/pdf_images)
- min_size: For "images" — minimum image dimension in pixels to extract (default: 100)
- data_path: Session working directory — set automatically by the harness; leave unset.

There is no offset/limit paging: use max_chars to bound a large file.

Returns JSON, varying by mode:
  text:   {content, path, size_bytes, format, status}
  tables: {tables, total_tables, file, format, status}
  images: {pdf_path, output_dir, images, image_count, status}"""

READ_FILE_MODES = ("text", "tables", "images")


def read_file_impl(
    path: Optional[str],
    mode: str,
    encoding: str,
    max_chars: int,
    table_index: Optional[int],
    output_format: str,
    output_dir: str,
    min_size: int,
    *,
    read_text: Callable[..., str],
    extract_tables: Callable[..., str],
    extract_images: Callable[..., str],
) -> str:
    """Route a read_file call to the reader its ``mode`` asks for.

    The three readers are injected because each server jails paths its own
    way; the routing — which mode means which reader, and what an unknown mode
    does — is the part that has to be identical everywhere, so it lives here.
    """
    if err := validate_required(path=path):
        return err
    if mode == "text":
        return read_text(path=path, encoding=encoding, max_chars=max_chars)
    if mode == "tables":
        return extract_tables(path=path, table_index=table_index,
                              output_format=output_format)
    if mode == "images":
        return extract_images(pdf_path=path, output_dir=output_dir,
                              min_size=min_size)
    return json.dumps({
        "error": f"Unknown mode '{mode}'. Use: {', '.join(READ_FILE_MODES)}",
        "status": "error"
    })


# =============================================================================
# SEARCH DOCUMENT  (one definition, registered by every server that offers it)
# =============================================================================

SEARCH_DOCUMENT_DESCRIPTION = """Search within a single document file. Supports text, PDF, and markdown.

Args:
- path: FULL absolute file path within the session working directory (required)
- mode: Search strategy — "pattern" (default) or "context"
  - "pattern" : Regex search. Returns matching lines with surrounding context lines.
  - "context" : Keyword/query-based. Returns the most relevant text sections.
- pattern: Regex to match (required for mode="pattern", e.g., "error.*timeout")
- query: Question or topic (required for mode="context", e.g., "what is the conclusion?")
- keywords: Extra keywords for mode="context" (comma-separated)
- case_sensitive: Case-sensitive matching for mode="pattern" (default: false)
- context_lines: Lines of context around each match for mode="pattern" (default: 3)
- max_matches: Max matches for mode="pattern" (default: 50)
- context_chars: Characters of context per section for mode="context" (default: 500)
- max_sections: Max sections for mode="context" (default: 5)
- data_path: Session working directory — set automatically by the harness; leave unset.

Returns JSON:
  pattern: {matches, total_matches, file, format, status}
  context: {sections, total_sections, query, file, format, status}"""

SEARCH_DOCUMENT_MODES = ("pattern", "context")


def search_document_dispatch_impl(
    path: Optional[str],
    mode: str,
    pattern: Optional[str],
    query: Optional[str],
    keywords: Optional[str],
    case_sensitive: bool,
    context_lines: int,
    max_matches: int,
    context_chars: int,
    max_sections: int,
    *,
    search_pattern: Callable[..., str],
    search_context: Callable[..., str],
) -> str:
    """Route a search_document call to the search its ``mode`` asks for.

    Same shape as ``read_file_impl`` and for the same reason: the routing has
    to be identical in every server that offers the tool, while the searches
    themselves are bound to each server's own path jail.
    """
    if err := validate_required(path=path):
        return err
    if mode == "pattern":
        if err := validate_required(pattern=pattern):
            return err
        return search_pattern(
            path=path, pattern=pattern, case_sensitive=case_sensitive,
            context_lines=context_lines, max_matches=max_matches)
    if mode == "context":
        # A model that reaches for "context" mode often fills in `pattern`
        # out of habit; treat it as the query rather than refusing.
        effective_query = query or pattern
        if not effective_query:
            return json.dumps({
                "error": "query (or pattern) is required for mode='context'",
                "status": "error"})
        return search_context(
            path=path, query=effective_query, keywords=keywords,
            context_chars=context_chars, max_sections=max_sections)
    return json.dumps({
        "error": f"Unknown mode '{mode}'. Use: {', '.join(SEARCH_DOCUMENT_MODES)}",
        "status": "error"
    })
