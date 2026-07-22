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

Local Search MCP Server

Search in-house documents (PDF, Markdown, Excel, text, Word) with a local
ingestion + retrieval pipeline modeled on Mistral's Search Toolkit
(https://mistral.ai/news/search-toolkit/). Everything runs on local
infrastructure: parsing, chunking, BM25 indexing, and (optionally) dense
embeddings from any OpenAI-compatible endpoint.

2 Core Tools:
1. index_documents - Ingest a directory of documents into the search index
2. local_search    - Query the index (bm25, dense, or hybrid retrieval)

The read-only DOCUMENTS_PATH corpus is indexed once into a shared server-level
index reused by all sessions; files a session indexes from its own jail go
into a per-session index. local_search queries both and merges the results.

Optional environment variables for dense/hybrid retrieval:
    ONIT_EMBEDDING_HOST    - OpenAI-compatible base URL (e.g. vLLM, Ollama)
    ONIT_EMBEDDING_MODEL   - embedding model name
    ONIT_EMBEDDING_API_KEY - API key if the endpoint requires one
'''

import contextlib
import json
import os
import shutil
import tempfile
import threading
from typing import Optional

from fastmcp import FastMCP

try:
    from .toolkit import (LocalSearchIndex, DEFAULT_CHUNK_SIZE,
                          DEFAULT_CHUNK_OVERLAP, INDEX_FILENAME)
except ImportError:
    from toolkit import (LocalSearchIndex, DEFAULT_CHUNK_SIZE,
                         DEFAULT_CHUNK_OVERLAP, INDEX_FILENAME)

import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("Local Search MCP Server")

# Data path for the index (set via options['data_path'] in run())
DATA_PATH = os.path.join(tempfile.gettempdir(), "onit", "data")

# Optional read-only documents directory (set via options['documents_path'])
DOCUMENTS_PATH = None

# Cached index instances, keyed by index directory so DATA_PATH changes and
# per-session jail roots each get their own index
_INDEXES: dict[str, LocalSearchIndex] = {}

# Serializes ingestion into the shared DOCUMENTS_PATH index so concurrent
# sessions don't parse/chunk/embed the same corpus twice
_SHARED_LOCK = threading.Lock()


def _validate_required(**kwargs) -> str:
    """Check for missing required arguments. Returns JSON error string or empty string."""
    missing = [name for name, value in kwargs.items() if value is None]
    if missing:
        return json.dumps({
            "error": f"Missing required argument(s): {', '.join(missing)}.",
            "status": "error"
        })
    return ""


def _in_container() -> bool:
    """True when running inside the onit container (ONIT_CONTAINER=1). In that
    case the container is the filesystem boundary, so DATA_PATH-only path
    allowlists are relaxed."""
    return os.environ.get("ONIT_CONTAINER") == "1"


def _session_base(data_path: str | None = None) -> str:
    """Resolve the jail root for a tool call.

    ``data_path`` is the session working directory injected by the OnIt
    harness (it overwrites any model-supplied value, so it is trusted). It
    must live inside the server-wide DATA_PATH so one session cannot index or
    search a sibling session's folder. Falls back to DATA_PATH when absent."""
    abs_data = os.path.realpath(os.path.expanduser(DATA_PATH))
    if not data_path:
        return abs_data
    base = os.path.realpath(os.path.expanduser(data_path))
    if _in_container():
        return base
    if base != abs_data and not base.startswith(abs_data + os.sep):
        raise ValueError(
            f"data_path must be within the server data directory {abs_data}. "
            f"Got: {base}"
        )
    return base


def _validate_corpus_path(dir_path: str, base: str | None = None) -> str:
    """Validate that the corpus directory is within the jail root (the
    per-session ``base`` when given, else DATA_PATH) or DOCUMENTS_PATH.
    Returns absolute path. Raises ValueError if outside allowed directories."""
    abs_path = os.path.realpath(os.path.expanduser(dir_path))
    if _in_container():
        return abs_path
    allowed = [base or os.path.realpath(os.path.expanduser(DATA_PATH))]
    if DOCUMENTS_PATH:
        allowed.append(os.path.realpath(os.path.expanduser(DOCUMENTS_PATH)))
    for root in allowed:
        if abs_path == root or abs_path.startswith(root + os.sep):
            return abs_path
    raise ValueError(
        f"Corpus path must be within: {' or '.join(allowed)}. Got: {abs_path}"
    )


def _index_dir(base: str | None = None) -> str:
    root = base or os.path.abspath(os.path.expanduser(DATA_PATH))
    return os.path.join(root, "local_search")


def _documents_root() -> Optional[str]:
    if not DOCUMENTS_PATH:
        return None
    return os.path.realpath(os.path.expanduser(DOCUMENTS_PATH))


def _shared_index_dir() -> str:
    # The DOCUMENTS_PATH corpus is identical for every session, so its index
    # lives once at the server level instead of being rebuilt per session jail.
    root = os.path.realpath(os.path.expanduser(DATA_PATH))
    return os.path.join(root, "local_search", "shared")


def _is_shared_corpus(corpus: str) -> bool:
    docs = _documents_root()
    return bool(docs) and (corpus == docs or corpus.startswith(docs + os.sep))


def _get_index(index_dir: str) -> LocalSearchIndex:
    # One cached index per directory so interleaved sessions don't evict
    # each other's in-memory index.
    if index_dir not in _INDEXES:
        _INDEXES[index_dir] = LocalSearchIndex(index_dir)
    return _INDEXES[index_dir]


def _contains_shared_documents(index_path: str, docs_root: str) -> bool:
    """True when an index file holds documents from the shared corpus, or is
    unreadable/empty (junk either way — the index is derived data)."""
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            documents = json.load(f).get("documents", {})
    except Exception:
        return True
    if not documents:
        return True
    return any(p == docs_root or p.startswith(docs_root + os.sep)
               for p in documents)


def _cleanup_legacy_session_indexes() -> int:
    """Delete legacy per-session copies of the shared corpus index.

    Before the shared index existed, every session jail under DATA_PATH (and
    the server-level fallback index) held its own full copy of the
    DOCUMENTS_PATH corpus index. Those copies waste disk and would duplicate
    shared results in merged searches. Session indexes holding only
    session-local documents are kept. Returns the number of indexes removed;
    affected sessions simply re-ingest on next use.
    """
    docs_root = _documents_root()
    if not docs_root:
        return 0
    root = os.path.realpath(os.path.expanduser(DATA_PATH))
    removed = 0

    # Server-level fallback index: remove just its index file, since the
    # shared index lives in a subdirectory of the same local_search dir
    fallback = os.path.join(root, "local_search", INDEX_FILENAME)
    if os.path.isfile(fallback) and _contains_shared_documents(fallback, docs_root):
        with contextlib.suppress(OSError):
            os.remove(fallback)
            removed += 1

    try:
        children = os.listdir(root)
    except OSError:
        return removed
    for child in children:
        if child == "local_search":
            continue
        index_dir = os.path.join(root, child, "local_search")
        index_path = os.path.join(index_dir, INDEX_FILENAME)
        if os.path.isfile(index_path) and _contains_shared_documents(index_path, docs_root):
            shutil.rmtree(index_dir, ignore_errors=True)
            removed += 1
    return removed


def _session_indexes(base: str) -> tuple[LocalSearchIndex, Optional[LocalSearchIndex]]:
    """The (session, shared) index pair visible to a session. The shared
    index exists only when a DOCUMENTS_PATH corpus is configured."""
    session_index = _get_index(_index_dir(base))
    shared_index = _get_index(_shared_index_dir()) if _documents_root() else None
    return session_index, shared_index


def _default_corpus(base: str | None = None) -> Optional[str]:
    """Default corpus directory: DOCUMENTS_PATH when set, else the jail root."""
    root = DOCUMENTS_PATH or base or DATA_PATH
    root = os.path.realpath(os.path.expanduser(root))
    return root if os.path.isdir(root) else None


def _index_documents_impl(
    path: Optional[str] = None,
    recursive: bool = True,
    rebuild: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    status_only: bool = False,
    data_path: str = "",
) -> str:
    """Core index_documents implementation."""
    try:
        base = _session_base(data_path)

        if status_only:
            return json.dumps({**_combined_status(base), "status": "success"}, indent=2)

        corpus = _validate_corpus_path(path, base=base) if path else _default_corpus(base)
        if not corpus or not os.path.isdir(corpus):
            return json.dumps({
                "error": f"Corpus directory not found: {path or corpus}. "
                         "Set documents_path (or ONIT_DOCUMENTS_PATH) or pass an explicit path.",
                "status": "error"
            })

        # The shared DOCUMENTS_PATH corpus is indexed once at the server level
        # and reused by all sessions; anything else goes to the session index.
        shared = _is_shared_corpus(corpus)
        session_index, shared_index = _session_indexes(base)
        index = shared_index if shared else session_index
        lock = _SHARED_LOCK if shared else contextlib.nullcontext()

        with lock:
            index.chunk_size = max(200, min(int(chunk_size), 8000))
            index.chunk_overlap = max(0, min(int(chunk_overlap), index.chunk_size // 2))
            result = index.index_directory(corpus, recursive=recursive, rebuild=rebuild)

        scope = "shared" if shared else "session"
        return json.dumps({**result, "scope": scope, "status": "success"}, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "path": path, "status": "error"})


def _combined_status(base: str) -> dict:
    """Merged statistics across the shared corpus index and the session index."""
    session_index, shared_index = _session_indexes(base)
    session_status = session_index.status()
    shared_status = shared_index.status() if shared_index else None
    statuses = [s for s in (shared_status, session_status) if s]

    by_format: dict[str, int] = {}
    for s in statuses:
        for fmt, n in s["documents_by_format"].items():
            by_format[fmt] = by_format.get(fmt, 0) + n

    return {
        "total_documents": sum(s["total_documents"] for s in statuses),
        "total_chunks": sum(s["total_chunks"] for s in statuses),
        "embedded_chunks": sum(s["embedded_chunks"] for s in statuses),
        "documents_by_format": by_format,
        "embedding_model": next(
            (s["embedding_model"] for s in statuses if s["embedding_model"]), None),
        "chunk_size": session_status["chunk_size"],
        "chunk_overlap": session_status["chunk_overlap"],
        "supported_formats": session_status["supported_formats"],
        "shared_index": shared_status,
        "session_index": session_status,
    }


def _local_search_impl(
    query: Optional[str] = None,
    top_k: int = 5,
    method: str = "hybrid",
    path: Optional[str] = None,
    data_path: str = "",
) -> str:
    """Core local_search implementation."""
    if err := _validate_required(query=query):
        return err
    if method not in ("bm25", "dense", "hybrid"):
        return json.dumps({
            "error": f"Unknown method '{method}'. Use: bm25, dense, hybrid",
            "status": "error"
        })
    try:
        base = _session_base(data_path)
        session_index, shared_index = _session_indexes(base)
        docs_root = _documents_root()

        if path:
            # Refresh an explicit corpus path in whichever index owns it
            corpus = _validate_corpus_path(path, base=base)
            if os.path.isdir(corpus):
                if _is_shared_corpus(corpus):
                    with _SHARED_LOCK:
                        shared_index.index_directory(corpus, recursive=True)
                else:
                    session_index.index_directory(corpus, recursive=True)
        elif shared_index is not None:
            # Auto-ingest the shared corpus on first use — one build serves
            # every session (double-checked under the lock)
            if not shared_index.chunks and os.path.isdir(docs_root):
                with _SHARED_LOCK:
                    if not shared_index.chunks:
                        shared_index.index_directory(docs_root, recursive=True)
        elif not session_index.chunks:
            corpus = _default_corpus(base)
            if corpus and os.path.isdir(corpus):
                session_index.index_directory(corpus, recursive=True)

        indexes = [i for i in (shared_index, session_index) if i is not None and i.chunks]
        if not indexes:
            return json.dumps({
                "error": "Search index is empty. Run index_documents first, or set "
                         "documents_path (or ONIT_DOCUMENTS_PATH) to your corpus.",
                "status": "error"
            })

        # Query the shared corpus index and the session index, then merge by
        # score. Hybrid scores are rank-based (RRF) and bm25 scores share a
        # scale, so cross-index comparison is reasonable.
        top_k = max(1, min(int(top_k), 20))
        merged = []
        for index in indexes:
            merged.extend(index.search(query, top_k=top_k, method=method))
        merged.sort(key=lambda r: -r["score"])
        merged = merged[:top_k]
        for rank, result in enumerate(merged, 1):
            result["rank"] = rank

        return json.dumps({
            "query": query,
            "method": method,
            "results": merged,
            "total_results": len(merged),
            "total_documents": sum(len(i.documents) for i in indexes),
            "total_chunks": sum(len(i.chunks) for i in indexes),
            "status": "success"
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "query": query, "status": "error"})


# Register as MCP tools only when local search is not disabled
if not os.environ.get('ONIT_DISABLE_LOCAL_SEARCH'):
    @mcp.tool(
        title="Index Local Documents",
        description="""Ingest in-house documents into the local search index.
Parses, chunks, and indexes files for BM25 and (when an embedding endpoint is
configured) dense retrieval. Unchanged files are skipped; deleted files are
dropped from the index. The shared documents_path corpus is indexed once and
reused across sessions; other paths go into the per-session index.

Supported formats: pdf, md, txt, csv, docx, xlsx

Args:
- path: Directory to index (default: documents_path, else data_path)
- recursive: Recurse into subdirectories (default: true)
- rebuild: Discard the existing index and re-ingest everything (default: false)
- chunk_size: Characters per chunk (default: 1600)
- chunk_overlap: Character overlap between chunks (default: 200)
- status_only: Only report index statistics without ingesting (default: false)

Returns JSON: {directory, indexed, skipped_unchanged, removed, errors,
total_documents, total_chunks, embedding_model, scope, status}"""
    )
    def index_documents(
        path: Optional[str] = None,
        recursive: bool = True,
        rebuild: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        status_only: bool = False,
        data_path: str = "",
    ) -> str:
        return _index_documents_impl(
            path=path, recursive=recursive, rebuild=rebuild,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            status_only=status_only, data_path=data_path,
        )

    @mcp.tool(
        title="Search Local Documents",
        description="""Search in-house documents (pdf, md, txt, csv, docx, xlsx)
using the local search index. Automatically ingests the default corpus on
first use. Use this for questions about internal/private data instead of
web search.

Args:
- query: Natural-language query or keywords (required)
- top_k: Number of results (default: 5, max: 20)
- method: "hybrid" (default; BM25 + embeddings fused), "bm25" (lexical only),
  or "dense" (embeddings only — requires ONIT_EMBEDDING_HOST/MODEL)
- path: Optional corpus directory to (re)index before searching

Returns JSON: {query, method, results: [{rank, score, file, location, text}],
total_results, total_documents, total_chunks, status}"""
    )
    def local_search(
        query: Optional[str] = None,
        top_k: int = 5,
        method: str = "hybrid",
        path: Optional[str] = None,
        data_path: str = "",
    ) -> str:
        return _local_search_impl(query=query, top_k=top_k, method=method, path=path,
                                  data_path=data_path)
else:
    # Provide plain function aliases so imports (e.g. from tools/mcp_server.py) still work
    index_documents = _index_documents_impl
    local_search = _local_search_impl


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18203,
    path: str = "/sse",
    options: dict = {}
) -> None:
    """Run the MCP server."""
    global DATA_PATH, DOCUMENTS_PATH

    if 'verbose' in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    if 'data_path' in options:
        DATA_PATH = options['data_path']
    elif os.environ.get('ONIT_DATA_PATH'):
        DATA_PATH = os.environ['ONIT_DATA_PATH']

    if 'documents_path' in options:
        DOCUMENTS_PATH = options['documents_path']
    elif os.environ.get('ONIT_DOCUMENTS_PATH'):
        DOCUMENTS_PATH = os.environ['ONIT_DOCUMENTS_PATH']

    _INDEXES.clear()  # Reset cached indexes in case DATA_PATH changed

    removed = _cleanup_legacy_session_indexes()
    if removed:
        logger.info(f"Removed {removed} legacy per-session local_search index(es)")

    logger.info(f"Starting Local Search MCP Server at {host}:{port}{path}")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Documents path: {DOCUMENTS_PATH}")
    logger.info("2 Core Tools: index_documents, local_search")

    quiet = 'verbose' not in options
    if quiet:
        import uvicorn.config
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    mcp.run(transport=transport, host=host, port=port, path=path,
            uvicorn_config={"access_log": False, "log_level": "warning"} if quiet else {})


if __name__ == "__main__":
    run()
