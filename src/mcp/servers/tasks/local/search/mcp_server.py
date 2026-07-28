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
into a per-session index. local_search queries both, fuses the results by rank,
and collapses a document held by both indexes into a single hit.
Every server start re-ingests the shared corpus into the index persisted under
DATA_PATH, so a restart picks up corpus changes made while the server was down
without discarding work (or a sibling service's index — one data directory is
commonly shared by several onit containers). While the server is up, every
search incrementally re-ingests
the corpora it queries, so documents added, edited, or deleted mid-run are
picked up without an explicit index_documents call.

Optional environment variables for dense/hybrid retrieval:
    ONIT_EMBEDDING_HOST    - OpenAI-compatible base URL (e.g. vLLM, Ollama)
    ONIT_EMBEDDING_MODEL   - embedding model name
    ONIT_EMBEDDING_API_KEY - API key if the endpoint requires one
'''

import contextlib
import json
import os
import tempfile
import threading
import time
from typing import Any, Dict, Optional, Tuple

from fastmcp import FastMCP

try:
    from .toolkit import (LocalSearchIndex, DEFAULT_CHUNK_SIZE,
                          DEFAULT_CHUNK_OVERLAP, MAX_DOCUMENT_SUMMARIES,
                          RRF_K, cap_per_file, file_content_hash)
except ImportError:
    from toolkit import (LocalSearchIndex, DEFAULT_CHUNK_SIZE,
                         DEFAULT_CHUNK_OVERLAP, MAX_DOCUMENT_SUMMARIES,
                         RRF_K, cap_per_file, file_content_hash)

import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("Local Search MCP Server")

# Data path for the index (set via options['data_path'] in run())
DATA_PATH = os.path.join(tempfile.gettempdir(), "onit", "data")

# Optional read-only documents directory (set via options['documents_path'])
DOCUMENTS_PATH = None

# Name of the shared index directory created inside DOCUMENTS_PATH. Dotted so
# the corpus walk in index_directory(), which skips dot-components, does not
# descend into the index it is writing.
SHARED_INDEX_DIRNAME = ".local_search"

# Cached index instances, keyed by index directory so DATA_PATH changes and
# per-session jail roots each get their own index
_INDEXES: dict[str, LocalSearchIndex] = {}

# Resolved shared-index location per (corpus, DATA_PATH), so the writability
# probe behind _shared_index_dir() runs once instead of once per query
_SHARED_INDEX_DIRS: dict[Tuple[str, str], str] = {}

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


def _fallback_shared_index_dir() -> str:
    """Where the shared index goes when it cannot live beside its corpus."""
    root = os.path.realpath(os.path.expanduser(DATA_PATH))
    return os.path.join(root, "local_search", "shared")


def _shared_index_dir() -> str:
    """Directory holding the index of the DOCUMENTS_PATH corpus.

    The corpus is identical for every session, so its index is built once at
    the server level rather than per session jail. It is stored *inside*
    DOCUMENTS_PATH, beside the documents it describes, so the index and the
    originals stay together: a corpus mounted into another deployment arrives
    already ingested, and the index does not accumulate under a DATA_PATH that
    is per-deployment scratch. The directory name is dotted because
    ``index_directory`` skips path components starting with a dot — otherwise
    the ingest walk would descend into the index it is writing.

    A read-only corpus is the normal container deployment — docker-compose
    mounts DOCUMENTS_PATH with ``:ro`` — so when it cannot be written the index
    goes under DATA_PATH instead, at the path it has always used. Creating the
    directory is not proof that the index can be written into it: a corpus
    remounted read-only still has the ``.local_search`` left by an earlier
    read-write run, and ``makedirs(exist_ok=True)`` returns happily for a
    directory that already exists. Writability is therefore checked, not
    inferred.

    The verdict is cached per (corpus, DATA_PATH) pair: this runs on every
    query, and re-probing each time would cost syscalls per search for an
    answer that cannot change while the server runs.

    An index that is not there yet — a fresh corpus, or one whose index was
    written by an older version somewhere else — is simply built from the
    corpus by the next ingest. Nothing is carried over from another directory:
    the corpus on disk is the only thing that defines the index, so rebuilding
    from it cannot resurrect entries for documents that have since changed.
    """
    docs = _documents_root()
    if not docs:
        return _fallback_shared_index_dir()

    cache_key = (docs, os.path.realpath(os.path.expanduser(DATA_PATH)))
    cached = _SHARED_INDEX_DIRS.get(cache_key)
    if cached:
        return cached

    resolved = os.path.join(docs, SHARED_INDEX_DIRNAME)
    reason = ""
    try:
        os.makedirs(resolved, mode=0o700, exist_ok=True)
        if not os.access(resolved, os.W_OK | os.X_OK):
            reason = "directory is not writable"
    except OSError as e:
        reason = str(e)

    if reason:
        resolved = _fallback_shared_index_dir()
        logger.info(
            f"Shared local_search index cannot be kept beside the corpus "
            f"{docs} ({reason}); using {resolved}"
        )

    _SHARED_INDEX_DIRS[cache_key] = resolved
    return resolved


def _is_shared_corpus(corpus: str) -> bool:
    docs = _documents_root()
    return bool(docs) and (corpus == docs or corpus.startswith(docs + os.sep))


def _get_index(index_dir: str) -> LocalSearchIndex:
    # One cached index per directory so interleaved sessions don't evict
    # each other's in-memory index.
    if index_dir not in _INDEXES:
        _INDEXES[index_dir] = LocalSearchIndex(index_dir)
    return _INDEXES[index_dir]


def rebuild_indexes(background: bool = True) -> Optional[threading.Thread]:
    """Re-ingest the shared corpus once per server start.

    The pass is incremental: ``index_directory`` re-parses the files whose
    mtime or size changed, drops the entries of files that no longer exist,
    and leaves everything else — including the embeddings already computed for
    it — untouched. That is enough to make a restart authoritative over a
    corpus edited while the server was down, and it does not make every
    container start pay to re-embed a corpus that did not change.

    Nothing persisted under DATA_PATH is deleted. Several onit services (web,
    a2a, gateway, terminal) routinely bind-mount one data directory, and each
    runs its own copy of this server: a start that wiped every index under
    that directory would delete the shared index a sibling had just built and
    the live per-session indexes of sibling sessions still serving traffic.

    The re-ingest runs in a daemon thread holding _SHARED_LOCK: binding the
    server port is not delayed by parsing and embedding a large corpus, and
    searches arriving mid-rebuild block on the lock instead of reading a
    half-built index. Returns the thread (None when the rebuild ran inline or
    there is no shared corpus to rebuild).
    """
    _INDEXES.clear()
    # run() assigns DATA_PATH/DOCUMENTS_PATH just before calling this, so the
    # cached shared-index location may predate the corpus now configured.
    _SHARED_INDEX_DIRS.clear()

    docs_root = _documents_root()
    if not docs_root or not os.path.isdir(docs_root):
        return None

    def _build() -> None:
        try:
            started = time.monotonic()
            with _SHARED_LOCK:
                index = _get_index(_shared_index_dir())
                result = index.index_directory(docs_root, recursive=True)
            elapsed = time.monotonic() - started
            if not result["total_documents"]:
                # Logged as an error because it is a misconfiguration, not a
                # state: the corpus is mounted but local_search can only answer
                # "index is empty" until it holds a supported document.
                logger.error(
                    f"Shared local_search corpus {docs_root} holds no indexable "
                    f"document — check that the corpus is mounted and contains "
                    f"pdf/md/txt/csv/docx/xlsx files"
                )
                return
            # The elapsed time is logged because a search taking the lock waits
            # exactly this long: an unwarmed corpus is paid for by whoever asks
            # the first question, and nothing else reports that cost.
            logger.info(
                f"Shared local_search index from {docs_root} ready in "
                f"{elapsed:.1f}s: "
                f"{result['total_documents']} document(s), "
                f"{result['total_chunks']} chunk(s); "
                f"{len(result['indexed'])} (re)ingested, "
                f"{result['skipped_unchanged']} unchanged, "
                f"{len(result['removed'])} dropped"
            )
            if result["no_text_extracted"]:
                # Warned, not merely logged: these files are in the corpus and
                # counted as documents, but no query can ever reach them.
                logger.warning(
                    f"{len(result['no_text_extracted'])} document(s) in "
                    f"{docs_root} yielded no extractable text and are not "
                    f"searchable (scanned PDFs need OCR): "
                    f"{', '.join(os.path.basename(p) for p in result['no_text_extracted'])}"
                )
        except Exception as e:
            logger.error(f"Failed to rebuild shared local_search index: {e}")

    if not background:
        _build()
        return None
    thread = threading.Thread(target=_build, name="local-search-rebuild", daemon=True)
    thread.start()
    return thread


def _session_indexes(base: str) -> tuple[LocalSearchIndex, Optional[LocalSearchIndex]]:
    """The (session, shared) index pair visible to a session. The shared
    index exists only when a DOCUMENTS_PATH corpus is configured."""
    session_index = _get_index(_index_dir(base))
    shared_index = _get_index(_shared_index_dir()) if _documents_root() else None
    return session_index, shared_index


def _refresh_corpus(corpus: str, session_index: LocalSearchIndex,
                    shared_index: Optional[LocalSearchIndex]) -> None:
    """Incrementally re-ingest ``corpus`` into whichever index owns it.

    Cheap enough to run on every query: ``index_directory`` re-parses only the
    files whose mtime or size changed and rewrites index.json only when one
    did, so an unchanged corpus costs a single stat-per-file walk — well under
    the BM25 pass the search itself runs. Ingestion into the shared index is
    serialized: taking the lock also parks a search behind an in-flight startup
    rebuild, so it never reads a partially populated index.

    A session walk never descends into DOCUMENTS_PATH. The corpus can sit
    inside the data directory, and in the terminal UI the jail root *is* that
    directory, so without the exclusion the session index would ingest a second
    copy of every shared document: the same file would then contribute to the
    fused ranking from both indexes, and every session would separately pay to
    parse and embed a corpus the shared index already holds.
    """
    if not corpus or not os.path.isdir(corpus):
        return
    shared = _is_shared_corpus(corpus)
    index = shared_index if shared else session_index
    if index is None:
        return
    docs_root = _documents_root()
    exclude = [docs_root] if docs_root and not shared else None
    with _SHARED_LOCK if shared else contextlib.nullcontext():
        index.index_directory(corpus, recursive=True, exclude=exclude)


def _summarize_documents(merged: list, indexes: list) -> list:
    """Collapse a page of chunks into the documents behind it.

    The results are excerpts chosen for repeating the query's words, which is
    not the same question as "what is this document, and does it apply?" —
    and answering that question was costing one round trip per document,
    serially, before the agent could say anything. Each document here carries
    its opening, so most of those reads no longer need to happen.

    Only the first MAX_DOCUMENT_SUMMARIES documents are described: a tool
    result has a size budget (the caller truncates at MAX_TOOL_RESPONSE), and
    spending it on the tail of the ranking would cost the top of it.
    """
    by_file: Dict[str, Dict[str, Any]] = {}
    for result in merged:
        entry = by_file.setdefault(result["file"], {
            "file": result["file"],
            "best_rank": result["rank"],
            "matched_at": [],
        })
        if result.get("location"):
            entry["matched_at"].append(result["location"])

    summaries = []
    for entry in list(by_file.values())[:MAX_DOCUMENT_SUMMARIES]:
        for index in indexes:
            opening = index.document_opening(entry["file"])
            if opening:
                entry.update(opening)
                break
        summaries.append(entry)
    return summaries


def _document_key(index: LocalSearchIndex, file_path: str) -> str:
    """Identity of a document, stable across indexes.

    The absolute path cannot identify a document during a cross-index merge:
    the shared corpus file and a session's copy of it are the same document
    under two paths, and returning both would hand the agent the same text
    twice. The content hash recorded at ingest is path-independent, so the
    copies collapse to one entry.

    An index persisted before hashes were recorded has none to key on, and
    reading the filename off the other index's entry would not match it, so
    the hash is computed on the spot and memoized. Only a file that has since
    been deleted or become unreadable falls through to its normalized name.
    """
    info = index.documents.get(file_path)
    content_hash = info.get("hash") if info else None
    if not content_hash:
        try:
            content_hash = file_content_hash(file_path)
        except OSError:
            return os.path.basename(file_path).casefold()
        if info is not None:
            info["hash"] = content_hash  # memoize, but never invent an entry
    return content_hash


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
        # As in _refresh_corpus: a session ingest leaves the shared corpus to
        # the shared index rather than taking a second copy of it.
        docs_root = _documents_root()
        exclude = [docs_root] if docs_root and not shared else None

        with lock:
            index.chunk_size = max(200, min(int(chunk_size), 8000))
            index.chunk_overlap = max(0, min(int(chunk_overlap), index.chunk_size // 2))
            result = index.index_directory(corpus, recursive=recursive,
                                           rebuild=rebuild, exclude=exclude)

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
        "no_text_extracted": sorted(
            p for s in statuses for p in s["no_text_extracted"]),
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
            _refresh_corpus(_validate_corpus_path(path, base=base),
                            session_index, shared_index)
        else:
            # Both corpora a query can reach are refreshed, not just whichever
            # one happens to be empty: a document added to the shared corpus or
            # dropped into the session's own folder after the server started is
            # otherwise invisible until a restart.
            if docs_root:
                _refresh_corpus(docs_root, session_index, shared_index)
            if not _is_shared_corpus(base):
                _refresh_corpus(base, session_index, shared_index)

        indexes = [i for i in (shared_index, session_index) if i is not None and i.chunks]
        if not indexes:
            return json.dumps({
                "error": "Search index is empty. Run index_documents first, or set "
                         "documents_path (or ONIT_DOCUMENTS_PATH) to your corpus.",
                "status": "error"
            })

        # Query the shared corpus index and the session index, then fuse the
        # two result lists by reciprocal rank. Raw scores are not comparable
        # across indexes — BM25 idf depends on the corpus statistics of the
        # index that produced it, and hybrid scores are already per-index RRF
        # sums — but ranks are. Passages are keyed by document content rather
        # than by path, so a session copy of a shared file is one result that
        # accumulates both indexes' contributions instead of two identical
        # ones. Equal ranks fuse to equal scores, so the merge interleaves the
        # two indexes, with the shared index taking ties on stable-sort order
        # (and its path, the canonical one, labelling any deduplicated pair).
        #
        # The passage text is part of the key because location does not
        # identify a chunk: a text or markdown file parses to a single
        # locationless block, and a pdf page or docx table is re-chunked, so
        # keying on (document, location) alone would merge every chunk of a
        # file into one entry. A long file whose chunks each match the query
        # weakly would then out-score a short, squarely relevant one by
        # summing contributions that belong to different passages.
        top_k = max(1, min(int(top_k), 20))
        fused: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for index in indexes:
            for rank, result in enumerate(
                    index.search(query, top_k=top_k, method=method)):
                key = (_document_key(index, result["file"]),
                       result["location"], result["text"])
                entry = fused.setdefault(key, {**result, "score": 0.0})
                entry["score"] += 1.0 / (RRF_K + rank + 1)

        # Each index capped its own page at MAX_CHUNKS_PER_FILE, but the two
        # pages fuse into one, so the cap is re-applied to the merged order.
        merged = cap_per_file(sorted(fused.values(), key=lambda r: -r["score"]),
                              top_k, lambda r: r["file"])
        for rank, result in enumerate(merged, 1):
            result["rank"] = rank
            result["score"] = round(result["score"], 6)

        # `results` is serialized ahead of `documents` so that the head of this
        # response is the ranking. A tool result is trimmed to its first
        # characters once the conversation moves past it (TOOL_RESULT_DECAY_CHARS),
        # and what survives that cut should be the best-ranked passages, whose
        # offset is fixed, rather than the document summaries, whose bulk grows
        # with top_k and pushes every passage out of the window: with summaries
        # first the top-ranked quotes sat at offset ~7.9k at top_k=5 and ~9.7k at
        # top_k=20, past any budget worth keeping; with the ranking first they
        # start at ~0.3k whatever top_k is. The summaries stay in the response in
        # full — this is byte order, not priority, and the guidance to read
        # `documents` first still stands. Each result also carries its `file`, so
        # a trimmed trace still records which documents were consulted.
        return json.dumps({
            "query": query,
            "method": method,
            "results": merged,
            "documents": _summarize_documents(merged, indexes),
            "total_results": len(merged),
            "total_documents": sum(len(i.documents) for i in indexes),
            "total_chunks": sum(len(i.chunks) for i in indexes),
            "status": "success"
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "query": query, "status": "error"})


# Tool descriptions live here as constants because tools/mcp_server.py re-registers
# the same two tools on the consolidated server the web UI talks to. Duplicated
# prose drifts: the retrieval guidance below was added here and the copy over
# there kept serving the older text, so the UI's model never saw it.
INDEX_DOCUMENTS_DESCRIPTION = """Ingest in-house documents into the local search index.
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


LOCAL_SEARCH_DESCRIPTION = """Search in-house documents (pdf, md, txt, csv, docx, xlsx)
using the local search index. Automatically ingests the default corpus on
first use. Use this for questions about internal/private data instead of
web search.

Args:
- query: Natural-language query or keywords (required)
- top_k: Number of results (default: 5, max: 20)
- method: "hybrid" (default; BM25 + embeddings fused), "bm25" (lexical only),
  or "dense" (embeddings only — requires ONIT_EMBEDDING_HOST/MODEL)
- path: Optional corpus directory to (re)index before searching

Returns JSON: {query, method,
results: [{rank, score, file, location, text}],
documents: [{file, best_rank, matched_at, opening, num_chunks}],
total_results, total_documents, total_chunks, status}

Read `documents` first. One entry per document behind the results, carrying
that document's opening — its title and usually the summary under it — which
is what identifies the document and what its matched excerpts routinely do
not show. `results` are the individual matched chunks, ranked; one long
document can fill several slots, so repeated hits there measure document
length, not relevance.

The opening plus the matched excerpts answer most questions outright. When
they do not, do not read the whole file: call `search_document` with
mode="context" and the question as `query` to pull the relevant passages out
of it. Reserve `read_file` for short documents, or when the passages you got
back point at something specific you still need.

A README or other index file names every topic in the corpus and so ranks
high on any query; follow it to the document it points at instead of
answering from it. If every top hit comes from the same document and none of
them answer the question, re-query with different terms."""


# Register as MCP tools only when local search is not disabled
if not os.environ.get('ONIT_DISABLE_LOCAL_SEARCH'):
    @mcp.tool(
        title="Index Local Documents",
        description=INDEX_DOCUMENTS_DESCRIPTION,
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
        description=LOCAL_SEARCH_DESCRIPTION,
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

    # Every start re-ingests the corpus as it is now.  In the background by
    # default, so the server answers other tools immediately — but a search
    # arriving meanwhile blocks on _SHARED_LOCK until the ingest finishes, so
    # the wait lands on the first question rather than on startup.  Set
    # warm_index (or ONIT_WARM_INDEX=1) to move it back to startup, where a
    # deployment can wait for it before taking traffic.
    warm = options.get('warm_index')
    if warm is None:
        warm = os.environ.get('ONIT_WARM_INDEX') == '1'
    if warm:
        logger.info("Warming the local_search index before serving...")
    rebuild_indexes(background=not warm)

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
