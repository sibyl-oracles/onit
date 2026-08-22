# Local Search over In-House Data


OnIt includes a local search toolkit modeled on the [Mistral Search Toolkit](https://mistral.ai/news/search-toolkit/): a composable pipeline that unifies **ingestion** (parse → chunk → embed/index) and **retrieval** (BM25 sparse, dense embeddings, hybrid fusion) behind a single interface. Everything runs on your own infrastructure — documents, index, and embeddings never leave your machine, so the agent can answer questions from private company data that web search cannot see.

| Format | Extension | Parser |
|--------|-----------|--------|
| PDF | `.pdf` | pypdf (per page) |
| Markdown | `.md`, `.markdown` | built-in |
| Text / CSV | `.txt`, `.text`, `.csv` | built-in |
| Word | `.docx` | python-docx (paragraphs and tables) |
| Excel | `.xlsx`, `.xlsm` | openpyxl (per sheet) |

## Quick start

```bash
# 1. Install the optional parsers for Word and Excel (PDF/md/txt work out of the box)
pip install "onit[search]"

# 2. Point OnIt at your document folder
export ONIT_DOCUMENTS_PATH=~/company-docs

# 3. Run and ask questions about your data
onit
> what is our vacation policy?
```

The agent uses two MCP tools, registered automatically in the tools servers:

| Tool | Description |
|------|-------------|
| `index_documents` | Ingest a directory: parse, chunk (default 1600 chars, 200 overlap), and index. Incremental — unchanged files are skipped, deleted files are dropped. Use `rebuild: true` to start fresh or `status_only: true` for index statistics. |
| `local_search` | Query the index and return ranked chunks with source file and location (page, sheet, table). Auto-ingests the default corpus on first use. |

## Retrieval methods

`local_search` supports three methods, selected with the `method` argument:

| Method | How it works | Requires |
|--------|--------------|----------|
| `bm25` | Okapi BM25 sparse lexical ranking (pure Python) | nothing |
| `dense` | Cosine similarity over chunk embeddings | an embedding endpoint |
| `hybrid` *(default)* | Reciprocal rank fusion of BM25 + dense rankings | falls back to `bm25` when no embedding endpoint is configured |

Dense and hybrid retrieval use any **OpenAI-compatible** `/embeddings` endpoint — a private vLLM or Ollama server keeps everything on-premises:

```bash
export ONIT_EMBEDDING_HOST=http://localhost:8000/v1   # vLLM, Ollama, etc.
export ONIT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
export ONIT_EMBEDDING_API_KEY=...                     # only if the endpoint needs one
                                                      # (falls back to VLLM_API_KEY)
```

When these are set, `index_documents` embeds chunks during ingestion and `local_search` embeds the query at search time. Without them, everything still works with BM25 — no network calls are made.

## How it works

```
Ingestion:  documents → parse (pdf/md/txt/csv/docx/xlsx) → chunk → [embed] → index
Retrieval:  query → BM25 ranking ─┐
            query → [dense ranking] ─┴→ reciprocal rank fusion → top-k chunks + sources
```

- The index is a single JSON file at `data_path/local_search/index.json` (owner-only permissions). Delete it or pass `rebuild: true` to re-ingest from scratch.
- Corpus directories must be inside `ONIT_DOCUMENTS_PATH` or `data_path` — the same filesystem sandbox that governs all OnIt file tools (relaxed inside `--container`).
- Set `ONIT_DISABLE_LOCAL_SEARCH=1` to unregister both tools.

## Adding a new document format

Parsers follow a small adapter interface: each returns a list of `(location, text)` blocks (e.g. `("page 3", ...)`, `("sheet Sales", ...)`). To support a new format, add a parser to `src/mcp/servers/tasks/local/search/toolkit.py`, register its extension in `SUPPORTED_EXTENSIONS`, and dispatch it from `parse_document()` — chunking, indexing, and retrieval pick it up automatically.

