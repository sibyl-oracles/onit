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

Tests for the Local Search toolkit and MCP tools (in-house document search).
'''

import json
import os

import pytest

from src.mcp.servers.tasks.local.search.toolkit import (
    BM25,
    LocalSearchIndex,
    chunk_text,
    parse_document,
    reciprocal_rank_fusion,
    tokenize,
)
import src.mcp.servers.tasks.local.search.mcp_server as local_mod


@pytest.fixture
def corpus(tmp_path):
    """A small corpus of markdown and text documents."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "vacation.md").write_text(
        "# Vacation Policy\n\n"
        "Employees accrue 15 days of paid vacation per year. "
        "Unused vacation days roll over up to a maximum of 30 days.\n"
    )
    (docs / "expenses.txt").write_text(
        "Expense reports must be submitted within 30 days of purchase. "
        "Meals are reimbursed up to 50 USD per day during business travel.\n"
    )
    (docs / "onboarding.md").write_text(
        "# Onboarding\n\n"
        "New hires receive a laptop and badge on day one. "
        "Security training is mandatory during the first week.\n"
    )
    return docs


@pytest.fixture
def search_env(tmp_path, corpus, monkeypatch):
    """Point the MCP module at a temp DATA_PATH/DOCUMENTS_PATH."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(local_mod, "DATA_PATH", str(data))
    monkeypatch.setattr(local_mod, "DOCUMENTS_PATH", str(corpus))
    monkeypatch.setattr(local_mod, "_INDEXES", {})
    monkeypatch.delenv("ONIT_EMBEDDING_HOST", raising=False)
    monkeypatch.delenv("ONIT_EMBEDDING_MODEL", raising=False)
    return data


# -- toolkit primitives -------------------------------------------------------

def test_tokenize():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_chunk_text_short_passthrough():
    assert chunk_text("short text", chunk_size=100) == ["short text"]


def test_chunk_text_splits_with_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 700 for c in chunks)  # chunk_size + carried overlap


def test_bm25_ranks_relevant_doc_first():
    corpus_tokens = [
        tokenize("the cat sat on the mat"),
        tokenize("dogs chase cats in the park"),
        tokenize("quantum computing uses qubits"),
    ]
    bm25 = BM25(corpus_tokens)
    scores = bm25.scores(tokenize("quantum qubits"))
    assert scores.index(max(scores)) == 2


def test_reciprocal_rank_fusion_prefers_agreement():
    fused = reciprocal_rank_fusion([[0, 1, 2], [1, 0, 2]])
    assert fused[0] == fused[1] > fused[2]


def test_parse_document_unsupported_extension(tmp_path):
    bad = tmp_path / "binary.exe"
    bad.write_text("data")
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_document(str(bad))


# -- index lifecycle -----------------------------------------------------------

def test_index_and_search(tmp_path, corpus):
    index = LocalSearchIndex(str(tmp_path / "idx"))
    result = index.index_directory(str(corpus))
    assert result["total_documents"] == 3
    assert result["total_chunks"] >= 3
    assert not result["errors"]

    results = index.search("how many vacation days do employees get", method="bm25")
    assert results
    assert results[0]["file"].endswith("vacation.md")


def test_index_persistence(tmp_path, corpus):
    index_dir = str(tmp_path / "idx")
    LocalSearchIndex(index_dir).index_directory(str(corpus))

    reloaded = LocalSearchIndex(index_dir)
    assert len(reloaded.documents) == 3
    results = reloaded.search("expense reimbursement for meals", method="bm25")
    assert results[0]["file"].endswith("expenses.txt")


def test_index_skips_unchanged_and_drops_deleted(tmp_path, corpus):
    index = LocalSearchIndex(str(tmp_path / "idx"))
    index.index_directory(str(corpus))

    result = index.index_directory(str(corpus))
    assert result["indexed"] == []
    assert result["skipped_unchanged"] == 3

    os.unlink(corpus / "onboarding.md")
    result = index.index_directory(str(corpus))
    assert len(result["removed"]) == 1
    assert result["total_documents"] == 2


def test_hybrid_falls_back_to_bm25_without_embeddings(tmp_path, corpus, monkeypatch):
    monkeypatch.delenv("ONIT_EMBEDDING_HOST", raising=False)
    monkeypatch.delenv("ONIT_EMBEDDING_MODEL", raising=False)
    index = LocalSearchIndex(str(tmp_path / "idx"))
    index.index_directory(str(corpus))
    results = index.search("security training", method="hybrid")
    assert results[0]["method"] == "bm25"
    assert results[0]["file"].endswith("onboarding.md")


def test_dense_method_errors_without_embeddings(tmp_path, corpus, monkeypatch):
    monkeypatch.delenv("ONIT_EMBEDDING_HOST", raising=False)
    monkeypatch.delenv("ONIT_EMBEDDING_MODEL", raising=False)
    index = LocalSearchIndex(str(tmp_path / "idx"))
    index.index_directory(str(corpus))
    with pytest.raises(ValueError, match="Dense retrieval unavailable"):
        index.search("anything", method="dense")


# -- MCP tool layer -------------------------------------------------------------

def test_mcp_index_documents_default_corpus(search_env):
    result = json.loads(local_mod._index_documents_impl())
    assert result["status"] == "success"
    assert result["total_documents"] == 3


def test_mcp_local_search_auto_indexes(search_env):
    result = json.loads(local_mod._local_search_impl(query="vacation days"))
    assert result["status"] == "success"
    assert result["total_results"] >= 1
    assert result["results"][0]["file"].endswith("vacation.md")


def test_mcp_local_search_requires_query(search_env):
    result = json.loads(local_mod._local_search_impl())
    assert result["status"] == "error"


def test_mcp_local_search_rejects_bad_method(search_env):
    result = json.loads(local_mod._local_search_impl(query="x", method="vector"))
    assert result["status"] == "error"


def test_mcp_index_documents_status_only(search_env):
    local_mod._index_documents_impl()
    status = json.loads(local_mod._index_documents_impl(status_only=True))
    assert status["status"] == "success"
    assert status["total_documents"] == 3
    assert status["embedded_chunks"] == 0


def test_mcp_rejects_path_outside_allowed_roots(search_env, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    result = json.loads(local_mod._index_documents_impl(path=str(outside)))
    assert result["status"] == "error"
    assert "must be within" in result["error"]


def test_mcp_rebuild(search_env):
    local_mod._index_documents_impl()
    result = json.loads(local_mod._index_documents_impl(rebuild=True))
    assert result["status"] == "success"
    assert len(result["indexed"]) == 3


# -- shared corpus index across sessions ---------------------------------------

def _make_session(data, name):
    session = data / name
    session.mkdir()
    return str(session)


def test_shared_corpus_indexed_once_across_sessions(search_env):
    session_a = _make_session(search_env, "session-a")
    session_b = _make_session(search_env, "session-b")

    result_a = json.loads(local_mod._index_documents_impl(data_path=session_a))
    assert result_a["scope"] == "shared"
    assert len(result_a["indexed"]) == 3

    # Second session reuses the shared index: nothing is re-parsed
    result_b = json.loads(local_mod._index_documents_impl(data_path=session_b))
    assert result_b["scope"] == "shared"
    assert result_b["indexed"] == []
    assert result_b["skipped_unchanged"] == 3

    # Only one shared index directory exists, none inside the session jails
    assert (search_env / "local_search" / "shared" / "index.json").is_file()
    assert not (search_env / "session-a" / "local_search").exists()
    assert not (search_env / "session-b" / "local_search").exists()


def test_search_uses_shared_index_across_sessions(search_env):
    session_a = _make_session(search_env, "session-a")
    session_b = _make_session(search_env, "session-b")

    result_a = json.loads(local_mod._local_search_impl(
        query="vacation days", data_path=session_a))
    assert result_a["status"] == "success"
    assert result_a["results"][0]["file"].endswith("vacation.md")

    # Session B searches the same shared index without re-ingesting
    shared_index = local_mod._get_index(local_mod._shared_index_dir())
    docs_before = dict(shared_index.documents)
    result_b = json.loads(local_mod._local_search_impl(
        query="vacation days", data_path=session_b))
    assert result_b["status"] == "success"
    assert result_b["results"][0]["file"].endswith("vacation.md")
    assert shared_index.documents == docs_before


def test_session_files_stay_isolated_and_merge_with_shared(search_env):
    session_a = _make_session(search_env, "session-a")
    session_b = _make_session(search_env, "session-b")

    private = search_env / "session-a" / "notes"
    private.mkdir()
    (private / "secret.md").write_text(
        "# Project Zebra\n\nProject Zebra launches in October with a 2M budget.\n"
    )
    result = json.loads(local_mod._index_documents_impl(
        path=str(private), data_path=session_a))
    assert result["scope"] == "session"
    assert len(result["indexed"]) == 1

    # Session A sees its private doc merged with the shared corpus
    found = json.loads(local_mod._local_search_impl(
        query="Project Zebra launch budget", data_path=session_a))
    assert found["results"][0]["file"].endswith("secret.md")
    assert found["total_documents"] == 4  # 3 shared + 1 session

    # Session B cannot see session A's private doc
    hidden = json.loads(local_mod._local_search_impl(
        query="Project Zebra launch budget", data_path=session_b))
    assert all(not r["file"].endswith("secret.md") for r in hidden["results"])
    assert hidden["total_documents"] == 3


def test_cleanup_removes_legacy_session_indexes(search_env, corpus):
    # Legacy layout: a full copy of the shared corpus index inside a session jail
    legacy = search_env / "old-session" / "local_search"
    LocalSearchIndex(str(legacy)).index_directory(str(corpus))
    assert (legacy / "index.json").is_file()

    # New-style session index holding only session-local documents
    private = search_env / "live-session" / "notes"
    private.mkdir(parents=True)
    (private / "own.md").write_text("Session-local notes about project alpha.\n")
    kept = search_env / "live-session" / "local_search"
    LocalSearchIndex(str(kept)).index_directory(str(private))

    # Legacy server-level fallback index, next to the new shared subdir
    fallback = search_env / "local_search"
    LocalSearchIndex(str(fallback)).index_directory(str(corpus))
    shared = search_env / "local_search" / "shared"
    LocalSearchIndex(str(shared)).index_directory(str(corpus))

    removed = local_mod._cleanup_legacy_session_indexes()
    assert removed == 2
    assert not legacy.exists()
    assert not (fallback / "index.json").exists()
    assert (kept / "index.json").is_file()       # session-only index kept
    assert (shared / "index.json").is_file()     # shared index untouched


def test_cleanup_noop_without_documents_path(search_env, corpus, monkeypatch):
    monkeypatch.setattr(local_mod, "DOCUMENTS_PATH", None)
    legacy = search_env / "old-session" / "local_search"
    LocalSearchIndex(str(legacy)).index_directory(str(corpus))

    assert local_mod._cleanup_legacy_session_indexes() == 0
    assert (legacy / "index.json").is_file()


def test_status_only_combines_shared_and_session(search_env):
    session_a = _make_session(search_env, "session-a")
    local_mod._index_documents_impl(data_path=session_a)

    status = json.loads(local_mod._index_documents_impl(
        status_only=True, data_path=session_a))
    assert status["status"] == "success"
    assert status["total_documents"] == 3
    assert status["shared_index"]["total_documents"] == 3
    assert status["session_index"]["total_documents"] == 0
