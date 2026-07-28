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
    DOCUMENT_OPENING_CHARS,
    MAX_CHUNKS_PER_FILE,
    MAX_DOCUMENT_SUMMARIES,
    RRF_K,
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


def _reset_search_globals(monkeypatch, data_path, documents_path):
    """Point the MCP module at a temp corpus and clear its module state.

    The caches and the refresh window are module globals that would otherwise
    carry from one test to the next; the refresh window in particular is
    wall-clock state, and a test that writes a file then searches for it needs
    the walk to actually happen.
    """
    monkeypatch.setattr(local_mod, "DATA_PATH", str(data_path))
    monkeypatch.setattr(local_mod, "DOCUMENTS_PATH", str(documents_path))
    monkeypatch.setattr(local_mod, "_INDEXES", {})
    monkeypatch.setattr(local_mod, "_SHARED_INDEX_DIRS", {})
    monkeypatch.setattr(local_mod, "_LAST_REFRESH", {})
    monkeypatch.setattr(local_mod, "REFRESH_MIN_INTERVAL", 0.0)


@pytest.fixture
def search_env(tmp_path, corpus, monkeypatch):
    """Point the MCP module at a temp DATA_PATH/DOCUMENTS_PATH."""
    data = tmp_path / "data"
    data.mkdir()
    _reset_search_globals(monkeypatch, data, corpus)
    monkeypatch.delenv("ONIT_EMBEDDING_HOST", raising=False)
    monkeypatch.delenv("ONIT_EMBEDDING_MODEL", raising=False)
    return data


# -- toolkit primitives -------------------------------------------------------

def test_tokenize():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_tokenize_folds_plurals_onto_the_singular():
    # A question asked in the plural has to reach a document titled in the
    # singular ("...-scholarship-..."), which BM25 treats as a separate term.
    assert tokenize("scholarships") == tokenize("scholarship")
    assert tokenize("policies") == tokenize("policy")
    assert tokenize("boxes") == tokenize("box")


def test_tokenize_leaves_non_plurals_alone():
    # Words that merely end in "s" must not be truncated into a different word.
    for word in ("process", "class", "campus", "syllabus", "gas", "ai"):
        assert tokenize(word) == [word]


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


def test_save_replaces_the_index_atomically(tmp_path, corpus):
    """A reader (a sibling onit service sharing the data directory) must never
    observe the truncated middle of a write."""
    index_dir = tmp_path / "idx"
    index = LocalSearchIndex(str(index_dir))
    index.index_directory(str(corpus))
    inode = (index_dir / "index.json").stat().st_ino

    (corpus / "parking.md").write_text("# Parking\n\nPermits from facilities.\n")
    index.index_directory(str(corpus))

    # A rename, not a truncate-in-place: the old file is never a partial one
    assert (index_dir / "index.json").stat().st_ino != inode
    assert json.loads((index_dir / "index.json").read_text())["version"] == 1
    assert [p.name for p in index_dir.iterdir()] == ["index.json"]


def test_index_of_an_older_version_is_not_read(tmp_path, corpus):
    index_dir = tmp_path / "idx"
    LocalSearchIndex(str(index_dir)).index_directory(str(corpus))
    persisted = index_dir / "index.json"
    data = json.loads(persisted.read_text())
    data["version"] = 0
    persisted.write_text(json.dumps(data))

    stale = LocalSearchIndex(str(index_dir))
    assert stale.documents == {} and stale.chunks == []
    # ...and the next ingest rebuilds it from the corpus rather than serving nothing
    assert stale.index_directory(str(corpus))["total_documents"] == 3


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


def test_index_records_path_independent_content_hash(tmp_path, corpus):
    copy = tmp_path / "copies"
    copy.mkdir()
    (copy / "vacation-copy.md").write_text((corpus / "vacation.md").read_text())
    (copy / "edited.md").write_text(
        (corpus / "vacation.md").read_text() + "Sabbaticals are unpaid.\n")

    index = LocalSearchIndex(str(tmp_path / "idx"))
    index.index_directory(str(corpus))
    index.index_directory(str(copy))

    hashes = {os.path.basename(p): info["hash"]
              for p, info in index.documents.items()}
    # Same bytes under a different name and directory hash alike; edited
    # content does not.
    assert hashes["vacation.md"] == hashes["vacation-copy.md"]
    assert hashes["vacation.md"] != hashes["edited.md"]
    assert hashes["vacation.md"] != hashes["expenses.txt"]


def test_filename_terms_make_document_discoverable(tmp_path):
    # A document named for its topic whose prose never states that topic: with
    # only the content indexed it scores zero and drops out of the ranking.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "06-ayala-scholarship-ai-data-science.md").write_text(
        "# Ayala Foundation Grant\n\n"
        "Covers full tuition for two years of graduate study, plus a monthly "
        "stipend for living costs.\n"
    )
    (docs / "parking.md").write_text("Parking permits are issued quarterly.\n")

    index = LocalSearchIndex(str(tmp_path / "idx"))
    index.index_directory(str(docs))

    results = index.search("scholarship ai", method="bm25")
    assert results
    assert (os.path.basename(results[0]["file"])
            == "06-ayala-scholarship-ai-data-science.md")


def test_folder_terms_make_document_discoverable(tmp_path):
    # The topic is named by the folder, not by the filename or the prose.
    docs = tmp_path / "docs"
    (docs / "AI-Program").mkdir(parents=True)
    (docs / "AI-Program" / "06-ayala-grant.md").write_text(
        "# Ayala Foundation Grant\n\n"
        "Covers full tuition for two years of graduate study.\n"
    )
    (docs / "parking.md").write_text("Parking permits are issued quarterly.\n")

    index = LocalSearchIndex(str(tmp_path / "idx"))
    index.index_directory(str(docs))

    results = index.search("ai program", method="bm25")
    assert results
    assert os.path.basename(results[0]["file"]) == "06-ayala-grant.md"


def test_plural_query_reaches_singular_filename(tmp_path):
    """The document is named "scholarship"; the question asks about
    "scholarships". Matched literally these are unrelated terms, so nothing in
    the corpus scores at all and the ranking comes back empty."""
    docs = tmp_path / "docs"
    (docs / "AI-Program").mkdir(parents=True)
    (docs / "AI-Program" / "06-ayala-scholarship-ai-data-science.md").write_text(
        "# Ayala Graduate Grant\n\nCovers tuition and a monthly stipend.\n")
    (docs / "parking.md").write_text("Parking permits are issued quarterly.\n")

    index = LocalSearchIndex(str(tmp_path / "idx"))
    index.index_directory(str(docs))

    results = index.search("scholarships", method="bm25")
    assert results
    assert (os.path.basename(results[0]["file"])
            == "06-ayala-scholarship-ai-data-science.md")


def test_document_outranks_a_catalog_that_merely_lists_it(tmp_path):
    """A corpus README naming every file must not outrank the file itself.

    The catalog mentions every topic in the corpus and is long enough to
    mention each more than once, so with the filename folded into the passage
    text it beat the documents it catalogues on their own subjects.
    """
    docs = tmp_path / "docs"
    (docs / "AI-Program").mkdir(parents=True)
    (docs / "AI-Program" / "06-ayala-scholarship-ai-data-science.md").write_text(
        "# Ayala Graduate Scholarship Program\n\n"
        "Covers tuition, stipends, and research support for graduate "
        "students in artificial intelligence and data science.\n")
    (docs / "README.md").write_text(
        "# Document Catalog\n\n"
        "### AI Program\n"
        "- Files: `AI-Program/06-ayala-scholarship-ai-data-science.md`, "
        "`AI-Program/06-mengg-ai-program-handbook-v1.pdf`\n"
        "- Note: the AI Program scholarship covers AI and data science "
        "students; other scholarships are listed under prefix 17.\n\n"
        "### Scholarships\n"
        "- Files: `02-sikap-annex-e-scholarship-privileges.pdf`\n"
        "- Note: SIKAP is a CHED scholarship; the AI Program scholarship "
        "is separate.\n")

    index = LocalSearchIndex(str(tmp_path / "idx"))
    index.index_directory(str(docs))

    results = index.search("UPD AI Program scholarships", method="bm25")
    assert results
    assert (os.path.basename(results[0]["file"])
            == "06-ayala-scholarship-ai-data-science.md")


def test_one_document_cannot_fill_the_whole_result_page(tmp_path):
    """A long catalogue matched the query in chunk after chunk and took every
    visible slot, so the document holding the answer sat below the fold."""
    docs = tmp_path / "docs"
    (docs / "AI-Program").mkdir(parents=True)
    (docs / "AI-Program" / "06-ayala-scholarship-ai-data-science.md").write_text(
        "# Ayala Graduate Scholarship Program\n\n"
        "Covers tuition and stipends for graduate students in artificial "
        "intelligence and data science at UP Diliman.\n")
    for i, name in enumerate(("policy.md", "stipends.md", "admissions.md",
                              "handbook.md")):
        (docs / name).write_text(
            f"# {name}\n\nAI Program scholarship note {i}: scholarships and "
            "stipends are released each semester at UPD.\n")
    # A catalogue long enough to be chunked many times over, every chunk of it
    # naming the query terms.
    (docs / "README.md").write_text(
        "# Document Catalog\n\n" + "".join(
            f"- `AI-Program/{i:02d}-scholarship.md` — UPD AI Program "
            f"scholarship entry {i}, listing scholarships for the AI Program "
            f"at UPD.\n" * 6
            for i in range(30)))

    index = LocalSearchIndex(str(tmp_path / "idx"))
    index.index_directory(str(docs))

    results = index.search("UPD AI Program scholarships", top_k=5, method="bm25")
    readme_hits = [r for r in results if os.path.basename(r["file"]) == "README.md"]
    assert len(readme_hits) <= MAX_CHUNKS_PER_FILE
    assert any(os.path.basename(r["file"])
               == "06-ayala-scholarship-ai-data-science.md" for r in results)


def test_named_document_is_shown_through_its_opening(tmp_path):
    """A document matched by name must be shown by the passage that identifies
    it, not only by whichever passages repeat the query words.

    The excerpt is picked by term frequency while the cap allows two per file,
    so a document could come back entirely through passages that never say what
    it is — here a list of degree programs and a page of contact links, with the
    title and the coverage it states deferred as a third chunk. Read through
    those two excerpts the document looks like unrelated boilerplate, and the
    one in-house document on the subject was dropped from the answer.
    """
    docs = tmp_path / "docs"
    (docs / "AI-Program").mkdir(parents=True)
    (docs / "AI-Program" / "06-ayala-scholarship-ai-data-science.md").write_text(
        # Opening: says what the document is, and matches the query weakly.
        "# Ayala Graduate Scholarship Program\n\n"
        "Covers tuition, a stipend, and research support.\n\n"
        + "filler " * 250 + "\n\n"
        # Two passages that repeat the query words far more often.
        "## Covered programs\n\nThe AI Program at UPD: the MEngg AI program, "
        "the PhD AI program, and the PhD Data Science program of the AI "
        "Program at UPD.\n\n" + "filler " * 150 + "\n\n"
        "## Contact\n\nAI Program office at UPD, the AI Program mailing list, "
        "and the AI Program page.\n")

    index = LocalSearchIndex(str(tmp_path / "idx"))
    index.index_directory(str(docs))

    results = index.search("UPD AI Program scholarships", top_k=2, method="bm25")
    assert results
    assert "Ayala Graduate Scholarship Program" in results[0]["text"]
    assert "Covers tuition, a stipend, and research support." in results[0]["text"]


def test_opening_is_not_promoted_for_a_document_matched_only_by_text(tmp_path):
    # Promotion follows the corpus's own vote on the document as a whole: with
    # nothing in the filename or folder matching, the passages keep the page.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "minutes-2024.md").write_text(
        "# Minutes\n\n" + "filler " * 250 + "\n\n"
        "## Item 4\n\nThe stipend was raised.\n")

    index = LocalSearchIndex(str(tmp_path / "idx"))
    index.index_directory(str(docs))

    results = index.search("stipend", top_k=1, method="bm25")
    assert results
    assert "The stipend was raised." in results[0]["text"]


def test_result_page_is_filled_when_few_documents_match(tmp_path):
    # The cap defers a document's extra chunks, it does not drop them: a corpus
    # where one file holds every match still returns a full page.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "handbook.md").write_text(
        "".join(f"## Section {i}\n\nScholarship terms, clause {i}. " + "filler " * 300
                + "\n\n" for i in range(6)))

    index = LocalSearchIndex(str(tmp_path / "idx"))
    index.index_directory(str(docs))

    results = index.search("scholarship", top_k=4, method="bm25")
    assert len(results) == 4
    assert all(os.path.basename(r["file"]) == "handbook.md" for r in results)
    assert [r["rank"] for r in results] == [1, 2, 3, 4]


def test_documents_yielding_no_text_are_reported(tmp_path, corpus):
    # A scanned PDF parses without error and yields no text: it is counted as
    # a document but can never be reached by a query, so it must be reported
    # rather than inflating total_documents in silence.
    (corpus / "scanned.txt").write_text("   \n")

    index = LocalSearchIndex(str(tmp_path / "idx"))
    result = index.index_directory(str(corpus))

    assert result["total_documents"] == 4
    assert [os.path.basename(p) for p in result["no_text_extracted"]] == ["scanned.txt"]
    assert [os.path.basename(p) for p in index.status()["no_text_extracted"]] \
        == ["scanned.txt"]


def test_name_label_backfilled_on_load_of_older_index(tmp_path, corpus):
    index_dir = tmp_path / "idx"
    LocalSearchIndex(str(index_dir)).index_directory(str(corpus))
    persisted = index_dir / "index.json"
    data = json.loads(persisted.read_text())
    # Same index version, written before name_label existed.
    for chunk in data["chunks"]:
        chunk.pop("name_label")
    persisted.write_text(json.dumps(data))

    reloaded = LocalSearchIndex(str(index_dir))
    assert all(c["name_label"] for c in reloaded.chunks)
    # "expenses" appears only in the filename — the prose says "Expense reports".
    results = reloaded.search("expenses", method="bm25")
    assert results and os.path.basename(results[0]["file"]) == "expenses.txt"


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


def test_shared_corpus_indexed_once_across_sessions(search_env, corpus):
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

    # One shared index, stored beside the corpus, none inside the session jails
    assert (corpus / ".local_search" / "index.json").is_file()
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


def test_cross_index_merge_uses_reciprocal_rank_fusion(search_env):
    session_a = _make_session(search_env, "session-a")

    private = search_env / "session-a" / "notes"
    private.mkdir()
    (private / "travel.md").write_text(
        "# Travel\n\nBusiness travel meals are capped at 50 USD per day.\n"
    )
    local_mod._index_documents_impl(path=str(private), data_path=session_a)

    found = json.loads(local_mod._local_search_impl(
        query="business travel meals per day", data_path=session_a))
    results = found["results"]
    assert len(results) > 1

    # Scores are rank-derived, not raw BM25, so they are comparable across
    # the shared and session indexes.
    assert results[0]["score"] == pytest.approx(1.0 / (RRF_K + 1), abs=1e-6)
    assert [r["score"] for r in results] == sorted(
        (r["score"] for r in results), reverse=True)
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))

    # Both indexes are represented: each contributes its own rank-1 hit, and
    # equal ranks fuse to equal scores.
    files = [os.path.basename(r["file"]) for r in results]
    assert "travel.md" in files and "expenses.txt" in files
    assert results[0]["score"] == results[1]["score"]

    # One entry per passage — the same chunk seen by both indexes is fused,
    # not concatenated and not repeated
    keys = [(r["file"], r["location"], r["text"]) for r in results]
    assert len(keys) == len(set(keys))


def test_cross_index_merge_boosts_chunks_ranked_by_both_indexes(
        search_env, monkeypatch):
    session_a = _make_session(search_env, "session-a")
    local_mod._index_documents_impl(data_path=session_a)

    # Hand the merge the same index twice: every chunk is then ranked by both
    # "indexes" at the same rank and must accumulate both RRF contributions.
    index = local_mod._get_index(local_mod._shared_index_dir())
    monkeypatch.setattr(local_mod, "_session_indexes", lambda base: (index, index))

    found = json.loads(local_mod._local_search_impl(
        query="vacation days", data_path=session_a))
    top = found["results"][0]
    assert top["file"].endswith("vacation.md")
    assert top["score"] == pytest.approx(2.0 / (RRF_K + 1), abs=1e-6)
    keys = [(r["file"], r["location"], r["text"]) for r in found["results"]]
    assert len(keys) == len(set(keys))


def test_cross_index_merge_keeps_chunks_of_one_file_apart(search_env, corpus):
    # A long catalog whose every chunk mentions the query terms in passing,
    # next to one short document that answers the query outright.
    sections = [
        f"## Entry {i}\n\nThe UPD directory lists the AI program and the "
        f"scholarship desk for unit {i}. " + "Archived record. " * 90
        for i in range(8)
    ]
    (corpus / "catalog.md").write_text("# Directory\n\n" + "\n\n".join(sections))
    (corpus / "ayala.md").write_text(
        "# Ayala Grant\n\nThe Ayala Foundation AI scholarship program at UPD "
        "funds ten graduate students each year.\n"
    )

    session_a = _make_session(search_env, "session-a")
    found = json.loads(local_mod._local_search_impl(
        query="UPD AI scholarship program", top_k=5, data_path=session_a))
    results = found["results"]

    # The catalog's chunks stay separate entries, each carrying only its own
    # rank contribution, so they cannot sum their way past the short document.
    assert results[0]["file"].endswith("ayala.md")
    assert all(r["score"] <= 2.0 / (RRF_K + 1) + 1e-9 for r in results)

    catalog = [r for r in results if r["file"].endswith("catalog.md")]
    assert len(catalog) > 1
    assert len({r["text"] for r in catalog}) == len(catalog)


def test_duplicate_document_across_indexes_returned_once(search_env, corpus):
    session_a = _make_session(search_env, "session-a")

    # The session keeps its own copy of a shared corpus file, under a
    # different name, so only the content marks the two as the same document.
    private = search_env / "session-a" / "notes"
    private.mkdir()
    (private / "vacation-copy.md").write_text((corpus / "vacation.md").read_text())
    local_mod._index_documents_impl(path=str(private), data_path=session_a)

    found = json.loads(local_mod._local_search_impl(
        query="vacation days roll over", data_path=session_a))
    results = found["results"]

    # The copy is not a second result: it merges into the shared corpus entry,
    # which keeps the canonical path and both indexes' RRF contributions.
    names = [os.path.basename(r["file"]) for r in results]
    assert names.count("vacation.md") == 1
    assert "vacation-copy.md" not in names
    assert results[0]["file"].endswith("vacation.md")
    assert results[0]["score"] == pytest.approx(2.0 / (RRF_K + 1), abs=1e-6)

    # Distinct documents still come back separately.
    assert "expenses.txt" in names or "onboarding.md" in names


def test_duplicate_document_dedup_survives_an_index_without_hashes(
        search_env, corpus):
    session_a = _make_session(search_env, "session-a")

    private = search_env / "session-a" / "notes"
    private.mkdir()
    (private / "vacation-copy.md").write_text((corpus / "vacation.md").read_text())
    local_mod._index_documents_impl(path=str(private), data_path=session_a)
    local_mod._index_documents_impl(data_path=session_a)  # shared corpus

    # An index persisted before hashes were recorded has none to key on, and
    # the filenames differ, so the hash has to be recomputed at merge time.
    session_index, _ = local_mod._session_indexes(local_mod._session_base(session_a))
    for info in session_index.documents.values():
        info.pop("hash", None)

    found = json.loads(local_mod._local_search_impl(
        query="vacation days roll over", data_path=session_a))
    names = [os.path.basename(r["file"]) for r in found["results"]]
    assert names.count("vacation.md") == 1
    assert "vacation-copy.md" not in names

    # The recomputed hash is memoized on the index entry, not re-read per query
    assert all(info.get("hash") for info in session_index.documents.values())


def test_document_key_falls_back_to_filename_for_unreadable_file(search_env):
    index = local_mod._get_index(local_mod._shared_index_dir())
    missing = os.path.join(str(search_env), "gone", "Vacation.MD")
    index.documents[missing] = {"size": 1}

    assert local_mod._document_key(index, missing) == "vacation.md"
    assert "hash" not in index.documents[missing]


# -- where the shared index is stored -------------------------------------------

def test_shared_index_lives_beside_the_corpus_not_under_data_path(search_env, corpus):
    """The index belongs with the documents it describes: a corpus mounted into
    another deployment must arrive already ingested, rather than leaving its
    index behind in a DATA_PATH that is per-deployment scratch."""
    local_mod.rebuild_indexes(background=False)

    assert (corpus / ".local_search" / "index.json").is_file()
    assert not (search_env / "local_search" / "shared").exists()


def test_shared_index_is_not_itself_indexed(search_env, corpus):
    """The index sits inside the directory being walked, so an ingest that did
    not skip it would index its own output and grow on every pass."""
    local_mod.rebuild_indexes(background=False)
    local_mod._INDEXES.clear()
    local_mod.rebuild_indexes(background=False)

    index = local_mod._get_index(local_mod._shared_index_dir())
    assert len(index.documents) == 3
    assert all(local_mod.SHARED_INDEX_DIRNAME not in p for p in index.documents)


def test_missing_shared_index_is_rebuilt_from_the_corpus(search_env, corpus):
    """A corpus with no index beside it — new, or last indexed by a version
    that stored the index elsewhere — is simply ingested. Nothing is adopted
    from another directory: the corpus on disk defines the index."""
    stale_elsewhere = search_env / "local_search" / "shared"
    LocalSearchIndex(str(stale_elsewhere)).index_directory(str(corpus))
    os.unlink(corpus / "onboarding.md")
    assert not (corpus / ".local_search" / "index.json").exists()

    result = json.loads(local_mod._local_search_impl(
        query="vacation days", data_path=str(search_env / "session-a")))

    assert (corpus / ".local_search" / "index.json").is_file()
    built = json.loads((corpus / ".local_search" / "index.json").read_text())
    # Built from the corpus as it is now, not inherited from the stale copy
    assert len(built["documents"]) == 2
    assert not any(p.endswith("onboarding.md") for p in built["documents"])
    assert result["results"][0]["file"].endswith("vacation.md")


def test_read_only_corpus_falls_back_to_data_path(search_env, corpus, monkeypatch):
    """A corpus mounted read-only is the normal container deployment. Search
    must still work — the index just goes under DATA_PATH instead."""
    def _refuse(path, *args, **kwargs):
        if str(path).startswith(str(corpus)):
            raise PermissionError(f"read-only corpus: {path}")
        return os.makedirs(path, *args, **kwargs)

    monkeypatch.setattr(local_mod.os, "makedirs", _refuse)

    assert local_mod._shared_index_dir() == str(search_env / "local_search" / "shared")


def test_read_only_corpus_that_already_has_an_index_dir_falls_back_too(
        search_env, corpus):
    """A corpus remounted read-only still carries the .local_search left by an
    earlier read-write run, and makedirs(exist_ok=True) succeeds for it. The
    index would then be unwritable at the very moment it is saved."""
    stale = corpus / ".local_search"
    stale.mkdir()
    stale.chmod(0o500)
    try:
        assert local_mod._shared_index_dir() == str(
            search_env / "local_search" / "shared")

        # ...and searching still works against the fallback location
        result = json.loads(local_mod._local_search_impl(
            query="vacation days", data_path=str(search_env / "session-a")))
        assert result["results"][0]["file"].endswith("vacation.md")
        assert (search_env / "local_search" / "shared" / "index.json").is_file()
    finally:
        stale.chmod(0o700)


# -- what a session walk covers -------------------------------------------------

@pytest.fixture
def nested_corpus(tmp_path, corpus, monkeypatch):
    """DOCUMENTS_PATH sitting inside DATA_PATH — the default layout, and the
    one the terminal UI hits when its jail root is the data directory itself."""
    data = tmp_path / "data"
    data.mkdir()
    nested = data / "docs"
    corpus.rename(nested)
    _reset_search_globals(monkeypatch, data, nested)
    (data / "session-notes.md").write_text(
        "# Notes\n\nDraft notes about the quarterly planning offsite.\n")
    return data, nested


def test_session_walk_leaves_the_shared_corpus_to_the_shared_index(nested_corpus):
    """The corpus lives inside the data directory, so a session walk rooted at
    the data directory would otherwise take a second copy of every shared
    document — paying to parse it per session and letting one file score from
    both indexes in the fused ranking."""
    data, nested = nested_corpus

    local_mod._local_search_impl(query="vacation days", data_path=str(data))

    session_index = local_mod._get_index(local_mod._index_dir(str(data)))
    shared_index = local_mod._get_index(local_mod._shared_index_dir())
    assert [os.path.basename(p) for p in session_index.documents] == ["session-notes.md"]
    assert len(shared_index.documents) == 3
    assert not any(str(nested) in p for p in session_index.documents)


def test_both_scopes_stay_searchable(nested_corpus):
    """Excluding the corpus from the session walk must not put it out of reach:
    the shared index still answers for it, the session index for its own files."""
    data, _ = nested_corpus

    shared_hit = json.loads(local_mod._local_search_impl(
        query="vacation days", data_path=str(data)))
    session_hit = json.loads(local_mod._local_search_impl(
        query="quarterly planning offsite", data_path=str(data)))

    assert shared_hit["results"][0]["file"].endswith("vacation.md")
    assert session_hit["results"][0]["file"].endswith("session-notes.md")
    # One entry per document, not one per index
    assert shared_hit["total_documents"] == 4


def test_documents_already_taken_by_the_session_index_are_dropped(nested_corpus):
    """An index written before the exclusion existed holds the shared corpus
    too. Those entries are dropped on the next pass rather than lingering as
    duplicates that outrank single-index hits."""
    data, nested = nested_corpus
    session_index = local_mod._get_index(local_mod._index_dir(str(data)))
    session_index.index_directory(str(data))  # the pre-exclusion pass
    assert len(session_index.documents) == 4

    local_mod._local_search_impl(query="vacation days", data_path=str(data))

    assert [os.path.basename(p) for p in session_index.documents] == ["session-notes.md"]


# -- startup rebuild ------------------------------------------------------------

def test_rebuild_reflects_corpus_changed_while_server_was_down(search_env, corpus):
    local_mod._index_documents_impl()  # index built by the "previous run"
    assert (corpus / ".local_search" / "index.json").is_file()

    os.unlink(corpus / "onboarding.md")
    (corpus / "parking.md").write_text(
        "# Parking\n\nParking permits are issued by facilities on the second floor.\n"
    )
    local_mod._INDEXES.clear()  # a restart starts with no in-memory index

    local_mod.rebuild_indexes(background=False)

    status = json.loads(local_mod._index_documents_impl(status_only=True))
    assert status["total_documents"] == 3
    found = json.loads(local_mod._local_search_impl(query="parking permits"))
    assert found["results"][0]["file"].endswith("parking.md")
    dropped = json.loads(local_mod._local_search_impl(query="security training laptop"))
    assert all(not r["file"].endswith("onboarding.md") for r in dropped["results"])


def test_rebuild_keeps_indexes_it_does_not_own(search_env, corpus):
    """One data directory is commonly shared by several onit services, each
    running its own copy of this server. A start must not delete the indexes
    of the sessions its siblings are still serving."""
    sibling = search_env / "other-session" / "local_search"
    LocalSearchIndex(str(sibling)).index_directory(str(corpus))
    fallback = search_env / "local_search"
    LocalSearchIndex(str(fallback)).index_directory(str(corpus))

    local_mod.rebuild_indexes(background=False)

    assert (sibling / "index.json").is_file()
    assert (fallback / "index.json").is_file()
    assert (corpus / ".local_search" / "index.json").is_file()


def test_rebuild_leaves_a_shared_index_a_sibling_just_wrote(search_env, corpus):
    """The shared index a sibling container built is refreshed in place, so it
    is never missing from disk while this server rebuilds its own copy."""
    local_mod._index_documents_impl()  # the sibling's build
    shared = corpus / ".local_search" / "index.json"
    before = json.loads(shared.read_text())
    local_mod._INDEXES.clear()

    local_mod.rebuild_indexes(background=False)

    assert shared.is_file()
    assert json.loads(shared.read_text())["documents"].keys() == before["documents"].keys()


def test_rebuild_reuses_embeddings_of_unchanged_documents(search_env):
    """An unchanged corpus is not re-embedded at every start — four containers
    restarting must not each re-embed a corpus that nobody edited."""
    local_mod.rebuild_indexes(background=False)
    index = local_mod._get_index(local_mod._shared_index_dir())
    for chunk in index.chunks:
        chunk["embedding"] = [1.0, 0.0]
    index.embedding_model = "test-embeddings"
    index.save()
    local_mod._INDEXES.clear()  # a restart starts with no in-memory index

    local_mod.rebuild_indexes(background=False)

    index = local_mod._get_index(local_mod._shared_index_dir())
    assert index.chunks and all("embedding" in c for c in index.chunks)
    assert index.embedding_model == "test-embeddings"


def test_rebuild_without_documents_path_is_a_noop(search_env, corpus, monkeypatch):
    monkeypatch.setattr(local_mod, "DOCUMENTS_PATH", None)
    session = search_env / "old-session" / "local_search"
    LocalSearchIndex(str(session)).index_directory(str(corpus))

    assert local_mod.rebuild_indexes(background=False) is None
    assert (session / "index.json").is_file()


def test_rebuild_in_background_completes(search_env):
    thread = local_mod.rebuild_indexes()
    thread.join(timeout=30)
    assert not thread.is_alive()
    status = json.loads(local_mod._index_documents_impl(status_only=True))
    assert status["shared_index"]["total_documents"] == 3


def test_status_only_combines_shared_and_session(search_env):
    session_a = _make_session(search_env, "session-a")
    local_mod._index_documents_impl(data_path=session_a)

    status = json.loads(local_mod._index_documents_impl(
        status_only=True, data_path=session_a))
    assert status["status"] == "success"
    assert status["total_documents"] == 3
    assert status["shared_index"]["total_documents"] == 3
    assert status["session_index"]["total_documents"] == 0


# -- searches keep their corpora current ---------------------------------------

def test_search_auto_indexes_session_file_alongside_shared_corpus(search_env):
    """A file dropped into the session's own folder is searchable without an
    explicit index_documents call, even though a shared corpus also exists."""
    session = _make_session(search_env, "session-a")
    (search_env / "session-a" / "uploaded.md").write_text(
        "# Project Falcon\n\nProject Falcon ships in November 2026.\n"
    )

    found = json.loads(local_mod._local_search_impl(
        query="Project Falcon November", data_path=session))
    assert found["status"] == "success"
    assert found["results"][0]["file"].endswith("uploaded.md")
    assert os.path.isfile(
        os.path.join(session, "local_search", "index.json"))


def test_search_picks_up_shared_corpus_file_added_while_server_runs(
        search_env, corpus):
    """The shared index refreshes on search, so a document added to the corpus
    after the startup rebuild does not wait for a restart to become visible."""
    session = _make_session(search_env, "session-a")
    local_mod.rebuild_indexes(background=False)

    (corpus / "falcon.md").write_text(
        "# Project Falcon\n\nProject Falcon ships in November 2026.\n"
    )

    found = json.loads(local_mod._local_search_impl(
        query="Project Falcon November", data_path=session))
    assert found["results"][0]["file"].endswith("falcon.md")
    assert found["total_documents"] == 4


def test_search_drops_shared_corpus_file_deleted_while_server_runs(
        search_env, corpus):
    session = _make_session(search_env, "session-a")
    local_mod.rebuild_indexes(background=False)

    (corpus / "vacation.md").unlink()

    found = json.loads(local_mod._local_search_impl(
        query="vacation policy accrue days", data_path=session))
    assert all(not r["file"].endswith("vacation.md") for r in found["results"])
    assert found["total_documents"] == 2


def test_search_reflects_edited_shared_document(search_env, corpus):
    session = _make_session(search_env, "session-a")
    local_mod.rebuild_indexes(background=False)

    (corpus / "vacation.md").write_text(
        "# Vacation Policy\n\nEmployees accrue 25 days of sabbatical leave.\n"
    )

    found = json.loads(local_mod._local_search_impl(
        query="sabbatical leave", data_path=session))
    assert found["results"][0]["file"].endswith("vacation.md")
    assert "sabbatical" in found["results"][0]["text"]


def test_session_refresh_does_not_leak_into_other_sessions(search_env):
    """Refreshing on search must not widen a session's jail: session A's file
    is still invisible to session B."""
    session_a = _make_session(search_env, "session-a")
    session_b = _make_session(search_env, "session-b")
    (search_env / "session-a" / "secret.md").write_text(
        "# Project Zebra\n\nProject Zebra launches in October with a 2M budget.\n"
    )

    seen = json.loads(local_mod._local_search_impl(
        query="Project Zebra October budget", data_path=session_a))
    assert seen["results"][0]["file"].endswith("secret.md")

    hidden = json.loads(local_mod._local_search_impl(
        query="Project Zebra October budget", data_path=session_b))
    assert all(not r["file"].endswith("secret.md") for r in hidden["results"])
    assert hidden["total_documents"] == 3


# -- documents section --------------------------------------------------------

def test_results_carry_the_documents_behind_them(search_env):
    """The excerpts are chosen for repeating the query's words; the opening is
    what says which document this is. Returning it here is what saves a round
    trip per document."""
    session = _make_session(search_env, "session-a")
    local_mod.rebuild_indexes(background=False)

    found = json.loads(local_mod._local_search_impl(
        query="vacation days accrue", data_path=session))

    assert found["documents"], "every result page describes its documents"
    vacation = next(d for d in found["documents"]
                    if d["file"].endswith("vacation.md"))
    assert "Vacation Policy" in vacation["opening"]
    assert vacation["num_chunks"] >= 1
    assert vacation["best_rank"] >= 1


def test_one_entry_per_document_however_many_chunks_matched(search_env):
    session = _make_session(search_env, "session-a")
    local_mod.rebuild_indexes(background=False)

    found = json.loads(local_mod._local_search_impl(
        query="days", top_k=10, data_path=session))

    files = [d["file"] for d in found["documents"]]
    assert len(files) == len(set(files))
    assert len(found["documents"]) <= len(found["results"])


def test_ranked_passages_are_serialized_before_the_document_summaries(search_env):
    """The head of the response is what survives being trimmed as an older tool
    result, so it has to be the ranking.

    The summaries grow with top_k; the top-ranked passages do not. With the
    summaries first, a passage's offset moves with the page size and outruns any
    trim budget — the model is left holding titles and no quotes.
    """
    session = _make_session(search_env, "session-a")
    local_mod.rebuild_indexes(background=False)

    raw = local_mod._local_search_impl(query="vacation days accrue",
                                       top_k=10, data_path=session)
    assert raw.index('"results"') < raw.index('"documents"')

    # The passages start at the same offset however big the page is.
    heads = {
        local_mod._local_search_impl(query="vacation days accrue", top_k=k,
                                     data_path=session).index('"results"')
        for k in (1, 5, 20)
    }
    assert len(heads) == 1


def test_document_openings_are_bounded(search_env, corpus):
    """A result page has a size budget: the caller truncates the whole thing."""
    long_doc = corpus / "handbook.md"
    long_doc.write_text("# Handbook\n\n" + ("policy detail " * 2000))
    session = _make_session(search_env, "session-a")
    local_mod.rebuild_indexes(background=False)

    found = json.loads(local_mod._local_search_impl(
        query="handbook policy detail", data_path=session))
    handbook = next(d for d in found["documents"]
                    if d["file"].endswith("handbook.md"))
    assert len(handbook["opening"]) <= DOCUMENT_OPENING_CHARS + 2
    assert len(found["documents"]) <= MAX_DOCUMENT_SUMMARIES


def test_opening_is_the_start_of_the_document_not_the_best_chunk(search_env, corpus):
    """The passage repeating the query's words can be anywhere; the title is
    at the top, and the title is what identifies the document."""
    (corpus / "benefits.md").write_text(
        "# Benefits Handbook\n\nThis handbook describes employee benefits.\n\n"
        + ("filler paragraph. " * 300)
        + "\n\nDental coverage includes two cleanings per year."
    )
    session = _make_session(search_env, "session-a")
    local_mod.rebuild_indexes(background=False)

    found = json.loads(local_mod._local_search_impl(
        query="dental cleanings coverage", data_path=session))
    benefits = next(d for d in found["documents"]
                    if d["file"].endswith("benefits.md"))
    assert benefits["opening"].startswith("# Benefits Handbook")


# -- refresh window -----------------------------------------------------------

def test_a_corpus_is_not_rewalked_on_every_search(search_env, corpus, monkeypatch):
    """A research answer searches several times in a row and the corpus does not
    change in between; the walk is a stat per file, which is not free on a large
    corpus."""
    monkeypatch.setattr(local_mod, "REFRESH_MIN_INTERVAL", 300.0)
    walks = []
    original = local_mod.LocalSearchIndex.index_directory

    def counted(self, directory, **kwargs):
        walks.append(directory)
        return original(self, directory, **kwargs)

    monkeypatch.setattr(local_mod.LocalSearchIndex, "index_directory", counted)
    for _ in range(4):
        local_mod._local_search_impl(query="vacation policy")
    # A search reaches two corpora — the shared one and the session's own — so
    # four searches walked eight times before the window existed.
    assert sorted(walks) == sorted(set(walks))
    assert len(walks) == 2


def test_the_window_expires(search_env, corpus, monkeypatch):
    monkeypatch.setattr(local_mod, "REFRESH_MIN_INTERVAL", 300.0)
    walks = []
    original = local_mod.LocalSearchIndex.index_directory

    def counted(self, directory, **kwargs):
        walks.append(directory)
        return original(self, directory, **kwargs)

    monkeypatch.setattr(local_mod.LocalSearchIndex, "index_directory", counted)
    local_mod._local_search_impl(query="vacation policy")
    walked_once = len(walks)
    local_mod._local_search_impl(query="vacation policy")
    assert len(walks) == walked_once
    # Age the stamps past the window rather than sleeping through it.
    local_mod._LAST_REFRESH.clear()
    local_mod._local_search_impl(query="vacation policy")
    assert len(walks) == walked_once * 2


def test_an_explicit_path_is_indexed_immediately(search_env, corpus, monkeypatch):
    """Passing `path` is a request to re-index, so the window must not defer it."""
    monkeypatch.setattr(local_mod, "REFRESH_MIN_INTERVAL", 300.0)
    local_mod._local_search_impl(query="vacation policy")

    (corpus / "sabbatical.md").write_text(
        "# Sabbatical Policy\n\nStaff may take a sabbatical after seven years.\n"
    )
    result = json.loads(local_mod._local_search_impl(query="sabbatical",
                                                    path=str(corpus)))
    assert result["status"] == "success"
    assert any("sabbatical" in r["file"].lower() for r in result["results"])


def test_a_new_file_is_still_found_once_the_window_lapses(search_env, corpus):
    """The window delays noticing a mid-answer addition; it does not hide it."""
    local_mod._local_search_impl(query="vacation policy")
    (corpus / "sabbatical.md").write_text(
        "# Sabbatical Policy\n\nStaff may take a sabbatical after seven years.\n"
    )
    result = json.loads(local_mod._local_search_impl(query="sabbatical"))
    assert any("sabbatical" in r["file"].lower() for r in result["results"])
