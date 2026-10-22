"""
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

Reuse the search toolkit's BM25, without importing the MCP package.

The RFC says to reuse ``local/search/toolkit.py``'s BM25 rather than write a
second one, and that is the right call: a second tokenizer is a second set of
plurals to fold, and the two sides of a match would stop lining up.

Getting at it is the awkward part, and the awkwardness is worth writing down.

``learn`` is imported under two names in this codebase -- ``learn`` (from
``src/`` on the path) and ``src.learn`` (from the repository root) -- and the
MCP package is shadowed by ``src/mcp`` whenever ``src/`` comes first on the
path.  So ``from mcp.servers...`` can resolve to the *PyPI* ``mcp`` SDK, which
has no ``servers`` submodule, and ``from src.mcp...`` only works when the
repository root happens to be on ``sys.path``.

Rather than add a third spelling to the pile, this module loads the toolkit
*by file path*, through :func:`importlib.util.spec_from_file_location`.  The
toolkit is a self-contained pure-Python module with no relative imports and no
fastmcp dependency, so loading it standalone is exact, and it resolves
identically no matter which spelling of ``learn`` got you here.

If the file cannot be found, :func:`tokenize` and :func:`bm25_scores` fall
back to local implementations.  The fallback is not as good -- it does not
fold plurals -- and it exists so that a missing toolkit degrades recall
rather than crashing the run.  Recall is an improvement to a working agent;
it must never be the thing that breaks it.
"""

import importlib.util
import logging
import math
import os
import re

logger = logging.getLogger(__name__)

# The toolkit, relative to this file: src/learn/_search.py ->
# src/mcp/servers/tasks/local/search/toolkit.py
_TOOLKIT_REL = os.path.join(
    "..", "mcp", "servers", "tasks", "local", "search", "toolkit.py")

# Word characters for the fallback tokenizer.  Deliberately the same shape as
# the toolkit's: letters, digits and underscores, so "don't" splits into
# "don" and "t" in both implementations.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

_TOOLKIT = None
_TOOLKIT_TRIED = False


def _load_toolkit():
    """Load the search toolkit by path, once.  Returns the module or None."""
    global _TOOLKIT, _TOOLKIT_TRIED
    if _TOOLKIT_TRIED:
        return _TOOLKIT
    _TOOLKIT_TRIED = True

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, _TOOLKIT_REL))
    if not os.path.isfile(path):
        logger.debug("search toolkit not found at %s", path)
        return None

    try:
        spec = importlib.util.spec_from_file_location(
            "onit_local_search_toolkit", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "BM25") or not hasattr(module, "tokenize"):
            return None
        _TOOLKIT = module
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("search toolkit failed to load: %s", e)
        _TOOLKIT = None
    return _TOOLKIT


def toolkit_available() -> bool:
    """Whether the shared BM25 implementation was found."""
    return _load_toolkit() is not None


# ─── fallback tokenizer ─────────────────────────────────────────────────────

def _fold_plural_fallback(word: str) -> str:
    """The toolkit's plural folding, reimplemented for the fallback path.

    Kept in step with ``toolkit._fold_plural`` deliberately: a query folded
    one way and a document folded another is a match that silently misses,
    and the symptom is "recall returns nothing" rather than an error.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and (word.endswith("ches") or word.endswith("shes")
                          or (word.endswith("es") and word[-3] in "sxz")):
        return word[:-2]
    if (len(word) > 3 and word.endswith("s")
            and not word.endswith("ss")
            and not word.endswith("us")):
        return word[:-1]
    return word


def tokenize(text: str) -> list:
    """Words of ``text``, lowercased, with plurals folded onto singulars.

    Delegates to the toolkit when it is available so that the two sides of a
    match are folded by the same function.
    """
    toolkit = _load_toolkit()
    if toolkit is not None:
        return list(toolkit.tokenize(text or ""))
    return [_fold_plural_fallback(w)
            for w in _TOKEN_RE.findall((text or "").lower())]


# ─── BM25 ───────────────────────────────────────────────────────────────────

def bm25_scores(corpus_tokens: list, query_tokens: list) -> list:
    """Score every document in ``corpus_tokens`` against ``query_tokens``.

    Uses the toolkit's BM25 when available; otherwise the same Okapi formula
    locally, which is a few dozen lines and keeps the promise that recall
    works without the optional search extras installed.
    """
    toolkit = _load_toolkit()
    if toolkit is not None:
        return list(toolkit.BM25(corpus_tokens).scores(query_tokens))
    return _bm25_fallback(corpus_tokens, query_tokens)


def _bm25_fallback(corpus_tokens: list, query_tokens: list,
                   k1: float = 1.5, b: float = 0.75) -> list:
    """Okapi BM25, matching the toolkit's parameters."""
    doc_lens = [len(t) for t in corpus_tokens]
    if not doc_lens:
        return []
    avg_len = sum(doc_lens) / len(doc_lens) or 1.0

    df: dict = {}
    for tokens in corpus_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    n = len(corpus_tokens)
    idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
           for term, freq in df.items()}

    results = []
    for tokens, doc_len in zip(corpus_tokens, doc_lens):
        if not tokens:
            results.append(0.0)
            continue
        tf: dict = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        norm = k1 * (1 - b + b * doc_len / avg_len)
        score = 0.0
        for term in query_tokens:
            freq = tf.get(term)
            if not freq:
                continue
            score += idf.get(term, 0.0) * freq * (k1 + 1) / (freq + norm)
        results.append(score)
    return results
