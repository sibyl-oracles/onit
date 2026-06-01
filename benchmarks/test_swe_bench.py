"""Offline tests for the SWE-bench runner — no Docker, no network, no git."""

from __future__ import annotations

from benchmarks.swe_bench_runner import DATASETS, _strip_test_hunks

_DIFF = """\
diff --git a/src/lib.py b/src/lib.py
index 111..222 100644
--- a/src/lib.py
+++ b/src/lib.py
@@ -1,2 +1,2 @@
-def f(): return 1
+def f(): return 2
diff --git a/tests/test_lib.py b/tests/test_lib.py
index 333..444 100644
--- a/tests/test_lib.py
+++ b/tests/test_lib.py
@@ -1,2 +1,2 @@
-assert f() == 1
+assert f() == 2
"""


def test_strip_test_hunks_keeps_source_drops_tests():
    out = _strip_test_hunks(_DIFF)
    assert "src/lib.py" in out
    assert "return 2" in out
    assert "tests/test_lib.py" not in out
    assert "assert f()" not in out


def test_strip_test_hunks_empty():
    assert _strip_test_hunks("") == ""


def test_dataset_ids():
    assert set(DATASETS) == {"lite", "verified", "full"}
    assert DATASETS["lite"] == "princeton-nlp/SWE-bench_Lite"
