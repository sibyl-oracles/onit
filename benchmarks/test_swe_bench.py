"""Offline tests for the SWE-bench runner — no Docker, no network, no git."""

from __future__ import annotations

import json

from pathlib import Path

from benchmarks import config as bench_config
from benchmarks.swe_bench_runner import (
    DATASETS,
    _load_existing,
    _strip_test_hunks,
    _write_predictions,
    build_parser,
)

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


def test_default_data_root_is_inside_the_mcp_jail():
    """Workspaces must sit under the jail root the MCP servers enforce.

    Each per-instance workspace is handed to OnIt as ``data_path``, and
    ``_session_base`` rejects anything outside the server-wide DATA_PATH. A
    default rooted elsewhere (it used to be ``$TMPDIR/onit-swebench``) fails
    every file tool call with "data_path must be within the server data
    directory" before the agent reads a line of source.
    """
    default = Path(build_parser().parse_args([]).data_root).resolve()
    root = bench_config.bench_data_root().resolve()
    assert default == root or default.is_relative_to(root)


def test_resume_loads_completed_skips_errored_and_truncated(tmp_path):
    """Only ``ok`` records count as done; errored/truncated ones get retried."""
    preds = tmp_path / "predictions_onit.jsonl"
    rows = [
        {"instance_id": "a", "model_name_or_path": "m", "model_patch": "x", "_onit_status": "ok"},
        {"instance_id": "b", "model_name_or_path": "m", "model_patch": "", "_onit_status": "error"},
    ]
    with preds.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.write('{"instance_id": "c", "model_pat')  # truncated crash line

    loaded = _load_existing(preds)
    done = {i for i, r in loaded.items() if r.get("_onit_status") == "ok"}
    assert set(loaded) == {"a", "b"}  # truncated "c" skipped
    assert done == {"a"}  # errored "b" is re-attempted, not skipped


def test_write_predictions_atomic_and_deduped(tmp_path):
    preds = tmp_path / "predictions_onit.jsonl"
    results = {
        "a": {"instance_id": "a", "model_name_or_path": "m", "model_patch": "x", "_onit_status": "ok"},
    }
    _write_predictions(preds, results)
    # Overwrite the same key: rewrite must dedupe, not append.
    results["a"]["model_patch"] = "y"
    _write_predictions(preds, results)

    lines = preds.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["model_patch"] == "y"
    assert not preds.with_suffix(".jsonl.tmp").exists()  # temp file cleaned up
