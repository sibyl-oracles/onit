"""The run directory the CLI prints must outlive the run.

A replay whose scratch directory is a temporary directory is deleted in
a ``finally`` when the CLI owns the root -- the cleanup is correct, but
it deletes the very path the command just printed and the operator just
quoted back.  So a run the CLI owns is copied into the checkout before
the sweep; a caller who named ``--eval-out`` owns the root and is
copied nowhere.
"""

import json
import os

from onit import cli


class _Args:
    """The eval-run arguments, shaped like argparse would produce."""

    def __init__(self, root):
        self.eval_run = True
        self.eval_split = "dev"
        self.eval_limit = None
        self.eval_out = root
        self.wiki_ab = False
        self.ab = False
        self.dry_run = False


def _seed_root(root, comparison):
    os.makedirs(os.path.join(root, "arms", "off"), exist_ok=True)
    with open(os.path.join(root, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f)
    with open(os.path.join(root, "arms", "off", "results.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": 1}], f)


def _checkout(tmp_path):
    """A hermetic stand-in for the repo's tmp/ directory."""
    d = tmp_path / "checkout"
    d.mkdir(exist_ok=True)
    return str(d)


def test_explicit_eval_out_is_not_copied(tmp_path, capsys):
    root = str(tmp_path / "run")
    _seed_root(root, {"delta": 0.5})

    # --eval-out was given, so the operator owns the root: no copy.
    cli._keep_eval_run_copy(_Args(root), str(root), owned=False, checkout=_checkout(tmp_path))
    assert not os.path.exists(os.path.join(_checkout(tmp_path), "tmp", "eval-runs"))
    assert capsys.readouterr().out == ""


def test_temp_run_is_copied_before_cleanup(tmp_path, capsys):
    root = str(tmp_path / "onit-evalrun-abcd")
    _seed_root(root, {"delta": 0.5})
    out = _checkout(tmp_path)

    cli._keep_eval_run_copy(_Args(""), str(root), owned=True, checkout=out)

    kept = os.path.join(out, "tmp", "eval-runs", os.path.basename(root))
    assert os.path.exists(os.path.join(kept, "comparison.json"))
    with open(os.path.join(kept, "comparison.json"), encoding="utf-8") as f:
        assert json.load(f)["delta"] == 0.5
    assert os.path.exists(os.path.join(kept, "arms", "off", "results.json"))
    printed = capsys.readouterr().out
    assert "Kept a copy" in printed and os.path.basename(root) in printed


def test_a_root_that_was_never_created_is_skipped(tmp_path, capsys):
    # An empty split or an early failure leaves no root behind.
    cli._keep_eval_run_copy(
        _Args(""), str(tmp_path / "never-made"), owned=True, checkout=_checkout(tmp_path)
    )
    assert not os.path.exists(os.path.join(_checkout(tmp_path), "tmp", "eval-runs"))
    assert capsys.readouterr().out == ""


def test_a_second_run_same_name_gets_a_distinct_copy(tmp_path):
    out = _checkout(tmp_path)
    root1 = str(tmp_path / "onit-evalrun-same")
    _seed_root(root1, {"delta": 0.1})
    cli._keep_eval_run_copy(_Args(""), root1, owned=True, checkout=out)

    root2 = str(tmp_path / "onit-evalrun-same")
    os.makedirs(root2, exist_ok=True)
    _seed_root(root2, {"delta": 0.2})
    cli._keep_eval_run_copy(_Args(""), root2, owned=True, checkout=out)

    base = os.path.join(out, "tmp", "eval-runs", "onit-evalrun-same")
    with open(os.path.join(base, "comparison.json"), encoding="utf-8") as f:
        assert json.load(f)["delta"] == 0.1
    suffixed = [
        d
        for d in os.listdir(os.path.join(out, "tmp", "eval-runs"))
        if d.startswith("onit-evalrun-same.")
    ]
    assert len(suffixed) == 1
    with open(
        os.path.join(out, "tmp", "eval-runs", suffixed[0], "comparison.json"), encoding="utf-8"
    ) as f:
        assert json.load(f)["delta"] == 0.2
