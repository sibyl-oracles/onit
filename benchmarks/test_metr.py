"""Offline tests for the METR time-horizon module — no Docker, no network.

Covers the logistic fit, horizon extraction, the bundled human-time label map,
and the per-family label collapse used by the bridge task.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.tasks.metr import (
    _family_human_minutes,
    _fit_logistic,
    horizon_summary,
    load_human_minutes,
    time_horizon_50,
)

from inspect_ai.scorer import CORRECT, INCORRECT, Score


# --------------------------------------------------------------------------- #
# Logistic fit
# --------------------------------------------------------------------------- #

def test_fit_logistic_recovers_known_curve():
    # P(success) = sigmoid(2 * (x - 5)) -> horizon at x0 = 5 (2**5 = 32 min).
    k_true, x0_true = 2.0, 5.0
    xs = [i * 0.5 for i in range(20)]  # 0 .. 9.5
    ys = [1.0 if _p >= 0.5 else 0.0 for _p in
          (1 / (1 + math.exp(-k_true * (x - x0_true))) for x in xs)]
    k, x0 = _fit_logistic(xs, ys)
    assert abs(x0 - x0_true) < 0.3
    assert k > 0


def test_fit_logistic_separates_high_low():
    # Easy tasks (short) all pass, hard tasks (long) all fail -> horizon
    # somewhere between the two clusters; slope negative (success falls with
    # duration).
    xs = [1.0, 1.5, 2.0, 2.5, 8.0, 8.5, 9.0, 9.5]
    ys = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    k, x0 = _fit_logistic(xs, ys)
    assert 2.5 < x0 < 8.0
    assert k < 0


def test_fit_logistic_one_class_is_degenerate():
    k, x0 = _fit_logistic([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    assert k == 0.0 and x0 == 0.0


# --------------------------------------------------------------------------- #
# Horizon extraction
# --------------------------------------------------------------------------- #

def _score(hm: float, ok: bool) -> Score:
    s = Score(value=CORRECT if ok else INCORRECT)
    s.metadata = {"human_minutes": hm}
    return s


def test_time_horizon_50_basic():
    scores = [
        _score(1, True), _score(2, True), _score(4, True),
        _score(32, False), _score(64, False), _score(128, False),
    ]
    th = time_horizon_50(scores)
    assert th is not None
    assert 4 < th < 32


def test_time_horizon_50_undefined_when_one_class():
    scores = [_score(1, True), _score(2, True), _score(4, True)]
    assert time_horizon_50(scores) is None


def test_time_horizon_50_skips_unlabeled():
    labeled = _score(2, True)
    unlabeled = Score(value=CORRECT)  # no human_minutes metadata
    th = time_horizon_50([labeled, unlabeled, _score(64, False)])
    assert th is not None  # unlabeled sample ignored, fit still defined


def test_horizon_summary_shape():
    summary = horizon_summary([(1, True), (2, True), (32, False), (64, False)])
    assert summary["n"] == 4
    assert summary["time_horizon_50"] is not None
    assert summary["fit"] is not None and len(summary["fit"]) == 2


# --------------------------------------------------------------------------- #
# Bundled label map
# --------------------------------------------------------------------------- #

def test_label_map_loads_and_is_sane():
    labels = load_human_minutes()
    assert len(labels) == 228  # METR Time Horizon v1.1 labeled set
    assert all(hm > 0 for hm in labels.values())
    # SWAA short tasks and RE-Bench long tasks both present.
    assert min(labels.values()) < 1.0
    assert max(labels.values()) >= 600.0


def test_label_map_keys_are_family_slash_task():
    labels = load_human_minutes()
    for key in list(labels)[:50]:
        fam, _, task_id = key.rpartition("/")
        assert fam and task_id, key


def test_family_collapse_gives_medians():
    labels = {"password_check/1": 10.0, "password_check/2": 30.0,
              "password_check/3": 20.0, "sadservers/a": 100.0}
    fams = _family_human_minutes(labels)
    assert fams["password_check"] == 20.0
    assert fams["sadservers"] == 100.0