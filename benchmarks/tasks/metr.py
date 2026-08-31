"""METR time-horizon tasks — long-horizon capability measurement.

Implements the METR time-horizon methodology (https://arxiv.org/abs/2503.14499,
analysis code: https://github.com/METR/eval-analysis-public) on top of METR's
public task suite (https://github.com/METR/public-tasks):

1. Run the agent on tasks with known human-expert completion times.
2. Record success/failure per task.
3. Fit a logistic curve of P(success) vs log2(human_minutes).
4. Report the **50%-time-horizon**: the task duration at which the agent
   succeeds half the time.

What is bundled here:

* :data:`benchmarks/data/metr_human_minutes.json` — the 228-task human-time
  label map extracted from METR's Time Horizon v1.1 release
  (``runs.jsonl``; sources HCAST, RE-Bench, SWAA; 0.02–1800 minutes).
* :func:`time_horizon_50` / :func:`horizon_summary` — the logistic fit and
  horizon extraction, usable standalone (tests, analysis scripts).
* :func:`metr_bridge` — a live Task-Standard task family run through METR's
  official Inspect bridge (``mtb.bridge``), with the horizon metric attached.

The bridge path requires ``pip install
git+https://github.com/METR/inspect-metr-task-bridge`` (package name ``mtb``),
a Docker daemon, and METR task images (pre-built from METR's ECR via
``secrets.env``, or built locally with ``mtb-build``). Task outcomes are then
scored by the task family's own scorer; ``human_minutes`` labels are attached
from the bundled map by task family.

Note the labeled set and the public suite overlap on 14 families
(password_check, sadservers, make_web_server, symbolic_regression, ...); those
are the tasks where live runs produce horizon points today. The remaining
labeled tasks (incl. all 66 SWAA short tasks) have no public runnable form —
they enter the curve only via recorded outcomes, not fabricated inputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

from inspect_ai import Task, task
from inspect_ai.scorer import CORRECT, Score, metric

try:  # numpy/sklearn ship with the [bench] extra
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover - fallback path exercised in tests
    np = None
    _HAVE_SKLEARN = False

# --------------------------------------------------------------------------- #
# Human-time labels (METR Time Horizon v1.1)
# --------------------------------------------------------------------------- #

_LABELS_PATH = Path(__file__).parent.parent / "data" / "metr_human_minutes.json"


def load_human_minutes() -> dict[str, float]:
    """Map ``task_family/task_id`` -> human-expert completion time (minutes)."""
    return json.loads(_LABELS_PATH.read_text())["labels"]


# --------------------------------------------------------------------------- #
# Logistic fit + time horizon
# --------------------------------------------------------------------------- #

def _logistic(x: float, k: float, x0: float) -> float:
    # Clamp the exponent: exp() overflows long before the fit is meaningful.
    z = max(-60.0, min(60.0, -k * (x - x0)))
    return 1.0 / (1.0 + math.exp(z))


def _fit_logistic(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Fit P(success) = sigmoid(k * (x - x0)) — METR's model shape.

    Uses sklearn's regularized logistic regression (robust on perfectly
    separated outcomes, where the unregularized MLE diverges). Returns
    ``(k, x0)``; one-class input has no finite horizon and yields ``(0, 0)``.
    """
    if not xs or len(set(ys)) < 2:
        return 0.0, 0.0
    if _HAVE_SKLEARN:
        X = np.asarray(xs, dtype=float).reshape(-1, 1)
        y = np.asarray(ys, dtype=int)
        # C inverse-regularization chosen small enough to keep separated data
        # finite, large enough that 200+ real samples are fit faithfully.
        clf = LogisticRegression(C=100.0, solver="lbfgs", max_iter=1000)
        clf.fit(X, y)
        k = float(clf.coef_[0, 0])
        x0 = float(-clf.intercept_[0] / k) if abs(k) > 1e-12 else 0.0
        return k, x0
    # Fallback: grid search over (k, x0) maximizing the log-likelihood.
    best = (0.0, 0.0, -math.inf)
    for k in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        for x0 in (min(xs) + i * (max(xs) - min(xs)) / 20 for i in range(21)):
            ll = 0.0
            for x, y in zip(xs, ys):
                p = min(max(_logistic(x, k, x0), 1e-9), 1 - 1e-9)
                ll += y * math.log(p) + (1 - y) * math.log(1 - p)
            if ll > best[2]:
                best = (k, x0, ll)
    return best[0], best[1]


def horizon_summary(
    attempts: list[tuple[float, bool]],
) -> dict:
    """50%-time-horizon from ``(human_minutes, succeeded)`` attempts.

    Returns ``{"time_horizon_50": minutes | None, "n": int, "fit": (k, x0)}``.
    ``time_horizon_50`` is ``None`` when all attempts share one outcome (the
    horizon is undefined — the agent succeeded/failed at every duration) or
    the fit degenerates (non-positive slope).
    """
    xs = [math.log2(max(hm, 1e-6)) for hm, _ in attempts]
    ys = [1.0 if ok else 0.0 for _, ok in attempts]
    if not attempts or len(set(ys)) < 2:
        return {"time_horizon_50": None, "n": len(attempts), "fit": None}
    k, x0 = _fit_logistic(xs, ys)
    # Success normally *decreases* with duration (k < 0); what matters is that
    # a slope exists at all. k ~ 0 means duration carries no signal.
    if abs(k) < 1e-6:
        return {"time_horizon_50": None, "n": len(attempts), "fit": (k, x0)}
    return {"time_horizon_50": float(2 ** x0), "n": len(attempts), "fit": (k, x0)}


def time_horizon_50(scores: list[Score]) -> float | None:
    """50%-time-horizon (minutes) from scored samples, or None if undefined.

    Each score must carry ``metadata["human_minutes"]`` (minutes) and a
    CORRECT/INCORRECT ``value``.
    """
    attempts = []
    for s in scores:
        hm = (s.metadata or {}).get("human_minutes")
        if hm is None:
            continue
        attempts.append((float(hm), s.value == CORRECT))
    return horizon_summary(attempts)["time_horizon_50"]


def _horizon_metric():
    """Inspect metric: 50%-time-horizon over samples carrying human_minutes."""

    @metric
    def time_horizon_50_metric(scores) -> float | None:
        return time_horizon_50(list(scores))

    return time_horizon_50_metric


# --------------------------------------------------------------------------- #
# Live METR Task-Standard family via the official bridge
# --------------------------------------------------------------------------- #

def _family_human_minutes(labels: dict[str, float]) -> dict[str, float]:
    """Collapse the per-task label map to per-family medians (minutes)."""
    from collections import defaultdict

    per_family: dict[str, list[float]] = defaultdict(list)
    for key, hm in labels.items():
        fam = key.rpartition("/")[0]
        per_family[fam].append(hm)
    medians = {}
    for fam, vals in per_family.items():
        vals.sort()
        n = len(vals)
        medians[fam] = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return medians


def metr_bridge(
    image_tag: str,
    agent: Callable | None = None,
    sample_id: str | None = None,
) -> Task:
    """Run one METR Task-Standard task family through the official bridge.

    Args:
        image_tag: METR task family + version, e.g. ``password_check-1.0.13``,
            resolved against ``INSPECT_METR_TASK_BRIDGE_REPOSITORY``, or a full
            image reference.
        agent: Optional Inspect solver factory to drive the task (e.g. OnIt's
            native-tools agent once Phase 4 lands). Defaults to the bridge's
            built-in basic agent (bash + python tools).
        sample_id: Optional single task name within the family (the bridge
            exposes each task in the family as a sample).

    Requires the ``mtb`` package, Docker, and METR task images. Samples are
    tagged with the family's median human time so the task-level
    ``time_horizon_50`` metric places outcomes on the duration axis.
    """
    try:
        from mtb import bridge as mtb_bridge
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "METR bridge not installed. Install with: "
            "pip install git+https://github.com/METR/inspect-metr-task-bridge"
        ) from exc

    kwargs: dict = {"image_tag": image_tag}
    if agent is not None:
        kwargs["agent"] = agent
    inner = mtb_bridge.bridge(**kwargs)

    labels = load_human_minutes()
    family = image_tag.rsplit("-", 1)[0]
    family_hm = _family_human_minutes(labels).get(family)
    if family_hm is not None:
        for sample in inner.dataset:
            sample.metadata = {**(sample.metadata or {}), "human_minutes": family_hm}

    if sample_id is not None:
        inner.dataset = inner.dataset.filter(lambda s: s.id == sample_id)

    t = Task(
        dataset=inner.dataset,
        solver=inner.solver,
        scorer=inner.scorer,
        setup=inner.setup,
        cleanup=inner.cleanup,
        name=f"metr_{inner.name}",
        version=inner.version,
    )
    t.metrics = [_horizon_metric()]
    return t


# Register as an Inspect task so `inspect eval benchmarks/tasks/metr.py` and
# `--tasks metr` both reach it. Kept as a thin @task wrapper so the required
# image_tag argument surfaces as a normal -T task arg.
@task
def metr(image_tag: str, agent: Callable | None = None,
         sample_id: str | None = None) -> Task:
    """METR Task-Standard family via the bridge (see :func:`metr_bridge`)."""
    return metr_bridge(image_tag=image_tag, agent=agent, sample_id=sample_id)