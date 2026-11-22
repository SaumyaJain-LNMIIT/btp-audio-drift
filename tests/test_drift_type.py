"""Unit tests for Phase 4 drift-type identification.

Synthetic mean-loss trajectories with a known shape — at realistic stream
proportions (transition widths mirroring the real scenarios: abrupt ~0%,
gradual ~12%, incremental ~60% of the stream) — must map to their label.
Baselines are low-level normal noise; drift raises the loss level.
"""

from __future__ import annotations

import numpy as np

from drift_detection.drift_type import classify_drift_type, DRIFT_TYPES

RNG = np.random.default_rng(0)
N = 1600
BASE = 0.07          # baseline mean-loss level
HIGH = 1.0           # drifted level
SD = 0.02            # per-step noise


def _noise(n):
    return RNG.normal(0.0, SD, size=n)


def _abrupt():
    # flat, instant step up at mid-stream, stays high.
    pre = BASE + _noise(N // 2)
    post = HIGH + _noise(N - N // 2)
    return np.concatenate([pre, post]), N // 2


def _gradual():
    # flat, ramp over ~12% of the stream centred at mid, stays high.
    width = int(0.12 * N)
    start = N // 2 - width // 2
    pre = BASE + _noise(start)
    ramp = np.linspace(BASE, HIGH, width) + _noise(width)
    post = HIGH + _noise(N - start - width)
    return np.concatenate([pre, ramp, post]), start + width


def _incremental():
    # flat, slow climb over ~60% of the stream, stays high.
    width = int(0.60 * N)
    start = int(0.20 * N)
    pre = BASE + _noise(start)
    ramp = np.linspace(BASE, HIGH, width) + _noise(width)
    post = HIGH + _noise(N - start - width)
    return np.concatenate([pre, ramp, post]), start + width // 2


def _recurring():
    # flat, up to a plateau, then back to baseline by the end.
    a1 = BASE + _noise(int(0.27 * N))
    b = HIGH + _noise(int(0.46 * N))
    a2 = BASE + _noise(N - len(a1) - len(b))
    return np.concatenate([a1, b, a2]), len(a1)


def _stationary():
    return BASE + _noise(N), N // 2


def test_abrupt():
    scores, det = _abrupt()
    assert classify_drift_type(scores, det).label == "abrupt"


def test_gradual():
    scores, det = _gradual()
    assert classify_drift_type(scores, det).label == "gradual"


def test_incremental():
    scores, det = _incremental()
    assert classify_drift_type(scores, det).label == "incremental"


def test_recurring():
    scores, det = _recurring()
    assert classify_drift_type(scores, det).label == "recurring"


def test_stationary_is_unknown():
    scores, det = _stationary()
    assert classify_drift_type(scores, det).label == "unknown"


def test_label_in_vocab():
    scores, det = _abrupt()
    assert classify_drift_type(scores, det).label in DRIFT_TYPES


def test_features_present():
    scores, det = _gradual()
    res = classify_drift_type(scores, det)
    assert {"amp", "amp_ratio", "end_level", "rise_frac"} <= set(res.features)


def test_too_few_points():
    assert classify_drift_type([0.1, 0.2], 1).label == "unknown"