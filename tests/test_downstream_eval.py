"""Tests for evaluation.downstream_eval — windowing, baseline, and the
pre-registered harmful / harmless / inconclusive rule + t_A onset.

A tiny stub classifier (always predicts class 0) lets us drive per-window
performance exactly by choosing the label stream: a window of all-0 labels
scores 1.0, a window of all-1 labels scores 0.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.downstream_eval import (
    assess_stream,
    baseline_performance,
    windowed_performance,
)


class StubClf:
    """Predicts constant class 0, so accuracy == fraction of zeros in y."""

    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def accuracy(self, X, y):
        return float(np.mean(np.asarray(y) == 0))

    def macro_f1(self, X, y):
        from sklearn.metrics import f1_score
        y = np.asarray(y)
        labels = np.union1d(self.classes_, np.unique(y))
        return float(f1_score(y, self.predict(X), labels=labels, average="macro", zero_division=0))


def _feats(n):
    return np.zeros((n, 1), dtype=np.float32)


# ------------------------------------------------------------------ windowing
def test_tumbling_windows_cover_and_place_on_timeline():
    y = np.zeros(10, dtype=int)
    ws = windowed_performance(StubClf(), _feats(10), y, window=5, metric="accuracy")
    assert [(w.start, w.end, w.count) for w in ws] == [(0, 4, 5), (5, 9, 5)]


def test_partial_tail_dropped_by_default():
    y = np.zeros(12, dtype=int)
    ws = windowed_performance(StubClf(), _feats(12), y, window=5, metric="accuracy")
    # samples 10,11 form a partial window of 2 < min_count=5 -> dropped
    assert len(ws) == 2
    assert ws[-1].end == 9


def test_sliding_stride():
    y = np.zeros(10, dtype=int)
    ws = windowed_performance(StubClf(), _feats(10), y, window=5, stride=2, metric="accuracy")
    assert [w.start for w in ws] == [0, 2, 4]  # 6,8 give partial tails (<5), dropped


# ------------------------------------------------------------------ baseline
def test_baseline_perfect_reference():
    y = np.zeros(20, dtype=int)
    b = baseline_performance(StubClf(), _feats(20), y, window=5, metric="accuracy")
    assert b["p_base"] == 1.0
    assert b["sigma_base"] == 0.0
    assert b["n_windows"] == 4


# ------------------------------------------------------------------ verdicts
def test_harmful_sustained_degradation_sets_t_A():
    # 15 good (acc 1.0) then 15 bad (acc 0.0); window=5 -> 3 bad windows in a row
    y = np.concatenate([np.zeros(15, dtype=int), np.ones(15, dtype=int)])
    ws = windowed_performance(StubClf(), _feats(30), y, window=5, metric="accuracy")
    a = assess_stream(ws, p_base=1.0, sigma_base=0.0, metric="accuracy",
                      k=3.0, tau_abs=0.10, s_sustained=3)
    assert a.verdict == "harmful"
    assert a.t_A == 15                      # first sample of the first bad window
    assert a.sustained_run == 3


def test_harmless_when_no_degradation():
    y = np.zeros(30, dtype=int)
    ws = windowed_performance(StubClf(), _feats(30), y, window=5, metric="accuracy")
    a = assess_stream(ws, p_base=1.0, sigma_base=0.0, metric="accuracy",
                      k=3.0, tau_abs=0.10, s_sustained=3)
    assert a.verdict == "harmless"
    assert a.t_A is None


def test_transient_dip_is_harmless():
    # one bad window only, surrounded by good -> a transient dip, not sustained
    y = np.zeros(30, dtype=int)
    y[10:15] = 1                            # window [10,14] scores 0.0
    ws = windowed_performance(StubClf(), _feats(30), y, window=5, metric="accuracy")
    a = assess_stream(ws, p_base=1.0, sigma_base=0.0, metric="accuracy",
                      k=3.0, tau_abs=0.10, s_sustained=3)
    assert a.verdict == "harmless"          # transient dips don't count
    assert a.t_A is None
    assert a.sustained_run == 1             # longest below-hard run was 1 window


def test_sustained_soft_but_not_hard_is_inconclusive():
    # 3 consecutive windows in the gray band (below soft, above hard):
    # p_base=1.0, sigma=0.1 -> k*sigma=0.3; tau=0.5 -> hard=0.5, soft=0.7.
    # labels [0,0,0,1,1] per window -> accuracy 0.6 (below 0.7, above 0.5).
    y = np.concatenate([np.tile([0, 0, 0, 1, 1], 3), np.zeros(10, dtype=int)])
    ws = windowed_performance(StubClf(), _feats(len(y)), y, window=5, metric="accuracy")
    a = assess_stream(ws, p_base=1.0, sigma_base=0.1, metric="accuracy",
                      k=3.0, tau_abs=0.5, s_sustained=3)
    assert a.verdict == "inconclusive"
    assert a.t_A is None
    assert "mild" in a.reason


def test_inconclusive_too_few_windows():
    # 2 windows, both bad, but S=3 can never be met
    y = np.ones(10, dtype=int)
    ws = windowed_performance(StubClf(), _feats(10), y, window=5, metric="accuracy")
    a = assess_stream(ws, p_base=1.0, sigma_base=0.0, metric="accuracy",
                      k=3.0, tau_abs=0.10, s_sustained=3)
    assert a.verdict == "inconclusive"
    assert "windows" in a.reason


def test_threshold_uses_max_of_sigma_and_tau():
    # p_base=1.0, sigma=0.1 -> k*sigma=0.3; tau=0.1; hard=0.7, soft=0.9
    # a window at value 0.8: below soft (0.8<0.9) but NOT below hard (0.8>=0.7)
    y = np.zeros(10, dtype=int)
    y[0] = 1  # window [0,4] -> acc 0.8
    ws = windowed_performance(StubClf(), _feats(10), y, window=5, metric="accuracy")
    a = assess_stream(ws, p_base=1.0, sigma_base=0.1, metric="accuracy",
                      k=3.0, tau_abs=0.10, s_sustained=3)
    w0 = a.windows[0]
    assert w0.value == pytest.approx(0.8)
    assert w0.below_soft and not w0.below_hard
    assert a.hard_threshold == pytest.approx(0.7)
    assert a.soft_threshold == pytest.approx(0.9)


def test_macro_f1_metric_runs():
    y = np.concatenate([np.zeros(10, dtype=int), np.ones(10, dtype=int)])
    ws = windowed_performance(StubClf(), _feats(20), y, window=5, metric="macro_f1")
    assert len(ws) == 4 and all(0.0 <= w.value <= 1.0 for w in ws)


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        windowed_performance(StubClf(), _feats(10), np.zeros(10, dtype=int),
                             window=5, metric="f2")