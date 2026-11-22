"""Offline downstream-performance evaluation for harmful-vs-harmless drift.

This module runs OFFLINE, after the stream has been built and after the
unsupervised detector has already produced its alerts. It reads ground-truth
labels (which the detector never sees) and answers: did downstream
classification performance degrade, and when?

Pipeline:
    1. ``baseline_performance`` — measure the frozen classifier's performance on
       a REFERENCE set (validation folds, or the pre-drift segment of the
       stationary stream) in windows, giving ``P_base`` and its between-window
       variability ``sigma_base``. This is done BEFORE looking at any drift
       scenario, so the harm threshold is pre-registered, not fitted to results.
    2. ``windowed_performance`` — measure the classifier over tumbling windows of
       the actual stream.
    3. ``assess_stream`` — apply the pre-registered rule to label the stream
       harmful / harmless / inconclusive and locate the degradation onset t_A.

Harm rule (pre-registered; constants live in the caller's config):
    hard threshold = P_base - max(k * sigma_base, tau_abs)
    soft threshold = P_base - min(k * sigma_base, tau_abs)
    * HARMFUL   : >= S consecutive windows below the HARD threshold.
                  t_A = first sample index of that first sustained run.
    * HARMLESS  : no window falls below the SOFT threshold (clearly fine).
    * INCONCLUSIVE: anything else — a short (< S) degraded run, windows only in
                  the soft/hard gray band, or too few windows (< S) to ever
                  confirm a sustained run.

The module depends only on an object exposing ``.macro_f1(X, y)`` and
``.accuracy(X, y)`` (e.g. models.downstream_classifier.DownstreamClassifier) —
no torch / detector / DP imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("audio_drift")

VALID_METRICS = ("macro_f1", "accuracy")


@dataclass
class WindowScore:
    """One tumbling-window measurement, placed on the sample timeline."""

    start: int          # first sample index (inclusive)
    end: int            # last sample index (inclusive)
    count: int          # samples in the window
    value: float        # metric value in the window
    drop: float         # P_base - value (NaN until baseline is known)
    below_soft: bool
    below_hard: bool


@dataclass
class StreamAssessment:
    """Verdict for one stream."""

    verdict: str                 # "harmful" | "harmless" | "inconclusive"
    t_A: int | None              # degradation-onset sample index, or None
    metric: str
    p_base: float
    sigma_base: float
    hard_threshold: float
    soft_threshold: float
    max_drop: float
    sustained_run: int           # length of the longest below-HARD run (windows)
    windows: list[WindowScore] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    reason: str = ""             # why "inconclusive", when applicable


def _metric_fn(clf, metric: str):
    if metric not in VALID_METRICS:
        raise ValueError(f"unknown metric '{metric}' (use one of {VALID_METRICS})")
    fn = getattr(clf, metric, None)
    if fn is None:
        raise AttributeError(f"classifier has no '{metric}' method")
    return fn


def _iter_windows(n: int, window: int, stride: int, min_count: int):
    """Yield (start, end_exclusive) index ranges over ``n`` samples."""
    if window <= 0:
        raise ValueError("window must be positive")
    start = 0
    while start < n:
        end = min(start + window, n)
        if end - start >= min_count:
            yield start, end
        start += stride


def windowed_performance(
    clf,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    window: int,
    metric: str = "macro_f1",
    stride: int | None = None,
    min_count: int | None = None,
) -> list[WindowScore]:
    """Measure ``clf`` over windows of a stream (tumbling by default).

    Args:
        clf: object with ``.macro_f1`` / ``.accuracy``.
        features, labels: stream arrays, in stream order.
        window: window size in samples.
        metric: ``"macro_f1"`` or ``"accuracy"``.
        stride: step between window starts (default = ``window``, i.e. tumbling;
            set < window for sliding windows).
        min_count: drop a trailing window with fewer than this many samples
            (default = ``window``, i.e. drop any partial tail).

    Returns:
        List of :class:`WindowScore` with ``drop`` / ``below_*`` left unset
        (NaN / False) — they are filled by :func:`assess_stream`.
    """
    features = np.asarray(features)
    labels = np.asarray(labels)
    fn = _metric_fn(clf, metric)
    stride = window if stride is None else stride
    min_count = window if min_count is None else min_count

    out: list[WindowScore] = []
    for s, e in _iter_windows(len(features), window, stride, min_count):
        val = float(fn(features[s:e], labels[s:e]))
        out.append(WindowScore(start=s, end=e - 1, count=e - s, value=val,
                               drop=float("nan"), below_soft=False, below_hard=False))
    return out


def baseline_performance(
    clf,
    ref_features: np.ndarray,
    ref_labels: np.ndarray,
    *,
    window: int,
    metric: str = "macro_f1",
    min_count: int | None = None,
) -> dict:
    """Baseline ``P_base`` and between-window ``sigma_base`` on a REFERENCE set.

    The reference is the validation folds (or the stationary stream's pre-drift
    segment) — never the drift scenario under test. Call this once, before
    assessing any scenario, so the harm threshold is pre-registered.

    Returns:
        ``{"p_base", "sigma_base", "n_windows", "per_window"}``. With a single
        window, ``sigma_base`` is 0.0 (and the caller should lean on ``tau_abs``).
    """
    scores = windowed_performance(clf, ref_features, ref_labels,
                                  window=window, metric=metric, min_count=min_count)
    vals = np.array([w.value for w in scores], dtype=float)
    if len(vals) == 0:
        raise ValueError("reference produced no windows; lower `window` or `min_count`")
    return {
        "p_base": float(vals.mean()),
        "sigma_base": float(vals.std(ddof=0)),
        "n_windows": int(len(vals)),
        "per_window": vals.tolist(),
    }


def _first_run_ge(flags: list[bool], s_min: int) -> tuple[int | None, int]:
    """First index of a run of ``True`` with length >= ``s_min``; and the longest
    run length seen. Returns (first_qualifying_start_index | None, longest_run)."""
    longest = run = 0
    first_start = None
    run_start = 0
    for i, f in enumerate(flags):
        if f:
            if run == 0:
                run_start = i
            run += 1
            if run > longest:
                longest = run
            if run >= s_min and first_start is None:
                first_start = run_start
        else:
            run = 0
    return first_start, longest


def assess_stream(
    scores: list[WindowScore],
    *,
    p_base: float,
    sigma_base: float,
    metric: str,
    k: float = 3.0,
    tau_abs: float = 0.10,
    s_sustained: int = 3,
) -> StreamAssessment:
    """Apply the pre-registered harm rule to a stream's window scores.

    Args:
        scores: output of :func:`windowed_performance` for the stream.
        p_base, sigma_base: from :func:`baseline_performance`.
        metric: for the record.
        k: multiples of ``sigma_base`` for the degradation band.
        tau_abs: absolute minimum drop (e.g. 0.10 macro-F1) to count as material.
        s_sustained: consecutive below-HARD windows required for "harmful".

    Returns:
        A :class:`StreamAssessment`. ``scores`` are annotated in place with
        ``drop`` / ``below_soft`` / ``below_hard``.
    """
    band = [k * sigma_base, tau_abs]
    hard = p_base - max(band)
    soft = p_base - min(band)

    for w in scores:
        w.drop = p_base - w.value
        w.below_soft = w.value < soft
        w.below_hard = w.value < hard

    below_hard = [w.below_hard for w in scores]
    below_soft = [w.below_soft for w in scores]
    first_hard, longest_hard = _first_run_ge(below_hard, s_sustained)
    first_soft, longest_soft = _first_run_ge(below_soft, s_sustained)
    max_drop = max((w.drop for w in scores), default=float("nan"))
    any_below_soft = any(below_soft)

    params = {"k": k, "tau_abs": tau_abs, "s_sustained": s_sustained}
    common = dict(metric=metric, p_base=p_base, sigma_base=sigma_base,
                  hard_threshold=hard, soft_threshold=soft, max_drop=max_drop,
                  sustained_run=longest_hard, windows=scores, params=params)

    # HARMFUL: a sustained run below the hard threshold. (Criterion unchanged.)
    if first_hard is not None:
        return StreamAssessment(verdict="harmful", t_A=scores[first_hard].start,
                                reason="", **common)

    # INCONCLUSIVE: persistent MILD degradation — a sustained run below the soft
    # threshold that never reaches the hard threshold (a real sag, not severe).
    if first_soft is not None:
        reason = (f"persistent mild degradation: >= {s_sustained} consecutive windows "
                  f"below the soft threshold but never a sustained run below hard "
                  f"(longest below-hard run = {longest_hard})")
        return StreamAssessment(verdict="inconclusive", t_A=None, reason=reason, **common)

    # INCONCLUSIVE: too few windows to ever confirm a sustained run, with dips present.
    if len(scores) < s_sustained and any_below_soft:
        reason = (f"only {len(scores)} windows (< s_sustained={s_sustained}) with sub-soft "
                  "dips present; cannot confirm whether degradation is sustained")
        return StreamAssessment(verdict="inconclusive", t_A=None, reason=reason, **common)

    # HARMLESS: no sustained sub-soft degradation. Transient dips (normal
    # baseline noise crossing the soft band briefly) are allowed here.
    reason = ("no sub-soft windows" if not any_below_soft
              else f"only transient sub-soft dips (longest below-soft run = {longest_soft})")
    return StreamAssessment(verdict="harmless", t_A=None, reason=reason, **common)