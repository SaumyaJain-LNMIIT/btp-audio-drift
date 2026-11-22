"""Phase 4 — drift-type identification.

Once the monitor has flagged that drift occurred, this module classifies *what
kind* of drift it was, from the shape of the student-teacher discrepancy (the
mean reconstruction loss) over the stream:

    abrupt       a single sharp step change (loss jumps, then stays high)
    gradual      a sustained ramp over a moderate window, then stays high
    incremental  a slow climb over a wide window, then stays high
    recurring    a rise followed by a return to the pre-drift baseline

The classifier is rule-based over interpretable, amplitude-free shape features,
so it needs no labels at inference time and its decisions can be explained in
the report. The true type is used only offline (Phase 4.2) to score it.

Why the mean loss (not the hybrid score)
----------------------------------------
The monitored hybrid score is z-normalised against a tight reference window, so
it is extremely spiky and its per-step shape is dominated by noise. The mean
reconstruction loss is the clean underlying discrepancy signal; smoothing it
recovers the drift *shape* (step vs ramp vs staircase vs return).

Method
------
1. Smooth the loss trajectory (moving average, ~3% of stream) to suppress
   per-sample noise.
2. Baseline level/scale from the first ``baseline_frac`` of the (smoothed)
   stream — assumed normal, pre-drift.
3. Amplitude gate: if the smoothed rise above baseline is not large relative to
   baseline fluctuation, there is no drift shape to name → ``unknown``.
4. Min-max normalise the smoothed rise to [0, 1] so decisions are amplitude-
   free (abrupt drift can be tiny or huge; only the *shape* should matter).
5. End level → recurring; otherwise rise duration (as a fraction of the stream)
   → abrupt / gradual / incremental.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

DRIFT_TYPES = ("abrupt", "gradual", "incremental", "recurring", "unknown")

# Coarse taxonomy — the reliably separable level. abrupt/gradual/incremental
# all end in a drifted state ("persistent"); recurring returns to baseline;
# unknown means no excursion above baseline noise ("no_drift").
COARSE_MAP = {
    "abrupt": "persistent",
    "gradual": "persistent",
    "incremental": "persistent",
    "recurring": "recurring",
    "unknown": "no_drift",
}
COARSE_TYPES = ("persistent", "recurring", "no_drift")


def coarse_type(label: str) -> str:
    """Map a fine drift-type label to the coarse, reliably-separable taxonomy."""
    return COARSE_MAP.get(label, "no_drift")


@dataclass
class DriftTypeResult:
    """Outcome of classifying one detected drift."""
    label: str
    features: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        feats = ", ".join(f"{k}={v:.3g}" if isinstance(v, (int, float)) else f"{k}={v}"
                           for k, v in self.features.items())
        return f"{self.label} ({feats})"


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average with edge-padding (no boundary dip toward zero)."""
    x = np.asarray(x, dtype=float)
    if w <= 1 or x.size < 3:
        return x
    w = min(w, x.size)
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    k = np.ones(w) / w
    return np.convolve(xp, k, mode="valid")[: x.size]


def classify_drift_type(
    signal: Sequence[float],
    detection_index: int = 0,
    *,
    baseline_frac: float = 0.15,
    smooth_frac: float = 0.03,
    gate_ratio: float = 3.0,
    recur_end: float = 0.30,
    abrupt_max_frac: float = 0.05,
    incr_min_frac: float = 0.20,
) -> DriftTypeResult:
    """Classify the drift type from a discrepancy (mean-loss) trajectory.

    Args:
        signal: full per-step mean-loss (or score) history for the stream.
        detection_index: index where drift was flagged (used only to require the
            peak to precede the stream end for recurrence).
        baseline_frac: fraction of the stream (from the start) used as the
            normal reference.
        smooth_frac: moving-average window as a fraction of the stream.
        gate_ratio: min (smoothed rise / baseline fluctuation) for a real drift
            shape; below this → ``unknown``.
        recur_end: if the normalised end level is below this, the drift returned
            to baseline → recurring.
        abrupt_max_frac: rise shorter than this fraction of the stream → abrupt.
        incr_min_frac: rise longer than this fraction of the stream → incremental.

    Returns:
        A :class:`DriftTypeResult` with the label and the shape features used.
    """
    s = np.asarray(signal, dtype=float)
    n = s.size
    if n < 10:
        return DriftTypeResult("unknown", {"reason": "too_few_points", "n": n})

    bl = max(5, int(baseline_frac * n))
    w = max(3, int(smooth_frac * n))
    sm = _smooth(s, w)

    # Baseline level, and the ABSOLUTE deviation from it. Using |loss - base|
    # makes the classifier direction-agnostic: the drifted distribution may
    # raise OR lower the reconstruction loss (it is seed-dependent), and both
    # are drift.
    b = float(np.median(sm[:bl]))
    dev = np.abs(sm - b)

    base_dev = dev[:bl]
    base_noise = float(np.median(base_dev)) * 1.4826
    if base_noise < 1e-9:
        base_noise = float(base_dev.std()) or 1e-9

    # Robust "elevated" level — the 90th percentile of the deviation, NOT the
    # max, so a transient spike at drift onset doesn't define the scale.
    high = float(np.percentile(dev, 90))
    amp = high

    features: dict = {
        "amp": amp,
        "base_noise": base_noise,
        "amp_ratio": amp / base_noise,
        "det_index": int(detection_index),
    }

    # Gate: is there a real excursion above baseline noise?
    if amp <= gate_ratio * base_noise:
        features["reason"] = "amp_below_gate"
        return DriftTypeResult("unknown", features)

    z = dev / amp  # 0 at baseline, ~1 at the elevated level

    late_w = max(5, int(0.12 * n))
    end_level = float(np.median(dev[n - late_w:]) / amp)
    features["end_level"] = end_level

    # Recurrence: the deviation returns to baseline by the end of the stream
    # (abrupt / gradual / incremental all END drifted, so their end_level ~1).
    if end_level < recur_end:
        features["rise_frac"] = float("nan")
        return DriftTypeResult("recurring", features)

    # Rise duration: first crossing of 0.2 to first crossing of 0.8 of the
    # deviation — the initial climb from baseline to the drifted level.
    lo_hits = np.where(z >= 0.2)[0]
    hi_hits = np.where(z >= 0.8)[0]
    lo = int(lo_hits[0]) if lo_hits.size else 0
    hi = int(hi_hits[0]) if hi_hits.size else n - 1
    if hi < lo:
        hi = lo
    rise_frac = max(1, hi - lo) / n
    features["rise_frac"] = rise_frac

    if rise_frac <= abrupt_max_frac:
        label = "abrupt"
    elif rise_frac >= incr_min_frac:
        label = "incremental"
    else:
        label = "gradual"
    return DriftTypeResult(label, features)


def classify_from_monitor(monitor, detection_index: int = 0, **kwargs) -> DriftTypeResult:
    """Classify using a :class:`DriftMonitor`'s mean-loss history.

    The mean reconstruction loss (not the spiky hybrid score) is the clean
    discrepancy signal the shape features are read from.

    Args:
        monitor: a DriftMonitor that has streamed a scenario.
        detection_index: which detected event to anchor on (0 = first). If the
            monitor flagged nothing, the trajectory peak is used.
    """
    signal = getattr(monitor, "mean_losses", None) or monitor.scores
    if getattr(monitor, "drift_events", None):
        idx = min(detection_index, len(monitor.drift_events) - 1)
        det = monitor.drift_events[idx].timestamp
    else:
        det = int(np.argmax(signal)) if len(signal) else 0
    return classify_drift_type(signal, det, **kwargs)