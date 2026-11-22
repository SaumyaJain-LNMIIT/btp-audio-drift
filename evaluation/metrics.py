"""Evaluation metrics for drift-detection performance.

This module defines a single, explicit event-matching policy and derives every
metric from it. It deliberately keeps three things separate that are easy to
conflate:

* **Detection delay** — how late a true event is caught. Reported on its own,
  never folded into F1.
* **Precision / recall / F1** — event-level detection quality.
* **False alarms** — reported two ways: the *false-discovery proportion*
  (FA / alarms) and the *false-alarm rate per 1,000 observations*
  (FA / steps x 1000), which is the meaningful quantity on long stationary
  streams.

Event-matching policy (fixed and documented)
--------------------------------------------
* A true drift event at time ``t`` is **detected** if at least one alarm fires
  in the window ``[t, t + tolerance]`` — at or after the true change and within
  ``tolerance`` observations of it. Alarms *before* a true change cannot have
  been caused by it and are counted as false alarms.
* Matching is **greedy and one-to-one**: processing true events in time order,
  each is matched to the earliest not-yet-used alarm inside its window. Every
  matched alarm is a true positive; every remaining alarm is a false alarm;
  every true event with no alarm in its window is a miss.
* ``tolerance`` is a fixed, explicit number of observations (default 100). It is
  NOT scaled to the stream length. (The previous implementation expanded it to
  one quarter of the stream, which made F1 trivially 1.0.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

DEFAULT_TOLERANCE: int = 100


# ======================================================================
# Core event matching
# ======================================================================
@dataclass
class MatchResult:
    """Outcome of matching detections to ground-truth drift events.

    Attributes:
        matches: ``(true_t, alarm_t, delay)`` triples, one per detected event.
        missed_events: true drift timestamps with no alarm in their window.
        false_alarms: alarm timestamps not matched to any event.
    """

    matches: list[tuple[int, int, int]]
    missed_events: list[int]
    false_alarms: list[int]

    @property
    def true_positives(self) -> int:
        return len(self.matches)

    @property
    def delays(self) -> list[int]:
        return [d for (_, _, d) in self.matches]


def _as_span(event) -> tuple[int, int]:
    """Normalise an event to a ``(start, end)`` window.

    A point drift ``t`` becomes ``(t, t)``; a gradual drift is given as a
    ``(onset, completion)`` tuple and kept as-is. A detection matches the event
    if it lands in ``[start, end + tolerance]``.
    """
    if isinstance(event, (tuple, list)) and len(event) == 2:
        return int(event[0]), int(event[1])
    return int(event), int(event)


def match_events(
    true_points: Iterable,
    detections: Iterable[int],
    tolerance: int = DEFAULT_TOLERANCE,
) -> MatchResult:
    """Greedily match detections to true drift events (one-to-one).

    Each event is either a point ``t`` (window ``[t, t + tolerance]``) or a
    ``(onset, completion)`` span (window ``[onset, completion + tolerance]``),
    which is how gradual/incremental drift is scored: a detection anywhere from
    the onset to ``tolerance`` past completion counts, and the delay is measured
    from the onset.

    Args:
        true_points: Ground-truth drift events (ints and/or (start, end) spans).
        detections: Alarm timestamps raised by the detector.
        tolerance: Max allowed delay past the event window for a valid match.

    Returns:
        A :class:`MatchResult`. Missed events report their onset.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    spans = [_as_span(e) for e in true_points]
    # process events in onset order
    event_order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    det_sorted = sorted(int(d) for d in detections)
    used = [False] * len(det_sorted)

    matches: list[tuple[int, int, int]] = []
    missed: list[int] = []

    for i in event_order:
        start, end = spans[i]
        chosen: Optional[int] = None
        for j, d in enumerate(det_sorted):
            if used[j] or d < start:
                continue
            if d > end + tolerance:
                break  # det_sorted is ascending: no later alarm can match
            chosen = j
            break
        if chosen is None:
            missed.append(start)
        else:
            used[chosen] = True
            matches.append((start, det_sorted[chosen], det_sorted[chosen] - start))

    false_alarms = [det_sorted[j] for j in range(len(det_sorted)) if not used[j]]
    return MatchResult(matches, missed, false_alarms)


# ======================================================================
# Derived metrics
# ======================================================================
def mean_detection_delay(match_result: MatchResult) -> float:
    """Mean delay over detected events (NaN if nothing was detected)."""
    delays = match_result.delays
    return float(np.mean(delays)) if delays else float("nan")


def false_discovery_proportion(true_positives: int, false_alarms: int) -> float:
    """FA / (TP + FA): fraction of alarms that were spurious.

    This is what the old ``false_positive_rate`` actually computed. It is a
    *proportion of alarms*, not the conventional false-positive rate.
    """
    total = true_positives + false_alarms
    return false_alarms / total if total > 0 else float("nan")


def false_alarm_rate_per_1k(false_alarm_count: int, total_steps: Optional[int]) -> float:
    """False alarms per 1,000 observations (NaN if ``total_steps`` unknown)."""
    if total_steps and total_steps > 0:
        return false_alarm_count / total_steps * 1000.0
    return float("nan")


def evaluate_detection(
    true_points: Iterable[int],
    detections: Iterable[int],
    total_steps: Optional[int] = None,
    tolerance: int = DEFAULT_TOLERANCE,
) -> dict:
    """Compute the full metric set from one event match.

    Precision = TP / #alarms, recall = TP / #events, F1 their harmonic mean.
    Where a denominator is zero the metric is NaN (reported as "n/a"), so a
    stationary stream with no true events does not silently score 0 or 1.

    Returns:
        A dict with the match counts, delay, P/R/F1, and both false-alarm views.
    """
    true_list = list(true_points)          # events may be ints or (start, end) spans
    det_list = [int(d) for d in detections]
    n_true = len(true_list)
    n_det = len(det_list)

    mr = match_events(true_list, det_list, tolerance)
    tp = mr.true_positives
    fa = len(mr.false_alarms)

    precision = tp / n_det if n_det > 0 else float("nan")
    recall = tp / n_true if n_true > 0 else float("nan")
    if n_true == 0 or n_det == 0:
        f1 = float("nan")
    elif (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "tolerance": tolerance,
        "num_true_events": n_true,
        "num_detections": n_det,
        "true_positives": tp,
        "false_alarms": fa,
        "missed_events": len(mr.missed_events),
        "detection_delay": mean_detection_delay(mr),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_discovery_proportion": false_discovery_proportion(tp, fa),
        "false_alarms_per_1k": false_alarm_rate_per_1k(fa, total_steps),
        "matched_delays": mr.delays,
        "missed_timestamps": mr.missed_events,
        "false_alarm_timestamps": mr.false_alarms,
    }


# ======================================================================
# Deprecated shims — kept so the current pipeline keeps importing cleanly
# until drift_analysis.py is migrated to evaluate_detection(). Do not use
# in new code; these preserve the OLD (symmetric-tolerance) behaviour.
# ======================================================================
def detection_delay(
    true_drift_points: list[int],
    detected_timestamps: list[int],
) -> float:
    """[DEPRECATED] Mean delay to first detection at/after each true point."""
    if not true_drift_points or not detected_timestamps:
        return float("nan")
    delays: list[int] = []
    det = sorted(detected_timestamps)
    for tp in true_drift_points:
        after = [d for d in det if d >= tp]
        if after:
            delays.append(after[0] - tp)
    return float(np.mean(delays)) if delays else float("nan")


def false_positive_rate(
    true_drift_points: list[int],
    detected_timestamps: list[int],
    total_steps: int,
    tolerance: int = 50,
) -> float:
    """[DEPRECATED] Fraction of detections that are false (misnamed as FPR)."""
    if not detected_timestamps:
        return 0.0
    fp = 0
    for d in detected_timestamps:
        if not any(abs(d - tp) <= tolerance for tp in true_drift_points):
            fp += 1
    return fp / len(detected_timestamps)


def precision_recall_f1(
    true_drift_points: list[int],
    detected_timestamps: list[int],
    tolerance: int = 50,
) -> dict[str, float]:
    """[DEPRECATED] Symmetric-tolerance P/R/F1 without one-to-one matching."""
    if not detected_timestamps:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp_det = sum(
        1 for d in detected_timestamps
        if any(abs(d - tp) <= tolerance for tp in true_drift_points)
    )
    tp_true = sum(
        1 for tp in true_drift_points
        if any(abs(d - tp) <= tolerance for d in detected_timestamps)
    )
    precision = tp_det / len(detected_timestamps) if detected_timestamps else 0.0
    recall = tp_true / len(true_drift_points) if true_drift_points else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}