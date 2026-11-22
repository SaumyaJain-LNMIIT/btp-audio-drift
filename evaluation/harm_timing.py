"""Lead/lag relationship between a drift ALERT and downstream DEGRADATION.

Pure, data-free logic shared by the harmful-vs-harmless harness and its tests.
It connects two quantities that are produced by two SEPARATE passes over the
same stream (see claude/harmful_vs_harmless_design.md):

* ``t_D`` — the detector's drift-alert time (features-only pass; no labels).
* ``t_A`` — the downstream degradation-onset time (offline, label-based pass).

Nothing here consumes labels or model state; it only reasons about timestamps,
so it is unit-testable without torch, sklearn or data.
"""

from __future__ import annotations

# Outcome buckets (drift scenarios). "clean" is the ideal stationary outcome.
EARLY = "early_alert"             # detected before harm began (lead > 0)
LATE = "late_alert"              # detected, but at/after harm began (lead <= 0)
MISSED = "missed_degradation"    # harm happened, detector never alerted (dangerous)
DETECTED_NO_HARM = "detected_no_harm"  # detector fired but no measurable harm (over-firing)
CLEAN = "clean"                  # no alert, no harm


def first_alert_at_or_after(detections, onset: int | None):
    """First alert timestamp at or after ``onset``; the drift-relevant alert.

    Args:
        detections: iterable of alert timestamps (sample indices).
        onset: the drift's ground-truth onset (sample index). If ``None`` (e.g.
            a stationary stream with no true drift), the earliest alert of any
            kind is returned so it can be reported as a false alarm.

    Returns:
        The timestamp (int) or ``None`` if there is no qualifying alert.
    """
    ts = sorted(int(d) for d in detections)
    if not ts:
        return None
    if onset is None:
        return ts[0]
    for t in ts:
        if t >= onset:
            return t
    return None


def classify_outcome(t_D: int | None, t_A: int | None):
    """Map (alert time, degradation-onset time) to an outcome bucket + lead time.

    ``detected`` is defined as ``t_D is not None`` and ``harmful`` as
    ``t_A is not None``. Lead time is only defined when both occur.

    Returns:
        ``(bucket, lead_time)`` where ``lead_time = t_A - t_D`` (positive = the
        alert preceded the degradation) when both are present, else ``None``.
    """
    detected = t_D is not None
    harmful = t_A is not None

    if detected and harmful:
        lead = int(t_A) - int(t_D)
        return (EARLY if lead > 0 else LATE), lead
    if harmful and not detected:
        return MISSED, None
    if detected and not harmful:
        return DETECTED_NO_HARM, None
    return CLEAN, None