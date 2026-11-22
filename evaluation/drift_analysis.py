"""Drift analysis — compare detected vs true drift points and produce a summary.

This is thin glue over :func:`evaluation.metrics.evaluate_detection`. The old
implementation inflated the matching tolerance to ``total_steps // 4`` (a
quarter of the whole stream), which made precision/recall/F1 trivially perfect.
That behaviour is removed: matching now uses a fixed, caller-supplied tolerance
(default :data:`evaluation.metrics.DEFAULT_TOLERANCE` = 100 observations).

The returned dict is a superset of the old one. The legacy key
``false_positive_rate`` is kept as an alias of ``false_discovery_proportion``
(what the old metric actually measured) so existing experiment runners keep
working until they are migrated; new code should read the explicit keys.
"""

from __future__ import annotations

import logging
from typing import Any

from evaluation.metrics import DEFAULT_TOLERANCE, evaluate_detection

logger = logging.getLogger("audio_drift")


def analyse_drift(
    true_drift_points: list[int],
    detected_timestamps: list[int],
    total_steps: int,
    tolerance: int = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Run the metric suite and return a summary dict.

    Args:
        true_drift_points: Ground-truth drift timestamps.
        detected_timestamps: Alarm timestamps from the detector.
        total_steps: Length of the stream (for the false-alarm rate).
        tolerance: Fixed event-matching tolerance in observations. NOT scaled
            to the stream length.

    Returns:
        Dict with match counts, detection delay, precision/recall/F1, and both
        false-alarm views. Includes a deprecated ``false_positive_rate`` alias.
    """
    m = evaluate_detection(
        true_points=true_drift_points,
        detections=detected_timestamps,
        total_steps=total_steps,
        tolerance=tolerance,
    )

    summary: dict[str, Any] = {
        "true_drift_points": list(true_drift_points),
        "detected_timestamps": list(detected_timestamps),
        "total_steps": total_steps,
        "tolerance": m["tolerance"],
        "num_detections": m["num_detections"],
        "num_true_events": m["num_true_events"],
        "true_positives": m["true_positives"],
        "missed_events": m["missed_events"],
        "detection_delay": m["detection_delay"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "false_discovery_proportion": m["false_discovery_proportion"],
        "false_alarms": m["false_alarms"],
        "false_alarms_per_1k": m["false_alarms_per_1k"],
        # --- deprecated alias: current runners still read this key ---
        "false_positive_rate": m["false_discovery_proportion"],
    }

    logger.info("=== Drift Analysis (tolerance=%d) ===", m["tolerance"])
    for k in (
        "num_detections", "true_positives", "missed_events",
        "detection_delay", "precision", "recall", "f1",
        "false_discovery_proportion", "false_alarms", "false_alarms_per_1k",
    ):
        logger.info("  %-27s : %s", k, summary[k])

    return summary