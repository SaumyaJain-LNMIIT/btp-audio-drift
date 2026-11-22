"""Unit tests for evaluation.metrics event-matching and derived metrics.

No dataset, torch, or network required — pure numpy. Run with:
    python -m pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import math

from evaluation.metrics import (
    DEFAULT_TOLERANCE,
    match_events,
    evaluate_detection,
    false_discovery_proportion,
    false_alarm_rate_per_1k,
)

TOL = 100


def test_correct_detection_within_tolerance():
    m = evaluate_detection(true_points=[1000], detections=[1050],
                           total_steps=5000, tolerance=TOL)
    assert m["true_positives"] == 1
    assert m["false_alarms"] == 0
    assert m["missed_events"] == 0
    assert m["detection_delay"] == 50
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_missed_detection_beyond_tolerance():
    # Alarm lands 300 steps after the event, tolerance is 100 -> miss + false alarm
    m = evaluate_detection([1000], [1300], total_steps=5000, tolerance=TOL)
    assert m["true_positives"] == 0
    assert m["missed_events"] == 1
    assert m["false_alarms"] == 1
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0


def test_detection_before_event_is_false_alarm():
    # Alarm fires BEFORE the true change -> cannot be caused by it
    m = evaluate_detection([1000], [950], total_steps=5000, tolerance=TOL)
    assert m["true_positives"] == 0
    assert m["missed_events"] == 1
    assert m["false_alarms"] == 1


def test_multiple_detections_for_one_drift():
    # First in-window alarm is the TP; the rest are false alarms
    m = evaluate_detection([1000], [1020, 1040, 1080], total_steps=5000, tolerance=TOL)
    assert m["true_positives"] == 1
    assert m["false_alarms"] == 2
    assert m["detection_delay"] == 20  # earliest in-window alarm
    assert math.isclose(m["precision"], 1 / 3)


def test_multiple_true_events_one_to_one():
    m = evaluate_detection([1000, 2000, 3000],
                           [1030, 2010, 3090],
                           total_steps=5000, tolerance=TOL)
    assert m["true_positives"] == 3
    assert m["false_alarms"] == 0
    assert m["missed_events"] == 0
    assert m["recall"] == 1.0
    assert math.isclose(m["detection_delay"], (30 + 10 + 90) / 3)


def test_one_alarm_cannot_cover_two_events():
    # Two true events, a single alarm -> only the first is matched
    m = evaluate_detection([1000, 1005], [1010], total_steps=5000, tolerance=TOL)
    assert m["true_positives"] == 1
    assert m["missed_events"] == 1
    assert m["false_alarms"] == 0  # the alarm was consumed by event 1000


def test_stationary_stream_no_true_events():
    # No drift; any alarm is a false alarm; recall/precision are n/a (NaN)
    m = evaluate_detection([], [500, 1500, 2500], total_steps=9000, tolerance=TOL)
    assert m["true_positives"] == 0
    assert m["false_alarms"] == 3
    assert math.isnan(m["recall"])
    assert math.isnan(m["f1"])
    assert math.isclose(m["false_alarms_per_1k"], 3 / 9000 * 1000)


def test_stationary_stream_no_alarms_is_clean():
    m = evaluate_detection([], [], total_steps=9000, tolerance=TOL)
    assert m["false_alarms"] == 0
    assert m["false_alarms_per_1k"] == 0.0


def test_match_events_structure():
    # event 100 <- alarm 140 (delay 40). Event 500 has no alarm in [500, 600]:
    # 480 is before it, 700 is beyond 500+100 -> 500 is missed, 480 & 700 false.
    mr = match_events([100, 500], [140, 480, 700], tolerance=TOL)
    assert mr.true_positives == 1
    assert mr.delays == [40]
    assert [t for (t, _, _) in mr.matches] == [100]
    assert mr.missed_events == [500]
    assert sorted(mr.false_alarms) == [480, 700]


def test_helpers():
    assert false_discovery_proportion(3, 1) == 0.25
    assert math.isnan(false_discovery_proportion(0, 0))
    assert false_alarm_rate_per_1k(5, 10000) == 0.5
    assert math.isnan(false_alarm_rate_per_1k(5, None))


def test_default_tolerance_constant():
    assert DEFAULT_TOLERANCE == 100


# ------------------- span (gradual) events -------------------
def test_span_event_detection_within_window():
    # gradual onset 700, completion 900; detection 950 is within [700, 900+100]
    m = evaluate_detection([(700, 900)], [950], total_steps=2000, tolerance=100)
    assert m["true_positives"] == 1
    assert m["detection_delay"] == 250  # measured from onset (700)
    assert m["recall"] == 1.0


def test_span_event_detection_past_window_is_miss():
    # detection 1050 > 900+100 -> miss + false alarm
    m = evaluate_detection([(700, 900)], [1050], total_steps=2000, tolerance=100)
    assert m["true_positives"] == 0
    assert m["missed_events"] == 1
    assert m["false_alarms"] == 1


def test_span_multiple_detections_one_tp_rest_fa():
    # two alarms inside the window -> first is TP, second a false alarm
    m = evaluate_detection([(700, 900)], [760, 980], total_steps=2000, tolerance=100)
    assert m["true_positives"] == 1
    assert m["false_alarms"] == 1
    assert m["detection_delay"] == 60