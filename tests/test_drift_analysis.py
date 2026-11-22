"""Tests for evaluation.drift_analysis.

The key regression test: a detection ~1,100 steps after the true drift in an
~8,700-step stream (exactly the saved 6th-sem abrupt result) must NOT score
F1 = 1.0 any more. Under the old dynamic tolerance it did.
"""

from __future__ import annotations

import math

from evaluation.drift_analysis import analyse_drift


def test_old_perfect_result_is_now_a_miss():
    # Saved abrupt run: true drift 4429, detected 5535, stream ~8732.
    s = analyse_drift([4429], [5535], total_steps=8732, tolerance=100)
    # 1106 >> 100 -> this is a MISS now, not a perfect detection
    assert s["true_positives"] == 0
    assert s["missed_events"] == 1
    assert s["recall"] == 0.0
    assert s["f1"] != 1.0
    # the alarm is counted as a false alarm instead
    assert s["false_alarms"] == 1


def test_genuine_timely_detection_scores_well():
    s = analyse_drift([4429], [4479], total_steps=8732, tolerance=100)
    assert s["true_positives"] == 1
    assert s["detection_delay"] == 50
    assert s["precision"] == 1.0
    assert s["recall"] == 1.0
    assert s["f1"] == 1.0


def test_backward_compatible_keys_present():
    # run_experiments.py / ablation_study.py read these exact keys
    s = analyse_drift([1000], [1050], total_steps=5000)
    for key in ("num_detections", "detection_delay", "false_positive_rate",
                "precision", "recall", "f1"):
        assert key in s
    # deprecated alias mirrors the honest metric
    assert s["false_positive_rate"] == s["false_discovery_proportion"]


def test_default_tolerance_is_fixed_not_scaled():
    # Two very different stream lengths, same geometry -> same verdict.
    # (Under the old code the tolerance scaled with total_steps.)
    short = analyse_drift([1000], [1150], total_steps=4000)   # delay 150 > 100
    long = analyse_drift([1000], [1150], total_steps=40000)   # delay 150 > 100
    assert short["true_positives"] == 0
    assert long["true_positives"] == 0


def test_stationary_stream_reports_false_alarm_rate():
    s = analyse_drift([], [500, 2500, 6000], total_steps=9000)
    assert s["false_alarms"] == 3
    assert math.isnan(s["recall"])
    assert math.isclose(s["false_alarms_per_1k"], 3 / 9000 * 1000)