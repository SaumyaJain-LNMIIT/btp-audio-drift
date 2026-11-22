"""Tests for evaluation.aggregate — mean ± std across seeds, NaN-safe."""

from __future__ import annotations

import math

from evaluation.aggregate import aggregate_runs, format_aggregate_table


def _rows():
    # abrupt detected in both seeds (delays 43, 53); stationary never detects.
    return [
        {"scenario": "abrupt", "seed": 1, "detection_delay": 43.0, "recall": 1.0,
         "precision": 1.0, "f1": 1.0, "false_alarms": 0, "false_alarms_per_1k": 0.0,
         "num_detections": 1, "true_positives": 1},
        {"scenario": "abrupt", "seed": 2, "detection_delay": 53.0, "recall": 1.0,
         "precision": 1.0, "f1": 1.0, "false_alarms": 0, "false_alarms_per_1k": 0.0,
         "num_detections": 1, "true_positives": 1},
        {"scenario": "stationary", "seed": 1, "detection_delay": float("nan"),
         "recall": float("nan"), "precision": float("nan"), "f1": float("nan"),
         "false_alarms": 0, "false_alarms_per_1k": 0.0, "num_detections": 0,
         "true_positives": 0},
        {"scenario": "stationary", "seed": 2, "detection_delay": float("nan"),
         "recall": float("nan"), "precision": float("nan"), "f1": float("nan"),
         "false_alarms": 1, "false_alarms_per_1k": 1.2, "num_detections": 1,
         "true_positives": 0},
    ]


def test_delay_mean_std_over_seeds():
    agg = aggregate_runs(_rows())
    assert agg["abrupt"]["n_seeds"] == 2
    assert math.isclose(agg["abrupt"]["detection_delay_mean"], 48.0)
    assert math.isclose(agg["abrupt"]["detection_delay_std"], 5.0)  # std of {43,53}
    assert agg["abrupt"]["detection_rate"] == 1.0
    assert math.isclose(agg["abrupt"]["f1_mean"], 1.0)


def test_stationary_detection_rate_is_zero():
    agg = aggregate_runs(_rows())
    assert agg["stationary"]["detection_rate"] == 0.0
    assert math.isnan(agg["stationary"]["detection_delay_mean"])
    assert math.isclose(agg["stationary"]["false_alarms_per_1k_mean"], 0.6)


def test_nan_values_are_ignored_not_poisoning():
    rows = [
        {"scenario": "x", "f1": 1.0, "detection_delay": 10.0, "true_positives": 1},
        {"scenario": "x", "f1": float("nan"), "detection_delay": float("nan"),
         "true_positives": 0},
    ]
    agg = aggregate_runs(rows)
    assert math.isclose(agg["x"]["f1_mean"], 1.0)
    assert agg["x"]["f1_n"] == 1
    assert agg["x"]["detection_delay_n"] == 1


def test_table_renders_without_error():
    table = format_aggregate_table(aggregate_runs(_rows()))
    assert "MULTI-SEED SUMMARY" in table
    assert "abrupt" in table and "stationary" in table