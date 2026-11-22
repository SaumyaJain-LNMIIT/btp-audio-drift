"""Tests for evaluation.harm_timing — alert/degradation lead-lag logic."""

from __future__ import annotations

from evaluation.harm_timing import (
    CLEAN,
    DETECTED_NO_HARM,
    EARLY,
    LATE,
    MISSED,
    classify_outcome,
    first_alert_at_or_after,
)


def test_first_alert_after_onset():
    assert first_alert_at_or_after([10, 200, 900], onset=788) == 900
    assert first_alert_at_or_after([10, 50], onset=788) is None   # all before onset
    assert first_alert_at_or_after([], onset=788) is None


def test_first_alert_no_onset_returns_earliest():
    # stationary stream: no true drift -> earliest alert (a false alarm)
    assert first_alert_at_or_after([300, 20, 80], onset=None) == 20
    assert first_alert_at_or_after([], onset=None) is None


def test_early_alert_positive_lead():
    bucket, lead = classify_outcome(t_D=800, t_A=950)
    assert bucket == EARLY and lead == 150


def test_late_alert_nonpositive_lead():
    bucket, lead = classify_outcome(t_D=1000, t_A=950)
    assert bucket == LATE and lead == -50
    # exactly simultaneous counts as late (no warning margin)
    bucket0, lead0 = classify_outcome(t_D=950, t_A=950)
    assert bucket0 == LATE and lead0 == 0


def test_missed_degradation():
    bucket, lead = classify_outcome(t_D=None, t_A=900)
    assert bucket == MISSED and lead is None


def test_detected_without_harm():
    bucket, lead = classify_outcome(t_D=800, t_A=None)
    assert bucket == DETECTED_NO_HARM and lead is None


def test_clean():
    bucket, lead = classify_outcome(t_D=None, t_A=None)
    assert bucket == CLEAN and lead is None