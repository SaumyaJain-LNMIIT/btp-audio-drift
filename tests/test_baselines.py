"""Tests for drift_detection.baselines — river detector wrappers."""

from __future__ import annotations

import pytest

from drift_detection.baselines import make_detector, detect_on_signal


def test_adwin_detects_a_mean_shift():
    signal = [0.0] * 300 + [5.0] * 300
    det = detect_on_signal(make_detector("adwin"), signal)
    assert any(d >= 300 for d in det)          # fires after the shift


def test_page_hinkley_detects_a_mean_shift():
    signal = [0.0] * 300 + [5.0] * 300
    det = detect_on_signal(make_detector("page_hinkley"), signal)
    assert len(det) >= 1


def test_no_detection_on_constant_signal():
    det = detect_on_signal(make_detector("adwin"), [1.0] * 500)
    assert det == []


def test_unknown_detector_raises():
    with pytest.raises(ValueError):
        make_detector("not_a_detector")


def test_all_three_names_build():
    for name in ("adwin", "kswin", "page_hinkley"):
        d = make_detector(name)
        d.update(0.0)
        assert hasattr(d, "drift_detected")