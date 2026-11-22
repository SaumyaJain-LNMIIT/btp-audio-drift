"""Standard streaming drift detectors as baselines (model-free).

Wraps ADWIN, KSWIN and Page-Hinkley from ``river`` so they can be run directly
on a 1-D signal — here, the mean of the MFCC feature vector per stream step —
with NO teacher/student models. This is the honest point of comparison for the
hybrid detector: does the student-teacher + ensemble machinery actually beat a
standard detector run on the raw feature signal?
"""

from __future__ import annotations

from typing import Iterable

from river.drift import ADWIN, KSWIN, PageHinkley

BASELINE_FACTORIES = {
    "adwin": lambda **p: ADWIN(**p),
    "kswin": lambda **p: KSWIN(**p),
    "page_hinkley": lambda **p: PageHinkley(**p),
}


def make_detector(name: str, **params):
    """Build a river drift detector by name ('adwin' / 'kswin' / 'page_hinkley')."""
    key = name.lower()
    if key not in BASELINE_FACTORIES:
        raise ValueError(
            f"Unknown detector '{name}'. Choose from {list(BASELINE_FACTORIES)}"
        )
    return BASELINE_FACTORIES[key](**params)


def detect_on_signal(detector, signal: Iterable[float]) -> list[int]:
    """Feed a 1-D signal to a river detector and return detection timestamps."""
    detections: list[int] = []
    for t, x in enumerate(signal):
        detector.update(float(x))
        if detector.drift_detected:
            detections.append(t)
    return detections