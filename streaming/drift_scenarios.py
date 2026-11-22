"""Drift scenarios — build stream orderings (and transforms) that inject drift.

Two kinds of drift are modelled, kept conceptually separate:

* **Distribution / class-order patterns** — ``abrupt``, ``gradual``,
  ``incremental``, ``recurring`` change *which* clips appear over time by moving
  between class group A (first half of classes) and group B (second half).
* **Feature-space transform** — ``noise`` keeps the class distribution fixed and
  perturbs the feature vectors after the drift point (sensor-degradation proxy).

Waveform-level acoustic transforms (SNR, reverb, codec) are deferred to a later
phase; this module operates on the feature matrix it is given. Which clips it
receives (train vs held-out streaming-test) is decided by the caller — these
generators are split-agnostic.

Every generator:
* accepts an explicit ``seed`` so multi-seed experiments vary the stream,
* returns ``(stream_features, stream_labels, drift_point, true_drift_indices)``,
* consumes every provided sample (nothing is silently dropped),
* reports ground-truth drift timestamps that match what it actually builds.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("audio_drift")


# ======================================================================
# Shared helpers
# ======================================================================
def _group_indices(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices into class-group A and B, each shuffled by ``seed``."""
    rng = np.random.RandomState(seed)
    unique = sorted(np.unique(labels))
    mid = len(unique) // 2
    group_a, group_b = unique[:mid], unique[mid:]
    idx_a = np.where(np.isin(labels, group_a))[0]
    idx_b = np.where(np.isin(labels, group_b))[0]
    rng.shuffle(idx_a)
    rng.shuffle(idx_b)
    return idx_a, idx_b


def _build_class_mix(
    features: np.ndarray,
    labels: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    trans_start: int,
    trans_end: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Build an A->B stream with a transition window, dropping nothing.

    Layout (by stream position):
      * ``[0, trans_start)``         : pure group A
      * ``[trans_start, trans_end)`` : mix, probability of B rising 0 -> 1
      * ``[trans_end, total)``       : pure group B

    All of ``idx_a`` and ``idx_b`` are consumed. The ground-truth drift
    timestamp is ``trans_start`` (onset of the change).
    """
    rng = np.random.RandomState(seed + 1)
    len_a, len_b = len(idx_a), len(idx_b)
    total = len_a + len_b

    trans_start = int(np.clip(trans_start, 0, len_a))
    trans_end = int(np.clip(trans_end, trans_start, total))

    pre = list(idx_a[:trans_start])
    remaining_a = list(idx_a[trans_start:])          # every leftover A kept
    remaining_b = list(idx_b)

    # Number of B to emit during the transition so that pre + transition leaves
    # exactly the right amount of B for the post segment.
    trans_len = trans_end - trans_start
    n_b_trans = int(np.clip(trans_len - len(remaining_a), 0, len(remaining_b)))

    mixed: list[int] = []
    a_ptr = b_ptr = 0
    emitted_b = 0
    span = max(1, trans_len)
    for pos in range(trans_len):
        want_b_density = pos / span            # target B fraction rises 0 -> 1
        a_left = len(remaining_a) - a_ptr
        b_left_budget = n_b_trans - emitted_b
        emit_b = False
        if b_left_budget > 0 and (
            a_left == 0 or (emitted_b < want_b_density * (a_ptr + emitted_b + 1))
        ):
            emit_b = True
        if emit_b:
            mixed.append(remaining_b[b_ptr]); b_ptr += 1; emitted_b += 1
        elif a_left > 0:
            mixed.append(remaining_a[a_ptr]); a_ptr += 1
        elif b_left_budget > 0:
            mixed.append(remaining_b[b_ptr]); b_ptr += 1; emitted_b += 1

    # Any remaining A (if the transition was shorter than #leftover A) and all
    # remaining B go into the post segment, A first so B still dominates the tail.
    post = list(remaining_a[a_ptr:]) + list(remaining_b[b_ptr:])

    stream_idx = np.array(pre + mixed + post, dtype=int)
    assert len(stream_idx) == total, "class-mix dropped samples"  # invariant

    # return the transition window bounds; wrappers build the (onset, end) span
    return features[stream_idx], labels[stream_idx], trans_start, trans_end


# ======================================================================
# Scenario generators
# ======================================================================
def abrupt_drift(
    features: np.ndarray,
    labels: np.ndarray,
    drift_point_ratio: float = 0.5,
    seed: int = 42,
    **_: Any,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Sudden A->B switch. Drift position honours ``drift_point_ratio``.

    To place the boundary at the requested ratio without polluting either side,
    the larger group is subsampled so that ``take_a / (take_a + take_b)`` equals
    the ratio while keeping as many samples as possible.
    """
    idx_a, idx_b = _group_indices(labels, seed)
    n_a, n_b = len(idx_a), len(idx_b)
    r = float(np.clip(drift_point_ratio, 0.05, 0.95))

    # Max total length T s.t. r*T <= n_a and (1-r)*T <= n_b.
    t_max = min(n_a / r, n_b / (1.0 - r))
    take_a = int(r * t_max)
    take_b = int(t_max) - take_a
    take_a, take_b = max(1, take_a), max(1, take_b)

    stream_idx = np.concatenate([idx_a[:take_a], idx_b[:take_b]])
    drift_point = take_a
    true_drifts = [drift_point]
    logger.info(
        "Abrupt drift: %d pre + %d post = %d samples, drift at t=%d (ratio=%.2f)",
        take_a, take_b, len(stream_idx), drift_point, r,
    )
    return features[stream_idx], labels[stream_idx], drift_point, true_drifts


def gradual_drift(
    features: np.ndarray,
    labels: np.ndarray,
    drift_point_ratio: float = 0.5,
    transition_width: int = 200,
    seed: int = 42,
    **_: Any,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Slow A->B mix over a short transition window (default 200 observations)."""
    idx_a, idx_b = _group_indices(labels, seed)
    total = len(idx_a) + len(idx_b)
    drift_point = int(total * drift_point_ratio)
    half = transition_width // 2
    stream_f, stream_l, start, end = _build_class_mix(
        features, labels, idx_a, idx_b,
        trans_start=drift_point - half, trans_end=drift_point + half, seed=seed,
    )
    logger.info(
        "Gradual drift: transition [%d, %d] (width %d) of %d",
        start, end, transition_width, total,
    )
    return stream_f, stream_l, start, [(start, end)]


def incremental_drift(
    features: np.ndarray,
    labels: np.ndarray,
    drift_point_ratio: float = 0.5,
    transition_fraction: float = 0.6,
    seed: int = 42,
    **_: Any,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """A->B mix stretched over a long interval (default 60% of the stream).

    Same mechanism as ``gradual`` but with a wide window, so the drift score
    rises slowly over a long span rather than in a short ramp. This shape
    difference is what later drift-type identification keys on.
    """
    idx_a, idx_b = _group_indices(labels, seed)
    total = len(idx_a) + len(idx_b)
    width = max(2, int(total * transition_fraction))
    drift_point = int(total * drift_point_ratio)
    half = width // 2
    stream_f, stream_l, start, end = _build_class_mix(
        features, labels, idx_a, idx_b,
        trans_start=drift_point - half, trans_end=drift_point + half, seed=seed,
    )
    logger.info(
        "Incremental drift: wide transition [%d, %d] (%.0f%%) of %d",
        start, end, transition_fraction * 100, total,
    )
    return stream_f, stream_l, start, [(start, end)]


def recurring_drift(
    features: np.ndarray,
    labels: np.ndarray,
    drift_point_ratio: float = 0.5,
    seed: int = 42,
    **_: Any,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Explicit A -> B -> A stream with two known ground-truth drift points."""
    idx_a, idx_b = _group_indices(labels, seed)
    half_a = len(idx_a) // 2
    a1, a2 = idx_a[:half_a], idx_a[half_a:]

    stream_idx = np.concatenate([a1, idx_b, a2])
    p1 = len(a1)                       # A -> B
    p2 = len(a1) + len(idx_b)          # B -> A
    true_drifts = [p1, p2]
    logger.info(
        "Recurring drift: A(%d) -> B(%d) -> A(%d), drifts at t=%d and t=%d",
        len(a1), len(idx_b), len(a2), p1, p2,
    )
    return features[stream_idx], labels[stream_idx], p1, true_drifts


def noise_drift(
    features: np.ndarray,
    labels: np.ndarray,
    drift_point_ratio: float = 0.5,
    noise_scale: float = 1.0,
    seed: int = 42,
    **_: Any,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Feature-corruption drift on NORMAL (group-A) data.

    Streams only group-A clips — the distribution the models know as "normal" —
    then adds Gaussian noise to the feature vectors after the drift point. This
    isolates sensor-degradation drift: without the fix, clean group-B clips would
    look anomalous to the group-A-trained models and cause pre-drift false alarms.
    """
    idx_a, _ = _group_indices(labels, seed)  # normal distribution = group A
    stream_feat = features[idx_a].copy()
    stream_lbl = labels[idx_a].copy()

    n = len(stream_feat)
    drift_point = int(n * drift_point_ratio)
    rng = np.random.RandomState(seed + 2)
    stream_feat[drift_point:] += rng.normal(
        0.0, noise_scale, size=stream_feat[drift_point:].shape
    ).astype(stream_feat.dtype)

    true_drifts = [drift_point]
    logger.info(
        "Noise drift (group-A normal): scale=%.2f on %d of %d samples, drift at t=%d",
        noise_scale, n - drift_point, n, drift_point,
    )
    return stream_feat, stream_lbl, drift_point, true_drifts


def stationary_stream(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    **_: Any,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """No-drift control: stream normal (group-A) clips with NO drift injected.

    Any detection on this stream is a false alarm. Used to measure the
    false-alarm rate and to confirm that detections in the real scenarios are
    not an artefact of the detector firing regardless of drift.
    """
    idx_a, _ = _group_indices(labels, seed)
    stream_feat = features[idx_a].copy()
    stream_lbl = labels[idx_a].copy()
    logger.info(
        "Stationary (no-drift) stream: %d group-A samples, no drift injected",
        len(stream_feat),
    )
    return stream_feat, stream_lbl, len(stream_feat) // 2, []  # no true drifts


# ======================================================================
# Dispatcher
# ======================================================================
SCENARIOS = {
    "abrupt": abrupt_drift,
    "gradual": gradual_drift,
    "incremental": incremental_drift,
    "recurring": recurring_drift,
    "noise": noise_drift,
    "stationary": stationary_stream,
}


def create_drift_scenario(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Create a named drift scenario. Unknown kwargs are ignored by generators.

    Returns:
        ``(stream_features, stream_labels, drift_point, true_drift_indices)``
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Choose from {list(SCENARIOS)}")
    return SCENARIOS[name](features, labels, **kwargs)