"""Tests for drift_detection.drift_score: raw score (legacy) and the
reference-window z-normaliser."""

from __future__ import annotations

import math

import numpy as np
import pytest

from drift_detection.drift_score import (
    compute_drift_score,
    DriftScoreNormalizer,
)


# ------------------------- legacy raw score -------------------------
def test_raw_score_is_weighted_sum():
    s = compute_drift_score(mean_loss=0.5, loss_variance=0.02, alpha=0.7, beta=0.3)
    assert math.isclose(s, 0.7 * 0.5 + 0.3 * 0.02)


# ------------------------- normaliser basics -------------------------
def test_zscore_matches_reference_stats():
    rng = np.random.RandomState(0)
    ref_mean = rng.normal(0.2, 0.05, size=500)
    ref_var = rng.normal(0.002, 0.0005, size=500)
    norm = DriftScoreNormalizer().fit(ref_mean, ref_var)

    # a value one std above the reference mean -> z ≈ 1
    one_std_above = ref_mean.mean() + ref_mean.std()
    assert abs(norm.z_mean(one_std_above) - 1.0) < 1e-3


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        DriftScoreNormalizer().score(0.3, 0.004)


def test_empty_reference_raises():
    with pytest.raises(ValueError):
        DriftScoreNormalizer().fit([], [])


def test_zero_std_is_safe():
    # constant reference -> std 0; eps must prevent division by zero
    norm = DriftScoreNormalizer(eps=1e-8).fit([0.2] * 10, [0.002] * 10)
    val = norm.score(0.2, 0.002)
    assert math.isfinite(val)


# ------------------- the core Phase-2 claim -------------------
def test_variance_now_contributes_meaningfully():
    """Under raw scoring, variance is negligible; after normalisation a
    variance spike of equal statistical size contributes comparably."""
    rng = np.random.RandomState(1)
    ref_mean = rng.normal(0.2, 0.05, size=1000)
    ref_var = rng.normal(0.002, 0.0005, size=1000)
    norm = DriftScoreNormalizer(alpha=0.7, beta=0.3).fit(ref_mean, ref_var)

    # a post-drift step: mean loss +2σ, variance +4σ
    ml = 0.2 + 2 * 0.05       # z_mean ≈ 2
    var = 0.002 + 4 * 0.0005  # z_var  ≈ 4

    # raw: variance term is a rounding error next to the mean term
    raw = compute_drift_score(ml, var, alpha=0.7, beta=0.3)
    raw_var_share = (0.3 * var) / raw
    assert raw_var_share < 0.02           # <2% of the raw score

    # normalised: variance contributes a real fraction
    zm, zv = norm.z_mean(ml), norm.z_var(var)
    norm_var_share = (0.3 * zv) / (0.7 * zm + 0.3 * zv)
    assert norm_var_share > 0.30          # >30% of the normalised score


def test_alpha_beta_override_for_ablation():
    norm = DriftScoreNormalizer(alpha=0.7, beta=0.3).fit(
        np.linspace(0.1, 0.3, 50), np.linspace(0.001, 0.003, 50)
    )
    studd_only = norm.score(0.25, 0.0025, alpha=1.0, beta=0.0)
    ens_only = norm.score(0.25, 0.0025, alpha=0.0, beta=1.0)
    assert math.isclose(studd_only, norm.z_mean(0.25))
    assert math.isclose(ens_only, norm.z_var(0.0025))


def test_save_and_load_roundtrip(tmp_path):
    norm = DriftScoreNormalizer(alpha=0.6, beta=0.4).fit(
        np.linspace(0.1, 0.3, 40), np.linspace(0.001, 0.004, 40)
    )
    ref = norm.score(0.28, 0.0035)
    path = tmp_path / "normalizer.npz"
    norm.save(path)
    loaded = DriftScoreNormalizer.load(path)
    assert math.isclose(loaded.score(0.28, 0.0035), ref, rel_tol=1e-6)