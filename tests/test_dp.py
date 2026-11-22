"""Unit tests for Phase 6.3 — DP sensitivity and analytic Gaussian calibration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from drift_detection import dp


# ---- variance bound ----
def test_population_variance_max_formula():
    assert dp.population_variance_max(5, 1.0) == pytest.approx(6 / 25)
    assert dp.population_variance_max(4, 1.0) == pytest.approx(4 / 16)
    assert dp.population_variance_max(3, 2.0) == pytest.approx(2 / 9 * 4)


def test_population_variance_max_is_achievable_upper_bound():
    # the worst-case split must not exceed the formula, for several K
    rng = np.random.default_rng(0)
    for k in (3, 4, 5, 7):
        c = 1.3
        vmax = dp.population_variance_max(k, c)
        # random loss vectors in [0, C] never exceed vmax
        for _ in range(200):
            losses = rng.uniform(0, c, size=k)
            assert float(np.var(losses)) <= vmax + 1e-9
        # the two-point split achieves it
        m = k // 2
        worst = [0.0] * (k - m) + [c] * m
        assert float(np.var(worst)) == pytest.approx(vmax)


# ---- sensitivity ----
def test_sensitivity_monotonic_in_c():
    kw = dict(alpha=0.7, beta=0.3, sigma_mean=0.1, sigma_var=0.01, k=5)
    assert dp.score_sensitivity(c=1.0, **kw) < dp.score_sensitivity(c=2.0, **kw)


def test_sensitivity_rejects_negative_weights():
    with pytest.raises(ValueError):
        dp.score_sensitivity(alpha=-0.1, beta=0.3, c=1.0, sigma_mean=0.1, sigma_var=0.01, k=5)


def test_sensitivity_depends_on_k():
    kw = dict(alpha=0.0, beta=1.0, c=1.0, sigma_mean=0.1, sigma_var=0.01)
    # pure variance term: Delta scales with Vmax(K), which differs by K
    d5 = dp.score_sensitivity(k=5, **kw)
    d4 = dp.score_sensitivity(k=4, **kw)
    assert d5 != d4


# ---- analytic Gaussian mechanism ----
def test_analytic_gaussian_roundtrip():
    sens, delta = 2.0, 1e-5
    for eps in (0.1, 0.5, 1.0, 2.0, 5.0):
        sigma = dp.analytic_gaussian_sigma(sens, eps, delta)
        # at this sigma the achieved delta is ~ the target
        assert dp._delta_of(eps, sigma, sens) == pytest.approx(delta, abs=1e-7)
        # inverting sigma recovers eps
        assert dp.epsilon_for_sigma(sens, sigma, delta) == pytest.approx(eps, rel=1e-3, abs=1e-3)


def test_more_noise_means_smaller_epsilon():
    sens, delta = 1.0, 1e-5
    e_small_sigma = dp.epsilon_for_sigma(sens, 1.0, delta)
    e_large_sigma = dp.epsilon_for_sigma(sens, 5.0, delta)
    assert e_large_sigma < e_small_sigma  # more noise -> more privacy -> smaller eps


def test_analytic_tighter_than_classical_for_small_eps():
    sens, delta, eps = 1.0, 1e-5, 0.5
    a = dp.analytic_gaussian_sigma(sens, eps, delta)
    c = dp.classical_gaussian_sigma(sens, eps, delta)
    assert a <= c + 1e-9  # analytic needs no more noise than the classical bound


# ---- clip estimation ----
def test_estimate_clip_c_percentile():
    losses = np.linspace(0, 100, 101)  # 0..100
    assert dp.estimate_clip_c(losses, percentile=99.0) == pytest.approx(99.0, abs=1.0)


def test_estimate_clip_c_empty_raises():
    with pytest.raises(ValueError):
        dp.estimate_clip_c([])


def test_estimate_clip_c_multiplier_scales():
    losses = np.linspace(0, 100, 101)
    base = dp.estimate_clip_c(losses, percentile=99.0, multiplier=1.0)
    scaled = dp.estimate_clip_c(losses, percentile=99.0, multiplier=3.0)
    assert scaled == pytest.approx(3.0 * base)


# ---- windowing (6.5) ----
def test_window_sensitivity_divides_by_w():
    assert dp.window_sensitivity(14.0, 50) == pytest.approx(14.0 / 50)
    assert dp.window_sensitivity(14.0, 1) == pytest.approx(14.0)


def test_window_sensitivity_rejects_bad_w():
    with pytest.raises(ValueError):
        dp.window_sensitivity(1.0, 0)


def test_tumbling_window_means_basic():
    vals = [1, 1, 1, 1, 2, 2, 2, 2]  # two full windows of 4
    means, ends = dp.tumbling_window_means(vals, 4)
    assert list(means) == [1.0, 2.0]
    assert list(ends) == [3, 7]


def test_tumbling_window_means_partial_tail():
    vals = [0, 0, 0, 6]  # window of 3 then a partial window of 1
    means, ends = dp.tumbling_window_means(vals, 3)
    assert list(means) == [0.0, 6.0]
    assert list(ends) == [2, 3]


def test_tumbling_window_w1_is_identity():
    vals = [0.5, 1.5, 2.5]
    means, ends = dp.tumbling_window_means(vals, 1)
    assert list(means) == [0.5, 1.5, 2.5]
    assert list(ends) == [0, 1, 2]


# ---- randomness separation ----
def test_secure_noise_shape_and_randomness():
    a = dp.draw_gaussian_noise(1.0, size=50, secure=True)
    b = dp.draw_gaussian_noise(1.0, size=50, secure=True)
    assert a.shape == (50,)
    assert not np.allclose(a, b)  # independent draws


def test_eval_noise_reproducible_with_seed():
    r1 = np.random.default_rng(42)
    r2 = np.random.default_rng(42)
    a = dp.draw_gaussian_noise(1.0, size=20, secure=False, rng=r1)
    b = dp.draw_gaussian_noise(1.0, size=20, secure=False, rng=r2)
    assert np.allclose(a, b)  # same seed -> same draws (evaluation only)


def test_eval_noise_requires_rng():
    with pytest.raises(ValueError):
        dp.draw_gaussian_noise(1.0, size=5, secure=False, rng=None)


def test_zero_sigma_zero_noise():
    assert np.all(dp.draw_gaussian_noise(0.0, size=10, secure=True) == 0.0)