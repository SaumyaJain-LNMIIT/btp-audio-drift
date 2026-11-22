"""Phase 6 — differential privacy mechanics for the drift score.

Everything here operates on the SCORE, the only quantity released across the
trust boundary (see claude/btp_dp_derivation.md). It provides:

* the K-dynamic population-variance bound,
* the conservative replace-one-clip sensitivity Delta,
* the analytic Gaussian mechanism (Balle & Wang, 2018) — exact for all eps>0,
  with the classical bound only as an eps<1 reference,
* the sigma <-> epsilon mapping used to label the privacy-utility sweep,
* randomness separation: a cryptographically secure path for real releases and
  a seeded path (flagged "evaluation only") for reproducible experiments,
* estimation of the clipping constant C from validation losses (a fixed
  auxiliary parameter, NOT derived from the protected streaming data).

No formal DP is claimed until the release boundary, randomness, and calibration
are verified end to end in the wired pipeline.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

_SYSRAND = random.SystemRandom()  # os.urandom-backed; for real releases


# ======================================================================
# Variance bound and sensitivity
# ======================================================================
def population_variance_max(k: int, c: float) -> float:
    """Max population variance (ddof=0) of K values in [0, C].

    Achieved by floor(K/2) at 0 and ceil(K/2) at C:
        floor(K/2) * ceil(K/2) / K**2 * C**2
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    return (k // 2) * ((k + 1) // 2) / (k ** 2) * (c ** 2)


def score_sensitivity(
    alpha: float,
    beta: float,
    c: float,
    sigma_mean: float,
    sigma_var: float,
    k: int,
    eps0: float = 1e-8,
) -> float:
    """Conservative replace-one-clip sensitivity of the normalised hybrid score.

        Delta <= alpha * C/(sigma_mean + eps0) + beta * Vmax(K)/(sigma_var + eps0)

    This upper-bounds the true (coupled) sensitivity, so it adds at least as
    much noise as needed — erring toward MORE privacy. alpha, beta must be >= 0,
    and C and the sigmas are fixed auxiliary parameters from validation data.
    """
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative for this bound.")
    if c <= 0:
        raise ValueError("clip constant C must be positive.")
    vmax = population_variance_max(k, c)
    return alpha * c / (sigma_mean + eps0) + beta * vmax / (sigma_var + eps0)


def estimate_clip_c(ref_losses, percentile: float = 99.0) -> float:
    """Clip constant C from reference (validation) per-student losses.

    C is a FIXED AUXILIARY PARAMETER derived from held-out validation data; it
    must never be computed from the protected streaming stream. A high
    percentile keeps almost all signal while bounding sensitivity.

    Args:
        ref_losses: array-like of per-student losses over validation data
            (any shape; flattened).
        percentile: percentile of the loss distribution to clip at.
    """
    arr = np.asarray(ref_losses, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("ref_losses is empty.")
    c = float(np.percentile(arr, percentile))
    return c if c > 0 else float(arr.max() or 1e-6)


# ======================================================================
# Analytic Gaussian mechanism (Balle & Wang, 2018)
# ======================================================================
def _Phi(x: float) -> float:
    """Standard normal CDF via erfc (stdlib only)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _delta_of(epsilon: float, sigma: float, sens: float) -> float:
    """Exact delta achieved by N(0, sigma^2) at privacy level epsilon.

        delta = Phi(s - t) - e^eps * Phi(-s - t),  s = sens/(2 sigma), t = eps sigma/sens
    Decreasing in both sigma and epsilon.
    """
    if sigma <= 0:
        return 1.0
    s = sens / (2.0 * sigma)
    t = epsilon * sigma / sens
    return _Phi(s - t) - math.exp(epsilon) * _Phi(-s - t)


def analytic_gaussian_sigma(sens: float, epsilon: float, delta: float) -> float:
    """Smallest sigma giving (epsilon, delta)-DP for L2-sensitivity ``sens``.

    Exact for all epsilon > 0 (unlike the classical closed form, which only
    holds for epsilon < 1). Found by bisection on the exact delta(sigma).
    """
    if sens <= 0:
        return 0.0
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1).")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    # delta(sigma) decreases from 1 (sigma->0) to 0 (sigma->inf).
    lo, hi = 1e-12, max(1.0, sens)
    while _delta_of(epsilon, hi, sens) > delta:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _delta_of(epsilon, mid, sens) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def epsilon_for_sigma(sens: float, sigma: float, delta: float) -> float:
    """Privacy budget epsilon achieved by N(0, sigma^2) at fixed ``delta``.

    Inverts the analytic mechanism: smallest epsilon with delta(epsilon) <= delta.
    Used to label each noise level on the privacy-utility curve with its eps.
    """
    if sens <= 0:
        return 0.0
    if sigma <= 0:
        return float("inf")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1).")
    # delta(epsilon) decreases in epsilon; already private enough at eps->0?
    if _delta_of(1e-12, sigma, sens) <= delta:
        return 0.0
    lo, hi = 0.0, 1.0
    while _delta_of(hi, sigma, sens) > delta:
        hi *= 2.0
        if hi > 1e9:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _delta_of(mid, sigma, sens) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def classical_gaussian_sigma(sens: float, epsilon: float, delta: float) -> float:
    """Classical Gaussian bound sigma = sens*sqrt(2 ln(1.25/delta))/epsilon.

    Reference ONLY — a sufficient condition valid for epsilon < 1; not exact and
    invalid for epsilon >= 1. Use :func:`analytic_gaussian_sigma` in practice.
    """
    if sens <= 0:
        return 0.0
    return sens * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


# ======================================================================
# Noise generation — randomness separation (safeguard 2)
# ======================================================================
def draw_gaussian_noise(sigma: float, size: int = 1, *, secure: bool = True, rng=None):
    """Draw Gaussian privacy noise.

    Args:
        sigma: noise standard deviation (score units).
        size: number of draws.
        secure: True -> cryptographically secure source (os.urandom via
            SystemRandom) for a REAL private release. False -> use the provided
            seeded ``rng`` for a REPRODUCIBLE EVALUATION ONLY (such a release is
            NOT private, because the seed is known).
        rng: a numpy Generator, required when ``secure`` is False.

    Returns:
        np.ndarray of shape (size,).
    """
    if sigma <= 0:
        return np.zeros(size)
    if secure:
        return np.array([_SYSRAND.gauss(0.0, sigma) for _ in range(size)])
    if rng is None:
        raise ValueError("evaluation mode (secure=False) requires a seeded rng.")
    return rng.normal(0.0, sigma, size)


@dataclass
class PrivacyAccount:
    """Bundle of the privacy parameters for one release configuration."""
    epsilon: float
    delta: float
    sigma: float
    sensitivity: float
    clip_c: float
    num_students: int

    def summary(self) -> str:  # pragma: no cover - formatting only
        return (f"(eps={self.epsilon:.3g}, delta={self.delta:.1e}) "
                f"sigma={self.sigma:.4g} Delta={self.sensitivity:.4g} "
                f"C={self.clip_c:.4g} K={self.num_students}")