"""Drift score computation.

Two scoring modes are provided:

Raw (legacy)::

    score = α · mean_loss + β · loss_variance

The two signals live on very different scales (mean loss ~0.1-1.0, loss
variance ~0.001-0.05), so β barely contributes and the "hybrid" behaves like
STUDD-only.

Normalised (Phase 2) — :class:`DriftScoreNormalizer`::

    z_mean = (mean_loss - μ_mean) / (σ_mean + ε)
    z_var  = (variance  - μ_var)  / (σ_var  + ε)
    score  = α · z_mean + β · z_var

μ/σ are estimated **once** from a reference window (the validation partition,
i.e. normal pre-drift data — never future streaming observations). After
normalisation both signals are in comparable z-units, so the ensemble-variance
signal can actually move the score.

Optional Gaussian noise can be added to the score for a privacy flavour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ======================================================================
# Raw (legacy) score
# ======================================================================
def compute_drift_score(
    mean_loss: float,
    loss_variance: float,
    alpha: float = 0.7,
    beta: float = 0.3,
    noise_sigma: float = 0.0,
) -> float:
    """Raw weighted-sum drift score (unnormalised). Kept for comparison."""
    score = alpha * mean_loss + beta * loss_variance
    if noise_sigma > 0:
        score += np.random.normal(0, noise_sigma)
    return float(score)


# ======================================================================
# Normalised score
# ======================================================================
class DriftScoreNormalizer:
    """Z-normalise both drift signals against reference statistics, then combine.

    Fit on reference (normal) observations, then call :meth:`score` per stream
    step. Reference statistics must come from data that is not part of the
    streaming-test set (we use the validation partition).
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-8) -> None:
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self._mu_mean: float | None = None
        self._sd_mean: float | None = None
        self._mu_var: float | None = None
        self._sd_var: float | None = None

    @property
    def fitted(self) -> bool:
        return self._mu_mean is not None

    def fit(self, ref_mean_losses, ref_variances) -> "DriftScoreNormalizer":
        """Estimate μ/σ of each signal from a reference window.

        Args:
            ref_mean_losses: per-step mean-loss values over the reference data.
            ref_variances: per-step loss-variance values over the reference data.
        """
        rm = np.asarray(ref_mean_losses, dtype=float)
        rv = np.asarray(ref_variances, dtype=float)
        if rm.size == 0 or rv.size == 0:
            raise ValueError("Reference arrays must be non-empty.")
        self._mu_mean, self._sd_mean = float(rm.mean()), float(rm.std())
        self._mu_var, self._sd_var = float(rv.mean()), float(rv.std())
        return self

    def z_mean(self, mean_loss: float) -> float:
        return (mean_loss - self._mu_mean) / (self._sd_mean + self.eps)

    def z_var(self, loss_variance: float) -> float:
        return (loss_variance - self._mu_var) / (self._sd_var + self.eps)

    def score(
        self,
        mean_loss: float,
        loss_variance: float,
        alpha: float | None = None,
        beta: float | None = None,
        noise_sigma: float = 0.0,
    ) -> float:
        """Combine the two z-normalised signals into one scalar.

        ``alpha`` / ``beta`` default to the instance values but can be overridden
        per call (used by the ablation, which reuses one fitted normaliser with
        different weights).
        """
        if not self.fitted:
            raise RuntimeError("DriftScoreNormalizer not fitted; call fit() first.")
        a = self.alpha if alpha is None else alpha
        b = self.beta if beta is None else beta
        s = a * self.z_mean(mean_loss) + b * self.z_var(loss_variance)
        if noise_sigma > 0:
            s += np.random.normal(0, noise_sigma)
        return float(s)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("Nothing to save; normaliser not fitted.")
        np.savez(
            path,
            mu_mean=self._mu_mean, sd_mean=self._sd_mean,
            mu_var=self._mu_var, sd_var=self._sd_var,
            alpha=self.alpha, beta=self.beta, eps=self.eps,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DriftScoreNormalizer":
        d = np.load(path)
        obj = cls(alpha=float(d["alpha"]), beta=float(d["beta"]), eps=float(d["eps"]))
        obj._mu_mean, obj._sd_mean = float(d["mu_mean"]), float(d["sd_mean"])
        obj._mu_var, obj._sd_var = float(d["mu_var"]), float(d["sd_var"])
        return obj