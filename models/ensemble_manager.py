"""Ensemble manager — manage N student models on disjoint data subsets.

Responsibilities:
  * Split data into disjoint partitions for privacy.
  * Collective inference: compute per-student reconstruction losses.
  * Aggregate statistics: mean loss & loss variance.

Phase 6 (differential privacy):
  * Per-student losses can be clipped to [0, C] so the drift score has a
    bounded, computable sensitivity (see claude/btp_dp_derivation.md). C is a
    fixed auxiliary parameter derived from held-out validation data, outside
    the protected streaming scope.
  * ``enforce_eval_mode`` guards the frozen-model assumption: both the teacher
    and the students contain BatchNorm, which is only memoryless (and only
    DP-safe) in eval mode. The guard raises if a model is in train mode.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.student_model import StudentModel

logger = logging.getLogger("audio_drift")


def enforce_eval_mode(*modules: nn.Module, strict: bool = True) -> None:
    """Ensure modules are in eval mode (frozen BatchNorm / no state updates).

    Args:
        modules: modules to check/set.
        strict: if True, raise RuntimeError when a module is in training mode
            (catches misuse that would break memorylessness and the DP claim);
            if False, silently switch it to eval.
    """
    for m in modules:
        if m.training:
            if strict:
                raise RuntimeError(
                    f"{type(m).__name__} is in train mode; streaming inference "
                    "must run in eval mode (frozen BatchNorm) for memorylessness "
                    "and the DP guarantee."
                )
            m.eval()


class EnsembleManager:
    """Orchestrate an ensemble of student models."""

    def __init__(
        self,
        num_students: int,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 64,
        device: str = "cpu",
    ) -> None:
        self.num_students = num_students
        self.device = device
        self.students: list[StudentModel] = []

        for _ in range(num_students):
            student = StudentModel(input_dim, hidden_dims, embedding_dim).to(device)
            self.students.append(student)

        logger.info("Initialised ensemble with %d students.", num_students)

    # ------------------------------------------------------------------
    # Data splitting
    # ------------------------------------------------------------------
    @staticmethod
    def split_data(
        features: np.ndarray,
        labels: np.ndarray,
        num_splits: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split data into ``num_splits`` disjoint subsets.

        Args:
            features: ``(N, D)``
            labels: ``(N,)``
            num_splits: Number of partitions.

        Returns:
            List of ``(features_subset, labels_subset)`` tuples.
        """
        indices = np.arange(len(features))
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_splits)
        return [(features[s], labels[s]) for s in splits]

    # ------------------------------------------------------------------
    # Per-sample loss computation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_losses(
        self,
        feature_tensor: torch.Tensor,
        teacher_embedding: torch.Tensor,
        clip_c: float | None = None,
    ) -> list[float]:
        """Compute MSE loss for each student against the teacher embedding.

        Args:
            feature_tensor: ``(1, input_dim)`` or ``(input_dim,)``
            teacher_embedding: ``(1, embedding_dim)`` or ``(embedding_dim,)``
            clip_c: if set, clip each per-student loss to ``[0, clip_c]``. This
                bounds the drift-score sensitivity for the DP guarantee. ``C`` is
                a fixed auxiliary parameter from validation data (see
                ``drift_detection.dp``); it must not depend on streaming data.

        Returns:
            List of per-student MSE losses (clipped if ``clip_c`` is given).
        """
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        if teacher_embedding.dim() == 1:
            teacher_embedding = teacher_embedding.unsqueeze(0)

        feature_tensor = feature_tensor.to(self.device)
        teacher_embedding = teacher_embedding.to(self.device)

        losses: list[float] = []
        mse = nn.MSELoss()
        for student in self.students:
            # Frozen-model guard: BatchNorm must use fixed running stats, never
            # update, and never couple across the batch. Force eval here; the
            # streaming entry point asserts strict=True once before the loop.
            enforce_eval_mode(student, strict=False)
            pred = student(feature_tensor)
            loss = mse(pred, teacher_embedding).item()
            if clip_c is not None:
                loss = min(max(loss, 0.0), float(clip_c))
            losses.append(loss)
        return losses

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_statistics(losses: list[float]) -> tuple[float, float]:
        """Compute mean and (population, ddof=0) variance of per-student losses.

        The population variance (``np.var`` default) is what the DP sensitivity
        bound is derived against: for K losses in [0, C] its maximum is
        ``floor(K/2)*ceil(K/2)/K**2 * C**2``.

        Returns:
            ``(mean_loss, loss_variance)``
        """
        arr = np.array(losses)
        return float(np.mean(arr)), float(np.var(arr))