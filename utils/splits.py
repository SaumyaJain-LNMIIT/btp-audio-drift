"""Fold-based data splitting for leakage-free evaluation.

UrbanSound8K ships 10 predefined folds. We map them into three disjoint
partitions so that clips used to train the models are never streamed during
evaluation:

* **train**      — teacher, students (and later the downstream classifier).
* **validation** — reference statistics, score weights, thresholds (used from
  Phase 2 onward).
* **streaming-test** — final evaluation only; never seen in training.

The class-group split (A = first half of classes, B = second half) is a
*separate* axis handled by the drift-scenario generators; this module only
partitions by fold. ``class_group_mask`` is provided so callers can intersect
the two axes (e.g. train the "normal" model on group-A ∩ train-fold clips).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("audio_drift")

DEFAULT_TRAIN_FOLDS = [1, 2, 3, 4, 5, 6]
DEFAULT_VAL_FOLDS = [7, 8]
DEFAULT_TEST_FOLDS = [9, 10]


@dataclass
class FoldSplit:
    """Index arrays for the three partitions."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train_idx),
            "val": len(self.val_idx),
            "test": len(self.test_idx),
        }


def split_by_folds(
    folds: np.ndarray,
    train_folds: list[int] | None = None,
    val_folds: list[int] | None = None,
    test_folds: list[int] | None = None,
) -> FoldSplit:
    """Partition sample indices by their fold number.

    Args:
        folds: ``(N,)`` fold number per sample.
        train_folds / val_folds / test_folds: fold numbers per partition.
            Must be mutually disjoint. Default: 1-6 / 7-8 / 9-10.

    Raises:
        ValueError: if any fold number appears in more than one partition.

    Returns:
        A :class:`FoldSplit`. Folds not assigned to any partition are dropped
        (with a warning); empty partitions are warned about.
    """
    folds = np.asarray(folds)
    tr = set(train_folds if train_folds is not None else DEFAULT_TRAIN_FOLDS)
    va = set(val_folds if val_folds is not None else DEFAULT_VAL_FOLDS)
    te = set(test_folds if test_folds is not None else DEFAULT_TEST_FOLDS)

    overlap = (tr & va) | (tr & te) | (va & te)
    if overlap:
        raise ValueError(f"Folds assigned to multiple partitions: {sorted(overlap)}")

    present = set(int(f) for f in np.unique(folds))
    unassigned = present - (tr | va | te)
    if unassigned:
        logger.warning("Folds present but unassigned (dropped): %s", sorted(unassigned))

    split = FoldSplit(
        train_idx=np.where(np.isin(folds, list(tr)))[0],
        val_idx=np.where(np.isin(folds, list(va)))[0],
        test_idx=np.where(np.isin(folds, list(te)))[0],
    )
    for name, idx in split.sizes().items():
        if idx == 0:
            logger.warning("Partition '%s' is empty for the given fold mapping.", name)
    logger.info(
        "Fold split -> train=%d, val=%d, test=%d (folds %s / %s / %s)",
        *split.sizes().values(), sorted(tr), sorted(va), sorted(te),
    )
    return split


def class_group_mask(
    labels: np.ndarray, half: str = "first", n_groups: int = 2
) -> np.ndarray:
    """Boolean mask selecting the first or second half of the class ids.

    Args:
        labels: ``(N,)`` integer labels.
        half: ``"first"`` (group A) or ``"second"`` (group B).
        n_groups: kept for clarity; only 2 groups are used here.

    Returns:
        Boolean mask ``(N,)``.
    """
    unique = sorted(np.unique(labels))
    mid = len(unique) // 2
    classes = unique[:mid] if half == "first" else unique[mid:]
    return np.isin(labels, classes)