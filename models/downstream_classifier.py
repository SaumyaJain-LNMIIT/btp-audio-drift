"""Frozen downstream classifier for the harmful-vs-harmless drift assessment.

A simple, deterministic supervised task model standing in for the deployed
environmental-sound classifier that drift could harm. It is trained ONCE on the
training folds, its single hyper-parameter ``C`` is selected on the validation
folds by macro-F1, and it is then FROZEN — never updated during streaming.

Role in the experiment (see claude/harmful_vs_harmless_design.md):
    * The unsupervised drift detector consumes FEATURES ONLY and never sees this
      classifier, its predictions, labels, or accuracy.
    * This classifier is applied OFFLINE to the already-constructed stream to
      measure whether a detected drift actually degraded downstream performance.
    * Information only ever flows one way at ANALYSIS time (we compare the
      detector's alert time against the measured degradation time); nothing from
      here reaches the detector.

Leakage-free by construction: training reads ``train_folds`` and selection reads
``val_folds``; the streaming-test folds are never touched in this module.

This module deliberately has no torch / detector / DP imports — it is a small,
self-contained, deterministic unit (sklearn + numpy only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from utils.splits import class_group_mask

logger = logging.getLogger("audio_drift")

# Regularisation strengths searched during validation selection. Small grid on
# purpose: this is a baseline task model, not the contribution.
DEFAULT_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)


@dataclass
class DownstreamClassifier:
    """A frozen, fitted task classifier plus the validation metrics behind it.

    ``classes_`` are the labels the model can emit. ``macro_f1`` / ``accuracy``
    evaluate on arbitrary ``(X, y)``; macro-F1 is averaged over the UNION of the
    model's known classes and the labels actually present in ``y``, so a model
    that cannot emit a class present in the stream (e.g. a Group-A-only model
    scored on Group-B clips) is correctly penalised rather than flattered.
    """

    model: LogisticRegression
    classes_: np.ndarray
    C: float
    label_space: str
    seed: int
    val_macro_f1: float
    val_accuracy: float
    scaler: object = None  # the SAME fitted transform used at train time
    meta: dict = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Apply the stored scaler so inference matches the (scaled) training
        # space; without this a scaler-trained model would mis-predict raw input.
        return self.model.predict(_apply(self.scaler, X))

    def macro_f1(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y)
        preds = self.predict(X)
        labels = np.union1d(self.classes_, np.unique(y))
        return float(f1_score(y, preds, labels=labels, average="macro", zero_division=0))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(np.asarray(y), self.predict(X)))


def _apply(scaler, X: np.ndarray) -> np.ndarray:
    """Apply an optional fitted transform (anything with ``.transform``)."""
    if scaler is None:
        return np.asarray(X)
    return np.asarray(scaler.transform(X))


def _fit_one(X: np.ndarray, y: np.ndarray, C: float, seed: int, max_iter: int) -> LogisticRegression:
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=seed,
        # lbfgs is deterministic; default handles multiclass as softmax.
    )
    clf.fit(X, y)
    return clf


def train_downstream_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    train_folds,
    val_folds,
    label_space: str = "all",
    scaler=None,
    C_grid: tuple[float, ...] = DEFAULT_C_GRID,
    seed: int = 42,
    max_iter: int = 1000,
) -> DownstreamClassifier:
    """Train and freeze the downstream classifier.

    Args:
        features: ``(N, D)`` feature matrix (e.g. 39-dim MFCC+Δ+ΔΔ).
        labels: ``(N,)`` integer class labels.
        folds: ``(N,)`` UrbanSound8K fold number per sample.
        train_folds / val_folds: fold numbers for training / selection. Must be
            disjoint. The streaming-test folds are intentionally NOT a parameter
            here — this module must never read them.
        label_space: ``"all"`` (every class — recommended; lets a pure class
            shift be *potentially harmless*) or ``"group_a"`` (restrict training
            to the first half of class ids, mirroring the detector's notion of
            "normal"; then any shift to Group B is unseen).
        scaler: optional fitted transform (anything with ``.transform``) applied
            to features before fit and at predict time, so the classifier sees
            the same scaled space as the stream. ``None`` = identity.
        C_grid: regularisation strengths searched on the validation set.
        seed: determinism for the solver and selection tie-breaks.
        max_iter: solver iteration cap.

    Returns:
        A frozen :class:`DownstreamClassifier`. ``C`` is the value with the
        highest validation macro-F1 (ties broken toward stronger regularisation,
        i.e. the smaller ``C``). The model is refit on the TRAINING folds only at
        the selected ``C`` (validation is left as a clean held-out reference for
        the baseline-variability threshold computed in a later step).

    Raises:
        ValueError: if train/val folds overlap, or a partition is empty, or an
            unknown ``label_space`` is given.
    """
    features = np.asarray(features)
    labels = np.asarray(labels)
    folds = np.asarray(folds)

    tr_set, va_set = set(train_folds), set(val_folds)
    if tr_set & va_set:
        raise ValueError(f"train/val folds overlap: {sorted(tr_set & va_set)}")
    if label_space not in ("all", "group_a"):
        raise ValueError(f"unknown label_space '{label_space}' (use 'all' or 'group_a')")

    train_mask = np.isin(folds, list(tr_set))
    val_mask = np.isin(folds, list(va_set))

    if label_space == "group_a":
        keep = class_group_mask(labels, half="first")
        train_mask = train_mask & keep
        val_mask = val_mask & keep

    if train_mask.sum() == 0:
        raise ValueError("no training samples after fold/label-space filtering")
    if val_mask.sum() == 0:
        raise ValueError("no validation samples after fold/label-space filtering")

    Xtr = _apply(scaler, features[train_mask])
    ytr = labels[train_mask]
    Xva = _apply(scaler, features[val_mask])
    yva = labels[val_mask]

    eval_labels = np.union1d(np.unique(ytr), np.unique(yva))

    best = None  # (macro_f1, -C -> prefer smaller C on tie, C, model)
    for C in C_grid:
        model = _fit_one(Xtr, ytr, C, seed, max_iter)
        preds = model.predict(Xva)
        mf1 = float(f1_score(yva, preds, labels=eval_labels, average="macro", zero_division=0))
        key = (mf1, -C)
        if best is None or key > best[0]:
            best = (key, C, model)
    assert best is not None
    _, best_C, best_model = best

    val_preds = best_model.predict(Xva)
    val_mf1 = float(f1_score(yva, val_preds, labels=eval_labels, average="macro", zero_division=0))
    val_acc = float(accuracy_score(yva, val_preds))

    logger.info(
        "Downstream classifier frozen: label_space=%s, C=%g, val_macroF1=%.3f, "
        "val_acc=%.3f, n_train=%d, n_val=%d, classes=%s",
        label_space, best_C, val_mf1, val_acc, int(train_mask.sum()),
        int(val_mask.sum()), best_model.classes_.tolist(),
    )

    return DownstreamClassifier(
        model=best_model,
        classes_=np.asarray(best_model.classes_),
        C=best_C,
        label_space=label_space,
        seed=seed,
        val_macro_f1=val_mf1,
        val_accuracy=val_acc,
        scaler=scaler,
        meta={
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "C_grid": list(C_grid),
            "max_iter": max_iter,
        },
    )