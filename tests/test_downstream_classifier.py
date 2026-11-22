"""Tests for models.downstream_classifier.

The downstream classifier is the frozen task model used OFFLINE to decide
whether detected drift was harmful. These tests check: it learns separable
classes, it is deterministic, it respects the label-space choice, it never reads
the streaming-test folds (leakage guard), and its macro-F1 correctly penalises a
model asked to classify clips from classes it was never trained on.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from models.downstream_classifier import (
    DownstreamClassifier,
    train_downstream_classifier,
)

TRAIN_FOLDS = [1, 2, 3, 4, 5, 6]
VAL_FOLDS = [7, 8]
TEST_FOLDS = [9, 10]


def _round_robin_folds(n: int) -> np.ndarray:
    """Assign folds 1..10 cyclically so every class lands in every partition."""
    return (np.arange(n) % 10) + 1


def _train_scaler(features: np.ndarray, folds: np.ndarray) -> StandardScaler:
    """Fit a scaler on TRAIN-fold rows only (leakage-free), mirroring real use
    where the pipeline's fitted scaler is passed in. Keeps lbfgs well-conditioned
    so the tests run in seconds and raise no convergence warning."""
    return StandardScaler().fit(features[np.isin(folds, TRAIN_FOLDS)])


def test_learns_separable_classes(synthetic_dataset):
    features, labels = synthetic_dataset
    folds = _round_robin_folds(len(labels))
    clf = train_downstream_classifier(
        features, labels, folds,
        train_folds=TRAIN_FOLDS, val_folds=VAL_FOLDS, label_space="all", seed=42,
        scaler=_train_scaler(features, folds),
    )
    assert isinstance(clf, DownstreamClassifier)
    # class means are separated in the fixture -> easy problem
    assert clf.val_macro_f1 > 0.8
    assert set(clf.classes_.tolist()) == set(range(10))


def test_deterministic_same_seed(synthetic_dataset):
    features, labels = synthetic_dataset
    folds = _round_robin_folds(len(labels))
    sc = _train_scaler(features, folds)
    a = train_downstream_classifier(features, labels, folds,
                                    train_folds=TRAIN_FOLDS, val_folds=VAL_FOLDS, seed=7, scaler=sc)
    b = train_downstream_classifier(features, labels, folds,
                                    train_folds=TRAIN_FOLDS, val_folds=VAL_FOLDS, seed=7, scaler=sc)
    assert a.C == b.C
    probe = features[:50]
    assert np.array_equal(a.predict(probe), b.predict(probe))


def test_frozen_predict_is_stable(synthetic_dataset):
    features, labels = synthetic_dataset
    folds = _round_robin_folds(len(labels))
    clf = train_downstream_classifier(features, labels, folds,
                                      train_folds=TRAIN_FOLDS, val_folds=VAL_FOLDS, seed=42,
                                      scaler=_train_scaler(features, folds))
    probe = features[10:60]
    first = clf.predict(probe)
    second = clf.predict(probe)
    assert np.array_equal(first, second)


def test_label_space_group_a_restricts_classes(synthetic_dataset):
    features, labels = synthetic_dataset
    folds = _round_robin_folds(len(labels))
    clf = train_downstream_classifier(features, labels, folds,
                                      train_folds=TRAIN_FOLDS, val_folds=VAL_FOLDS,
                                      label_space="group_a", seed=42,
                                      scaler=_train_scaler(features, folds))
    # Group A = first half of class ids (0..4)
    assert set(clf.classes_.tolist()) == {0, 1, 2, 3, 4}


def test_group_a_model_penalised_on_unseen_classes(synthetic_dataset):
    features, labels = synthetic_dataset
    folds = _round_robin_folds(len(labels))
    clf = train_downstream_classifier(features, labels, folds,
                                      train_folds=TRAIN_FOLDS, val_folds=VAL_FOLDS,
                                      label_space="group_a", seed=42,
                                      scaler=_train_scaler(features, folds))
    # Evaluate on Group-B clips (classes 5..9) it was never trained on.
    group_b = labels >= 5
    acc_b = clf.accuracy(features[group_b], labels[group_b])
    mf1_b = clf.macro_f1(features[group_b], labels[group_b])
    assert acc_b == 0.0           # cannot emit any correct Group-B label
    assert mf1_b < 0.2            # macro-F1 collapses (union-label averaging)


def test_no_test_fold_leakage(synthetic_dataset):
    """Corrupting the test-fold rows must not affect training at all —
    proving the module never reads folds 9/10."""
    features, labels = synthetic_dataset
    folds = _round_robin_folds(len(labels))
    clean = train_downstream_classifier(features, labels, folds,
                                        train_folds=TRAIN_FOLDS, val_folds=VAL_FOLDS, seed=42,
                                        scaler=_train_scaler(features, folds))

    poisoned = features.copy()
    poisoned[np.isin(folds, TEST_FOLDS)] = np.nan   # would crash sklearn if read
    # Scaler is fit on TRAIN rows only (unaffected by the poisoned test rows).
    dirty = train_downstream_classifier(poisoned, labels, folds,
                                        train_folds=TRAIN_FOLDS, val_folds=VAL_FOLDS, seed=42,
                                        scaler=_train_scaler(poisoned, folds))

    assert clean.C == dirty.C
    probe = features[:40]
    assert np.array_equal(clean.predict(probe), dirty.predict(probe))


def test_overlapping_train_val_raises(synthetic_dataset):
    features, labels = synthetic_dataset
    folds = _round_robin_folds(len(labels))
    with pytest.raises(ValueError):
        train_downstream_classifier(features, labels, folds,
                                    train_folds=[1, 2], val_folds=[2, 3], seed=42)