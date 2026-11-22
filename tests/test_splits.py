"""Tests for utils.splits: disjoint fold partitions and class-group masks."""

from __future__ import annotations

import numpy as np
import pytest

from utils.splits import split_by_folds, class_group_mask, FoldSplit


def _folds(per_fold=5):
    # folds 1..10, each repeated `per_fold` times
    return np.repeat(np.arange(1, 11), per_fold)


def test_default_split_is_disjoint_and_covers_all():
    folds = _folds()
    s = split_by_folds(folds)
    all_idx = np.concatenate([s.train_idx, s.val_idx, s.test_idx])
    # no index appears twice
    assert len(all_idx) == len(set(all_idx.tolist()))
    # default 1-6 / 7-8 / 9-10 covers every sample
    assert len(all_idx) == len(folds)
    assert s.sizes() == {"train": 30, "val": 10, "test": 10}


def test_partitions_map_to_right_folds():
    folds = _folds()
    s = split_by_folds(folds, train_folds=[1, 2], val_folds=[3], test_folds=[4])
    assert set(folds[s.train_idx]) == {1, 2}
    assert set(folds[s.val_idx]) == {3}
    assert set(folds[s.test_idx]) == {4}


def test_overlapping_folds_raise():
    folds = _folds()
    with pytest.raises(ValueError):
        split_by_folds(folds, train_folds=[1, 2], val_folds=[2, 3], test_folds=[4])


def test_train_and_test_share_no_clip():
    folds = _folds()
    s = split_by_folds(folds)
    assert set(s.train_idx.tolist()).isdisjoint(s.test_idx.tolist())
    assert set(s.train_idx.tolist()).isdisjoint(s.val_idx.tolist())


def test_class_group_mask_splits_halves():
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    a = class_group_mask(labels, half="first")
    b = class_group_mask(labels, half="second")
    assert labels[a].tolist() == [0, 1, 2, 3, 4]
    assert labels[b].tolist() == [5, 6, 7, 8, 9]
    assert not (a & b).any()  # disjoint


def test_leakage_free_intersection():
    # The real guarantee: model-training clips (group-A ∩ train folds) never
    # appear in the streaming-test partition.
    rng = np.random.RandomState(0)
    n = 200
    folds = rng.randint(1, 11, size=n)
    labels = rng.randint(0, 10, size=n)
    s = split_by_folds(folds)
    group_a = class_group_mask(labels, half="first")

    train_mask = np.zeros(n, dtype=bool)
    train_mask[s.train_idx] = True
    model_train = np.where(train_mask & group_a)[0]

    assert set(model_train.tolist()).isdisjoint(s.test_idx.tolist())