"""Shared pytest fixtures — synthetic data, no dataset/torch/network needed."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_dataset():
    """A small labelled feature matrix: 10 classes x 20 samples, 39-dim.

    Class means are separated so group-A vs group-B streams differ, mirroring
    the real 39-dim MFCC+Δ+ΔΔ layout.
    """
    rng = np.random.RandomState(0)
    n_classes, per_class, dim = 10, 20, 39
    feats, labs = [], []
    for c in range(n_classes):
        centre = rng.normal(c, 0.1, size=dim)
        feats.append(rng.normal(centre, 0.5, size=(per_class, dim)))
        labs.append(np.full(per_class, c))
    features = np.vstack(feats).astype(np.float32)
    labels = np.concatenate(labs).astype(np.int64)
    return features, labels