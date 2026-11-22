"""Shared per-seed training pipeline.

Trains the full stack (scaler, teacher, students, reference-window normaliser)
for one seed and returns the pieces needed to run streaming experiments. Used by
both the multi-seed harness and the sensitivity sweep so they don't duplicate —
or drift apart from — the training logic in main.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from utils.seed import set_global_seed
from utils.splits import (
    split_by_folds, class_group_mask,
    DEFAULT_TRAIN_FOLDS, DEFAULT_VAL_FOLDS, DEFAULT_TEST_FOLDS,
)
from features.feature_extractor import FeatureExtractor
from models.ensemble_manager import EnsembleManager
from drift_detection.drift_score import DriftScoreNormalizer
from training.train_teacher import train_teacher
from training.train_students import train_students


@dataclass
class SeedComponents:
    teacher: object
    ensemble: EnsembleManager
    normalizer: object  # DriftScoreNormalizer | None
    test_features: np.ndarray
    test_labels: np.ndarray
    tolerance: int
    input_dim: int
    # Phase 6: per-student reconstruction losses over the validation reference
    # set — a FIXED AUXILIARY parameter source for the DP clip constant C
    # (never derived from streaming data). None if the normaliser was disabled.
    ref_student_losses: object = None


def prepare_seed(cfg, features, labels, folds, seed, device) -> SeedComponents:
    """Train scaler + teacher + students (+ normaliser) for one seed."""
    set_global_seed(seed)
    input_dim = features.shape[1]

    split_cfg = cfg.get("split", {})
    fold_split = split_by_folds(
        folds,
        split_cfg.get("train_folds", DEFAULT_TRAIN_FOLDS),
        split_cfg.get("val_folds", DEFAULT_VAL_FOLDS),
        split_cfg.get("test_folds", DEFAULT_TEST_FOLDS),
    )
    group_a = class_group_mask(labels, half="first")

    train_mask = np.zeros(len(labels), dtype=bool)
    train_mask[fold_split.train_idx] = True
    model_train_mask = train_mask & group_a

    fe = FeatureExtractor(
        n_mfcc=cfg["features"]["mfcc_features"], n_fft=cfg["features"]["n_fft"],
        hop_length=cfg["features"]["hop_length"], use_deltas=cfg["features"]["use_deltas"],
    )
    fe.fit_scaler(features[model_train_mask])
    feats = fe.transform(features)
    train_features, train_labels = feats[model_train_mask], labels[model_train_mask]
    test_features, test_labels = feats[fold_split.test_idx], labels[fold_split.test_idx]

    teacher = train_teacher(
        train_features, input_dim, cfg["teacher"]["hidden_dims"],
        cfg["teacher"]["embedding_dim"], cfg["teacher"]["epochs"],
        cfg["teacher"]["batch_size"], cfg["teacher"]["learning_rate"], device,
    )
    ensemble = EnsembleManager(
        cfg["students"]["num_students"], input_dim, cfg["students"]["hidden_dims"],
        cfg["students"]["embedding_dim"], device,
    )
    train_students(
        ensemble, teacher, train_features, train_labels, cfg["students"]["epochs"],
        cfg["students"]["batch_size"], cfg["students"]["learning_rate"], device,
    )

    normalizer = None
    ref_student_losses = None
    if cfg["drift"].get("normalize", True):
        val_mask = np.zeros(len(labels), dtype=bool)
        val_mask[fold_split.val_idx] = True
        ref = feats[val_mask & group_a]
        teacher.eval()
        mls, vrs, per_student = [], [], []
        for i in range(len(ref)):
            x = torch.tensor(ref[i], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                h = teacher(x)
            ls = ensemble.compute_losses(x, h)
            per_student.extend(ls)
            m, v = EnsembleManager.compute_statistics(ls)
            mls.append(m); vrs.append(v)
        normalizer = DriftScoreNormalizer(cfg["drift"]["alpha"], cfg["drift"]["beta"]).fit(mls, vrs)
        ref_student_losses = np.asarray(per_student, dtype=float)

    return SeedComponents(
        teacher=teacher, ensemble=ensemble, normalizer=normalizer,
        test_features=test_features, test_labels=test_labels,
        tolerance=cfg.get("eval", {}).get("tolerance", 100), input_dim=input_dim,
        ref_student_losses=ref_student_losses,
    )