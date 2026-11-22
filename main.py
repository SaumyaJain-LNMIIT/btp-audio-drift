"""Main entry point — run the privacy-preserving drift-detection pipeline.

Leakage-free design (Phase 1):
    1. Features are extracted RAW (unscaled) with fold metadata.
    2. Folds are split into train / validation / streaming-test partitions.
    3. The StandardScaler is fit on the TRAIN partition only, then applied to all.
    4. Teacher + students train on group-A clips from TRAIN folds (the "normal"
       distribution the models are supposed to know).
    5. Drift experiments stream only HELD-OUT test-fold clips, so the pre-drift
       stream is never data the models were trained on.

Usage:
    python main.py
    python main.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from utils.helpers import load_config, ensure_dir, get_device
from utils.logger import setup_logger
from utils.seed import set_global_seed
from utils.reproducibility import write_run_manifest
from utils.splits import (
    split_by_folds,
    class_group_mask,
    DEFAULT_TRAIN_FOLDS,
    DEFAULT_VAL_FOLDS,
    DEFAULT_TEST_FOLDS,
)
from features.audio_loader import AudioLoader
from features.feature_extractor import FeatureExtractor
from models.ensemble_manager import EnsembleManager
from drift_detection.drift_score import DriftScoreNormalizer
from training.train_teacher import train_teacher
from training.train_students import train_students
from experiments.run_experiments import run_all_experiments
from experiments.ablation_study import run_ablation


def _load_or_extract_features(cfg, logger):
    """Return raw (unscaled) features, labels, label_names, folds.

    Re-extracts if the cache is missing, was scaled in place (legacy), or lacks
    fold metadata — because the leakage-free flow needs raw features + folds.
    """
    processed_dir = cfg["dataset"]["processed_dir"]
    cache_file = Path(processed_dir) / "features.npz"
    config_file = Path(processed_dir) / "feature_config.json"

    cache_usable = False
    if cache_file.exists():
        try:
            data = FeatureExtractor.load_cached_full(cache_file)
            scaled_in_place = False
            if config_file.exists():
                import json
                scaled_in_place = json.loads(config_file.read_text()).get("scaled_in_place", True)
            cache_usable = ("folds" in data) and (not scaled_in_place)
        except Exception as exc:
            logger.warning("Could not read cache (%s); will re-extract.", exc)

    if cache_usable:
        logger.info("Loading cached RAW features from %s", cache_file)
        return data["features"], data["labels"], data["label_names"], data["folds"]

    logger.info("=" * 60)
    logger.info("STEP 1-2 — Load dataset & extract RAW features (+ fold metadata)")
    logger.info("=" * 60)
    loader = AudioLoader(data_home=cfg["dataset"]["data_home"])
    dataset_path = Path(cfg["dataset"]["data_home"]) / "UrbanSound8K"
    if not dataset_path.exists():
        logger.info("Dataset not found. Downloading...")
        loader.download()
    # soundata needs its index JSON present; fetch it if missing.
    # download() validates existing audio and only pulls what's absent (the index).
    try:
        _ = loader.dataset.clip_ids
        logger.info("Dataset index found.")
    except Exception:
        logger.info("Dataset index missing — downloading index/metadata...")
        loader.download()
    clips = loader.load_all_clips(
        max_duration=cfg["dataset"]["max_duration"],
        sample_rate=cfg["dataset"]["sample_rate"],
    )
    fe = FeatureExtractor(
        n_mfcc=cfg["features"]["mfcc_features"],
        n_fft=cfg["features"]["n_fft"],
        hop_length=cfg["features"]["hop_length"],
        use_deltas=cfg["features"]["use_deltas"],
        normalize=cfg["features"]["normalize"],
    )
    features, labels, label_names, meta = fe.extract_batch(
        clips, save_path=processed_dir, scale=False, return_meta=True
    )
    return features, labels, label_names, meta["folds"]


def main(config_path: str = "configs/config.yaml") -> None:
    """Run the complete pipeline end-to-end."""
    cfg = load_config(config_path)
    seed = cfg["experiment"]["seed"]
    set_global_seed(seed)
    device = get_device()

    output_dir = cfg["experiment"]["output_dir"]
    ensure_dir(output_dir)
    logger = setup_logger(log_file=str(Path(output_dir) / "pipeline.log"))
    logger.info("Device: %s", device)
    logger.info("Configuration loaded from %s", config_path)

    # Reproducibility snapshot — config, seed, library versions, git commit.
    manifest_path, manifest = write_run_manifest(output_dir, config=cfg, seeds=[seed])
    logger.info("Reproducibility manifest -> %s (config_hash=%s, git=%s)",
                manifest_path, manifest["config_hash"], manifest["git_commit"])

    # ── 1-2. Features (raw) + folds ──────────────────────────────────
    features, labels, label_names, folds = _load_or_extract_features(cfg, logger)
    input_dim = features.shape[1]
    logger.info("Feature dim: %d | Samples: %d | Classes: %d",
                input_dim, len(features), len(label_names))

    # ── 3. Leakage-free split ────────────────────────────────────────
    split_cfg = cfg.get("split", {})
    fold_split = split_by_folds(
        folds,
        train_folds=split_cfg.get("train_folds", DEFAULT_TRAIN_FOLDS),
        val_folds=split_cfg.get("val_folds", DEFAULT_VAL_FOLDS),
        test_folds=split_cfg.get("test_folds", DEFAULT_TEST_FOLDS),
    )
    group_a = class_group_mask(labels, half="first")

    # Model-training set: group-A clips from TRAIN folds only ("normal").
    train_mask = np.zeros(len(labels), dtype=bool)
    train_mask[fold_split.train_idx] = True
    model_train_mask = train_mask & group_a
    logger.info("Model-training samples (group-A ∩ train folds): %d", int(model_train_mask.sum()))

    # ── Scaler: fit on model-training rows ONLY, then transform all ──
    fe = FeatureExtractor(
        n_mfcc=cfg["features"]["mfcc_features"],
        n_fft=cfg["features"]["n_fft"],
        hop_length=cfg["features"]["hop_length"],
        use_deltas=cfg["features"]["use_deltas"],
    )
    fe.fit_scaler(features[model_train_mask])
    features_scaled = fe.transform(features)
    fe.save_scaler(Path(cfg["dataset"]["processed_dir"]) / "scaler_trainfit.npz")

    train_features = features_scaled[model_train_mask]
    train_labels = labels[model_train_mask]

    # Held-out streaming-test partition (ALL classes, unseen folds).
    test_features = features_scaled[fold_split.test_idx]
    test_labels = labels[fold_split.test_idx]
    logger.info("Streaming-test samples (held-out folds, all classes): %d", len(test_features))

    # ── 3b. Train Teacher ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Pretrain teacher on %d normal samples", len(train_features))
    logger.info("=" * 60)
    teacher = train_teacher(
        features=train_features,
        input_dim=input_dim,
        hidden_dims=cfg["teacher"]["hidden_dims"],
        embedding_dim=cfg["teacher"]["embedding_dim"],
        epochs=cfg["teacher"]["epochs"],
        batch_size=cfg["teacher"]["batch_size"],
        learning_rate=cfg["teacher"]["learning_rate"],
        device=device,
    )

    # ── 4. Train Student Ensemble ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Train student ensemble (%d students)", cfg["students"]["num_students"])
    logger.info("=" * 60)
    ensemble = EnsembleManager(
        num_students=cfg["students"]["num_students"],
        input_dim=input_dim,
        hidden_dims=cfg["students"]["hidden_dims"],
        embedding_dim=cfg["students"]["embedding_dim"],
        device=device,
    )
    train_students(
        ensemble=ensemble,
        teacher=teacher,
        features=train_features,
        labels=train_labels,
        epochs=cfg["students"]["epochs"],
        batch_size=cfg["students"]["batch_size"],
        learning_rate=cfg["students"]["learning_rate"],
        device=device,
    )

    # ── 4b. Reference statistics for normalised scoring (val, group-A) ──
    normalizer = None
    tolerance = cfg.get("eval", {}).get("tolerance", 100)
    if cfg["drift"].get("normalize", True):
        val_mask = np.zeros(len(labels), dtype=bool)
        val_mask[fold_split.val_idx] = True
        ref_mask = val_mask & group_a  # normal (group-A) validation clips
        ref_feats = features_scaled[ref_mask]
        teacher.eval()
        ref_mean_losses, ref_variances = [], []
        for i in range(len(ref_feats)):
            x = torch.tensor(ref_feats[i], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                h = teacher(x)
            losses = ensemble.compute_losses(x, h)
            m, v = EnsembleManager.compute_statistics(losses)
            ref_mean_losses.append(m)
            ref_variances.append(v)
        normalizer = DriftScoreNormalizer(
            alpha=cfg["drift"]["alpha"], beta=cfg["drift"]["beta"]
        ).fit(ref_mean_losses, ref_variances)
        normalizer.save(Path(cfg["dataset"]["processed_dir"]) / "drift_normalizer.npz")
        logger.info(
            "Fitted drift-score normaliser on %d reference (val, group-A) samples",
            len(ref_feats),
        )

    # ── 5. Drift Detection Experiments (on HELD-OUT test partition) ──
    logger.info("=" * 60)
    logger.info("STEP 5 — Drift detection experiments (held-out stream)")
    logger.info("=" * 60)
    run_all_experiments(
        features=test_features,
        labels=test_labels,
        teacher=teacher,
        ensemble=ensemble,
        scenarios=cfg["experiment"]["scenarios"],
        alpha=cfg["drift"]["alpha"],
        beta=cfg["drift"]["beta"],
        adwin_delta=cfg["drift"]["adwin_delta"],
        noise_sigma=cfg["drift"]["noise_sigma"],
        drift_point_ratio=cfg["streaming"]["drift_point_ratio"],
        tolerance=tolerance,
        normalizer=normalizer,
        seed=seed,
        output_dir=output_dir,
        device=device,
    )

    # ── 6. Ablation Study (on HELD-OUT test partition) ──────────────
    logger.info("=" * 60)
    logger.info("STEP 6 — Ablation study (held-out stream)")
    logger.info("=" * 60)
    drift_scenarios = [s for s in cfg["experiment"]["scenarios"] if s != "stationary"]
    for scen in drift_scenarios:
        run_ablation(
            features=test_features,
            labels=test_labels,
            teacher=teacher,
            ensemble=ensemble,
            scenario=scen,
            adwin_delta=cfg["drift"]["adwin_delta"],
            noise_sigma=cfg["drift"]["noise_sigma"],
            drift_point_ratio=cfg["streaming"]["drift_point_ratio"],
            tolerance=tolerance,
            normalizer=normalizer,
            seed=seed,
            output_dir=output_dir,
            device=device,
            ablation_modes=cfg["experiment"]["ablation_modes"],
        )

    logger.info("Pipeline complete. Results saved to %s/", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privacy-Preserving Concept Drift Detection")
    default_config = str(Path(__file__).parent / "configs/config.yaml")
    parser.add_argument("--config", default=default_config)
    args = parser.parse_args()
    main(args.config)