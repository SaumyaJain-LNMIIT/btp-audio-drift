"""Baseline comparison: standard detectors on the raw MFCC-mean signal.

Runs ADWIN / KSWIN / Page-Hinkley (model-free) on the SAME held-out streams the
hybrid detector sees, so the two can be compared directly. Replaces the old
report's paper-quoted comparison table with measured numbers.

Requires cached features (run `python main.py` once first).

Usage:
    python scripts/run_baselines.py --seeds 42 7 13 101 202
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.helpers import load_config, ensure_dir
from utils.logger import setup_logger
from utils.seed import set_global_seed
from utils.splits import (
    split_by_folds, class_group_mask,
    DEFAULT_TRAIN_FOLDS, DEFAULT_VAL_FOLDS, DEFAULT_TEST_FOLDS,
)
from features.feature_extractor import FeatureExtractor
from streaming.drift_scenarios import create_drift_scenario
from drift_detection.baselines import make_detector, detect_on_signal
from evaluation.drift_analysis import analyse_drift
from evaluation.aggregate import aggregate_runs, format_aggregate_table

logger = setup_logger()
BASELINES = ["adwin", "kswin", "page_hinkley"]


def _load_cached(cfg):
    cache = Path(cfg["dataset"]["processed_dir"]) / "features.npz"
    if not cache.exists():
        raise SystemExit("No cached features found. Run `python main.py` once first.")
    data = FeatureExtractor.load_cached_full(cache)
    if "folds" not in data:
        raise SystemExit("Cache lacks fold metadata; re-run `python main.py`.")
    return data["features"], data["labels"], data["folds"]


def _scaled_test_partition(cfg, features, labels, folds):
    """Reproduce the hybrid's held-out test stream input (train-fit scaling)."""
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
    return feats[fold_split.test_idx], labels[fold_split.test_idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Model-free baseline detectors")
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 101, 202])
    args = ap.parse_args()

    cfg = load_config(args.config)
    features, labels, folds = _load_cached(cfg)
    tolerance = cfg.get("eval", {}).get("tolerance", 100)
    scenarios = cfg["experiment"]["scenarios"]

    test_features, test_labels = _scaled_test_partition(cfg, features, labels, folds)
    out_dir = Path(cfg["experiment"]["output_dir"]) / "baselines"
    ensure_dir(out_dir)

    # Detector parameters. ADWIN is matched to the hybrid's delta so the only
    # difference is the SIGNAL (raw MFCC-mean vs hybrid score), not sensitivity.
    # KSWIN / Page-Hinkley use library defaults (standard out-of-the-box usage).
    detector_params = {
        "adwin": {"delta": cfg["drift"]["adwin_delta"]},
        "kswin": {},
        "page_hinkley": {},
    }

    rows: list[dict] = []
    for seed in args.seeds:
        set_global_seed(seed)
        for scen in scenarios:
            stream_feat, _, _, true_drifts = create_drift_scenario(
                scen, test_features, test_labels,
                drift_point_ratio=cfg["streaming"]["drift_point_ratio"], seed=seed,
            )
            signal = stream_feat.mean(axis=1)  # model-free: mean of MFCC vector
            for method in BASELINES:
                det = detect_on_signal(make_detector(method, **detector_params[method]), signal)
                res = analyse_drift(true_drifts, det, len(signal), tolerance=tolerance)
                res.update({"method": method, "scenario": scen, "seed": seed})
                rows.append(res)
        logger.info("Seed %d done.", seed)

    # Per-run CSV
    keys = ["seed", "method", "scenario", "num_detections", "true_positives",
            "detection_delay", "recall", "f1", "false_alarms", "false_alarms_per_1k"]
    with open(out_dir / "baselines_per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate + print one table per method (reuse the scenario aggregator)
    full_agg: dict = {}
    logger.info("\n" + "=" * 78)
    logger.info("BASELINE DETECTORS on raw MFCC-mean (model-free)")
    logger.info("=" * 78)
    for method in BASELINES:
        agg = aggregate_runs([r for r in rows if r["method"] == method])
        full_agg[method] = agg
        logger.info("\n--- %s ---\n%s", method.upper(), format_aggregate_table(agg))

    with open(out_dir / "baselines_aggregate.json", "w") as f:
        json.dump(full_agg, f, indent=2, default=str)
    logger.info("\nSaved baseline CSV + aggregate JSON to %s", out_dir)
    logger.info("Compare these against the hybrid table from run_multiseed.py.")


if __name__ == "__main__":
    main()