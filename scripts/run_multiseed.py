"""Multi-seed evaluation harness.

Retrains the FULL pipeline (scaler, teacher, students, normaliser) for each seed
and aggregates per-scenario metrics as mean ± std. This is what shows the
seed-42 results are robust rather than a fluke.

Requires features to be cached first — run `python main.py` once so that
`data/processed_features/features.npz` (with fold metadata) exists.

Usage:
    python scripts/run_multiseed.py --seeds 42 7 13 101 202
    python scripts/run_multiseed.py --config configs/config.yaml --seeds 42 7 13
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.helpers import load_config, get_device, ensure_dir
from utils.logger import setup_logger
from features.feature_extractor import FeatureExtractor
from experiments.pipeline import prepare_seed
from experiments.run_experiments import run_single_experiment
from evaluation.aggregate import aggregate_runs, format_aggregate_table
from utils.reproducibility import write_run_manifest

logger = setup_logger()


def _load_cached(cfg):
    cache = Path(cfg["dataset"]["processed_dir"]) / "features.npz"
    if not cache.exists():
        raise SystemExit("No cached features found. Run `python main.py` once first.")
    data = FeatureExtractor.load_cached_full(cache)
    if "folds" not in data:
        raise SystemExit("Cache lacks fold metadata; re-run `python main.py` to re-extract.")
    return data["features"], data["labels"], data["folds"]


def run_one_seed(cfg, features, labels, folds, seed, device, out_dir) -> list[dict]:
    """Train + evaluate the whole pipeline for one seed; return scenario rows."""
    comp = prepare_seed(cfg, features, labels, folds, seed, device)
    rows: list[dict] = []
    for scen in cfg["experiment"]["scenarios"]:
        res = run_single_experiment(
            scen, comp.test_features, comp.test_labels, comp.teacher, comp.ensemble,
            alpha=cfg["drift"]["alpha"], beta=cfg["drift"]["beta"],
            adwin_delta=cfg["drift"]["adwin_delta"], noise_sigma=cfg["drift"]["noise_sigma"],
            drift_point_ratio=cfg["streaming"]["drift_point_ratio"],
            tolerance=comp.tolerance, normalizer=comp.normalizer, seed=seed,
            output_dir=str(Path(out_dir) / f"seed_{seed}"), device=device,
        )
        res["seed"] = seed
        rows.append(res)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed drift-detection evaluation")
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 101, 202])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    features, labels, folds = _load_cached(cfg)

    out_dir = Path(cfg["experiment"]["output_dir"]) / "multiseed"
    ensure_dir(out_dir)
    write_run_manifest(out_dir, config=cfg, seeds=list(args.seeds))

    all_rows: list[dict] = []
    for seed in args.seeds:
        logger.info("=" * 60)
        logger.info("SEED %d", seed)
        logger.info("=" * 60)
        all_rows.extend(run_one_seed(cfg, features, labels, folds, seed, device, out_dir))

    # Per-seed CSV
    keys = ["seed", "scenario", "num_detections", "true_positives", "detection_delay",
            "recall", "f1", "false_alarms", "false_alarms_per_1k"]
    with open(out_dir / "per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    agg = aggregate_runs(all_rows)
    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)

    logger.info("\n" + format_aggregate_table(agg))
    logger.info("Saved per-seed CSV + aggregate JSON to %s", out_dir)


if __name__ == "__main__":
    main()