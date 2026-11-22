"""Sensitivity sweeps over ADWIN delta and the alpha/beta weighting.

For each seed the stack is trained ONCE (via experiments.pipeline.prepare_seed),
then every delta / alpha value is run cheaply by re-streaming with a different
monitor sensitivity or score weighting. Answers:
  * does a different delta make abrupt reliable without raising stationary FAs?
  * how does the alpha/beta split affect detection vs false alarms?

Requires cached features (run `python main.py` once first).

Usage:
    python scripts/run_sensitivity.py --seeds 42 7 13 101 202
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

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

DELTA_SWEEP = [0.002, 0.01, 0.05, 0.1, 0.2]
ALPHA_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]


def _load_cached(cfg):
    cache = Path(cfg["dataset"]["processed_dir"]) / "features.npz"
    if not cache.exists():
        raise SystemExit("No cached features found. Run `python main.py` once first.")
    data = FeatureExtractor.load_cached_full(cache)
    if "folds" not in data:
        raise SystemExit("Cache lacks fold metadata; re-run `python main.py`.")
    return data["features"], data["labels"], data["folds"]


def _run(comp, scen, cfg, seed, device, *, adwin_delta, alpha, beta, out_dir):
    return run_single_experiment(
        scen, comp.test_features, comp.test_labels, comp.teacher, comp.ensemble,
        alpha=alpha, beta=beta, adwin_delta=adwin_delta,
        noise_sigma=cfg["drift"]["noise_sigma"],
        drift_point_ratio=cfg["streaming"]["drift_point_ratio"],
        tolerance=comp.tolerance, normalizer=comp.normalizer, seed=seed,
        save_plots=False, output_dir=str(out_dir), device=device,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Delta / alpha sensitivity sweeps")
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 101, 202])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    features, labels, folds = _load_cached(cfg)
    scenarios = cfg["experiment"]["scenarios"]
    base_alpha = cfg["drift"]["alpha"]
    base_beta = cfg["drift"]["beta"]
    base_delta = cfg["drift"]["adwin_delta"]
    out_dir = Path(cfg["experiment"]["output_dir"]) / "sensitivity"
    ensure_dir(out_dir)
    write_run_manifest(out_dir, config=cfg, seeds=list(args.seeds),
                       extra={"delta_sweep": DELTA_SWEEP, "alpha_sweep": ALPHA_SWEEP})
    scratch = out_dir / "_scratch"

    n_scen = len(scenarios)
    total_runs = len(args.seeds) * (len(DELTA_SWEEP) + len(ALPHA_SWEEP)) * n_scen
    logger.info("Sweep plan: %d seeds x (%d delta + %d alpha) x %d scenarios = %d streaming runs",
                len(args.seeds), len(DELTA_SWEEP), len(ALPHA_SWEEP), n_scen, total_runs)
    done = 0

    # Mute the per-run analysis dumps during sweeping; progress goes to stdout
    # via print() so the console shows a single advancing counter, not a wall
    # of repeating "Drift Analysis" blocks. Restored to INFO before the tables.
    logging.getLogger("audio_drift").setLevel(logging.ERROR)

    delta_rows: list[dict] = []
    alpha_rows: list[dict] = []
    for seed in args.seeds:
        logger.info("=" * 60)
        logger.info("SEED %d (training once, then sweeping)", seed)
        logger.info("=" * 60)
        print(f"\n=== SEED {seed}: training once, then sweeping ===", flush=True)
        comp = prepare_seed(cfg, features, labels, folds, seed, device)

        # --- delta sweep (alpha/beta fixed at config) ---
        for d in DELTA_SWEEP:
            print(f"  [{done:3d}/{total_runs}] seed {seed} | delta-sweep delta={d}", flush=True)
            for scen in scenarios:
                r = _run(comp, scen, cfg, seed, device,
                         adwin_delta=d, alpha=base_alpha, beta=base_beta, out_dir=scratch)
                r.update({"seed": seed, "scenario": scen, "delta": d})
                delta_rows.append(r)
                done += 1

        # --- alpha sweep (delta fixed at config) ---
        for a in ALPHA_SWEEP:
            print(f"  [{done:3d}/{total_runs}] seed {seed} | alpha-sweep alpha={a:.2f}", flush=True)
            for scen in scenarios:
                r = _run(comp, scen, cfg, seed, device,
                         adwin_delta=base_delta, alpha=a, beta=1.0 - a, out_dir=scratch)
                r.update({"seed": seed, "scenario": scen, "alpha": a})
                alpha_rows.append(r)
                done += 1

    # --- save raw rows ---
    keys = ["seed", "scenario", "delta", "alpha", "num_detections", "true_positives",
            "detection_delay", "recall", "f1", "false_alarms", "false_alarms_per_1k"]
    with open(out_dir / "sensitivity_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in delta_rows + alpha_rows:
            w.writerow(r)

    # restore normal logging so the summary tables below are visible
    logging.getLogger("audio_drift").setLevel(logging.INFO)

    # --- aggregate + print: one table per delta, one per alpha ---
    summary: dict = {"delta": {}, "alpha": {}}
    logger.info("\n" + "=" * 78)
    logger.info("DELTA SWEEP (alpha=%.2f fixed)", base_alpha)
    logger.info("=" * 78)
    for d in DELTA_SWEEP:
        agg = aggregate_runs([r for r in delta_rows if r["delta"] == d])
        summary["delta"][str(d)] = agg
        logger.info("\n--- delta = %s ---\n%s", d, format_aggregate_table(agg))

    logger.info("\n" + "=" * 78)
    logger.info("ALPHA SWEEP (delta=%s fixed; beta = 1 - alpha)", base_delta)
    logger.info("=" * 78)
    for a in ALPHA_SWEEP:
        agg = aggregate_runs([r for r in alpha_rows if r["alpha"] == a])
        summary["alpha"][str(a)] = agg
        logger.info("\n--- alpha = %.2f (beta = %.2f) ---\n%s", a, 1.0 - a, format_aggregate_table(agg))

    with open(out_dir / "sensitivity_aggregate.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("\nSaved sensitivity CSV + aggregate JSON to %s", out_dir)


if __name__ == "__main__":
    main()