"""Phase 6.4 — privacy-utility evaluation across epsilon.

For each seed: train once, estimate the clip constant C from the validation
reference losses (a fixed auxiliary parameter), compute the conservative
sensitivity Delta, and for each target epsilon calibrate the Gaussian noise via
the ANALYTIC mechanism. Then stream every scenario with per-student loss
clipping and DP noise added to the score, and record detection quality.

Only the noised score crosses into the monitor (release boundary). The DP noise
is drawn from a SEEDED rng for reproducibility and is therefore an EVALUATION of
the mechanism, NOT an actual private release (a real release uses the secure
path in drift_detection.dp). See claude/btp_dp_derivation.md.

Requires cached features and the normaliser enabled (drift.normalize: true).

Usage:
    python scripts/run_privacy.py --seeds 42 7 13 101 202
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.helpers import load_config, get_device, ensure_dir
from utils.logger import setup_logger
from utils.reproducibility import write_run_manifest
from features.feature_extractor import FeatureExtractor
from experiments.pipeline import prepare_seed
from models.ensemble_manager import EnsembleManager, enforce_eval_mode
from streaming.drift_scenarios import create_drift_scenario
from drift_detection.drift_monitor import DriftMonitor
from evaluation.drift_analysis import analyse_drift
from drift_detection import dp

logger = setup_logger()

EPS_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
DELTA = 1e-5


def _load_cached(cfg):
    cache = Path(cfg["dataset"]["processed_dir"]) / "features.npz"
    if not cache.exists():
        raise SystemExit("No cached features found. Run `python main.py` once first.")
    data = FeatureExtractor.load_cached_full(cache)
    if "folds" not in data:
        raise SystemExit("Cache lacks fold metadata; re-run `python main.py`.")
    return data["features"], data["labels"], data["folds"]


def _stream_detect(comp, scen, cfg, seed, device, *, clip_c, sigma_noise, eval_rng):
    """Stream one scenario with loss clipping + DP noise on the score.

    Returns the drift-analysis metrics dict. Only the noised score reaches the
    monitor (the DP release boundary); raw mean/var stay local.
    """
    stream_feat, stream_lbl, _, true_drifts = create_drift_scenario(
        scen, comp.test_features, comp.test_labels,
        drift_point_ratio=cfg["streaming"]["drift_point_ratio"], seed=seed,
    )
    teacher = comp.teacher
    teacher.eval()
    # Frozen-model guard (strict) once before streaming.
    enforce_eval_mode(teacher, *comp.ensemble.students, strict=True)

    alpha, beta = cfg["drift"]["alpha"], cfg["drift"]["beta"]
    monitor = DriftMonitor(delta=cfg["drift"]["adwin_delta"])
    total = len(stream_feat)
    for t in range(total):
        x = torch.tensor(stream_feat[t], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            h = teacher(x)
        losses = comp.ensemble.compute_losses(x, h, clip_c=clip_c)
        m, v = EnsembleManager.compute_statistics(losses)
        clean = comp.normalizer.score(m, v, alpha=alpha, beta=beta, noise_sigma=0.0)
        if sigma_noise > 0:
            clean += float(dp.draw_gaussian_noise(sigma_noise, 1, secure=False, rng=eval_rng)[0])
        monitor.update(t, clean, m, v)

    return analyse_drift(true_drifts, monitor.detected_timestamps, total, tolerance=comp.tolerance)


def _nanmean(xs):
    arr = np.array([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Privacy-utility sweep over epsilon")
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 101, 202])
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--epsilons", type=float, nargs="+", default=EPS_GRID)
    ap.add_argument("--clip-percentile", type=float, default=99.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not cfg["drift"].get("normalize", True):
        raise SystemExit("Privacy sweep needs the normaliser: set drift.normalize: true.")
    device = get_device()
    features, labels, folds = _load_cached(cfg)
    out_dir = Path(cfg["experiment"]["output_dir"]) / "privacy"
    ensure_dir(out_dir)
    write_run_manifest(out_dir, config=cfg, seeds=list(args.seeds),
                       extra={"epsilons": args.epsilons, "delta": args.delta,
                              "clip_percentile": args.clip_percentile,
                              "note": "DP noise seeded for reproducibility — evaluation, not a private release"})

    scenarios = cfg["experiment"]["scenarios"]
    drift_scens = [s for s in scenarios if s != "stationary"]
    alpha, beta = cfg["drift"]["alpha"], cfg["drift"]["beta"]

    # eps = inf (sigma = 0) is the no-privacy reference.
    eps_levels = list(args.epsilons) + [float("inf")]
    rows: list[dict] = []
    per_seed_accounts: list[dict] = []

    logging.getLogger("audio_drift").setLevel(logging.ERROR)
    for seed in args.seeds:
        print(f"\n=== SEED {seed}: training once, then privacy sweep ===", flush=True)
        comp = prepare_seed(cfg, features, labels, folds, seed, device)
        if comp.normalizer is None or comp.ref_student_losses is None:
            raise SystemExit("normaliser/reference losses unavailable; set drift.normalize: true.")

        # Fixed auxiliary parameters (from validation reference, not streaming):
        clip_c = dp.estimate_clip_c(comp.ref_student_losses, percentile=args.clip_percentile)
        k = comp.ensemble.num_students
        sd_mean = comp.normalizer._sd_mean
        sd_var = comp.normalizer._sd_var
        eps0 = comp.normalizer.eps
        sensitivity = dp.score_sensitivity(alpha, beta, clip_c, sd_mean, sd_var, k, eps0)
        per_seed_accounts.append({"seed": seed, "clip_c": clip_c, "sensitivity": sensitivity,
                                  "sd_mean": sd_mean, "sd_var": sd_var, "k": k})
        print(f"  C={clip_c:.4g}  Delta={sensitivity:.4g}  (sd_mean={sd_mean:.3g}, sd_var={sd_var:.3g}, K={k})",
              flush=True)

        for eps in eps_levels:
            if math.isinf(eps):
                sigma = 0.0
            else:
                sigma = dp.analytic_gaussian_sigma(sensitivity, eps, args.delta)
            # dedicated, seeded rng for the DP noise (evaluation only).
            eval_rng = np.random.default_rng(abs(hash((seed, round(eps, 6)))) % (2**32))
            tag = "inf" if math.isinf(eps) else f"{eps:g}"
            print(f"  [seed {seed}] eps={tag:>4}  sigma={sigma:.4g}", flush=True)
            for scen in scenarios:
                r = _stream_detect(comp, scen, cfg, seed, device,
                                   clip_c=clip_c, sigma_noise=sigma, eval_rng=eval_rng)
                detected = 1.0 if r.get("true_positives", 0) > 0 else 0.0
                rows.append({
                    "seed": seed, "scenario": scen, "epsilon": eps, "sigma": sigma,
                    "delta": args.delta, "clip_c": clip_c, "sensitivity": sensitivity,
                    "detected": detected, "f1": r.get("f1"), "recall": r.get("recall"),
                    "detection_delay": r.get("detection_delay"),
                    "false_alarms_per_1k": r.get("false_alarms_per_1k"),
                })

    logging.getLogger("audio_drift").setLevel(logging.INFO)

    # ---- save raw rows ----
    keys = ["seed", "scenario", "epsilon", "sigma", "delta", "clip_c", "sensitivity",
            "detected", "f1", "recall", "detection_delay", "false_alarms_per_1k"]
    with open(out_dir / "privacy_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- aggregate per epsilon ----
    summary = {"delta": args.delta, "accounts": per_seed_accounts, "by_epsilon": {}}
    curve_eps, curve_f1, curve_det, curve_fa = [], [], [], []
    logger.info("\n" + "=" * 78)
    logger.info("PRIVACY-UTILITY SWEEP (delta=%.0e; DP noise = analytic Gaussian)", args.delta)
    logger.info("=" * 78)
    logger.info(f"{'epsilon':>8} {'drift F1':>10} {'drift DetRate':>14} {'stationary FA/1k':>18}")
    logger.info("-" * 54)
    for eps in eps_levels:
        er = [r for r in rows if r["epsilon"] == eps]
        f1_drift = _nanmean([r["f1"] for r in er if r["scenario"] in drift_scens])
        det_drift = _nanmean([r["detected"] for r in er if r["scenario"] in drift_scens])
        fa_stat = _nanmean([r["false_alarms_per_1k"] for r in er if r["scenario"] == "stationary"])
        tag = "inf" if math.isinf(eps) else f"{eps:g}"
        summary["by_epsilon"][tag] = {"drift_f1": f1_drift, "drift_detrate": det_drift,
                                      "stationary_fa_per_1k": fa_stat}
        logger.info(f"{tag:>8} {f1_drift:>10.3f} {det_drift:>14.2f} {fa_stat:>18.3f}")
        if not math.isinf(eps):
            curve_eps.append(eps); curve_f1.append(f1_drift)
            curve_det.append(det_drift); curve_fa.append(fa_stat)

    ref = summary["by_epsilon"]["inf"]
    logger.info("-" * 54)
    logger.info(f"{'no-priv':>8} {ref['drift_f1']:>10.3f} {ref['drift_detrate']:>14.2f} {ref['stationary_fa_per_1k']:>18.3f}")

    with open(out_dir / "privacy_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- plot: detection F1 and stationary FA vs epsilon ----
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xscale("log")
    ax1.plot(curve_eps, curve_f1, "o-", color="#2a6", label="drift F1")
    ax1.axhline(ref["drift_f1"], ls="--", color="#2a6", alpha=0.5, label="F1 (no privacy)")
    ax1.set_xlabel("privacy budget  ε   (smaller = more private)")
    ax1.set_ylabel("mean F1 over drift scenarios", color="#2a6")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(curve_eps, curve_fa, "s--", color="#c33", label="stationary FA/1k")
    ax2.set_ylabel("false alarms / 1k (stationary)", color="#c33")
    ax1.set_title(f"Privacy–utility tradeoff (δ={args.delta:.0e}, analytic Gaussian)")
    ax1.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "privacy_utility.png", dpi=130)
    plt.close(fig)

    logger.info("\nSaved privacy CSV + summary JSON + curve PNG to %s", out_dir)


if __name__ == "__main__":
    main()