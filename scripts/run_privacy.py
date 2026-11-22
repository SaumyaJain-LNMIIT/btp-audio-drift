"""Phase 6.4/6.5 — privacy-utility evaluation across epsilon, with optional
tumbling-window aggregated release.

For each seed: train once, estimate the clip constant C from the validation
reference losses (a fixed auxiliary parameter), compute the per-step sensitivity
Delta_step, and — for a tumbling window of size W — the release sensitivity
Delta_win = Delta_step / W. For each target epsilon the Gaussian noise is
calibrated via the ANALYTIC mechanism to Delta_win. Then every scenario is
streamed with per-student loss clipping; the per-step clean scores are averaged
over non-overlapping windows of W samples and noise is added to each window mean
(the only quantity released to the monitor). A window-level detection is placed
back on the sample timeline at the window's last sample.

W=1 reduces to the per-step release. Tumbling (disjoint) windows keep each clip
in exactly one release -> parallel composition, event-level DP, no cross-window
budget. The DP noise is drawn from a SEEDED rng for reproducibility and is
therefore an EVALUATION, NOT a private release. See claude/btp_dp_derivation.md.

CHECK THE eps=inf (sigma=0) ROW FIRST: it is the no-privacy baseline. With
clipping it must recover roughly the unclipped detector; if it does not, raise
--clip-mult (C is clipping away drift signal).

Usage:
    python scripts/run_privacy.py --seeds 42 7 13 101 202 --window 50 --clip-mult 3
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


def _stream_detect(comp, scen, cfg, seed, device, *, clip_c, sigma_noise, eval_rng, window):
    """Stream one scenario: clip losses, score per step, aggregate into tumbling
    windows of ``window`` samples, add DP noise to each window mean, and run the
    monitor on the noised window means. Only the noised window means reach the
    monitor (the DP release boundary). Returns drift-analysis metrics.
    """
    stream_feat, stream_lbl, _, true_drifts = create_drift_scenario(
        scen, comp.test_features, comp.test_labels,
        drift_point_ratio=cfg["streaming"]["drift_point_ratio"], seed=seed,
    )
    teacher = comp.teacher
    teacher.eval()
    enforce_eval_mode(teacher, *comp.ensemble.students, strict=True)
    alpha, beta = cfg["drift"]["alpha"], cfg["drift"]["beta"]

    # Per-step CLEAN (un-noised) scores — stay inside the trusted environment.
    clean, mls, vrs = [], [], []
    total = len(stream_feat)
    for t in range(total):
        x = torch.tensor(stream_feat[t], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            h = teacher(x)
        losses = comp.ensemble.compute_losses(x, h, clip_c=clip_c)
        m, v = EnsembleManager.compute_statistics(losses)
        clean.append(comp.normalizer.score(m, v, alpha=alpha, beta=beta, noise_sigma=0.0))
        mls.append(m); vrs.append(v)

    monitor = DriftMonitor(delta=cfg["drift"]["adwin_delta"])
    if window > 1:
        means, ends = dp.tumbling_window_means(clean, window)
        wm, _ = dp.tumbling_window_means(mls, window)
        wv, _ = dp.tumbling_window_means(vrs, window)
        noise = (dp.draw_gaussian_noise(sigma_noise, len(means), secure=False, rng=eval_rng)
                 if sigma_noise > 0 else np.zeros(len(means)))
        for j in range(len(means)):
            monitor.update(int(ends[j]), float(means[j] + noise[j]), float(wm[j]), float(wv[j]))
    else:
        noise = (dp.draw_gaussian_noise(sigma_noise, total, secure=False, rng=eval_rng)
                 if sigma_noise > 0 else np.zeros(total))
        for t in range(total):
            monitor.update(t, float(clean[t] + noise[t]), mls[t], vrs[t])

    return analyse_drift(true_drifts, monitor.detected_timestamps, total, tolerance=comp.tolerance)


def _nanmean(xs):
    arr = np.array([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Privacy-utility sweep over epsilon (windowed release)")
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 101, 202])
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--epsilons", type=float, nargs="+", default=EPS_GRID)
    ap.add_argument("--clip-percentile", type=float, default=99.0)
    ap.add_argument("--clip-mult", type=float, default=1.0,
                    help="scale C up to preserve drift signal (budget freed by windowing)")
    ap.add_argument("--window", type=int, default=1, help="tumbling window size W (1 = per-step)")
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
                              "clip_percentile": args.clip_percentile, "clip_mult": args.clip_mult,
                              "window": args.window,
                              "note": "DP noise seeded for reproducibility — evaluation, not a private release"})

    scenarios = cfg["experiment"]["scenarios"]
    drift_scens = [s for s in scenarios if s != "stationary"]
    alpha, beta = cfg["drift"]["alpha"], cfg["drift"]["beta"]
    W = max(1, args.window)

    eps_levels = list(args.epsilons) + [float("inf")]
    rows: list[dict] = []
    accounts: list[dict] = []

    logging.getLogger("audio_drift").setLevel(logging.ERROR)
    for seed in args.seeds:
        print(f"\n=== SEED {seed}: training once, then privacy sweep (W={W}) ===", flush=True)
        comp = prepare_seed(cfg, features, labels, folds, seed, device)
        if comp.normalizer is None or comp.ref_student_losses is None:
            raise SystemExit("normaliser/reference losses unavailable; set drift.normalize: true.")

        clip_c = dp.estimate_clip_c(comp.ref_student_losses, args.clip_percentile, args.clip_mult)
        k = comp.ensemble.num_students
        sd_mean, sd_var, eps0 = comp.normalizer._sd_mean, comp.normalizer._sd_var, comp.normalizer.eps
        delta_step = dp.score_sensitivity(alpha, beta, clip_c, sd_mean, sd_var, k, eps0)
        delta_win = dp.window_sensitivity(delta_step, W)
        accounts.append({"seed": seed, "clip_c": clip_c, "delta_step": delta_step,
                         "delta_win": delta_win, "window": W, "k": k})
        print(f"  C={clip_c:.4g}  Delta_step={delta_step:.4g}  Delta_win={delta_win:.4g}  (W={W}, K={k})",
              flush=True)

        for eps in eps_levels:
            sigma = 0.0 if math.isinf(eps) else dp.analytic_gaussian_sigma(delta_win, eps, args.delta)
            eval_rng = np.random.default_rng(abs(hash((seed, round(eps, 6)))) % (2**32))
            tag = "inf" if math.isinf(eps) else f"{eps:g}"
            print(f"  [seed {seed}] eps={tag:>4}  sigma={sigma:.4g}", flush=True)
            for scen in scenarios:
                r = _stream_detect(comp, scen, cfg, seed, device,
                                   clip_c=clip_c, sigma_noise=sigma, eval_rng=eval_rng, window=W)
                detected = 1.0 if r.get("true_positives", 0) > 0 else 0.0
                rows.append({"seed": seed, "scenario": scen, "epsilon": eps, "sigma": sigma,
                             "delta": args.delta, "clip_c": clip_c, "delta_win": delta_win,
                             "window": W, "detected": detected, "f1": r.get("f1"),
                             "recall": r.get("recall"), "detection_delay": r.get("detection_delay"),
                             "false_alarms_per_1k": r.get("false_alarms_per_1k")})

    logging.getLogger("audio_drift").setLevel(logging.INFO)

    keys = ["seed", "scenario", "epsilon", "sigma", "delta", "clip_c", "delta_win", "window",
            "detected", "f1", "recall", "detection_delay", "false_alarms_per_1k"]
    with open(out_dir / "privacy_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    summary = {"delta": args.delta, "window": W, "clip_mult": args.clip_mult,
               "accounts": accounts, "by_epsilon": {}}
    curve_eps, curve_det, curve_fa = [], [], []
    logger.info("\n" + "=" * 78)
    logger.info("PRIVACY-UTILITY SWEEP (delta=%.0e, W=%d; analytic Gaussian)", args.delta, W)
    logger.info("=" * 78)
    logger.info(f"{'epsilon':>8} {'drift DetRate':>14} {'drift F1*':>10} {'stationary FA/1k':>18}")
    logger.info("-" * 54)
    for eps in eps_levels:
        er = [r for r in rows if r["epsilon"] == eps]
        det = _nanmean([r["detected"] for r in er if r["scenario"] in drift_scens])
        f1 = _nanmean([r["f1"] for r in er if r["scenario"] in drift_scens])
        fa = _nanmean([r["false_alarms_per_1k"] for r in er if r["scenario"] == "stationary"])
        tag = "inf" if math.isinf(eps) else f"{eps:g}"
        summary["by_epsilon"][tag] = {"drift_detrate": det, "drift_f1": f1, "stationary_fa_per_1k": fa}
        label = "no-priv" if math.isinf(eps) else tag
        logger.info(f"{label:>8} {det:>14.2f} {f1:>10.3f} {fa:>18.3f}")
        if not math.isinf(eps):
            curve_eps.append(eps); curve_det.append(det); curve_fa.append(fa)
    logger.info("-" * 54)
    logger.info("* DetRate (fraction of drift events caught) is the honest metric; "
                "F1 is averaged over detected runs only.")
    ref = summary["by_epsilon"]["inf"]
    logger.info("Baseline check: no-privacy DetRate = %.2f (should recover the unclipped detector; "
                "if low, raise --clip-mult).", ref["drift_detrate"])

    with open(out_dir / "privacy_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- plot: detection rate and stationary FA vs epsilon ----
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xscale("log")
    ax1.plot(curve_eps, curve_det, "o-", color="#2a6", label="drift DetRate")
    ax1.axhline(ref["drift_detrate"], ls="--", color="#2a6", alpha=0.5, label="DetRate (no privacy)")
    ax1.set_xlabel("privacy budget  ε   (smaller = more private)")
    ax1.set_ylabel("drift detection rate", color="#2a6")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(curve_eps, curve_fa, "s--", color="#c33", label="stationary FA/1k")
    ax2.set_ylabel("false alarms / 1k (stationary)", color="#c33")
    ax1.set_title(f"Privacy–utility (δ={args.delta:.0e}, W={W}, analytic Gaussian)")
    ax1.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "privacy_utility.png", dpi=130)
    plt.close(fig)

    logger.info("\nSaved privacy CSV + summary JSON + curve PNG to %s", out_dir)


if __name__ == "__main__":
    main()