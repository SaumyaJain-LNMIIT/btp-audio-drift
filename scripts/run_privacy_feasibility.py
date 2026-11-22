"""Phase 6.5 — controlled DP feasibility experiment.

Asks: is there ANY configuration giving (a) a real privacy guarantee, (b)
detection within the ORIGINAL 100-observation tolerance, (c) acceptable false
alarms, (d) useful detection on the project streams? We do not relax the
tolerance or hide misses; if nothing works we report the privacy-latency
trade-off honestly.

Design decisions (held constant across all configs for fair comparison):
* ADWIN clock=1 — check every observation, so detection latency reflects true
  dynamics, not the default clock=32 check interval (which inflated windowed
  latency). main.py is unchanged (it keeps clock=32).
* Private releases use FRESH cryptographically secure noise (secrets-backed),
  so reproducibility seeds never make the privacy noise predictable. Streams
  and model checkpoints are seeded and fixed across comparable configs.
* Tumbling (non-overlapping) windows only; each clip is in exactly one window.
  A window-level detection is timestamped at the window's LAST sample index, so
  a detection in the drift's own window carries delay <= W-1 (never credited
  before the window completes). Delay is measured in original sample units.
* Analytic Gaussian mechanism; delta = 1e-5.

Configs (primary = D):
  Baseline  W=1 hybrid     no DP      - detection ceiling
  A         W=1 mean-only  no DP
  B         W=1 mean-only  DP (loss-clip)
  C         W=5 mean-only  no DP
  D         W=5 mean-only  DP (loss-clip)      <-- PRIMARY
  E         W=5 hybrid     DP (loss-clip)      <-- exploratory
  F         W=1 mean-only  DP (score-clip)     <-- pointwise alternative
DP configs also run an eps=inf (clip, no noise) reference to separate the cost
of clipping from the cost of noise.

Requires cached features and drift.normalize: true.

Usage:
    python scripts/run_privacy_feasibility.py --seeds 42 7 13 101 202
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

EPS_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]
DELTA = 1e-5
ADWIN_CLOCK = 1
SCORE_CLIP_PCT = (0.5, 99.5)  # validation score percentiles for pointwise clip


def _load_cached(cfg):
    cache = Path(cfg["dataset"]["processed_dir"]) / "features.npz"
    if not cache.exists():
        raise SystemExit("No cached features found. Run `python main.py` once first.")
    data = FeatureExtractor.load_cached_full(cache)
    if "folds" not in data:
        raise SystemExit("Cache lacks fold metadata; re-run `python main.py`.")
    return data["features"], data["labels"], data["folds"]


def _run(comp, cfg, scen, seed, device, *, alpha, beta, window, clip_mode,
         clip_c, score_lo, score_hi, sigma):
    """Stream one scenario under one configuration. Returns analysis metrics.

    clip_mode: 'none' | 'loss' (clip per-student losses to [0,C]) | 'score'
    (clip the per-step score to [score_lo, score_hi]). Noise (secure) is added
    to the released value (per step, or per window mean if window>1).
    """
    stream_feat, _, _, true_drifts = create_drift_scenario(
        scen, comp.test_features, comp.test_labels,
        drift_point_ratio=cfg["streaming"]["drift_point_ratio"], seed=seed,
    )
    teacher = comp.teacher
    teacher.eval()
    enforce_eval_mode(teacher, *comp.ensemble.students, strict=True)

    clean, mls, vrs = [], [], []
    total = len(stream_feat)
    for t in range(total):
        x = torch.tensor(stream_feat[t], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            h = teacher(x)
        losses = comp.ensemble.compute_losses(x, h, clip_c=clip_c if clip_mode == "loss" else None)
        m, v = EnsembleManager.compute_statistics(losses)
        s = comp.normalizer.score(m, v, alpha=alpha, beta=beta, noise_sigma=0.0)
        if clip_mode == "score":
            s = min(max(s, score_lo), score_hi)
        clean.append(s); mls.append(m); vrs.append(v)

    monitor = DriftMonitor(delta=cfg["drift"]["adwin_delta"], clock=ADWIN_CLOCK)
    if window > 1:
        means, ends = dp.tumbling_window_means(clean, window)
        wm, _ = dp.tumbling_window_means(mls, window)
        wv, _ = dp.tumbling_window_means(vrs, window)
        noise = dp.draw_gaussian_noise(sigma, len(means), secure=True) if sigma > 0 else np.zeros(len(means))
        for j in range(len(means)):
            monitor.update(int(ends[j]), float(means[j] + noise[j]), float(wm[j]), float(wv[j]))
    else:
        noise = dp.draw_gaussian_noise(sigma, total, secure=True) if sigma > 0 else np.zeros(total)
        for t in range(total):
            monitor.update(t, float(clean[t] + noise[t]), mls[t], vrs[t])

    res = analyse_drift(true_drifts, monitor.detected_timestamps, total, tolerance=comp.tolerance)
    res["_total"] = total  # stream length (stationary exposure / FA denominator)
    return res


def _nanmean(xs):
    arr = np.array([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="DP feasibility experiment (controlled)")
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 101, 202])
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--epsilons", type=float, nargs="+", default=EPS_GRID)
    ap.add_argument("--clip-mult", type=float, default=3.0)
    ap.add_argument("--scenarios", nargs="+", default=["abrupt", "stationary"],
                    help="add 'gradual' once a config looks promising")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not cfg["drift"].get("normalize", True):
        raise SystemExit("Feasibility experiment needs the normaliser: set drift.normalize: true.")
    device = get_device()
    features, labels, folds = _load_cached(cfg)
    out_dir = Path(cfg["experiment"]["output_dir"]) / "privacy_feasibility"
    ensure_dir(out_dir)
    write_run_manifest(out_dir, config=cfg, seeds=list(args.seeds),
                       extra={"epsilons": args.epsilons, "delta": args.delta,
                              "clip_mult": args.clip_mult, "adwin_clock": ADWIN_CLOCK,
                              "scenarios": args.scenarios,
                              "note": "private releases use FRESH secure noise (not reproducible by design)"})

    Ha, Hb = cfg["drift"]["alpha"], cfg["drift"]["beta"]
    # name, alpha, beta, window, clip_mode, dp?
    CONFIGS = [
        ("Baseline", Ha, Hb, 1, "none", False),
        ("A", 1.0, 0.0, 1, "none", False),
        ("B", 1.0, 0.0, 1, "loss", True),
        ("C", 1.0, 0.0, 5, "none", False),
        ("D", 1.0, 0.0, 5, "loss", True),   # PRIMARY
        ("E", Ha, Hb, 5, "loss", True),     # exploratory hybrid
        ("F", 1.0, 0.0, 1, "score", True),  # pointwise score clip
    ]
    drift_scens = [s for s in args.scenarios if s != "stationary"]
    rows: list[dict] = []

    logging.getLogger("audio_drift").setLevel(logging.ERROR)
    for seed in args.seeds:
        print(f"\n=== SEED {seed}: training once ===", flush=True)
        comp = prepare_seed(cfg, features, labels, folds, seed, device)
        if comp.normalizer is None:
            raise SystemExit("normaliser unavailable; set drift.normalize: true.")
        k = comp.ensemble.num_students
        sd_mean, sd_var, eps0 = comp.normalizer._sd_mean, comp.normalizer._sd_var, comp.normalizer.eps
        clip_c = dp.estimate_clip_c(comp.ref_student_losses, 99.0, args.clip_mult)

        for (name, a, b, W, clip_mode, is_dp) in CONFIGS:
            # sensitivity + score-clip bounds for this config/seed
            score_lo = score_hi = 0.0
            if clip_mode == "loss":
                delta_step = dp.score_sensitivity(a, b, clip_c, sd_mean, sd_var, k, eps0)
                sens = dp.window_sensitivity(delta_step, W)
            elif clip_mode == "score":
                val_scores = np.array([comp.normalizer.score(comp.ref_mean_losses[i], comp.ref_variances[i],
                                                             alpha=a, beta=b, noise_sigma=0.0)
                                       for i in range(len(comp.ref_mean_losses))])
                score_lo = float(np.percentile(val_scores, SCORE_CLIP_PCT[0]))
                score_hi = float(np.percentile(val_scores, SCORE_CLIP_PCT[1]))
                sens = dp.window_sensitivity(score_hi - score_lo, W)
            else:
                sens = float("nan")

            eps_levels = ([float("inf")] + list(args.epsilons)) if is_dp else [None]
            for eps in eps_levels:
                if eps is None:            # non-DP config
                    sigma = 0.0
                elif math.isinf(eps):      # clip-only reference
                    sigma = 0.0
                else:
                    sigma = dp.analytic_gaussian_sigma(sens, eps, args.delta)
                for scen in args.scenarios:
                    r = _run(comp, cfg, scen, seed, device, alpha=a, beta=b, window=W,
                             clip_mode=clip_mode, clip_c=clip_c, score_lo=score_lo,
                             score_hi=score_hi, sigma=sigma)
                    ndet = r.get("num_detections", 0)
                    tp = r.get("true_positives", 0)
                    is_drift = scen in drift_scens
                    if is_drift:
                        # hit = matched within tolerance; late = fired but none matched;
                        # missed = never fired. (Tolerance is the unchanged 100.)
                        outcome = "hit" if tp > 0 else ("late" if ndet > 0 else "missed")
                    else:
                        outcome = "false_alarm" if ndet > 0 else "clean"
                    rows.append({
                        "config": name, "alpha": a, "beta": b, "window": W,
                        "clip_mode": clip_mode, "dp": is_dp,
                        "epsilon": ("none" if eps is None else ("inf" if math.isinf(eps) else eps)),
                        "sensitivity": sens, "sigma": sigma, "seed": seed, "scenario": scen,
                        "num_detections": ndet, "true_positives": tp, "outcome": outcome,
                        "detected": 1.0 if tp > 0 else 0.0,
                        "delay": r.get("detection_delay"),       # matched-event delay only (nan if no TP)
                        "false_alarms": r.get("false_alarms", 0),
                        "false_alarms_per_1k": r.get("false_alarms_per_1k"),
                        "stream_len": r.get("_total"),
                    })
            print(f"  seed {seed} config {name} done", flush=True)

    logging.getLogger("audio_drift").setLevel(logging.INFO)

    # raw per-seed CSV (includes num_detections, outcome, raw FA count, exposure)
    keys = ["config", "alpha", "beta", "window", "clip_mode", "dp", "epsilon",
            "sensitivity", "sigma", "seed", "scenario", "num_detections", "true_positives",
            "outcome", "detected", "delay", "false_alarms", "false_alarms_per_1k", "stream_len"]
    with open(out_dir / "feasibility_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # aggregate by (config, epsilon): hit/late/missed counts, matched-only delay, raw FA + exposure
    def agg(cfg_name, eps):
        sel = [r for r in rows if r["config"] == cfg_name and r["epsilon"] == eps]
        drift_rows = [r for r in sel if r["scenario"] in drift_scens]
        stat_rows = [r for r in sel if r["scenario"] == "stationary"]
        n_drift = len(drift_rows)
        hits = sum(1 for r in drift_rows if r["outcome"] == "hit")
        late = sum(1 for r in drift_rows if r["outcome"] == "late")
        missed = sum(1 for r in drift_rows if r["outcome"] == "missed")
        det = (hits / n_drift) if n_drift else float("nan")
        delay = _nanmean([r["delay"] for r in drift_rows if r["outcome"] == "hit"])  # matched only
        fa_cnt = sum(r["false_alarms"] for r in stat_rows)
        exposure = sum(r["stream_len"] for r in stat_rows)  # total stationary observations
        sens = _nanmean([r["sensitivity"] for r in sel])
        sigma = _nanmean([r["sigma"] for r in sel])
        a = sel[0]["alpha"]; b = sel[0]["beta"]; W = sel[0]["window"]
        return dict(cfg=cfg_name, ab=f"{a:.2g}/{b:.2g}", W=W, eps=eps, sens=sens, sigma=sigma,
                    det=det, hits=hits, late=late, missed=missed, n=n_drift,
                    delay=delay, fa_cnt=fa_cnt, exposure=exposure)

    table = []
    for (name, a, b, W, clip_mode, is_dp) in CONFIGS:
        eps_levels = (["inf"] + list(args.epsilons)) if is_dp else ["none"]
        for eps in eps_levels:
            table.append(agg(name, eps))

    logger.info("\n" + "=" * 110)
    logger.info("DP FEASIBILITY — comparison (delta=%.0e, clock=%d, tol=100, secure noise, %d seeds)",
                args.delta, ADWIN_CLOCK, len(args.seeds))
    logger.info("=" * 110)
    hdr = (f"{'Method':<9}{'a/b':>9}{'W':>3}{'eps':>6}{'Delta':>9}{'sigma':>9}"
           f"{'DetRate':>9}{'hit/late/miss':>14}{'Delay*':>8}{'FAcnt':>7}{'exposure':>10}")
    logger.info(hdr)
    logger.info("-" * len(hdr))
    for t in table:
        eps = t["eps"]
        eps_s = eps if isinstance(eps, str) else f"{eps:g}"
        sens_s = "   -   " if math.isnan(t["sens"]) else f"{t['sens']:.3g}"
        delay_s = "  -  " if math.isnan(t["delay"]) else f"{t['delay']:.0f}"
        hlm = f"{t['hits']}/{t['late']}/{t['missed']}"
        logger.info(f"{t['cfg']:<9}{t['ab']:>9}{t['W']:>3}{eps_s:>6}{sens_s:>9}{t['sigma']:>9.3g}"
                    f"{t['det']:>9.2f}{hlm:>14}{delay_s:>8}{t['fa_cnt']:>7}{t['exposure']:>10}")
    logger.info("-" * len(hdr))
    logger.info("* Delay = mean over MATCHED (hit) events only. hit/late/miss out of %d drift streams; "
                "FAcnt = raw false alarms over 'exposure' stationary observations.", len(args.seeds))
    logger.info("Mean-only DP = configs B, D, F; Hybrid DP = config E. eps=inf = clip, no noise. "
                "DP-noise rows use fresh secure noise (one realization; non-DP and eps=inf rows are deterministic).")

    with open(out_dir / "feasibility_summary.json", "w") as f:
        json.dump(table, f, indent=2, default=str)
    logger.info("\nSaved feasibility CSV + summary JSON to %s", out_dir)


if __name__ == "__main__":
    main()