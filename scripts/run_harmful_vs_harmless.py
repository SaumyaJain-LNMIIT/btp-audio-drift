"""Phase 5 — Harmful vs Harmless drift assessment.

Does the UNSUPERVISED drift detector fire on drift that MATERIALLY degrades a
downstream environmental-sound classifier, and does it fire BEFORE the
degradation? See claude/harmful_vs_harmless_design.md.

Two strictly separated passes over the same, seed-fixed stream:
  1. DETECTOR pass  — features only (no labels); yields alert times.
  2. OFFLINE pass   — the frozen classifier reads ground-truth labels and scores
     the stream in windows; a PRE-REGISTERED rule locates the degradation onset.
Information only flows one way at analysis time (compare t_D vs t_A); nothing
from the offline pass reaches the detector.

Pre-registered harm rule (constants fixed on the CLI BEFORE running):
  degraded window : P_w < P_base - max(k*sigma_base, tau_abs)
  harmful         : >= S consecutive degraded windows  -> t_A = run start
Baseline P_base/sigma_base come from the validation folds (Group A = "normal").

Verdict metric is ACCURACY by default: once the stream shifts to Group B a
window holds only a subset of classes, which deflates macro-F1 by averaging over
absent classes. Macro-F1 is still logged per window for reference.

ADWIN clock defaults to 1 (check every observation) so lead/lag reflects true
dynamics rather than the default clock=32 check interval; main.py is unchanged.

Requires cached features (`python main.py` once) and drift.normalize: true.

Usage:
    python scripts/run_harmful_vs_harmless.py --seeds 42 7 13 101 202
    python scripts/run_harmful_vs_harmless.py --noise-scales 0.5 2.0 --window 100
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
from utils.splits import (
    split_by_folds, class_group_mask,
    DEFAULT_TRAIN_FOLDS, DEFAULT_VAL_FOLDS, DEFAULT_TEST_FOLDS,
)
from features.feature_extractor import FeatureExtractor
from experiments.pipeline import prepare_seed
from models.ensemble_manager import EnsembleManager, enforce_eval_mode
from models.downstream_classifier import train_downstream_classifier
from streaming.drift_scenarios import create_drift_scenario
from drift_detection.drift_monitor import DriftMonitor
from evaluation.drift_analysis import analyse_drift
from evaluation.downstream_eval import (
    baseline_performance, windowed_performance, assess_stream,
)
from evaluation.harm_timing import first_alert_at_or_after, classify_outcome

logger = setup_logger()

ADWIN_CLOCK = 1


def _load_cached(cfg):
    cache = Path(cfg["dataset"]["processed_dir"]) / "features.npz"
    if not cache.exists():
        raise SystemExit("No cached features found. Run `python main.py` once first.")
    data = FeatureExtractor.load_cached_full(cache)
    if "folds" not in data:
        raise SystemExit("Cache lacks fold metadata; re-run `python main.py`.")
    return data["features"], data["labels"], data["folds"]


def _scaled_feats(cfg, features, labels, folds):
    """Rebuild the detector's scaler (fit on train-folds ∩ Group A) and return
    the fully scaled feature matrix, matching comp.test_features exactly so the
    classifier and the stream live in the same space. Reads no test-fold stats."""
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

    val_mask = np.zeros(len(labels), dtype=bool)
    val_mask[fold_split.val_idx] = True
    return feats, (val_mask & group_a)


def _onset(true_drifts):
    """Ground-truth drift onset (sample index) or None (stationary)."""
    if not true_drifts:
        return None
    d0 = true_drifts[0]
    return int(d0[0]) if isinstance(d0, (tuple, list)) else int(d0)


def _detect(comp, cfg, stream_feat, device, clock):
    """DETECTOR pass — features only. Returns (detections, analyse_drift dict)."""
    teacher = comp.teacher
    teacher.eval()
    enforce_eval_mode(teacher, *comp.ensemble.students, strict=True)
    alpha, beta = cfg["drift"]["alpha"], cfg["drift"]["beta"]
    monitor = DriftMonitor(delta=cfg["drift"]["adwin_delta"], clock=clock)
    total = len(stream_feat)
    for t in range(total):
        x = torch.tensor(stream_feat[t], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            h = teacher(x)
        losses = comp.ensemble.compute_losses(x, h)
        m, v = EnsembleManager.compute_statistics(losses)
        s = comp.normalizer.score(m, v, alpha=alpha, beta=beta, noise_sigma=0.0)
        monitor.update(t, s, m, v)
    return list(monitor.detected_timestamps), total


def _expand_scenarios(scenarios, noise_scales):
    """Turn 'noise' into one entry per requested scale; keep others as-is."""
    out = []
    for s in scenarios:
        if s == "noise":
            for ns in noise_scales:
                out.append(("noise", ns, f"noise_s{ns:g}"))
        else:
            out.append((s, None, s))
    return out


def _plot(out_dir, seed, label, prim, ws_prim, p_base, hard, onset, t_D, t_A):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = [w.end for w in ws_prim]
    ys = [w.value for w in ws_prim]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(xs, ys, marker="o", ms=3, lw=1, label=prim)
    ax.axhline(p_base, color="green", ls="--", lw=1, label="P_base")
    ax.axhline(hard, color="red", ls=":", lw=1, label="hard threshold")
    if onset is not None:
        ax.axvline(onset, color="black", lw=1, label="true drift onset")
    if t_D is not None:
        ax.axvline(t_D, color="blue", lw=1.2, label="t_D (alert)")
    if t_A is not None:
        ax.axvline(t_A, color="red", lw=1.2, label="t_A (degradation)")
    ax.set_title(f"seed {seed} — {label}")
    ax.set_xlabel("stream sample"); ax.set_ylabel(prim)
    ax.legend(fontsize=7, ncol=3); fig.tight_layout()
    fig.savefig(out_dir / f"hvh_{label}_seed{seed}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Harmful vs harmless drift assessment")
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 101, 202])
    ap.add_argument("--scenarios", nargs="+", default=["stationary", "abrupt", "noise"])
    ap.add_argument("--noise-scales", type=float, nargs="+", default=[0.5, 2.0],
                    help="PROVISIONAL low/high; calibrate on validation, do not assume harmless")
    ap.add_argument("--label-space", choices=["all", "group_a"], default="all")
    ap.add_argument("--metric", choices=["accuracy", "macro_f1"], default="accuracy",
                    help="verdict metric; accuracy avoids macro-F1's absent-class deflation")
    ap.add_argument("--window", type=int, default=100, help="W_eval, offline window size")
    ap.add_argument("--stride", type=int, default=None, help="window stride (default = tumbling)")
    ap.add_argument("--k", type=float, default=3.0, help="sigma multiplier for harm band")
    ap.add_argument("--tau", type=float, default=0.10, help="absolute min drop (material)")
    ap.add_argument("--s-sustained", type=int, default=3, help="consecutive degraded windows")
    ap.add_argument("--adwin-clock", type=int, default=ADWIN_CLOCK)
    ap.add_argument("--plot-seed", type=int, default=None, help="seed to plot (default: first)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not cfg["drift"].get("normalize", True):
        raise SystemExit("Needs the normaliser: set drift.normalize: true.")
    device = get_device()
    features, labels, folds = _load_cached(cfg)
    split_cfg = cfg.get("split", {})
    train_folds = split_cfg.get("train_folds", DEFAULT_TRAIN_FOLDS)
    val_folds = split_cfg.get("val_folds", DEFAULT_VAL_FOLDS)

    out_dir = Path(cfg["experiment"]["output_dir"]) / "harmful_vs_harmless"
    ensure_dir(out_dir)
    secondary = "macro_f1" if args.metric == "accuracy" else "accuracy"
    plot_seed = args.plot_seed if args.plot_seed is not None else args.seeds[0]

    write_run_manifest(out_dir, config=cfg, seeds=list(args.seeds), extra={
        "label_space": args.label_space, "verdict_metric": args.metric,
        "window": args.window, "stride": args.stride, "k": args.k, "tau_abs": args.tau,
        "s_sustained": args.s_sustained, "adwin_clock": args.adwin_clock,
        "noise_scales": args.noise_scales, "scenarios": args.scenarios,
        "note": "thresholds pre-registered here before any scenario was scored",
    })

    scen_plan = _expand_scenarios(args.scenarios, args.noise_scales)
    drift_labels = {lbl for (nm, _, lbl) in scen_plan if nm != "stationary"}
    window_rows, event_rows = [], []

    logging.getLogger("audio_drift").setLevel(logging.ERROR)
    for seed in args.seeds:
        print(f"\n=== SEED {seed}: training detector + downstream classifier ===", flush=True)
        comp = prepare_seed(cfg, features, labels, folds, seed, device)
        if comp.normalizer is None:
            raise SystemExit("normaliser unavailable; set drift.normalize: true.")

        feats_scaled, val_a_mask = _scaled_feats(cfg, features, labels, folds)
        clf = train_downstream_classifier(
            feats_scaled, labels, folds,
            train_folds=train_folds, val_folds=val_folds,
            label_space=args.label_space, scaler=None, seed=seed,
        )
        base = baseline_performance(
            clf, feats_scaled[val_a_mask], labels[val_a_mask],
            window=args.window, metric=args.metric,
        )
        p_base, sigma_base = base["p_base"], base["sigma_base"]
        print(f"  baseline {args.metric}: P_base={p_base:.3f} sigma={sigma_base:.3f} "
              f"(n_windows={base['n_windows']}, val macroF1={clf.val_macro_f1:.3f})", flush=True)

        for (name, ns, label) in scen_plan:
            kwargs = dict(drift_point_ratio=cfg["streaming"]["drift_point_ratio"], seed=seed)
            if ns is not None:
                kwargs["noise_scale"] = ns
            stream_feat, stream_lbl, _, true_drifts = create_drift_scenario(
                name, comp.test_features, comp.test_labels, **kwargs)
            onset = _onset(true_drifts)

            # ---- pass 1: detector (features only) ----
            detections, total = _detect(comp, cfg, stream_feat, device, args.adwin_clock)
            det_metrics = analyse_drift(true_drifts, detections, total, tolerance=comp.tolerance)
            t_D = first_alert_at_or_after(detections, onset)

            # ---- pass 2: offline downstream performance (labels) ----
            ws = windowed_performance(clf, stream_feat, stream_lbl,
                                      window=args.window, metric=args.metric, stride=args.stride)
            ws_sec = windowed_performance(clf, stream_feat, stream_lbl,
                                          window=args.window, metric=secondary, stride=args.stride)
            assessment = assess_stream(ws, p_base=p_base, sigma_base=sigma_base,
                                       metric=args.metric, k=args.k, tau_abs=args.tau,
                                       s_sustained=args.s_sustained)
            t_A = assessment.t_A
            bucket, lead = classify_outcome(t_D, t_A)

            for w, w2 in zip(ws, ws_sec):
                window_rows.append({
                    "seed": seed, "scenario": label, "window_start": w.start,
                    "window_end": w.end, "count": w.count,
                    args.metric: round(w.value, 6), secondary: round(w2.value, 6),
                    "drop": round(w.drop, 6), "below_soft": int(w.below_soft),
                    "below_hard": int(w.below_hard),
                })
            event_rows.append({
                "seed": seed, "scenario": label,
                "detected": int(t_D is not None), "harmful": int(t_A is not None),
                "verdict": assessment.verdict, "bucket": bucket,
                "t_D": ("" if t_D is None else t_D), "t_A": ("" if t_A is None else t_A),
                "lead_time": ("" if lead is None else lead),
                "onset": ("" if onset is None else onset),
                "perf_drop": round(assessment.max_drop, 6),
                "p_base": round(p_base, 6), "sigma_base": round(sigma_base, 6),
                "hard_threshold": round(assessment.hard_threshold, 6),
                "num_detections": det_metrics.get("num_detections", 0),
                "false_alarms": det_metrics.get("false_alarms", 0),
                "stream_len": total, "n_windows": len(ws),
                "reason": assessment.reason,
            })
            print(f"  {label:<12} verdict={assessment.verdict:<12} bucket={bucket:<18} "
                  f"t_D={t_D} t_A={t_A} lead={lead}", flush=True)

            if seed == plot_seed:
                _plot(out_dir, seed, label, args.metric, ws, p_base,
                      assessment.hard_threshold, onset, t_D, t_A)

    logging.getLogger("audio_drift").setLevel(logging.INFO)

    # ---- write outputs ----
    wkeys = ["seed", "scenario", "window_start", "window_end", "count",
             args.metric, secondary, "drop", "below_soft", "below_hard"]
    with open(out_dir / "windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=wkeys, extrasaction="ignore")
        w.writeheader(); w.writerows(window_rows)

    ekeys = ["seed", "scenario", "detected", "harmful", "verdict", "bucket",
             "t_D", "t_A", "lead_time", "onset", "perf_drop", "p_base", "sigma_base",
             "hard_threshold", "num_detections", "false_alarms", "stream_len",
             "n_windows", "reason"]
    with open(out_dir / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ekeys, extrasaction="ignore")
        w.writeheader(); w.writerows(event_rows)

    # ---- aggregate: per-scenario contingency + lead stats ----
    def _agg(label):
        rows = [r for r in event_rows if r["scenario"] == label]
        n = len(rows)
        harmful = sum(r["harmful"] for r in rows)
        detected = sum(r["detected"] for r in rows)
        buckets = {}
        for r in rows:
            buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
        leads = [r["lead_time"] for r in rows if r["lead_time"] != ""]
        lead_mean = float(np.mean(leads)) if leads else float("nan")
        return dict(scenario=label, n=n, detected=detected, harmful=harmful,
                    buckets=buckets, lead_mean=lead_mean,
                    is_drift=(label in drift_labels))

    summary = [_agg(lbl) for (_, _, lbl) in scen_plan]
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"params": vars(args), "per_scenario": summary}, f, indent=2, default=str)

    logger.info("\n" + "=" * 92)
    logger.info("HARMFUL vs HARMLESS — %d seeds, metric=%s, clock=%d, k=%.1f tau=%.2f S=%d, W=%d",
                len(args.seeds), args.metric, args.adwin_clock, args.k, args.tau,
                args.s_sustained, args.window)
    logger.info("=" * 92)
    hdr = f"{'scenario':<14}{'n':>3}{'detected':>9}{'harmful':>8}{'lead_mean':>10}  buckets"
    logger.info(hdr); logger.info("-" * len(hdr))
    for s in summary:
        lm = "   -   " if math.isnan(s["lead_mean"]) else f"{s['lead_mean']:.0f}"
        logger.info(f"{s['scenario']:<14}{s['n']:>3}{s['detected']:>9}{s['harmful']:>8}"
                    f"{lm:>10}  {s['buckets']}")
    logger.info("-" * len(hdr))
    logger.info("buckets: early_alert / late_alert = detected & harmful (lead>0 / <=0); "
                "missed_degradation = harm but no alert; detected_no_harm = alert, no harm; clean = neither.")
    logger.info("Saved windows.csv, events.csv, summary.json (+ plots) to %s", out_dir)


if __name__ == "__main__":
    main()