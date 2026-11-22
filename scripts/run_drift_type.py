"""Phase 4.2 — validate drift-type identification on real streams.

For each seed the stack is trained once (via prepare_seed), then each labelled
scenario is streamed, the drift-type classifier is applied at the detection,
and the predicted type is compared against the known true type. Produces a
confusion matrix (predicted vs true) aggregated over seeds, plus a per-run CSV
of the shape features so the thresholds can be inspected / tuned on real data.

The type-ID evaluation covers the four scenarios with a clean ground-truth
drift *type*: abrupt, gradual, incremental, recurring. `noise` (feature
corruption) and `stationary` (no drift) are reported separately as a sanity
check — they are not drift *types*, so they are not scored in the matrix.

Requires cached features (run `python main.py` once first).

Usage:
    python scripts/run_drift_type.py --seeds 42 7 13 101 202
"""

from __future__ import annotations

import argparse
import csv
import json
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
from drift_detection.drift_type import (
    classify_from_monitor, DRIFT_TYPES, COARSE_TYPES, coarse_type,
)
from utils.reproducibility import write_run_manifest

logger = setup_logger()

# Scenarios with a clean ground-truth drift TYPE (scored in the matrix).
TYPE_SCENARIOS = {
    "abrupt": "abrupt",
    "gradual": "gradual",
    "incremental": "incremental",
    "recurring": "recurring",
}
# Reported for sanity but not scored (not drift types).
SANITY_SCENARIOS = ["noise", "stationary"]

# Column order for the confusion matrix.
PRED_LABELS = ["abrupt", "gradual", "incremental", "recurring", "unknown"]


def _load_cached(cfg):
    cache = Path(cfg["dataset"]["processed_dir"]) / "features.npz"
    if not cache.exists():
        raise SystemExit("No cached features found. Run `python main.py` once first.")
    data = FeatureExtractor.load_cached_full(cache)
    if "folds" not in data:
        raise SystemExit("Cache lacks fold metadata; re-run `python main.py`.")
    return data["features"], data["labels"], data["folds"]


def _stream_and_classify(comp, scen, cfg, seed, device, scratch, cls_kwargs):
    """Run one scenario, return (predicted_type, features, n_detections)."""
    _, monitor = run_single_experiment(
        scen, comp.test_features, comp.test_labels, comp.teacher, comp.ensemble,
        alpha=cfg["drift"]["alpha"], beta=cfg["drift"]["beta"],
        adwin_delta=cfg["drift"]["adwin_delta"],
        noise_sigma=cfg["drift"]["noise_sigma"],
        drift_point_ratio=cfg["streaming"]["drift_point_ratio"],
        tolerance=comp.tolerance, normalizer=comp.normalizer, seed=seed,
        save_plots=False, return_monitor=True, output_dir=str(scratch), device=device,
    )
    res = classify_from_monitor(monitor, **cls_kwargs)
    return res.label, res.features, len(monitor.drift_events)


def main() -> None:
    ap = argparse.ArgumentParser(description="Drift-type identification validation")
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 101, 202])
    ap.add_argument("--baseline-frac", type=float, default=0.15)
    ap.add_argument("--smooth-frac", type=float, default=0.03)
    ap.add_argument("--gate-ratio", type=float, default=3.0)
    ap.add_argument("--recur-end", type=float, default=0.30)
    ap.add_argument("--abrupt-max-frac", type=float, default=0.05)
    ap.add_argument("--incr-min-frac", type=float, default=0.20)
    args = ap.parse_args()

    cls_kwargs = dict(
        baseline_frac=args.baseline_frac,
        smooth_frac=args.smooth_frac,
        gate_ratio=args.gate_ratio,
        recur_end=args.recur_end,
        abrupt_max_frac=args.abrupt_max_frac,
        incr_min_frac=args.incr_min_frac,
    )

    cfg = load_config(args.config)
    device = get_device()
    features, labels, folds = _load_cached(cfg)
    out_dir = Path(cfg["experiment"]["output_dir"]) / "drift_type"
    ensure_dir(out_dir)
    write_run_manifest(out_dir, config=cfg, seeds=list(args.seeds), extra={"classifier": cls_kwargs})
    scratch = out_dir / "_scratch"

    # confusion[true][pred] counts; per-run rows for the CSV
    confusion = {t: {p: 0 for p in PRED_LABELS} for t in TYPE_SCENARIOS.values()}
    sanity: dict[str, dict[str, int]] = {s: {p: 0 for p in PRED_LABELS} for s in SANITY_SCENARIOS}
    rows: list[dict] = []

    all_scen = list(TYPE_SCENARIOS) + SANITY_SCENARIOS
    total = len(args.seeds) * len(all_scen)
    done = 0
    for seed in args.seeds:
        print(f"\n=== SEED {seed}: training once, then classifying ===", flush=True)
        comp = prepare_seed(cfg, features, labels, folds, seed, device)
        for scen in all_scen:
            pred, feats, n_det = _stream_and_classify(comp, scen, cfg, seed, device, scratch, cls_kwargs)
            done += 1
            true_type = TYPE_SCENARIOS.get(scen)
            print(f"  [{done:2d}/{total}] seed {seed} | {scen:12s} true={true_type or '-':11s} "
                  f"pred={pred:11s} (dets={n_det})", flush=True)
            if true_type is not None:
                confusion[true_type][pred] += 1
            else:
                sanity[scen][pred] += 1
            rows.append({
                "seed": seed, "scenario": scen, "true_type": true_type or "",
                "pred_type": pred, "n_detections": n_det,
                **{f"feat_{k}": v for k, v in feats.items()},
            })

    # ---- save per-run rows ----
    field_order = ["seed", "scenario", "true_type", "pred_type", "n_detections"]
    extra_fields = sorted({k for r in rows for k in r if k not in field_order})
    with open(out_dir / "drift_type_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order + extra_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- per-type accuracy + overall ----
    per_type_acc = {}
    correct = total_scored = 0
    for t, preds in confusion.items():
        n = sum(preds.values())
        hits = preds.get(t, 0)
        per_type_acc[t] = hits / n if n else float("nan")
        correct += hits
        total_scored += n
    overall = correct / total_scored if total_scored else float("nan")

    # ---- print confusion matrix ----
    logger.info("\n" + "=" * 72)
    logger.info("DRIFT-TYPE CONFUSION MATRIX  (rows = true, cols = predicted)")
    logger.info("=" * 72)
    header = f"{'true \\ pred':<14}" + "".join(f"{p:>12}" for p in PRED_LABELS)
    logger.info(header)
    logger.info("-" * len(header))
    for t in TYPE_SCENARIOS.values():
        line = f"{t:<14}" + "".join(f"{confusion[t][p]:>12}" for p in PRED_LABELS)
        logger.info(line + f"   acc={per_type_acc[t]:.2f}")
    logger.info("-" * len(header))
    logger.info("OVERALL type-ID accuracy: %.2f  (%d/%d)", overall, correct, total_scored)

    logger.info("\nSanity (not scored):")
    for s in SANITY_SCENARIOS:
        dist = ", ".join(f"{p}={sanity[s][p]}" for p in PRED_LABELS if sanity[s][p])
        logger.info("  %-12s -> %s", s, dist or "(none)")

    # ---- coarse 3-way matrix (the reliably-separable taxonomy) ----
    coarse_conf = {t: {p: 0 for p in COARSE_TYPES} for t in COARSE_TYPES}
    for r in rows:
        if not r["true_type"]:
            continue
        ct = coarse_type(r["true_type"])
        cp = coarse_type(r["pred_type"])
        coarse_conf[ct][cp] += 1
    c_correct = sum(coarse_conf[t][t] for t in COARSE_TYPES)
    c_total = sum(sum(coarse_conf[t].values()) for t in COARSE_TYPES)
    coarse_overall = c_correct / c_total if c_total else float("nan")

    logger.info("\n" + "=" * 72)
    logger.info("COARSE 3-WAY MATRIX  (persistent / recurring / no_drift)")
    logger.info("=" * 72)
    chead = f"{'true \\ pred':<14}" + "".join(f"{p:>12}" for p in COARSE_TYPES)
    logger.info(chead)
    logger.info("-" * len(chead))
    for t in COARSE_TYPES:
        row_n = sum(coarse_conf[t].values())
        if row_n == 0:
            continue
        acc = coarse_conf[t][t] / row_n
        logger.info(f"{t:<14}" + "".join(f"{coarse_conf[t][p]:>12}" for p in COARSE_TYPES)
                    + f"   acc={acc:.2f}")
    logger.info("-" * len(chead))
    logger.info("COARSE accuracy: %.2f  (%d/%d)", coarse_overall, c_correct, c_total)

    summary = {
        "confusion": confusion,
        "per_type_accuracy": per_type_acc,
        "overall_accuracy": overall,
        "coarse_confusion": coarse_conf,
        "coarse_accuracy": coarse_overall,
        "sanity": sanity,
        "classifier": cls_kwargs,
        "seeds": list(args.seeds),
    }
    with open(out_dir / "drift_type_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("\nSaved confusion matrix + per-run CSV to %s", out_dir)


if __name__ == "__main__":
    main()