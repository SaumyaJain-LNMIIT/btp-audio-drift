"""Aggregate per-seed drift-detection metrics into mean ± std summaries.

Pure numpy (no torch/dataset), so it is unit-testable in isolation. NaN values
(e.g. delay when nothing was detected, recall on a no-drift stream) are ignored
in the mean/std rather than poisoning them.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

_AGG_KEYS = [
    "detection_delay", "recall", "precision", "f1",
    "false_alarms", "false_alarms_per_1k", "num_detections", "true_positives",
]


def _mean_std(vals) -> tuple[float, float, int]:
    arr = np.asarray(vals, dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    return float(valid.mean()), float(valid.std()), int(valid.size)


def aggregate_runs(rows: list[dict]) -> dict[str, dict]:
    """Group per-(seed, scenario) metric dicts by scenario and summarise.

    Args:
        rows: each dict has at least ``scenario`` plus metric keys.

    Returns:
        ``{scenario: {n_seeds, detection_rate, <key>_mean/_std/_n, ...}}``.
    """
    by_scen: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scen[r["scenario"]].append(r)

    out: dict[str, dict] = {}
    for scen, rs in by_scen.items():
        entry: dict = {"n_seeds": len(rs)}
        for k in _AGG_KEYS:
            m, s, n = _mean_std([r.get(k, float("nan")) for r in rs])
            entry[f"{k}_mean"], entry[f"{k}_std"], entry[f"{k}_n"] = m, s, n
        # detection_rate: fraction of seeds with at least one true positive.
        tps = [r.get("true_positives", 0) for r in rs]
        entry["detection_rate"] = float(np.mean([1.0 if tp >= 1 else 0.0 for tp in tps]))
        out[scen] = entry
    return out


def format_aggregate_table(agg: dict[str, dict]) -> str:
    """Render the aggregate dict as a readable fixed-width table."""
    lines = ["MULTI-SEED SUMMARY (mean ± std over seeds)"]
    lines.append(
        f"{'Scenario':<12} {'Seeds':>5} {'DetRate':>8} "
        f"{'Delay(mean±std)':>18} {'F1(mean±std)':>14} {'FA/1k(mean±std)':>16}"
    )
    lines.append("-" * 76)
    for scen, e in agg.items():
        d_m, d_s = e["detection_delay_mean"], e["detection_delay_std"]
        delay = "n/a" if np.isnan(d_m) else f"{d_m:.1f}±{d_s:.1f}"
        f_m, f_s = e["f1_mean"], e["f1_std"]
        f1 = "n/a" if np.isnan(f_m) else f"{f_m:.2f}±{f_s:.2f}"
        fa_m, fa_s = e["false_alarms_per_1k_mean"], e["false_alarms_per_1k_std"]
        fa = "n/a" if np.isnan(fa_m) else f"{fa_m:.2f}±{fa_s:.2f}"
        lines.append(
            f"{scen:<12} {e['n_seeds']:>5} {e['detection_rate']:>8.2f} "
            f"{delay:>18} {f1:>14} {fa:>16}"
        )
    return "\n".join(lines)