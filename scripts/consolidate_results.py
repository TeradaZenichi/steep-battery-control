"""Consolidate validation (eval) and annual test results into one table.

For every (method, architecture, tariff) it collects:
  - val_best : best validation eval_mean_reward over training (from the audit)
  - test_mean / test_worst : annual held-out test (best checkpoint, raw mode)
and brackets them with the MILP teacher upper bound and the RB/RBS lower bound,
reporting the optimality gap to MILP and the improvement over RBS.

Reads:
  paper/train/<method>/<arch>/<tariff>/audit_training.csv   (val_best)
  paper/test/<method>/<arch>/<tariff>/summary_overall.json  (test)
  paper/test/baselines/{RB,RBS}/<tariff>/summary_overall.json
  paper/test/teacher/<tariff>/summary_overall.json          (run teacher_test.py first)

Writes:
  paper/results_consolidated.csv
and prints a per-tariff markdown table to stdout.

Run (from project root):
    python scripts/consolidate_results.py
    python scripts/consolidate_results.py --checkpoint best --mode raw
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METHODS = [
    "supervised", "sac_penalty", "sac_cmdp",
    "sac_penalty_ft", "sac_cmdp_ft", "sac_cmdp_rlpd", "sac_cmdp_ema",
]
ARCHS = ["GRU", "MHA", "TCN"]
TARIFFS = ["tar_flat", "tar_s", "tar_w", "tar_sw", "tar_tou"]


def _val_best(method, arch, tariff):
    p = PROJECT_ROOT / "paper" / "train" / method / arch / tariff / "audit_training.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if "eval_mean_reward" not in df.columns:
        return None
    col = df["eval_mean_reward"].dropna()
    return float(col.max()) if len(col) else None


def _test_summary(path, checkpoint, mode):
    if not path.exists():
        return None
    data = json.load(open(path, encoding="utf-8"))
    # Baselines/teacher use checkpoint="rule"/"teacher"; learned use the given one.
    for row in data:
        ck = row.get("checkpoint")
        md = row.get("mode", mode)
        if ck in (checkpoint, "rule", "teacher") and md == mode:
            return row
    # Fallback: first row.
    return data[0] if data else None


def _bound(name, tariff, checkpoint, mode):
    p = PROJECT_ROOT / "paper" / "test" / name / tariff / "summary_overall.json"
    row = _test_summary(p, checkpoint, mode)
    return float(row["mean_reward"]) if row else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--mode", default="raw")
    args = parser.parse_args()

    teacher = {t: _bound("teacher", t, "teacher", args.mode) for t in TARIFFS}
    rb = {t: _bound("baselines/RB", t, "rule", args.mode) for t in TARIFFS}
    rbs = {t: _bound("baselines/RBS", t, "rule", args.mode) for t in TARIFFS}

    rows = []
    for method in METHODS:
        for arch in ARCHS:
            for tariff in TARIFFS:
                test_path = PROJECT_ROOT / "paper" / "test" / method / arch / tariff / "summary_overall.json"
                trow = _test_summary(test_path, args.checkpoint, args.mode)
                vb = _val_best(method, arch, tariff)
                if trow is None and vb is None:
                    continue
                test_mean = float(trow["mean_reward"]) if trow else None
                test_worst = float(trow.get("worst_reward")) if trow and "worst_reward" in trow else None

                gap_milp = None
                if test_mean is not None and teacher.get(tariff):
                    c_m, c_t = -test_mean, -teacher[tariff]
                    gap_milp = 100.0 * (c_m - c_t) / c_t
                impr_rbs = None
                if test_mean is not None and rbs.get(tariff):
                    c_m, c_r = -test_mean, -rbs[tariff]
                    impr_rbs = 100.0 * (c_r - c_m) / c_r

                rows.append({
                    "method": method, "arch": arch, "tariff": tariff,
                    "val_best": round(vb, 2) if vb is not None else "",
                    "test_mean": round(test_mean, 2) if test_mean is not None else "",
                    "test_worst": round(test_worst, 2) if test_worst is not None else "",
                    "teacher_milp": round(teacher[tariff], 2) if teacher.get(tariff) else "",
                    "rbs": round(rbs[tariff], 2) if rbs.get(tariff) else "",
                    "gap_to_milp_pct": round(gap_milp, 1) if gap_milp is not None else "",
                    "improvement_vs_rbs_pct": round(impr_rbs, 1) if impr_rbs is not None else "",
                })

    out_csv = PROJECT_ROOT / "paper" / "results_consolidated.csv"
    fields = ["method", "arch", "tariff", "val_best", "test_mean", "test_worst",
              "teacher_milp", "rbs", "gap_to_milp_pct", "improvement_vs_rbs_pct"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)\n")

    # Markdown summary per tariff (test_mean + gap to MILP).
    missing_teacher = [t for t in TARIFFS if not teacher.get(t)]
    if missing_teacher:
        print(f"[warn] no teacher upper bound for: {missing_teacher} "
              f"(run scripts/teacher_test.py) — gap_to_milp left blank\n")
    for tariff in TARIFFS:
        sub = [r for r in rows if r["tariff"] == tariff and r["test_mean"] != ""]
        if not sub:
            continue
        sub.sort(key=lambda r: r["test_mean"])  # most negative first
        print(f"### {tariff}  (MILP={teacher.get(tariff)}, RBS={rbs.get(tariff)})")
        print(f"{'method/arch':<26} {'test_mean':>10} {'gap_MILP%':>10} {'vs_RBS%':>9}")
        for r in sub:
            print(f"{r['method']+'/'+r['arch']:<26} {r['test_mean']:>10} "
                  f"{r['gap_to_milp_pct']:>10} {r['improvement_vs_rbs_pct']:>9}")
        print()


if __name__ == "__main__":
    main()
