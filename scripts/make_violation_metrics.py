"""Expanded native grid-limit violation metrics (reviewer item 11).

From the RAW (pre-projection) deployed operation traces, per controller:
  - import / export peak overshoot (kW);
  - per-step violation probability;
  - number of violation events and mean duration (min);
  - violation energy split into import and export (kWh, per held-out year);
  - worst single window (kWh);
  - mean projection correction on the grid power (kW), from raw vs safe traces.

Reported for the demonstration-free CMDP and the fine-tuning reference, over the five
tariffs and three encoders.

  python scripts/make_violation_metrics.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PMAX_IMP = 10.0
PMAX_EXP = 2.0
DT_H = 5.0 / 60.0          # hours per step
DT_MIN = 5.0
TOL = 1e-6
TARIFFS = ["tar_flat", "tar_tou", "tar_s", "tar_w", "tar_sw"]
ARCHS = ["GRU", "MHA", "TCN"]
METHODS = [("u_cmdp", "CMDP (demo-free)"), ("u_cmdp_ft", "CMDP+FT (reference)")]
WINDOWS = [f"test_{t}_{m:02d}" for t in ("cy", "wy") for m in range(1, 13)]


def overshoot(pg):
    imp = np.maximum(pg - PMAX_IMP, 0.0)
    exp = np.maximum(-PMAX_EXP - pg, 0.0)
    return imp, exp


def events(mask):
    n, dur, run = 0, [], 0
    for v in mask:
        if v:
            run += 1
        elif run:
            n += 1; dur.append(run); run = 0
    if run:
        n += 1; dur.append(run)
    return n, (np.mean(dur) * DT_MIN if dur else 0.0)


def analyze(method):
    pk_i = pk_e = 0.0
    e_i = e_e = 0.0
    steps = viol_steps = n_ev = 0
    dur_acc = []
    worst = 0.0
    corr = []
    for arch in ARCHS:
        for tar in TARIFFS:
            base = ROOT / "paper" / "test" / method / arch / tar / "operations"
            for w in WINDOWS:
                praw = base / f"{w}_best_raw.csv.gz"
                if not praw.exists():
                    continue
                raw = pd.read_csv(praw, compression="gzip", parse_dates=["timestamp"]).set_index("timestamp")
                pg = raw["PGrid"].to_numpy()
                imp, exp = overshoot(pg)
                pk_i = max(pk_i, imp.max()); pk_e = max(pk_e, exp.max())
                e_i += imp.sum() * DT_H; e_e += exp.sum() * DT_H
                m = (imp + exp) > TOL
                steps += len(pg); viol_steps += int(m.sum())
                ne, _ = events(m); n_ev += ne
                worst = max(worst, (imp + exp).sum() * DT_H)
                _, md = events(m)
                if ne:
                    dur_acc.append(md)
                psafe = base / f"{w}_best_safe.csv.gz"
                if psafe.exists():
                    safe = pd.read_csv(psafe, compression="gzip", parse_dates=["timestamp"]).set_index("timestamp")
                    j = safe["PGrid"].reindex(raw.index)
                    corr.append(np.nanmean(np.abs(pg - j.to_numpy())))
    return {
        "peak_imp_kW": pk_i, "peak_exp_kW": pk_e,
        "viol_prob_%": 100 * viol_steps / max(steps, 1),
        "events": n_ev, "mean_dur_min": np.mean(dur_acc) if dur_acc else 0.0,
        "energy_imp_kWh": e_i, "energy_exp_kWh": e_e,
        "worst_window_kWh": worst, "mean_corr_kW": np.mean(corr) if corr else 0.0,
    }


def main():
    print("Native grid-limit violation metrics (RAW, deployed; 5 tariffs x 3 encoders)\n")
    rows = {ml: analyze(mk) for mk, ml in METHODS}
    keys = ["peak_imp_kW", "peak_exp_kW", "viol_prob_%", "events", "mean_dur_min",
            "energy_imp_kWh", "energy_exp_kWh", "worst_window_kWh", "mean_corr_kW"]
    print(f"{'metric':18} | " + " | ".join(f"{ml:>20}" for _, ml in METHODS))
    for k in keys:
        print(f"{k:18} | " + " | ".join(f"{rows[ml][k]:20.3f}" for _, ml in METHODS))


if __name__ == "__main__":
    main()
