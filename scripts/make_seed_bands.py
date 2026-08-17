"""Late-training seed band per method on the lead tariff (reproducible).

Definition (consistent with the mean +/- std band plotted in fig13/training_sw_3seeds):
  - per (method, encoder, seed): validation reward (eval_mean_reward), lightly smoothed
    (centered rolling mean, window 3, as in make_training_curves);
  - per episode, the std across the 3 seeds (sample, ddof=1, exactly as the band in
    make_training_curves uses pandas .std) -- the band half-width;
  - averaged over the final LATE episodes, then averaged over the three encoders.

  python scripts/make_seed_bands.py
"""
from __future__ import annotations
import statistics as st
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TR = ROOT / "paper" / "train"
ARCHS = ["GRU", "MHA", "TCN"]
SEEDS = ["tar_sw", "tar_sw-s7", "tar_sw-s13"]
METHODS = [("u_penalty", "Penalty"), ("u_cmdp", "CMDP"), ("u_cmdp_shaped", "PBRS"),
           ("u_cmdp_droq", "DroQ"), ("u_cmdp_redq", "REDQ"), ("u_cmdp_ft", "FT")]
SMOOTH = 3
LATE = 20
METRIC = "eval_mean_reward"


def band(method):
    per_enc = []
    for a in ARCHS:
        curves = []
        for sd in SEEDS:
            p = TR / method / a / sd / "audit_training.csv"
            if not p.exists():
                continue
            ev = pd.read_csv(p)[METRIC].dropna().reset_index(drop=True)
            ev = ev.rolling(SMOOTH, center=True, min_periods=1).mean()
            curves.append(ev)
        if len(curves) < 2:
            continue
        n = min(len(c) for c in curves)
        df = pd.DataFrame({i: c.iloc[:n].values for i, c in enumerate(curves)})
        per_ep_std = df.std(axis=1)                    # band half-width per episode (ddof=1, as fig13)
        per_enc.append(float(per_ep_std.iloc[-LATE:].mean()))
    return st.mean(per_enc) if per_enc else float("nan")


def main():
    print(f"Late-training seed band (smooth={SMOOTH}, last {LATE} episodes, mean over encoders):")
    for mk, ml in METHODS:
        print(f"  {ml:8} {band(mk):.1f}")


if __name__ == "__main__":
    main()
