"""TEST A: do EV departures actually meet the trip requirement (operational EV safety)?

The CMDP EV dual variable saturates (lambda_EV = lambda_max) and the cost-to-go exceeds
the nominal budget, so the constraint is not satisfied in the budget sense. This checks the
OPERATIONAL question instead: at each departure, was the EV SoC enough for the trip?

Uses existing deployed (safe, best) operation traces joined with the dataset's ev_conn /
ev_arrival by timestamp. A departure is conn {1,2} -> 0; the trip fraction consumed is
ev_arrival at the following arrival. Sufficient iff departure SoC >= trip fraction.

  python scripts/check_ev_departures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARIFFS = ["tar_flat", "tar_tou", "tar_s", "tar_w", "tar_sw"]
ARCHS = ["GRU", "MHA", "TCN"]
DATASETS = {
    "cy": "data/Simulation_CY_Fut_HP__PV5000-HB5000.csv",
    "wy": "data/Simulation_WY_Fut_HP__PV5000-HB5000.csv",
}
TOL = 1e-3

ds = {}
for k, f in DATASETS.items():
    d = pd.read_csv(ROOT / f, sep=";", parse_dates=["timestamp"], dayfirst=True, index_col="timestamp")
    ds[k] = d[["ev_conn", "ev_arrival"]]


def analyze(arch, tariff):
    base = ROOT / "paper" / "test" / "u_cmdp" / arch / tariff / "operations"
    dep_soc, trip = [], []
    for tag in ("cy", "wy"):
        for m in range(1, 13):
            p = base / f"test_{tag}_{m:02d}_best_safe.csv.gz"
            if not p.exists():
                continue
            tr = pd.read_csv(p, compression="gzip", parse_dates=["timestamp"]).set_index("timestamp")
            conn = ds[tag]["ev_conn"].reindex(tr.index).to_numpy()
            arr = ds[tag]["ev_arrival"].reindex(tr.index).to_numpy()
            soc = tr["SoCEV"].to_numpy()
            n = len(conn)
            for i in range(n - 1):
                if conn[i] in (1, 2) and conn[i + 1] == 0:          # departure
                    j = i + 1
                    while j < n and conn[j] == 0:
                        j += 1
                    if j < n and conn[j] in (1, 2) and arr[j] > 0:  # next trip's consumption
                        dep_soc.append(float(soc[i]))
                        trip.append(float(arr[j]))
    return np.array(dep_soc), np.array(trip)


def main():
    print("TEST A -- EV departures meeting the trip requirement (u_cmdp, best/safe)\n")
    print(f"{'tariff':8} {'enc':4} | {'#dep':>5} | {'%suff':>6} | {'dep_soc':>8} | {'#insuf':>6} | {'max_def':>7}")
    tot_dep = tot_insuf = 0
    for t in TARIFFS:
        for a in ARCHS:
            dep, trip = analyze(a, t)
            if len(dep) == 0:
                print(f"{t:8} {a:4} | no traces"); continue
            deficit = np.maximum(trip - dep, 0.0)
            insuf = deficit > TOL
            tot_dep += len(dep); tot_insuf += int(insuf.sum())
            print(f"{t:8} {a:4} | {len(dep):5d} | {100*np.mean(~insuf):6.1f} | {np.mean(dep):8.3f} | "
                  f"{int(insuf.sum()):6d} | {deficit.max():7.3f}")
    print(f"\nOVERALL: {tot_dep} departures, {tot_insuf} insufficient "
          f"({100*(tot_dep-tot_insuf)/max(tot_dep,1):.2f}% met the trip requirement)")


if __name__ == "__main__":
    main()
