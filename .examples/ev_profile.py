#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Add a stochastic EV worker profile to a time series CSV.

Encoding in column `ev_status`:
-  0   : EV is away from home
- -1   : time step immediately before arrival
-  0<x<1 : EV is at home; value is SoC on arrival (replicated while at home)
-  1   : departure time step

Assumes:
- Input CSV has a `timestamp` column.
- Time step is regular (e.g., 5, 10, 15 or 60 minutes).
- Pattern: worker leaves home in the morning of day d+1 and arrives in
  the evening of day d (so each arrival at day d is paired with departure
  next morning at day d+1).
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_CSV = Path("data/Simulation_WY_Cur_HP__PV5000-HB5000.csv")
OUTPUT_CSV = Path("data/Simulation_WY_Cur_HP__PV5000-HB5000_EV.csv")

RNG_SEED = 123

# Arrival (evening) and departure (next morning) parameters in hours
ARR_PARAMS = {
    "weekday": {"mu": 18.5, "sigma": 1.0},  # ~18:30
    "weekend": {"mu": 20.0, "sigma": 2.0},  # mais tarde
}
DEP_PARAMS = {
    "weekday": {"mu": 8.0, "sigma": 0.75},  # ~08:00
    "weekend": {"mu": 9.0, "sigma": 1.0},   # um pouco mais tarde
}

# Bounds to truncar horários (sempre em horas)
ARR_MIN_H, ARR_MAX_H = 15.0, 23.5
DEP_MIN_H, DEP_MAX_H = 5.0, 12.0

# SoC on arrival distribution
MU_SOC = 0.30
SIGMA_SOC = 0.10


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sample_hour(mu: float, sigma: float, h_min: float, h_max: float,
                rng: np.random.Generator) -> float:
    """Sample an hour in [h_min, h_max] from a truncated normal."""
    h = rng.normal(mu, sigma)
    return float(np.clip(h, h_min, h_max))


def main():
    rng = np.random.default_rng(RNG_SEED)

    # --------------------------------------------------------
    # Read CSV and set timestamp as index
    # --------------------------------------------------------
    df = pd.read_csv(INPUT_CSV, sep=";", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Initialize ev_status = 0 (away)
    df["ev_status"] = 0.0

    # Sorted unique days in the index
    days = df.index.normalize().unique()
    n_days = len(days)
    if n_days < 2:
        raise ValueError("Need at least 2 days in the time series.")

    # --------------------------------------------------------
    # Sample arrivals for days[0..n_days-2] and departures for days[1..n_days-1]
    # Pattern: arrival on day i, departure next morning on day i+1.
    # --------------------------------------------------------
    arrivals = {}
    departures = {}
    soc_arrival = {}

    # Sample arrivals for all days except the last
    for i in range(n_days - 1):
        date = days[i]
        wtype = "weekday" if date.weekday() < 5 else "weekend"
        pars = ARR_PARAMS[wtype]
        h_arr = sample_hour(pars["mu"], pars["sigma"], ARR_MIN_H, ARR_MAX_H, rng)
        arrivals[i] = date + pd.Timedelta(hours=h_arr)
        # SoC associado a essa chegada
        soc_arrival[i] = float(np.clip(rng.normal(MU_SOC, SIGMA_SOC), 0.0, 1.0))

    # Sample departures for all days except the first
    for i in range(1, n_days):
        date = days[i]
        wtype = "weekday" if date.weekday() < 5 else "weekend"
        pars = DEP_PARAMS[wtype]
        h_dep = sample_hour(pars["mu"], pars["sigma"], DEP_MIN_H, DEP_MAX_H, rng)
        departures[i] = date + pd.Timedelta(hours=h_dep)

    # --------------------------------------------------------
    # Build sessions: for each i in [0..n_days-2],
    #   arrival on day i   -> arrivals[i]
    #   departure on day i+1 -> departures[i+1]
    # --------------------------------------------------------
    full_index = df.index

    for i in range(n_days - 1):
        arr_ts = arrivals[i]
        dep_ts = departures[i + 1]
        soc = soc_arrival[i]

        # Map to nearest time steps in the full index
        arr_pos = full_index.get_indexer([arr_ts], method="nearest")[0]
        dep_pos = full_index.get_indexer([dep_ts], method="nearest")[0]

        arr_idx = full_index[arr_pos]
        dep_idx = full_index[dep_pos]

        # 1) Pre-arrival: time step immediately before arr_idx -> -1
        prev_arr_pos = arr_pos - 1
        if prev_arr_pos >= 0:
            prev_arr_idx = full_index[prev_arr_pos]
            df.loc[prev_arr_idx, "ev_status"] = -1.0

        # 2) At home: from arr_idx inclusive up to dep_idx exclusive -> SoC
        if arr_idx < dep_idx:
            mask = (full_index >= arr_idx) & (full_index < dep_idx)
            df.loc[mask, "ev_status"] = soc
        else:
            # (Caso patológico: se por algum motivo arr >= dep, apenas ignore sessão)
            continue

        # 3) Departure: dep_idx -> 1
        df.loc[dep_idx, "ev_status"] = 1.0

    # --------------------------------------------------------
    # Save updated CSV
    # --------------------------------------------------------
    df.reset_index().to_csv(OUTPUT_CSV, sep=";", index=False)
    print(f"Saved CSV with EV profile to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
