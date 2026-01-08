# plot_il_csv_multi.py
#
# Edit the variables in the CONFIG section below, then run:
#   python plot_il_csv_multi.py
#
# For each (MODEL_NAME, TARIFF_NAME, START_DATE) triplet, this script:
#   1) Loads the IL-style CSV at CSV_PATHS[i]
#   2) Slices the time window [START_DATE, START_DATE + N_DAYS)
#   3) Plots (top) PV, PGRID, PBESS, PLOAD, PV*(1-XPV)
#      and (bottom) SoC of BESS and EV
#   4) Saves to Figures/{tariff}/{model}-{day_start}-{day_end}.pdf
#
# Notes:
# - All lists must have the same length.
# - CSV must have a timestamp column or timestamp index.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG (EDIT HERE)
# =========================
CSV_PATHS = [
    r"",  # e.g.: r"C:\Users\Lucas\Code\steep-battery-control\Results\1_MLP_IL\teacher_tar_s_CY_year.csv"
]
MODEL_NAMES = [
    "1_MLP_IL",
]
TARIFF_NAMES = [
    "tar_s",
]
START_DATES = [
    "2000-01-01 00:00:00",
]

# Window selection (inclusive start, for N days)
N_DAYS = 3

# If your CSV uses different column names, edit these candidate lists.
CAND_PGRID = ["act_pgrid", "Pgrid", "pgrid"]
CAND_PBESS = ["act_PBESS", "PBESS_env", "PBESS"]
CAND_PLOAD = ["load_kw", "Load", "Pload", "load"]
CAND_PV_USED = ["act_ppv_used", "Ppv", "ppv_used"]
CAND_PV_CURTAILED = ["act_ppv_curtailed", "ppv_curtailed", "pv_curtailed_kw", "pv_curtailed"]
CAND_XPV = ["act_XPV", "XPV", "act_XPV_cmd"]
CAND_SOC_BESS = ["soc_bess", "EBESS", "bess_soc", "SoC_BESS"]
CAND_SOC_EV = ["soc_ev", "Eev", "ev_soc", "SoC_EV"]

# Figure style
FIGSIZE = (14, 8)
DPI = 200
# =========================


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _require(col: str | None, what: str, df: pd.DataFrame, candidates: list[str]) -> str:
    if col is None:
        raise KeyError(
            f"Missing column for '{what}'.\n"
            f"Candidates tried: {candidates}\n"
            f"Available columns ({len(df.columns)}): {list(df.columns)}"
        )
    return col


def load_il_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()].sort_index()
    df.index.name = "timestamp"
    return df


def slice_window(df: pd.DataFrame, start_date: str, n_days: int) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = start + pd.Timedelta(days=int(n_days))
    w = df.loc[(df.index >= start) & (df.index < end)].copy()
    if w.empty:
        raise ValueError(
            f"No data in selected window.\n"
            f"start={start} end(exclusive)={end}\n"
            f"CSV time range: {df.index.min()} .. {df.index.max()}"
        )
    return w


def build_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    col_pgrid = _require(_pick_first_existing(df, CAND_PGRID), "PGRID", df, CAND_PGRID)
    col_pbess = _require(_pick_first_existing(df, CAND_PBESS), "PBESS (env)", df, CAND_PBESS)
    col_pload = _require(_pick_first_existing(df, CAND_PLOAD), "PLOAD", df, CAND_PLOAD)
    col_pv_used = _require(_pick_first_existing(df, CAND_PV_USED), "PV used (env)", df, CAND_PV_USED)

    col_xpv = _pick_first_existing(df, CAND_XPV)
    col_pv_curtailed = _pick_first_existing(df, CAND_PV_CURTAILED)

    # PV available reconstruction
    if col_pv_curtailed is not None:
        pv_available = df[col_pv_used].astype(float) + df[col_pv_curtailed].astype(float)
    else:
        pv_available = df[col_pv_used].astype(float)

    # XPV (curtailment fraction)
    if col_xpv is not None:
        xpv = df[col_xpv].astype(float).clip(0.0, 1.0)
    else:
        denom = pv_available.replace(0.0, np.nan)
        xpv = (1.0 - (df[col_pv_used].astype(float) / denom)).fillna(0.0).clip(0.0, 1.0)

    col_soc_bess = _require(_pick_first_existing(df, CAND_SOC_BESS), "SoC BESS", df, CAND_SOC_BESS)
    col_soc_ev = _require(_pick_first_existing(df, CAND_SOC_EV), "SoC EV", df, CAND_SOC_EV)

    return {
        "PV_available": pv_available.astype(float),
        "PV_used_env": df[col_pv_used].astype(float),
        "PGRID": df[col_pgrid].astype(float),
        "PBESS_env": df[col_pbess].astype(float),
        "PLOAD": df[col_pload].astype(float),
        "XPV": xpv.astype(float),
        "PV_times_(1-XPV)": (pv_available.astype(float) * (1.0 - xpv.astype(float))).astype(float),
        "SoC_BESS": df[col_soc_bess].astype(float),
        "SoC_EV": df[col_soc_ev].astype(float),
    }


def build_output_path(tariff: str, model: str, start: pd.Timestamp, n_days: int) -> Path:
    fig_root = Path("Figures") / str(tariff)
    fig_root.mkdir(parents=True, exist_ok=True)

    day_start = start.strftime("%Y-%m-%d")
    day_end = (start + pd.Timedelta(days=int(n_days)) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fname = f"{model}-{day_start}-{day_end}.pdf"
    return fig_root / fname


def plot_and_save(series: dict[str, pd.Series], out_pdf: Path, title: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    ax1.plot(series["PV_available"].index, series["PV_available"].values, label="PV (available)")
    ax1.plot(series["PV_used_env"].index, series["PV_used_env"].values, label="PV used (env)")
    ax1.plot(series["PV_times_(1-XPV)"].index, series["PV_times_(1-XPV)"].values, label="PV * (1 - XPV)")
    ax1.plot(series["PGRID"].index, series["PGRID"].values, label="PGRID")
    ax1.plot(series["PBESS_env"].index, series["PBESS_env"].values, label="PBESS")
    ax1.plot(series["PLOAD"].index, series["PLOAD"].values, label="PLOAD")
    ax1.set_ylabel("Power (kW)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(series["SoC_BESS"].index, series["SoC_BESS"].values, label="SoC BESS")
    ax2.plot(series["SoC_EV"].index, series["SoC_EV"].values, label="SoC EV")
    ax2.set_ylabel("State of Charge")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")


def validate_config():
    n = len(CSV_PATHS)
    if not (len(MODEL_NAMES) == len(TARIFF_NAMES) == len(START_DATES) == n):
        raise ValueError(
            "CONFIG lists must have the same length:\n"
            f"len(CSV_PATHS)={len(CSV_PATHS)}\n"
            f"len(MODEL_NAMES)={len(MODEL_NAMES)}\n"
            f"len(TARIFF_NAMES)={len(TARIFF_NAMES)}\n"
            f"len(START_DATES)={len(START_DATES)}"
        )
    if n == 0:
        raise ValueError("CSV_PATHS is empty. Add at least one entry.")


def main():
    validate_config()

    for csv_p, model, tariff, start_date in zip(CSV_PATHS, MODEL_NAMES, TARIFF_NAMES, START_DATES):
        if not str(csv_p).strip():
            raise ValueError("One of the CSV_PATHS entries is empty.")
        csv_path = Path(str(csv_p)).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = load_il_csv(csv_path)
        w = slice_window(df, start_date, N_DAYS)
        series = build_series(w)

        start = pd.to_datetime(start_date)
        out_pdf = build_output_path(tariff, model, start, N_DAYS)
        title = f"{model} | {tariff} | {start.strftime('%Y-%m-%d')} +{N_DAYS}d"

        plot_and_save(series, out_pdf=out_pdf, title=title)


if __name__ == "__main__":
    main()
