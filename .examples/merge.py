#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge EPW weather data with building demand and PV generation
from an Excel simulation file, using interpolation to a common
time resolution for all series.

- EPW: hourly data (8760 samples).
- Excel: 15-minute data (expected 35040 samples).
- TARGET_STEP_MIN controls the target time step (e.g., 5, 10, 15, 30, 60).

Output: CSV with ';' as delimiter.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# GLOBAL SETTINGS
# ============================================================

# Target time step in minutes for interpolation (e.g. 5, 10, 15, 30, 60)
TARGET_STEP_MIN = 5

# Paths to input files (adjust as needed)
EPW_PATH = Path(".examples/epw/Extreme_Warm_Year_Uccle_Current.epw")
EXCEL_PATH = Path(".examples/sim/Simulation_WY_Cur_HP__PV5000-HB5000.xlsx")

# Output CSV path
OUTPUT_CSV_PATH = Path("Simulation_WY_Cur_HP__PV5000-HB5000.csv")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_excel_data(excel_path: Path) -> pd.DataFrame:
    """
    Load the Excel file and keep only the relevant columns.
    Assumes data is in sheet 'Data'.
    """
    df = pd.read_excel(excel_path, sheet_name="Data")

    # Keep only demand and PV generation (rename if needed)
    cols = [
        "electricity_demand_rate_W",
        "produced_electricity_rate_W",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in Excel: {missing}")

    df_num = df[cols].apply(pd.to_numeric, errors="coerce")
    return df_num.reset_index(drop=True)


def load_epw_data(epw_path: Path) -> pd.DataFrame:
    """
    Load EPW file, skipping header, and select a subset of climate variables.
    EPW format: no header row, data starts after 8 header lines.
    """
    epw = pd.read_csv(epw_path, skiprows=8, header=None)

    # Rename standard EPW columns we care about
    epw = epw.rename(
        columns={
            0: "year",
            1: "month",
            2: "day",
            3: "hour",
            4: "minute",
            6: "drybulb_C",
            7: "dewpoint_C",
            8: "relhum_percent",
            13: "ghi_Wm2",
            14: "dni_Wm2",
            15: "dhi_Wm2",
            20: "wspd_m_s",
            21: "wdir_deg",
        }
    )

    # Choose here which climate columns you want in the final CSV
    climate_cols = [
        "drybulb_C",
        "relhum_percent",
        "ghi_Wm2",
        "dni_Wm2",
        "dhi_Wm2",
        "wspd_m_s",
        "wdir_deg",
    ]

    missing = [c for c in climate_cols if c not in epw.columns]
    if missing:
        raise ValueError(f"Missing expected columns in EPW: {missing}")

    df_climate = epw[climate_cols].apply(pd.to_numeric, errors="coerce")
    return df_climate.reset_index(drop=True)


def interpolate_to_length(df: pd.DataFrame, target_len: int) -> pd.DataFrame:
    """
    Interpolate each column of df to have exactly target_len samples,
    assuming the original samples span the same total duration.

    Uses a normalized 1D parameterization [0, 1] for the interpolation
    to avoid dealing explicitly with calendar dates here.
    """
    n_src = len(df)
    if n_src < 2:
        raise ValueError("Source DataFrame needs at least 2 rows for interpolation.")

    x_src = np.linspace(0.0, 1.0, num=n_src, endpoint=True)
    x_tgt = np.linspace(0.0, 1.0, num=target_len, endpoint=True)

    data_interp = {}
    for col in df.columns:
        y_src = df[col].to_numpy(dtype=float)
        # Optional: basic NaN handling before interpolation
        # If you expect NaNs, you can fill them here:
        # mask = np.isnan(y_src)
        # if mask.any():
        #     y_src[mask] = np.interp(x_src[mask], x_src[~mask], y_src[~mask])
        data_interp[col] = np.interp(x_tgt, x_src, y_src)

    return pd.DataFrame(data_interp)


def main():
    # --------------------------------------------------------
    # Load original data
    # --------------------------------------------------------
    excel_df = load_excel_data(EXCEL_PATH)
    epw_df = load_epw_data(EPW_PATH)

    # --------------------------------------------------------
    # Check basic assumptions
    # --------------------------------------------------------
    n_epw = len(epw_df)       # expected 8760 (hours in a year)
    if n_epw == 0:
        raise ValueError("EPW DataFrame is empty.")

    # Total minutes in the EPW year (assuming 1-hour step)
    total_minutes = n_epw * 60

    if total_minutes % TARGET_STEP_MIN != 0:
        raise ValueError(
            f"TARGET_STEP_MIN={TARGET_STEP_MIN} does not divide total_minutes={total_minutes}."
        )

    # Number of points at the target time step over the same total duration
    n_target = total_minutes // TARGET_STEP_MIN

    # --------------------------------------------------------
    # Interpolate EPW (hourly) and Excel (15-min) to n_target
    # --------------------------------------------------------
    epw_interp = interpolate_to_length(epw_df, n_target)
    excel_interp = interpolate_to_length(excel_df, n_target)

    # --------------------------------------------------------
    # Build a synthetic time index for the final DataFrame
    # --------------------------------------------------------
    start_time = pd.Timestamp("2000-01-01 00:00")  # arbitrary but consistent
    time_index = pd.date_range(
        start=start_time, periods=n_target, freq=f"{TARGET_STEP_MIN}min"
    )

    # --------------------------------------------------------
    # Merge into a single DataFrame
    # --------------------------------------------------------
    merged = pd.concat([excel_interp, epw_interp], axis=1)
    merged.index = time_index

    # --------------------------------------------------------
    # Save to CSV with ';' as delimiter
    # --------------------------------------------------------
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV_PATH, sep=";", index_label="timestamp")

    print(f"Saved merged CSV to: {OUTPUT_CSV_PATH.resolve()}")
    print(f"Shape: {merged.shape}, time step: {TARGET_STEP_MIN} minutes")


if __name__ == "__main__":
    main()
