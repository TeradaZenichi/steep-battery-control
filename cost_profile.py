#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate stochastic tariffs based on solar (GHI) and wind speed
from a merged CSV (building + weather + EV).

Input:
    - CSV with ';' separator, containing at least:
        * timestamp
        * ghi_Wm2
        * wspd_m_s

Output:
    - CSV with ';' separator, containing:
        * timestamp
        * tar_s   : solar-based dynamic tariff [€/kWh]
        * tar_w   : wind-based dynamic tariff [€/kWh]
        * tar_sw  : combined solar+wind dynamic tariff [€/kWh]
        * tar_flat: flat tariff [€/kWh]
        * tar_tou : time-of-use tariff [€/kWh]

File paths (input / output) are read from parameters.json:
    {
        "input file": "Simulation_CY_Cur_HP__PV5000-HB5000_EV.csv",
        "output file": "Simulation_CY_Cur_HP__PV5000-HB5000_tariffs.csv",
        ...
    }

All tariff parameters (base_price, dependencies, ndays) are hardcoded below.

Note: All costs and tariffs in this module are expressed in €/kWh (energy basis).
      When computing total costs, multiply tariff by energy consumed in kWh.
"""

import pandas as pd
import json
from pathlib import Path

# ============================================================
# Load only paths from JSON (keep previous keys for file paths)
# ============================================================



# Use configuration settings for file paths only
filepath  =   "data/Simulation_CY_Cur_HP__PV5000-HB5000_EV.csv"
fileoutput   =  "data/Simulation_CY_Cur_HP__PV5000-HB5000_tariffs.csv"

# ============================================================
# Fixed tariff parameters (all prices in €/kWh - energy basis)
# ============================================================

base_price = 1.0   # Base tariff [€/kWh]
min_price  = -1e6  # Minimum tariff [€/kWh]
max_price  = 1e6   # Maximum tariff [€/kWh]

# Sensitivities to normalized variations
αs  = 0.074   # solar energy dependency
αw  = 0.120   # wind energy dependency
αsd = 0.074   # combined solar energy dependency
αwd = 0.120   # combined wind energy dependency

# Moving-average window in days
ndays = 7     # number of samples [day] for moving average window


# ============================================================
# Functions adapted to the merged CSV (with timestamp, ghi_Wm2, wspd_m_s)
# ============================================================

def read_weather_data(filepath: str) -> pd.DataFrame:
    """
    Read merged CSV (building + weather + EV) with ';' separator.
    Expected columns include:
      - timestamp
      - ghi_Wm2
      - wspd_m_s

    Renames for internal processing:
      ghi_Wm2   -> Global Horizontal Radiation
      wspd_m_s  -> Wind Speed (m/s)
    """
    df = pd.read_csv(filepath, sep=';', parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    rename_map = {
        'ghi_Wm2': 'Global Horizontal Radiation',
        'wspd_m_s': 'Wind Speed (m/s)',
    }
    df = df.rename(columns=rename_map)

    required = ['timestamp', 'Global Horizontal Radiation', 'Wind Speed (m/s)']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input CSV: {missing}")

    return df


def calculate_wrapped_moving_average(data: pd.DataFrame, ndays: int) -> pd.DataFrame:
    """
    Compute wrapped moving averages for radiation and wind over ndays.
    Works for arbitrary time step by inferring Δt from timestamp.
    """
    df = data.copy()

    ts = pd.to_datetime(df['timestamp'])
    if len(ts) < 2:
        raise ValueError("Not enough rows to infer time step.")
    dt_min = (ts.iloc[1] - ts.iloc[0]).total_seconds() / 60.0
    if dt_min <= 0:
        raise ValueError("Non-positive time step inferred from timestamps.")

    steps_per_day = int(round(24 * 60 / dt_min))
    wma = int(ndays * steps_per_day)

    # Keep window within dataset size
    wma = max(1, min(wma, len(df)))

    # Wrap: pad at beginning with last wma samples
    padding = df.tail(wma)
    data_padded = pd.concat([padding, df], ignore_index=True)

    rad_ma = data_padded['Global Horizontal Radiation'].rolling(
        window=wma, min_periods=1
    ).mean()[wma:].reset_index(drop=True)

    wind_ma = data_padded['Wind Speed (m/s)'].rolling(
        window=wma, min_periods=1
    ).mean()[wma:].reset_index(drop=True)

    df['Moving Average Radiation'] = rad_ma
    df['Moving Average Wind'] = wind_ma

    return df


def add_percentage_variation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized variation for radiation and wind based on moving averages.
    """
    df = data.copy()

    df['Normalized Variation % Radiation'] = (
        (df['Global Horizontal Radiation'] - df['Moving Average Radiation']) /
        df['Moving Average Radiation']
    )

    df['Normalized Variation % Wind'] = (
        (df['Wind Speed (m/s)'] - df['Moving Average Wind']) /
        df['Moving Average Wind']
    )

    return df


def add_tariff_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add dynamic tariff columns based on normalized variations.
    Tariffs are clipped between min_price and max_price.
    Also add flat and TOU tariffs.

    All tariffs are expressed in €/kWh (energy basis).
    To compute total cost, multiply tariff by energy consumed in kWh.

    Columns created:
        - tar_s    : solar-based dynamic tariff [€/kWh]
        - tar_w    : wind-based dynamic tariff [€/kWh]
        - tar_sw   : combined solar+wind dynamic tariff [€/kWh]
        - tar_flat : flat tariff [€/kWh]
        - tar_tou  : time-of-use tariff [€/kWh]
    """
    df = data.copy()

    # Dynamic tariffs (solar, wind, combined) [€/kWh]
    df['tar_s'] = (
        base_price - αs * df['Normalized Variation % Radiation']
    ).clip(min_price, max_price)

    df['tar_w'] = (
        base_price - αw * df['Normalized Variation % Wind']
    ).clip(min_price, max_price)

    df['tar_sw'] = (
        base_price
        - αsd * df['Normalized Variation % Radiation']
        - αwd * df['Normalized Variation % Wind']
    ).clip(min_price, max_price)

    # Flat tariff (constant) [€/kWh]
    df['tar_flat'] = base_price

    # Time-of-use tariff (simple example based on hour of day) [€/kWh]
    hours = df['timestamp'].dt.hour

    # Start with off-peak
    tou = pd.Series(base_price * 0.7, index=df.index)

    # Mid-period
    tou[(hours >= 10) & (hours < 18)] = base_price * 1.0

    # Peak period
    tou[(hours >= 18) & (hours < 22)] = base_price * 1.3

    df['tar_tou'] = tou

    return df


def save_all_columns_to_csv(data: pd.DataFrame):
    """
    Save full DataFrame to CSV with ';' as delimiter,
    keeping all original columns plus the new tariff columns.
    """
    data.to_csv(fileoutput, index=False, sep=';')


# ============================
# Main processing
# ============================

def main():
    df = read_weather_data(filepath)
    df = calculate_wrapped_moving_average(df, ndays)
    df = add_percentage_variation(df)
    df = add_tariff_columns(df)

    save_all_columns_to_csv(df)
    print(df[['tar_s', 'tar_w', 'tar_sw', 'tar_flat', 'tar_tou']].describe())


if __name__ == "__main__":
    main()