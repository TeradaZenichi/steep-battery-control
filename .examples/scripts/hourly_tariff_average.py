"""Utility to homogenize tariff columns within each hour.

The script reads a ';'-separated CSV, computes the mean value of
`tar_s`, `tar_w`, and `tar_sw` for every hour, and assigns that hourly
mean to all rows that fall inside the same hour. Other columns remain untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

TAR_COLUMNS: List[str] = ["tar_s", "tar_w", "tar_sw"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fix tariff columns so they remain constant inside each hour by "
            "using the mean tariff of that hour."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the input CSV (expects a ';' separator by default).",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path. If omitted, the input file is overwritten "
            "after processing."
        ),
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Name of the timestamp column used to build the hourly buckets.",
    )
    parser.add_argument(
        "--separator",
        "-s",
        default=";",
        help="Column separator used in the CSV file (default ';').",
    )
    parser.add_argument(
        "--tar-columns",
        nargs="+",
        default=TAR_COLUMNS,
        help="Tariff columns that must be averaged per hour.",
    )
    parser.add_argument(
        "--dayfirst",
        action="store_true",
        help="Treat timestamps as day-first (e.g. DD/MM/YYYY).",
    )
    return parser.parse_args()


def ensure_columns_present(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def apply_hourly_means(
    df: pd.DataFrame,
    timestamp_column: str,
    tar_columns: Iterable[str],
    dayfirst: bool,
) -> pd.DataFrame:
    ensure_columns_present(df, [timestamp_column, *tar_columns])

    result = df.copy()
    timestamp = pd.to_datetime(result[timestamp_column], dayfirst=dayfirst)
    result["_hour_bucket"] = timestamp.dt.floor("H")

    # Compute the hourly mean for the tariff columns and broadcast back.
    hourly_means = result.groupby("_hour_bucket")[list(tar_columns)].transform("mean")
    result[list(tar_columns)] = hourly_means

    return result.drop(columns="_hour_bucket")


def main() -> None:
    args = parse_arguments()
    input_path = args.input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = args.output_path or input_path

    df = pd.read_csv(input_path, sep=args.separator)
    adjusted = apply_hourly_means(
        df,
        timestamp_column=args.timestamp_column,
        tar_columns=args.tar_columns,
        dayfirst=args.dayfirst,
    )

    adjusted.to_csv(output_path, sep=args.separator, index=False)
    print(f"Hourly tariffs written to {output_path}")


if __name__ == "__main__":
    main()
