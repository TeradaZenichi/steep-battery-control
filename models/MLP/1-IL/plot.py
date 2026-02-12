# models/MLP/1-IL/plot.py
from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates


# =============================================================================
# PREAMBLE (same pattern as train/test)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../steep-battery-control
MODEL_ROOT   = Path(__file__).resolve().parents[2]  # .../models
MLP_ROOT     = Path(__file__).resolve().parent.parent   # .../models/MLP
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MLP_ROOT))
sys.path.insert(0, str(MODEL_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))


# =============================================================================
# CONFIG (EDIT HERE)
# =============================================================================
MODEL_NAME = Path(__file__).resolve().parent.name     # e.g., "1-IL"
SPLIT = "test"                                       # "test", "run", ...

TARIFFS = ["tar_s"] #, "tar_w", "tar_sw", "tar_tou", "tar_flat"]

CONFIG_JSON = Path(__file__).resolve().parent / "config.json"
RESULTS_ROOT = PROJECT_ROOT / "Results"

PLOT_KEY = "plot"           # will read config[PLOT_KEY]
ONLY_NAMES = []             # e.g., ["test_1", "test_4"]; empty => all entries in config["plot"]

# Output format
FIGSIZE = (14, 8)
DPI = 200
EXT = "pdf"                 # "pdf" or "png"

# Which files to plot
ACTOR_SUFFIX   = "_actor_env_operation.csv"
TEACHER_SUFFIX = "_teacher_operation.csv"

# Strict schema (no candidates)
POWER_COLS = ["PPV", "PLoad", "PBESS", "PEV", "PGrid"]
SOC_COLS_ACTOR = ["SoCBESS", "SoCEV"]
SOC_COLS_TEACHER = ["SoCBESS", "SoCEV"]
CMD_COLS_ACTOR = ["bess_cmd", "ev_cmd", "pv_cmd"]
CURTAIL_COL = "χPV"

# If you want PV_available, we compute PPV / (1 - χPV)
PLOT_PV_AVAILABLE = True
# =============================================================================


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if PLOT_KEY not in cfg or not isinstance(cfg[PLOT_KEY], list):
        raise KeyError(f"Config must contain a list at key '{PLOT_KEY}': {path}")
    return cfg


def _read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        sep=",",
        parse_dates=["timestamp"],
        index_col="timestamp",
    ).sort_index()
    df.index.name = "timestamp"
    return df


def _require_cols(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns for {context}: {missing}\n"
            f"Available columns ({len(df.columns)}): {list(df.columns)}"
        )


def _slice_window(df: pd.DataFrame, start_date: str, days: int) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = start + pd.Timedelta(days=int(days))
    w = df.loc[(df.index >= start) & (df.index < end)].copy()
    if w.empty:
        raise ValueError(
            "No data in selected window.\n"
            f"start={start} end(exclusive)={end}\n"
            f"CSV time range: {df.index.min()} .. {df.index.max()}"
        )
    return w


def _plot_actor(df: pd.DataFrame, title: str, out_path: Path) -> None:
    _require_cols(df, POWER_COLS + SOC_COLS_ACTOR + CMD_COLS_ACTOR + [CURTAIL_COL], "actor plot")

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)

    colors = {
        "BESS": "#1f77b4",
        "EV": "#ff7f0e",
        "PV": "#2ca02c",
        "Load": "#d62728",
        "Grid": "#9467bd",
    }

    if len(df.index) >= 2:
        step = (df.index[1] - df.index[0]).total_seconds() / 86400.0
        bar_width = step * 0.8
    else:
        bar_width = 0.02

    # Commands (top)
    ax0.bar(
        df.index,
        df["bess_cmd"].astype(float).values,
        width=bar_width,
        color=colors["BESS"],
        alpha=0.7,
        label="bess_cmd",
    )
    ax0.bar(
        df.index,
        df["ev_cmd"].astype(float).values,
        width=bar_width,
        color=colors["EV"],
        alpha=0.7,
        label="ev_cmd",
    )
    ax0.plot(
        df.index,
        df["pv_cmd"].astype(float).values,
        color=colors["PV"],
        label="pv_cmd",
    )
    ax0.axhline(0.0, linewidth=1.0)
    ax0.set_ylabel("Commands")
    ax0.set_ylim(-1.0, 1.0)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    # Power (operation)
    if PLOT_PV_AVAILABLE:
        denom = (1.0 - df[CURTAIL_COL].astype(float)).replace(0.0, np.nan)
        pv_available = (df["PPV"].astype(float) / denom).fillna(df["PPV"].astype(float))
        ax1.plot(
            df.index,
            pv_available.values,
            label="PV_available",
            color=colors["PV"],
            linestyle="--",
        )

    ax1.plot(df.index, df["PPV"].astype(float).values, label="PPV", color=colors["PV"])
    ax1.plot(df.index, df["PLoad"].astype(float).values, label="PLoad", color=colors["Load"])
    ax1.plot(df.index, df["PGrid"].astype(float).values, label="PGrid", color=colors["Grid"])

    ax1.bar(
        df.index,
        df["PBESS"].astype(float).values,
        width=bar_width,
        color=colors["BESS"],
        alpha=0.7,
        label="PBESS",
    )
    ax1.bar(
        df.index,
        df["PEV"].astype(float).values,
        width=bar_width,
        color=colors["EV"],
        alpha=0.7,
        label="PEV",
    )

    ax1.axhline(0.0, linewidth=1.0)
    ax1.set_ylabel("Power (kW)")
    ax1.set_ylim(-2.5, 5.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # SoC (bottom)
    for c in SOC_COLS_ACTOR:
        ax2.plot(df.index, df[c].astype(float).values, label=c)

    ax2.set_ylabel("SoC")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    fig.suptitle(title)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_teacher(df: pd.DataFrame, title: str, out_path: Path) -> None:
    _require_cols(df, POWER_COLS + SOC_COLS_TEACHER + [CURTAIL_COL], "teacher plot")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    # Power
    if PLOT_PV_AVAILABLE:
        denom = (1.0 - df[CURTAIL_COL].astype(float)).replace(0.0, np.nan)
        pv_available = (df["PPV"].astype(float) / denom).fillna(df["PPV"].astype(float))
        ax1.plot(df.index, pv_available.values, label="PV_available")

    for c in POWER_COLS:
        ax1.plot(df.index, df[c].astype(float).values, label=c)
    ax1.axhline(0.0, linewidth=1.0)
    ax1.set_ylabel("Power (kW)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # SoC (teacher has no commands)
    for c in SOC_COLS_TEACHER:
        ax2.plot(df.index, df[c].astype(float).values, label=c)

    ax2.set_ylabel("SoC")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = _load_config(CONFIG_JSON)
    entries = cfg[PLOT_KEY]

    if ONLY_NAMES:
        keep = set(ONLY_NAMES)
        entries = [e for e in entries if str(e.get("name")) in keep]

    if not entries:
        raise ValueError(f"No plot entries selected. Check config['{PLOT_KEY}'] and ONLY_NAMES.")

    for tariff in TARIFFS:
        out_dir = RESULTS_ROOT / "figures" / "MLP" / MODEL_NAME / tariff

        out_dir.mkdir(parents=True, exist_ok=True)

        for entry in entries:
            name = str(entry["name"])
            start_date = str(entry["date"])
            days = int(entry["days"])

            actor_csv = RESULTS_ROOT / SPLIT / "MLP" / MODEL_NAME / tariff / f"{name}{ACTOR_SUFFIX}"
            teacher_csv = RESULTS_ROOT / SPLIT / "MLP" / MODEL_NAME / tariff / f"{name}{TEACHER_SUFFIX}"

            # Actor
            df_actor = _read_csv(actor_csv)
            w_actor = _slice_window(df_actor, start_date, days)
            start_ts = pd.to_datetime(start_date)
            out_actor = out_dir / f"{name}__{start_ts.strftime('%Y-%m-%d')}__{days}d__actor.{EXT}"
            title_actor = f"ACTOR | model={MODEL_NAME} | tariff={tariff} | {name} | {start_ts.strftime('%Y-%m-%d')} +{days}d"
            _plot_actor(w_actor, title_actor, out_actor)
            print(f"[OK] Saved: {out_actor}")

            # Teacher
            df_teacher = _read_csv(teacher_csv)
            w_teacher = _slice_window(df_teacher, start_date, days)
            out_teacher = out_dir / f"{name}__{start_ts.strftime('%Y-%m-%d')}__{days}d__teacher.{EXT}"
            title_teacher = f"TEACHER | model={MODEL_NAME} | tariff={tariff} | {name} | {start_ts.strftime('%Y-%m-%d')} +{days}d"
            _plot_teacher(w_teacher, title_teacher, out_teacher)
            print(f"[OK] Saved: {out_teacher}")


if __name__ == "__main__":
    main()
