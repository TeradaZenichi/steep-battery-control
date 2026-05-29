"""Plot two representative days of operation for every model/architecture/tariff.

Scans paper/test/<method>/<arch>/<tariff>/operations/ for the per-month operation
traces written by the test pipeline (test_{cy,wy}_{MM}_best_{raw,safe}.csv.gz),
picks a single shared 48 h window (the one with the most EV-connected steps, taken
from the dataset's ev_conn so it is identical across methods and shows V2G), and
renders one figure per (method, arch, tariff, mode) into paper/plots/.

Figure layout follows .bkp/models/GRU/2-RL/plot.py:
  - top  : powers (PBESS/PEV bars; PPV, PV_available, PLoad, PGrid steps) with the
           tariff overlaid on a secondary axis; EV-connected periods lightly shaded.
  - bottom: SoCBESS / SoCEV.

Run (from project root):
    python scripts/plot_operations.py
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = PROJECT_ROOT / "paper" / "test"
OUT_ROOT = PROJECT_ROOT / "paper" / "plots"

CKPT = "best"
MODES = ["raw", "safe"]
WINDOW_DAYS = 2
FIGSIZE = (14, 8)
DPI = 200
EXT = "pdf"

DATASETS = {
    "cy": PROJECT_ROOT / "data" / "Simulation_CY_Fut_HP__PV5000-HB5000.csv",
    "wy": PROJECT_ROOT / "data" / "Simulation_WY_Fut_HP__PV5000-HB5000.csv",
}

COLORS = {
    "BESS": "#1f77b4", "EV": "#ff7f0e", "PV": "#2ca02c",
    "Load": "#d62728", "Grid": "#9467bd", "Tariff": "#7f7f7f",
}

POWER_COLS = ["PPV", "PLoad", "PBESS", "PEV", "PGrid"]
SOC_COLS = ["SoCBESS", "SoCEV"]
CURTAIL_COL = "pv_cmd"  # action[2] == χPV in this pipeline

_RUN_RE = re.compile(r"^test_(cy|wy)_(\d{2})_" + re.escape(CKPT) + r"_(raw|safe)\.csv\.gz$")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def _setup_font() -> None:
    font_path = PROJECT_ROOT / "data" / "Gulliver.otf"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        prop = font_manager.FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "axes.unicode_minus": False,
    })


# ---------------------------------------------------------------------------
# ev_conn (window selection + shading)
# ---------------------------------------------------------------------------
_EVCONN_CACHE: dict[str, pd.Series] = {}


def _ev_conn(tag: str) -> pd.Series | None:
    if tag in _EVCONN_CACHE:
        return _EVCONN_CACHE[tag]
    path = DATASETS.get(tag)
    if path is None or not path.exists():
        _EVCONN_CACHE[tag] = None
        return None
    s = pd.read_csv(
        path, sep=";", parse_dates=["timestamp"], dayfirst=True,
        index_col="timestamp", usecols=["timestamp", "ev_conn"],
    )["ev_conn"].sort_index()
    _EVCONN_CACHE[tag] = s
    return s


def _best_window_start(tag: str, month: int) -> pd.Timestamp:
    """48 h window (within the first 7 days of the month) with most EV-connected steps."""
    start = pd.Timestamp(year=2000, month=month, day=1)
    s = _ev_conn(tag)
    default = start
    if s is None:
        return default
    best_off, best_score = 0, -1
    for off in range(0, 7 - WINDOW_DAYS + 1):
        w0 = start + pd.Timedelta(days=off)
        w1 = w0 + pd.Timedelta(days=WINDOW_DAYS)
        seg = s.loc[(s.index >= w0) & (s.index < w1)]
        score = int((seg != 0).sum())
        if score > best_score:
            best_score, best_off = score, off
    return start + pd.Timedelta(days=best_off)


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------
def _discover():
    """Return {(method, arch, tariff): {mode: {(tag, month): path}}} and present runs."""
    combos: dict[tuple, dict] = {}
    runs_present: set[tuple] = set()
    for op_dir in TEST_ROOT.glob("*/*/*/operations"):
        rel = op_dir.relative_to(TEST_ROOT).parts  # (method, arch, tariff, "operations")
        if len(rel) != 4:
            continue
        method, arch, tariff = rel[0], rel[1], rel[2]
        modes: dict[str, dict] = {}
        for f in op_dir.glob(f"*_{CKPT}_*.csv.gz"):
            m = _RUN_RE.match(f.name)
            if not m:
                continue
            tag, month, mode = m.group(1), int(m.group(2)), m.group(3)
            modes.setdefault(mode, {})[(tag, month)] = f
            runs_present.add((tag, month))
        if modes:
            combos[(method, arch, tariff)] = modes
    return combos, runs_present


def _global_window(runs_present: set[tuple]):
    """Pick one (tag, month, start) shared across all combos: max EV-connected steps."""
    best = None  # (score, tag, month, start)
    for tag, month in sorted(runs_present):
        start = _best_window_start(tag, month)
        s = _ev_conn(tag)
        if s is None:
            score = 0
        else:
            seg = s.loc[(s.index >= start) & (s.index < start + pd.Timedelta(days=WINDOW_DAYS))]
            score = int((seg != 0).sum())
        if best is None or score > best[0]:
            best = (score, tag, month, start)
    return best  # may be None if nothing present


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _read_trace(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
    return df.sort_index()


def _slice(df: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    end = start + pd.Timedelta(days=WINDOW_DAYS)
    return df.loc[(df.index >= start) & (df.index < end)].copy()


def _shade_ev(ax, tag: str, w: pd.DataFrame) -> None:
    s = _ev_conn(tag)
    if s is None or w.empty:
        return
    seg = s.reindex(w.index, method="nearest")
    connected = (seg != 0).to_numpy()
    idx = w.index.to_numpy()
    i = 0
    labeled = False
    while i < len(connected):
        if connected[i]:
            j = i
            while j + 1 < len(connected) and connected[j + 1]:
                j += 1
            ax.axvspan(idx[i], idx[j], color=COLORS["EV"], alpha=0.06,
                       label=("EV connected" if not labeled else None))
            labeled = True
            i = j + 1
        else:
            i += 1


def _plot(w: pd.DataFrame, tag: str, title: str, out_path: Path) -> None:
    missing = [c for c in POWER_COLS + SOC_COLS + [CURTAIL_COL] if c not in w.columns]
    if missing:
        print(f"[skip] {out_path.name}: missing columns {missing}")
        return

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    bar_width = ((w.index[1] - w.index[0]).total_seconds() / 86400.0 * 0.8) if len(w.index) >= 2 else 0.01

    _shade_ev(ax0, tag, w)

    # PV available (pre-curtailment), dashed
    denom = (1.0 - w[CURTAIL_COL].astype(float)).replace(0.0, np.nan)
    pv_avail = (w["PPV"].astype(float) / denom).fillna(w["PPV"].astype(float))
    ax0.step(w.index, pv_avail.values, label="PV available", color=COLORS["PV"], linestyle="--", where="mid")
    ax0.step(w.index, w["PPV"].astype(float).values, label="PPV", color=COLORS["PV"], where="mid")
    ax0.step(w.index, w["PLoad"].astype(float).values, label="PLoad", color=COLORS["Load"], where="mid")
    ax0.step(w.index, w["PGrid"].astype(float).values, label="PGrid", color=COLORS["Grid"], where="mid")
    ax0.bar(w.index, w["PBESS"].astype(float).values, width=bar_width, color=COLORS["BESS"], alpha=0.7, label="PBESS")
    ax0.bar(w.index, w["PEV"].astype(float).values, width=bar_width, color=COLORS["EV"], alpha=0.7, label="PEV (neg = V2G)")
    ax0.axhline(0.0, linewidth=0.8, color="k")
    ax0.set_ylabel("Power (kW)")
    ax0.grid(True, alpha=0.3)

    # Tariff on secondary axis
    axt = ax0.twinx()
    axt.step(w.index, w["tariff"].astype(float).values, label="Tariff", color=COLORS["Tariff"],
             linestyle=":", linewidth=1.4, where="mid")
    axt.set_ylabel("Tariff")
    h0, l0 = ax0.get_legend_handles_labels()
    ht, lt = axt.get_legend_handles_labels()
    ax0.legend(h0 + ht, l0 + lt, loc="upper left", ncol=4)

    for c in SOC_COLS:
        ax1.step(w.index, w[c].astype(float).values, label=c, where="mid")
    ax1.set_ylabel("SoC")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup_font()
    combos, runs_present = _discover()
    if not combos:
        print(f"No operation traces found under {TEST_ROOT}")
        return

    g = _global_window(runs_present)
    if g is None:
        print("Could not determine a representative window.")
        return
    _, g_tag, g_month, g_start = g
    print(f"Shared window: {g_tag} month {g_month:02d}  "
          f"{g_start.strftime('%Y-%m-%d')} +{WINDOW_DAYS}d  ({len(combos)} combos)")

    n = 0
    for (method, arch, tariff), modes in sorted(combos.items()):
        for mode in MODES:
            files = modes.get(mode, {})
            if not files:
                continue
            # Prefer the shared window; else fall back to this combo's own best.
            if (g_tag, g_month) in files:
                tag, month, start, path = g_tag, g_month, g_start, files[(g_tag, g_month)]
            else:
                (tag, month), path = sorted(files.items())[0]
                start = _best_window_start(tag, month)
            df = _read_trace(path)
            w = _slice(df, start)
            if w.empty:
                print(f"[skip] {method}/{arch}/{tariff}/{mode}: empty window")
                continue
            title = (f"{method} | {arch} | {tariff} | {mode} | "
                     f"{tag} {start.strftime('%Y-%m-%d')} +{WINDOW_DAYS}d")
            out_path = OUT_ROOT / method / arch / tariff / f"op_{mode}.{EXT}"
            _plot(w, tag, title, out_path)
            n += 1
    print(f"Saved {n} figures under {OUT_ROOT}")


if __name__ == "__main__":
    main()
