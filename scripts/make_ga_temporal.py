"""Graphical-abstract temporal panel (ga_temporal) with the CORRECT native-violation
energy in kWh (integrated over the 5-min step, i.e. summed kW * dt).

Two panels on a representative 48 h window (GRU encoder, tar_sw):
  left  - behavior-cloning fine-tuning: raw policy vs safety-projected action; the
          raw actions breach the export limit, so it "needs the projection".
  right - demonstration-free policy: stays inside the limits by itself.

Rationale: the previous hand-made panel summed instantaneous kW WITHOUT the * dt
factor and mislabeled the result as kWh (reported 7.94 kWh; the integrated value is
0.66 kWh, which is what Fig. 10 / op1 already reports). This script restores a
versioned, reproducible generator with the correct energy.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "paper"
TEST = ROOT / "paper" / "test"
PMAX_IMP, PMAX_EXP = 10.0, 2.0
SCN = "test_wy_12"
RAW_C, PROJ_C, DF_C, LIM_C = "#9467bd", "#000000", "#2ca02c", "#d62728"

_fp = str(ROOT / "data" / "Gulliver.otf")
font_manager.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "Gulliver"
plt.rcParams["font.sans-serif"] = font_manager.FontProperties(fname=_fp).get_name()
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})


def load(method, mode, tariff="tar_sw", arch="GRU"):
    f = TEST / method / arch / tariff / "operations" / f"{SCN}_best_{mode}.csv.gz"
    return pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()


def viol_kwh(pgrid, dt):
    """Native grid-limit violation as ENERGY: sum of instantaneous overshoot (kW) * dt (h)."""
    return float(((np.maximum(-PMAX_EXP - pgrid, 0) + np.maximum(pgrid - PMAX_IMP, 0)) * dt).sum())


def hours(idx):
    return (idx - idx[0]).total_seconds() / 3600.0


def main():
    ft_raw, ft_safe = load("u_cmdp_ft", "raw"), load("u_cmdp_ft", "safe")
    df_raw = load("u_cmdp_droq", "raw")
    dt = (ft_raw.index[1] - ft_raw.index[0]).total_seconds() / 3600.0  # 5 min = 1/12 h

    # 48 h window centered on the FT native-violation peak (same window as Fig. 10 / op1)
    vt = np.maximum(-PMAX_EXP - ft_raw["PGrid"], 0) + np.maximum(ft_raw["PGrid"] - PMAX_IMP, 0)
    around = vt.idxmax()
    t0, t1 = around - pd.Timedelta(hours=24), around + pd.Timedelta(hours=24)
    wr, ws, wd = ft_raw.loc[t0:t1], ft_safe.loc[t0:t1], df_raw.loc[t0:t1]

    v_ft, v_df = viol_kwh(wr["PGrid"], dt), viol_kwh(wd["PGrid"], dt)

    # wide-and-short aspect so the two panels sit as a slim strip in the graphical abstract
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(14, 1.9), sharey=True)

    aL.step(hours(wr.index), wr["PGrid"], color=RAW_C, where="mid", lw=1.3, label="raw policy")
    aL.step(hours(ws.index), ws["PGrid"], color=PROJ_C, ls=":", where="mid", lw=1.2,
            label="after safety projection")
    aL.set_title(f"behavior cloning (fine-tuning): native violation = {v_ft:.2f} kWh  "
                 f"(needs the projection)", color=LIM_C, fontsize=9)
    aL.legend(loc="lower right", ncol=2, fontsize=7.5, framealpha=0.9)

    aR.step(hours(wd.index), wd["PGrid"], color=DF_C, where="mid", lw=1.3)
    aR.set_title(f"demonstration-free policy: native violation = {v_df:.2f} kWh  "
                 f"(stays inside the limits)", color=DF_C, fontsize=9)

    for ax in (aL, aR):
        ax.axhline(-PMAX_EXP, color=LIM_C, ls="--", lw=1)
        ax.axhline(PMAX_IMP, color=LIM_C, ls="--", lw=1)
        ax.set_yticks([-PMAX_EXP, 0, PMAX_IMP]); ax.set_yticklabels(["-2", "0", "+10"])
        ax.set_xticks([0, 12, 24, 36, 48]); ax.set_xlim(0, 48)
        ax.set_xlabel("time (h)"); ax.grid(True, alpha=0.3)
    aL.set_ylabel("grid power (kW)")

    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"ga_temporal.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote ga_temporal.{{svg,pdf,png}}  ->  FT={v_ft:.2f} kWh (was mislabeled 7.94), "
          f"demo-free={v_df:.2f} kWh")


if __name__ == "__main__":
    main()
