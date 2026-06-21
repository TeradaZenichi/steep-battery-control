"""Four operation figures into paper/paper/ (Gulliver font, GRU traces).

  op1_projection_reliance - native grid power vs grid limits, FT vs DroQ: the
      demonstration-based policy's raw actions breach the export limit (clamped by
      the projection); the demonstration-free policy stays feasible by itself.
  op2_dispatch            - representative 48 h dispatch (PV/Load/Grid + BESS/EV) and
      SoC for the demonstration-free method (DroQ).
  op3_bess_conservative   - BESS power + SoC, DroQ vs FT: less cycling (degradation).
  op4_tariff_adaptive     - DroQ operation across flat / ToU / sol+wind tariffs.
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates, font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "paper"
TEST = ROOT / "paper" / "test"
PMAX_IMP, PMAX_EXP = 10.0, 2.0

_fp = str(ROOT / "data" / "Gulliver.otf")
font_manager.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "Gulliver"
plt.rcParams["font.sans-serif"] = font_manager.FontProperties(fname=_fp).get_name()
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})
C = {"BESS": "#1f77b4", "EV": "#ff7f0e", "PV": "#2ca02c", "Load": "#d62728", "Grid": "#9467bd"}


def load(method, tariff, scenario, arch="GRU"):
    f = TEST / method / arch / tariff / "operations" / f"{scenario}_best_raw.csv.gz"
    return pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()


def win(df, hours=48, around=None):
    if around is None:
        t0 = df.index[0]
    else:
        t0 = around - pd.Timedelta(hours=hours / 2)
    return df.loc[t0:t0 + pd.Timedelta(hours=hours)]


def fmt_x(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.grid(True, alpha=0.3)


# ---- op1: projection reliance -------------------------------------------------
def op1():
    scn = "test_wy_12"
    ft, dq = load("u_cmdp_ft", "tar_sw", scn), load("u_cmdp_droq", "tar_sw", scn)
    vt = (np.maximum(-PMAX_EXP - ft["PGrid"], 0) + np.maximum(ft["PGrid"] - PMAX_IMP, 0))
    wft = win(ft, 48, around=vt.idxmax())
    wdq = dq.loc[wft.index[0]:wft.index[-1]]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, sharey=True)
    for ax, w, name in ((axes[0], wft, "CMDP+FT (demonstration-based)"),
                        (axes[1], wdq, "CMDP+DroQ (demonstration-free)")):
        ax.step(w.index, w["PGrid"], color=C["Grid"], where="mid", label="Grid power")
        ax.axhline(-PMAX_EXP, color="r", ls="--", lw=1, label="grid limits")
        ax.axhline(PMAX_IMP, color="r", ls="--", lw=1)
        ax.fill_between(w.index, w["PGrid"], -PMAX_EXP, where=(w["PGrid"] < -PMAX_EXP),
                        color="r", alpha=0.35, step="mid", label="native violation")
        v = (np.maximum(-PMAX_EXP - w["PGrid"], 0) + np.maximum(w["PGrid"] - PMAX_IMP, 0)).sum()
        ax.set_title(f"{name}  -  native violation = {v:.2f} kWh", fontsize=10)
        ax.set_ylabel("Grid power (kW)")
        fmt_x(ax)
    axes[0].legend(loc="lower right", ncol=3, fontsize=8, framealpha=0.9)
    axes[1].set_xlabel("Time")
    fig.tight_layout()
    _save(fig, "op1_projection_reliance")


# ---- dispatch helper (power + SoC) -------------------------------------------
def _dispatch(w, title, name):
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.5, 1]))
    bw = (w.index[1] - w.index[0]).total_seconds() / 86400.0 * 0.8
    a0.step(w.index, w["PPV"], color=C["PV"], where="mid", label="PV")
    a0.step(w.index, w["PLoad"], color=C["Load"], where="mid", label="Load")
    a0.step(w.index, w["PGrid"], color=C["Grid"], where="mid", label="Grid")
    a0.bar(w.index, w["PBESS"], width=bw, color=C["BESS"], alpha=0.7, label="BESS")
    a0.bar(w.index, w["PEV"], width=bw, color=C["EV"], alpha=0.7, label="EV")
    a0.axhline(0, color="k", lw=0.8, alpha=0.5)
    a0.set_ylabel("Power (kW)"); a0.set_title(title)
    a0.legend(loc="upper right", ncol=3, fontsize=8); a0.grid(True, alpha=0.3)
    a1.step(w.index, w["SoCBESS"], color=C["BESS"], where="mid", label="SoC BESS")
    a1.step(w.index, w["SoCEV"], color=C["EV"], where="mid", label="SoC EV")
    a1.axhline(w["SoCBESS"].max(), ls=":", color=C["BESS"], lw=1, alpha=0.7)
    a1.set_ylim(-0.05, 1.05); a1.set_ylabel("SoC"); a1.set_xlabel("Time")
    a1b = a1.twinx()  # import price on the secondary axis (arbitrage logic)
    a1b.step(w.index, w["tariff"], color="0.35", lw=1.3, ls="--", where="mid", label="import price")
    a1b.set_ylabel("import price (€/kWh)", color="0.35")
    a1b.tick_params(axis="y", labelcolor="0.35")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a1b.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, framealpha=0.9)
    fmt_x(a1)
    fig.tight_layout()
    _save(fig, name)


def op2():
    _dispatch(win(load("u_cmdp_droq", "tar_sw", "test_cy_03"), 48),
              "CMDP+DroQ - 48 h dispatch (tar_sw, high-PV window)", "op2_dispatch")


def op6():
    df = load("u_cmdp_droq", "tar_sw", "test_wy_03")
    w = win(df, 48, around=df["SoCBESS"].idxmax())
    _dispatch(w, f"CMDP+DroQ - BESS charged to maximum (SoC -> {df['SoCBESS'].max():.2f}, tar_sw)",
              "op6_bess_full_charge")


# ---- op3: BESS cycling per method (aggregate, the year-long effect) ----------
def op3():
    archs = ["GRU", "MHA", "TCN"]
    tars = ["tar_flat", "tar_s", "tar_w", "tar_sw", "tar_tou"]
    methods = [("u_penalty", "Penalty", "0.6"), ("u_cmdp", "CMDP", "0.6"),
               ("u_cmdp_shaped", "+PBRS", "0.6"), ("u_cmdp_droq", "+DroQ", C["BESS"]),
               ("u_cmdp_redq", "+REDQ", C["BESS"]), ("u_cmdp_ft", "+FT", C["Load"])]

    def rec(m, a, t):
        for r in json.load(open(TEST / m / a / t / "summary_overall.json", encoding="utf-8")):
            if r.get("checkpoint") == "best" and r.get("mode") == "safe":
                return r

    labels, thru, cost, colors = [], [], [], []
    for m, lbl, c in methods:
        labels.append(lbl); colors.append(c)
        thru.append(st.mean(rec(m, a, t)["mean_bess_charge_kwh"] + rec(m, a, t)["mean_bess_discharge_kwh"]
                            for a in archs for t in tars))
        cost.append(st.mean(rec(m, a, t)["mean_bess_cost"] for a in archs for t in tars))
    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(labels)))
    ax.bar(x, thru, color=colors, edgecolor="k", linewidth=0.5)
    for xi, tv in zip(x, thru):
        ax.text(xi, tv + 2, f"{tv:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("BESS throughput (kWh, mean)")
    ax.set_title("Battery cycling per method (mean over 3 encoders × 5 tariffs)")
    ax.grid(True, axis="y", alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(x, cost, "D", color="k", ms=6)
    ax2.set_ylabel("degradation cost (◆)")
    fig.tight_layout()
    _save(fig, "op3_bess_cycling")


# ---- op4: tariff-adaptive -----------------------------------------------------
def op4():
    tariffs = [("tar_flat", "flat"), ("tar_tou", "ToU"), ("tar_sw", "sol+wind")]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True)
    for ax, (t, lbl) in zip(axes, tariffs):
        w = win(load("u_cmdp_droq", t, "test_cy_07"), 48)
        bw = (w.index[1] - w.index[0]).total_seconds() / 86400.0 * 0.8
        ax.bar(w.index, w["PBESS"], width=bw, color=C["BESS"], alpha=0.8, label="BESS")
        ax.step(w.index, w["PGrid"], color=C["Grid"], where="mid", lw=1, label="Grid")
        ax.axhline(0, color="k", lw=0.8, alpha=0.5)
        ax.set_ylabel(f"{lbl}\n(kW)"); ax.grid(True, alpha=0.3)
    axes[0].set_title("CMDP+DroQ operation across tariffs (48 h, July)")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time"); fmt_x(axes[-1])
    fig.tight_layout()
    _save(fig, "op4_tariff_adaptive")


# ---- op5: structural export, PV surplus vastly exceeds BESS capacity ---------
def op5():
    import glob, os
    dt = 5 / 60.0
    BESS_USABLE = 8.0  # Emax 10 x DoD 0.8
    scns = sorted(os.path.basename(p).replace("_best_raw.csv.gz", "")
                  for p in glob.glob(str(TEST / "u_cmdp_droq" / "GRU" / "tar_sw" / "operations" / "*_best_raw.csv.gz")))
    sur, exp = [], []
    for scn in scns:
        w = load("u_cmdp_droq", "tar_sw", scn)
        sur.append(float((np.maximum(w["PPV"] - w["PLoad"], 0) * dt).sum()))
        exp.append(float((np.maximum(-w["PGrid"], 0) * dt).sum()))
    order = np.argsort(sur)
    sur, exp = np.array(sur)[order], np.array(exp)[order]
    x = np.arange(len(sur))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.fill_between(x, exp, sur, color=C["PV"], alpha=0.45, label="self-consumed (BESS + EV)")
    ax.fill_between(x, 0, exp, color=C["Load"], alpha=0.45, label="exported to grid (paid 0.2x)")
    ax.plot(x, sur, color="k", lw=1.2, label="PV surplus")
    ax.axhline(BESS_USABLE, ls="--", color=C["BESS"], lw=1.4, label="BESS usable (~8 kWh)")
    ax.set_xlabel("test windows (sorted by PV surplus)")
    ax.set_ylabel("energy per window (kWh)")
    ax.set_title("PV surplus vastly exceeds storage: most surplus must be exported")
    ax.set_xlim(0, len(sur) - 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "op5_structural_export")


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    op1(); op2(); op3(); op4(); op5(); op6()
