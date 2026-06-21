"""Operation-plot library into paper/paper/opt/.

Layout:
  opt/{arch}/{method}/{tariff}/dispatch.png    - PV/Load/BESS/EV/Grid + SoC + price
  opt/{arch}/{method}/{tariff}/bess_max.png     - window around the BESS SoC peak
  opt/{arch}/{method}/{tariff}/projection.png   - native grid power vs limits + violation
  opt/{arch}/{tariff}/compare.png               - one column per method (same window)

PNG only (browse library). Fixed representative windows keep it fast:
  dispatch/bess_max from a high-PV window; projection from a violation-prone window.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates, font_manager

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "paper" / "test"
OUT = ROOT / "paper" / "paper" / "opt"
PMAX_IMP, PMAX_EXP = 10.0, 2.0
SCN_DISPATCH, SCN_VIOL = "test_cy_03", "test_wy_12"

_fp = str(ROOT / "data" / "Gulliver.otf")
font_manager.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "Gulliver"
plt.rcParams["font.sans-serif"] = font_manager.FontProperties(fname=_fp).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
C = {"BESS": "#1f77b4", "EV": "#ff7f0e", "PV": "#2ca02c", "Load": "#d62728", "Grid": "#9467bd"}
ARCHS = ["GRU", "MHA", "TCN"]
METHODS = [("u_penalty", "Penalty"), ("u_cmdp", "CMDP"), ("u_cmdp_shaped", "CMDP+PBRS"),
           ("u_cmdp_droq", "CMDP+DroQ"), ("u_cmdp_redq", "CMDP+REDQ"), ("u_cmdp_ft", "CMDP+FT")]
TARIFFS = ["tar_flat", "tar_s", "tar_w", "tar_sw", "tar_tou"]


def load(method, arch, tariff, scn, mode="raw"):
    f = TEST / method / arch / tariff / "operations" / f"{scn}_best_{mode}.csv.gz"
    if not f.exists():
        return None
    return pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp").sort_index()


def win(df, hours=48, around=None):
    t0 = df.index[0] if around is None else around - pd.Timedelta(hours=hours / 2)
    return df.loc[t0:t0 + pd.Timedelta(hours=hours)]


def fmt_x(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.grid(True, alpha=0.3)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")              # png (browse)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")   # pdf (paper)
    plt.close(fig)


def fig_dispatch(w, title, path):
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
    bw = (w.index[1] - w.index[0]).total_seconds() / 86400.0 * 0.8
    a0.step(w.index, w["PPV"], color=C["PV"], where="mid", label="PV")
    a0.step(w.index, w["PLoad"], color=C["Load"], where="mid", label="Load")
    a0.step(w.index, w["PGrid"], color=C["Grid"], where="mid", label="Grid")
    a0.bar(w.index, w["PBESS"], width=bw, color=C["BESS"], alpha=0.7, label="BESS")
    a0.bar(w.index, w["PEV"], width=bw, color=C["EV"], alpha=0.7, label="EV")
    a0.axhline(0, color="k", lw=0.7, alpha=0.5)
    a0.set_ylabel("Power (kW)"); a0.set_title(title, fontsize=9)
    a0.legend(loc="upper right", ncol=3, fontsize=7); a0.grid(True, alpha=0.3)
    a1.step(w.index, w["SoCBESS"], color=C["BESS"], where="mid", label="SoC BESS")
    a1.step(w.index, w["SoCEV"], color=C["EV"], where="mid", label="SoC EV")
    a1.set_ylim(-0.05, 1.05); a1.set_ylabel("SoC"); a1.set_xlabel("Time")
    a1b = a1.twinx()
    a1b.step(w.index, w["tariff"], color="0.35", lw=1.1, ls="--", where="mid", label="price")
    a1b.set_ylabel("price (EUR/kWh)", color="0.35"); a1b.tick_params(axis="y", labelcolor="0.35")
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a1b.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=7, framealpha=0.9)
    fmt_x(a1); fig.tight_layout(); save(fig, path)


def fig_projection(method, arch, tariff, title, path):
    """Two panels: (top) native grid power vs limits with violation shading (raw
    rollout); (bottom) raw vs projected BESS/EV action (safe rollout) - how much the
    feasibility projection corrects the policy's actions."""
    rw = load(method, arch, tariff, SCN_VIOL, "raw")
    sw = load(method, arch, tariff, SCN_VIOL, "safe")
    if rw is None:
        return
    vt = (np.maximum(-PMAX_EXP - rw["PGrid"], 0) + np.maximum(rw["PGrid"] - PMAX_IMP, 0))
    around = vt.idxmax() if vt.sum() > 0 else None
    wr = win(rw, 48, around=around)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True)
    a0.step(wr.index, wr["PGrid"], color=C["Grid"], where="mid", label="Grid power")
    a0.axhline(-PMAX_EXP, color="r", ls="--", lw=1, label="grid limits"); a0.axhline(PMAX_IMP, color="r", ls="--", lw=1)
    a0.fill_between(wr.index, wr["PGrid"], -PMAX_EXP, where=(wr["PGrid"] < -PMAX_EXP), color="r", alpha=0.35, step="mid")
    a0.fill_between(wr.index, wr["PGrid"], PMAX_IMP, where=(wr["PGrid"] > PMAX_IMP), color="r", alpha=0.35, step="mid")
    v = (np.maximum(-PMAX_EXP - wr["PGrid"], 0) + np.maximum(wr["PGrid"] - PMAX_IMP, 0)).sum()
    a0.set_title(f"{title}  -  native violation = {v:.2f} kWh", fontsize=9)
    a0.set_ylabel("Grid power (kW)"); a0.legend(loc="lower right", fontsize=7); a0.grid(True, alpha=0.3)
    if sw is not None:
        ws = sw.loc[wr.index[0]:wr.index[-1]]
        a1.step(ws.index, ws["raw_bess_cmd"], color=C["BESS"], ls="--", where="mid", lw=1, label="BESS raw")
        a1.step(ws.index, ws["bess_cmd"], color=C["BESS"], where="mid", lw=1.4, label="BESS proj.")
        a1.step(ws.index, ws["raw_ev_cmd"], color=C["EV"], ls="--", where="mid", lw=1, label="EV raw")
        a1.step(ws.index, ws["ev_cmd"], color=C["EV"], where="mid", lw=1.4, label="EV proj.")
        corr = (np.abs(ws["raw_bess_cmd"] - ws["bess_cmd"]) + np.abs(ws["raw_ev_cmd"] - ws["ev_cmd"])).sum()
        a1.set_title(f"raw vs projected action (total correction = {corr:.2f})", fontsize=9)
        a1.set_ylabel("action (norm.)"); a1.legend(loc="lower right", ncol=2, fontsize=6.5); a1.grid(True, alpha=0.3)
    a1.set_xlabel("Time"); fmt_x(a1); fig.tight_layout(); save(fig, path)


def fig_compare(arch, tariff, path):
    """Paper-quality, 2-column-width (~7.5in) method comparison: power + (SoC BESS,
    SoC EV, import price). One column per method. Saved as PDF + PNG."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    cols = [(m, lbl, load(m, arch, tariff, SCN_DISPATCH)) for m, lbl in METHODS]
    cols = [c for c in cols if c[2] is not None]
    if not cols:
        return
    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(7.5, 2.5), sharex="col", sharey="row",
                             gridspec_kw=dict(height_ratios=[1.3, 1]))
    w0 = win(cols[0][2], 48)  # price is identical across methods -> shared scale
    pmin, pmax = float(w0["tariff"].min()), float(w0["tariff"].max())
    for j, (m, lbl, df) in enumerate(cols):
        w = win(df, 48)
        bw = (w.index[1] - w.index[0]).total_seconds() / 86400.0 * 0.8
        a0, a1 = axes[0, j], axes[1, j]
        a0.step(w.index, w["PPV"], color=C["PV"], where="mid", lw=0.7)
        a0.step(w.index, w["PGrid"], color=C["Grid"], where="mid", lw=0.7)
        a0.bar(w.index, w["PBESS"], width=bw, color=C["BESS"], alpha=0.7)
        a0.bar(w.index, w["PEV"], width=bw, color=C["EV"], alpha=0.7)
        a0.axhline(0, color="k", lw=0.5, alpha=0.5)
        a0.set_title(lbl, fontsize=8); a0.grid(True, alpha=0.3); a0.tick_params(labelsize=6)
        a1.step(w.index, w["SoCBESS"], color=C["BESS"], where="mid", lw=0.9)
        a1.step(w.index, w["SoCEV"], color=C["EV"], where="mid", lw=0.9)
        a1.set_ylim(-0.05, 1.05); a1.grid(True, alpha=0.3); a1.tick_params(labelsize=6)
        ab = a1.twinx()
        ab.step(w.index, w["tariff"], color="0.35", lw=0.8, ls="--", where="mid")
        ab.set_ylim(pmin * 0.95, pmax * 1.05); ab.tick_params(labelsize=6, labelcolor="0.35")
        if j < n - 1:
            ab.set_yticklabels([])
        else:
            ab.set_ylabel("price\n(EUR/kWh)", fontsize=6, color="0.35")
        a1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        a1.xaxis.set_major_locator(mdates.DayLocator())
        a1.tick_params(axis="x", labelsize=6)
    axes[0, 0].set_ylabel("Power (kW)", fontsize=7)
    axes[1, 0].set_ylabel("SoC", fontsize=7)
    leg = [Line2D([], [], color=C["PV"], lw=1.2, label="PV"),
           Line2D([], [], color=C["Grid"], lw=1.2, label="Grid"),
           Patch(facecolor=C["BESS"], alpha=0.7, label="BESS"),
           Patch(facecolor=C["EV"], alpha=0.7, label="EV"),
           Line2D([], [], color=C["BESS"], lw=1.2, label="SoC BESS"),
           Line2D([], [], color=C["EV"], lw=1.2, label="SoC EV"),
           Line2D([], [], color="0.35", ls="--", lw=1, label="price")]
    fig.tight_layout()
    # legend sits just above the panels; arch/tariff is left to the figure caption
    fig.legend(handles=leg, loc="upper center", ncol=7, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    for arch in ARCHS:
        for m, lbl in METHODS:
            for tariff in TARIFFS:
                d = OUT / arch / m / tariff
                dd = load(m, arch, tariff, SCN_DISPATCH)
                if dd is not None:
                    fig_dispatch(win(dd, 48), f"{arch} / {lbl} / {tariff} - dispatch", d / "dispatch.png")
                    pk = dd["SoCBESS"].idxmax()
                    fig_dispatch(win(dd, 48, around=pk),
                                 f"{arch} / {lbl} / {tariff} - BESS max (SoC->{dd['SoCBESS'].max():.2f})",
                                 d / "bess_max.png")
                fig_projection(m, arch, tariff, f"{arch} / {lbl} / {tariff}", d / "projection.png")
            print(f"  {arch}/{lbl}: done", flush=True)
        for tariff in TARIFFS:
            fig_compare(arch, tariff, OUT / arch / tariff / "compare.png")
        print(f"== {arch} complete ==", flush=True)


if __name__ == "__main__":
    import sys
    if "--compare-only" in sys.argv:
        for arch in ARCHS:
            for tariff in TARIFFS:
                fig_compare(arch, tariff, OUT / arch / tariff / "compare.png")
            print(f"  {arch} compares done", flush=True)
    else:
        main()
