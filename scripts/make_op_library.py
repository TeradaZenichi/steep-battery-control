"""Cross-method comparison operation figure (fig15) into paper/paper/opt/.

Layout:
  opt/{arch}/{tariff}/compare.pdf  - one column per method over the same 48 h window:
      power (PV/Grid + BESS/EV bars) on top, SoC (BESS, EV) with import price below.

The per-method browsing library (dispatch/bess_max/projection) was pruned; only the
paper's cross-method comparison figure is generated here.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates, font_manager

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "paper" / "test"
OUT = ROOT / "paper" / "paper" / "opt"
SCN_DISPATCH = "test_cy_03"

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
        for tariff in TARIFFS:
            fig_compare(arch, tariff, OUT / arch / tariff / "compare.png")
        print(f"== {arch} compares done ==", flush=True)


if __name__ == "__main__":
    main()
