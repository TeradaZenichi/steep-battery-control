"""CMDP constraint-satisfaction diagnostics on the lead tariff (reproducible).

Shows that the active EV departure-risk constraint is satisfied during training:
  (a) the constrained quantity [Q_EV]_+ (ev_q_cost_pi_pos_mean) vs the budget b_EV;
  (b) the dual multiplier lambda_EV (ev_lambda_value) trajectory.

Both are read from audit_training.csv (u_cmdp, lead tariff tar_sw, three encoders).

  python scripts/make_cmdp_diagnostics.py
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
TR = ROOT / "paper" / "train" / "u_cmdp"
OUT = ROOT / "paper" / "paper"
ARCHS = ["GRU", "MHA", "TCN"]
SEED = "tar_sw"
BUDGET = 0.01            # b_EV from config.json (cmdp.costs.ev.budget)
JC = "ev_q_cost_pi_pos_mean"
LAM = "ev_lambda_value"
SMOOTH = 3

_fp = str(ROOT / "data" / "Gulliver.otf")
if Path(_fp).exists():
    font_manager.fontManager.addfont(_fp)
    plt.rcParams["font.family"] = "Gulliver"
    plt.rcParams["font.sans-serif"] = font_manager.FontProperties(fname=_fp).get_name()
plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False})

COLORS = {"GRU": "#1f77b4", "MHA": "#2ca02c", "TCN": "#d62728"}


def load(arch, col):
    p = TR / arch / SEED / "audit_training.csv"
    if not p.exists():
        return None, None
    df = pd.read_csv(p)
    s = pd.to_numeric(df[col], errors="coerce")
    ep = pd.to_numeric(df["episode"], errors="coerce")
    m = s.notna() & ep.notna()
    return ep[m].to_numpy(), s[m].rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()


def main():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    print("Final [Q_EV]_+ (last-10-episode mean) vs budget b_EV =", BUDGET)
    for a in ARCHS:
        ep, jc = load(a, JC)
        if ep is not None:
            axa.plot(ep, jc, color=COLORS[a], lw=1.6, label=a)
            print(f"  {a}: {np.mean(jc[-10:]):.4f}  ({'within' if np.mean(jc[-10:]) <= BUDGET else 'ABOVE'} budget)")
        ep, lam = load(a, LAM)
        if ep is not None:
            axb.plot(ep, lam, color=COLORS[a], lw=1.6, label=a)

    axa.axhline(BUDGET, color="#222222", ls="--", lw=1.2)
    axa.text(0.98, BUDGET, r" $b_{\mathrm{EV}}$", va="bottom", ha="right",
             transform=axa.get_yaxis_transform(), fontsize=9)
    axa.set_xlabel("episode"); axa.set_ylabel(r"$[\,\widehat{Q}_{\mathrm{EV}}\,]_+$  (constrained cost)")
    axa.set_title("(a) EV constraint vs budget", fontsize=10.5)
    axb.set_xlabel("episode"); axb.set_ylabel(r"$\lambda_{\mathrm{EV}}$")
    axb.set_title("(b) dual multiplier", fontsize=10.5)
    for ax in (axa, axb):
        ax.grid(True, alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"cmdp_diagnostics.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote cmdp_diagnostics.{pdf,png} to", OUT)


if __name__ == "__main__":
    main()
