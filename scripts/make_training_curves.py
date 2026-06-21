"""Training learning curves (validation reward vs episode) from audit_training.csv.

Two figures, both 6 method-panels x 3 architecture series, on the lead tariff sw:
  Fig 1  training_seed42       : seed 42 only -- clean single lines (convergence shape).
  Fig 2  training_sw_3seeds    : seeds 42/7/13 -- mean +/- std band (seed robustness).

y = eval_mean_reward (the 24-window validation mean), x = episode (0..79).

  python scripts/make_training_curves.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "paper" / "training"
OUT.mkdir(parents=True, exist_ok=True)

_font_path = str(ROOT / "data" / "Gulliver.otf")
if Path(_font_path).exists():
    font_manager.fontManager.addfont(_font_path)
    plt.rcParams["font.family"] = "Gulliver"
plt.rcParams["axes.unicode_minus"] = False

METHODS = [("u_penalty", "Penalty"), ("u_cmdp", "CMDP"), ("u_cmdp_shaped", "CMDP+PBRS"),
           ("u_cmdp_droq", "CMDP+DroQ"), ("u_cmdp_redq", "CMDP+REDQ"), ("u_cmdp_ft", "CMDP+FT")]
ARCHS = ["GRU", "MHA", "TCN"]
# acronyms here: the compact single-column legend cannot fit the full encoder names
ARCH_LABEL = {"GRU": "GRU", "MHA": "MHA", "TCN": "TCN"}
COLORS = {"GRU": "#1f77b4", "MHA": "#d62728", "TCN": "#2ca02c"}
SEED_DIRS = {42: "tar_sw", 7: "tar_sw-s7", 13: "tar_sw-s13"}
METRIC = "eval_mean_reward"
SMOOTH = 3  # light centered rolling mean: kills single-episode eval spikes, keeps the
            # multi-episode dips (which recover) and the FT actor-start jump.


def _curve(folder, arch, tariff_dir):
    p = ROOT / "paper" / "train" / folder / arch / tariff_dir / "audit_training.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)[["episode", METRIC]].dropna()
    d[METRIC] = d[METRIC].rolling(SMOOTH, center=True, min_periods=1).mean()
    return d


def _ylim(curves):
    lo = np.nanpercentile(np.concatenate(curves), 2)
    hi = np.nanpercentile(np.concatenate(curves), 100)
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _grid(figname, draw_panel):
    # 3x2 single-column layout, compact to one column width
    fig, axes = plt.subplots(3, 2, figsize=(3.4, 4.7), sharex=True, sharey=True)
    all_vals = []
    for ax, (folder, lbl) in zip(axes.flat, METHODS):
        vals = draw_panel(ax, folder)
        all_vals.extend(vals)
        ax.set_title(lbl, fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        ax.grid(True, lw=0.3, alpha=0.4)
    if all_vals:
        lo, hi = _ylim(all_vals)
        axes.flat[0].set_ylim(lo, hi)
    for ax in axes[-1]:
        ax.set_xlabel("Episode", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Val. reward", fontsize=7)
    handles = [plt.Line2D([0], [0], color=COLORS[a], lw=2, label=ARCH_LABEL[a]) for a in ARCHS]
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.savefig(OUT / f"{figname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{figname}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] wrote {figname}.pdf/.png")


def fig_seed42():
    def panel(ax, folder):
        vals = []
        for a in ARCHS:
            c = _curve(folder, a, SEED_DIRS[42])
            if c is None:
                continue
            ax.plot(c["episode"], c[METRIC], color=COLORS[a], lw=1.6)
            vals.append(c[METRIC].to_numpy())
        return vals
    _grid("training_seed42", panel)


def fig_3seeds():
    def panel(ax, folder):
        vals = []
        for a in ARCHS:
            series = [_curve(folder, a, d) for d in SEED_DIRS.values()]
            series = [s for s in series if s is not None]
            if not series:
                continue
            m = pd.concat([s.set_index("episode")[METRIC] for s in series], axis=1)
            mean, std = m.mean(axis=1), m.std(axis=1)
            ax.plot(mean.index, mean.values, color=COLORS[a], lw=1.6)
            ax.fill_between(mean.index, mean - std, mean + std, color=COLORS[a], alpha=0.18, lw=0)
            vals.append(mean.to_numpy())
        return vals
    _grid("training_sw_3seeds", panel)


def main():
    fig_seed42()
    fig_3seeds()
    print(f"[train] -> {OUT}")


if __name__ == "__main__":
    main()
