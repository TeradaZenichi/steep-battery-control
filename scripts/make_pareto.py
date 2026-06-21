"""Cross-architecture Pareto: operating cost (x) vs native grid violation (y).

Demonstration-free methods cluster near y=0 (low native violation); CMDP+FT sits
higher on the violation axis (it relies on the feasibility projection). One panel
per encoder.

Two figures:
  pareto_cost_violation        - 5-tariff mean, seed 42 (qualitative, all tariffs).
  pareto_cost_violation_seed   - lead tariff (tar_sw), n=3 seeds, mean +/- std error
                                 bars on both axes (the violation separation persists
                                 under seed variance).
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "paper"

# -- Gulliver font (paper-wide standard; see existing paper/analysis plots) --
_font_path = str(ROOT / "data" / "Gulliver.otf")
font_manager.fontManager.addfont(_font_path)
_prop = font_manager.FontProperties(fname=_font_path)
plt.rcParams["font.family"] = "Gulliver"
plt.rcParams["font.sans-serif"] = _prop.get_name()

ARCHS = ["GRU", "MHA", "TCN"]
ARCH_LABEL = {"GRU": "GRU", "MHA": "Multi-Head Attention", "TCN": "TCN"}
BASE = ["tar_flat", "tar_s", "tar_w", "tar_sw", "tar_tou"]
SW3 = ["tar_sw", "tar_sw-s7", "tar_sw-s13"]
METHODS = [
    ("u_penalty", "Penalty", "#7f7f7f", "o"),
    ("u_cmdp", "CMDP", "#1f77b4", "o"),
    ("u_cmdp_shaped", "+PBRS", "#2ca02c", "s"),
    ("u_cmdp_droq", "+DroQ", "#9467bd", "^"),
    ("u_cmdp_redq", "+REDQ", "#17becf", "v"),
    ("u_cmdp_ft", "+FT (demo)", "#d62728", "*"),
]


def rec(method, arch, tariff, mode):
    p = ROOT / "paper" / "test" / method / arch / tariff / "summary_overall.json"
    for r in json.load(open(p, encoding="utf-8")):
        if r.get("checkpoint") == "best" and r.get("mode") == mode:
            return r


def vals(method, arch, key, mode, tariffs):
    return [rec(method, arch, t, mode)[key] for t in tariffs]


def _decorate(axes, fig):
    axes[0].set_ylabel("native grid violation [kWh]\n(lower better)")
    axes[0].annotate("intrinsically\nfeasible", (0.02, 0.02), xycoords="axes fraction",
                     fontsize=8, color="#2ca02c", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.tight_layout()


def pareto_mean():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, arch in zip(axes, ARCHS):
        for m, lbl, c, mk in METHODS:
            x = -st.mean(vals(m, arch, "mean_reward", "safe", BASE))
            y = st.mean(vals(m, arch, "mean_grid_violation_kwh", "raw", BASE))
            ax.scatter(x, y, s=260 if mk == "*" else 90, c=c, marker=mk,
                       edgecolors="k", linewidths=0.6, zorder=3, label=lbl)
        ax.axhspan(-0.005, 0.02, color="#2ca02c", alpha=0.08, zorder=0)
        ax.set_title(ARCH_LABEL[arch], fontsize=11, fontweight="bold")
        ax.set_xlabel("operating cost  (lower better)")
        ax.grid(True, alpha=0.3)
    _decorate(axes, fig)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"pareto_cost_violation.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def pareto_seed():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, arch in zip(axes, ARCHS):
        for m, lbl, c, mk in METHODS:
            rx = [-v for v in vals(m, arch, "mean_reward", "safe", SW3)]
            ry = vals(m, arch, "mean_grid_violation_kwh", "raw", SW3)
            x, xe = st.mean(rx), st.pstdev(rx)
            y, ye = st.mean(ry), st.pstdev(ry)
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=mk, color=c,
                        markersize=15 if mk == "*" else 9, markeredgecolor="k",
                        markeredgewidth=0.6, ecolor="0.5", elinewidth=1,
                        capsize=3, zorder=3, label=lbl)
        ax.axhspan(-0.005, 0.02, color="#2ca02c", alpha=0.08, zorder=0)
        ax.set_title(ARCH_LABEL[arch], fontsize=11, fontweight="bold")
        ax.set_xlabel("operating cost  (lower better)")
        ax.grid(True, alpha=0.3)
    _decorate(axes, fig)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"pareto_cost_violation_seed.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    pareto_mean()
    pareto_seed()
    print("wrote pareto_cost_violation{,_seed}.{pdf,png} to", OUT)
