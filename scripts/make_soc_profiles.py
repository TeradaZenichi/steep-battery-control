"""Typical SoC operation profiles (mean diurnal cycle) into paper/paper/soc/.

For each (method, arch, tariff) the mean daily profile of SoC BESS and SoC EV is
computed by averaging over all 24 held-out windows x 7 days, grouped by time-of-day
(5-min resolution), from the best/raw checkpoint (same source as the compare figure).

Three layouts (all generated):
  V1: rows=arch(3), cols=tariff(5), 6 method lines   -> soc_bess_v1.pdf, soc_ev_v1.pdf
  V2: rows=method(6), cols=tariff(5), 3 arch lines    -> soc_bess_v2.pdf, soc_ev_v2.pdf
  V3: rows=arch(3), cols=tariff(5), BESS solid + EV dashed, 6 methods -> soc_v3.pdf
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "paper" / "test"
OUT = ROOT / "paper" / "paper" / "soc"

_fp = str(ROOT / "data" / "Gulliver.otf")
font_manager.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "Gulliver"
plt.rcParams["font.sans-serif"] = font_manager.FontProperties(fname=_fp).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})

ARCHS = ["GRU", "MHA", "TCN"]
# full encoder names; multi-line form for the rotated row labels so they fit the panel height
ARCH_FULL = {"GRU": "Gated Recurrent Unit", "MHA": "Multi-Head Attention",
             "TCN": "Temporal Convolutional Network"}
ARCH_FULL_ML = {"GRU": "Gated\nRecurrent\nUnit", "MHA": "Multi-Head\nAttention",
                "TCN": "Temporal\nConvolutional\nNetwork"}
METHODS = [("u_penalty", "Penalty"), ("u_cmdp", "CMDP"), ("u_cmdp_shaped", "CMDP+PBRS"),
           ("u_cmdp_droq", "CMDP+DroQ"), ("u_cmdp_redq", "CMDP+REDQ"), ("u_cmdp_ft", "CMDP+FT")]
TARIFFS = [("tar_flat", "flat"), ("tar_tou", "ToU"), ("tar_s", "solar"),
           ("tar_w", "wind"), ("tar_sw", "sol+wind")]
SCNS = [f"test_{ds}_{m:02d}" for ds in ("cy", "wy") for m in range(1, 13)]

MCOL = {m: c for (m, _), c in zip(METHODS,
        ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#ff7f0e"])}
ACOL = {"GRU": "#1f77b4", "MHA": "#2ca02c", "TCN": "#d62728"}


def diurnal(method, arch, tariff):
    """Mean SoC-vs-time-of-day profile over all windows; index = hour in [0,24)."""
    frames = []
    for scn in SCNS:
        f = TEST / method / arch / tariff / "operations" / f"{scn}_best_raw.csv.gz"
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=["timestamp", "SoCBESS", "SoCEV"], parse_dates=["timestamp"])
        df["tod"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
        frames.append(df[["tod", "SoCBESS", "SoCEV"]])
    if not frames:
        return None
    g = pd.concat(frames).groupby("tod")
    return pd.DataFrame({"bess": g["SoCBESS"].mean(), "ev": g["SoCEV"].mean()})


def _ax_cosmetics(ax, i, j, nrows, row_label, col_label):
    ax.set_xlim(0, 24); ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 6, 12, 18, 24]); ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=6.5)
    if i == 0 and col_label:
        ax.set_title(col_label, fontsize=8)
    if j == 0 and row_label:
        ax.set_ylabel(row_label, fontsize=7.5, labelpad=3)
    if i == nrows - 1:
        ax.set_xlabel("hour", fontsize=7)


def fig_grid(prof, rows, cols, line_dim, device, path, title):
    """Generic grid. rows/cols are (key,label) lists; line_dim picks the overlaid series."""
    nr, nc = len(rows), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(2.1 * nc, 1.5 * nr + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    for i, (rk, rl) in enumerate(rows):
        for j, (ck, cl) in enumerate(cols):
            ax = axes[i][j]
            for key, lbl, col in line_dim:
                p = prof(rk, rl, ck, cl, key)
                if p is None:
                    continue
                ax.step(p.index, p[device], color=col, lw=1.0, where="mid")
            _ax_cosmetics(ax, i, j, nr, rl, cl)
    handles = [Line2D([], [], color=c, lw=1.4, label=l) for _, l, c in line_dim]
    # legend sits just above the panels
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.935))
    fig.suptitle(title, y=0.999, fontsize=9)
    save(fig, path)


def fig_v3(profiles, path):
    """rows=arch, cols=tariff; per panel BESS (solid) + EV (dotted), 6 methods."""
    # BESS solid and thicker (the focus); EV dotted, thin, translucent since it stays high/flat
    nr, nc = len(ARCHS), len(TARIFFS)
    fig, axes = plt.subplots(nr, nc, figsize=(2.35 * nc, 1.95 * nr + 0.7),
                             sharex=True, sharey=True, squeeze=False)
    for i, a in enumerate(ARCHS):
        for j, (tk, tl) in enumerate(TARIFFS):
            ax = axes[i][j]
            for m, _ in METHODS:
                p = profiles.get((m, a, tk))
                if p is None:
                    continue
                ax.step(p.index, p["ev"], color=MCOL[m], lw=0.7, ls=":", alpha=0.5, where="mid")
                ax.step(p.index, p["bess"], color=MCOL[m], lw=1.1, ls="-", where="mid")
            _ax_cosmetics(ax, i, j, nr, ARCH_FULL_ML[a], tl)
    mh = [Line2D([], [], color=MCOL[m], lw=1.4, label=l) for m, l in METHODS]
    dh = [Line2D([], [], color="0.3", lw=1.4, ls="-", label="BESS"),
          Line2D([], [], color="0.3", lw=1.0, ls=":", label="EV")]
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(handles=mh + dh, loc="lower center", ncol=8, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.935))
    fig.suptitle("Typical SoC daily profile  (solid: BESS,  dotted: EV)", y=0.999, fontsize=9)
    save(fig, path)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    print("computing diurnal profiles (90 cells)...", flush=True)
    profiles = {}
    for a in ARCHS:
        for m, _ in METHODS:
            for tk, _ in TARIFFS:
                profiles[(m, a, tk)] = diurnal(m, a, tk)
        print(f"  {a} done", flush=True)

    # V1: rows=arch, cols=tariff, methods overlaid (one fig per device)
    method_lines = [(m, l, MCOL[m]) for m, l in METHODS]
    for dev, devname in [("bess", "BESS"), ("ev", "EV")]:
        fig_grid(lambda rk, rl, ck, cl, key, dev=dev: profiles.get((key, rk, ck)),
                 rows=[(a, ARCH_FULL_ML[a]) for a in ARCHS], cols=TARIFFS, line_dim=method_lines,
                 device=dev, path=OUT / f"soc_{dev}_v1.png",
                 title=f"Typical {devname} SoC daily profile  (rows: encoder, cols: tariff)")

    # V2: rows=method, cols=tariff, archs overlaid (one fig per device)
    arch_lines = [(a, ARCH_FULL[a], ACOL[a]) for a in ARCHS]
    for dev, devname in [("bess", "BESS"), ("ev", "EV")]:
        fig_grid(lambda rk, rl, ck, cl, key, dev=dev: profiles.get((rk, key, ck)),
                 rows=METHODS, cols=TARIFFS, line_dim=arch_lines,
                 device=dev, path=OUT / f"soc_{dev}_v2.png",
                 title=f"Typical {devname} SoC daily profile  (rows: method, cols: tariff)")

    # V3: combined BESS+EV, rows=arch, cols=tariff, methods overlaid
    fig_v3(profiles, OUT / "soc_v3.png")
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
