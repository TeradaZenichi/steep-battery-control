"""Exploratory metric correlation matrices (Spearman) for the annual test grid.

Curated, NON-definitional metric set (reward / total_cost / grid_penalty are excluded
because they are linear functions of the others):
  energy_cost, ev_cost, bess_cycling(=charge+discharge kWh), native grid violation
  (RAW, pre-projection), projection delta (SAFE, how much the projection corrects),
  pv_curtailed.
All extensive metrics are normalized per step to remove the 28--31-day window-length
confound; projection_delta is already a per-step mean. Native violation is taken from the
RAW trace; every other metric from the deployed SAFE trace, merged per window.

Two views (the question each answers differs):
  A) one matrix per ARCHITECTURE  (pooled over methods x tariffs x windows)
  B) one matrix per METHOD         (pooled over architectures x tariffs x windows)
Outputs heatmaps (PDF+PNG, Gulliver) + CSVs into paper/paper/correlation/, and prints a
key-pair comparison so we can see whether A and B give different conclusions.

  python scripts/make_correlation.py
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
OUT = ROOT / "paper" / "paper" / "correlation"
OUT.mkdir(parents=True, exist_ok=True)

_font_path = str(ROOT / "data" / "Gulliver.otf")
if Path(_font_path).exists():
    font_manager.fontManager.addfont(_font_path)
    plt.rcParams["font.family"] = "Gulliver"
plt.rcParams["axes.unicode_minus"] = False  # Gulliver lacks U+2212; use ASCII hyphen-minus

ARCHS = ["GRU", "MHA", "TCN"]
BASE = ["tar_flat", "tar_s", "tar_w", "tar_sw", "tar_tou"]
METHODS = [("u_penalty", "Penalty"), ("u_cmdp", "CMDP"), ("u_cmdp_shaped", "CMDP+PBRS"),
           ("u_cmdp_droq", "CMDP+DroQ"), ("u_cmdp_redq", "CMDP+REDQ"), ("u_cmdp_ft", "CMDP+FT")]

# display order + labels of the curated metrics
METRICS = ["energy_cost", "ev_cost", "bess_cycle", "native_viol", "proj_delta", "pv_curtail"]
LABELS = {"energy_cost": "Energy cost", "ev_cost": "EV cost", "bess_cycle": "BESS cycling",
          "native_viol": "Native viol.", "proj_delta": "Proj. $\\Delta$", "pv_curtail": "PV curtail"}


def _cell_rows(folder, arch, tariff):
    p = ROOT / "paper" / "test" / folder / arch / tariff / "summary_monthly.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d["checkpoint"] == "best"]
    raw = d[d["mode"] == "raw"].set_index("scenario")
    safe = d[d["mode"] == "safe"].set_index("scenario")
    if raw.empty or safe.empty:
        return None
    steps = safe["steps"].clip(lower=1)
    out = pd.DataFrame({
        "energy_cost": safe["energy_cost"] / steps,
        "ev_cost": safe["ev_cost"] / steps,
        "bess_cycle": (safe["bess_charge_kwh"] + safe["bess_discharge_kwh"]) / steps,
        "native_viol": raw["grid_violation_kwh"].reindex(safe.index) / steps,
        "proj_delta": safe["projection_delta_mean"],
        "pv_curtail": safe["pv_curtailed_kwh"] / steps,
    })
    out["arch"], out["method"], out["tariff"] = arch, folder, tariff
    return out.reset_index(drop=True)


def load_all():
    frames = []
    for folder, _ in METHODS:
        for arch in ARCHS:
            for t in BASE:
                r = _cell_rows(folder, arch, t)
                if r is not None:
                    frames.append(r)
    return pd.concat(frames, ignore_index=True)


def corr(df):
    return df[METRICS].corr(method="spearman")


def heatmap(cmat, title, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    M = cmat.loc[METRICS, METRICS].to_numpy(dtype=float)
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(METRICS))); ax.set_yticks(range(len(METRICS)))
    labs = [LABELS[m] for m in METRICS]
    ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labs, fontsize=8)
    for i in range(len(METRICS)):
        for j in range(len(METRICS)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.55 else "black")
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(f"{path}.pdf"); fig.savefig(f"{path}.png", dpi=160)
    plt.close(fig)


KEY_PAIRS = [("energy_cost", "native_viol"), ("energy_cost", "bess_cycle"),
             ("bess_cycle", "native_viol"), ("proj_delta", "native_viol"),
             ("ev_cost", "native_viol"), ("energy_cost", "ev_cost")]


def keytable(label, cmat):
    return [f"{cmat.loc[a, b]:+.2f}" if np.isfinite(cmat.loc[a, b]) else "  nan"
            for a, b in KEY_PAIRS]


def main():
    df = load_all()
    print(f"[corr] loaded {len(df)} window rows across {df['arch'].nunique()} archs "
          f"x {df['method'].nunique()} methods x {df['tariff'].nunique()} tariffs\n")

    heatmap(corr(df), "All (pooled)", OUT / "corr_all")
    cmat_all = corr(df)

    # View A: per architecture
    rowsA = []
    for arch in ARCHS:
        c = corr(df[df["arch"] == arch])
        c.to_csv(OUT / f"corr_arch_{arch}.csv")
        heatmap(c, f"Architecture: {arch}", OUT / f"corr_arch_{arch}")
        rowsA.append((arch, keytable(arch, c)))
    # View B: per method
    rowsB = []
    for folder, lbl in METHODS:
        c = corr(df[df["method"] == folder])
        c.to_csv(OUT / f"corr_method_{folder}.csv")
        heatmap(c, lbl, OUT / f"corr_method_{folder}")
        rowsB.append((lbl, keytable(lbl, c)))

    hdr = ["en-viol", "en-cyc", "cyc-viol", "pdl-viol", "ev-viol", "en-ev"]
    print("Key Spearman pairs (en=energy_cost, viol=native, cyc=BESS cycling, pdl=proj.delta, ev=EV cost)\n")
    print("== VIEW A: per architecture ==")
    print(f"{'':12s}" + "".join(f"{h:>10s}" for h in hdr))
    for name, vals in rowsA:
        print(f"{name:12s}" + "".join(f"{v:>10s}" for v in vals))
    print(f"{'ALL':12s}" + "".join(f"{v:>10s}" for v in keytable('ALL', cmat_all)))
    print("\n== VIEW B: per method ==")
    print(f"{'':12s}" + "".join(f"{h:>10s}" for h in hdr))
    for name, vals in rowsB:
        print(f"{name:12s}" + "".join(f"{v:>10s}" for v in vals))
    print(f"\n[corr] wrote {1 + len(ARCHS) + len(METHODS)} heatmaps (pdf+png) + CSVs to {OUT}")


if __name__ == "__main__":
    main()
