"""Operational analysis (BESS cycling + V2G) into paper/paper/ as CSVs.

  table_bess_v2g.csv        - per-method (mean over 3 archs x 5 tariffs):
                              BESS charge/discharge/throughput, EV charge,
                              V2G (EV discharge) + %, BESS/EV cost.
  table_tariff_profile.csv  - per-tariff (mean over 6 methods x 3 archs):
                              BESS throughput, V2G, grid import/export,
                              PV curtailed, energy cost.
  table_bess_by_tariff.csv  - BESS throughput, method x tariff (consistency check).
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

ARCHS = ["GRU", "MHA", "TCN"]
TARS = ["tar_flat", "tar_tou", "tar_s", "tar_w", "tar_sw"]
TNAME = {"tar_flat": "flat", "tar_tou": "ToU", "tar_s": "solar", "tar_w": "wind", "tar_sw": "sol+wind"}
METHODS = [("u_penalty", "penalty"), ("u_cmdp", "CMDP"), ("u_cmdp_shaped", "PBRS"),
           ("u_cmdp_droq", "DroQ"), ("u_cmdp_redq", "REDQ"), ("u_cmdp_ft", "FT")]


def rec(m, a, t):
    for r in json.load(open(ROOT / "paper" / "test" / m / a / t / "summary_overall.json", encoding="utf-8")):
        if r.get("checkpoint") == "best" and r.get("mode") == "safe":
            return r


def thru(m, a, t):
    r = rec(m, a, t)
    return r["mean_bess_charge_kwh"] + r["mean_bess_discharge_kwh"]


def write_csv(name, header, rows):
    (OUT / name).write_text("\n".join([",".join(header)] + [",".join(map(str, r)) for r in rows]), encoding="utf-8")


def t_bess_v2g():
    rows = []
    for m, lbl in METHODS:
        def a(key):
            return st.mean(rec(m, ar, t)[key] for ar in ARCHS for t in TARS)
        bc, bd = a("mean_bess_charge_kwh"), a("mean_bess_discharge_kwh")
        ec, ed = a("mean_ev_charge_kwh"), a("mean_ev_discharge_kwh")
        rows.append([lbl, f"{bc:.1f}", f"{bd:.1f}", f"{bc+bd:.1f}", f"{ec:.1f}",
                     f"{ed:.2f}", f"{100*ed/ec:.2f}", f"{a('mean_bess_cost'):.3f}", f"{a('mean_ev_cost'):.2f}"])
    write_csv("table_bess_v2g.csv",
              ["method", "bess_charge_kwh", "bess_discharge_kwh", "bess_throughput_kwh",
               "ev_charge_kwh", "v2g_discharge_kwh", "v2g_pct", "bess_cost", "ev_cost"], rows)


def t_tariff_profile():
    rows = []
    for t in TARS:
        def a(key):
            return st.mean(rec(m, ar, t)[key] for m, _ in METHODS for ar in ARCHS)
        bt = a("mean_bess_charge_kwh") + a("mean_bess_discharge_kwh")
        ec, ed = a("mean_ev_charge_kwh"), a("mean_ev_discharge_kwh")
        rows.append([TNAME[t], f"{bt:.1f}", f"{ed:.2f}", f"{100*ed/ec:.2f}",
                     f"{a('mean_grid_import_kwh'):.1f}", f"{a('mean_grid_export_kwh'):.1f}",
                     f"{a('mean_pv_curtailed_kwh'):.1f}", f"{a('mean_energy_cost'):.1f}"])
    write_csv("table_tariff_profile.csv",
              ["tariff", "bess_throughput_kwh", "v2g_discharge_kwh", "v2g_pct",
               "grid_import_kwh", "grid_export_kwh", "pv_curtailed_kwh", "energy_cost"], rows)


def t_bess_by_tariff():
    rows = []
    for t in TARS:
        rows.append([TNAME[t]] + [f"{st.mean(thru(m, ar, t) for ar in ARCHS):.0f}" for m, _ in METHODS])
    write_csv("table_bess_by_tariff.csv", ["tariff"] + [lbl for _, lbl in METHODS], rows)


if __name__ == "__main__":
    t_bess_v2g()
    t_tariff_profile()
    t_bess_by_tariff()
    print("wrote operational CSVs to", OUT)
