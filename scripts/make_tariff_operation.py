"""Emit LaTeX for the per-tariff operating regime (demonstration-free class).

Aggregates the demonstration-free CMDP variants (CMDP, +PBRS, +DroQ, +REDQ) over the
3 encoders, deployed (safe) mode. Descriptive, system-level (no method ranking).

  python scripts/make_tariff_operation.py
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "paper" / "test"
ARCHS = ["GRU", "MHA", "TCN"]
# system-level regime: average over ALL methods (matches the per-tariff prose in 5.5);
# method-level cycling differences live in the cycling paragraph / Fig. op3.
DEMO = ["u_penalty", "u_cmdp", "u_cmdp_shaped", "u_cmdp_droq", "u_cmdp_redq", "u_cmdp_ft"]
TARIFFS = [("tar_flat", "flat"), ("tar_tou", "ToU"), ("tar_s", "solar"),
           ("tar_w", "wind"), ("tar_sw", "sol--wind")]


def rec(m, a, t):
    p = TEST / m / a / t / "summary_overall.json"
    if not p.exists():
        return None
    for r in json.load(open(p, encoding="utf-8")):
        if r.get("checkpoint") == "best" and r.get("mode") == "safe":
            return r
    return None


def agg(tariff, key):
    vals = [rec(m, a, tariff)[key] for m in DEMO for a in ARCHS if rec(m, a, tariff)]
    return st.mean(vals) if vals else float("nan")


def main():
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Operating regime under each tariff, averaged over all six methods and the "
           r"three encoders (deployed/safety-projected). The entries describe how each tariff "
           r"shapes operation; they are not a method comparison.}",
           r"\label{tab:tariff_operation}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r"Tariff & BESS throughput & V2G & Grid import & Grid export & PV curtailed & Energy cost \\",
           r" & [kWh] & [\%] & [kWh] & [kWh] & [kWh] & \\", r"\midrule"]
    for tk, tl in TARIFFS:
        thru = agg(tk, "mean_bess_charge_kwh") + agg(tk, "mean_bess_discharge_kwh")
        evc, evd = agg(tk, "mean_ev_charge_kwh"), agg(tk, "mean_ev_discharge_kwh")
        v2g = 100 * evd / evc if evc else float("nan")
        out.append(f"{tl} & {thru:.1f} & {v2g:.2f} & {agg(tk,'mean_grid_import_kwh'):.1f} & "
                   f"{agg(tk,'mean_grid_export_kwh'):.1f} & {agg(tk,'mean_pv_curtailed_kwh'):.1f} & "
                   f"{agg(tk,'mean_energy_cost'):.1f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print("\n".join(out))


if __name__ == "__main__":
    main()
