"""Emit LaTeX for the honest seed-42 grid (cost + native violation) and the
per-method means. Prints to stdout for inlining into main.tex.

  python scripts/make_seed42_grid.py
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "paper" / "test"
ARCHS = ["GRU", "MHA", "TCN"]
METHODS = [("u_penalty", "Penalty"), ("u_cmdp", "CMDP"), ("u_cmdp_shaped", "CMDP+PBRS"),
           ("u_cmdp_droq", "CMDP+DroQ"), ("u_cmdp_redq", "CMDP+REDQ"), ("u_cmdp_ft", "CMDP+FT")]
TARIFFS = [("tar_flat", "flat"), ("tar_tou", "ToU"), ("tar_s", "solar"),
           ("tar_w", "wind"), ("tar_sw", "sol--wind")]


def rec(m, a, t, mode):
    p = TEST / m / a / t / "summary_overall.json"
    if not p.exists():
        return None
    for r in json.load(open(p, encoding="utf-8")):
        if r.get("checkpoint") == "best" and r.get("mode") == mode:
            return r
    return None


def reward(m, a, t):
    r = rec(m, a, t, "safe")
    return None if r is None else r["mean_reward"]


def viol(m, a, t):
    r = rec(m, a, t, "raw")
    return None if r is None else r["mean_grid_violation_kwh"]


def grid(fn, fmt, label, caption):
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\begin{tabular}{l" + "r" * len(TARIFFS) + "}", r"\toprule",
           "Method (encoder) & " + " & ".join(tl for _, tl in TARIFFS) + r" \\", r"\midrule"]
    for i, (mk, ml) in enumerate(METHODS):
        if i:
            out.append(r"\addlinespace")
        for a in ARCHS:
            cells = []
            for tk, _ in TARIFFS:
                v = fn(mk, a, tk)
                cells.append(fmt.format(v) if v is not None else "--")
            out.append(f"{ml} ({a}) & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def means():
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Per-method means over the seed-42 grid (3 encoders $\times$ 5 tariffs): "
           r"operating reward (higher is better) and native pre-projection grid violation, "
           r"with the per-cell violation range. Reward differences among the demonstration-free "
           r"variants are within seed noise (Table~\ref{tab:seed}); the full grid is in "
           r"Appendix~\ref{app:grid}.}",
           r"\label{tab:method_means}", r"\begin{tabular}{lrrl}", r"\toprule",
           r"Method & Reward & Native violation [kWh] & Violation range [kWh] \\",
           r"\midrule"]
    for mk, ml in METHODS:
        cs = [reward(mk, a, t) for a in ARCHS for t, _ in TARIFFS if reward(mk, a, t) is not None]
        vs = [viol(mk, a, t) for a in ARCHS for t, _ in TARIFFS if viol(mk, a, t) is not None]
        out.append(f"{ml} & {st.mean(cs):.1f} & {st.mean(vs):.3f} & "
                   f"{min(vs):.3f}--{max(vs):.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def main():
    print("%%%%%%%%%%%%%%%%%%%% MEANS (body) %%%%%%%%%%%%%%%%%%%%")
    print(means())
    print("\n%%%%%%%%%%%%%%%%%%%% VIOLATION GRID (appendix) %%%%%%%%%%%%%%%%%%%%")
    print(grid(viol, "{:.3f}", "tab:grid_violation",
               "Native grid violation [kWh] for every method, encoder, and tariff (seed 42, "
               "raw policy before the safety projection). Among the demonstration-free controllers "
               "the only persistently nonzero entries are the base CMDP on GRU; CMDP+FT is elevated "
               "across the grid."))
    print("\n%%%%%%%%%%%%%%%%%%%% REWARD GRID (appendix) %%%%%%%%%%%%%%%%%%%%")
    print(grid(reward, "{:.1f}", "tab:grid_reward",
               "Operating reward (higher is better) for every method, encoder, and tariff "
               "(seed 42, safety-projected). The penalty baseline is the lowest-reward outlier on "
               "GRU and TCN; the other methods overlap."))


if __name__ == "__main__":
    main()
