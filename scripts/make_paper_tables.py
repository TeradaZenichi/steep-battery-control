"""Generate paper-ready cross-architecture tables into paper/paper/.

Outputs (CSV + LaTeX booktabs):
  1. table_results       - per-arch, per-method: operating reward + native grid
                           violation (5-tariff mean, seed 42). Native violation
                           (raw, pre-projection) is the primary safety axis.
  2. table_seed          - tar_sw n=3: reward mean +/- std per method (reliability).
  3. table_encoders      - encoder config + parameter counts (capacity transparency).

Reward differences within an architecture are often within seed noise (see table_seed
std), so reward "winners" are not bolded; native violation is the reported safety axis.
"""
from __future__ import annotations
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

ARCHS = ["GRU", "MHA", "TCN"]
BASE = ["tar_flat", "tar_s", "tar_w", "tar_sw", "tar_tou"]
SW3 = ["tar_sw", "tar_sw-s7", "tar_sw-s13"]
METHODS = [  # (folder, label) in presentation order
    ("u_penalty", "Penalty"), ("u_cmdp", "CMDP"), ("u_cmdp_shaped", "CMDP+PBRS"),
    ("u_cmdp_droq", "CMDP+DroQ"), ("u_cmdp_redq", "CMDP+REDQ"), ("u_cmdp_ft", "CMDP+FT (demo)"),
]


def rec(method, arch, tariff, mode):
    p = ROOT / "paper" / "test" / method / arch / tariff / "summary_overall.json"
    if not p.exists():
        return None
    for r in json.load(open(p, encoding="utf-8")):
        if r.get("checkpoint") == "best" and r.get("mode") == mode:
            return r
    return None


def mean_over(method, arch, key, mode, tariffs):
    vals = [rec(method, arch, t, mode)[key] for t in tariffs if rec(method, arch, t, mode)]
    return st.mean(vals) if len(vals) == len(tariffs) else float("nan")


def _rb_reward():
    """Rule-based floor: 5-tariff mean reward (encoder-independent). None if absent."""
    vals = []
    for t in BASE:
        p = ROOT / "paper" / "ablation" / "test" / "baselines" / "RB" / t / "summary_overall.json"
        if not p.exists():
            return None
        vals.append(json.load(open(p, encoding="utf-8"))[0]["mean_reward"])
    return st.mean(vals)


def _milp_field(field):
    """5-tariff mean of a MILP-baseline summary field, if the annual-test summary exists.
    field='mean_reward' -> idealized perfect-foresight objective (loose, optimistic bound:
    no SoC-derating, EV charged at the normal tariff, no fast-charge premium).
    field='openloop_replay_mean' -> the SAME plan executed open-loop in the env (realized,
    on the env's exact physics) -- the number comparable to RB and the learned policies."""
    for cand in ("MILP", "milp", "oracle"):
        vals, ok = [], True
        for t in BASE:
            p = ROOT / "paper" / "ablation" / "test" / "baselines" / cand / t / "summary_overall.json"
            if not p.exists():
                ok = False
                break
            vals.append(json.load(open(p, encoding="utf-8"))[0][field])
        if ok:
            return st.mean(vals)
    return None


def latex_table(caption, label, header_rows, body_rows, colspec):
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    lines += header_rows
    lines.append(r"\midrule")
    lines += [" & ".join(r) + r" \\" for r in body_rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- Table 1: results
def table_results():
    csv = ["method," + ",".join(f"{a}_reward,{a}_energy_cost,{a}_native_viol_kwh" for a in ARCHS)]
    body = []
    for m, lbl in METHODS:
        cells = [lbl]
        row_csv = [lbl]
        for a in ARCHS:
            rew = mean_over(m, a, "mean_reward", "safe", BASE)
            en = mean_over(m, a, "mean_energy_cost", "safe", BASE)
            nv = mean_over(m, a, "mean_grid_violation_kwh", "raw", BASE)
            cells += [f"{rew:.1f}", f"{nv:.3f}"]
            row_csv += [f"{rew:.2f}", f"{en:.2f}", f"{nv:.4f}"]
        body.append(cells)
        csv.append(",".join(row_csv))
    # --- reference rows (encoder-independent) ---
    # Operative comparison is RB floor vs the MILP open-loop replay (both realized in the
    # env, same ruler as the learned policies): the perfect-foresight plan executed
    # open-loop is WORSE than the naive rule-based controller -> closed-loop learning
    # matters. The idealized MILP objective is reported only as a loose optimistic bound
    # (perfect foresight + idealized physics: no SoC-derating, EV at normal tariff).
    ref = []
    rb = _rb_reward()
    if rb is not None:
        ref.append([r"\midrule RB (rule-based, floor)"] + [f"{rb:.1f}", "--"] * 3)
        csv.append(",".join(["RB_floor"] + [f"{rb:.2f}", "", "--"] * 3))
    milp_ol = _milp_field("openloop_replay_mean")
    if milp_ol is not None:
        ref.append([r"MILP plan, open-loop"] + [f"{milp_ol:.1f}", "--"] * 3)
        csv.append(",".join(["MILP_openloop"] + [f"{milp_ol:.2f}", "", "--"] * 3))
    milp = _milp_field("mean_reward")
    if milp is not None:
        ref.append([r"MILP optimum (idealized)"] + [f"{milp:.1f}", "--"] * 3)
        csv.append(",".join(["MILP_idealized"] + [f"{milp:.2f}", "", "--"] * 3))
    body += ref
    (OUT / "table_results.csv").write_text("\n".join(csv), encoding="utf-8")
    header = [
        r"& \multicolumn{2}{c}{GRU} & \multicolumn{2}{c}{MHA} & \multicolumn{2}{c}{TCN} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Method & Reward & Viol$_\text{nat}$ & Reward & Viol$_\text{nat}$ & Reward & Viol$_\text{nat}$ \\",
    ]
    tex = latex_table(
        r"Deployed annual performance (reward) and \emph{native} grid violation "
        r"(kWh, raw policy before the feasibility projection), 5-tariff mean, seed~42. "
        r"Native violation is the safety axis: demonstration-free methods learn "
        r"intrinsically feasible policies ($\approx 0$) while CMDP+FT relies on the "
        r"projection. Reward differences within a column are within seed noise "
        r"(Table~\ref{tab:seed}). Reference rows are encoder-independent: RB is the "
        r"rule-based floor; \emph{MILP plan, open-loop} executes the perfect-foresight "
        r"plan without feedback (realized in the simulator---worse than RB, so closed-loop "
        r"control matters); \emph{MILP optimum} is the idealized objective, a loose "
        r"optimistic bound (perfect foresight, no state-dependent power derating, \gls{EV} "
        r"charged at the normal tariff), reported over the $20/24$ windows where the MILP "
        r"is feasible.",
        "tab:results", header, body, "l" + "rr" * 3)
    (OUT / "table_results.tex").write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------- Table 2: seed n=3
def table_seed():
    csv = ["method," + ",".join(f"{a}_reward_mean,{a}_reward_std" for a in ARCHS)]
    body = []
    for m, lbl in METHODS:
        cells = [lbl]
        row_csv = [lbl]
        for a in ARCHS:
            rs = [rec(m, a, t, "safe")["mean_reward"] for t in SW3 if rec(m, a, t, "safe")]
            mu, sd = (st.mean(rs), st.pstdev(rs)) if len(rs) == 3 else (float("nan"), float("nan"))
            cells.append(f"${mu:.1f}\\pm{sd:.1f}$")
            row_csv += [f"{mu:.2f}", f"{sd:.2f}"]
        body.append(cells)
        csv.append(",".join(row_csv))
    (OUT / "table_seed.csv").write_text("\n".join(csv), encoding="utf-8")
    header = [r"Method & GRU & MHA & TCN \\"]
    tex = latex_table(
        r"Seed robustness on the lead tariff (tar\_sw), reward mean~$\pm$~std over "
        r"3 seeds. The spread across methods is comparable to the per-method seed "
        r"std, so within-architecture reward rankings are not statistically "
        r"separable (no single cheapest method is claimed).",
        "tab:seed", header, body, "lccc")
    (OUT / "table_seed.tex").write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------- Table 3: encoders
def table_encoders():
    import importlib
    rows = []
    csv = ["encoder,hidden_dim,layers,actor_params,critic_params"]
    for a in ARCHS:
        mod = importlib.import_module(f"models.{a}.model")
        mc = json.load(open(ROOT / "models" / a / "model.json", encoding="utf-8"))
        ac = dict(mc["actor"]); ac["parameters"] = str(ROOT / "data" / "parameters.json")
        cc = dict(mc["critic"])
        na = sum(p.numel() for p in mod.load_actor(ac).parameters())
        nc = sum(p.numel() for p in mod.load_critic(cc).parameters())
        h, L = ac.get("hidden_dim"), ac.get("num_layers")
        rows.append([a, str(h), str(L), f"{na:,}", f"{nc:,}"])
        csv.append(f"{a},{h},{L},{na},{nc}")
    (OUT / "table_encoders.csv").write_text("\n".join(csv), encoding="utf-8")
    header = [r"Encoder & $d_\text{hidden}$ & Layers & Actor params & Critic params \\"]
    tex = latex_table(
        r"Temporal encoder configurations and parameter counts. GRU and MHA are "
        r"capacity-matched ($\approx$\,256k actor); TCN uses 3 dilated-convolution "
        r"blocks and is $\approx$2$\times$ larger, so the encoder axis is reported "
        r"as a robustness check across diverse encoders, not a capacity-controlled "
        r"ranking.",
        "tab:encoders", header, rows, "lrrrr")
    (OUT / "table_encoders.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    table_results()
    table_seed()
    table_encoders()
    print("wrote tables to", OUT)
    for f in sorted(OUT.glob("table_*")):
        print("  ", f.relative_to(ROOT))
