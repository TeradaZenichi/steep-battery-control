"""Seed-level statistics on the lead tariff (reviewer item 8).

For tar_sw with three seeds (42/7/13), per method: the seed mean +/- 95% CI of the
deployed reward (best/safe), where each seed value is averaged over the three encoders.
Pairwise differences vs the demonstration-free CMDP are reported paired by seed, with the
95% CI of the difference and Cohen's d, to test whether any cost ranking is supported.

  python scripts/make_seed_stats.py
"""
from __future__ import annotations
import json
import statistics as st
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "paper" / "test"
ARCHS = ["GRU", "MHA", "TCN"]
SEEDS = ["tar_sw", "tar_sw-s7", "tar_sw-s13"]
METHODS = [("u_cmdp", "CMDP"), ("u_penalty", "Penalty"), ("u_cmdp_shaped", "PBRS"),
           ("u_cmdp_droq", "DroQ"), ("u_cmdp_redq", "REDQ"), ("u_cmdp_ft", "FT")]
T975_2 = 4.303      # Student t, 0.975 quantile, 2 dof (n=3)


def reward(m, a, sd):
    f = TEST / m / a / sd / "summary_overall.json"
    if not f.exists():
        return None
    for r in json.load(open(f, encoding="utf-8")):
        if r.get("checkpoint") == "best" and r.get("mode") == "safe":
            return r["mean_reward"]


def seed_values(m):
    """One reward per seed, averaged over the three encoders."""
    vals = []
    for sd in SEEDS:
        per_enc = [reward(m, a, sd) for a in ARCHS if reward(m, a, sd) is not None]
        if len(per_enc) == len(ARCHS):
            vals.append(st.mean(per_enc))
    return vals


def main():
    data = {ml: seed_values(mk) for mk, ml in METHODS}

    print("Lead tariff (tar_sw), n=3 seeds, reward averaged over encoders\n")
    print(f"{'method':8} | {'mean':>8} | {'95% CI':>14}")
    for _, ml in METHODS:
        v = data[ml]
        m, sd = st.mean(v), st.stdev(v)
        ci = T975_2 * sd / sqrt(len(v))
        print(f"{ml:8} | {m:8.1f} | +/- {ci:8.1f}")

    print("\nPairwise vs CMDP (paired by seed): mean diff, 95% CI of diff, Cohen's d")
    base = data["CMDP"]
    for _, ml in METHODS:
        if ml == "CMDP":
            continue
        diffs = [a - b for a, b in zip(data[ml], base)]
        md, sdd = st.mean(diffs), st.stdev(diffs)
        ci = T975_2 * sdd / sqrt(len(diffs))
        d = md / sdd if sdd > 0 else float("nan")
        sig = "DIFFERS" if abs(md) > ci else "within noise"
        print(f"  {ml:8} vs CMDP: {md:+7.1f}  CI +/-{ci:6.1f}  d={d:+5.2f}  -> {sig}")


if __name__ == "__main__":
    main()
