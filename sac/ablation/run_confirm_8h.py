"""~6.5h single-machine confirmation of the pivot (fits in 8h).

Adds ONE extra seed (seed 7) to each arm on tar_sw -> n=2 per arm (the existing
runs are seed 42), then runs the annual test of the seed-42 best checkpoints.
Keeps the exact same config (50 ep / 6-val) as the n=1 runs, so seeds are
comparable (no episode-budget confound). Trades a 3rd seed for staying under 8h.

  arms: hp_ent2 (from-scratch, cmdp, ent=-2)  |  cmp_ft (BC+FT)
  ~2.75h per training run x 2 = ~5.5h  +  annual test (~1h, best ckpt only)

Run:
    python sac/run_confirm_8h.py
    python sac/run_confirm_8h.py --analyze-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SAC_ROOT = Path(__file__).resolve().parent            # .../sac/ablation
PROJECT_ROOT = SAC_ROOT.parent.parent                 # .../<project>
ARCH = "GRU"
TARIFF = "tar_sw"
SEED2 = 7
ARMS = {"hp_ent2": "from-scratch", "cmp_ft": "BC+FT"}


def _run(script: Path, extra_env: dict) -> int:
    env = dict(os.environ, RUN_TARIFFS=TARIFF, PYTHONIOENCODING="utf-8", **extra_env)
    return subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), env=env).returncode


def _trim_test_to_best(exp: str):
    p = SAC_ROOT / exp / ARCH / "config.json"
    c = json.load(open(p, encoding="utf-8"))
    c.setdefault("test_io", {})["checkpoints"] = ["best"]
    json.dump(c, open(p, "w", encoding="utf-8"), indent=4)


def train_seed():
    t0 = time.time()
    for exp in ARMS:
        print(f"\n>>> TRAIN {exp} seed{SEED2} {TARIFF}", flush=True)
        rc = _run(SAC_ROOT / exp / ARCH / "train.py", {"RUN_SEED": str(SEED2), "RUN_SUFFIX": f"-s{SEED2}"})
        print(f"<<< {exp} seed{SEED2}: rc={rc}  (elapsed {time.time()-t0:.0f}s)", flush=True)


def test_seed42():
    t0 = time.time()
    for exp in ARMS:
        _trim_test_to_best(exp)
        print(f"\n>>> TEST {exp} (seed42 best, annual) {TARIFF}", flush=True)
        rc = _run(SAC_ROOT / exp / ARCH / "test.py", {})
        print(f"<<< TEST {exp}: rc={rc}  (elapsed {time.time()-t0:.0f}s)", flush=True)


def analyze():
    import pandas as pd
    print("\n=== seed confirmation — val drift (tar_sw, ent=-2) ===")
    print(f"{'arm / seed':<24}{'peak':>9}{'end':>9}{'drift':>8}")
    summary = {}
    for exp, label in ARMS.items():
        for suf, sd in (("", "42"), (f"-s{SEED2}", str(SEED2))):
            f = PROJECT_ROOT / "paper" / "train" / exp / ARCH / f"{TARIFF}{suf}" / "audit_training.csv"
            if not f.exists():
                print(f"{label+' s'+sd:<24}  (missing)"); continue
            e = pd.read_csv(f)["eval_mean_reward"].dropna()
            if not len(e):
                print(f"{label+' s'+sd:<24}  (no eval)"); continue
            summary.setdefault(exp, []).append((sd, e.max(), e.iloc[-1], e.iloc[-1] - e.max()))
            print(f"{label+' s'+sd:<24}{e.max():>9.1f}{e.iloc[-1]:>9.1f}{e.iloc[-1]-e.max():>8.1f}")

    print("\n=== annual test — best ckpt, raw (tar_sw) ===")
    for exp, label in ARMS.items():
        f = PROJECT_ROOT / "paper" / "test" / exp / ARCH / TARIFF / "summary_overall.json"
        if not f.exists():
            print(f"{label:<14}: (no test)"); continue
        d = json.load(open(f, encoding="utf-8"))
        r = [x for x in d if x.get("checkpoint") == "best" and x.get("mode") == "raw"]
        if r:
            print(f"{label:<14}: annual mean_reward = {r[0]['mean_reward']:.1f}")

    # verdict heuristic on drift consistency
    if "hp_ent2" in summary and "cmp_ft" in summary and len(summary["hp_ent2"]) == 2 and len(summary["cmp_ft"]) == 2:
        sc_drift = [d for _, _, _, d in summary["hp_ent2"]]
        ft_drift = [d for _, _, _, d in summary["cmp_ft"]]
        print(f"\nscratch drift (both seeds): {sc_drift[0]:.1f}, {sc_drift[1]:.1f}")
        print(f"FT      drift (both seeds): {ft_drift[0]:.1f}, {ft_drift[1]:.1f}")
        if max(sc_drift) > -5 and min(ft_drift) < -5:
            print("-> CONSISTENT across seeds: scratch stable, FT drifts -> pivot confirmed (n=2).")
        else:
            print("-> seeds DISAGREE -> need more seeds before concluding.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if args.analyze_only:
        analyze(); return
    train_seed()
    test_seed42()
    analyze()


if __name__ == "__main__":
    main()
