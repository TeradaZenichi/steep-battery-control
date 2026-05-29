"""Phase 0 de-risk: multi-seed drift characterization on tar_sw/GRU.

Self-contained — run on any machine that has the repo + data. Generates the
configs, trains {scratch, shaped, FT} x seeds (default 42,7,13), skipping any
run already finished (.train_done), then prints mean +/- std of the eval drift
per method so we can see which method is RELIABLY stable (low |drift|, low
variance) vs the high-variance from-scratch.

Methods:
  hp_ent2   = from-scratch CMDP (ent=-2)                 [scratch]
  hp_shaped = hp_ent2 + PBRS Phi_bess                    [shaping]
  cmp_ft    = CMDP + BC warm-start + FT profile          [BC+FT]

Run on the other machine:
    python sac/run_phase0.py
    python sac/run_phase0.py --seeds 42,7,13,17,23      # n=5
    python sac/run_phase0.py --analyze-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SAC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SAC_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARCH = "GRU"
TARIFF = "tar_sw"
METHODS = {"hp_ent2": "scratch", "hp_shaped": "shaping(PBRS)", "cmp_ft": "BC+FT"}


def _ensure_configs():
    """Generate the experiment configs so this works on a fresh clone."""
    from sac.run_hp_ablation import generate as gen_ablation   # makes hp_ent2, hp_shaped, ...
    from sac.run_ft_comparison import generate as gen_ft       # makes cmp_ft
    gen_ablation()
    gen_ft()


def _suffix(seed: int) -> str:
    return "" if seed == 42 else f"-s{seed}"


def _run_dir(method: str, seed: int) -> Path:
    return PROJECT_ROOT / "paper" / "train" / method / ARCH / f"{TARIFF}{_suffix(seed)}"


def train(seeds):
    t0 = time.time()
    plan = [(m, s) for m in METHODS for s in seeds]
    todo = [(m, s) for m, s in plan if not (_run_dir(m, s) / ".train_done").exists()]
    print(f"[phase0] {len(plan)} (method,seed) cells; {len(plan)-len(todo)} already done; {len(todo)} to run "
          f"(~{len(todo)*2.75:.0f}h serial)\n", flush=True)
    for i, (method, seed) in enumerate(todo, 1):
        script = SAC_ROOT / method / ARCH / "train.py"
        env = dict(os.environ, RUN_TARIFFS=TARIFF, RUN_SEED=str(seed),
                   RUN_SUFFIX=_suffix(seed), PYTHONIOENCODING="utf-8")
        print(f">>> [{i}/{len(todo)}] {method} seed{seed}", flush=True)
        rc = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), env=env).returncode
        print(f"<<< {method} seed{seed}: rc={rc}  (elapsed {time.time()-t0:.0f}s)\n", flush=True)


def analyze(seeds):
    import pandas as pd, numpy as np
    print("\n=== Phase 0 drift characterization (tar_sw/GRU, ent=-2) ===")
    print(f"{'method':<16}{'seeds':>10}{'drift each':>26}{'mean':>8}{'std':>7}{'end mean':>10}")
    for method, label in METHODS.items():
        drifts, ends = [], []
        for s in seeds:
            f = _run_dir(method, s) / "audit_training.csv"
            if not f.exists():
                continue
            e = pd.read_csv(f)["eval_mean_reward"].dropna()
            if len(e):
                drifts.append(e.iloc[-1] - e.max()); ends.append(e.iloc[-1])
        if not drifts:
            print(f"{label:<16}{'(none)':>10}"); continue
        d = np.array(drifts)
        each = ",".join(f"{x:.0f}" for x in d)
        print(f"{label:<16}{len(d):>10}{each:>26}{d.mean():>8.1f}{d.std():>7.1f}{np.mean(ends):>10.1f}")
    print("\n(|drift| menor e std menor = mais confiavel; compara shaping/FT vs scratch)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,13")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.analyze_only:
        analyze(seeds); return
    _ensure_configs()
    train(seeds)
    analyze(seeds)


if __name__ == "__main__":
    main()
