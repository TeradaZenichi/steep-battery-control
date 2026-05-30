"""Apples-to-apples: does BC+FT still beat a properly-trained from-scratch?

Both arms use the NEW unified harness (online, UTD 1/4), tar_sw/GRU, target_entropy=-2,
same 6-month val, 50 episodes:
  scratch arm = hp_ent2  (already trained: cmdp from-scratch, ent=-2)  -> reused
  ft arm      = cmp_ft   (cmdp + BC warm-start + FT profile)           -> trained here

This decides whether the paper's "BC warm-start is needed" thesis survives once the
training-schedule artifact (see memory: finding-drift-training-artifact) is removed.

Run:
    python sac/run_ft_comparison.py                # generate cmp_ft, train, compare
    python sac/run_ft_comparison.py --analyze-only
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
sys.path.insert(0, str(PROJECT_ROOT))

from sac.make_configs import BASE_TRAIN, _val_list, _deep_merge, TRAIN_PY, TEST_PY, BASE_TEST_IO, FT_PROFILE, BC_INIT
from sac.ablation.run_hp_ablation import SHORT, VAL_SUBSET

ARCH = "GRU"
TARIFF = "tar_sw"
SCRATCH = "hp_ent2"       # reused from the HP ablation (cmdp, ent=-2, from-scratch)
FT = "cmp_ft"             # cmdp + BC warm-start + FT profile


def generate():
    val = [v for v in _val_list() if v["name"] in VAL_SUBSET]
    train_cfg = _deep_merge(_deep_merge(BASE_TRAIN, SHORT), {"bc_init": BC_INIT, **FT_PROFILE})
    cfg = {"train": train_cfg, "val": val, "test": [], "plot": [], "test_io": BASE_TEST_IO}
    d = SAC_ROOT / FT / ARCH
    d.mkdir(parents=True, exist_ok=True)
    (SAC_ROOT / FT / "__init__.py").write_text("", encoding="utf-8")
    (d / "__init__.py").write_text("", encoding="utf-8")
    json.dump(cfg, open(d / "config.json", "w", encoding="utf-8"), indent=4)
    (d / "train.py").write_text(TRAIN_PY, encoding="utf-8")
    (d / "test.py").write_text(TEST_PY, encoding="utf-8")
    print(f"[gen] {FT}: safety={train_cfg['safety']} bc_init=yes ent={train_cfg['target_entropy']} "
          f"actor_start={train_cfg['actor_update_start_episode']} actor_lr={train_cfg['actor_lr']} "
          f"| {len(val)} val | {train_cfg['train_episodes']} ep")


def train():
    script = SAC_ROOT / FT / ARCH / "train.py"
    env = dict(os.environ, RUN_TARIFFS=TARIFF, PYTHONIOENCODING="utf-8")
    print(f"\n>>> TRAIN {FT} {TARIFF}", flush=True)
    t0 = time.time()
    rc = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), env=env).returncode
    print(f"<<< {FT}: rc={rc} ({time.time()-t0:.0f}s)", flush=True)


def analyze():
    import pandas as pd
    print("\n=== BC+FT vs from-scratch (tar_sw, ent=-2, same harness/val) ===")
    print(f"{'arm':<22}{'peak':>9}{'@ep':>5}{'end':>9}{'drift':>8}")
    rows = {}
    for label, name in (("from-scratch (hp_ent2)", SCRATCH), ("BC+FT (cmp_ft)", FT)):
        f = PROJECT_ROOT / "paper" / "train" / name / ARCH / TARIFF / "audit_training.csv"
        if not f.exists():
            print(f"{label:<22}  (no audit yet)"); continue
        e = pd.read_csv(f)["eval_mean_reward"].dropna()
        if not len(e):
            print(f"{label:<22}  (no eval rows)"); continue
        rows[name] = (e.max(), int(e.idxmax()), e.iloc[-1], e.iloc[-1] - e.max(), len(e))
        print(f"{label:<22}{e.max():>9.1f}{int(e.idxmax()):>5}{e.iloc[-1]:>9.1f}{e.iloc[-1]-e.max():>8.1f}")
    if SCRATCH in rows and FT in rows:
        sc_end, ft_end = rows[SCRATCH][2], rows[FT][2]
        gap = ft_end - sc_end
        print(f"\nFT end − scratch end = {gap:+.1f}  ", end="")
        if gap > 1.0:
            print("-> FT still wins: thesis survives (warm-start helps even with proper training).")
        elif gap < -1.0:
            print("-> from-scratch wins: thesis must PIVOT to 'training schedule > BC'.")
        else:
            print("-> ~tie: BC warm-start no longer necessary; thesis pivots. (Confirm w/ seeds + annual test.)")
        ft_peak_ep, ft_n = rows[FT][1], rows[FT][4]
        if ft_peak_ep >= ft_n - 3:
            print(f"[warn] FT peak at ep {ft_peak_ep} of {ft_n} — still climbing; consider more episodes before concluding.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if args.analyze_only:
        analyze(); return
    generate()
    train()
    analyze()


if __name__ == "__main__":
    main()
