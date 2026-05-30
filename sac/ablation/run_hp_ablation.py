"""Short hyperparameter ablation on tar_sw/GRU to test the drift hypotheses.

Three variants (all safety=cmdp, from-scratch), isolating the effect:
  hp_ctrl            : current unified base (target_entropy=-1, gamma=0.999, reward_scale=0.1)
  hp_ent2            : target_entropy=-2                      (less entropy pressure)
  hp_ent2_g99_rs03   : target_entropy=-2, gamma=0.99, reward_scale=0.3  (full drift attack)

Short run: 50 episodes, reduced 6-month validation set (consistent across variants),
so the eval trajectory's peak-vs-end drift is comparable. Trains only (no annual
test); the drift signal lives in audit_training.csv's eval_mean_reward.

Run (generates configs, trains the 3 variants sequentially on tar_sw, prints drift):
    python sac/run_hp_ablation.py
    python sac/run_hp_ablation.py --generate-only
    python sac/run_hp_ablation.py --analyze-only
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SAC_ROOT = Path(__file__).resolve().parent            # .../sac/ablation (where this script + ablation configs live)
PROJECT_ROOT = SAC_ROOT.parent.parent                 # .../<project>
sys.path.insert(0, str(PROJECT_ROOT))

from sac.make_configs import BASE_TRAIN, _val_list, _deep_merge, TRAIN_PY, TEST_PY, BASE_TEST_IO

ARCH = "GRU"
TARIFF = "tar_sw"
SHORT = {"train_episodes": 50, "evaluate_every": 1, "min_episodes_before_early_stop": 50, "early_stop_patience": 50}
VAL_SUBSET = {"val_cy_01", "val_cy_04", "val_cy_07", "val_cy_10", "val_wy_04", "val_wy_10"}

VARIANTS = {
    "hp_ctrl": {},
    "hp_ent2": {"target_entropy": -2.0},
    "hp_ent2_g99_rs03": {"target_entropy": -2.0, "gamma": 0.99, "reward_scale": 0.3},
    "hp_shaped": {"target_entropy": -2.0, "shaping_bess_omega": 0.2},  # = hp_ent2 + PBRS Φ_bess
}


def generate():
    val = [v for v in _val_list() if v["name"] in VAL_SUBSET]
    for name, override in VARIANTS.items():
        train_cfg = _deep_merge(_deep_merge(BASE_TRAIN, SHORT), override)
        cfg = {"train": train_cfg, "val": val, "test": [], "plot": [], "test_io": BASE_TEST_IO}
        d = SAC_ROOT / name / ARCH
        d.mkdir(parents=True, exist_ok=True)
        (SAC_ROOT / name / "__init__.py").write_text("", encoding="utf-8")
        (d / "__init__.py").write_text("", encoding="utf-8")
        json.dump(cfg, open(d / "config.json", "w", encoding="utf-8"), indent=4)
        (d / "train.py").write_text(TRAIN_PY, encoding="utf-8")
        (d / "test.py").write_text(TEST_PY, encoding="utf-8")
        print(f"[gen] {name}: target_entropy={train_cfg['target_entropy']} gamma={train_cfg['gamma']} "
              f"reward_scale={train_cfg['reward_scale']} | {len(val)} val | {train_cfg['train_episodes']} ep")


def train():
    for name in VARIANTS:
        script = SAC_ROOT / name / ARCH / "train.py"
        env = dict(os.environ, RUN_TARIFFS=TARIFF, PYTHONIOENCODING="utf-8")
        print(f"\n>>> TRAIN {name} {TARIFF}", flush=True)
        t0 = time.time()
        rc = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), env=env).returncode
        print(f"<<< {name}: rc={rc} ({time.time()-t0:.0f}s)", flush=True)


def analyze():
    import pandas as pd
    print("\n=== drift comparison (eval_mean_reward) ===")
    print(f"{'variant':<20}{'peak':>9}{'@ep':>5}{'end':>9}{'drift':>8}{'alpha_end':>10}")
    for name in VARIANTS:
        f = PROJECT_ROOT / "paper" / "train" / name / ARCH / TARIFF / "audit_training.csv"
        if not f.exists():
            print(f"{name:<20}  (no audit yet)")
            continue
        df = pd.read_csv(f)
        e = df["eval_mean_reward"].dropna() if "eval_mean_reward" in df.columns else None
        a = df["alpha"].dropna() if "alpha" in df.columns else None
        if e is None or not len(e):
            print(f"{name:<20}  (no eval rows)")
            continue
        peak, peak_ep, end = e.max(), int(e.idxmax()), e.iloc[-1]
        drift = end - peak
        ae = a.iloc[-1] if a is not None and len(a) else float("nan")
        print(f"{name:<20}{peak:>9.1f}{peak_ep:>5}{end:>9.1f}{drift:>8.1f}{ae:>10.3f}")
    print("\n(drift closer to 0 = more stable; less negative end = better final policy)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if args.analyze_only:
        analyze(); return
    generate()
    if args.generate_only:
        return
    train()
    analyze()


if __name__ == "__main__":
    main()
