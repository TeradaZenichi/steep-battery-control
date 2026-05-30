"""Generate coherent per-experiment configs + thin scripts under sac/.

Single source of truth: every experiment is BASE + a small switch override, so
all share an identical training harness and hyperparameters. Only the listed
switches differ. Experiment dirs are prefixed `u_` (unified) so they never
collide with the legacy results under reinforcement/ (paper/{train,test}/u_*).

Run:
    python sac/make_configs.py            # all archs in ARCHS
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # .../sac
PROJECT_ROOT = ROOT.parent
ARCHS = ["GRU", "MHA", "TCN"]                    # full grid (3 encoders)


def _val_list():
    runs = []
    for tag, ds, socs in (("cy", "CY", [0.2, 0.5, 0.8]), ("wy", "WY", [0.5, 0.8, 0.2])):
        for m in range(1, 13):
            runs.append({
                "name": f"val_{tag}_{m:02d}",
                "dataset": f"data/Simulation_{ds}_Cur_HP__PV5000-HB5000.csv",
                "date": f"2000-{m:02d}-01 00:00:00",
                "days": 7,
                "soc": socs[(m - 1) % 3],
            })
    return runs


# --- Unified base (from-scratch profile). Identical across every experiment. ---
BASE_TRAIN = {
    "seed": 42, "days": 7, "train_episodes": 80, "warmup_episodes": 5,
    "min_episodes_before_early_stop": 30, "early_stop_patience": 15,
    "evaluate_every": 1, "eval_workers": 6, "max_pending_evals": 2,
    "target_entropy": -1.0, "init_alpha": 0.3, "log_std_min": -10.0, "log_std_max": 2.0,
    "buffer_size": 200000, "batch_size": 512, "history_len": 96,
    "n_step": 36, "update_every_steps": 4, "actor_update_every": 8, "actor_update_start_episode": 10,
    "gamma": 0.999, "tau": 0.005, "actor_lr": 1e-4, "critic_lr": 1e-4, "alpha_lr": 3e-4,
    "reward_scale": 0.1, "grad_clip": True, "ema_tau": 0.99,
    "precision": "bf16",  # "fp32" | "bf16" | "fp16" (Ampere recommends bf16 - no GradScaler needed)
    "droq_dropout": 0.0,  # 0 = vanilla twin Q; >0 = DroQ-style dropout in critic MLP (e.g., 0.01)
    "n_critics": 2,       # 2 = twin Q (default); 4 = REDQ-light (4 critics, take subset min for target)
    "m_target": None,     # None = use all critics; integer = REDQ subset min size (e.g., 2 of 4)
    "shaping_omega": 0.0, "shaping_typical_drop": 0.137, "shaping_bess_omega": 0.0,
    # switches
    "safety": "cmdp", "ema_backup": False, "bc_init": None,
    "rlpd": {"enabled": False, "prior_ratio": 0.5, "sources": ["RB", "RBS"], "demos_root": "paper/demos"},
    "cmdp": {
        "safe_rollout": False,
        "costs": {
            "ev": {"enabled": True, "target_soc": 0.5, "t_remain_floor": 0.05,
                   "budget": 0.01, "lambda_init": 0.1, "lambda_lr": 3e-4, "lambda_max": 10.0},
            "grid": {"enabled": False, "budget": 0.0, "lambda_init": 0.1, "lambda_lr": 3e-4, "lambda_max": 10.0},
            "pv": {"enabled": False, "budget": 0.01, "lambda_init": 0.05, "lambda_lr": 1e-4, "lambda_max": 0.5},
        },
    },
}

BASE_TEST_IO = {
    "checkpoints": ["best", "best_robust", "best_worst", "last"],
    "tariffs": ["tar_s", "tar_w", "tar_sw", "tar_flat", "tar_tou"],
    "save_plot_operation": True, "plot_operation_days": 7, "test_workers": 4,
}

# Fine-tuning profile: gentle refinement from the BC warm-start.
FT_PROFILE = {"actor_update_start_episode": 20, "init_alpha": 0.01,
              "target_entropy": -2.0, "actor_lr": 5e-5}
BC_INIT = "paper/train/supervised/{arch}/{tariff}/best.pt"

# --- Experiments: name -> switch override (deep-merged onto BASE_TRAIN) ---
EXPERIMENTS = {
    "u_cmdp": {},
    "u_penalty": {"safety": "penalty"},
    "u_cmdp_rlpd": {"n_step": 3, "rlpd": {"enabled": True, "prior_ratio": 0.25}},
    "u_cmdp_ema": {"ema_backup": True},
    "u_cmdp_shaped": {"shaping_bess_omega": 0.2},
    "u_cmdp_droq": {"droq_dropout": 0.01},   # DroQ-style anti-drift: dropout in critic MLP
    "u_cmdp_redq": {"n_critics": 4, "m_target": 2},   # REDQ-light: 4 critics, target = min of random 2
    "u_cmdp_ft": {"bc_init": BC_INIT, **FT_PROFILE},
    "u_penalty_ft": {"safety": "penalty", "bc_init": BC_INIT, **FT_PROFILE},
}

TRAIN_PY = '''"""Auto-generated thin entrypoint — unified SAC trainer."""
import sys
from pathlib import Path

# Walk up until we find the project root (depth-agnostic: works for sac/<exp>/<arch>/ and sac/ablation/<exp>/<arch>/).
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "sac" / "common" / "trainer.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sac.common.trainer import run_train

if __name__ == "__main__":
    run_train(Path(__file__).resolve().parent)
'''

TEST_PY = '''"""Auto-generated thin entrypoint — reuses the shared annual test."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "sac" / "common" / "trainer.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reinforcement.sac_cmdp.utils.test_eval import run_test

if __name__ == "__main__":
    run_test(Path(__file__).resolve().parent)
'''


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def main():
    val = _val_list()
    n = 0
    for exp, override in EXPERIMENTS.items():
        train_cfg = _deep_merge(BASE_TRAIN, override)
        cfg = {"train": train_cfg, "val": val, "test": [], "plot": [], "test_io": BASE_TEST_IO}
        for arch in ARCHS:
            d = ROOT / exp / arch
            d.mkdir(parents=True, exist_ok=True)
            (d / "__init__.py").write_text("", encoding="utf-8")
            with open(d / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4)
            (d / "train.py").write_text(TRAIN_PY, encoding="utf-8")
            (d / "test.py").write_text(TEST_PY, encoding="utf-8")
            n += 1
        (ROOT / exp / "__init__.py").write_text("", encoding="utf-8")
    print(f"Generated {len(EXPERIMENTS)} experiments x {len(ARCHS)} arch = {n} config dirs under {ROOT}")
    print("Experiments:", ", ".join(EXPERIMENTS))


if __name__ == "__main__":
    main()
