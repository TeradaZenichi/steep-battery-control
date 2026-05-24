"""GRUAttn experiment pack. Run: python scripts/run_GRUAttn.py

Completed experiments are skipped via .train_done/.test_done markers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner import run_experiments, ALL_TARIFFS, TAR_S_ONLY


EXPERIMENTS = [
    # (label, method, tariffs, seed, shaping_omega, suffix)
    # ---- Baselines: all 5 tariffs ----
    ("baseline penalty",   "sac_penalty", ALL_TARIFFS, 42, 0.00, ""),
    ("baseline cmdp",      "sac_cmdp",    ALL_TARIFFS, 42, 0.00, ""),

    # ---- Shaping ablation (tar_s only) ----
    ("penalty shaping005", "sac_penalty", TAR_S_ONLY,  42, 0.05, "-shaping005"),
    ("penalty shaping010", "sac_penalty", TAR_S_ONLY,  42, 0.10, "-shaping010"),
    ("penalty shaping030", "sac_penalty", TAR_S_ONLY,  42, 0.30, "-shaping030"),
    ("cmdp    shaping003", "sac_cmdp",    TAR_S_ONLY,  42, 0.03, "-shaping003"),
    ("cmdp    shaping010", "sac_cmdp",    TAR_S_ONLY,  42, 0.10, "-shaping010"),
    ("cmdp    shaping020", "sac_cmdp",    TAR_S_ONLY,  42, 0.20, "-shaping020"),

    # ---- Seed variance (tar_s only) ----
    ("penalty seed1",      "sac_penalty", TAR_S_ONLY,   1, 0.00, "-seed1"),
    ("penalty seed7",      "sac_penalty", TAR_S_ONLY,   7, 0.00, "-seed7"),
    ("cmdp    seed1",      "sac_cmdp",    TAR_S_ONLY,   1, 0.00, "-seed1"),
    ("cmdp    seed7",      "sac_cmdp",    TAR_S_ONLY,   7, 0.00, "-seed7"),
]


if __name__ == "__main__":
    run_experiments("GRUAttn", EXPERIMENTS)
