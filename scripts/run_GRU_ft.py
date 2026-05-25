"""GRU fine-tuning pack. Run: python scripts/run_GRU_ft.py

Trains the supervised IL teacher per tariff (if missing) and then runs the
SAC fine-tuning variants (penalty + CMDP) starting from the corresponding
BC checkpoint. Completed runs are skipped via the usual .train_done /
.test_done markers and via the presence of best.pth for the supervised
stage.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner import (
    run_supervised, run_supervised_tests, run_experiments_ft,
    ALL_TARIFFS, TAR_S_ONLY, TAR_SW_ONLY,
)


EXPERIMENTS = [
    # (label, method, tariffs, seed, shaping_omega, suffix)
    # ---- Baselines: all 5 tariffs (expanded internally to one-per-tariff) ----
    ("ft baseline penalty",   "sac_penalty_ft", ALL_TARIFFS, 42, 0.00, ""),
    ("ft baseline cmdp",      "sac_cmdp_ft",    ALL_TARIFFS, 42, 0.00, ""),

    # ---- Bridge: shaping010 on tar_s ----
    ("ft penalty shaping010 s", "sac_penalty_ft", TAR_S_ONLY,  42, 0.10, "-shaping010"),
    ("ft cmdp shaping010 s",    "sac_cmdp_ft",    TAR_S_ONLY,  42, 0.10, "-shaping010"),

    # ---- Main ablations on tar_sw ----
    ("ft penalty shaping005",   "sac_penalty_ft", TAR_SW_ONLY, 42, 0.05, "-shaping005"),
    ("ft penalty shaping030",   "sac_penalty_ft", TAR_SW_ONLY, 42, 0.30, "-shaping030"),
    ("ft cmdp shaping003",      "sac_cmdp_ft",    TAR_SW_ONLY, 42, 0.03, "-shaping003"),
    ("ft cmdp shaping020",      "sac_cmdp_ft",    TAR_SW_ONLY, 42, 0.20, "-shaping020"),

    # ---- Seed variance on tar_sw ----
    ("ft penalty seed1",        "sac_penalty_ft", TAR_SW_ONLY,  1, 0.00, "-seed1"),
    ("ft penalty seed7",        "sac_penalty_ft", TAR_SW_ONLY,  7, 0.00, "-seed7"),
    ("ft cmdp seed1",           "sac_cmdp_ft",    TAR_SW_ONLY,  1, 0.00, "-seed1"),
    ("ft cmdp seed7",           "sac_cmdp_ft",    TAR_SW_ONLY,  7, 0.00, "-seed7"),
]


def _tariffs_needed(experiments):
    needed = set()
    for _, _, tariffs, *_ in experiments:
        needed.update(tariffs.split(","))
    return sorted(needed)


if __name__ == "__main__":
    arch = "GRU"
    tariffs = _tariffs_needed(EXPERIMENTS)
    run_supervised(arch, tariffs)
    run_supervised_tests(arch, tariffs)
    run_experiments_ft(arch, EXPERIMENTS)
