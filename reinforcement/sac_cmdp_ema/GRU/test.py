"""Annual raw/safe checkpoint test for GRU (CMDP + EMA-backup)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from reinforcement.sac_cmdp.utils.test_eval import run_test


if __name__ == "__main__":
    run_test(Path(__file__).resolve().parent)
