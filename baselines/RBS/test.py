"""Annual test for the RBS baseline."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines._test_eval import run_baseline_test
from baselines.RBS.model import load_actor


if __name__ == "__main__":
    run_baseline_test("RBS", "baselines.RBS.model", load_actor)
