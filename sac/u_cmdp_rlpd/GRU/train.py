"""Auto-generated thin entrypoint — unified SAC trainer."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from sac.common.trainer import run_train

if __name__ == "__main__":
    run_train(Path(__file__).resolve().parent)
