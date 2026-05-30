"""Auto-generated thin entrypoint — unified SAC trainer."""
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
