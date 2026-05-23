"""IL baseline for the active MHA actor."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from supervised.utils.il import train_il


if __name__ == "__main__":
    train_il(Path(__file__).resolve().parent)
