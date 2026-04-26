from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_split_testing import run_split_tests


def main() -> None:
    run_split_tests(
        machine="A",
        stage="all",
        python_exe=sys.executable,
        dry_run=False,
        stop_on_error=False,
        resume=True,
        live_output=True,
        rl_variants=("combo", "det", "stoch"),
    )


if __name__ == "__main__":
    main()
