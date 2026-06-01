"""Supervised BC training — TCN machine. Hardcoded config (no argparse).

Trains the 5 tariffs of TCN via supervised/TCN/train.py. Forces CPU
(CUDA_VISIBLE_DEVICES="") so it never competes with a running RL grid
for the GPU. Idempotent: if all 5 tariffs' best.pt already exist,
the script exits without retraining.

Output: paper/train/supervised/TCN/<tariff>/best.pt

Run from project root:
    python supervised/run_supervised_tcn.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SUPERVISED_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SUPERVISED_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---- machine config -----------------------------------------------------------
ARCH = "TCN"
TARIFFS = ["tar_flat", "tar_s", "tar_w", "tar_sw", "tar_tou"]
FORCE_CPU = True  # safe with a running RL grid on the GPU
# ------------------------------------------------------------------------------


def _all_tariffs_done():
    base = PROJECT_ROOT / "paper" / "train" / "supervised" / ARCH
    missing = [t for t in TARIFFS if not (base / t / "best.pt").exists()]
    return missing


def main():
    print(f"=== supervised BC | {ARCH} machine ===", flush=True)
    print(f"tariffs: {TARIFFS}")
    print(f"output : paper/train/supervised/{ARCH}/<tariff>/best.pt")
    print(f"device : {'CPU (forced)' if FORCE_CPU else 'auto'}")

    missing = _all_tariffs_done()
    if not missing:
        print(f"\n[{ARCH}] all 5 tariffs already trained, nothing to do.")
        return 0

    print(f"\n[{ARCH}] missing tariffs: {missing}", flush=True)
    print(f"[{ARCH}] note: supervised/{ARCH}/train.py trains all tariffs listed in"
          f" supervised/{ARCH}/config.json (no per-tariff skip).", flush=True)

    script = SUPERVISED_ROOT / ARCH / "train.py"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if FORCE_CPU:
        env["CUDA_VISIBLE_DEVICES"] = ""

    print(f"\n>>> launching {script.relative_to(PROJECT_ROOT)}", flush=True)
    t0 = time.time()
    rc = subprocess.run([sys.executable, str(script)],
                        cwd=str(PROJECT_ROOT), env=env).returncode
    print(f"<<< {ARCH}: rc={rc} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    base = PROJECT_ROOT / "paper" / "train" / "supervised" / ARCH
    final = [t for t in TARIFFS if (base / t / "best.pt").exists()]
    print(f"\n[{ARCH}] final: {len(final)}/{len(TARIFFS)} tariffs in"
          f" paper/train/supervised/{ARCH}/")
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main())
