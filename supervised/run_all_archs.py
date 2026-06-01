"""Generate supervised BC checkpoints for all 3 archs in parallel.

Workflow per arch (GRU, MHA, TCN):
  1. If `paper/ablation/train/supervised/<arch>` exists and the canonical
     `paper/train/supervised/<arch>` does not, move it to the canonical
     location (no re-training needed).
  2. If all 5 tariffs' best.pt already exist at the canonical location,
     skip this arch.
  3. Otherwise, launch `supervised/<arch>/train.py` (which generates the
     5 tariffs' best.pt in one go).

All launched trainings run in parallel as subprocesses with CUDA disabled
(CUDA_VISIBLE_DEVICES=""), so they NEVER compete with a running RL grid for
the GPU. The teacher's MILP is CPU-only anyway, and the IL model is small
enough to train fast on CPU (~5-15 min per arch on a modern machine).

Logs: one file per arch at `_supervised_<arch>.log` in the project root.

Output: `paper/train/supervised/<arch>/<tariff>/best.pt` for each arch and
tariff in {tar_flat, tar_s, tar_w, tar_sw, tar_tou}. After this script
finishes, copy `paper/train/supervised/MHA/` to the MHA machine and
`paper/train/supervised/TCN/` to the TCN machine (or use a shared FS).

Run from project root:
    python supervised/run_all_archs.py
    python supervised/run_all_archs.py --archs MHA          # only MHA
    python supervised/run_all_archs.py --max-parallel 1     # serial
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHS_DEFAULT = ["GRU", "MHA", "TCN"]
TARIFFS = ["tar_flat", "tar_s", "tar_w", "tar_sw", "tar_tou"]


def _canonical(arch: str) -> Path:
    return PROJECT_ROOT / "paper" / "train" / "supervised" / arch


def _ablation(arch: str) -> Path:
    return PROJECT_ROOT / "paper" / "ablation" / "train" / "supervised" / arch


def _has_all_tariffs(base: Path) -> bool:
    return base.exists() and all((base / t / "best.pt").exists() for t in TARIFFS)


def restore_from_ablation(arch: str) -> str:
    """If ablation has the data and canonical doesn't, move it. Returns status string."""
    src = _ablation(arch)
    dst = _canonical(arch)
    if _has_all_tariffs(dst):
        return "skip (canonical already complete)"
    if _has_all_tariffs(src):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        return f"restored from ablation ({sum(1 for _ in dst.rglob('best.pt'))} ckpts)"
    return "needs training"


def train_arch(arch: str) -> tuple[str, int, float, str]:
    """Spawn `supervised/<arch>/train.py` with CUDA disabled. Returns (arch, rc, duration_s, log_path)."""
    script = PROJECT_ROOT / "supervised" / arch / "train.py"
    log_path = PROJECT_ROOT / f"_supervised_{arch}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""        # force CPU — never compete with RL grid GPU
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    print(f"[{arch}] training ... (log: {log_path.name})", flush=True)
    with open(log_path, "w", encoding="utf-8") as f:
        rc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT), env=env,
            stdout=f, stderr=subprocess.STDOUT,
        ).returncode
    return arch, rc, time.time() - t0, str(log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default=",".join(ARCHS_DEFAULT),
                    help="comma-separated subset of GRU,MHA,TCN")
    ap.add_argument("--max-parallel", type=int, default=3,
                    help="how many archs to train in parallel (default: 3)")
    ap.add_argument("--no-restore", action="store_true",
                    help="do not move data from paper/ablation/ — only retrain")
    args = ap.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    print("=" * 60)
    print(f"Supervised BC checkpoint generation (parallel)")
    print(f"Archs: {archs} | max-parallel: {args.max_parallel}")
    print(f"Tariffs per arch: {TARIFFS}")
    print(f"Output: paper/train/supervised/<arch>/<tariff>/best.pt")
    print(f"Mode: CPU only (CUDA disabled) — safe with running RL grid")
    print("=" * 60)

    to_train = []
    print("\n--- Phase 1: restore from ablation where possible ---")
    for arch in archs:
        if args.no_restore:
            status = "skip (--no-restore)" if _has_all_tariffs(_canonical(arch)) else "needs training"
        else:
            status = restore_from_ablation(arch)
        print(f"  {arch}: {status}")
        if status == "needs training":
            to_train.append(arch)

    if not to_train:
        print("\n  All archs are ready. Nothing to train.")
        print("\nFinal locations:")
        for arch in archs:
            base = _canonical(arch)
            n = sum(1 for _ in base.rglob("best.pt")) if base.exists() else 0
            print(f"  {arch}: {n}/5 tariffs at {base.relative_to(PROJECT_ROOT)}")
        return 0

    print(f"\n--- Phase 2: training {to_train} in parallel ---")
    t_all = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as pool:
        futures = {pool.submit(train_arch, arch): arch for arch in to_train}
        for fut in as_completed(futures):
            arch, rc, dur, log_path = fut.result()
            tag = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  {arch}: {tag}  ({dur / 60:.1f} min)  -> {Path(log_path).name}", flush=True)
            results.append((arch, rc, dur))

    print(f"\nTotal wall time: {(time.time() - t_all) / 60:.1f} min")
    print("\nFinal status:")
    any_fail = False
    for arch in archs:
        base = _canonical(arch)
        n = sum(1 for _ in base.rglob("best.pt")) if base.exists() else 0
        ok = (n == 5)
        if not ok:
            any_fail = True
        print(f"  {arch}: {'COMPLETE' if ok else 'INCOMPLETE'} ({n}/5)  -> {base.relative_to(PROJECT_ROOT) if base.exists() else 'missing'}")

    print()
    print("Next step:")
    print("  Copy paper/train/supervised/MHA/  ->  MHA machine's paper/train/supervised/MHA/")
    print("  Copy paper/train/supervised/TCN/  ->  TCN machine's paper/train/supervised/TCN/")
    print("  (if all machines share NFS this is automatic)")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
