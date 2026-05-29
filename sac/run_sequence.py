"""Run unified SAC experiments sequentially (train then test) for one tariff.

Each experiment is launched as a subprocess (sac/<exp>/<arch>/{train,test}.py) so
the Windows multiprocessing eval works and failures stay isolated. Defaults to
the tar_sw validation trio.

Examples:
    python sac/run_sequence.py                          # u_cmdp,u_penalty,u_cmdp_ft on tar_sw
    python sac/run_sequence.py --tariff tar_s
    python sac/run_sequence.py --exps u_cmdp,u_cmdp_shaped --skip-test
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAC_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPS = ["u_cmdp", "u_penalty", "u_cmdp_ft"]


def _bc_ok(exp: str, arch: str, tariff: str) -> bool | None:
    """None if exp needs no bc_init; True/False if it does and it exists/not."""
    cfg = json.load(open(SAC_ROOT / exp / arch / "config.json", encoding="utf-8"))
    bc = cfg["train"].get("bc_init")
    if not bc:
        return None
    path = bc[tariff] if isinstance(bc, dict) else bc
    p = PROJECT_ROOT / str(path).format(arch=arch, tariff=tariff)
    return p.exists()


def _run(script: Path, tariff: str) -> tuple[int, float]:
    env = dict(os.environ, RUN_TARIFFS=tariff, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), env=env)
    return proc.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tariff", default="tar_sw")
    ap.add_argument("--arch", default="GRU")
    ap.add_argument("--exps", default=",".join(DEFAULT_EXPS))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    exps = [e.strip() for e in args.exps.split(",") if e.strip()]
    results = []
    print(f"=== sequential run | tariff={args.tariff} arch={args.arch} | exps={exps} ===\n", flush=True)

    for exp in exps:
        d = SAC_ROOT / exp / args.arch
        if not (d / "train.py").exists():
            print(f"[skip] {exp}/{args.arch}: not found (run sac/make_configs.py?)", flush=True)
            results.append((exp, "missing", "-"))
            continue

        bc = _bc_ok(exp, args.arch, args.tariff)
        if bc is False:
            print(f"[warn] {exp}: bc_init checkpoint absent for {args.tariff} -> would train WITHOUT warm-start. Skipping.", flush=True)
            results.append((exp, "no-bc", "-"))
            continue
        if bc is True:
            print(f"[info] {exp}: bc_init found -> FT warm-start enabled.", flush=True)

        tr = te = "skip"
        if not args.skip_train:
            print(f"\n>>> TRAIN {exp}/{args.arch} {args.tariff}", flush=True)
            rc, dt = _run(d / "train.py", args.tariff)
            tr = f"ok({dt:.0f}s)" if rc == 0 else f"FAIL(rc={rc})"
            print(f"<<< TRAIN {exp}: {tr}", flush=True)
            if rc != 0:
                results.append((exp, tr, "skipped-after-train-fail"))
                continue
        if not args.skip_test:
            print(f"\n>>> TEST  {exp}/{args.arch} {args.tariff}", flush=True)
            rc, dt = _run(d / "test.py", args.tariff)
            te = f"ok({dt:.0f}s)" if rc == 0 else f"FAIL(rc={rc})"
            print(f"<<< TEST  {exp}: {te}", flush=True)
        results.append((exp, tr, te))

    print("\n=== summary ===", flush=True)
    for exp, tr, te in results:
        print(f"  {exp:<16} train={tr:<14} test={te}", flush=True)


if __name__ == "__main__":
    main()
