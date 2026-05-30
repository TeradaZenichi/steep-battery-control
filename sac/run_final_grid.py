"""Final grid orchestrator — train + annual test, distributable across machines.

For each (method, arch, tariff, seed) cell:
  1. Train if `.train_done` is missing.
  2. Run annual test if `.test_done` is missing on the matching test dir.
Skip-if-done is automatic on both phases.

Distribution: with `--machine k/N`, this machine takes cells where
`cell_index % N == k-1`. Run the same command on each of N machines with k=1..N.

Default scope (lean grid):
  methods: u_cmdp, u_cmdp_ft, u_cmdp_shaped, u_penalty
  archs:   GRU, MHA, TCN
  tariffs: tar_flat, tar_s, tar_w, tar_sw, tar_tou
  seeds:   42 everywhere; +seeds 7,13 only on tar_sw (robustness on lead tariff).

Examples:
    python sac/run_final_grid.py --machine 1/3                 # machine 1 of 3
    python sac/run_final_grid.py --methods u_cmdp,u_cmdp_ft     # subset of methods
    python sac/run_final_grid.py --seeds 42 --extra-seeds none  # only n=1
    python sac/run_final_grid.py --plan-only                    # just print the cells
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SAC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SAC_ROOT.parent

DEFAULT_METHODS = ["u_cmdp", "u_cmdp_ft", "u_cmdp_shaped", "u_penalty", "u_cmdp_droq", "u_cmdp_redq"]
DEFAULT_ARCHS = ["GRU", "MHA", "TCN"]
DEFAULT_TARIFFS = ["tar_flat", "tar_s", "tar_w", "tar_sw", "tar_tou"]
LEAD_TARIFF = "tar_sw"
DEFAULT_BASE_SEEDS = [42]
DEFAULT_EXTRA_SEEDS = [7, 13]   # added only on LEAD_TARIFF


def _suffix(seed):
    return "" if seed == 42 else f"-s{seed}"


def _train_dir(method, arch, tariff, seed):
    return PROJECT_ROOT / "paper" / "train" / method / arch / f"{tariff}{_suffix(seed)}"


def _test_dir(method, arch, tariff, seed):
    return PROJECT_ROOT / "paper" / "test" / method / arch / f"{tariff}{_suffix(seed)}"


def _has_bc_init(method, arch, tariff):
    """Check the supervised BC checkpoint exists for FT methods, else None."""
    cfg = json.load(open(SAC_ROOT / method / arch / "config.json", encoding="utf-8"))
    bc = cfg["train"].get("bc_init")
    if not bc:
        return None
    p = PROJECT_ROOT / str(bc).format(arch=arch, tariff=tariff)
    return p.exists()


def build_cells(methods, archs, tariffs, base_seeds, extra_seeds, lead_tariff):
    cells = []
    for m in methods:
        for a in archs:
            for t in tariffs:
                seeds = list(base_seeds)
                if t == lead_tariff:
                    seeds += [s for s in extra_seeds if s not in seeds]
                for s in seeds:
                    cells.append((m, a, t, s))
    return cells


def _run(script, extra_env):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", **extra_env)
    return subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), env=env).returncode


def run_cell(method, arch, tariff, seed, skip_test):
    """Train then test one cell. Skip phase if already done."""
    base_env = {"RUN_TARIFFS": tariff, "RUN_SEED": str(seed), "RUN_SUFFIX": _suffix(seed)}
    statuses = []
    # Train
    if (_train_dir(method, arch, tariff, seed) / ".train_done").exists():
        statuses.append("train=skip")
    else:
        bc = _has_bc_init(method, arch, tariff)
        if bc is False:
            return ["train=skip(no-bc)", "test=skip"]
        t0 = time.time()
        rc = _run(SAC_ROOT / method / arch / "train.py", base_env)
        statuses.append(f"train={'ok' if rc == 0 else 'FAIL'}({time.time()-t0:.0f}s)")
        if rc != 0:
            return statuses + ["test=skip(train-fail)"]
    # Test
    if skip_test:
        statuses.append("test=skipped")
    elif (_test_dir(method, arch, tariff, seed) / ".test_done").exists():
        statuses.append("test=skip")
    else:
        t0 = time.time()
        rc = _run(SAC_ROOT / method / arch / "test.py", base_env)
        statuses.append(f"test={'ok' if rc == 0 else 'FAIL'}({time.time()-t0:.0f}s)")
    return statuses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--archs", default=",".join(DEFAULT_ARCHS))
    ap.add_argument("--tariffs", default=",".join(DEFAULT_TARIFFS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_BASE_SEEDS))
    ap.add_argument("--extra-seeds", default=",".join(str(s) for s in DEFAULT_EXTRA_SEEDS),
                    help="seeds added only on the lead tariff (tar_sw); 'none' to skip")
    ap.add_argument("--lead-tariff", default=LEAD_TARIFF)
    ap.add_argument("--machine", default="1/1", help="this machine's share, e.g. 1/3")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    tariffs = [t.strip() for t in args.tariffs.split(",") if t.strip()]
    base_seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    extra_seeds = [] if args.extra_seeds.lower() == "none" else [int(s) for s in args.extra_seeds.split(",") if s.strip()]
    k, N = (int(x) for x in args.machine.split("/"))
    if not (1 <= k <= N):
        raise SystemExit(f"--machine must satisfy 1<=k<=N (got {k}/{N})")

    cells = build_cells(methods, archs, tariffs, base_seeds, extra_seeds, args.lead_tariff)
    mine = [c for i, c in enumerate(cells) if i % N == k - 1]

    print(f"=== final grid (machine {k}/{N}) ===")
    print(f"methods: {methods}")
    print(f"archs:   {archs}")
    print(f"tariffs: {tariffs}  (lead={args.lead_tariff} with extra seeds {extra_seeds})")
    print(f"base seeds: {base_seeds}")
    print(f"total cells: {len(cells)}  |  this machine: {len(mine)}  |  est ~{len(mine)*3:.0f}h sequential")
    print()
    for i, (m, a, t, s) in enumerate(mine, 1):
        train_done = (_train_dir(m, a, t, s) / ".train_done").exists()
        test_done = (_test_dir(m, a, t, s) / ".test_done").exists()
        marker = "[done]" if train_done and test_done else "[todo]" if not train_done else "[test-only]"
        print(f"  {i:>3}. {marker} {m}/{a}/{t}-s{s}")
    print()

    if args.plan_only:
        return

    summary = []
    t_all = time.time()
    for i, (m, a, t, s) in enumerate(mine, 1):
        print(f"\n>>> [{i}/{len(mine)}] {m}/{a}/{t}-s{s}", flush=True)
        st = run_cell(m, a, t, s, args.skip_test)
        summary.append((m, a, t, s, st))
        print(f"<<< {m}/{a}/{t}-s{s}: {' | '.join(st)} (elapsed total {(time.time()-t_all)/3600:.1f}h)", flush=True)

    print("\n=== summary ===")
    for m, a, t, s, st in summary:
        print(f"  {m:<18} {a:<4} {t:<10} s{s:<3} : {' | '.join(st)}")
    print(f"total elapsed: {(time.time()-t_all)/3600:.1f}h")


if __name__ == "__main__":
    main()
