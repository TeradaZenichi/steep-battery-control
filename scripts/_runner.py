"""Shared launcher for scripts/run_<ARCH>.py.

Each per-arch script defines its EXPERIMENTS list and calls run_experiments.
Completed runs are skipped via .train_done/.test_done marker files; delete a
marker to force a redo.

Experiment tuple: (label, method, tariffs, seed, shaping_omega, suffix)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALL_TARIFFS = "tar_s,tar_w,tar_sw,tar_flat,tar_tou"
TAR_S_ONLY = "tar_s"


def patch_config(config_path: Path, **kv) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for k, v in kv.items():
        cfg["train"][k] = v
    config_path.write_text(json.dumps(cfg, indent=4, ensure_ascii=False) + "\n",
                           encoding="utf-8")


def run_with_env(cmd: list[str], env_extra: dict[str, str]) -> int:
    env = os.environ.copy()
    env.update(env_extra)
    print(f"\n$ {' '.join(cmd)}")
    print(f"  env: {env_extra}")
    return subprocess.call(cmd, env=env, cwd=str(PROJECT_ROOT))


def banner(text: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{text}\n{line}")


def _all_train_done(method: str, arch: str, tariffs: list[str], suffix: str) -> bool:
    base = PROJECT_ROOT / "paper" / "train" / method / arch
    return all((base / f"{t}{suffix}" / ".train_done").exists() for t in tariffs)


def _all_test_done(method: str, arch: str, tariffs: list[str], suffix: str) -> bool:
    base = PROJECT_ROOT / "paper" / "test" / method / arch
    return all((base / f"{t}{suffix}" / ".test_done").exists() for t in tariffs)


def run_experiments(arch: str, experiments: list[tuple]) -> None:
    valid = {"GRU", "GRUAttn", "MHA", "TCN"}
    if arch not in valid:
        raise ValueError(f"unknown arch={arch}; expected one of {valid}")

    start = time.time()
    n = len(experiments)
    failures: list[str] = []
    skipped: list[str] = []

    banner(f"[{arch}] starting machine pack: {n} experiments")

    for idx, (label, method, tariffs, seed, omega, suffix) in enumerate(experiments, 1):
        tariff_list = tariffs.split(",")
        train_done = _all_train_done(method, arch, tariff_list, suffix)
        test_done = _all_test_done(method, arch, tariff_list, suffix)

        if train_done and test_done:
            print(f"\n[{arch}] {idx}/{n}  {label}  -- SKIP (already done)")
            skipped.append(label)
            continue

        banner(f"[{arch}] {idx}/{n}  {label}  "
               f"(method={method} tariffs={tariffs} seed={seed} omega={omega})")
        print(f"  train_done={train_done} test_done={test_done}")

        cfg_path = PROJECT_ROOT / "reinforcement" / method / arch / "config.json"
        patch_config(cfg_path, seed=seed, shaping_omega=omega)

        env_extra = {"RUN_TARIFFS": tariffs, "RUN_SUFFIX": suffix}

        if not train_done:
            rc = run_with_env(
                [sys.executable, f"reinforcement/{method}/{arch}/train.py"],
                env_extra,
            )
            if rc != 0:
                failures.append(f"train: {label} (rc={rc})")
                print(f"!! TRAIN FAILED for {label} -- skipping test, continuing")
                continue
        else:
            print("  train artifacts present, skipping training")

        if not test_done:
            rc = run_with_env(
                [sys.executable, f"reinforcement/{method}/{arch}/test.py"],
                env_extra,
            )
            if rc != 0:
                failures.append(f"test: {label} (rc={rc})")
                print(f"!! TEST FAILED for {label}, continuing")

    # Restore default config values so the repo is left clean
    for method in ("sac_penalty", "sac_cmdp"):
        cfg_path = PROJECT_ROOT / "reinforcement" / method / arch / "config.json"
        if cfg_path.exists():
            patch_config(cfg_path, seed=42, shaping_omega=0.0)

    elapsed_h = (time.time() - start) / 3600.0
    banner(f"[{arch}] DONE in {elapsed_h:.1f} h "
           f"(ran={n - len(skipped)} skipped={len(skipped)} failed={len(failures)})")
    if skipped:
        print("Skipped (already done):")
        for s in skipped:
            print(f"  - {s}")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
