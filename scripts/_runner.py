"""Shared launcher for scripts/run_<ARCH>.py.

Each per-arch script defines its EXPERIMENTS list and calls run_experiments.
Completed runs are skipped via .train_done/.test_done marker files; delete a
marker to force a redo.

Experiment tuple: (label, method, tariffs, seed, shaping_omega, suffix)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALL_TARIFFS = "tar_s,tar_w,tar_sw,tar_flat,tar_tou"
TAR_S_ONLY = "tar_s"
TAR_SW_ONLY = "tar_sw"


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


# ----------------------------------------------------------------------------
# Supervised + fine-tuning helpers
# ----------------------------------------------------------------------------

def _supervised_ckpt(arch: str, tariff: str) -> Path:
    return PROJECT_ROOT / "paper" / "train" / "supervised" / arch / tariff / "best.pth"


def _supervised_done(arch: str, tariff: str) -> bool:
    return _supervised_ckpt(arch, tariff).exists()


def _supervised_test_done(arch: str, tariff: str) -> bool:
    return (
        PROJECT_ROOT / "paper" / "test" / "supervised" / arch / tariff / ".test_done"
    ).exists()


def _bc_history_len(arch: str, tariff: str) -> int:
    """Return the history_len that BC actually used for this (arch, tariff).

    The supervised pipeline (optionally with HPO) writes best_params.json with
    the selected configuration. FT must propagate that L to the RL config so
    the loaded BC actor and the RL encoder share the same temporal context.
    """
    best_params = PROJECT_ROOT / "paper" / "train" / "supervised" / arch / tariff / "best_params.json"
    if best_params.exists():
        data = json.loads(best_params.read_text(encoding="utf-8"))
        if "history_len" in data:
            return int(data["history_len"])
    # Fallback: read from the supervised config (no HPO scenario).
    sup_cfg_path = PROJECT_ROOT / "supervised" / arch / "config.json"
    sup_cfg = json.loads(sup_cfg_path.read_text(encoding="utf-8"))
    return int(sup_cfg["training"]["history_len"])


def _supervised_train_dir(arch: str, tariff: str) -> Path:
    return PROJECT_ROOT / "paper" / "train" / "supervised" / arch / tariff


def _materialize_supervised_rl_checkpoint(arch: str, tariff: str) -> None:
    """Expose the IL checkpoint in the .pt shape expected by test_eval."""
    train_dir = _supervised_train_dir(arch, tariff)
    src = train_dir / "best.pth"
    dst = train_dir / "best.pt"
    if not src.exists():
        raise FileNotFoundError(f"missing supervised checkpoint: {src}")
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)
    (train_dir / ".train_done").touch()


def _write_supervised_test_config(arch: str, tariff: str, history_len: int) -> Path:
    """Create a temporary algo_root whose parent name is 'supervised'.

    The shared test_eval helper infers output locations from algo_root:
    paper/test/<method>/<arch>/<tariff>. Placing the config under
    paper/_test_configs/supervised/<arch> makes method='supervised' without
    adding repo source files.
    """
    source_cfg_path = PROJECT_ROOT / "reinforcement" / "sac_penalty" / arch / "config.json"
    if source_cfg_path.exists():
        cfg = json.loads(source_cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = {"train": {}, "test_io": {}, "test": []}

    cfg.setdefault("train", {})
    cfg["train"]["history_len"] = int(history_len)
    cfg.setdefault("test_io", {})
    cfg["test_io"]["tariffs"] = [tariff]
    cfg["test_io"]["checkpoints"] = ["best"]

    algo_root = PROJECT_ROOT / "paper" / "_test_configs" / "supervised" / arch
    algo_root.mkdir(parents=True, exist_ok=True)
    (algo_root / "config.json").write_text(
        json.dumps(cfg, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return algo_root


def run_supervised_tests(arch: str, tariffs: list[str]) -> None:
    """Evaluate each pure BC checkpoint in the closed-loop environment.

    Results go to paper/test/supervised/<arch>/<tariff>. This runs before FT
    so we can separate "BC already works/fails" from the effect of SAC
    fine-tuning.
    """
    from reinforcement.sac_penalty.utils.test_eval import run_test

    banner(f"[{arch}] supervised closed-loop test: {len(tariffs)} tariffs")
    failures: list[str] = []
    for tariff in tariffs:
        if not _supervised_done(arch, tariff):
            failures.append(f"missing supervised {arch}/{tariff}")
            print(f"!! missing supervised checkpoint for {arch}/{tariff}")
            continue
        if _supervised_test_done(arch, tariff):
            print(f"  SKIP supervised test {arch}/{tariff} (.test_done present)")
            continue

        try:
            _materialize_supervised_rl_checkpoint(arch, tariff)
            algo_root = _write_supervised_test_config(
                arch, tariff, _bc_history_len(arch, tariff)
            )
            env_prev = os.environ.copy()
            os.environ["RUN_TARIFFS"] = tariff
            os.environ["RUN_SUFFIX"] = ""
            try:
                run_test(algo_root)
            finally:
                os.environ.clear()
                os.environ.update(env_prev)
        except Exception as exc:
            failures.append(f"supervised test {arch}/{tariff}: {exc}")
            print(f"!! SUPERVISED TEST FAILED for {arch}/{tariff}: {exc}")

    if failures:
        print("Supervised test failures:")
        for f in failures:
            print(f"  - {f}")
        raise RuntimeError("supervised closed-loop test failed; aborting FT")


def run_supervised(arch: str, tariffs: list[str]) -> None:
    """Run the supervised IL stage once for each tariff in *tariffs*.

    Outputs go to paper/train/supervised/<arch>/<tariff>/best.pth. Tariffs that
    already have a checkpoint are skipped.
    """
    banner(f"[{arch}] supervised: {len(tariffs)} tariffs")
    sup_cfg_path = PROJECT_ROOT / "supervised" / arch / "config.json"
    failures: list[str] = []
    for t in tariffs:
        if _supervised_done(arch, t):
            print(f"  SKIP supervised {arch}/{t} (best.pth already present)")
            continue
        # Pin the tariff in the supervised config so its internal loop only runs t
        sup_cfg = json.loads(sup_cfg_path.read_text(encoding="utf-8"))
        sup_cfg["tariffs"] = [t]
        sup_cfg_path.write_text(json.dumps(sup_cfg, indent=4, ensure_ascii=False) + "\n",
                                encoding="utf-8")
        rc = run_with_env(
            [sys.executable, f"supervised/{arch}/train.py"],
            {},
        )
        if rc != 0 or not _supervised_done(arch, t):
            failures.append(f"supervised {arch}/{t} (rc={rc})")
            print(f"!! SUPERVISED FAILED for {arch}/{t}")
    if failures:
        print("Supervised failures:")
        for f in failures:
            print(f"  - {f}")
        raise RuntimeError("supervised stage failed; aborting FT")


def _expand_ft_experiments(experiments: list[tuple]) -> list[tuple]:
    """Expand any multi-tariff entry (e.g. ALL_TARIFFS) into one-tariff-per-entry,
    since each FT run needs a single bc_init_checkpoint."""
    out = []
    for label, method, tariffs, seed, omega, suffix in experiments:
        tlist = tariffs.split(",")
        if len(tlist) == 1:
            out.append((label, method, tlist[0], seed, omega, suffix))
        else:
            for t in tlist:
                out.append((f"{label} ({t})", method, t, seed, omega, suffix))
    return out


def run_experiments_ft(arch: str, experiments: list[tuple]) -> None:
    """Chained supervised + FT runner.

    For each FT experiment in *experiments* (single tariff per entry after
    expansion), points the FT config's bc_init_checkpoint at the corresponding
    supervised artifact and then triggers train+test. Assumes
    run_supervised(arch, ...) has been called for all tariffs in advance.
    """
    valid = {"GRU", "GRUAttn", "MHA", "TCN"}
    if arch not in valid:
        raise ValueError(f"unknown arch={arch}; expected one of {valid}")

    expanded = _expand_ft_experiments(experiments)
    start = time.time()
    n = len(expanded)
    failures: list[str] = []
    skipped: list[str] = []

    banner(f"[{arch}] FT pack: {n} experiments (expanded)")

    for idx, (label, method, tariff, seed, omega, suffix) in enumerate(expanded, 1):
        if not method.endswith("_ft"):
            raise RuntimeError(f"{label}: run_experiments_ft expects _ft methods only")

        train_done = _all_train_done(method, arch, [tariff], suffix)
        test_done = _all_test_done(method, arch, [tariff], suffix)
        if train_done and test_done:
            print(f"\n[{arch}] {idx}/{n}  {label}  -- SKIP (already done)")
            skipped.append(label)
            continue

        bc_ckpt = _supervised_ckpt(arch, tariff)
        if not bc_ckpt.exists():
            failures.append(f"{label}: missing BC checkpoint {bc_ckpt}")
            print(f"!! BC CHECKPOINT MISSING for {arch}/{tariff}; skipping")
            continue

        banner(f"[{arch}] {idx}/{n}  {label}  "
               f"(method={method} tariff={tariff} seed={seed} omega={omega})")

        cfg_path = PROJECT_ROOT / "reinforcement" / method / arch / "config.json"
        bc_rel = bc_ckpt.relative_to(PROJECT_ROOT).as_posix()
        bc_L = _bc_history_len(arch, tariff)
        print(f"  using history_len={bc_L} from BC for {arch}/{tariff}")
        patch_config(
            cfg_path,
            seed=seed,
            shaping_omega=omega,
            bc_init_checkpoint=bc_rel,
            history_len=bc_L,
        )

        env_extra = {"RUN_TARIFFS": tariff, "RUN_SUFFIX": suffix}

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

    # Restore defaults in FT configs so the repo stays clean
    for method in ("sac_cmdp_ft", "sac_penalty_ft"):
        cfg_path = PROJECT_ROOT / "reinforcement" / method / arch / "config.json"
        if cfg_path.exists():
            patch_config(cfg_path, seed=42, shaping_omega=0.0, bc_init_checkpoint=None)

    elapsed_h = (time.time() - start) / 3600.0
    banner(f"[{arch}] FT DONE in {elapsed_h:.1f} h "
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
