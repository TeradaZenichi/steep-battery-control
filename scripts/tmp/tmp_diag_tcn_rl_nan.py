from __future__ import annotations

import argparse
import importlib.util
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]

NUMERIC_COLUMNS_OF_INTEREST = [
    "electricity_demand_rate_W",
    "produced_electricity_rate_W",
    "ev_conn",
    "ev_arrival",
    "ev_departure",
    "drybulb_C",
    "relhum_percent",
    "Global Horizontal Radiation",
    "dni_Wm2",
    "dhi_Wm2",
    "Wind Speed (m/s)",
    "wdir_deg",
    "tar_s",
    "tar_w",
    "tar_sw",
    "tar_tou",
    "tar_flat",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_train_module(model_name: str):
    train_path = ROOT / "models" / model_name / "2-RL" / "train.py"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find train.py at: {train_path}")

    spec = importlib.util.spec_from_file_location(f"{model_name}_train_module", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {train_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _finite_np(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr)
    if a.size == 0:
        return {
            "shape": tuple(a.shape),
            "all_finite": True,
            "nan_count": 0,
            "inf_count": 0,
            "max_abs": 0.0,
        }
    return {
        "shape": tuple(a.shape),
        "all_finite": bool(np.isfinite(a).all()),
        "nan_count": int(np.isnan(a).sum()),
        "inf_count": int(np.isinf(a).sum()),
        "max_abs": float(np.nanmax(np.abs(a))),
    }


def _scan_datasets_for_nonfinite(train_cfg: dict) -> dict[str, Any]:
    paths: set[Path] = {
        ROOT / "data" / "Simulation_CY_Cur_HP__PV5000-HB5000.csv",
        ROOT / "data" / "Simulation_WY_Cur_HP__PV5000-HB5000.csv",
    }

    for group in ("val", "test"):
        for run in train_cfg.get(group, []):
            p = Path(run["dataset"])
            if not p.is_absolute():
                p = ROOT / p
            paths.add(p)

    reports: list[dict[str, Any]] = []
    for p in sorted(paths):
        row: dict[str, Any] = {"path": str(p), "exists": p.exists()}
        if not p.exists():
            reports.append(row)
            continue

        df = pd.read_csv(p, sep=";")
        present = [c for c in NUMERIC_COLUMNS_OF_INTEREST if c in df.columns]
        missing = [c for c in NUMERIC_COLUMNS_OF_INTEREST if c not in df.columns]

        if present:
            sub = df[present]
            arr = sub.to_numpy(dtype=np.float64)
            nonfinite = _finite_np(arr)
            nan_cols = {
                col: int(sub[col].isna().sum())
                for col in present
                if int(sub[col].isna().sum()) > 0
            }
            row.update(
                {
                    "rows": int(len(df)),
                    "present_cols": present,
                    "missing_cols": missing,
                    "nan_total": int(np.isnan(arr).sum()),
                    "inf_total": int(np.isinf(arr).sum()),
                    "max_abs": nonfinite["max_abs"],
                    "all_finite": bool(nonfinite["all_finite"]),
                    "nan_by_col": nan_cols,
                }
            )
        else:
            row.update(
                {
                    "rows": int(len(df)),
                    "present_cols": [],
                    "missing_cols": missing,
                    "nan_total": 0,
                    "inf_total": 0,
                    "max_abs": 0.0,
                    "all_finite": True,
                    "nan_by_col": {},
                }
            )

        reports.append(row)

    all_finite = all(r.get("all_finite", False) for r in reports if r.get("exists", False))
    return {"all_finite": bool(all_finite), "datasets": reports}


def _collect_buffer_with_finite_checks(trainer, train_mod, steps_limit: int) -> dict[str, Any]:
    trainer._reset_train_envs()
    histories = trainer._init_histories()
    env_dones = {key: False for key in trainer.envs.keys()}

    steps = 0
    first_failure: dict[str, Any] | None = None

    while (not all(env_dones.values())) and steps < steps_limit:
        active = [(key, env) for key, env in trainer.envs.items() if not env_dones[key]]

        obs_map = {key: np.stack(histories[key], axis=0) for key, _ in active}
        obs_batch = np.stack([obs_map[key] for key, _ in active], axis=0)
        obs_stats = _finite_np(obs_batch)
        if not obs_stats["all_finite"]:
            first_failure = {
                "stage": "obs_batch",
                "details": obs_stats,
            }
            break

        obs_t = torch.as_tensor(obs_batch, device=train_mod.DEVICE)
        with torch.no_grad():
            action_t, _, _, _ = trainer.actor.sample(obs_t)
        action_batch = action_t.detach().cpu().numpy()
        act_stats = _finite_np(action_batch)
        if not act_stats["all_finite"]:
            first_failure = {
                "stage": "action_batch",
                "details": act_stats,
            }
            break

        for i, (key, env) in enumerate(active):
            action_exec = np.clip(action_batch[i], env.action_space.low, env.action_space.high)
            next_obs, rew, done, truncated, _ = env.step(action_exec)
            next_obs_vec = trainer._obs_vector(next_obs)

            if not np.isfinite(float(rew)):
                first_failure = {
                    "stage": "reward",
                    "env": key,
                    "value": float(rew),
                    "step": int(steps),
                }
                break

            next_obs_stats = _finite_np(next_obs_vec)
            if not next_obs_stats["all_finite"]:
                first_failure = {
                    "stage": "next_obs",
                    "env": key,
                    "step": int(steps),
                    "details": next_obs_stats,
                }
                break

            histories[key].append(next_obs_vec.copy())
            next_obs_seq = np.stack(histories[key], axis=0)
            trainer.buffer.add(
                obs_map[key],
                action_exec,
                float(rew) * trainer.hp.reward_scale,
                next_obs_seq,
                bool(done or truncated),
                stream_id=key,
            )

            env_dones[key] = bool(done or truncated)
            steps += 1

        if first_failure is not None:
            break

    return {
        "collected_steps": int(steps),
        "buffer_size": int(trainer.buffer.size),
        "first_failure": first_failure,
    }


def _stress_updates(trainer, updates: int, log_every: int) -> dict[str, Any]:
    if int(trainer.buffer.size) < int(trainer.hp.batch_size):
        return {
            "status": "skipped",
            "reason": "buffer_size < batch_size",
            "updates_done": 0,
            "snapshots": [],
            "failure": None,
        }

    snapshots: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    updates_done = 0

    for idx in range(1, updates + 1):
        try:
            trainer.update()
        except Exception as exc:  # noqa: BLE001
            failure = {
                "update_index": int(idx),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
            break

        updates_done += 1
        if idx % log_every == 0 or idx == updates:
            snapshots.append(
                {
                    "update_index": int(idx),
                    "q1_last": float(trainer.q1_values[-1]) if trainer.q1_values else None,
                    "backup_abs_max_last": float(trainer.backup_abs_maxs[-1]) if trainer.backup_abs_maxs else None,
                    "logp_min_last": float(trainer.logp_mins[-1]) if trainer.logp_mins else None,
                    "critic_loss_last": float(trainer.critic_losses[-1]) if trainer.critic_losses else None,
                    "actor_loss_last": float(trainer.actor_losses[-1]) if trainer.actor_losses else None,
                    "alpha": float(trainer.temperature.alpha.detach().cpu()),
                    "lambda": float(trainer.lmbda.detach().cpu().item()),
                }
            )

    return {
        "status": "failed" if failure is not None else "ok",
        "updates_done": int(updates_done),
        "snapshots": snapshots,
        "failure": failure,
    }


def _eval_sweep_finite(trainer, train_mod, max_runs: int) -> dict[str, Any]:
    runs = list(trainer.train_cfg.get("val", []))
    if max_runs > 0:
        runs = runs[:max_runs]

    actor_state_cpu = {k: v.detach().cpu() for k, v in trainer.actor.state_dict().items()}

    reports: list[dict[str, Any]] = []
    for deterministic in (True, False):
        mode = "det" if deterministic else "stoch"
        for run in runs:
            name = str(run.get("name", "unknown"))
            try:
                reward = float(
                    train_mod._eval_worker(
                        run,
                        trainer.parameters,
                        trainer.tariff,
                        trainer.actor_cfg,
                        actor_state_cpu,
                        trainer.episode_length,
                        trainer.history_len,
                        deterministic,
                    )
                )
                finite = math.isfinite(reward)
                reports.append(
                    {
                        "mode": mode,
                        "scenario": name,
                        "reward": reward,
                        "is_finite": bool(finite),
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                reports.append(
                    {
                        "mode": mode,
                        "scenario": name,
                        "reward": None,
                        "is_finite": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    all_finite = all(r["is_finite"] for r in reports)
    return {
        "all_finite": bool(all_finite),
        "runs_checked": int(len(reports)),
        "reports": reports,
    }


def run_diagnostic(
    model: str,
    tariff: str,
    batch_size: int,
    steps_limit: int,
    updates: int,
    log_every: int,
    max_eval_runs: int,
) -> dict[str, Any]:
    train_mod = _load_train_module(model)
    trainer = train_mod.Train(tariff)

    trainer.hp.batch_size = int(batch_size)
    trainer.eval_workers = 1
    trainer.train_env_workers = 1

    summary: dict[str, Any] = {
        "started_at": _now_iso(),
        "model": model,
        "tariff": tariff,
        "device": str(train_mod.DEVICE),
        "config_snapshot": {
            "actor_lr": float(trainer.hp.actor_lr),
            "critic_lr": float(trainer.hp.critic_lr),
            "alpha_lr": float(trainer.hp.α_lr),
            "history_len": int(trainer.history_len),
            "n_step": int(trainer.hp.n_step),
            "batch_size_effective": int(trainer.hp.batch_size),
            "reward_scale": float(trainer.hp.reward_scale),
        },
    }

    summary["dataset_scan"] = _scan_datasets_for_nonfinite(trainer.train_cfg)
    summary["buffer_collection"] = _collect_buffer_with_finite_checks(trainer, train_mod, steps_limit)
    summary["update_stress"] = _stress_updates(trainer, updates=updates, log_every=log_every)
    summary["eval_sweep"] = _eval_sweep_finite(trainer, train_mod, max_runs=max_eval_runs)
    summary["finished_at"] = _now_iso()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick non-finite diagnostic for RL training (TCN/TCNv2)."
    )
    parser.add_argument("--model", default="TCN", choices=["TCN", "TCNv2"])
    parser.add_argument("--tariff", default="tar_s")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps-limit", type=int, default=256)
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-eval-runs", type=int, default=8)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path. Default: Results/analysis/<auto_name>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_diagnostic(
        model=args.model,
        tariff=args.tariff,
        batch_size=args.batch_size,
        steps_limit=args.steps_limit,
        updates=args.updates,
        log_every=max(1, int(args.log_every)),
        max_eval_runs=max(1, int(args.max_eval_runs)),
    )

    out_path = (
        Path(args.out)
        if args.out is not None
        else ROOT
        / "Results"
        / "analysis"
        / f"diag_rl_nonfinite_{args.model}_{args.tariff}.json"
    )
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
