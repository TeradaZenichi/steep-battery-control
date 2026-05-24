"""Annual test: monthly metrics + weekly operation traces, raw and safety-projected."""
from __future__ import annotations

import csv
import os
import gzip
import importlib
import json
from calendar import monthrange
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from environment import SmartHomeEnv
from models._physics import Battery, EV
from reinforcement.sac_penalty.utils.safety import SafetyLayer


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default_monthly_runs():
    runs = []
    for tag, dataset in [
        ("cy", "data/Simulation_CY_Fut_HP__PV5000-HB5000.csv"),
        ("wy", "data/Simulation_WY_Fut_HP__PV5000-HB5000.csv"),
    ]:
        for month in range(1, 13):
            days = monthrange(2000, month)[1]
            runs.append({
                "name": f"test_{tag}_{month:02d}",
                "dataset": dataset,
                "date": f"2000-{month:02d}-01 00:00:00",
                "days": days,
                "soc": 0.5,
            })
    return runs


def _load_actor(arch_name, checkpoint, model_cfg, parameters_path):
    mod = importlib.import_module(f"models.{arch_name}.model")
    actor_cfg = dict(model_cfg["actor"])
    actor_cfg["parameters"] = str(parameters_path)
    actor = mod.load_actor(actor_cfg, device=torch.device("cpu"))
    state = torch.load(checkpoint, map_location="cpu")
    actor.load_state_dict(state["actor"] if isinstance(state, dict) and "actor" in state else state)
    actor.eval()
    return actor, actor_cfg


def _step_metrics(metrics, info, obs, action, params, safety_violation=np.nan, projection_delta=np.nan):
    dt = float(params["general"]["timestep"]) / 60.0
    pgrid = float(info["pgrid"])
    pbess = float(info["pbess"])
    pev = float(info["pev"])
    pv_avail = max(0.0, float(obs[11]) * float(params["general"]["Pnorm"]))

    metrics["reward"] += float(info["reward"])
    metrics["energy_cost"] += float(info["energy_cost"])
    metrics["bess_cost"] += float(info["bess_cost"])
    metrics["ev_cost"] += float(info["ev_cost"])
    metrics["pv_cost"] += float(info["pv_cost"])
    metrics["grid_penalty"] += float(info["penalty"])
    metrics["bess_abs_power"] += abs(pbess)
    metrics["ev_abs_power"] += abs(pev)
    metrics["bess_charge_kwh"] += max(pbess, 0.0) * dt
    metrics["bess_discharge_kwh"] += max(-pbess, 0.0) * dt
    metrics["ev_charge_kwh"] += max(pev, 0.0) * dt
    metrics["ev_discharge_kwh"] += max(-pev, 0.0) * dt
    metrics["pv_used_kwh"] += max(float(info["ppv"]), 0.0) * dt
    metrics["pv_curtailed_kwh"] += pv_avail * float(action[2]) * dt
    metrics["grid_import_kwh"] += max(pgrid, 0.0) * dt
    metrics["grid_export_kwh"] += max(-pgrid, 0.0) * dt
    metrics["grid_violation_kwh"] += (
        max(pgrid - float(params["Grid"]["Pmax_import"]), 0.0)
        + max(float(-params["Grid"]["Pmax_export"]) - pgrid, 0.0)
    ) * dt
    if np.isfinite(safety_violation):
        metrics["safety_violation"] += float(safety_violation)
    if np.isfinite(projection_delta):
        metrics["projection_delta"] += float(projection_delta)
    metrics["steps"] += 1


def _finish_metrics(metrics):
    steps = max(1, int(metrics["steps"]))
    out = dict(metrics)
    out["total_cost"] = -out["reward"]
    for key in ["bess_abs_power", "ev_abs_power", "safety_violation", "projection_delta"]:
        out[f"{key}_mean"] = out.pop(key) / steps
    return out


def _eval_run(run, tariff, actor, history_len, params, mode, save_operation=False, operation_days=None):
    df = pd.read_csv(
        PROJECT_ROOT / run["dataset"],
        sep=";",
        parse_dates=["timestamp"],
        dayfirst=True,
        index_col="timestamp",
    )
    days = int(run["days"] if operation_days is None else min(run["days"], operation_days))
    env = SmartHomeEnv(
        df, params, start=pd.to_datetime(run["date"], format="%Y-%m-%d %H:%M:%S"),
        days=days, BESS_SoC=float(run.get("soc", 0.5)), tariff=tariff,
        track_operation=False,
    )
    safety = None
    if mode == "safe":
        dt = float(params["general"]["timestep"]) / 60.0
        safety = SafetyLayer(Battery(params["BESS"], dt), EV(params["EV"], dt), params)
        safety.eval()

    obs, _ = env.reset()
    hist = deque([obs.copy() for _ in range(history_len)], maxlen=history_len)
    metrics = {k: 0.0 for k in [
        "reward", "energy_cost", "bess_cost", "ev_cost", "pv_cost",
        "grid_penalty", "bess_abs_power", "ev_abs_power", "bess_charge_kwh",
        "bess_discharge_kwh", "ev_charge_kwh", "ev_discharge_kwh",
        "pv_used_kwh", "pv_curtailed_kwh", "grid_import_kwh",
        "grid_export_kwh", "grid_violation_kwh", "safety_violation",
        "projection_delta",
    ]}
    metrics["steps"] = 0
    rows = []
    done = truncated = False
    while not (done or truncated):
        x = torch.as_tensor(np.stack(hist), dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            raw = actor.act(x, deterministic=True)
            action_t = raw
            violation = torch.zeros((1, 1))
            delta = torch.zeros((1, 1))
            if safety is not None:
                action_t, violation = safety.project(x, raw)
                delta = torch.mean(torch.abs(action_t - raw), dim=-1, keepdim=True)
        action = action_t.squeeze(0).cpu().numpy()
        raw_action = raw.squeeze(0).cpu().numpy()
        obs_before = obs.copy()
        obs, reward, done, truncated, info = env.step(action)
        info = dict(info)
        info["reward"] = float(reward)
        _step_metrics(
            metrics, info, obs_before, action, params,
            safety_violation=float(violation.squeeze().cpu()) if safety is not None else np.nan,
            projection_delta=float(delta.squeeze().cpu()) if safety is not None else np.nan,
        )
        if save_operation:
            rows.append({
                "timestamp": info["timestep"],
                "reward": float(reward),
                "tariff": float(obs_before[15]),
                "PLoad": float(obs_before[10]) * float(params["general"]["Pnorm"]),
                "PPV": float(info["ppv"]),
                "PBESS": float(info["pbess"]),
                "PEV": float(info["pev"]),
                "PGrid": float(info["pgrid"]),
                "SoCBESS": float(env.bess.soc),
                "SoCEV": float(env.ev.soc),
                "bess_cmd": float(action[0]),
                "ev_cmd": float(action[1]),
                "pv_cmd": float(action[2]),
                "raw_bess_cmd": float(raw_action[0]),
                "raw_ev_cmd": float(raw_action[1]),
                "raw_pv_cmd": float(raw_action[2]),
                "projection_delta": float(delta.squeeze().cpu()) if safety is not None else 0.0,
                "safety_violation": float(violation.squeeze().cpu()) if safety is not None else 0.0,
                "energy_cost": float(info["energy_cost"]),
                "bess_cost": float(info["bess_cost"]),
                "ev_cost": float(info["ev_cost"]),
                "pv_cost": float(info["pv_cost"]),
                "grid_penalty": float(info["penalty"]),
            })
        hist.append(obs.copy())
    env.close()
    return _finish_metrics(metrics), rows


def _aggregate(rows):
    rewards = np.asarray([r["reward"] for r in rows], dtype=np.float64)
    tail = min(2, len(rewards))
    worst = np.sort(rewards)[:tail]
    out = {
        "months": len(rows),
        "mean_reward": float(np.mean(rewards)),
        "worst_reward": float(np.min(rewards)),
        "robust_reward": float(np.mean(worst)),
        "std_reward": float(np.std(rewards)),
    }
    for key in rows[0]:
        if key in {"architecture", "tariff", "checkpoint", "mode", "scenario"}:
            continue
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            out[f"mean_{key}"] = float(np.mean(vals))
    return out


def run_test(algo_root: Path):
    algo_root = Path(algo_root)
    arch_name = algo_root.name
    model_root = PROJECT_ROOT / "models" / arch_name
    cfg = _read_json(algo_root / "config.json")
    model_cfg = _read_json(model_root / "model.json")
    params_path = PROJECT_ROOT / "data" / "parameters.json"
    params = _read_json(params_path)
    io = cfg.get("test_io", {})
    checkpoints = io.get("checkpoints", ["best", "best_robust", "best_worst", "last"])
    modes = ["raw", "safe"]
    suffix = os.environ.get("RUN_SUFFIX", "")
    env_tariffs = os.environ.get("RUN_TARIFFS")
    if env_tariffs:
        tariffs = env_tariffs.split(",")
    else:
        tariffs = io.get("tariffs", ["tar_s"])
    runs = cfg.get("test") or _default_monthly_runs()
    plot_days = int(io.get("plot_operation_days", 7))
    save_plot = bool(io.get("save_plot_operation", True))
    history_len = int(cfg["train"].get("history_len", 96))

    for tariff in tariffs:
        train_dir = PROJECT_ROOT / "paper" / "train" / "sac_penalty" / arch_name / f"{tariff}{suffix}"
        out_dir = PROJECT_ROOT / "paper" / "test" / "sac_penalty" / arch_name / f"{tariff}{suffix}"
        op_dir = out_dir / "operations"
        out_dir.mkdir(parents=True, exist_ok=True)
        op_dir.mkdir(parents=True, exist_ok=True)

        monthly_rows = []
        aggregate_rows = []
        for ckpt in checkpoints:
            ckpt_path = train_dir / f"{ckpt}.pt"
            if not ckpt_path.exists():
                continue
            actor, _ = _load_actor(arch_name, ckpt_path, model_cfg, params_path)
            for mode in modes:
                rows_for_agg = []
                for run in tqdm(runs, desc=f"{arch_name} {tariff} {ckpt} {mode}", dynamic_ncols=True):
                    metrics, _ = _eval_run(run, tariff, actor, history_len, params, mode)
                    row = {
                        "architecture": arch_name,
                        "tariff": tariff,
                        "checkpoint": ckpt,
                        "mode": mode,
                        "scenario": run["name"],
                        **metrics,
                    }
                    monthly_rows.append(row)
                    rows_for_agg.append(row)

                    if save_plot:
                        _, op_rows = _eval_run(
                            run, tariff, actor, history_len, params, mode,
                            save_operation=True, operation_days=plot_days,
                        )
                        op_path = op_dir / f"{run['name']}_{ckpt}_{mode}.csv.gz"
                        with gzip.open(op_path, "wt", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=list(op_rows[0].keys()))
                            writer.writeheader()
                            writer.writerows(op_rows)

                aggregate_rows.append({
                    "architecture": arch_name,
                    "tariff": tariff,
                    "checkpoint": ckpt,
                    "mode": mode,
                    **_aggregate(rows_for_agg),
                })

        if monthly_rows:
            with open(out_dir / "summary_monthly.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(monthly_rows[0].keys()))
                writer.writeheader()
                writer.writerows(monthly_rows)
        if aggregate_rows:
            with open(out_dir / "summary_by_checkpoint.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
                writer.writeheader()
                writer.writerows(aggregate_rows)
            with open(out_dir / "summary_overall.json", "w", encoding="utf-8") as f:
                json.dump(aggregate_rows, f, indent=2)
        (out_dir / ".test_done").touch()
