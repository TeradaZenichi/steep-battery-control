from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from environment import SmartHomeEnv
from opt import Teacher

TEACHER_SUMMARY_FILENAME = "teacher_summary.json"


def _resolve_dataset_path(dataset: str, project_root: Path) -> Path:
    dataset_path = Path(dataset)
    if dataset_path.is_absolute():
        return dataset_path
    candidate = project_root / dataset_path
    if candidate.exists():
        return candidate
    return dataset_path


def mask_operation_with_ev_conn(operation: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    op = operation.copy()

    if "ev_status" in raw_df.columns:
        ev_status = raw_df.loc[op.index, "ev_status"].astype(float)
    else:
        ev_status = (op["SoCEV"].astype(float) > 0.0).astype(float)

    ev_present = ev_status > 0.01
    prev_present = ev_present.shift(1, fill_value=False)
    connected_mask = (ev_present & prev_present).astype(float)

    op["PEV"] = op["PEV"].astype(float) * connected_mask
    op["SoCEV"] = op["SoCEV"].astype(float) * connected_mask
    op["EEV"] = op["EEV"].astype(float) * connected_mask

    return op


def enrich_operation_with_reward_breakdown(
    operation: pd.DataFrame,
    raw_df: pd.DataFrame,
    par: dict,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    op = operation.copy()

    dt = float(par["general"]["timestep"]) / 60.0

    if "ev_status" in raw_df.columns:
        ev_status = raw_df.loc[op.index, "ev_status"].astype(float)
    else:
        ev_status = (op["SoCEV"].astype(float) > 0.0).astype(float)

    ev_present = ev_status > 0.01
    prev_present = ev_present.shift(1, fill_value=False)
    connected_mask = (ev_present & prev_present).astype(float)

    op["PEV"] = op["PEV"].astype(float) * connected_mask
    op["SoCEV"] = op["SoCEV"].astype(float) * connected_mask

    tar = op["tariff"].astype(float)
    op["energy_cost_load"] = op["PLoad"].astype(float) * tar * dt
    op["energy_cost_bess"] = op["PBESS"].astype(float) * tar * dt
    op["energy_cost_ev"] = op["PEV"].astype(float) * tar * dt
    op["energy_cost_pv"] = op["energy_cost"].astype(float) - (
        op["energy_cost_load"] + op["energy_cost_bess"] + op["energy_cost_ev"]
    )
    op["energy_cost_recon"] = (
        op["energy_cost_load"]
        + op["energy_cost_bess"]
        + op["energy_cost_ev"]
        + op["energy_cost_pv"]
    )
    op["energy_cost_err"] = op["energy_cost"].astype(float) - op["energy_cost_recon"]

    bess = par["BESS"]
    wear_coeff_b = float(bess["capex"]) / (float(bess["Emax"]) * float(bess["ncycles"]))

    p_b = op["PBESS"].astype(float)
    op["bess_wear_cost"] = wear_coeff_b * p_b.abs() * dt
    op["bess_cost_recon"] = op["bess_wear_cost"]
    op["bess_cost_err"] = op["bess_cost"].astype(float) - op["bess_cost_recon"]

    ev = par["EV"]
    wear_coeff_e = float(ev["capex"]) / (float(ev["Emax"]) * float(ev["ncycles"]))

    p_ev = op["PEV"].astype(float)
    op["ev_wear_cost"] = wear_coeff_e * p_ev.abs() * dt

    emax_e = float(ev["Emax"])
    soc_min = float(ev["soc_min"])
    sev = (emax_e * soc_min - op["EEV"].astype(float)).clip(lower=0.0)
    op["ev_soc_min_cost"] = sev * float(ev["penalty"]) * dt * connected_mask

    if "ev_conn" in raw_df.columns and "ev_arrival" in raw_df.columns:
        conn_t = raw_df.loc[op.index, "ev_conn"].astype(int)
        connected_t = conn_t.isin([1, 2])
        connected_prev_t = conn_t.shift(1, fill_value=0).isin([1, 2])
        is_start_t = op.index == op.index.min()
        arrival_mask = connected_t & (~connected_prev_t) & (~is_start_t)

        e_dep = op["EEV"].astype(float).shift(1).fillna(0.0)
        e_trip = emax_e * raw_df.loc[op.index, "ev_arrival"].astype(float)
        ecrit = float(ev["soc_critical"]) * emax_e
        e_leg = emax_e - ecrit

        fast_tariff = float(ev["fast_tariff"])
        tariff_t = op["tariff"].astype(float)

        e_pre = (e_dep - ecrit).clip(lower=0.0)
        r_need = (e_trip - e_pre).clip(lower=0.0)

        n_fast = np.ceil(np.where(e_leg > 1e-9, r_need / e_leg, 0.0)).astype(float)
        arrival_cost = n_fast * fast_tariff * tariff_t * (e_leg if e_leg > 1e-9 else 0.0)
        op["ev_arrival_fast_cost"] = pd.Series(arrival_cost, index=op.index) * arrival_mask.astype(float)
    else:
        op["ev_arrival_fast_cost"] = 0.0

    op["ev_cost_recon"] = (
        op["ev_wear_cost"]
        + op["ev_soc_min_cost"]
        + op["ev_arrival_fast_cost"]
    )
    op["ev_cost_err"] = op["ev_cost"].astype(float) - op["ev_cost_recon"]

    totals: dict[str, Any] = {
        "total_reward": float(op["reward"].astype(float).sum()),
    }
    totals["total_cost"] = float(-totals["total_reward"])

    for key in ["energy_cost", "grid_penalty", "bess_cost", "ev_cost"]:
        totals[key] = float(op[key].astype(float).sum())

    for key in ["energy_cost_load", "energy_cost_bess", "energy_cost_ev", "energy_cost_pv"]:
        totals[key] = float(op[key].astype(float).sum())

    totals["bess_wear_cost"] = float(op["bess_wear_cost"].astype(float).sum())

    for key in ["ev_wear_cost", "ev_soc_min_cost", "ev_arrival_fast_cost"]:
        totals[key] = float(op[key].astype(float).sum())

    totals["total_penalties"] = float(
        totals["grid_penalty"]
        + totals["ev_soc_min_cost"]
        + totals["ev_arrival_fast_cost"]
    )

    totals["reward_components"] = {
        "reward_from_energy": -totals["energy_cost"],
        "reward_from_grid_penalty": -totals["grid_penalty"],
        "reward_from_bess": -totals["bess_cost"],
        "reward_from_ev": -totals["ev_cost"],
        "reward_from_demand_energy": -totals["energy_cost_load"],
        "reward_from_ev_energy": -totals["energy_cost_ev"],
        "reward_from_penalties_total": -totals["total_penalties"],
        "reward_from_ev_soc_min": -totals["ev_soc_min_cost"],
        "reward_from_ev_arrival_fast": -totals["ev_arrival_fast_cost"],
    }

    denom = abs(totals["total_cost"]) if abs(totals["total_cost"]) > 1e-12 else 1.0
    totals["cost_shares_abs"] = {
        "energy_cost": totals["energy_cost"] / denom,
        "grid_penalty": totals["grid_penalty"] / denom,
        "bess_cost": totals["bess_cost"] / denom,
        "ev_cost": totals["ev_cost"] / denom,
        "demand_energy_cost": totals["energy_cost_load"] / denom,
        "ev_energy_cost": totals["energy_cost_ev"] / denom,
        "penalties_total": totals["total_penalties"] / denom,
        "ev_soc_min_cost": totals["ev_soc_min_cost"] / denom,
        "ev_arrival_fast_cost": totals["ev_arrival_fast_cost"] / denom,
    }

    totals["max_abs_energy_cost_err"] = float(op["energy_cost_err"].abs().max())
    totals["max_abs_bess_cost_err"] = float(op["bess_cost_err"].abs().max())
    totals["max_abs_ev_cost_err"] = float(op["ev_cost_err"].abs().max())

    return op, totals


def run_teacher_runs(
    runs: list[dict[str, Any]],
    tariff: str,
    par: dict[str, Any],
    folder: Path,
    save_operation_csv: bool = True,
    save_breakdown_csv: bool = True,
    include_breakdown_summary: bool = True,
    show_progress: bool = True,
    pbar_position: int = 1,
) -> dict[str, dict[str, Any]]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[2]
    summary: dict[str, dict[str, Any]] = {}

    run_iter = runs
    if show_progress:
        run_iter = tqdm(
            runs,
            desc=f"{tariff} teacher runs (sequential)",
            position=pbar_position,
            dynamic_ncols=True,
            leave=False,
        )

    for run in run_iter:
        start = datetime.strptime(run["date"], "%Y-%m-%d %H:%M:%S")
        days = run["days"]
        bess_soc = run["soc"]
        max_steps = int((24 * 60 * float(days)) / float(par["general"]["timestep"]))

        dataset_path = _resolve_dataset_path(run["dataset"], project_root)

        with tqdm(
            total=4,
            desc=f"{run['name']} teacher stages",
            position=pbar_position + 1,
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        ) as pbar_teacher_stage:
            pbar_teacher_stage.set_postfix_str("loading data")
            df = pd.read_csv(
                dataset_path,
                sep=";",
                parse_dates=["timestamp"],
                dayfirst=True,
                index_col="timestamp",
            )
            pbar_teacher_stage.update(1)

            pbar_teacher_stage.set_postfix_str("building MILP")
            teacher = Teacher(df, par, start, days, bess_soc, tariff)
            teacher.build()
            pbar_teacher_stage.update(1)

            pbar_teacher_stage.set_postfix_str("solving MILP")
            teacher.solve()
            teacher_operation = teacher.get_operation()
            pbar_teacher_stage.update(1)

            if save_operation_csv:
                teacher_operation.to_csv(
                    folder / f"{run['name']}_teacher_operation.csv",
                    index_label="timestamp",
                )

            teacher_env = SmartHomeEnv(df, par, start, days, bess_soc, tariff)
            done = False
            teacher_reward = 0.0

            pbar_teacher_stage.set_postfix_str("env rollout")
            with tqdm(
                total=max_steps,
                desc=f"{run['name']} teacher",
                position=pbar_position + 2,
                dynamic_ncols=True,
                leave=False,
                disable=not show_progress,
            ) as pbar_teacher:
                while not done:
                    ts = teacher_env.sim.step
                    if ts not in teacher_operation.index:
                        raise KeyError(f"Timestamp {ts} not found in teacher operation index.")

                    action = teacher.get_actions(ts)
                    _, reward, terminated, truncated, _ = teacher_env.step(action)
                    done = terminated or truncated
                    teacher_reward += reward
                    pbar_teacher.update(1)

            pbar_teacher_stage.update(1)

        if save_operation_csv:
            teacher_op_masked = mask_operation_with_ev_conn(teacher_env.operation, df)
            teacher_op_masked.to_csv(
                folder / f"{run['name']}_env_operation.csv",
                index_label="timestamp",
            )

        teacher_totals = None
        if save_breakdown_csv or include_breakdown_summary:
            teacher_op_break, teacher_totals = enrich_operation_with_reward_breakdown(
                teacher_env.operation,
                df,
                par,
            )
            if save_breakdown_csv:
                teacher_op_break.to_csv(
                    folder / f"{run['name']}_env_operation_breakdown.csv",
                    index_label="timestamp",
                )

        summary[run["name"]] = {
            "teacher_reward": float(teacher_reward),
            "teacher_breakdown": teacher_totals if include_breakdown_summary else None,
            "dataset": run["dataset"],
            "date": run["date"],
            "days": run["days"],
            "soc": run["soc"],
        }

    return summary


def save_teacher_summary(folder: Path, summary: dict[str, Any]) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / TEACHER_SUMMARY_FILENAME, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


def load_teacher_summary(folder: Path) -> dict[str, Any]:
    summary_path = Path(folder) / TEACHER_SUMMARY_FILENAME
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)
