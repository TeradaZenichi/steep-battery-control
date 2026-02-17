from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../steep-battery-control
MODEL_ROOT   = Path(__file__).resolve().parents[2]  # .../models
MLP_ROOT     = Path(__file__).resolve().parent.parent   # .../models/MLP
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MLP_ROOT))
sys.path.insert(0, str(MODEL_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from environment import SmartHomeEnv
from model import load_actor
from opt import Teacher

# ------------------------------------------------------------
# Reward breakdown helper
# ------------------------------------------------------------

def enrich_operation_with_reward_breakdown(operation: pd.DataFrame, raw_df: pd.DataFrame, par: dict):
    """Add per-step reward/cost breakdown columns to SmartHomeEnv.operation.

    Reward in the env is:
        reward_t = -(energy_cost_t + grid_penalty_t + bess_cost_t + ev_cost_t)

    This function reconstructs additional interpretable components:
      - energy_cost split into: load, bess, ev, pv (exact, linear)
      - bess_cost split into: wear, saturation penalty (reconstructed)
      - ev_cost split into: wear, saturation penalty, departure penalty (reconstructed)

    Returns:
      op_enriched: DataFrame with extra columns
      totals: dict with summed components (costs and reward contributions)
    """
    op = operation.copy()

    # dt in hours (consistent with environment.Simulation.Δt)
    dt = float(par["general"]["timestep"]) / 60.0

    # -------------------------
    # EV connection mask
    # -------------------------
    if "ev_status" in raw_df.columns:
        ev_status = raw_df.loc[op.index, "ev_status"].astype(float)
    else:
        ev_status = (op["SoCEV"].astype(float) > 0.0).astype(float)

    ev_present = ev_status > 0.01
    prev_present = ev_present.shift(1, fill_value=False)
    connected_mask = (ev_present & prev_present).astype(float)
    departing_mask = ((~ev_present) & prev_present).astype(float)

    # Mask EV power and SoC between departure and arrival
    op["PEV"] = op["PEV"].astype(float) * connected_mask
    op["SoCEV"] = op["SoCEV"].astype(float) * connected_mask

    # -------------------------
    # Energy cost decomposition
    # -------------------------
    tar = op["tariff"].astype(float)
    op["energy_cost_load"] = op["PLoad"].astype(float) * tar * dt
    op["energy_cost_bess"] = op["PBESS"].astype(float) * tar * dt
    op["energy_cost_ev"]   = op["PEV"].astype(float) * tar * dt
    op["energy_cost_pv"]   = -op["PPV"].astype(float) * tar * dt

    op["energy_cost_recon"] = (
        op["energy_cost_load"]
        + op["energy_cost_bess"]
        + op["energy_cost_ev"]
        + op["energy_cost_pv"]
    )
    op["energy_cost_err"] = op["energy_cost"].astype(float) - op["energy_cost_recon"]

    # -------------------------
    # BESS cost decomposition
    # -------------------------
    bess = par["BESS"]
    Pmax_b = float(bess["Pmax"])
    wear_coeff_b = float(bess["capex"]) / (float(bess["Emax"]) * float(bess["ncycles"]))

    Pcmd_b = op["bess_cmd"].astype(float) * Pmax_b
    P_b = op["PBESS"].astype(float)
    eps = 1e-3
    sat_b = (P_b - Pcmd_b).abs()
    sat_b = (sat_b - eps).clip(lower=0.0)

    op["bess_wear_cost"] = wear_coeff_b * P_b.abs() * dt
    op["bess_sat_cost"]  = sat_b * float(bess["sat_penalty"]) * dt
    op["bess_cost_recon"] = op["bess_wear_cost"] + op["bess_sat_cost"]
    op["bess_cost_err"] = op["bess_cost"].astype(float) - op["bess_cost_recon"]

    # -------------------------
    # EV cost decomposition
    # -------------------------
    ev = par["EV"]
    Pmax_c = float(ev["Pmax_c"])
    Pmax_d = float(ev["Pmax_d"])
    wear_coeff_e = float(ev["capex"]) / (float(ev["Emax"]) * float(ev["ncycles"]))

    a_ev = op["ev_cmd"].astype(float)
    cmd_ev = np.where(a_ev >= 0.0, a_ev * Pmax_c, a_ev * Pmax_d)
    cmd_ev = pd.Series(cmd_ev, index=op.index)
    P_ev = op["PEV"].astype(float)

    sat_ev = (P_ev - cmd_ev).abs()
    sat_ev = (sat_ev - eps).clip(lower=0.0)

    op["ev_wear_cost"] = wear_coeff_e * P_ev.abs() * dt
    op["ev_sat_cost"]  = sat_ev * float(ev["sat_penalty"]) * dt * connected_mask

    # SoC min penalty (connected only)
    Emax_e = float(ev["Emax"])
    soc_min = float(ev["soc_min"])
    sev = (Emax_e * soc_min - op["EEV"].astype(float)).clip(lower=0.0)
    op["ev_soc_min_cost"] = sev * float(ev["penalty"]) * dt * connected_mask

    # Arrival fast-charging penalty
    if "ev_conn" in raw_df.columns and "ev_arrival" in raw_df.columns:
        conn_t = raw_df.loc[op.index, "ev_conn"].astype(int)
        connected_t = conn_t.isin([1, 2])
        connected_prev_t = conn_t.shift(1, fill_value=0).isin([1, 2])
        is_start_t = op.index == op.index.min()
        arrival_mask = connected_t & (~connected_prev_t) & (~is_start_t)

        E_dep = op["EEV"].astype(float).shift(1).fillna(0.0)
        E_trip = Emax_e * raw_df.loc[op.index, "ev_arrival"].astype(float)
        Ecrit = float(ev["soc_critical"]) * Emax_e
        E_leg = Emax_e - Ecrit

        fast_tariff = float(ev["fast_tariff"])
        tariff_t = op["tariff"].astype(float)

        E_pre = (E_dep - Ecrit).clip(lower=0.0)
        R = (E_trip - E_pre).clip(lower=0.0)

        n_fast = np.ceil(np.where(E_leg > 1e-9, R / E_leg, 0.0)).astype(float)
        arrival_cost = n_fast * fast_tariff * tariff_t * (E_leg if E_leg > 1e-9 else 0.0)
        op["ev_arrival_fast_cost"] = pd.Series(arrival_cost, index=op.index) * arrival_mask.astype(float)
    else:
        op["ev_arrival_fast_cost"] = 0.0

    op["ev_cost_recon"] = (
        op["ev_wear_cost"]
        + op["ev_sat_cost"]
        + op["ev_soc_min_cost"]
        + op["ev_arrival_fast_cost"]
    )
    op["ev_cost_err"] = op["ev_cost"].astype(float) - op["ev_cost_recon"]

    # -------------------------
    # Totals (costs and rewards)
    # -------------------------
    totals = {
        "total_reward": float(op["reward"].astype(float).sum()),
    }
    totals["total_cost"] = float(-totals["total_reward"])

    for k in ["energy_cost", "grid_penalty", "bess_cost", "ev_cost"]:
        totals[k] = float(op[k].astype(float).sum())

    for k in ["energy_cost_load", "energy_cost_bess", "energy_cost_ev", "energy_cost_pv"]:
        totals[k] = float(op[k].astype(float).sum())

    for k in ["bess_wear_cost", "bess_sat_cost"]:
        totals[k] = float(op[k].astype(float).sum())

    for k in ["ev_wear_cost", "ev_sat_cost", "ev_soc_min_cost", "ev_arrival_fast_cost"]:
        totals[k] = float(op[k].astype(float).sum())

    totals["total_penalties"] = float(
        totals["grid_penalty"]
        + totals["bess_sat_cost"]
        + totals["ev_sat_cost"]
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


def mask_operation_with_ev_conn(operation: pd.DataFrame, raw_df: pd.DataFrame):
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("data/parameters.json", encoding="utf-8") as f:
    par = json.load(f)

with open("models/MLP/model.json") as f:
    model_cfg = json.load(f)

with open("models/MLP/2-RL/config.json") as f:
    cfg = json.load(f)

seed = int(cfg["train"]["seed"])
torch.manual_seed(seed)
np.random.seed(seed)

# I/O and breakdown toggles to speed up tests
SAVE_OPERATION_CSV = True
SAVE_BREAKDOWN_CSV = False
INCLUDE_BREAKDOWN_SUMMARY = False

# for tariff in ["tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat"]:
for tariff in ["tar_s"]:

    folder = PROJECT_ROOT / "Results" / "test" / "MLP" / "2-RL" / tariff
    folder.mkdir(parents=True, exist_ok=True)

    summary = {}

    actor = load_actor(
        model_cfg["actor"],
        weights_path=f"Results/train/MLP/2-RL/{tariff}/best_actor_eval.pt",
        device=DEVICE,
    )

    # 1 ambiente (e 1 df) por dict em cfg["test"]
    for run in cfg["test"]:
        start = datetime.strptime(run["date"], "%Y-%m-%d %H:%M:%S")
        days = run["days"]
        BESS_SoC = run["soc"]

        df = pd.read_csv(
            run["dataset"],
            sep=";",
            parse_dates=["timestamp"],
            dayfirst=True,
            index_col="timestamp",
        )

        teacher = Teacher(df, par, start, days, BESS_SoC, tariff)
        teacher.build()
        teacher.solve()

        teacher_operation = teacher.get_operation()
        if SAVE_OPERATION_CSV:
            teacher_operation.to_csv(
                folder / f"{run['name']}_teacher_operation.csv",
                index_label="timestamp",
            )

        teacher_env = SmartHomeEnv(df, par, start, days, BESS_SoC, tariff)
        actor_env   = SmartHomeEnv(df, par, start, days, BESS_SoC, tariff)

        print(f"[{tariff}] {run['name']} - Starting teacher evaluation...")
        done = False
        teacher_reward = 0.0
        while not done:
            # Normalize actions to match SmartHomeEnv semantics (EV has asymmetric charge/discharge limits)
            pbess = float(teacher_operation.loc[teacher_env.sim.step, "PBESS"])
            pev   = float(teacher_operation.loc[teacher_env.sim.step, "PEV"])
            chi   = float(teacher_operation.loc[teacher_env.sim.step, "χPV"])

            a_bess = float(np.clip(pbess / teacher_env.bess.Pmax, -1.0, 1.0))
            if pev >= 0.0:
                a_ev = pev / teacher_env.ev.Pmax_c
            else:
                a_ev = pev / teacher_env.ev.Pmax_d
            a_ev = float(np.clip(a_ev, -1.0, 1.0))
            chi = float(np.clip(chi, 0.0, 1.0))

            action = [a_bess, a_ev, chi]
            state, reward, terminated, truncated, info = teacher_env.step(action)
            done = terminated or truncated
            teacher_reward += reward

        if SAVE_OPERATION_CSV:
            teacher_op_masked = mask_operation_with_ev_conn(teacher_env.operation, df)
            teacher_op_masked.to_csv(
                folder / f"{run['name']}_env_operation.csv",
                index_label="timestamp",
            )

        # Reward breakdown (teacher executed in env)
        teacher_totals = None
        if SAVE_BREAKDOWN_CSV or INCLUDE_BREAKDOWN_SUMMARY:
            teacher_op_break, teacher_totals = enrich_operation_with_reward_breakdown(
                teacher_env.operation,
                df,
                par,
            )
            if SAVE_BREAKDOWN_CSV:
                teacher_op_break.to_csv(
                    folder / f"{run['name']}_env_operation_breakdown.csv",
                    index_label="timestamp",
                )

        print(f"[{tariff}] {run['name']} - Starting actor evaluation...")
        done = False
        actor_reward = 0.0
        while not done:
            state = actor_env._get_observation()
            action = actor.action(state)  # determinístico + projeção
            state, reward, terminated, truncated, info = actor_env.step(action)
            done = terminated or truncated
            actor_reward += reward

        if SAVE_OPERATION_CSV:
            actor_op_masked = mask_operation_with_ev_conn(actor_env.operation, df)
            actor_op_masked.to_csv(
                folder / f"{run['name']}_actor_env_operation.csv",
                index_label="timestamp",
            )
        # Reward breakdown (actor executed in env)
        actor_totals = None
        if SAVE_BREAKDOWN_CSV or INCLUDE_BREAKDOWN_SUMMARY:
            actor_op_break, actor_totals = enrich_operation_with_reward_breakdown(
                actor_env.operation,
                df,
                par,
            )
            if SAVE_BREAKDOWN_CSV:
                actor_op_break.to_csv(
                    folder / f"{run['name']}_actor_env_operation_breakdown.csv",
                    index_label="timestamp",
                )

        summary[run["name"]] = {
            "teacher_reward": float(teacher_reward),
            "actor_reward": float(actor_reward),
            "reward_diff": float(actor_reward - teacher_reward),
            "teacher_breakdown": teacher_totals if INCLUDE_BREAKDOWN_SUMMARY else None,
            "actor_breakdown": actor_totals if INCLUDE_BREAKDOWN_SUMMARY else None,
            "dataset": run["dataset"],
            "date": run["date"],
            "days": run["days"],
            "soc": run["soc"],
        }

    with open(folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
