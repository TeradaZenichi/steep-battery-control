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
    op = operation.copy()

    # dt in hours (consistent with environment.Simulation.Δt)
    dt = float(par["general"]["timestep"]) / 60.0

    # -------------------------
    # Energy cost decomposition
    # -------------------------
    tar = op["tariff"].astype(float)
    op["energy_cost_load"] = op["PLoad"].astype(float) * tar * dt
    op["energy_cost_bess"] = op["PBESS"].astype(float) * tar * dt
    op["energy_cost_ev"]   = op["PEV"].astype(float) * tar * dt
    # PV reduces net grid import; keep the sign consistent with PGrid = load + bess + ev - pv
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

    # EV presence derived from raw data (matches EVEnv finite-state machine)
    if "ev_status" in raw_df.columns:
        ev_status = raw_df.loc[op.index, "ev_status"].astype(float)
    else:
        # Fallback: infer presence from SoCEV > 0 (less reliable)
        ev_status = (op["SoCEV"].astype(float) > 0.0).astype(float)

    ev_present = ev_status > 0.01
    prev_present = ev_present.shift(1).fillna(False)
    # arriving step: first positive
    connected_mask = (ev_present & prev_present).astype(float)
    departing_mask = ((~ev_present) & prev_present).astype(float)

    a_ev = op["ev_cmd"].astype(float)
    cmd_ev = np.where(a_ev >= 0.0, a_ev * Pmax_c, a_ev * Pmax_d)
    cmd_ev = pd.Series(cmd_ev, index=op.index)
    P_ev = op["PEV"].astype(float)

    sat_ev = (P_ev - cmd_ev).abs()
    sat_ev = (sat_ev - eps).clip(lower=0.0)

    op["ev_wear_cost"] = wear_coeff_e * P_ev.abs() * dt
    op["ev_sat_cost"]  = sat_ev * float(ev["sat_penalty"]) * dt * connected_mask

    # Departure penalty occurs on the first step after presence turns off.
    # EVEnv uses the SoC *before* resetting to 0. We recover it from SoCEV shifted by 1.
    thresholds = np.array(ev["departure_penalty"]["thresholds"], dtype=float)
    weights = np.array(ev["departure_penalty"]["weights"], dtype=float)
    soc_before_depart = op["SoCEV"].astype(float).shift(1).fillna(0.0).to_numpy()
    idx = np.searchsorted(thresholds, soc_before_depart, side="right")
    idx = np.clip(idx, 0, len(weights) - 1)
    dep_cost = weights[idx] * (1.0 - soc_before_depart)
    op["ev_departure_cost"] = pd.Series(dep_cost, index=op.index) * departing_mask

    op["ev_cost_recon"] = op["ev_wear_cost"] + op["ev_sat_cost"] + op["ev_departure_cost"]
    op["ev_cost_err"] = op["ev_cost"].astype(float) - op["ev_cost_recon"]

    # -------------------------
    # Totals (costs and rewards)
    # -------------------------
    totals = {
        "total_reward": float(op["reward"].astype(float).sum()),
    }
    totals["total_cost"] = float(-totals["total_reward"])

    # Base components used by the env
    for k in ["energy_cost", "grid_penalty", "bess_cost", "ev_cost"]:
        totals[k] = float(op[k].astype(float).sum())

    # Energy breakdown
    for k in ["energy_cost_load", "energy_cost_bess", "energy_cost_ev", "energy_cost_pv"]:
        totals[k] = float(op[k].astype(float).sum())

    # BESS breakdown
    for k in ["bess_wear_cost", "bess_sat_cost"]:
        totals[k] = float(op[k].astype(float).sum())

    # EV breakdown
    for k in ["ev_wear_cost", "ev_sat_cost", "ev_departure_cost"]:
        totals[k] = float(op[k].astype(float).sum())

    totals["total_penalties"] = float(
        totals["grid_penalty"]
        + totals["bess_sat_cost"]
        + totals["ev_sat_cost"]
        + totals["ev_departure_cost"]
    )

    # Reward contributions (negative of costs)
    totals["reward_components"] = {
        "reward_from_energy": -totals["energy_cost"],
        "reward_from_grid_penalty": -totals["grid_penalty"],
        "reward_from_bess": -totals["bess_cost"],
        "reward_from_ev": -totals["ev_cost"],
        "reward_from_demand_energy": -totals["energy_cost_load"],
        "reward_from_ev_energy": -totals["energy_cost_ev"],
        "reward_from_penalties_total": -totals["total_penalties"],
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
    }

    # Sanity checks (should be near 0)
    totals["max_abs_energy_cost_err"] = float(op["energy_cost_err"].abs().max())
    totals["max_abs_bess_cost_err"] = float(op["bess_cost_err"].abs().max())
    totals["max_abs_ev_cost_err"] = float(op["ev_cost_err"].abs().max())

    return op, totals

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("data/parameters.json", encoding="utf-8") as f:
    par = json.load(f)

with open("models/MLP/1-IL/config.json") as f:
    test_cfg = json.load(f)

with open("models/MLP/model.json") as f:
    model_cfg = json.load(f)

torch.manual_seed(test_cfg["seed"])
np.random.seed(test_cfg["seed"])

# Toggle: in IL evaluation, you may want to run with the same feasibility projection as RL.
# - False: uses actor.predict(state_t) (no projection)
# - True:  uses actor.action(state) (deterministic + projection)
USE_PROJECTION = False

for tariff in ["tar_s"]: #, "tar_w", "tar_sw", "tar_tou", "tar_flat"]:
    folder = PROJECT_ROOT / "Results" / "test" / "MLP" / "1-IL" / tariff
    folder.mkdir(parents=True, exist_ok=True)

    summary = {}

    for run in test_cfg["test"]:
        start = datetime.strptime(run["date"], "%Y-%m-%d %H:%M:%S")
        days = run["days"]
        BESS_SoC = run["soc"]

        # NEW: load a dataset per test dict
        df = pd.read_csv(
            run["dataset"],
            sep=";",
            parse_dates=["timestamp"],
            dayfirst=True,
            index_col="timestamp",
        )

        actor = load_actor(
            model_cfg["actor"],
            weights_path=f"Results/train/MLP/1-IL/{tariff}/best.pth",
            device=DEVICE,
        )

        teacher = Teacher(df, par, start, days, BESS_SoC, tariff)
        teacher.build()
        teacher.solve()

        teacher_operation = teacher.get_operation()
        teacher_operation.to_csv(
            f"Results/test/MLP/1-IL/{tariff}/{run['name']}_teacher_operation.csv",
            index_label="timestamp",
        )

        teacher_env = SmartHomeEnv(df, par, start, days, BESS_SoC, tariff)
        actor_env = SmartHomeEnv(df, par, start, days, BESS_SoC, tariff)

        print("Starting teacher evaluation...")
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

        teacher_env.operation.to_csv(
            f"Results/test/MLP/1-IL/{tariff}/{run['name']}_env_operation.csv",
            index_label="timestamp",
        )

        # Reward breakdown (teacher executed in env)
        teacher_op_break, teacher_totals = enrich_operation_with_reward_breakdown(teacher_env.operation, df, par)
        teacher_op_break.to_csv(
            f"Results/test/MLP/1-IL/{tariff}/{run['name']}_env_operation_breakdown.csv",
            index_label="timestamp",
        )

        print("Starting actor evaluation...")
        done = False
        actor_reward = 0.0
        while not done:
            state = actor_env._get_observation()

            if USE_PROJECTION:
                action = actor.action(state)  # deterministic + projection
            else:
                # Minimal device fix: ensure obs is on the same device as the actor
                state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
                action = actor.predict(state_t)  # no projection

            state, reward, terminated, truncated, info = actor_env.step(action)
            done = terminated or truncated
            actor_reward += reward

        actor_env.operation.to_csv(
            f"Results/test/MLP/1-IL/{tariff}/{run['name']}_actor_env_operation.csv",
            index_label="timestamp",
        )

        # Reward breakdown (actor executed in env)
        actor_op_break, actor_totals = enrich_operation_with_reward_breakdown(actor_env.operation, df, par)
        actor_op_break.to_csv(
            f"Results/test/MLP/1-IL/{tariff}/{run['name']}_actor_env_operation_breakdown.csv",
            index_label="timestamp",
        )

        summary[run["name"]] = {
            "teacher_reward": teacher_reward,
            "actor_reward": actor_reward,
            "reward_diff": actor_reward - teacher_reward,
            "teacher_breakdown": teacher_totals,
            "actor_breakdown": actor_totals,
            "use_projection": USE_PROJECTION,
            "dataset": run["dataset"],
            "date": run["date"],
            "days": run["days"],
            "soc": run["soc"],
        }

    with open(folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
