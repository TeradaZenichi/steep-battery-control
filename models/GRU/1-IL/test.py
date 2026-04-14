from datetime import datetime
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import torch
import json
import sys
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../steep-battery-control
MODEL_ROOT   = Path(__file__).resolve().parents[2]  # .../models
GRU_ROOT     = Path(__file__).resolve().parents[1]  # .../models/GRU
ALGO_ROOT    = Path(__file__).resolve().parent      # .../models/GRU/1-IL
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GRU_ROOT))
sys.path.insert(0, str(MODEL_ROOT))
sys.path.append(str(ALGO_ROOT))

from environment import SmartHomeEnv
from model import load_actor
from test_utils.teacher_eval import load_teacher_summary

# ------------------------------------------------------------
# Reward breakdown helper
# ------------------------------------------------------------

def enrich_operation_with_reward_breakdown(operation: pd.DataFrame, raw_df: pd.DataFrame, par: dict):
    op = operation.copy()

    # dt in hours (consistent with environment.Simulation.?t)
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
    # arriving step: first positive
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
    op["bess_wear_cost"] = wear_coeff_b * P_b.abs() * dt
    op["bess_cost_recon"] = op["bess_wear_cost"]
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
    op["ev_wear_cost"] = wear_coeff_e * P_ev.abs() * dt
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

    # Base components used by the env
    for k in ["energy_cost", "grid_penalty", "bess_cost", "ev_cost"]:
        totals[k] = float(op[k].astype(float).sum())

    # Energy breakdown
    for k in ["energy_cost_load", "energy_cost_bess", "energy_cost_ev", "energy_cost_pv"]:
        totals[k] = float(op[k].astype(float).sum())

    # BESS breakdown
    for k in ["bess_wear_cost"]:
        totals[k] = float(op[k].astype(float).sum())

    # EV breakdown
    for k in ["ev_wear_cost", "ev_soc_min_cost", "ev_arrival_fast_cost"]:
        totals[k] = float(op[k].astype(float).sum())

    totals["total_penalties"] = float(
        totals["grid_penalty"]
        + totals["ev_soc_min_cost"]
        + totals["ev_arrival_fast_cost"]
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

    # Sanity checks (should be near 0)
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


def eval_actor_run_parallel(run: dict, tariff: str, par: dict, actor_cfg: dict, actor_state_dict: dict, use_projection: bool, save_operation_csv: bool, save_breakdown_csv: bool, include_breakdown_summary: bool, folder: Path, show_step_pbar: bool, history_len: int, pbar_position: int):
    start = datetime.strptime(run["date"], "%Y-%m-%d %H:%M:%S")
    days = run["days"]
    bess_soc = run["soc"]

    df = pd.read_csv(
        run["dataset"],
        sep=";",
        parse_dates=["timestamp"],
        dayfirst=True,
        index_col="timestamp",
    )

    actor = load_actor(actor_cfg, device=torch.device("cpu"))
    actor.load_state_dict(actor_state_dict, strict=True)
    actor.eval()

    actor_env = SmartHomeEnv(df, par, start, days, bess_soc, tariff)
    max_steps = int((24 * 60 * float(days)) / float(par["general"]["timestep"]))
    history_len = max(1, int(history_len))

    state0 = np.asarray(actor_env._get_observation(), dtype=np.float32).reshape(-1)
    history = deque([state0.copy() for _ in range(history_len)], maxlen=history_len)

    done = False
    actor_reward = 0.0
    with tqdm(
        total=max_steps,
        desc=f"{run['name']} actor",
        position=pbar_position,
        dynamic_ncols=True,
        leave=False,
        disable=not show_step_pbar,
    ) as pbar_actor:
        while not done:
            state_seq = np.stack(history, axis=0)
            state_t = torch.as_tensor(state_seq, dtype=torch.float32, device=torch.device("cpu")).unsqueeze(0)
            with torch.no_grad():
                if use_projection:
                    _, _, action_t, _ = actor.sample(state_t)  # deterministic + projection
                else:
                    action_t, _, _, _ = actor.sample(state_t)  # stochastic + projection
            action = action_t.squeeze(0).detach().cpu().numpy()

            next_state, reward, terminated, truncated, info = actor_env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
            history.append(next_state.copy())
            done = terminated or truncated
            actor_reward += reward
            pbar_actor.update(1)

    if save_operation_csv:
        actor_op_masked = mask_operation_with_ev_conn(actor_env.operation, df)
        actor_op_masked.to_csv(
            folder / f"{run['name']}_actor_env_operation.csv",
            index_label="timestamp",
        )

    actor_totals = None
    if save_breakdown_csv or include_breakdown_summary:
        actor_op_break, actor_totals = enrich_operation_with_reward_breakdown(
            actor_env.operation,
            df,
            par,
        )
        if save_breakdown_csv:
            actor_op_break.to_csv(
                folder / f"{run['name']}_actor_env_operation_breakdown.csv",
                index_label="timestamp",
            )

    return {
        "name": run["name"],
        "actor_reward": float(actor_reward),
        "actor_breakdown": actor_totals if include_breakdown_summary else None,
    }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8") as f:
    par = json.load(f)

with open(ALGO_ROOT / "config.json", encoding="utf-8") as f:
    test_cfg = json.load(f)

with open(GRU_ROOT / "model.json", encoding="utf-8") as f:
    model_cfg = json.load(f)

model_cfg["actor"]["parameters"] = str(PROJECT_ROOT / "data" / "parameters.json")

torch.manual_seed(test_cfg["seed"])
np.random.seed(test_cfg["seed"])
EVAL_WORKERS = int(test_cfg.get("eval_workers", 1))
SHOW_ACTOR_STEP_PBAR = bool(test_cfg.get("show_actor_step_pbar", True))

# Toggle: in IL evaluation, you may want to run with the same feasibility projection as RL.
# - False: uses actor.predict(state_t) (no projection)
# - True:  uses actor.action(state) (deterministic + projection)
USE_PROJECTION = True

# I/O and breakdown toggles to speed up tests
SAVE_OPERATION_CSV = True
SAVE_BREAKDOWN_CSV = True
INCLUDE_BREAKDOWN_SUMMARY = True

for tariff in tqdm(["tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat"], desc="Tariffs", position=0, dynamic_ncols=True):
    folder = PROJECT_ROOT / "Results" / "test" / "GRU" / "1-IL" / tariff
    folder.mkdir(parents=True, exist_ok=True)

    history_len = int(test_cfg.get("training", {}).get("history_len", 1))
    best_params_path = PROJECT_ROOT / "Results" / "train" / "GRU" / "1-IL" / tariff / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path, "r", encoding="utf-8") as f:
            history_len = int(json.load(f).get("history_len", history_len))

    summary = {}

    actor_state_dict = torch.load(
        PROJECT_ROOT / "Results" / "train" / "GRU" / "1-IL" / tariff / "best.pth",
        map_location=torch.device("cpu"),
    )

    runs = list(test_cfg["test"])
    actor_results = {}
    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as actor_pool:
        futures = {
            actor_pool.submit(
                eval_actor_run_parallel,
                run,
                tariff,
                par,
                model_cfg["actor"],
                actor_state_dict,
                USE_PROJECTION,
                SAVE_OPERATION_CSV,
                SAVE_BREAKDOWN_CSV,
                INCLUDE_BREAKDOWN_SUMMARY,
                folder,
                SHOW_ACTOR_STEP_PBAR,
                history_len,
                2 + idx,
            ): run["name"]
            for idx, run in enumerate(runs)
        }

        with tqdm(total=len(futures), desc=f"{tariff} actor runs (parallel, w={EVAL_WORKERS})", position=1, dynamic_ncols=True, leave=False) as pbar_actor_runs:
            for future in as_completed(futures):
                result = future.result()
                actor_results[result["name"]] = result
                pbar_actor_runs.update(1)

    teacher_summary = load_teacher_summary(folder)
    missing_teacher_runs = [run["name"] for run in runs if run["name"] not in teacher_summary]
    if missing_teacher_runs:
        preview = ", ".join(missing_teacher_runs[:5])
        raise RuntimeError(
            f"Teacher summary mismatch for {tariff}: missing {len(missing_teacher_runs)} of {len(runs)} runs ({preview}). "
            "Run generate_teacher_test_baseline.py so teacher uses all tariffs and the same test set."
        )
    for run in runs:
        teacher_info = teacher_summary[run["name"]]
        teacher_reward = float(teacher_info["teacher_reward"])
        teacher_totals = teacher_info.get("teacher_breakdown", None)

        actor_info = actor_results.get(run["name"], {})
        actor_reward = float(actor_info.get("actor_reward", np.nan))
        actor_totals = actor_info.get("actor_breakdown", None)

        summary[run["name"]] = {
            "teacher_reward": teacher_reward,
            "actor_reward": actor_reward,
            "reward_diff": actor_reward - teacher_reward,
            "teacher_breakdown": teacher_totals if INCLUDE_BREAKDOWN_SUMMARY else None,
            "actor_breakdown": actor_totals if INCLUDE_BREAKDOWN_SUMMARY else None,
            "use_projection": USE_PROJECTION,
            "dataset": run["dataset"],
            "date": run["date"],
            "days": run["days"],
            "soc": run["soc"],
        }

    with open(folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

