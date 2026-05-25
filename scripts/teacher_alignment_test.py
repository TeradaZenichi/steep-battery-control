"""Short teacher-vs-environment alignment check.

The MILP teacher is intentionally idealized, but it should still produce
actions that are useful when replayed in SmartHomeEnv. This script solves a few
short MILP windows, replays the teacher actions in the environment, and reports
where objective/cost/state trajectories diverge.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment import SmartHomeEnv
from opt import Teacher


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_df(run: dict) -> pd.DataFrame:
    return pd.read_csv(
        PROJECT_ROOT / run["dataset"],
        sep=";",
        parse_dates=["timestamp"],
        dayfirst=True,
        index_col="timestamp",
    )


def _solve_teacher(teacher: Teacher, solver_name: str):
    teacher.build()
    solver = pyo.SolverFactory(solver_name)
    if not solver.available(False):
        raise RuntimeError(f"Solver not available: {solver_name}")
    teacher.results = solver.solve(teacher.model, tee=False)
    teacher.get_operation()
    return teacher


def _replay_teacher_actions(df, params, run, tariff, teacher: Teacher, days: int):
    start = datetime.strptime(run["date"], "%Y-%m-%d %H:%M:%S")
    env = SmartHomeEnv(
        df, params, start, days, float(run["soc"]), tariff, track_operation=True
    )
    obs, _ = env.reset()
    total_reward = 0.0
    n = 0

    for t in teacher.model.Ωt:
        action = np.asarray(teacher.get_actions(t), dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        n += 1
        if terminated or truncated:
            break

    env_op = env.operation.copy()
    env.close()
    return total_reward, env_op, n


def _sum_col(df: pd.DataFrame, col: str) -> float:
    return float(df[col].sum()) if col in df.columns else float("nan")


def _mae_joined(left: pd.DataFrame, right: pd.DataFrame, col: str) -> float:
    if col not in left.columns or col not in right.columns:
        return float("nan")
    joined = left[[col]].join(right[[col]], how="inner", lsuffix="_teacher", rsuffix="_env")
    if joined.empty:
        return float("nan")
    return float(np.mean(np.abs(joined[f"{col}_teacher"] - joined[f"{col}_env"])))


def _socev_connected_mae(df: pd.DataFrame, teacher_op: pd.DataFrame, env_op: pd.DataFrame) -> float:
    if "SoCEV" not in teacher_op.columns or "SoCEV" not in env_op.columns:
        return float("nan")
    joined = teacher_op[["SoCEV"]].join(
        env_op[["SoCEV"]], how="inner", lsuffix="_teacher", rsuffix="_env"
    )
    if joined.empty:
        return float("nan")
    conn = df.loc[joined.index, "ev_conn"].astype(int)
    mask = conn.isin([1, 2]).to_numpy()
    if not np.any(mask):
        return float("nan")
    delta = joined["SoCEV_teacher"].to_numpy()[mask] - joined["SoCEV_env"].to_numpy()[mask]
    return float(np.mean(np.abs(delta)))


def evaluate_run(run: dict, tariff: str, params: dict, solver: str, days: int) -> dict:
    run_eval = dict(run)
    run_eval["days"] = int(days)
    df = _load_df(run_eval)
    start = datetime.strptime(run_eval["date"], "%Y-%m-%d %H:%M:%S")

    teacher = Teacher(df, params, start, int(days), float(run_eval["soc"]), tariff)
    teacher = _solve_teacher(teacher, solver)
    env_reward, env_op, steps = _replay_teacher_actions(
        df, params, run_eval, tariff, teacher, int(days)
    )

    teacher_op = teacher.operation.copy()
    milp_objective = float(pyo.value(teacher.model.objective))
    env_total_cost = -float(env_reward)

    row = {
        "run": run_eval.get("name", "run"),
        "tariff": tariff,
        "date": run_eval["date"],
        "days": int(days),
        "steps": int(steps),
        "milp_objective": milp_objective,
        "env_replay_reward": float(env_reward),
        "env_replay_cost": env_total_cost,
        "env_minus_milp_cost": env_total_cost - milp_objective,
        "env_energy_cost": _sum_col(env_op, "energy_cost"),
        "env_bess_cost": _sum_col(env_op, "bess_cost"),
        "env_ev_cost": _sum_col(env_op, "ev_cost"),
        "env_pv_cost": _sum_col(env_op, "pv_cost"),
        "env_grid_penalty": _sum_col(env_op, "grid_penalty"),
        "teacher_energy_cost": _sum_col(teacher_op, "energy_cost"),
        "teacher_bess_cost": _sum_col(teacher_op, "bess_cost"),
        "teacher_ev_cost": _sum_col(teacher_op, "ev_cost"),
        "env_minus_teacher_ev_cost": _sum_col(env_op, "ev_cost") - _sum_col(teacher_op, "ev_cost"),
        "pbess_mae": _mae_joined(teacher_op, env_op, "PBESS"),
        "pev_mae": _mae_joined(teacher_op, env_op, "PEV"),
        "pgrid_mae": _mae_joined(teacher_op, env_op, "PGrid"),
        "socbess_mae": _mae_joined(teacher_op, env_op, "SoCBESS"),
        "socev_mae": _mae_joined(teacher_op, env_op, "SoCEV"),
        "socev_connected_mae": _socev_connected_mae(df, teacher_op, env_op),
        "env_bess_abs_power_mean": float(env_op["PBESS"].abs().mean()),
        "env_ev_abs_power_mean": float(env_op["PEV"].abs().mean()),
    }
    return row


def _write_markdown(rows: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    cols = [
        "run", "days", "milp_objective", "env_replay_cost",
        "env_minus_milp_cost", "env_ev_cost", "env_grid_penalty",
        "env_minus_teacher_ev_cost", "pbess_mae", "pev_mae",
        "socbess_mae", "socev_connected_mae",
    ]
    def md_table(frame: pd.DataFrame) -> str:
        frame = frame.copy()
        for col in frame.columns:
            if pd.api.types.is_float_dtype(frame[col]):
                frame[col] = frame[col].map(lambda v: f"{v:.4f}")
        headers = list(frame.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in frame.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    aggregate = pd.DataFrame([
        df.drop(columns=["run", "tariff", "date"]).mean(numeric_only=True).to_dict()
    ])
    lines = [
        "# Teacher Alignment Test",
        "",
        "MILP actions replayed in `SmartHomeEnv`. Positive `env_minus_milp_cost` means the environment made the teacher trajectory more expensive than the MILP objective.",
        "",
        md_table(df[cols]),
        "",
        "## Aggregate",
        "",
        md_table(aggregate),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tariff", default="tar_s")
    parser.add_argument("--arch-config", default="supervised/TCN/config.json")
    parser.add_argument("--solver", default="gurobi")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--max-runs", type=int, default=3)
    parser.add_argument("--out-dir", default=".review/teacher_alignment")
    args = parser.parse_args()

    params = _read_json(PROJECT_ROOT / "data" / "parameters.json")
    cfg = _read_json(PROJECT_ROOT / args.arch_config)
    runs = list(cfg["runs"])[: int(args.max_runs)]

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in runs:
        print(f"[teacher-align] {args.tariff} {run['name']} {args.days}d")
        rows.append(evaluate_run(run, args.tariff, params, args.solver, args.days))

    stem = f"{args.tariff}_{args.days}d_{len(rows)}runs"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_markdown(rows, md_path)

    print(f"[teacher-align] wrote {csv_path}")
    print(f"[teacher-align] wrote {md_path}")


if __name__ == "__main__":
    main()
