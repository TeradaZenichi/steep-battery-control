"""MILP teacher upper bound on the held-out test windows.

Solves the perfect-foresight MILP (opt.Teacher) on the same monthly test runs
used by the RL test pipeline, replays the teacher actions in SmartHomeEnv, and
writes a summary in the EXACT format of the RL test summaries so it slots into
the consolidation script and the paper/test tree as the optimization upper
bound.

It reuses `reinforcement.sac_cmdp.utils.test_eval._eval_run` and `_aggregate`,
so the per-run metrics and the annual aggregation are computed identically to
the learned controllers (apples-to-apples). The teacher is architecture- and
method-independent: one solve per (tariff, run) is reused for all comparisons.

Output:
    paper/test/teacher/<tariff>/summary_overall.json   # checkpoint="teacher"
    paper/test/teacher/<tariff>/.test_done

Run (from project root):
    python scripts/teacher_test.py                       # all 5 tariffs, gurobi
    python scripts/teacher_test.py --tariffs tar_sw --solver appsi_highs
    python scripts/teacher_test.py --overwrite
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from opt import Teacher
from reinforcement.sac_cmdp.utils.test_eval import (
    _eval_run, _aggregate, _default_monthly_runs,
)

DEFAULT_TARIFFS = ["tar_s", "tar_w", "tar_sw", "tar_flat", "tar_tou"]


class TeacherActor:
    """Replays a precomputed MILP action sequence through the `.act` interface
    expected by `_eval_run`. Ignores the observation; returns the next action."""

    def __init__(self, actions):
        self._actions = actions
        self._i = 0

    def act(self, x, deterministic=True):
        if self._i >= len(self._actions):
            # Env stepped past the solved horizon — should not happen if the
            # teacher window matches the env window. Hold the last action.
            a = self._actions[-1]
        else:
            a = self._actions[self._i]
            self._i += 1
        return torch.as_tensor(np.asarray(a, dtype=np.float32)).unsqueeze(0)


def _load_run_df(run):
    return pd.read_csv(
        PROJECT_ROOT / run["dataset"], sep=";",
        parse_dates=["timestamp"], dayfirst=True, index_col="timestamp",
    )


def _solve_teacher_actions(run, tariff, params, solver):
    df = _load_run_df(run)
    start = datetime.strptime(run["date"], "%Y-%m-%d %H:%M:%S")
    teacher = Teacher(df, params, start, run["days"], run.get("soc", 0.5), tariff)
    teacher.build()
    teacher.solve(solver=solver)
    teacher.get_operation()
    return [np.asarray(teacher.get_actions(t), dtype=np.float32) for t in teacher.model.Ωt]


def run_tariff(tariff, params, solver, overwrite):
    out_dir = PROJECT_ROOT / "paper" / "test" / "teacher" / tariff
    summary_path = out_dir / "summary_overall.json"
    if summary_path.exists() and not overwrite:
        print(f"[teacher] {tariff}: summary exists, skipping (use --overwrite)")
        return

    runs = _default_monthly_runs()
    per_run = []
    for run in tqdm(runs, desc=f"teacher {tariff}", dynamic_ncols=True):
        actions = _solve_teacher_actions(run, tariff, params, solver)
        actor = TeacherActor(actions)
        metrics, _ = _eval_run(run, tariff, actor, history_len=1, params=params, mode="raw")
        metrics["scenario"] = run["name"]
        per_run.append(metrics)

    agg = _aggregate(per_run)
    record = {
        "architecture": "teacher",
        "tariff": tariff,
        "checkpoint": "teacher",
        "mode": "raw",
        **agg,
        "mean_total_cost": -float(agg["mean_reward"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump([record], f, indent=2)
    (out_dir / ".test_done").touch()
    print(f"[teacher] {tariff}: mean_reward={agg['mean_reward']:.2f} "
          f"worst={agg['worst_reward']:.2f} -> {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tariffs", type=str, default=",".join(DEFAULT_TARIFFS))
    parser.add_argument("--solver", type=str, default="gurobi",
                        help="Pyomo solver name (gurobi, appsi_highs, cbc, ...).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tariffs = [t.strip() for t in args.tariffs.split(",") if t.strip()]
    with open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8") as f:
        params = json.load(f)

    for tariff in tariffs:
        run_tariff(tariff, params, args.solver, args.overwrite)


if __name__ == "__main__":
    main()
