"""MILP perfect-foresight reference over the annual test windows.

We record TWO numbers per window:

  (1) milp_objective  -- the idealized perfect-foresight optimum on the MILP model.
      This is the standard HEMS/MPC oracle benchmark and serves as the cost CEILING
      (lower bound on cost / upper bound on reward). Optimistic by construction.

  (2) openloop replay -- the SAME teacher actions replayed open-loop in SmartHomeEnv
      (through the safety projection the learned policies use). This is NOT a valid
      ceiling: open-loop replay lets the EV SoC drift from the MILP plan (teacher-env
      dynamics mismatch), so the EV departure-SoC penalty inflates cost far above the
      objective and above the closed-loop learned policies. We keep it only to quantify
      the open-loop gap -- itself evidence for closed-loop learning over open-loop
      optimization.

The summary's mean_reward = -mean(milp_objective) (the idealized ceiling), which
scripts/make_paper_tables.py picks up automatically; openloop_* fields document the gap.

Writes paper/ablation/test/baselines/MILP/{tariff}/summary_overall.json.

  python scripts/make_milp_ceiling.py --test1   # time a single full-month solve
  python scripts/make_milp_ceiling.py            # full: 24 windows x 5 tariffs
"""
from __future__ import annotations
import json
import statistics as st
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyomo.opt import TerminationCondition
from scripts.teacher_alignment_test import _load_df  # reuse loader
from reinforcement.sac_penalty.utils.test_eval import _default_monthly_runs
from reinforcement.sac_penalty.utils.safety import SafetyLayer
from models._physics import Battery, EV
from environment import SmartHomeEnv
from opt import Teacher

TARIFFS = ["tar_flat", "tar_s", "tar_w", "tar_sw", "tar_tou"]
PARAMS = json.load(open(ROOT / "data" / "parameters.json", encoding="utf-8"))
GMIN = float(PARAMS["Grid"]["Pmax_export"])
GMAX = float(PARAMS["Grid"]["Pmax_import"])


def _replay_projected(df, params, run, tariff, teacher, days):
    """Replay teacher actions through the safety projection; return realized env reward,
    plus the raw (pre-projection) native grid violation in kWh for context."""
    start = datetime.strptime(run["date"], "%Y-%m-%d %H:%M:%S")
    env = SmartHomeEnv(df, params, start, days, float(run["soc"]), tariff, track_operation=False)
    dt = float(params["general"]["timestep"]) / 60.0
    safety = SafetyLayer(Battery(params["BESS"], dt), EV(params["EV"], dt), params)
    safety.eval()

    obs, _ = env.reset()
    total_reward = 0.0
    raw_viol_kwh = 0.0
    n = 0
    for t in teacher.model.Ωt:
        raw = torch.as_tensor(np.asarray(teacher.get_actions(t), dtype=np.float32)).unsqueeze(0)
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            proj, _ = safety.project(x, raw)
        action = proj.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        pg = float(info["pgrid"])
        raw_viol_kwh += (max(pg - GMAX, 0.0) + max(-GMIN - pg, 0.0)) * dt
        n += 1
        if terminated or truncated:
            break
    env.close()
    return total_reward, raw_viol_kwh, n


def one(run, tariff, days):
    """Solve + replay one window. Returns (obj, replay_reward, viol, dt) or None if the
    MILP is infeasible (a hard EV departure-SoC constraint the env handles as a soft
    penalty -- these windows are reported and excluded from the ceiling mean)."""
    df = _load_df({**run, "days": int(days)})
    start = datetime.strptime(run["date"], "%Y-%m-%d %H:%M:%S")
    # Clip the window to available data (the env truncates at data end; the MILP would
    # otherwise index past it). Only the December window overshoots by one day.
    ts = float(PARAMS["general"]["timestep"])  # minutes
    last = df.index[-1].to_pydatetime()
    max_days = int(((last + timedelta(minutes=ts)) - start).total_seconds() // 86400)
    days = min(int(days), max_days)
    teacher = Teacher(df, PARAMS, start, int(days), float(run["soc"]), tariff)
    teacher.build()
    solver = pyo.SolverFactory("gurobi")
    teacher.results = solver.solve(teacher.model, tee=False)
    if teacher.results.solver.termination_condition != TerminationCondition.optimal:
        return None
    teacher.get_operation()
    obj = float(pyo.value(teacher.model.objective))          # idealized optimum (cost)
    t0 = time.time()
    rew, viol, _ = _replay_projected(df, PARAMS, run, tariff, teacher, int(days))
    return obj, rew, viol, time.time() - t0


def main():
    if "--test1" in sys.argv:
        r = _default_monthly_runs()[0]
        print(f"[milp] timing one window: {r['name']} ({r['days']}d) tar_sw ...", flush=True)
        obj, rew, viol, dt = one(r, "tar_sw", r["days"])
        print(f"[milp] objective(ceiling)={-obj:.1f}  openloop_replay={rew:.1f}  "
              f"post-proj viol={viol:.2f} kWh  {dt:.0f}s -> 120 windows ~ {120*dt/3600:.1f} h", flush=True)
        return
    runs = _default_monthly_runs()
    t_all = time.time()
    for tariff in TARIFFS:
        objs, replays, infeasible = [], [], []
        for run in runs:
            res = one(run, tariff, run["days"])
            if res is None:
                infeasible.append(run["name"])
                print(f"[milp] {tariff}/{run['name']}: INFEASIBLE (skipped)", flush=True)
                continue
            obj, rew, viol, dt = res
            objs.append(-obj)            # reward-convention ceiling = -cost
            replays.append(rew)
            print(f"[milp] {tariff}/{run['name']}: ceiling={-obj:.1f} openloop={rew:.1f} ({dt:.0f}s)", flush=True)
        mean_ceiling = st.mean(objs) if objs else float("nan")
        out = ROOT / "paper" / "ablation" / "test" / "baselines" / "MILP" / tariff
        out.mkdir(parents=True, exist_ok=True)
        summary = [{"architecture": "MILP", "tariff": tariff, "checkpoint": "milp",
                    "mode": "perfect_foresight", "mean_reward": mean_ceiling,
                    "worst_reward": min(objs) if objs else float("nan"),
                    "months": len(objs), "infeasible_windows": infeasible,
                    "n_infeasible": len(infeasible),
                    "openloop_replay_mean": st.mean(replays) if replays else float("nan"),
                    "openloop_replay_worst": min(replays) if replays else float("nan")}]
        json.dump(summary, open(out / "summary_overall.json", "w", encoding="utf-8"), indent=2)
        print(f"[milp] == {tariff}: ceiling={mean_ceiling:.1f} openloop={st.mean(replays) if replays else float('nan'):.1f} "
              f"(feasible {len(objs)}/{len(runs)}, infeasible={len(infeasible)}) ==", flush=True)
    print(f"[milp] total wall: {(time.time()-t_all)/3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
