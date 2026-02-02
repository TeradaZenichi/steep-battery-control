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

for tariff in ["tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat"]:
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
            action = [
                teacher_operation.loc[teacher_env.sim.step, "PBESS"] / teacher_env.bess.Pmax,
                teacher_operation.loc[teacher_env.sim.step, "PEV"]   / teacher_env.ev.Pmax_c,
                teacher_operation.loc[teacher_env.sim.step, "χPV"],
            ]
            state, reward, terminated, truncated, info = teacher_env.step(action)
            done = terminated or truncated
            teacher_reward += reward

        teacher_env.operation.to_csv(
            folder / f"{run['name']}_env_operation.csv",
            index_label="timestamp",
        )

        print(f"[{tariff}] {run['name']} - Starting actor evaluation...")
        done = False
        actor_reward = 0.0
        while not done:
            state = actor_env._get_observation()
            state = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
            action = actor.predict(state)  # determinístico (avaliação)
            state, reward, terminated, truncated, info = actor_env.step(action)
            done = terminated or truncated
            actor_reward += reward

        actor_env.operation.to_csv(
            folder / f"{run['name']}_actor_env_operation.csv",
            index_label="timestamp",
        )

        summary[run["name"]] = {
            "teacher_reward": float(teacher_reward),
            "actor_reward": float(actor_reward),
            "reward_diff": float(actor_reward - teacher_reward),
            "dataset": run["dataset"],
            "date": run["date"],
            "days": run["days"],
            "soc": run["soc"],
        }

    with open(folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
