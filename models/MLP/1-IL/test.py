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

df = pd.read_csv(
    "data/Simulation_WY_Cur_HP__PV5000-HB5000.csv",
    sep=";",
    parse_dates=["timestamp"],
    dayfirst=True,               # datas no formato dd/mm/aaaa
    index_col="timestamp",
)
with open("data/parameters.json", encoding="utf-8") as f:
	par = json.load(f)

with open("models/MLP/1-IL/config.json") as f:
    test_cfg = json.load(f)

with open("models/MLP/model.json") as f:
    model_cfg = json.load(f)

torch.manual_seed(test_cfg["seed"])
np.random.seed(test_cfg["seed"])

for tariff in ["tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat"]:
    folder = PROJECT_ROOT / "Results" / "test" / "MLP" / "1-IL" / tariff
    folder.mkdir(parents=True, exist_ok=True)
    summary = {}
    for run in test_cfg["test"]:
        start = datetime.strptime(run["date"], "%Y-%m-%d %H:%M:%S")
        days = run["days"]
        BESS_SoC = run["soc"]

        actor = load_actor(model_cfg["actor"], weights_path=f"Results/train/MLP/1-IL/{tariff}/best.pth")
        teacher = Teacher(df, par, start, days, BESS_SoC, tariff)
        teacher.build()
        teacher.solve()

        teacher_operation = teacher.get_operation()
        teacher_operation.to_csv(f"Results/test/MLP/1-IL/{tariff}/{run['name']}_teacher_operation.csv", index_label="timestamp")

        teacher_env = SmartHomeEnv(df, par, start, days, BESS_SoC, tariff)
        actor_env = SmartHomeEnv(df, par, start, days, BESS_SoC, tariff)

        print("Starting teacher evaluation...")
        done = False
        teacher_reward = 0
        while not done:
            action = [
                teacher_operation.loc[teacher_env.sim.step, "PBESS"] / teacher_env.bess.Pmax,
                teacher_operation.loc[teacher_env.sim.step, "PEV"] / teacher_env.ev.Pmax_c,
                teacher_operation.loc[teacher_env.sim.step, "χPV"],
            ]
            state, reward, terminated, truncated, info = teacher_env.step(action)
            actor_action = actor.predict(state)
            done = terminated or truncated
            teacher_reward += reward

        teacher_env.operation.to_csv(f"Results/test/MLP/1-IL/{tariff}/{run['name']}_env_operation.csv", index_label="timestamp")

        print("Starting actor evaluation...")
        done = False
        actor_reward = 0
        while not done:
            state = actor_env._get_observation()
            action = actor.predict(state)
            state, reward, terminated, truncated, info = actor_env.step(action)
            done = terminated or truncated
            actor_reward += reward

        actor_env.operation.to_csv(f"Results/test/MLP/1-IL/{tariff}/{run['name']}_actor_env_operation.csv", index_label="timestamp")
        summary[run["name"]] = {
            "teacher_reward": teacher_reward,
            "actor_reward": actor_reward,
            "reward_diff": actor_reward - teacher_reward,
        }
    with open(folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


