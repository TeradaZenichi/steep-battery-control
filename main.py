from environment import SmartHomeEnv
from datetime import datetime
from opt import Teacher
import pandas as pd
import json


df = pd.read_csv(
    "data/Simulation_WY_Cur_HP__PV5000-HB5000.csv",
    sep=";",
    parse_dates=["timestamp"],
    dayfirst=True,               # datas no formato dd/mm/aaaa
    index_col="timestamp",
)
with open("data/parameters.json", encoding="utf-8") as f:
	par = json.load(f)
	
start = datetime.strptime("2000-09-01 00:00:00", "%Y-%m-%d %H:%M:%S")
days = 10
tariff = "tar_s"
BESS_SoC = 0.5


# teacher = Teacher(df, par, start, days, BESS_SoC, tariff)

teacher = Teacher(df, par, start, days, BESS_SoC, tariff)
teacher.build()
teacher.solve()

teacher_operation = teacher.get_operation()
teacher_operation.to_csv("teacher_operation.csv", index_label="timestamp")

env = SmartHomeEnv(df, par, start, days, BESS_SoC, tariff)
done = False

while not done:
    print([
        teacher_operation.loc[env.sim.step, "PBESS"],
        teacher_operation.loc[env.sim.step, "PEV"],
        teacher_operation.loc[env.sim.step, "χPV"],
    ])
    action = [
        teacher_operation.loc[env.sim.step, "PBESS"] / env.bess.Pmax,
        teacher_operation.loc[env.sim.step, "PEV"] / env.ev.Pmax_c,
        teacher_operation.loc[env.sim.step, "χPV"],
    ]
    print(action)
    state, reward, terminated, truncated, info = env.step(action)
    print(f"State: {state}, Reward: {reward}")
    done = terminated or truncated

# save with index as timestamp
env.operation.to_csv("env_operation.csv", index_label="timestamp")

x_train, y_train = teacher.get_training_data()

a = 1


