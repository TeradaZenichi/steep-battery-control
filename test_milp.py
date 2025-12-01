from env.environment import SmartHomeEnv
from opt import Teacher
import pandas as pd
import numpy as np
import json


PARAM = "data/parameters.json"
DATA = "data/Simulation_CY_Cur_HP__PV5000-HB5000.csv"
NDAYS = 5

with open(PARAM, 'r') as f:
    config = json.load(f)

dataweather = pd.read_csv(DATA, sep=';')
state_mask = None

startdate = "08/01/2000 00:00"
startdate = pd.to_datetime(startdate, format="%d/%m/%Y %H:%M", dayfirst=True)
teacher = Teacher(config, dataframe=dataweather, start_date=startdate, days=NDAYS)
teacher.build(start_soc=0.5)
results = teacher.solve(solver_name="gurobi")

# Testing MILP results in environment

home = SmartHomeEnv(config, dataframe=dataweather, days=NDAYS, state_mask=None, start_date=startdate)

state, info = home.reset()
home.bess.reset(soc_init=0.5)


results = teacher.results_df()
results.to_csv("milp_planner_results.csv")

while not home.done:
    action = results.loc[home.sim.current_datetime, ['PBESS', 'Pev']].values
    action = [action[0], action[1], 0]  # Convert to list to append
    state, reward, done, info = home.step(action)
    # print(f"Step: {home.sim.current_step}, Action: {action}, Reward: {reward:.4f}")

env_results_df = home.build_operation_dataframe()

print("\nEnvironment replay results (first 5 rows):")
print(env_results_df.head())
env_results_df.to_csv("milp_env_replay_results.csv")


