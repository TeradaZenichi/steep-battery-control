from env.environment import SmartHomeEnv
import pandas as pd
import json

PARAM = "data/parameters.json"
DATA = "data/Simulation_CY_Cur_HP__PV5000-HB5000.csv"


with open(PARAM, 'r') as f:
    config = json.load(f)

dataweather = pd.read_csv(DATA)


home = SmartHomeEnv(config, dataframe=dataweather)


a = 1