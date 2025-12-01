from env.environment import SmartHomeEnv
from env.mask import get_state_mask
import pandas as pd
import json

PARAM = "data/parameters.json"
DATA = "data/Simulation_CY_Cur_HP__PV5000-HB5000.csv"


def main() -> None:
    with open(PARAM, 'r') as f:
        config = json.load(f)

    dataweather = pd.read_csv(DATA, sep=';')

    state_mask = get_state_mask()
    state_mask = None

    dataweather = pd.read_csv(DATA, sep=';', parse_dates=['timestamp']).set_index('timestamp').sort_index()    
    home = SmartHomeEnv(config, dataframe=dataweather, days=100, state_mask=state_mask)
    

    state, info = home.reset()
    
    while not home.done:
        action = home.action_space.sample()
        state, reward, done, info = home.step(action)
        print(f"Step: {home.sim.current_step}, Action: {action}, Reward: {reward:.4f}")

    hist = home.sim.get_state_history(10)
    print("Last 10 recorded states:")
    print(hist)
if __name__ == "__main__":
    main()