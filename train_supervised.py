from opt import Teacher
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
    
    startdate = "09/01/2000 00:00"
    startdate = pd.to_datetime(startdate, format="%d/%m/%Y %H:%M", dayfirst=True)
    teacher = Teacher(config, dataframe=dataweather, start_date=startdate, days=5)
    teacher.build(start_soc=0.5)
    results = teacher.solve(solver_name="gurobi")
    
    teacher.results_df().to_csv("supervised_results.csv", index=True)
    teacher.results_df().to_excel("supervised_results.xlsx", index=True)
    print("Results saved to results/supervised_results.csv")


    print("Last 10 recorded states:")


if __name__ == "__main__":
    main()