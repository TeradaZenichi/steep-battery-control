import pandas as pd
from pathlib import Path

files = [
    Path('data/Simulation_CY_Cur_HP__PV5000-HB5000.csv'),
    Path('data/Simulation_WY_Cur_HP__PV5000-HB5000.csv'),
]

for path in files:
    df = pd.read_csv(path, sep=';', parse_dates=['timestamp'], dayfirst=True).sort_values('timestamp')
    dt_h = float(df['timestamp'].diff().dropna().dt.total_seconds().mode().iloc[0] / 3600.0)
    load_kwh_step = pd.to_numeric(df['electricity_demand_rate_W'], errors='coerce').fillna(0.0) / 1000.0 * dt_h

    print('\n' + path.name)
    for tariff in ['tar_flat', 'tar_tou', 'tar_s', 'tar_w', 'tar_sw']:
        series = pd.to_numeric(df[tariff], errors='coerce').dropna()
        monthly = (load_kwh_step * series).groupby(df['timestamp'].dt.to_period('M')).sum()
        print(
            f"{tariff}: min={series.min():.4f} mean={series.mean():.4f} max={series.max():.4f} "
            f"| month_mean_eur={monthly.mean():.2f}"
        )
