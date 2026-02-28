import pandas as pd

files = [
    "data/Simulation_CY_Fut_HP__PV5000-HB5000.csv",
    "data/Simulation_WY_Fut_HP__PV5000-HB5000.csv",
]

for f in files:
    df = pd.read_csv(f, sep=";")
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    ts = df["timestamp"].sort_values().reset_index(drop=True)
    print(f"=== {f} ===")
    print(f"  Range: {ts.iloc[0]} .. {ts.iloc[-1]}")
    print(f"  Rows: {len(ts)}")

    diff = ts.diff().dropna()
    expected = pd.Timedelta(minutes=5)
    gaps = diff[diff != expected]
    if len(gaps) > 0:
        print(f"  Gaps (non-5min): {len(gaps)}")
        for i, (idx, val) in enumerate(gaps.items()):
            if i >= 10:
                break
            prev = ts.iloc[idx - 1].strftime("%Y-%m-%d %H:%M")
            curr = ts.iloc[idx].strftime("%Y-%m-%d %H:%M")
            print(f"    row {idx}: {prev} -> {curr} (delta={val})")
    else:
        print("  No gaps (all 5-min intervals)")

    for m in range(1, 13):
        month_ts = ts[ts.dt.month == m]
        if len(month_ts) > 0:
            first = month_ts.iloc[0].strftime("%Y-%m-%d %H:%M")
            last = month_ts.iloc[-1].strftime("%Y-%m-%d %H:%M")
            last_day = month_ts.iloc[-1].day
            print(f"    Month {m:02d}: {first} .. {last} ({len(month_ts)} rows, last_day={last_day})")
        else:
            print(f"    Month {m:02d}: MISSING")
    print()
