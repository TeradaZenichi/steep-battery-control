import pandas as pd
import os

base = "Results/train/ATT_old/2-RL"
tariffs = sorted(os.listdir(base))

for t in tariffs:
    f = os.path.join(base, t, "audit_training.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    print(f"=== {t} ({len(df)} episodes) ===")

    for ep in [0, 10, 25, 50, 100, 150, 200, len(df) - 1]:
        if ep >= len(df):
            continue
        row = df.iloc[ep]
        episode = int(row["episode"])
        lam = row.get("lambda", float("nan"))
        cm = row.get("cost_mean", float("nan"))
        cp = row.get("cost_p95", float("nan"))
        fv = row.get("frac_violation", float("nan"))
        er = row.get("eval_reward_ma", "")
        dl = row.get("dual_enabled", "")
        print(
            f"  ep={episode:3d}  "
            f"lambda={lam:8.4f}  "
            f"cost_mean={cm:.6f}  "
            f"cost_p95={cp:.4f}  "
            f"frac_viol={fv:.4f}  "
            f"eval_ma={er}"
        )

    # Summary
    print(f"  --- Summary ---")
    print(f"  lambda final: {df['lambda'].iloc[-1]:.4f}")
    print(f"  lambda max:   {df['lambda'].max():.4f}")
    print(f"  cost_mean final: {df['cost_mean'].iloc[-1]:.6f}")
    print(f"  cost_mean min:   {df['cost_mean'].min():.6f}")
    print(f"  frac_violation final: {df['frac_violation'].iloc[-1]:.4f}")
    print(f"  frac_violation min:   {df['frac_violation'].min():.4f}")
    print()
