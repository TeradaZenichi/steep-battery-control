import pandas as pd

df = pd.read_csv("Results/train/ATT_old/2-RL/tar_s/audit_training.csv")

dlam = df["lambda"].diff()
print("Lambda growth per episode (first 10):")
for i in range(min(10, len(dlam))):
    lam = df["lambda"].iloc[i]
    delta = f"{dlam.iloc[i]:.6f}" if i > 0 else "N/A"
    print(f"  ep {i}: lambda={lam:.6f}  delta={delta}")

print()
print(f"Mean lambda delta: {dlam.dropna().mean():.6f}")
print(f"Max lambda delta:  {dlam.dropna().max():.6f}")
print()
print(f"Final lambda: {df['lambda'].iloc[-1]:.4f}")
print(f"Final cost_mean: {df['cost_mean'].iloc[-1]:.6f}")
print(f"cost_limit in old config: 0.0001")
print(f"cost_mean is {df['cost_mean'].iloc[-1] / 0.0001:.0f}x above cost_limit")
print()
print("=== Implication ===")
print("cost_mean never drops below cost_limit (0.0001).")
print("Lambda keeps growing because cost > limit always.")
print("With lambda_lr=5e-4 instead of 1e-3, lambda grows ~2x slower.")
print("This means lambda would be ~1.75 at ep 150 instead of ~3.5.")
print("Lower lambda = less penalty pressure = higher cost tolerance.")
print()

# Check what happens to reward at different lambda levels
for lam_thresh in [1.0, 2.0, 3.0]:
    rows = df[df["lambda"] >= lam_thresh]
    if len(rows) > 0:
        ep = rows.iloc[0]["episode"]
        er = rows.iloc[0].get("eval_reward_ma", "N/A")
        cm = rows.iloc[0]["cost_mean"]
        print(f"  lambda >= {lam_thresh:.0f} at ep {ep:.0f}: cost_mean={cm:.6f}, eval_ma={er}")
