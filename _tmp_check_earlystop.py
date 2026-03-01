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

    # Track best eval_reward_ma and when it improved
    best_ma = float("-inf")
    last_improve_ep = 0
    improvements = []

    for i, row in df.iterrows():
        ep = int(row["episode"])
        ma = row.get("best_eval_ma", float("nan"))
        if pd.notna(ma) and ma > best_ma:
            gap = ep - last_improve_ep
            improvements.append((ep, ma, gap))
            best_ma = ma
            last_improve_ep = ep

    print(f"  Total improvements: {len(improvements)}")
    print(f"  Last improvement at ep: {last_improve_ep}")
    print(f"  Best eval_ma: {best_ma:.2f}")
    print()
    print(f"  {'ep':>5s}  {'best_eval_ma':>14s}  {'gap_since_prev':>14s}")
    for ep, ma, gap in improvements:
        print(f"  {ep:5d}  {ma:14.2f}  {gap:14d}")

    # Gaps between improvements
    gaps = [g for _, _, g in improvements]
    if gaps:
        print()
        print(f"  Max gap between improvements: {max(gaps)} episodes")
        print(f"  Mean gap: {sum(gaps)/len(gaps):.1f}")
        print(f"  Gaps > 20: {[g for g in gaps if g > 20]}")
        print(f"  Gaps > 30: {[g for g in gaps if g > 30]}")
        print(f"  Gaps > 50: {[g for g in gaps if g > 50]}")

    # no_improve_evals column
    if "no_improve_evals" in df.columns:
        max_no_improve = df["no_improve_evals"].max()
        print(f"  Max no_improve_evals: {max_no_improve}")

    print()
