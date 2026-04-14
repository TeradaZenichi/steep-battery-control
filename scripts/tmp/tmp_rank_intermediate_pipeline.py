import pandas as pd
from pathlib import Path

p = Path("Results/analysis/intermediate_pipeline_v2_rl_summary.csv")
df = pd.read_csv(p)

print("=== Ranking por test_actor_reward (maior é melhor) ===")
for _, r in df.sort_values("test_actor_reward", ascending=False).iterrows():
    print(
        f"{r['model']}: reward={r['test_actor_reward']:.3f}, chi={r['test_mean_chiPV']:.3f}, "
        f"pv_curtail_kwh={r['test_pv_curtail_kwh']:.3f}, frac_violation={r['train_frac_violation']:.3f}, "
        f"lambda={r['train_lambda']:.6f}"
    )

print("\n=== Ranking por pv_curtail_kwh (menor é melhor) ===")
for _, r in df.sort_values("test_pv_curtail_kwh", ascending=True).iterrows():
    print(
        f"{r['model']}: pv_curtail_kwh={r['test_pv_curtail_kwh']:.3f}, reward={r['test_actor_reward']:.3f}, "
        f"chi={r['test_mean_chiPV']:.3f}, bess_pos={r['test_bess_pos_kwh']:.3f}, bess_neg={r['test_bess_neg_kwh']:.3f}"
    )
