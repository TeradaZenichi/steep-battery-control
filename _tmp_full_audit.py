import pandas as pd
import numpy as np
import os

base = "Results/train/ATT_old/2-RL"
tariffs = sorted(os.listdir(base))

for t in tariffs:
    f = os.path.join(base, t, "audit_training.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    print(f"{'='*60}")
    print(f"=== {t} ({len(df)} episodes) ===")
    print(f"{'='*60}")

    # 1. Steps per episode & buffer usage
    steps_per_ep = df["steps"].diff().dropna()
    print(f"\n[STEPS / BUFFER]")
    print(f"  steps/ep: {steps_per_ep.mean():.0f} (constant={steps_per_ep.std() < 1})")
    print(f"  total steps at end: {df['steps'].iloc[-1]}")
    print(f"  buffer_size at end: {df['buffer_size'].iloc[-1]}")
    print(f"  buffer fill ratio: {df['buffer_size'].iloc[-1] / 200000:.2f} (of 200K)")

    # 2. Alpha (entropy temperature)
    print(f"\n[ALPHA (entropy temp)]")
    print(f"  start: {df['alpha'].iloc[0]:.4f}")
    print(f"  end:   {df['alpha'].iloc[-1]:.4f}")
    print(f"  min:   {df['alpha'].min():.4f}")
    print(f"  Decay ratio: {df['alpha'].iloc[-1] / df['alpha'].iloc[0]:.2f}x")

    # 3. logp (entropy proxy)
    print(f"\n[ENTROPY (logp_mean)]")
    print(f"  start: {df['logp_mean'].iloc[0]:.2f}")
    print(f"  end:   {df['logp_mean'].iloc[-1]:.2f}")
    print(f"  target_entropy in config: -1.5")
    print(f"  logp converges toward target? {'YES' if abs(df['logp_mean'].iloc[-1] - (-1.5)) < abs(df['logp_mean'].iloc[0] - (-1.5)) else 'NO'}")

    # 4. Critic loss
    print(f"\n[CRITIC LOSS]")
    print(f"  start: {df['critic_loss'].iloc[0]:.2f}")
    print(f"  end:   {df['critic_loss'].iloc[-1]:.2f}")
    print(f"  mean:  {df['critic_loss'].mean():.2f}")
    print(f"  std:   {df['critic_loss'].std():.2f}")

    # 5. Q values
    print(f"\n[Q VALUES]")
    print(f"  q1_mean start: {df['q1_mean'].iloc[0]:.2f}, end: {df['q1_mean'].iloc[-1]:.2f}")
    print(f"  backup_abs_max start: {df['backup_abs_max'].iloc[0]:.1f}, end: {df['backup_abs_max'].iloc[-1]:.1f}, max: {df['backup_abs_max'].max():.1f}")

    # 6. Actor loss components
    print(f"\n[ACTOR LOSS COMPONENTS]")
    print(f"  actor_loss end:         {df['actor_loss'].iloc[-1]:.3f}")
    print(f"  actor_term_entropy end: {df['actor_term_entropy'].iloc[-1]:.3f}")
    print(f"  actor_term_q end:       {df['actor_term_q'].iloc[-1]:.3f}")
    print(f"  actor_term_dual end:    {df['actor_term_dual'].iloc[-1]:.3f}")
    ratio_dual_q = abs(df['actor_term_dual'].iloc[-1]) / (abs(df['actor_term_q'].iloc[-1]) + 1e-8)
    print(f"  |dual/q| ratio at end:  {ratio_dual_q:.3f}")
    if ratio_dual_q > 0.5:
        print(f"  WARNING: dual term is {ratio_dual_q:.1f}x of q term - may dominate actor")

    # 7. Cost & violation trajectory
    print(f"\n[COST / VIOLATION]")
    print(f"  cost_mean: {df['cost_mean'].iloc[0]:.4f} -> {df['cost_mean'].iloc[-1]:.4f}")
    print(f"  frac_violation: {df['frac_violation'].iloc[0]:.4f} -> {df['frac_violation'].iloc[-1]:.4f}")
    print(f"  frac_violation increases over time? {df['frac_violation'].iloc[-1] > df['frac_violation'].iloc[0]}")

    # 8. n_updates per episode
    updates_per_ep = df["n_updates"].diff().dropna()
    print(f"\n[UPDATES]")
    print(f"  updates/ep: {updates_per_ep.mean():.0f}")
    print(f"  update_ratio (updates/steps): {updates_per_ep.mean() / steps_per_ep.mean():.3f}")

    # 9. Reward trajectory
    print(f"\n[REWARD]")
    print(f"  train_reward start: {df['train_reward_total'].iloc[0]:.1f}")
    print(f"  train_reward end:   {df['train_reward_total'].iloc[-1]:.1f}")
    print(f"  best_train_reward:  {df['best_train_reward'].iloc[-1]:.1f}")
    eval_ma = df["eval_reward_ma"].dropna()
    if len(eval_ma) > 0:
        print(f"  best_eval_ma:       {df['best_eval_ma'].iloc[-1]:.1f}")

    # 10. warmup assessment
    warmup_eps = 10  # from config
    if len(df) > warmup_eps:
        pre = df.iloc[:warmup_eps]["train_reward_total"].mean()
        post = df.iloc[warmup_eps:warmup_eps+10]["train_reward_total"].mean()
        print(f"\n[WARMUP]")
        print(f"  mean reward ep 0-9 (warmup): {pre:.1f}")
        print(f"  mean reward ep 10-19 (post):  {post:.1f}")
        print(f"  improvement after warmup:     {post - pre:.1f}")

    print()
