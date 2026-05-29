"""SAC-CMDP + EMA-backup stack.

Re-exports the shared sac_cmdp utilities unchanged, then overrides only the two
critic-step functions with EMA-backup-aware versions (see cmdp_ema.py). This
keeps the baseline `sac_cmdp/utils` completely untouched.
"""
from reinforcement.sac_cmdp.utils import (
    SafetyLayer,
    Hyperparameters,
    ReplayBuffer,
    Temperature,
    CostLambda,
    soft_update,
    actor_step,
    alpha_step,
    lambda_step,
    cost_tensor,
    EpisodeGen,
    Audit,
    AsyncEval,
    EvalRunner,
    summarize,
    collect_episode,
    collect_streams,
    EMA,
    run_test,
)
from .cmdp_ema import reward_critic_step, cost_critic_step

__all__ = [
    "SafetyLayer",
    "Hyperparameters", "ReplayBuffer",
    "Temperature", "CostLambda", "soft_update",
    "reward_critic_step", "cost_critic_step", "actor_step",
    "alpha_step", "lambda_step",
    "cost_tensor",
    "EpisodeGen", "Audit", "AsyncEval", "EvalRunner", "summarize",
    "collect_episode", "collect_streams",
    "EMA",
    "run_test",
]
