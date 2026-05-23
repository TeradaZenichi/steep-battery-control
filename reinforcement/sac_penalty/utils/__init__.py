"""Penalty-based SAC stack (KL anchor + phases). Physics primitives live
under models/_physics.py — import from there if you need Battery/EV."""
from .safety import SafetyLayer
from .hyperparams import Hyperparameters
from .replay import ReplayBuffer
from .sac import Temperature, soft_update, alpha_step, critic_step_ens, actor_step_ens
from .episode import EpisodeGen
from .audit import Audit
from .eval import AsyncEval, EvalRunner, summarize
from .rollout import collect_episode, collect_streams
from .ema import EMA
from .early_stop import EarlyStop
from .critic_ensemble import CriticEnsemble
from .lagrangian import PIDLambda
from .anchor import KLAnchor

__all__ = [
    "SafetyLayer",
    "Hyperparameters", "ReplayBuffer",
    "Temperature", "soft_update", "alpha_step",
    "critic_step_ens", "actor_step_ens",
    "EpisodeGen", "Audit", "AsyncEval", "EvalRunner", "summarize",
    "collect_episode", "collect_streams",
    "EMA", "EarlyStop", "CriticEnsemble", "PIDLambda", "KLAnchor",
]
