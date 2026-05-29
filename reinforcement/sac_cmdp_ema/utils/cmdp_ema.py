"""EMA-backup variants of the CMDP critic steps.

Isolated from `sac_cmdp/utils/cmdp.py` so the baseline experiments are never
touched. The ONLY difference vs the shared versions: the continuation action
a' ~ π(s') in the Bellman target is sampled from `backup_actor` (a lagged EMA
copy of the policy) instead of the live actor.

Rationale (from Results/Insights v9): the standard SAC target re-evaluates
transitions using the *current* policy's continuation. When the live policy
degrades post-peak, the target value of good transitions collapses with it,
driving the documented late critic drift. A lagged EMA continuation keeps the
target anchored to the better (peak-era) policy longer.

If `backup_actor is None`, behaviour is identical to the shared baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from reinforcement.sac_cmdp.utils.cmdp import _min_q, cost_tensor


def reward_critic_step(critic, target, actor, batch, temperature, opt, grad_clip=True, backup_actor=None):
    obs, act = batch["obs"], batch["act"]
    rew, next_obs = batch["rew"], batch["next_obs"]
    done, gamma_pow = batch["done"], batch["gamma_pow"]
    ba = backup_actor if backup_actor is not None else actor
    with torch.inference_mode():
        next_action, logp_next, _, _ = ba.sample(next_obs, with_mu=False)
        q_next, _, _ = _min_q(target, next_obs, next_action)
        backup = rew + gamma_pow * (1.0 - done) * (q_next - temperature.alpha * logp_next)
    backup = backup.clone()
    q1, q2 = critic(obs, act)
    loss = F.smooth_l1_loss(q1, backup) + F.smooth_l1_loss(q2, backup)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    opt.step()
    return {
        "reward_critic_loss": float(loss.detach()),
        "q_reward_mean": float(torch.min(q1, q2).mean().detach()),
        "reward_backup_mean": float(backup.mean().detach()),
    }


def cost_critic_step(name, critic, target, actor, batch, opt, safety, env_params, cfg, grad_clip=True, backup_actor=None):
    obs, act = batch["obs"], batch["act"]
    next_obs = batch["next_obs"]
    done, gamma_pow = batch["done"], batch["gamma_pow"]
    ba = backup_actor if backup_actor is not None else actor
    with torch.inference_mode():
        c = batch.get(f"{name}_cost")
        if c is None:
            c = cost_tensor(name, obs, act, safety, env_params, cfg)
        next_action, _, _, _ = ba.sample(next_obs, with_mu=False)
        q_next, _, _ = _min_q(target, next_obs, next_action)
        q_next = torch.relu(q_next)
        backup = c + gamma_pow * (1.0 - done) * q_next
    backup = backup.clone()
    q1, q2 = critic(obs, act)
    loss = F.smooth_l1_loss(q1, backup) + F.smooth_l1_loss(q2, backup)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    opt.step()
    return {
        f"{name}_critic_loss": float(loss.detach()),
        f"{name}_cost_mean": float(c.mean().detach()),
        f"{name}_q_cost_mean": float(torch.min(q1, q2).mean().detach()),
        f"{name}_cost_backup_mean": float(backup.mean().detach()),
    }
