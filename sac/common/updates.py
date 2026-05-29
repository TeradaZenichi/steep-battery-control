"""Critic update steps that accept an explicit `backup_actor`.

Identical math to reinforcement.sac_cmdp.utils.cmdp, but the action used in the
Bellman backup comes from `backup_actor` (the live actor by default, or a lagged
EMA actor when the `ema_backup` switch is on). Actor/alpha/lambda steps are
reused unchanged from the sac_cmdp stack.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _min_q(critic, obs, act):
    q1, q2 = critic(obs, act)
    return torch.min(q1, q2), q1, q2


def reward_critic_step(critic, target, backup_actor, batch, temperature, opt, grad_clip=True):
    obs, act = batch["obs"], batch["act"]
    rew, next_obs = batch["rew"], batch["next_obs"]
    done, gamma_pow = batch["done"], batch["gamma_pow"]
    with torch.inference_mode():
        next_action, logp_next, _, _ = backup_actor.sample(next_obs, with_mu=False)
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


def cost_critic_step(name, critic, target, backup_actor, batch, opt, grad_clip=True):
    obs, act = batch["obs"], batch["act"]
    next_obs = batch["next_obs"]
    done, gamma_pow = batch["done"], batch["gamma_pow"]
    with torch.inference_mode():
        c = batch.get(f"{name}_cost")
        next_action, _, _, _ = backup_actor.sample(next_obs, with_mu=False)
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
    }
