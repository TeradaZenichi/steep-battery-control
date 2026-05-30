"""SAC update steps with ensemble-aware critic support.

The Critic class returns a tuple of N q-values (twin = N=2; REDQ-light = N=4).
For target, optionally take min over a random subset of M (when m_target<N).
For training, all N critics are trained against the same scalar target.
Backward compatible with twin Q (N=2, m_target=None → identical math).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stack_q(critic, obs, act):
    """Stack the N q-outputs of the (twin or ensemble) critic."""
    return torch.stack(list(critic(obs, act)), dim=0)   # [N, batch, 1]


def _min_subset(q_stack, m=None):
    """min over a random subset of M of the N q estimates (m=None → min over all)."""
    n = q_stack.shape[0]
    if m and m < n:
        idx = torch.randperm(n, device=q_stack.device)[:m]
        q_stack = q_stack[idx]
    return q_stack.min(dim=0).values


def reward_critic_step(critic, target, backup_actor, batch, temperature, opt, grad_clip=True, m_target=None):
    obs, act = batch["obs"], batch["act"]
    rew, next_obs = batch["rew"], batch["next_obs"]
    done, gamma_pow = batch["done"], batch["gamma_pow"]
    with torch.no_grad():
        next_action, logp_next, _, _ = backup_actor.sample(next_obs, with_mu=False)
        q_next = _min_subset(_stack_q(target, next_obs, next_action), m=m_target)
        backup = rew + gamma_pow * (1.0 - done) * (q_next - temperature.alpha * logp_next)
    backup = backup.clone()
    q_all = critic(obs, act)
    loss = sum(F.smooth_l1_loss(q, backup) for q in q_all)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    opt.step()
    q_stack_train = torch.stack(list(q_all), dim=0)
    return {
        "reward_critic_loss": float(loss.detach()),
        "q_reward_mean": float(q_stack_train.min(dim=0).values.mean().detach()),
        "reward_backup_mean": float(backup.mean().detach()),
    }


def cost_critic_step(name, critic, target, backup_actor, batch, opt, grad_clip=True, m_target=None):
    obs, act = batch["obs"], batch["act"]
    next_obs = batch["next_obs"]
    done, gamma_pow = batch["done"], batch["gamma_pow"]
    with torch.no_grad():
        c = batch.get(f"{name}_cost")
        next_action, _, _, _ = backup_actor.sample(next_obs, with_mu=False)
        q_next = _min_subset(_stack_q(target, next_obs, next_action), m=m_target)
        q_next = torch.relu(q_next)
        backup = c + gamma_pow * (1.0 - done) * q_next
    backup = backup.clone()
    q_all = critic(obs, act)
    loss = sum(F.smooth_l1_loss(q, backup) for q in q_all)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    opt.step()
    q_stack_train = torch.stack(list(q_all), dim=0)
    return {
        f"{name}_critic_loss": float(loss.detach()),
        f"{name}_cost_mean": float(c.mean().detach()),
        f"{name}_q_cost_mean": float(q_stack_train.min(dim=0).values.mean().detach()),
    }


def actor_step(actor, reward_critic, cost_critics, lambdas, batch, temperature, opt, grad_clip=True, m_actor=None):
    """Ensemble-aware actor step (replaces cmdp.actor_step which assumes twin Q).

    With m_actor=None: min over all N reward critic outputs (and all M of each cost critic).
    With m_actor<N: min over a random subset (REDQ-style).
    """
    obs = batch["obs"]
    action_pi, logp_pi, _, _ = actor.sample(obs, with_mu=False)
    q_reward = _min_subset(_stack_q(reward_critic, obs, action_pi), m=m_actor)
    loss_terms = temperature.alpha.detach() * logp_pi - q_reward
    info = {
        "logp": logp_pi.detach(),
        "logp_mean": float(logp_pi.mean().detach()),
        "q_reward_pi_mean": float(q_reward.mean().detach()),
    }
    q_costs = {}
    for name, critic in cost_critics.items():
        q_cost = _min_subset(_stack_q(critic, obs, action_pi), m=m_actor)
        q_cost_pos = torch.relu(q_cost)
        q_costs[name] = q_cost_pos
        loss_terms = loss_terms + lambdas[name].value.detach() * q_cost_pos
        info[f"{name}_q_cost_pi_mean"] = float(q_cost.mean().detach())
        info[f"{name}_q_cost_pi_pos_mean"] = float(q_cost_pos.mean().detach())
        info[f"{name}_lambda_value"] = float(lambdas[name].value.detach())
    loss = loss_terms.mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    opt.step()
    info["actor_loss"] = float(loss.detach())
    info["q_costs"] = {k: v.detach() for k, v in q_costs.items()}
    return info
