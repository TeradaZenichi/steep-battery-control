"""SAC-CMDP update pieces: one reward critic and optional cost critics."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Temperature(nn.Module):
    def __init__(self, init_alpha=0.1):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(init_alpha), dtype=torch.float32)))

    @property
    def alpha(self):
        return self.log_alpha.exp()


class CostLambda(nn.Module):
    def __init__(self, init_lambda=0.1, budget=0.0, max_lambda=None):
        super().__init__()
        init_lambda = max(float(init_lambda), 1e-8)
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(init_lambda, dtype=torch.float32)))
        self.budget = float(budget)
        self.max_lambda = None if max_lambda is None else float(max_lambda)

    @property
    def value(self):
        val = self.log_lambda.exp()
        return val if self.max_lambda is None else val.clamp(max=self.max_lambda)

    def loss(self, q_cost):
        return -(self.log_lambda * (q_cost.detach() - self.budget)).mean()

    def clamp_(self):
        if self.max_lambda is None:
            return
        with torch.no_grad():
            self.log_lambda.clamp_(max=float(torch.log(torch.tensor(max(self.max_lambda, 1e-8)))))


def soft_update(net, target, tau):
    with torch.no_grad():
        for p, pt in zip(net.parameters(), target.parameters()):
            pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def _min_q(critic, obs, act):
    q1, q2 = critic(obs, act)
    return torch.min(q1, q2), q1, q2


def _last(obs):
    return obs if obs.dim() == 2 else obs[:, -1, :]


def ev_deficit_cost(obs, target_soc=0.5, floor=0.05):
    """Dense EV departure-risk cost from the current observation.

    Uses the current EV SoC, controllability flag, and normalized time until
    departure. In the current env layout: SoC EV=13, EV on=14, until dep=26.
    """
    s = _last(obs)
    soc = s[..., 13:14]
    ev_on = s[..., 14:15]
    deficit = torch.clamp(float(target_soc) - soc, min=0.0)
    if s.shape[-1] > 26:
        remain = torch.clamp(s[..., 26:27], min=float(floor))
    else:
        remain = torch.full_like(soc, 0.5)
    return ev_on * deficit / remain


def grid_cost(obs, act, safety):
    if safety is None:
        return torch.zeros((act.shape[0], 1), device=act.device, dtype=act.dtype)
    _, viol = safety.project(obs, act)
    return viol


def pv_cost(obs, act, params):
    """Immediate PV curtailment energy in kWh; optional diagnostic cost."""
    s = _last(obs)
    ppv = torch.clamp(s[..., 11:12] * float(params["general"]["Pnorm"]), min=0.0)
    dt = float(params["general"]["timestep"]) / 60.0
    return torch.clamp(act[..., 2:3], min=0.0) * ppv * dt


def cost_tensor(name, obs, act, safety, env_params, cfg):
    if name == "ev":
        return ev_deficit_cost(
            obs,
            target_soc=float(cfg.get("target_soc", env_params["EV"].get("soc_min", 0.5))),
            floor=float(cfg.get("t_remain_floor", 0.05)),
        )
    if name == "grid":
        return grid_cost(obs, act, safety)
    if name == "pv":
        return pv_cost(obs, act, env_params)
    raise KeyError(f"Unknown CMDP cost: {name}")


def reward_critic_step(critic, target, actor, batch, temperature, opt, grad_clip=True):
    obs, act = batch["obs"], batch["act"]
    rew, next_obs = batch["rew"], batch["next_obs"]
    done, gamma_pow = batch["done"], batch["gamma_pow"]
    with torch.inference_mode():
        next_action, logp_next, _, _ = actor.sample(next_obs, with_mu=False)
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


def cost_critic_step(name, critic, target, actor, batch, opt, safety, env_params, cfg, grad_clip=True):
    obs, act = batch["obs"], batch["act"]
    next_obs = batch["next_obs"]
    done, gamma_pow = batch["done"], batch["gamma_pow"]
    with torch.inference_mode():
        c = batch.get(f"{name}_cost")
        if c is None:
            c = cost_tensor(name, obs, act, safety, env_params, cfg)
        next_action, _, _, _ = actor.sample(next_obs, with_mu=False)
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


def actor_step(actor, reward_critic, cost_critics, lambdas, batch, temperature, opt, grad_clip=True):
    obs = batch["obs"]
    action_pi, logp_pi, _, _ = actor.sample(obs, with_mu=False)
    q_reward, _, _ = _min_q(reward_critic, obs, action_pi)
    loss_terms = temperature.alpha.detach() * logp_pi - q_reward
    info = {
        "logp": logp_pi.detach(),
        "logp_mean": float(logp_pi.mean().detach()),
        "q_reward_pi_mean": float(q_reward.mean().detach()),
    }
    q_costs = {}
    for name, critic in cost_critics.items():
        q_cost, _, _ = _min_q(critic, obs, action_pi)
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


def alpha_step(temperature, logp, target_entropy, opt, log_alpha_min=-3.0):
    loss = -(temperature.log_alpha * (logp + target_entropy)).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():
        temperature.log_alpha.clamp_(min=log_alpha_min)
    return {"alpha_loss": float(loss.detach()), "alpha": float(temperature.alpha.detach())}


def lambda_step(lam, q_cost, opt):
    loss = lam.loss(q_cost)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    lam.clamp_()
    return {"lambda_loss": float(loss.detach()), "lambda_value": float(lam.value.detach())}
