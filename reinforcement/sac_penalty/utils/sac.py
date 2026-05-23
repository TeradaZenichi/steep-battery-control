"""SAC update pieces used by the standard GRU trainer."""
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


def soft_update(net, target, tau):
    with torch.no_grad():
        for p, pt in zip(net.parameters(), target.parameters()):
            pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def alpha_step(temperature, logp, target_entropy, opt, log_alpha_min=-3.0):
    """Auto-tune alpha with a small floor to avoid entropy collapse."""
    loss = -(temperature.log_alpha * (logp + target_entropy)).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():
        temperature.log_alpha.clamp_(min=log_alpha_min)
    return {"alpha_loss": float(loss.detach()), "alpha": float(temperature.alpha.detach())}


def critic_step_ens(ensemble, actor, batch, temperature, opt, grad_clip=True, m=2, safety=None):
    obs, act = batch["obs"], batch["act"]
    rew, next_obs = batch["rew"], batch["next_obs"]
    done, gamma_pow = batch["done"], batch["gamma_pow"]
    with torch.inference_mode():
        next_action, logp_next, _, _ = actor.sample(next_obs, with_mu=False)
        if safety is not None:
            next_action, _ = safety.project(next_obs, next_action)
        q_next = ensemble.target_q(next_obs, next_action, m=m)
        backup = rew + gamma_pow * (1.0 - done) * (q_next - temperature.alpha * logp_next)
    backup = backup.clone()
    q_all = ensemble.all_q(obs, act)
    loss = sum(F.smooth_l1_loss(q_all[i], backup) for i in range(q_all.shape[0]))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(ensemble.parameters(), max_norm=1.0)
    opt.step()
    return {
        "critic_loss": float(loss.detach()),
        "critic_main_loss": float(loss.detach()),
        "q1_mean": float(q_all[0].mean().detach()),
        "q2_mean": float(q_all[-1].mean().detach()),
        "q_mean": float(q_all.mean().detach()),
        "backup_mean": float(backup.mean().detach()),
    }


def actor_step_ens(actor, ensemble, safety, batch, temperature, lam, opt,
                   kl_anchor=None, grad_clip=True, project_q=False,
                   proj_penalty_coef=0.0):
    obs = batch["obs"]
    action_pi, logp_pi, _, _ = actor.sample(obs, with_mu=False)
    action_safe, viol = safety.project(obs, action_pi)
    q_action = action_safe if project_q else action_pi
    q_all = ensemble.all_q(obs, q_action)
    q_mean = q_all.mean(dim=0)
    q_std = q_all.std(dim=0, unbiased=False)
    loss = (temperature.alpha.detach() * logp_pi - q_mean + lam.value.detach() * viol).mean()

    if proj_penalty_coef > 0.0:
        proj_delta = torch.abs(action_safe - action_pi).mean(dim=-1, keepdim=True)
        loss = loss + float(proj_penalty_coef) * proj_delta.mean()
    else:
        proj_delta = torch.zeros_like(viol)

    info = {}
    if kl_anchor is not None and kl_anchor.ready():
        kl_val = kl_anchor.kl(obs, actor)
        loss = loss + kl_anchor.beta * kl_val
        info["kl"] = float(kl_val.detach())
        info["kl_beta"] = float(kl_anchor.beta)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    opt.step()
    info.update({
        "actor_loss": float(loss.detach()),
        "logp_mean": float(logp_pi.mean().detach()),
        "q_pi_mean": float(q_mean.mean().detach()),
        "q_pi_std_mean": float(q_std.mean().detach()),
        "proj_delta_mean": float(proj_delta.mean().detach()),
        "viol_mean": float(viol.mean().detach()),
        "logp": logp_pi.detach(),
        "viol": viol.detach(),
    })
    return info
