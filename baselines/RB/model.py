"""Rule-based controller for HEMS as evaluation baseline.

Strategy:
- BESS:
  1) If PV > load AND BESS has room → charge (absorb local surplus)
  2) Else if tariff is in lower quartile (past 24h) AND BESS has room → charge
  3) Else if tariff is in upper quartile AND BESS has charge AND no PV surplus → discharge
  4) Else idle
- EV: charge whenever connected and SoC below target (greedy, simple baseline).
- PV: no curtailment (env clips via export limit when needed).

Tariff-percentile (obs[23]) is scale-invariant, so this controller behaves
sensibly across tar_s, tar_w, tar_sw, tar_tou, and tar_flat without retuning.

Exposes `load_actor(cfg)` so it slots into the same EvalRunner pipeline used
for SAC architectures — no training, no critic.
"""
import torch
import torch.nn as nn


def _last(x):
    return x if x.dim() == 2 else x[:, -1, :]


class RuleBased(nn.Module):
    def __init__(self, soc_ev_target=0.8, cheap=0.25, expensive=0.75,
                 soc_bess_min=0.05, soc_bess_max=0.95):
        super().__init__()
        self.soc_ev_target = float(soc_ev_target)
        self.cheap = float(cheap)
        self.expensive = float(expensive)
        self.soc_bess_min = float(soc_bess_min)
        self.soc_bess_max = float(soc_bess_max)
        # Dummy buffer so .state_dict() / .load_state_dict() work without errors
        self.register_buffer("_dummy", torch.zeros(1))

    def act(self, x, deterministic=True):
        s = _last(x)
        load = s[..., 10:11]
        pv = s[..., 11:12]
        soc_bess = s[..., 12:13]
        soc_ev = s[..., 13:14]
        ev_ctrl = s[..., 14:15]
        tar_pct = s[..., 23:24]

        zero = torch.zeros_like(soc_bess)
        one = torch.ones_like(soc_bess)
        pv_surplus = pv > (load + 1e-3)

        bess_room = soc_bess < self.soc_bess_max
        bess_have = soc_bess > self.soc_bess_min
        cheap = tar_pct < self.cheap
        expensive = tar_pct > self.expensive

        charge = (pv_surplus | cheap) & bess_room
        discharge = expensive & bess_have & (~pv_surplus)

        a_bess = torch.where(charge, one, zero)
        a_bess = torch.where(discharge, -one, a_bess)

        a_ev = ((ev_ctrl > 0.5) & (soc_ev < self.soc_ev_target)).to(x.dtype)
        a_pv = zero
        return torch.cat([a_bess, a_ev, a_pv], dim=-1)

    def forward(self, x):
        return self.act(x, deterministic=True)


def load_actor(cfg, device=None):
    rb = RuleBased(
        soc_ev_target=cfg.get("soc_ev_target", 0.8),
        cheap=cfg.get("cheap", 0.25),
        expensive=cfg.get("expensive", 0.75),
    )
    return rb if device is None else rb.to(device)


def load_critic(cfg, device=None):
    raise NotImplementedError("Rule-based controller has no critic; eval-only.")
