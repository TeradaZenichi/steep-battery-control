"""Rule-based Smart controller for HEMS as evaluation baseline.

Differences from RB:
- **EV**: respects tariff (charge mostly when cheap) AND urgency (force charge
  if departure is imminent and SoC below safety).
- **BESS**: reserve management (don't discharge below `soc_bess_reserve`) and
  coordination with EV (back off charge rate when EV is charging at full).
- **PV**: still no curtailment.

This is the "smart heuristic" baseline — better than the naive RB but still
purely rule-based (no optimization, no learning).
"""
import torch
import torch.nn as nn


def _last(x):
    return x if x.dim() == 2 else x[:, -1, :]


class RuleBasedSmart(nn.Module):
    def __init__(self,
                 soc_ev_target=0.8,
                 soc_ev_safe=0.55,  # buffer above env's soc_min=0.5 to avoid penalty oscillation
                 soc_bess_reserve=0.3,
                 soc_bess_max=0.95,
                 cheap=0.25,
                 medium=0.5,
                 expensive=0.75,
                 ev_charge_horizon=0.25,  # ≈ 3h before departure → charge regardless of tariff
                 bess_coord_factor=0.0):
        super().__init__()
        self.soc_ev_target = float(soc_ev_target)
        self.soc_ev_safe = float(soc_ev_safe)
        self.soc_bess_reserve = float(soc_bess_reserve)
        self.soc_bess_max = float(soc_bess_max)
        self.cheap = float(cheap)
        self.medium = float(medium)
        self.expensive = float(expensive)
        self.ev_charge_horizon = float(ev_charge_horizon)
        self.bess_coord_factor = float(bess_coord_factor)
        self.register_buffer("_dummy", torch.zeros(1))

    def act(self, x, deterministic=True):
        s = _last(x)
        load = s[..., 10:11]
        pv = s[..., 11:12]
        soc_bess = s[..., 12:13]
        soc_ev = s[..., 13:14]
        ev_ctrl = s[..., 14:15]
        tar_pct = s[..., 23:24]
        t_until_dep = s[..., 26:27]

        zero = torch.zeros_like(soc_bess)
        one = torch.ones_like(soc_bess)
        connected = ev_ctrl > 0.5

        # ---- EV smart charging (3-tier priority) ----
        # 1) Safety: keep SoC ≥ soc_safe (buffer above env's soc_min) to avoid
        #    the per-step `penalty · (soc_min·Emax − E) · Δt` term.
        must_charge = connected & (soc_ev < self.soc_ev_safe)
        # 2) Pre-departure: if departure is close, ensure target is reached
        #    regardless of tariff (otherwise post-trip SoC may drop below safety).
        pre_dep = connected & (soc_ev < self.soc_ev_target) & (t_until_dep < self.ev_charge_horizon)
        # 3) Opportunistic: top up to target when tariff is NOT in upper quartile.
        ok_charge = connected & (soc_ev < self.soc_ev_target) & (tar_pct < self.expensive)
        a_ev = (must_charge | pre_dep | ok_charge).to(x.dtype)

        # ---- BESS: PV-first, arbitrage, reserve, EV coordination ----
        pv_surplus = pv > (load + 1e-3)
        bess_room = soc_bess < self.soc_bess_max
        bess_above_reserve = soc_bess > self.soc_bess_reserve
        ev_busy = a_ev > 0.5

        # PV surplus always charges (free energy, independent of EV).
        charge_pv = pv_surplus & bess_room
        # Arbitrage charge: cheap tariff AND not from grid coincidence with EV.
        # Use `bess_coord_factor` ∈ [0, 1] to scale the charge rate when EV is busy.
        charge_arb_solo = (tar_pct < self.cheap) & bess_room & ~pv_surplus & ~ev_busy
        charge_arb_coord = (tar_pct < self.cheap) & bess_room & ~pv_surplus & ev_busy
        discharge_arb = (tar_pct > self.expensive) & bess_above_reserve & ~pv_surplus

        a_bess = zero.clone()
        a_bess = torch.where(charge_pv, one, a_bess)
        a_bess = torch.where(charge_arb_solo, one, a_bess)
        a_bess = torch.where(charge_arb_coord, self.bess_coord_factor * one, a_bess)
        a_bess = torch.where(discharge_arb, -one, a_bess)

        a_pv = zero
        return torch.cat([a_bess, a_ev, a_pv], dim=-1)

    def forward(self, x):
        return self.act(x, deterministic=True)


def load_actor(cfg, device=None):
    rb = RuleBasedSmart(
        soc_ev_target=cfg.get("soc_ev_target", 0.8),
        soc_ev_safe=cfg.get("soc_ev_safe", 0.55),
        soc_bess_reserve=cfg.get("soc_bess_reserve", 0.3),
        soc_bess_max=cfg.get("soc_bess_max", 0.95),
        cheap=cfg.get("cheap", 0.25),
        medium=cfg.get("medium", 0.5),
        expensive=cfg.get("expensive", 0.75),
        ev_charge_horizon=cfg.get("ev_charge_horizon", 0.25),
        bess_coord_factor=cfg.get("bess_coord_factor", 0.0),
    )
    return rb if device is None else rb.to(device)


def load_critic(cfg, device=None):
    raise NotImplementedError("Rule-based controller has no critic; eval-only.")
