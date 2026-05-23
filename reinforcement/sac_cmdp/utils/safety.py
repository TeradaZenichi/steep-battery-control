"""Closed-form projection onto box ∩ grid-slab; also returns the violation
that the Lagrangian dual penalizes."""
import torch.nn as nn
import torch


def _last(x):
    return x if x.dim() == 2 else x[:, -1, :]


class SafetyLayer(nn.Module):
    def __init__(self, bess, ev, params):
        super().__init__()
        self.bess, self.ev = bess, ev
        f = lambda v: torch.tensor(v, dtype=torch.float32)
        self.register_buffer("Pnorm", f(params["general"]["Pnorm"]))
        self.register_buffer("gmin", f(-params["Grid"]["Pmax_export"]))
        self.register_buffer("gmax", f(params["Grid"]["Pmax_import"]))

    def project(self, x, action):
        s = _last(x)
        ab_min, ab_max = self.bess.limits(s[..., 12:13])
        ae_min, ae_max = self.ev.limits(s[..., 13:14], s[..., 14:15])
        pload = s[..., 10:11] * self.Pnorm
        ppv_avail = torch.clamp(s[..., 11:12] * self.Pnorm, min=0.0)
        L, U = self.gmin - pload, self.gmax - pload
        eps = action.new_tensor(1e-12)

        a_b = torch.clamp(action[..., 0:1], ab_min, ab_max)
        a_e = torch.clamp(action[..., 1:2], ae_min, ae_max)
        a_v = torch.clamp(action[..., 2:3], 0.0, 1.0)
        pb0 = a_b * self.bess.Pmax
        pe0 = torch.where(a_e >= 0.0, a_e * self.ev.Pmax_c, a_e * self.ev.Pmax_d)
        ypv0 = -ppv_avail * (1.0 - a_v)
        x0 = torch.cat([pb0, pe0, ypv0], dim=-1)

        lo = torch.cat([ab_min * self.bess.Pmax, ae_min * self.ev.Pmax_d, -ppv_avail], dim=-1)
        hi = torch.cat([ab_max * self.bess.Pmax, ae_max * self.ev.Pmax_c, torch.zeros_like(ppv_avail)], dim=-1)

        s0 = x0.sum(dim=-1, keepdim=True)
        target = torch.where(s0 < L, L, torch.where(s0 > U, U, s0))
        violation = torch.clamp(L - s0, min=0.0) + torch.clamp(s0 - U, min=0.0)

        mu_lo = (lo - x0).amin(dim=-1, keepdim=True)
        mu_hi = (hi - x0).amax(dim=-1, keepdim=True)
        for _ in range(16):
            mu = 0.5 * (mu_lo + mu_hi)
            cand_sum = torch.clamp(x0 + mu, lo, hi).sum(dim=-1, keepdim=True)
            mu_lo = torch.where(cand_sum < target, mu, mu_lo)
            mu_hi = torch.where(cand_sum >= target, mu, mu_hi)
        y = torch.clamp(x0 + 0.5 * (mu_lo + mu_hi), lo, hi)

        pb, pe, ypv = y[..., 0:1], y[..., 1:2], y[..., 2:3]
        ppv = -ypv
        b_proj = torch.clamp(pb / self.bess.Pmax.clamp_min(eps), ab_min, ab_max)
        e_scale = torch.where(pe >= 0.0, self.ev.Pmax_c, self.ev.Pmax_d).clamp_min(eps)
        e_proj = torch.clamp(pe / e_scale, ae_min, ae_max)
        v_proj = torch.where(ppv_avail > eps, 1.0 - ppv / ppv_avail.clamp_min(eps), torch.zeros_like(ppv))
        v_proj = torch.clamp(v_proj, 0.0, 1.0)
        return torch.cat([b_proj, e_proj, v_proj], dim=-1), violation
