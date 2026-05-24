"""Battery and EV per-unit limits as a function of SoC. Shared across methods."""
import torch.nn as nn
import torch


def _idx(grid, soc):
    return torch.clamp(torch.searchsorted(grid, soc) - 1, 0, grid.numel() - 1).long()


class Battery(nn.Module):
    def __init__(self, p, dt):
        super().__init__()
        f = lambda v: torch.tensor(v, dtype=torch.float32)
        for k, v in dict(dt=dt, Pmax=p["Pmax"], Emax=p["Emax"], DoD=p["DoD"], eta=p["η"], beta=p["β"]).items():
            self.register_buffer(k, f(v))
        sp = p["soc_power_curve_pu"]
        self.register_buffer("soc_grid", f(sp["soc"]))
        self.register_buffer("ch_pu", f(sp["charge_pu"]))
        self.register_buffer("dis_pu", f(sp["discharge_pu"]))

    def limits(self, soc):
        soc = torch.clamp(soc.to(torch.float32), 0.0, 1.0)
        i = _idx(self.soc_grid, soc)
        Emin = self.Emax * (1.0 - self.DoD)
        E = soc * self.Emax * (1.0 - self.beta * self.dt)
        eps = soc.new_tensor(1e-12)
        ch = torch.minimum(self.ch_pu[i] * self.Pmax, torch.clamp((self.Emax - E) / (self.dt * self.eta + eps), min=0.0))
        dis = torch.maximum(-self.dis_pu[i] * self.Pmax, (Emin - E) * self.eta / (self.dt + eps))
        return torch.clamp(dis / (self.Pmax + eps), -1.0, 0.0), torch.clamp(ch / (self.Pmax + eps), 0.0, 1.0)


class EV(nn.Module):
    def __init__(self, p, dt):
        super().__init__()
        f = lambda v: torch.tensor(v, dtype=torch.float32)
        for k, v in dict(dt=dt, Pmax_c=p["Pmax_c"], Pmax_d=p["Pmax_d"], Emax=p["Emax"],
                         DoD=p["DoD"], eta=p["η"], beta=p["β"]).items():
            self.register_buffer(k, f(v))
        sp = p["soc_power_curve_pu"]
        self.register_buffer("soc_grid", f(sp["soc"]))
        self.register_buffer("ch_pu", f(sp["charge_pu"]))
        self.register_buffer("dis_pu", f(sp["discharge_pu"]))

    def limits(self, soc, status):
        soc = torch.clamp(soc.to(torch.float32), 0.0, 1.0)
        status = torch.clamp(status.to(torch.float32), 0.0, 1.0)
        i = _idx(self.soc_grid, soc)
        Emin = self.Emax * (1.0 - self.DoD)
        E = soc * self.Emax * (1.0 - self.beta * self.dt)
        eps = soc.new_tensor(1e-12)
        ch = torch.minimum(self.ch_pu[i] * self.Pmax_c, torch.clamp((self.Emax - E) / (self.dt * self.eta + eps), min=0.0))
        dis = torch.maximum(-self.dis_pu[i] * self.Pmax_d, (Emin - E) * self.eta / (self.dt + eps))
        return (torch.clamp(dis / (self.Pmax_d + eps), -1.0, 0.0) * status,
                torch.clamp(ch / (self.Pmax_c + eps), 0.0, 1.0) * status)
