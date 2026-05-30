"""GRU SAC: actor samples 2 stochastic dims (BESS, EV); PV curtailment is
closed deterministically as the minimum required by the export limit, plus an
economic component when the effective export price is worse than curtailing.

The executed action seen by the env and the critic is 3-D [BESS, EV, PV];
the policy entropy only counts the 2 strategic dimensions.
"""
from torch.distributions import Normal
import torch.nn as nn
import torch
import json

from models._physics import Battery, EV


def _last(x):
    return x if x.dim() == 2 else x[:, -1, :]


def _scale(z, lo, hi):
    return lo + 0.5 * (z + 1.0) * (hi - lo)


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, head_dim,
                 log_std_min=-10.0, log_std_max=2.0, init_log_std_bias=-2.0,
                 parameters="data/parameters.json", use_layer_norm=False):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.mu = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 2))
        self.logstd = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 2))
        nn.init.zeros_(self.logstd[-1].weight)
        nn.init.constant_(self.logstd[-1].bias, float(init_log_std_bias))

        with open(parameters, "r", encoding="utf-8") as f:
            p = json.load(f)
        dt = p["general"]["timestep"] / 60.0
        self.bess = Battery(p["BESS"], dt)
        self.ev = EV(p["EV"], dt)
        self.register_buffer("Pnorm", torch.tensor(float(p["general"]["Pnorm"]), dtype=torch.float32))
        self.register_buffer("gmin", torch.tensor(float(-p["Grid"]["Pmax_export"]), dtype=torch.float32))
        self.register_buffer("export_factor", torch.tensor(float(p["Grid"].get("export_tariff_factor", 1.0)), dtype=torch.float32), persistent=False)
        # SmartHomeEnv hardcodes the marginal PV curtailment penalty as 0.01/kWh.
        self.register_buffer("pv_cut_cost", torch.tensor(0.01, dtype=torch.float32), persistent=False)
        self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)

    def _enc(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.ln(h) if self.ln is not None else h

    def policy(self, x):
        h = self._enc(x)
        return self.mu(h), torch.clamp(self.logstd(h), self.log_std_min, self.log_std_max)

    def _box(self, x):
        s = _last(x)
        bmin, bmax = self.bess.limits(s[..., 12:13])
        emin, emax = self.ev.limits(s[..., 13:14], s[..., 14:15])
        return torch.cat([bmin, emin], dim=-1), torch.cat([bmax, emax], dim=-1)

    def _pv_curtailment(self, x, be_action):
        s = _last(x)
        pload = s[..., 10:11] * self.Pnorm
        ppv = torch.clamp(s[..., 11:12] * self.Pnorm, min=0.0)
        pbess = be_action[..., 0:1] * self.bess.Pmax
        a_ev = be_action[..., 1:2]
        pev = torch.where(a_ev >= 0.0, a_ev * self.ev.Pmax_c, a_ev * self.ev.Pmax_d)
        pgrid_full_pv = pload + pbess + pev - ppv
        pcut_physical = torch.clamp(self.gmin - pgrid_full_pv, min=0.0)

        pbess_c = torch.clamp(pbess, min=0.0)
        pev_c = torch.clamp(pev, min=0.0)
        pgrid_export = torch.clamp(-pgrid_full_pv, min=0.0)
        ppv_export_cap = torch.clamp(ppv - pload - pbess_c - pev_c, min=0.0)
        ppv_export = torch.minimum(pgrid_export, ppv_export_cap)

        buy_price = s[..., 15:16]
        export_value = self.export_factor * buy_price
        cut_export = export_value < -self.pv_cut_cost
        pcut_economic = torch.where(cut_export, ppv_export, torch.zeros_like(ppv_export))
        pcut = torch.maximum(pcut_physical, pcut_economic)
        eps = ppv.new_tensor(1e-12)
        ppv_safe = ppv.clamp_min(eps)
        return torch.where(ppv > eps, torch.clamp(pcut / ppv_safe, 0.0, 1.0), torch.zeros_like(ppv))

    def _action(self, x, z):
        lo, hi = self._box(x)
        be_action = _scale(z, lo, hi)
        pv_action = self._pv_curtailment(x, be_action)
        return torch.cat([be_action, pv_action], dim=-1), (hi - lo)

    def forward(self, x):
        mu, _ = self.policy(x)
        a, _ = self._action(x, torch.tanh(mu))
        return a

    def act(self, x, deterministic=False):
        if deterministic:
            return self.forward(x)
        mu, log_std = self.policy(x)
        raw = Normal(mu, log_std.exp()).rsample()
        a, _ = self._action(x, torch.tanh(raw))
        return a

    def sample(self, x, with_mu=True):
        mu, log_std = self.policy(x)
        dist = Normal(mu, log_std.exp())
        raw = dist.rsample()
        z = torch.tanh(raw)
        action, span = self._action(x, z)

        eps = raw.new_tensor(1e-6)
        thr = raw.new_tensor(1e-3)
        logp_unit = dist.log_prob(raw) - torch.log(1.0 - z.pow(2) + eps)
        # If a BESS/EV box collapses (span ≈ 0), that dim is effectively
        # deterministic and contributes 0 to the policy entropy. PV is not
        # sampled — its log-prob is always 0.
        per_dim = torch.where(
            span > thr,
            logp_unit - torch.log((0.5 * span).clamp_min(eps)),
            torch.zeros_like(logp_unit),
        )
        logp = per_dim.sum(dim=-1, keepdim=True)
        cost = torch.zeros_like(logp)
        mu_action = self.forward(x) if with_mu else None
        return action, logp, mu_action, cost


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, head_dim,
                 use_layer_norm=False, dropout_rate=0.0, n_critics=2):
        super().__init__()
        p = float(dropout_rate)

        def build():
            gru = nn.GRU(state_dim, hidden_dim, num_layers, batch_first=True)
            ln = nn.LayerNorm(hidden_dim) if use_layer_norm else None
            layers = [nn.Linear(hidden_dim + action_dim, hidden_dim), nn.ReLU()]
            if p > 0: layers.append(nn.Dropout(p))
            layers += [nn.Linear(hidden_dim, head_dim), nn.ReLU()]
            if p > 0: layers.append(nn.Dropout(p))
            layers += [nn.Linear(head_dim, 1)]
            return gru, ln, nn.Sequential(*layers)

        self.n_critics = int(n_critics)
        self.grus = nn.ModuleList(); self.lns = nn.ModuleList(); self.mlps = nn.ModuleList()
        for _ in range(self.n_critics):
            gru, ln, mlp = build()
            self.grus.append(gru)
            self.lns.append(ln if ln is not None else nn.Identity())
            self.mlps.append(mlp)
        # Backward-compatible aliases for num_heads==2 (so old code with .gru1/.gru2 still works).
        if self.n_critics == 2:
            self.gru1, self.ln1, self.mlp1 = self.grus[0], self.lns[0], self.mlps[0]
            self.gru2, self.ln2, self.mlp2 = self.grus[1], self.lns[1], self.mlps[1]

    def _q(self, gru, ln, mlp, obs, act):
        x = obs.unsqueeze(1) if obs.dim() == 2 else obs
        out, _ = gru(x)
        h = out[:, -1, :]
        if ln is not None and not isinstance(ln, nn.Identity):
            h = ln(h)
        return mlp(torch.cat([h, act], dim=-1))

    def forward(self, obs, act):
        return tuple(self._q(self.grus[i], self.lns[i], self.mlps[i], obs, act) for i in range(self.n_critics))


def load_actor(cfg, device=None):
    a = Actor(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg.get("hidden_dim", 256),
        num_layers=cfg.get("num_layers", 1),
        head_dim=cfg.get("head_dim", 64),
        log_std_min=cfg.get("log_std_min", -10.0),
        log_std_max=cfg.get("log_std_max", 2.0),
        init_log_std_bias=cfg.get("init_log_std_bias", -2.0),
        parameters=cfg.get("parameters", "data/parameters.json"),
        use_layer_norm=cfg.get("use_layer_norm", False),
    )
    return a if device is None else a.to(device)


def load_critic(cfg, device=None):
    c = Critic(
        state_dim=int(cfg["state_dim"]),
        action_dim=int(cfg.get("action_dim", 3)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 1)),
        head_dim=int(cfg.get("head_dim", 64)),
        use_layer_norm=bool(cfg.get("use_layer_norm", False)),
        dropout_rate=float(cfg.get("dropout_rate", 0.0)),
        n_critics=int(cfg.get("n_critics", 2)),
    )
    return c if device is None else c.to(device)
