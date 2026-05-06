from torch.distributions import Normal
import torch.nn as nn
import torch
import json


class Battery:
    def __init__(self, general):
        p = general["BESS"]
        self.Δt   = torch.tensor(general["general"]["timestep"]/60, dtype=torch.float32)
        self.Pmax = torch.tensor(p["Pmax"], dtype=torch.float32)
        self.Emax = torch.tensor(p["Emax"], dtype=torch.float32)
        self.DoD  = torch.tensor(p["DoD"], dtype=torch.float32)
        self.η    = torch.tensor(p["η"], dtype=torch.float32)
        self.β    = torch.tensor(p["β"], dtype=torch.float32)

        sp = p["soc_power_curve_pu"]
        self.soc_grid = torch.tensor(sp["soc"], dtype=torch.float32)
        self.ch_pu    = torch.tensor(sp["charge_pu"], dtype=torch.float32)
        self.dis_pu   = torch.tensor(sp["discharge_pu"], dtype=torch.float32)

    @staticmethod
    def _like(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return c.to(device=x.device, dtype=x.dtype)

    def limits(self, soc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        soc = torch.clamp(soc.to(torch.float32), 0.0, 1.0)

        dt   = self._like(soc, self.Δt)
        Pmax = self._like(soc, self.Pmax)
        Emax = self._like(soc, self.Emax)
        DoD  = self._like(soc, self.DoD)
        eta  = self._like(soc, self.η)
        beta = self._like(soc, self.β)

        soc_grid = self._like(soc, self.soc_grid)
        ch_pu    = self._like(soc, self.ch_pu)
        dis_pu   = self._like(soc, self.dis_pu)

        idx = torch.searchsorted(soc_grid, soc) - 1
        idx = torch.clamp(idx, 0, soc_grid.numel() - 1).long()

        Emin = Emax * (1.0 - DoD)

        E = soc * Emax
        E = E * (1.0 - beta * dt)

        eps = soc.new_tensor(1e-12)

        P_curve_ch = ch_pu[idx] * Pmax
        P_head_ch  = torch.clamp((Emax - E) / (dt * eta + eps), min=0.0)
        Pcmd_max   = torch.minimum(P_curve_ch, P_head_ch)
        amax       = torch.clamp(Pcmd_max / (Pmax + eps), 0.0, 1.0)

        P_curve_dis = -dis_pu[idx] * Pmax
        P_head_dis  = (Emin - E) * eta / (dt + eps)
        Pcmd_min    = torch.maximum(P_curve_dis, P_head_dis)
        amin        = torch.clamp(Pcmd_min / (Pmax + eps), -1.0, 0.0)

        return amin, amax


class EV:
    def __init__(self, general):
        p = general["EV"]
        self.Δt     = torch.tensor(general["general"]["timestep"]/60, dtype=torch.float32)
        self.Pmax_c = torch.tensor(p["Pmax_c"], dtype=torch.float32)
        self.Pmax_d = torch.tensor(p["Pmax_d"], dtype=torch.float32)
        self.Emax   = torch.tensor(p["Emax"], dtype=torch.float32)
        self.DoD    = torch.tensor(p["DoD"], dtype=torch.float32)
        self.η      = torch.tensor(p["η"], dtype=torch.float32)
        self.β      = torch.tensor(p["β"], dtype=torch.float32)

        sp = p["soc_power_curve_pu"]
        self.soc_grid = torch.tensor(sp["soc"], dtype=torch.float32)
        self.ch_pu    = torch.tensor(sp["charge_pu"], dtype=torch.float32)
        self.dis_pu   = torch.tensor(sp["discharge_pu"], dtype=torch.float32)

    @staticmethod
    def _like(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return c.to(device=x.device, dtype=x.dtype)

    def get_limits(self, soc: torch.Tensor, status: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        soc    = torch.clamp(soc.to(torch.float32), 0.0, 1.0)
        status = torch.clamp(status.to(torch.float32), 0.0, 1.0)

        dt   = self._like(soc, self.Δt)
        Emax = self._like(soc, self.Emax)
        DoD  = self._like(soc, self.DoD)
        eta  = self._like(soc, self.η)
        beta = self._like(soc, self.β)

        Pmax_c = self._like(soc, self.Pmax_c)
        Pmax_d = self._like(soc, self.Pmax_d)

        soc_grid = self._like(soc, self.soc_grid)
        ch_pu    = self._like(soc, self.ch_pu)
        dis_pu   = self._like(soc, self.dis_pu)

        idx = torch.searchsorted(soc_grid, soc) - 1
        idx = torch.clamp(idx, 0, soc_grid.numel() - 1).long()

        Emin = Emax * (1.0 - DoD)

        E = soc * Emax
        E = E * (1.0 - beta * dt)

        eps = soc.new_tensor(1e-12)

        P_curve_ch = ch_pu[idx] * Pmax_c
        P_head_ch  = torch.clamp((Emax - E) / (dt * eta + eps), min=0.0)
        Pcmd_max   = torch.minimum(P_curve_ch, P_head_ch)
        amax       = torch.clamp(Pcmd_max / (Pmax_c + eps), 0.0, 1.0)

        P_curve_dis = -dis_pu[idx] * Pmax_d
        P_head_dis  = (Emin - E) * eta / (dt + eps)
        Pcmd_min    = torch.maximum(P_curve_dis, P_head_dis)
        amin        = torch.clamp(Pcmd_min / (Pmax_d + eps), -1.0, 0.0)

        amin = amin * status
        amax = amax * status

        return amin, amax



class Actor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        head_dim: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        init_log_std_bias: float = -2.0,
        parameters = "data/parameters.json",
    ):
        super().__init__()

        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU()]
            prev = dim
        self.backbone = nn.Sequential(*layers)

        self.mu_head = nn.Sequential(
            nn.Linear(prev, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 3),
        )
        self.logstd_head = nn.Sequential(
            nn.Linear(prev, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 3),
        )

        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self._init_logstd(init_log_std_bias)
        with open(parameters, 'r', encoding="utf-8") as f:
            params  = json.load(f)
        self.bess   = Battery(params)
        self.ev     = EV(params)

    def _init_logstd(self, init_log_std_bias: float) -> None:
        last = self.logstd_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, float(init_log_std_bias))

    @staticmethod
    def _flatten_history(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            b, t, d = x.shape
            return x.reshape(b, t * d)
        raise ValueError(f"Expected x with 2 or 3 dims, got {x.dim()}")

    @staticmethod
    def _last_step(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x[:, -1, :]
        raise ValueError(f"Expected x with 2 or 3 dims, got {x.dim()}")

    def _dist_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_flat = self._flatten_history(x)
        h = self.backbone(x_flat)
        mu = self.mu_head(h)
        log_std = self.logstd_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    @staticmethod
    def _map_pv(z_pv: torch.Tensor) -> torch.Tensor:
        return torch.exp(5.0 * (z_pv - 1.0)).clamp(0.0, 1.0)


    @staticmethod
    def _last_obs(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x[:, -1, :]
        raise ValueError(f"Expected x with 2 or 3 dims, got {x.dim()}")

    @staticmethod
    def _scale_unit(z: torch.Tensor, amin: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
        return amin + 0.5 * (z + 1.0) * (amax - amin)

    def _action_bounds(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_last = self._last_obs(x)
        soc_bess = x_last[..., 12:13]
        soc_ev = x_last[..., 13:14]
        ev_on = x_last[..., 14:15]
        amin_b, amax_b = self.bess.limits(soc_bess)
        amin_e, amax_e = self.ev.get_limits(soc_ev, ev_on)
        return amin_b, amax_b, amin_e, amax_e

    def _action_from_unit(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        amin_b, amax_b, amin_e, amax_e = self._action_bounds(x)
        a_bess = self._scale_unit(z[..., 0:1], amin_b, amax_b)
        a_ev = self._scale_unit(z[..., 1:2], amin_e, amax_e)
        a_pv = self._map_pv(z[..., 2:3])
        return torch.cat([a_bess, a_ev, a_pv], dim=-1)

    def _action_and_logp(self, x: torch.Tensor, raw: torch.Tensor, dist: Normal) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.tanh(raw)
        amin_b, amax_b, amin_e, amax_e = self._action_bounds(x)

        a_bess = self._scale_unit(z[..., 0:1], amin_b, amax_b)
        a_ev = self._scale_unit(z[..., 1:2], amin_e, amax_e)
        a_pv = self._map_pv(z[..., 2:3])
        action = torch.cat([a_bess, a_ev, a_pv], dim=-1)

        eps = raw.new_tensor(1e-6)
        logp_unit = dist.log_prob(raw) - torch.log(1.0 - z.pow(2) + eps)

        span_b = torch.clamp(amax_b - amin_b, min=0.0)
        span_e = torch.clamp(amax_e - amin_e, min=0.0)
        active_b = span_b > eps
        active_e = span_e > eps

        logp_b = torch.where(
            active_b,
            logp_unit[..., 0:1] - torch.log((0.5 * span_b).clamp_min(eps)),
            torch.zeros_like(logp_unit[..., 0:1]),
        )
        logp_e = torch.where(
            active_e,
            logp_unit[..., 1:2] - torch.log((0.5 * span_e).clamp_min(eps)),
            torch.zeros_like(logp_unit[..., 1:2]),
        )
        logp_pv = logp_unit[..., 2:3] - torch.log(5.0 * a_pv.clamp_min(eps))
        logp = logp_b + logp_e + logp_pv
        return action, logp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self._dist_params(x)
        action = self._action_from_unit(x, torch.tanh(mu))
        return action

    def _project(self, x: torch.Tensor, action_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Projects raw actions to the feasible set induced by the observation.
        Returns: (action_proj, cost) where cost = violation^2.
        Observation indices:
        12 -> bess_soc
        13 -> ev_soc
        14 -> ev_present (0/1)
        """
        x_last = self._last_step(x)
        soc_bess = x_last[..., 12:13]
        soc_ev   = x_last[..., 13:14]
        ev_on    = x_last[..., 14:15]

        amin_b, amax_b = self.bess.limits(soc_bess)
        amin_e, amax_e = self.ev.get_limits(soc_ev, ev_on)

        a_bess = torch.clamp(action_raw[..., 0:1], amin_b, amax_b)
        a_ev   = torch.clamp(action_raw[..., 1:2], amin_e, amax_e)
        a_pv   = torch.clamp(action_raw[..., 2:3], 0.0, 1.0)

        action_proj = torch.cat([a_bess, a_ev, a_pv], dim=-1)

        # --- CHANGE: gate EV violation when disconnected ---
        viol_bess = (action_raw[..., 0:1] - a_bess).abs()
        viol_ev   = (action_raw[..., 1:2] - a_ev).abs() * ev_on
        viol = viol_bess + viol_ev
        # --- END CHANGE ---

        cost = viol.pow(2)
        return action_proj, cost

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self._dist_params(x)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        raw = dist.rsample()
        action, logp = self._action_and_logp(x, raw, dist)
        action_proj, cost = self._project(x, action)

        mu_action = self._action_from_unit(x, torch.tanh(mu))
        mu_action_proj, _ = self._project(x, mu_action)

        return action_proj, logp, mu_action_proj, cost


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], head_dim: int):
        super().__init__()

        def build():
            layers = []
            prev = input_dim
            for dim in hidden_dims:
                layers += [nn.Linear(prev, dim), nn.ReLU()]
                prev = dim
            layers += [nn.Linear(prev, head_dim), nn.ReLU(), nn.Linear(head_dim, 1)]
            return nn.Sequential(*layers)

        self.q1 = build()
        self.q2 = build()

    @staticmethod
    def _flatten_history(obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            return obs
        if obs.dim() == 3:
            b, t, d = obs.shape
            return obs.reshape(b, t * d)
        raise ValueError(f"Expected obs with 2 or 3 dims, got {obs.dim()}")

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_flat = self._flatten_history(obs)
        x = torch.cat([obs_flat, act], dim=-1)
        return self.q1(x), self.q2(x)


def load_actor(actor_cfg: dict, device=None):
    actor = Actor(
        input_dim=actor_cfg["input_dim"],
        hidden_dims=actor_cfg["hidden_dims"],
        head_dim=actor_cfg["head_dim"],
        log_std_min=actor_cfg.get("log_std_min", -20.0),
        log_std_max=actor_cfg.get("log_std_max", 2.0),
        init_log_std_bias=actor_cfg.get("init_log_std_bias", -2.0),
        parameters=actor_cfg.get("parameters", "data/parameters.json"),
    )
    if device is not None:
        actor = actor.to(device)
    return actor


def load_critic(critic_cfg: dict, device=None):
    input_dim = critic_cfg.get("input_dim")
    if input_dim is None:
        state_dim = critic_cfg.get("state_dim")
        action_dim = critic_cfg.get("action_dim")
        if state_dim is None or action_dim is None:
            raise KeyError("critic config must provide either 'input_dim' or both 'state_dim' and 'action_dim'")
        input_dim = int(state_dim) + int(action_dim)

    head_dim = critic_cfg.get("head_dim")
    if head_dim is None:
        hidden_dims = critic_cfg.get("hidden_dims", [])
        if not hidden_dims:
            raise KeyError("critic config must provide 'head_dim' or non-empty 'hidden_dims'")
        head_dim = int(hidden_dims[-1])

    critics = Critic(
        input_dim=input_dim,
        hidden_dims=critic_cfg["hidden_dims"],
        head_dim=head_dim,
    )
    if device is not None:
        critics = critics.to(device)
    return critics
