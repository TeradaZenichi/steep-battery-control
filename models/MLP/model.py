from torch.distributions import Normal
import torch.nn as nn
import torch
import math
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

    def _dist_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_std = self.logstd_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    @staticmethod
    def _map_pv(z_pv: torch.Tensor) -> torch.Tensor:
        return 0.5 * (z_pv + 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self._dist_params(x)
        z = torch.tanh(mu)  # [-1,1] for all three dims
        pv = self._map_pv(z[..., 2:3])  # [0,1]
        return torch.cat([z[..., 0:1], z[..., 1:2], pv], dim=-1)

    def _project(self, x: torch.Tensor, action_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Projects raw actions to the feasible set induced by the observation.
        Returns: (action_proj, cost) where cost = violation^2.
        Observation indices:
        12 -> bess_soc
        13 -> ev_soc
        14 -> ev_present (0/1)
        """
        soc_bess = x[..., 12:13]
        soc_ev   = x[..., 13:14]
        ev_on    = x[..., 14:15]

        bmin, bmax = self.bess.limits(soc_bess)
        emin, emax = self.ev.get_limits(soc_ev, ev_on)

        a_bess = torch.clamp(action_raw[..., 0:1], bmin, bmax)
        a_ev   = torch.clamp(action_raw[..., 1:2], emin, emax)
        a_pv   = torch.clamp(action_raw[..., 2:3], 0.0, 1.0)

        action_proj = torch.cat([a_bess, a_ev, a_pv], dim=-1)

        viol = (action_raw[..., 0:1] - a_bess).abs() + (action_raw[..., 1:2] - a_ev).abs()
        cost = viol.pow(2)

        return action_proj, cost


    def sample(
        self,
        x: torch.Tensor,
        eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SAC sample with feasibility projection.
        Returns:
        action_proj      : action used by env/Q (projected)
        logp_raw         : log pi(a_raw|s) (pre-projection, used in SAC)
        mu_action_proj   : deterministic projected action (for eval/debug)
        cost             : violation^2 for Lagrange multiplier
        """
        mu, log_std = self._dist_params(x)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        u = dist.rsample()          # reparameterized sample
        z = torch.tanh(u)           # [-1,1]

        pv = self._map_pv(z[..., 2:3])
        action_raw = torch.cat([z[..., 0:1], z[..., 1:2], pv], dim=-1)

        # log pi(z|s) with tanh correction + PV affine map correction
        logp_u = dist.log_prob(u).sum(dim=-1, keepdim=True)
        log_det_tanh = torch.log(1.0 - z.pow(2) + eps).sum(dim=-1, keepdim=True)
        logp = (logp_u - log_det_tanh) + math.log(2.0)

        # deterministic (mean) action, then project it too
        mu_z = torch.tanh(mu)
        mu_pv = self._map_pv(mu_z[..., 2:3])
        mu_action_raw = torch.cat([mu_z[..., 0:1], mu_z[..., 1:2], mu_pv], dim=-1)

        action_proj, cost = self._project(x, action_raw)
        mu_action_proj, _ = self._project(x, mu_action_raw)

        return action_proj, logp, mu_action_proj, cost


    def predict(self, obs):
        self.eval()
        with torch.no_grad():
            if isinstance(obs, torch.Tensor):
                x = obs
                if x.ndim == 1:
                    x = x.unsqueeze(0)
            else:
                x = torch.as_tensor(obs, dtype=torch.float32)
                if x.ndim == 1:
                    x = x.unsqueeze(0)
            out = self.forward(x)
            return out.cpu().numpy().squeeze()

    def action(self, obs):
        """
        Deterministic action with feasibility projection.

        - Uses the policy mean (mu) deterministically (no sampling).
        - Applies tanh squashing and PV mapping to [0,1].
        - Projects the raw deterministic action to the feasible set.
        - Returns a numpy array (squeezed), consistent with predict().
        """
        self.eval()
        with torch.no_grad():
            if isinstance(obs, torch.Tensor):
                x = obs
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                x = x.to(next(self.parameters()).device, dtype=torch.float32)
            else:
                x = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
                if x.ndim == 1:
                    x = x.unsqueeze(0)

            mu, _ = self._dist_params(x)

            mu_z = torch.tanh(mu)  # [-1,1] for first two dims and PV latent
            mu_pv = self._map_pv(mu_z[..., 2:3])  # [0,1]
            action_raw = torch.cat([mu_z[..., 0:1], mu_z[..., 1:2], mu_pv], dim=-1)

            action_proj, _ = self._project(x, action_raw)

            return action_proj.cpu().numpy().squeeze()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()

        layers = []
        prev = state_dim + action_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU()]
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class DoubleCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.q1 = Critic(state_dim, action_dim, hidden_dims)
        self.q2 = Critic(state_dim, action_dim, hidden_dims)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


def load_actor(config, weights_path=None, device=None):
    model = Actor(
        config["input_dim"],
        config["hidden_dims"],
        config["head_dim"],
        # Default bounds match the Actor constructor (and common SAC settings).
        log_std_min=config.get("log_std_min", -20.0),
        log_std_max=config.get("log_std_max", 2.0),
        init_log_std_bias=config.get("init_log_std_bias", -2.0),
    )

    if weights_path:
        state = torch.load(weights_path, map_location=device if device is not None else "cpu")
        model.load_state_dict(state, strict=True)

    if device is not None:
        model.to(device)

    return model


def load_critic(config, weights_path=None, device=None):
    model = DoubleCritic(config["state_dim"], config["action_dim"], config["hidden_dims"])

    if weights_path:
        state = torch.load(weights_path, map_location=device if device is not None else "cpu")
        model.load_state_dict(state, strict=True)

    if device is not None:
        model.to(device)

    return model
