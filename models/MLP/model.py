import math
import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """
    Unified Actor for IL (deterministic forward) and SAC (stochastic sample).
    - forward(): deterministic action in env bounds (BESS/EV in [-1,1], PV in [0,1])
    - sample(): stochastic action + log_prob consistent with tanh + PV affine mapping
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        head_dim: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        init_log_std_bias: float = -2.0,
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

        # Initialize log-std to a small value (near-deterministic) by default.
        self._init_logstd(init_log_std_bias)

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
        # Map PV from [-1,1] to [0,1]
        return 0.5 * (z_pv + 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Deterministic policy used by IL (and by SAC evaluation if desired).
        mu, _ = self._dist_params(x)
        z = torch.tanh(mu)  # [-1,1] for all three dims
        pv = self._map_pv(z[..., 2:3])  # [0,1]
        return torch.cat([z[..., 0:1], z[..., 1:2], pv], dim=-1)

    def sample(self, x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          action:     [batch,3] with (bess, ev) in [-1,1] and pv in [0,1]
          log_prob:   [batch,1] log pi(action|state) with correct change-of-variables
          mu_action:  [batch,3] deterministic action from mu (for eval/logging)
        """
        mu, log_std = self._dist_params(x)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        u = dist.rsample()          # reparameterized sample
        z = torch.tanh(u)           # [-1,1]

        # Build final action with PV mapped to [0,1]
        pv = self._map_pv(z[..., 2:3])
        action = torch.cat([z[..., 0:1], z[..., 1:2], pv], dim=-1)

        # log_prob in u-space
        logp_u = dist.log_prob(u).sum(dim=-1, keepdim=True)

        # tanh change-of-variables: z = tanh(u)
        log_det = torch.log(1.0 - z.pow(2) + eps).sum(dim=-1, keepdim=True)
        logp_z = logp_u - log_det

        # PV affine mapping: pv = 0.5*(z_pv + 1) -> z_pv = 2*pv - 1, |dz/dpv| = 2
        # Therefore log pi(pv) = log pi(z_pv) + log(2). Constant but included for consistency.
        logp = logp_z + math.log(2.0)

        # Deterministic action from mu
        mu_z = torch.tanh(mu)
        mu_pv = self._map_pv(mu_z[..., 2:3])
        mu_action = torch.cat([mu_z[..., 0:1], mu_z[..., 1:2], mu_pv], dim=-1)

        return action, logp, mu_action

    def predict(self, obs):
        # Deterministic prediction (IL-style): uses forward()
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

