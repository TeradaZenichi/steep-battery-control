from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
import torch
import json


class Battery:
    def __init__(self, general):
        p = general["BESS"]
        self.Δt = torch.tensor(general["general"]["timestep"] / 60, dtype=torch.float32)
        self.Pmax = torch.tensor(p["Pmax"], dtype=torch.float32)
        self.Emax = torch.tensor(p["Emax"], dtype=torch.float32)
        self.DoD = torch.tensor(p["DoD"], dtype=torch.float32)
        self.η = torch.tensor(p["η"], dtype=torch.float32)
        self.β = torch.tensor(p["β"], dtype=torch.float32)

        sp = p["soc_power_curve_pu"]
        self.soc_grid = torch.tensor(sp["soc"], dtype=torch.float32)
        self.ch_pu = torch.tensor(sp["charge_pu"], dtype=torch.float32)
        self.dis_pu = torch.tensor(sp["discharge_pu"], dtype=torch.float32)

    @staticmethod
    def _like(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return c.to(device=x.device, dtype=x.dtype)

    def limits(self, soc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        soc = torch.clamp(soc.to(torch.float32), 0.0, 1.0)

        dt = self._like(soc, self.Δt)
        Pmax = self._like(soc, self.Pmax)
        Emax = self._like(soc, self.Emax)
        DoD = self._like(soc, self.DoD)
        eta = self._like(soc, self.η)
        beta = self._like(soc, self.β)

        soc_grid = self._like(soc, self.soc_grid)
        ch_pu = self._like(soc, self.ch_pu)
        dis_pu = self._like(soc, self.dis_pu)

        idx = torch.searchsorted(soc_grid, soc) - 1
        idx = torch.clamp(idx, 0, soc_grid.numel() - 1).long()

        Emin = Emax * (1.0 - DoD)

        E = soc * Emax
        E = E * (1.0 - beta * dt)

        eps = soc.new_tensor(1e-12)

        P_curve_ch = ch_pu[idx] * Pmax
        P_head_ch = torch.clamp((Emax - E) / (dt * eta + eps), min=0.0)
        Pcmd_max = torch.minimum(P_curve_ch, P_head_ch)
        amax = torch.clamp(Pcmd_max / (Pmax + eps), 0.0, 1.0)

        P_curve_dis = -dis_pu[idx] * Pmax
        P_head_dis = (Emin - E) * eta / (dt + eps)
        Pcmd_min = torch.maximum(P_curve_dis, P_head_dis)
        amin = torch.clamp(Pcmd_min / (Pmax + eps), -1.0, 0.0)

        return amin, amax


class EV:
    def __init__(self, general):
        p = general["EV"]
        self.Δt = torch.tensor(general["general"]["timestep"] / 60, dtype=torch.float32)
        self.Pmax_c = torch.tensor(p["Pmax_c"], dtype=torch.float32)
        self.Pmax_d = torch.tensor(p["Pmax_d"], dtype=torch.float32)
        self.Emax = torch.tensor(p["Emax"], dtype=torch.float32)
        self.DoD = torch.tensor(p["DoD"], dtype=torch.float32)
        self.η = torch.tensor(p["η"], dtype=torch.float32)
        self.β = torch.tensor(p["β"], dtype=torch.float32)

        sp = p["soc_power_curve_pu"]
        self.soc_grid = torch.tensor(sp["soc"], dtype=torch.float32)
        self.ch_pu = torch.tensor(sp["charge_pu"], dtype=torch.float32)
        self.dis_pu = torch.tensor(sp["discharge_pu"], dtype=torch.float32)

    @staticmethod
    def _like(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return c.to(device=x.device, dtype=x.dtype)

    def get_limits(self, soc: torch.Tensor, status: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        soc = torch.clamp(soc.to(torch.float32), 0.0, 1.0)
        status = torch.clamp(status.to(torch.float32), 0.0, 1.0)

        dt = self._like(soc, self.Δt)
        Emax = self._like(soc, self.Emax)
        DoD = self._like(soc, self.DoD)
        eta = self._like(soc, self.η)
        beta = self._like(soc, self.β)

        Pmax_c = self._like(soc, self.Pmax_c)
        Pmax_d = self._like(soc, self.Pmax_d)

        soc_grid = self._like(soc, self.soc_grid)
        ch_pu = self._like(soc, self.ch_pu)
        dis_pu = self._like(soc, self.dis_pu)

        idx = torch.searchsorted(soc_grid, soc) - 1
        idx = torch.clamp(idx, 0, soc_grid.numel() - 1).long()

        Emin = Emax * (1.0 - DoD)

        E = soc * Emax
        E = E * (1.0 - beta * dt)

        eps = soc.new_tensor(1e-12)

        P_curve_ch = ch_pu[idx] * Pmax_c
        P_head_ch = torch.clamp((Emax - E) / (dt * eta + eps), min=0.0)
        Pcmd_max = torch.minimum(P_curve_ch, P_head_ch)
        amax = torch.clamp(Pcmd_max / (Pmax_c + eps), 0.0, 1.0)

        P_curve_dis = -dis_pu[idx] * Pmax_d
        P_head_dis = (Emin - E) * eta / (dt + eps)
        Pcmd_min = torch.maximum(P_curve_dis, P_head_dis)
        amin = torch.clamp(Pcmd_min / (Pmax_d + eps), -1.0, 0.0)

        amin = amin * status
        amax = amax * status

        return amin, amax


# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ TCN building blocks Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

class CausalConv1d(nn.Module):
    """Conv1d with left-side causal padding (no future leakage)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, use_weight_norm: bool = False):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                         dilation=dilation)
        if use_weight_norm:
            self.conv = nn.utils.parametrizations.weight_norm(conv)
        else:
            self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Residual block: two causal convolutions + skip connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.0, use_weight_norm: bool = False):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation, use_weight_norm=use_weight_norm)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation, use_weight_norm=use_weight_norm)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        res = self.skip(x)

        out = F.relu(self.conv1(x))
        out = self.drop(out)

        out = F.relu(self.conv2(out))
        out = self.drop(out)

        return F.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation."""

    def __init__(self, input_dim: int, num_channels: list[int],
                 kernel_size: int = 3, dropout: float = 0.0, use_weight_norm: bool = False):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout, use_weight_norm=use_weight_norm))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)  Ã¢â€ â€™ (B, C_out, T)
        return self.network(x)


# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ TCN Actor Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

class TCNActor(nn.Module):
    """Temporal Convolutional Network actor for SAC.

    Architecture:
        Input (B, T, input_dim)
        Ã¢â€ â€™ transpose to (B, input_dim, T)
        Ã¢â€ â€™ TemporalConvNet (causal dilated convolutions)
        Ã¢â€ â€™ last-timestep pooling Ã¢â€ â€™ (B, hidden)
        Ã¢â€ â€™ mu_head / logstd_head Ã¢â€ â€™ 3-dim action
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
        head_dim: int = 128,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        init_log_std_bias: float = -2.0,
        use_weight_norm: bool = False,
        parameters: str = "data/parameters.json",
    ):
        super().__init__()

        if num_channels is None:
            num_channels = [128, 128, 128]

        self.tcn = TemporalConvNet(
            input_dim,
            num_channels,
            kernel_size,
            dropout,
            use_weight_norm=use_weight_norm,
        )

        tcn_out_dim = num_channels[-1]

        self.mu_head = nn.Sequential(
            nn.Linear(tcn_out_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 3),
        )
        self.logstd_head = nn.Sequential(
            nn.Linear(tcn_out_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 3),
        )

        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self._init_logstd(init_log_std_bias)

        with open(parameters, "r", encoding="utf-8") as f:
            params = json.load(f)
        self.bess = Battery(params)
        self.ev = EV(params)

    def _init_logstd(self, init_log_std_bias: float) -> None:
        last = self.logstd_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, float(init_log_std_bias))

    @staticmethod
    def _map_pv(z_pv: torch.Tensor) -> torch.Tensor:
        return 0.5 * (z_pv + 1.0)

    @staticmethod
    def _to_seq(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.dim() == 2:
            return x.unsqueeze(1), True
        if x.dim() == 3:
            return x, False
        raise ValueError(f"Expected x with 2 or 3 dims, got {x.dim()}")

    def _dist_params(self, x: torch.Tensor):
        x_seq, squeezed = self._to_seq(x)
        # x_seq: (B, T, input_dim) Ã¢â€ â€™ transpose to (B, input_dim, T) for Conv1d
        h = self.tcn(x_seq.transpose(1, 2))  # (B, C_out, T)
        # Last-timestep pooling
        last = h[:, :, -1]  # (B, C_out)

        mu = self.mu_head(last)
        log_std = self.logstd_head(last)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mu, log_std

    def forward(self, x: torch.Tensor):
        mu, _ = self._dist_params(x)
        z = torch.tanh(mu)
        pv = self._map_pv(z[..., 2:3])
        action = torch.cat([z[..., 0:1], z[..., 1:2], pv], dim=-1)
        return action

    def _project(self, x: torch.Tensor, action_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x_last = x[:, -1, :]
        else:
            x_last = x

        soc_bess = x_last[..., 12:13]
        soc_ev = x_last[..., 13:14]
        ev_on = x_last[..., 14:15]

        amin_b, amax_b = self.bess.limits(soc_bess)
        amin_e, amax_e = self.ev.get_limits(soc_ev, ev_on)

        a_bess = torch.clamp(action_raw[..., 0:1], amin_b, amax_b)
        a_ev = torch.clamp(action_raw[..., 1:2], amin_e, amax_e)
        a_pv = torch.clamp(action_raw[..., 2:3], 0.0, 1.0)

        action_proj = torch.cat([a_bess, a_ev, a_pv], dim=-1)

        viol_bess = (action_raw[..., 0:1] - a_bess).abs()
        viol_ev = (action_raw[..., 1:2] - a_ev).abs() * ev_on
        cost = (viol_bess + viol_ev).pow(2)
        return action_proj, cost

    def sample(self, x: torch.Tensor):
        mu, log_std = self._dist_params(x)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        raw = dist.rsample()
        z = torch.tanh(raw)
        z_pv = self._map_pv(z[..., 2:3])
        z = torch.cat([z[..., 0:1], z[..., 1:2], z_pv], dim=-1)

        action_proj, cost = self._project(x, z)

        logp = dist.log_prob(raw).sum(dim=-1, keepdim=True)
        logp -= torch.log(1.0 - torch.tanh(raw).pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        mu_tanh = torch.tanh(mu)
        mu_pv = self._map_pv(mu_tanh[..., 2:3])
        mu_tanh = torch.cat([mu_tanh[..., 0:1], mu_tanh[..., 1:2], mu_pv], dim=-1)
        mu_action_proj, _ = self._project(x, mu_tanh)

        return action_proj, logp, mu_action_proj, cost


class TCNQNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        head_dim: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [128, 128, 128]

        self.tcn = TemporalConvNet(state_dim, num_channels, kernel_size, dropout)
        tcn_out_dim = num_channels[-1]

        layers = []
        prev = tcn_out_dim + action_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU()]
            prev = dim
        layers += [nn.Linear(prev, head_dim), nn.ReLU(), nn.Linear(head_dim, 1)]
        self.q_head = nn.Sequential(*layers)

    @staticmethod
    def _to_seq(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.dim() == 2:
            return x.unsqueeze(1), True
        if x.dim() == 3:
            return x, False
        raise ValueError(f"Expected x with 2 or 3 dims, got {x.dim()}")

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x_seq, _ = self._to_seq(obs)
        h = self.tcn(x_seq.transpose(1, 2))
        summary = h[:, :, -1]
        q_input = torch.cat([summary, act], dim=-1)
        return self.q_head(q_input)


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        head_dim: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.q1 = TCNQNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            head_dim=head_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.q2 = TCNQNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            head_dim=head_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        return self.q1(obs, act), self.q2(obs, act)


def load_actor(actor_cfg: dict, device=None):
    actor = TCNActor(
        input_dim=actor_cfg["input_dim"],
        num_channels=actor_cfg.get("num_channels", [128, 128, 128]),
        kernel_size=actor_cfg.get("kernel_size", 3),
        dropout=actor_cfg.get("dropout", 0.0),
        head_dim=actor_cfg.get("head_dim", 128),
        log_std_min=actor_cfg.get("log_std_min", -20.0),
        log_std_max=actor_cfg.get("log_std_max", 2.0),
        init_log_std_bias=actor_cfg.get("init_log_std_bias", -2.0),
        use_weight_norm=actor_cfg.get("use_weight_norm", False),
        parameters=actor_cfg.get("parameters", "data/parameters.json"),
    )
    if device is not None:
        actor = actor.to(device)
    return actor


def _extract_actor_state_dict(checkpoint: dict) -> dict:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dict")

    if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    for key in ("actor_state_dict", "state_dict", "model_state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            return nested

    return checkpoint


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict

    out = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            out[key[len("module."):]] = value
        else:
            out[key] = value
    return out


def _is_legacy_tcn_actor_state_dict(state_dict: dict) -> bool:
    has_legacy_norm = any(".norm1." in k or ".norm2." in k for k in state_dict.keys())
    has_legacy_conv_weight = any(
        ("tcn.network." in k) and k.endswith(".conv.weight") and (".parametrizations.weight." not in k)
        for k in state_dict.keys()
    )
    return has_legacy_norm or has_legacy_conv_weight


def _convert_legacy_tcn_actor_state_dict(state_dict: dict) -> dict:
    converted = {}
    for key, value in state_dict.items():
        if ".norm1." in key or ".norm2." in key:
            continue

        if ("tcn.network." in key) and key.endswith(".conv.weight") and (".parametrizations.weight." not in key):
            prefix = key[:-len("weight")]
            converted[prefix + "parametrizations.weight.original1"] = value
            if isinstance(value, torch.Tensor) and value.dim() > 1:
                dims = tuple(range(1, value.dim()))
                converted[prefix + "parametrizations.weight.original0"] = torch.linalg.vector_norm(
                    value,
                    ord=2,
                    dim=dims,
                    keepdim=True,
                )
            else:
                converted[prefix + "parametrizations.weight.original0"] = value.abs()
            continue

        converted[key] = value

    return converted


def _state_dict_uses_weight_norm(state_dict: dict) -> bool:
    return any(".parametrizations.weight.original" in k for k in state_dict.keys())


def _actor_expects_weight_norm(actor: nn.Module) -> bool:
    actor_state = actor.state_dict()
    return any(".parametrizations.weight.original" in k for k in actor_state.keys())


def _convert_weight_norm_tcn_actor_state_dict_to_plain(state_dict: dict) -> dict:
    converted = {}
    consumed = set()

    suffix_v = "parametrizations.weight.original1"
    suffix_g = "parametrizations.weight.original0"

    for key, value in state_dict.items():
        if key in consumed:
            continue

        if key.endswith(suffix_v):
            prefix = key[:-len(suffix_v)]
            g_key = prefix + suffix_g
            v = value
            g = state_dict.get(g_key)

            if isinstance(v, torch.Tensor) and isinstance(g, torch.Tensor):
                if v.dim() > 1:
                    dims = tuple(range(1, v.dim()))
                    denom = torch.linalg.vector_norm(v, ord=2, dim=dims, keepdim=True).clamp_min(1e-12)
                else:
                    denom = v.abs().clamp_min(1e-12)
                weight = g * (v / denom)
            else:
                weight = v

            converted[prefix + "weight"] = weight
            consumed.add(key)
            if g_key in state_dict:
                consumed.add(g_key)
            continue

        if key.endswith(suffix_g):
            continue

        converted[key] = value

    return converted


def load_actor_state_dict_compat(actor: nn.Module, checkpoint: dict, strict: bool = True):
    state_dict = _extract_actor_state_dict(checkpoint)
    state_dict = _strip_module_prefix(state_dict)

    expects_weight_norm = _actor_expects_weight_norm(actor)
    incoming_weight_norm = _state_dict_uses_weight_norm(state_dict)

    if expects_weight_norm and (not incoming_weight_norm) and _is_legacy_tcn_actor_state_dict(state_dict):
        state_dict = _convert_legacy_tcn_actor_state_dict(state_dict)
    elif (not expects_weight_norm) and incoming_weight_norm:
        state_dict = _convert_weight_norm_tcn_actor_state_dict_to_plain(state_dict)

    return actor.load_state_dict(state_dict, strict=strict)


def load_critic(critic_cfg: dict, device=None):
    state_dim = critic_cfg.get("state_dim")
    action_dim = int(critic_cfg.get("action_dim", 3))
    if state_dim is None:
        input_dim = critic_cfg.get("input_dim")
        if input_dim is None:
            raise KeyError("critic config must provide either 'state_dim' or 'input_dim'")
        state_dim = int(input_dim) - action_dim
        if state_dim <= 0:
            raise ValueError("critic input_dim must be greater than action_dim")

    head_dim = critic_cfg.get("head_dim")
    if head_dim is None:
        hidden_dims = critic_cfg.get("hidden_dims", [])
        if not hidden_dims:
            raise KeyError("critic config must provide 'head_dim' or non-empty 'hidden_dims'")
        head_dim = int(hidden_dims[-1])

    critics = Critic(
        state_dim=int(state_dim),
        action_dim=action_dim,
        hidden_dims=critic_cfg["hidden_dims"],
        head_dim=head_dim,
        num_channels=critic_cfg.get("num_channels", [128, 128, 128]),
        kernel_size=critic_cfg.get("kernel_size", 3),
        dropout=critic_cfg.get("dropout", 0.0),
    )
    if device is not None:
        critics = critics.to(device)
    return critics
