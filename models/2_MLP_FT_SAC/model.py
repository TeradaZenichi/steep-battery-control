# models/2_MLP_FT_SAC/model.py
from __future__ import annotations

import math
import torch
import torch.nn as nn


def _act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    return nn.ReLU()


def _sizes(x) -> list[int]:
    if x is None:
        return [256, 256]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


class Actor(nn.Module):
    def __init__(self, hp) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(getattr(hp, "input_dim", 0))
        out_dim = int(getattr(hp, "output_dim", 0))
        dropout = float(getattr(hp, "dropout", 0.0))
        act = str(getattr(hp, "activation", "relu"))
        out_act = str(getattr(hp, "output_activation", "none")).lower()

        for h in _sizes(getattr(hp, "hidden_sizes", None)):
            layers.append(nn.Linear(prev, h))
            layers.append(_act(act))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, out_dim))
        if out_act == "tanh":
            layers.append(nn.Tanh())
        elif out_act == "sigmoid":
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorGaussian(nn.Module):
    def __init__(self, hp) -> None:
        super().__init__()
        self.mean_net = Actor(hp)
        self.log_std = nn.Parameter(torch.full((int(getattr(hp, "output_dim", 0)),), float(getattr(hp, "log_std_init", -3.0))))
        self.log_std_min = float(getattr(hp, "log_std_min", -10.0))
        self.log_std_max = float(getattr(hp, "log_std_max", 2.0))
        self.squash_tanh = bool(getattr(hp, "squash_tanh", False))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.mean_net(obs)
        s = self.log_std.clamp(self.log_std_min, self.log_std_max).expand_as(m)
        return m, s

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        m, s = self(obs)
        a = m if deterministic else (m + torch.exp(s) * torch.randn_like(m))
        return torch.tanh(a) if self.squash_tanh else a

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m, s = self(obs)
        std = torch.exp(s)
        z = m + std * torch.randn_like(m)
        if not self.squash_tanh:
            return z, (-0.5 * (((z - m) / (std + 1e-8)) ** 2) - s - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1, keepdim=True)
        a = torch.tanh(z)
        lp = (-0.5 * (((z - m) / (std + 1e-8)) ** 2) - s - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1, keepdim=True) - torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return a, lp


class CriticQ(nn.Module):
    def __init__(self, hp) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(getattr(hp, "input_dim", 0)) + int(getattr(hp, "output_dim", 0))
        dropout = float(getattr(hp, "critic_dropout", getattr(hp, "dropout", 0.0)))
        act = str(getattr(hp, "critic_activation", getattr(hp, "activation", "relu")))
        hids = getattr(hp, "critic_hidden_sizes", None)
        if hids is None:
            hids = getattr(hp, "hidden_sizes", None)

        for h in _sizes(hids):
            layers.append(nn.Linear(prev, h))
            layers.append(_act(act))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class TwinCritic(nn.Module):
    def __init__(self, hp) -> None:
        super().__init__()
        self.q1 = CriticQ(hp)
        self.q2 = CriticQ(hp)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, act), self.q2(obs, act)
