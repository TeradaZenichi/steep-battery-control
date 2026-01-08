from __future__ import annotations

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


def _to_sizes(x) -> list[int]:
    if x is None:
        return [256, 256]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


class Actor(nn.Module):
    """
    Simple MLP actor used by both IL and TD3.
    - Input:  obs tensor (B, obs_dim)
    - Output: action tensor (B, act_dim)
    """

    def __init__(self, hp) -> None:
        super().__init__()
        self.hp = hp

        in_dim = int(getattr(hp, "input_dim", 0))
        out_dim = int(getattr(hp, "output_dim", 0))
        hidden = _to_sizes(getattr(hp, "hidden_sizes", None))
        dropout = float(getattr(hp, "dropout", 0.0))
        activation = str(getattr(hp, "activation", "relu"))
        out_act = str(getattr(hp, "output_activation", "none")).lower()

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(_act(activation))
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
