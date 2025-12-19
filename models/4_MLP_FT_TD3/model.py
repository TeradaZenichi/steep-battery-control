from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn

try:  # pragma: no cover
    from .hp import HyperParameters
except ImportError:  # pragma: no cover
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from hp import HyperParameters


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


def _get_activation_cls(name: str) -> type[nn.Module]:
    return _ACTIVATIONS[name.lower()]


def build_mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for width in hidden_sizes:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ImitationMLP(nn.Module):
    def __init__(self, hparams: HyperParameters):
        super().__init__()
        self.hparams = hparams
        layers: list[nn.Module] = []
        in_dim = hparams.input_dim
        dropout = float(hparams.dropout)
        activation_cls = _get_activation_cls(hparams.activation)
        hidden = list(hparams.hidden_sizes)
        if not hidden:
            hidden = [max(2 * hparams.output_dim, 32)]
        for width in hidden:
            layers.append(nn.Linear(in_dim, width))
            layers.append(activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, hparams.output_dim)

    @classmethod
    def from_json(cls, json_path: str | Path) -> "ImitationMLP":
        hparams = HyperParameters.from_json(json_path)
        return cls(hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.backbone(x)
        return self.head(features)


class Actor(nn.Module):
    """Actor wrapper around the imitation MLP with action clamping."""

    def __init__(self, hparams: HyperParameters, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.core = ImitationMLP(hparams)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.core(x)
        return torch.min(torch.max(raw, self.action_low), self.action_high)


class Critic(nn.Module):
    """Twin-MLP critic mirroring the actor's hidden depth."""

    def __init__(self, hparams: HyperParameters, action_dim: int, hidden_sizes: Iterable[int], activation: str = "relu"):
        super().__init__()
        _ = activation  # kept for API symmetry
        self.q1 = build_mlp(hparams.input_dim + action_dim, tuple(hidden_sizes), 1, dropout=hparams.dropout)
        self.q2 = build_mlp(hparams.input_dim + action_dim, tuple(hidden_sizes), 1, dropout=hparams.dropout)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


if __name__ == "__main__":  # pragma: no cover
    cfg = Path(__file__).with_name("model.json")
    m = ImitationMLP.from_json(cfg)
    dummy = torch.randn(4, m.hparams.input_dim)
    out = m(dummy)
    print(out.shape)
