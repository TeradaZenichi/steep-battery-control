"""Lightweight multi-head attention SAC model.

This is intentionally smaller than a Transformer encoder: one input projection,
one self-attention block, and the last contextual token feeds the heads. The
observation already carries calendar encodings, so no extra positional encoding
is added here.
"""
import torch
import torch.nn as nn

from models.GRU.model import Actor as BaseActor


class Actor(BaseActor):
    def __init__(self, input_dim, hidden_dim, num_layers, head_dim, num_heads=4,
                 dropout=0.05, **kwargs):
        super().__init__(input_dim, hidden_dim, 1, head_dim, **kwargs)
        del self.gru
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _enc(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        z = self.proj(x)
        h, _ = self.attn(z, z, z, need_weights=False)
        h = h + self.ff(h)
        h = h[:, -1, :]
        return self.ln(h) if self.ln is not None else h


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, head_dim,
                 num_heads=4, dropout=0.05, use_layer_norm=False):
        super().__init__()

        def build():
            proj = nn.Linear(state_dim, hidden_dim)
            attn = nn.MultiheadAttention(
                hidden_dim, int(num_heads), dropout=float(dropout), batch_first=True
            )
            ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, hidden_dim),
            )
            ln = nn.LayerNorm(hidden_dim) if use_layer_norm else None
            mlp = nn.Sequential(
                nn.Linear(hidden_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, head_dim), nn.ReLU(),
                nn.Linear(head_dim, 1),
            )
            return proj, attn, ff, ln, mlp

        self.proj1, self.attn1, self.ff1, self.ln1, self.mlp1 = build()
        self.proj2, self.attn2, self.ff2, self.ln2, self.mlp2 = build()

    def _q(self, proj, attn, ff, ln, mlp, obs, act):
        x = obs.unsqueeze(1) if obs.dim() == 2 else obs
        z = proj(x)
        h, _ = attn(z, z, z, need_weights=False)
        h = h + ff(h)
        h = h[:, -1, :]
        if ln is not None:
            h = ln(h)
        return mlp(torch.cat([h, act], dim=-1))

    def forward(self, obs, act):
        return (
            self._q(self.proj1, self.attn1, self.ff1, self.ln1, self.mlp1, obs, act),
            self._q(self.proj2, self.attn2, self.ff2, self.ln2, self.mlp2, obs, act),
        )


def load_actor(cfg, device=None):
    a = Actor(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg.get("hidden_dim", 192),
        num_layers=cfg.get("num_layers", 1),
        head_dim=cfg.get("head_dim", 64),
        num_heads=cfg.get("num_heads", 4),
        dropout=cfg.get("dropout", 0.05),
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
        hidden_dim=int(cfg.get("hidden_dim", 192)),
        num_layers=int(cfg.get("num_layers", 1)),
        head_dim=int(cfg.get("head_dim", 64)),
        num_heads=int(cfg.get("num_heads", 4)),
        dropout=float(cfg.get("dropout", 0.05)),
        use_layer_norm=bool(cfg.get("use_layer_norm", False)),
    )
    return c if device is None else c.to(device)
