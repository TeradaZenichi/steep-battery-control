"""Multi-head attention SAC model (single pre-LN Transformer encoder block).

Design for stable training under SAC-CMDP:
  * sinusoidal positional encoding added to the projected history, so the
    attention sees explicit sequence order / recency (the calendar features give
    absolute wall-clock time, but not relative position within the window);
  * one `nn.TransformerEncoderLayer(norm_first=True)` block, i.e. pre-LN with a
    residual + LayerNorm around BOTH the self-attention and the feed-forward
    sublayers. An earlier hand-rolled block without the residual-around-attention
    collapsed the moment the actor started updating.

Capacity stays matched to the other encoders: 1 layer, `dim_feedforward =
hidden_dim`, and the positional encoding is a non-learnable buffer (0 params).
The last contextual token feeds the heads.
"""
import math

import torch
import torch.nn as nn

from models.GRU.model import Actor as BaseActor


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al.); 0 learnable params."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(int(max_len), int(d_model))
        pos = torch.arange(int(max_len)).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, int(d_model), 2).float() * (-math.log(10000.0) / int(d_model)))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


def _make_encoder(hidden_dim, num_heads, dropout, num_layers):
    layer = nn.TransformerEncoderLayer(
        d_model=int(hidden_dim), nhead=int(num_heads), dim_feedforward=int(hidden_dim),
        dropout=float(dropout), activation="relu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)),
                                 enable_nested_tensor=False)


class Actor(BaseActor):
    def __init__(self, input_dim, hidden_dim, num_layers, head_dim, num_heads=4,
                 dropout=0.05, **kwargs):
        super().__init__(input_dim, hidden_dim, 1, head_dim, **kwargs)
        del self.gru
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pe = SinusoidalPE(hidden_dim)
        self.enc = _make_encoder(hidden_dim, num_heads, dropout, num_layers)

    def _enc(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        # Attention is numerically fragile in bf16 (the policy collapses the moment
        # the actor starts updating). Run the whole encoder in fp32 even under a
        # bf16 autocast; the rest of the actor stays in the outer precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.pe(self.proj(x.float()))
            h = self.enc(z)[:, -1, :]
            if self.ln is not None:
                h = self.ln(h)
        return h


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, head_dim,
                 num_heads=4, dropout=0.05, use_layer_norm=False, dropout_rate=0.0, n_critics=2):
        super().__init__()
        # `dropout` is the architectural attention/feed-forward dropout; `dropout_rate`
        # is the DroQ head dropout, kept active at the Bellman target (see trainer).
        pq = float(dropout_rate)

        def build():
            proj = nn.Linear(state_dim, hidden_dim)
            pe = SinusoidalPE(hidden_dim)
            enc = _make_encoder(hidden_dim, num_heads, dropout, num_layers)
            ln = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            head = [nn.Linear(hidden_dim + action_dim, hidden_dim), nn.ReLU()]
            if pq > 0: head.append(nn.Dropout(pq))
            head += [nn.Linear(hidden_dim, head_dim), nn.ReLU()]
            if pq > 0: head.append(nn.Dropout(pq))
            head += [nn.Linear(head_dim, 1)]
            mlp = nn.Sequential(*head)
            return proj, pe, enc, ln, mlp

        self.n_critics = int(n_critics)
        self.projs = nn.ModuleList(); self.pes = nn.ModuleList(); self.encs = nn.ModuleList()
        self.lns = nn.ModuleList(); self.mlps = nn.ModuleList()
        for _ in range(self.n_critics):
            proj, pe, enc, ln, mlp = build()
            self.projs.append(proj); self.pes.append(pe); self.encs.append(enc)
            self.lns.append(ln); self.mlps.append(mlp)
        if self.n_critics == 2:
            self.proj1, self.pe1, self.enc1, self.ln1, self.mlp1 = self.projs[0], self.pes[0], self.encs[0], self.lns[0], self.mlps[0]
            self.proj2, self.pe2, self.enc2, self.ln2, self.mlp2 = self.projs[1], self.pes[1], self.encs[1], self.lns[1], self.mlps[1]

    def _q(self, proj, pe, enc, ln, mlp, obs, act):
        x = obs.unsqueeze(1) if obs.dim() == 2 else obs
        # Encoder in fp32 (see Actor._enc); the Q head runs in the outer precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = pe(proj(x.float()))
            h = enc(z)[:, -1, :]
            if ln is not None and not isinstance(ln, nn.Identity):
                h = ln(h)
        return mlp(torch.cat([h.to(act.dtype), act], dim=-1))

    def forward(self, obs, act):
        return tuple(self._q(self.projs[i], self.pes[i], self.encs[i], self.lns[i], self.mlps[i], obs, act)
                     for i in range(self.n_critics))


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
        dropout_rate=float(cfg.get("dropout_rate", 0.0)),
        n_critics=int(cfg.get("n_critics", 2)),
    )
    return c if device is None else c.to(device)
