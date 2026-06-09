"""Lightweight TCN SAC model.

Uses dilated causal convolutions over the observation history. This is a fast
sequence baseline for BESS/EV control without recurrent state or full
Transformer blocks.
"""
import torch
import torch.nn as nn

from models.GRU.model import Actor as BaseActor


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (int(kernel_size) - 1) * int(dilation)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.net(x)
        r = x if self.down is None else self.down(x)
        return y + r


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, kernel_size=3, dropout=0.05):
        super().__init__()
        layers = []
        in_ch = int(input_dim)
        for i in range(int(num_layers)):
            layers.append(TemporalBlock(
                in_ch, int(hidden_dim), kernel_size,
                dilation=2 ** i, dropout=dropout,
            ))
            in_ch = int(hidden_dim)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        y = self.net(x.transpose(1, 2))
        return y[:, :, -1]


class Actor(BaseActor):
    def __init__(self, input_dim, hidden_dim, num_layers, head_dim, kernel_size=3,
                 dropout=0.05, **kwargs):
        super().__init__(input_dim, hidden_dim, 1, head_dim, **kwargs)
        del self.gru
        self.tcn = TemporalEncoder(input_dim, hidden_dim, num_layers, kernel_size, dropout)

    def _enc(self, x):
        h = self.tcn(x)
        return self.ln(h) if self.ln is not None else h


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, head_dim,
                 kernel_size=3, dropout=0.05, use_layer_norm=False, dropout_rate=0.0, n_critics=2):
        super().__init__()
        # `dropout` is the architectural temporal-convolution dropout; `dropout_rate`
        # is the DroQ head dropout, kept active at the Bellman target (see trainer).
        pq = float(dropout_rate)

        def build():
            enc = TemporalEncoder(state_dim, hidden_dim, num_layers, kernel_size, dropout)
            ln = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            head = [nn.Linear(hidden_dim + action_dim, hidden_dim), nn.ReLU()]
            if pq > 0: head.append(nn.Dropout(pq))
            head += [nn.Linear(hidden_dim, head_dim), nn.ReLU()]
            if pq > 0: head.append(nn.Dropout(pq))
            head += [nn.Linear(head_dim, 1)]
            mlp = nn.Sequential(*head)
            return enc, ln, mlp

        self.n_critics = int(n_critics)
        self.encs = nn.ModuleList(); self.lns = nn.ModuleList(); self.mlps = nn.ModuleList()
        for _ in range(self.n_critics):
            enc, ln, mlp = build()
            self.encs.append(enc); self.lns.append(ln); self.mlps.append(mlp)
        if self.n_critics == 2:
            self.enc1, self.ln1, self.mlp1 = self.encs[0], self.lns[0], self.mlps[0]
            self.enc2, self.ln2, self.mlp2 = self.encs[1], self.lns[1], self.mlps[1]

    def _q(self, enc, ln, mlp, obs, act):
        h = enc(obs)
        if ln is not None and not isinstance(ln, nn.Identity):
            h = ln(h)
        return mlp(torch.cat([h, act], dim=-1))

    def forward(self, obs, act):
        return tuple(self._q(self.encs[i], self.lns[i], self.mlps[i], obs, act)
                     for i in range(self.n_critics))


def load_actor(cfg, device=None):
    a = Actor(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg.get("hidden_dim", 192),
        num_layers=cfg.get("num_layers", 3),
        head_dim=cfg.get("head_dim", 64),
        kernel_size=cfg.get("kernel_size", 3),
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
        num_layers=int(cfg.get("num_layers", 3)),
        head_dim=int(cfg.get("head_dim", 64)),
        kernel_size=int(cfg.get("kernel_size", 3)),
        dropout=float(cfg.get("dropout", 0.05)),
        use_layer_norm=bool(cfg.get("use_layer_norm", False)),
        dropout_rate=float(cfg.get("dropout_rate", 0.0)),
        n_critics=int(cfg.get("n_critics", 2)),
    )
    return c if device is None else c.to(device)
