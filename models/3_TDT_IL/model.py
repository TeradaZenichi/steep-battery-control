from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

try:  # pragma: no cover
    from .hp import HyperParameters
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from hp import HyperParameters


class TimeDecisionTransformer(nn.Module):
    def __init__(self, hparams: HyperParameters):
        super().__init__()
        self.hparams = hparams
        d_model = hparams.d_model
        self.state_encoder = nn.Linear(hparams.input_dim, d_model)
        self.action_encoder = nn.Linear(hparams.output_dim, d_model)
        self.rtg_encoder = nn.Linear(1, d_model)
        self.time_embedding = nn.Embedding(max(1, hparams.max_timestep), d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=hparams.n_heads,
            dim_feedforward=hparams.dim_feedforward,
            dropout=hparams.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=hparams.n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(hparams.dropout)
        self.head = nn.Linear(d_model, hparams.output_dim)

    @classmethod
    def from_json(cls, json_path: str | Path) -> "TimeDecisionTransformer":
        hparams = HyperParameters.from_json(json_path)
        return cls(hparams)

    def _shift_actions(self, actions: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(actions.size(0), 1, actions.size(2), device=actions.device, dtype=actions.dtype)
        shifted = torch.cat([pad, actions[:, :-1, :]], dim=1)
        return shifted

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_tokens = self._shift_actions(actions)
        state_embed = self.state_encoder(states)
        action_embed = self.action_encoder(action_tokens)
        scale = max(self.hparams.rtg_scale, 1e-6)
        rtg_norm = returns_to_go.unsqueeze(-1) / scale
        rtg_embed = self.rtg_encoder(rtg_norm)
        capped_steps = torch.clamp(timesteps.long(), min=0, max=self.hparams.max_timestep - 1)
        time_embed = self.time_embedding(capped_steps)
        tokens = state_embed + action_embed + rtg_embed + time_embed
        tokens = self.dropout(tokens)
        if padding_mask is not None:
            padding_mask = padding_mask.bool()
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)
        return self.head(encoded)


if __name__ == "__main__":  # pragma: no cover
    cfg = Path(__file__).with_name("model.json")
    model = TimeDecisionTransformer.from_json(cfg)
    batch = 2
    seq_len = model.hparams.seq_len
    dummy_states = torch.randn(batch, seq_len, model.hparams.input_dim)
    dummy_actions = torch.randn(batch, seq_len, model.hparams.output_dim)
    dummy_returns = torch.randn(batch, seq_len)
    dummy_steps = torch.arange(seq_len).repeat(batch, 1)
    out = model(dummy_states, dummy_actions, dummy_returns, dummy_steps)
    print(out.shape)
