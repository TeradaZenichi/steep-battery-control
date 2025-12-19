from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:  # pragma: no cover - convenience for relative imports
    from .hp import HyperParameters
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from hp import HyperParameters


class ImitationLSTM(nn.Module):
    def __init__(self, hparams: HyperParameters):
        super().__init__()
        self.hparams = hparams
        lstm_dropout = float(hparams.dropout) if hparams.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hparams.input_dim,
            hidden_size=hparams.hidden_size,
            num_layers=hparams.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.post_dropout = nn.Dropout(float(hparams.dropout)) if hparams.dropout > 0 else nn.Identity()
        self.head = nn.Linear(hparams.hidden_size, hparams.output_dim)

    @classmethod
    def from_json(cls, json_path: str | os.PathLike[str]) -> "ImitationLSTM":
        hparams = HyperParameters.from_json(json_path)
        return cls(hparams)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.dim() == 2:
            sequences = sequences.unsqueeze(0)
        outputs, _ = self.lstm(sequences)
        last_hidden = outputs[:, -1, :]
        return self.head(self.post_dropout(last_hidden))


if __name__ == "__main__":  # pragma: no cover
    config_path = Path(__file__).with_name("model.json")
    hparams = HyperParameters.from_json(config_path)
    model = ImitationLSTM(hparams)
    dummy = torch.randn(4, hparams.seq_len, hparams.input_dim)
    out = model(dummy)
    print("Input:", dummy.shape)
    print("Output:", out.shape)
