from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict


@dataclass
class HyperParameters:
    """Hyperparameters for the Time Decision Transformer imitation learner."""

    input_dim: int
    output_dim: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dim_feedforward: int = 1024
    seq_len: int = 32
    dropout: float = 0.1
    max_timestep: int = 288
    rtg_scale: float = 1.0
    target_return: float = 0.0
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 20
    seed: int = 42
    val_split: float = 0.2
    patience: int = 0
    improvement_tol: float = 1e-5

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "HyperParameters":
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> "HyperParameters":
        import json

        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return cls.from_dict(payload)

    def to_json(self, path: str | Path) -> None:
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2)
