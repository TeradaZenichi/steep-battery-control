from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict


@dataclass
class HyperParameters:
    """Hyperparameters for the LSTM imitation learner."""

    input_dim: int
    output_dim: int
    hidden_size: int = 256
    num_layers: int = 2
    seq_len: int = 12
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 15
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
