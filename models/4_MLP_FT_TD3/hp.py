from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Sequence


@dataclass
class HyperParameters:
    """Hyperparameters for the MLP imitation learner."""

    input_dim: int
    output_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    dropout: float = 0.0
    activation: str = "relu"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 30
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
        # hidden_sizes may come as list from JSON
        if "hidden_sizes" in filtered and isinstance(filtered["hidden_sizes"], list):
            filtered["hidden_sizes"] = tuple(int(x) for x in filtered["hidden_sizes"])
        if "hidden_sizes" in filtered and isinstance(filtered["hidden_sizes"], Sequence) and not isinstance(filtered["hidden_sizes"], tuple):
            filtered["hidden_sizes"] = tuple(filtered["hidden_sizes"])
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
