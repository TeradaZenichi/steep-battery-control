from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class HyperParameters:
	"""Container for model and training settings used by the imitation MLP."""

	input_dim: int
	output_dim: int
	hidden_sizes: Sequence[int]
	dropout: float = 0.0
	activation: str = "relu"
	learning_rate: float = 1e-3
	weight_decay: float = 0.0
	batch_size: int = 256
	epochs: int = 10
	seed: int = 42
	val_split: float = 0.2
	patience: int = 0
	improvement_tol: float = 1e-5

	def to_dict(self) -> Dict[str, object]:
		return asdict(self)

	@classmethod
	def from_dict(cls, data: Dict[str, object]) -> "HyperParameters":
		field_names = {f.name for f in fields(cls)}
		filtered = {key: value for key, value in data.items() if key in field_names}
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
