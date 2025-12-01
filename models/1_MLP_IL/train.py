from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

if str(Path(__file__).resolve().parent) not in sys.path:
	sys.path.append(str(Path(__file__).resolve().parent))

from opt import Teacher  # type: ignore  # pylint: disable=wrong-import-position
from hp import HyperParameters  # type: ignore  # pylint: disable=wrong-import-position
from model import ImitationMLP  # type: ignore  # pylint: disable=wrong-import-position


# Training configuration (adjust as needed)
CONFIG_PATH = Path("data/parameters.json")
DATA_PATH = Path("data/Simulation_CY_Cur_HP__PV5000-HB5000.csv")
START_DATE = "08/01/2000 00:00"
DAYS = 100
SOLVER_NAME = "gurobi"
MODEL_JSON = Path(__file__).with_name("model.json")
SAVE_PATH = Path("Results/1_MLP_IL/best.pt")

TRAIN_SCENARIOS: List[Dict[str, object]] = [
	{
		"name": "base",
		"config_path": CONFIG_PATH,
		"data_path": Path("data/Simulation_CY_Cur_HP__PV5000-HB5000.csv"),
		"start_date": START_DATE,
		"days": DAYS,
		"solver_name": SOLVER_NAME,
		"start_soc": 0.5,
	},
	{
		"name": "extended",
		"config_path": CONFIG_PATH,
		"data_path": Path("data/Simulation_WY_Cur_HP__PV5000-HB5000.csv"),
		"start_date": START_DATE,
		"days": DAYS,
		"solver_name": SOLVER_NAME,
        "start_soc": 0.5,
    }
]


def _load_config(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as fp:
		return json.load(fp)


def build_teacher_dataset(scenarios: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
	feature_chunks: List[np.ndarray] = []
	target_chunks: List[np.ndarray] = []
	for idx, scenario in enumerate(scenarios, start=1):
		config_path = Path(scenario.get("config_path", CONFIG_PATH))
		data_path = Path(scenario.get("data_path", DATA_PATH))
		start_date = str(scenario.get("start_date", START_DATE))
		days = int(scenario.get("days", DAYS))
		solver_name = str(scenario.get("solver_name", SOLVER_NAME))
		start_soc = float(scenario.get("start_soc", 0.5))
		label = str(scenario.get("name", f"scenario_{idx}"))

		print(f"Running teacher scenario '{label}' with data={data_path.name}, days={days}, start_soc={start_soc}")
		config = _load_config(config_path)
		dataframe = pd.read_csv(data_path, sep=";")
		teacher = Teacher(config, dataframe=dataframe, start_date=start_date, days=days)
		teacher.build(start_soc=start_soc)
		teacher.solve(solver_name=solver_name)

		states, _ = teacher.get_masked_observations(state_mask=None)
		actions_df = teacher.results_df()[["PBESS", "Pev", "chi_pv"]]
		feature_chunks.append(states.astype(np.float32))
		target_chunks.append(actions_df.to_numpy(dtype=np.float32))

	features = np.concatenate(feature_chunks, axis=0)
	targets = np.concatenate(target_chunks, axis=0)
	print(f"Aggregated dataset shape -> features: {features.shape}, targets: {targets.shape}")
	return features, targets


def make_loaders(features: np.ndarray, targets: np.ndarray, batch_size: int, val_split: float, seed: int) -> Tuple[DataLoader, DataLoader]:
	total = len(features)
	assert total == len(targets), "Feature/target length mismatch"
	rng = np.random.default_rng(seed)
	indices = rng.permutation(total)
	split = int(total * (1 - val_split))
	train_idx, val_idx = indices[:split], indices[split:]
	if len(val_idx) == 0:
		val_idx = train_idx

	train_ds = TensorDataset(
		torch.from_numpy(features[train_idx]),
		torch.from_numpy(targets[train_idx]),
	)
	val_ds = TensorDataset(
		torch.from_numpy(features[val_idx]),
		torch.from_numpy(targets[val_idx]),
	)
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size)
	return train_loader, val_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
	model.train()
	running_loss = 0.0
	for xb, yb in loader:
		xb = xb.to(device)
		yb = yb.to(device)
		optimizer.zero_grad()
		preds = model(xb)
		loss = criterion(preds, yb)
		loss.backward()
		optimizer.step()
		running_loss += loss.item() * xb.size(0)
	return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
	model.eval()
	total_loss = 0.0
	with torch.no_grad():
		for xb, yb in loader:
			xb = xb.to(device)
			yb = yb.to(device)
			preds = model(xb)
			loss = criterion(preds, yb)
			total_loss += loss.item() * xb.size(0)
	return total_loss / len(loader.dataset)



def main() -> None:
	hparams = HyperParameters.from_json(MODEL_JSON)

	features, targets = build_teacher_dataset(TRAIN_SCENARIOS)
	state_dim = features.shape[1]
	if state_dim != hparams.input_dim:
		print(f"Detected state_dim={state_dim}; updating hyperparameters (was {hparams.input_dim}).")
		hparams.input_dim = state_dim
	if targets.shape[1] != hparams.output_dim:
		raise ValueError(f"Target dimension {targets.shape[1]} does not match HyperParameters.output_dim={hparams.output_dim}")

	val_split = getattr(hparams, "val_split", 0.2)
	train_loader, val_loader = make_loaders(features, targets, hparams.batch_size, val_split, hparams.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = ImitationMLP(hparams).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
	criterion = nn.MSELoss()
	patience = getattr(hparams, "patience", 0)
	improvement_tol = getattr(hparams, "improvement_tol", 1e-5)
	best_val_loss = float("inf")
	best_state_dict = None
	patience_counter = 0
	best_epoch = 0
	train_history: list[float] = []
	val_history: list[float] = []

	for epoch in range(1, hparams.epochs + 1):
		train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
		val_loss = evaluate(model, val_loader, criterion, device)
		print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
		train_history.append(train_loss)
		val_history.append(val_loss)

		if val_loss + improvement_tol < best_val_loss:
			best_val_loss = val_loss
			best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
			patience_counter = 0
			best_epoch = epoch
		else:
			patience_counter += 1
			if patience and patience_counter >= patience:
				print(f"Early stopping triggered after {epoch} epochs (no val improvement in {patience} epochs).")
				break

	save_path = SAVE_PATH
	save_path.parent.mkdir(parents=True, exist_ok=True)
	state_to_save = best_state_dict if best_state_dict is not None else model.state_dict()
	torch.save({"state_dict": state_to_save, "hparams": hparams.to_dict(), "best_val_loss": best_val_loss}, save_path)
	print(f"Model saved to {save_path}")
	graphs_dir = SAVE_PATH.parent
	graphs_dir.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=(8, 4))
	ax.plot(range(1, len(train_history) + 1), train_history, label="train_loss")
	ax.plot(range(1, len(val_history) + 1), val_history, label="val_loss")
	ax.axvline(best_epoch, color="green", linestyle="--", label=f"best_val@{best_epoch}")
	ax.set_xlabel("epoch")
	ax.set_ylabel("mse loss")
	ax.set_title("MLP imitation training")
	ax.legend()
	ax.grid(True, alpha=0.3)
	graph_path = graphs_dir / "training_curves.png"
	fig.tight_layout()
	fig.savefig(graph_path)
	plt.close(fig)
	print(f"Training curves saved to {graph_path}")
	print(f"Best validation loss {best_val_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
	main()
