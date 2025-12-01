from typing import Callable, Tuple
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

if str(Path(__file__).resolve().parent) not in sys.path:
	sys.path.append(str(Path(__file__).resolve().parent))

from env.environment import SmartHomeEnv
from hp import HyperParameters  
from model import ImitationMLP    
from opt import Teacher  



# Evaluation configuration
CONFIG_PATH = Path("data/parameters.json")
DATA_PATH = Path("data/Simulation_CY_Fut_HP__PV5000-HB5000.csv")
START_DATE = "08/01/2000 00:00"
DAYS = 5
SOLVER_NAME = "gurobi"
MODEL_JSON = Path(__file__).with_name("model.json")
CHECKPOINT_PATH = Path("Results/1_MLP_IL/best.pt")
EXPORT_DIR = CHECKPOINT_PATH.parent


def load_config_and_data() -> Tuple[dict, pd.DataFrame]:
	with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
		config = json.load(fp)
	dataframe = pd.read_csv(DATA_PATH, sep=";")
	return config, dataframe


def solve_teacher(config: dict, dataframe: pd.DataFrame) -> pd.DataFrame:
	teacher = Teacher(config, dataframe=dataframe, start_date=START_DATE, days=DAYS)
	teacher.build(start_soc=0.5)
	teacher.solve(solver_name=SOLVER_NAME)
	return teacher.results_df()


def load_model(device: torch.device) -> ImitationMLP:
	hparams = HyperParameters.from_json(MODEL_JSON)
	checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
	model = ImitationMLP(hparams).to(device)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	return model


def rollout_env(config: dict, dataframe: pd.DataFrame, policy_fn: Callable, label: str) -> Tuple[float, pd.DataFrame, Path]:
	env = SmartHomeEnv(config, dataframe=dataframe, days=DAYS, state_mask=None, start_date=START_DATE)
	obs, _ = env.reset()
	env.bess.reset(soc_init=0.5)
	total_reward = 0.0
	while not env.done:
		action = policy_fn(env, obs)
		obs, reward, done, _ = env.step(action)
		total_reward += reward
	export_path = EXPORT_DIR / f"{label}_env_replay.csv"
	export_path.parent.mkdir(parents=True, exist_ok=True)
	df = env.build_operation_dataframe()
	df.to_csv(export_path)
	return total_reward, df, export_path


def teacher_policy(results_df: pd.DataFrame) -> Callable:
	def _policy(env: SmartHomeEnv, _obs: np.ndarray) -> np.ndarray:
		ts = env.sim.current_datetime
		row = results_df.loc[ts, ["PBESS", "Pev", "chi_pv"]]
		return row.to_numpy(dtype=np.float32)

	return _policy


def mlp_policy(model: ImitationMLP, device: torch.device) -> Callable:
	first_call = {"checked": False}

	def _policy(env: SmartHomeEnv, obs: np.ndarray) -> np.ndarray:
		if not first_call["checked"]:
			if obs.shape[0] != model.hparams.input_dim:
				raise ValueError(
					f"Observation dimension {obs.shape[0]} does not match model input {model.hparams.input_dim}."
				)
			first_call["checked"] = True
		tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
		with torch.no_grad():
			action = model(tensor).squeeze(0).cpu().numpy()
		low = env.action_space.low
		high = env.action_space.high
		return np.clip(action, low, high)

	return _policy


def enrich_operation_df(df: pd.DataFrame, config: dict) -> pd.DataFrame:
	enriched = df.copy()
	if "ppv_curtailed" in enriched:
		enriched["Ppv_available"] = enriched["Ppv"].fillna(0.0) + enriched["ppv_curtailed"].fillna(0.0)
	elif "produced_electricity_rate_W" in enriched:
		enriched["Ppv_available"] = enriched["produced_electricity_rate_W"].astype(float) / 1000.0
	else:
		enriched["Ppv_available"] = enriched["Ppv"].fillna(0.0)

	bess_emax = float(config.get("BESS", {}).get("Emax", 1.0))
	ev_emax = float(config.get("EV", {}).get("Emax", 1.0))
	enriched["soc_bess"] = enriched.get("EBESS", 0.0) / max(bess_emax, 1e-6)
	enriched["soc_ev"] = enriched.get("Eev", 0.0) / max(ev_emax, 1e-6)
	return enriched


def compute_power_limits(df_teacher: pd.DataFrame, df_mlp: pd.DataFrame) -> Tuple[float, float]:
	power_cols = ["Pgrid", "PBESS", "Load", "Ppv", "Ppv_available"]
	combined = []
	for df in (df_teacher, df_mlp):
		combined.append(df[power_cols].to_numpy(dtype=float).flatten())
	series = np.concatenate(combined)
	finite = series[np.isfinite(series)]
	if finite.size == 0:
		return -1.0, 1.0
	ymin = float(finite.min())
	ymax = float(finite.max())
	if np.isclose(ymin, ymax):
		margin = max(1.0, abs(ymin) * 0.1)
		return ymin - margin, ymax + margin
	span = ymax - ymin
	padding = 0.05 * span
	return ymin - padding, ymax + padding


def plot_power_and_soc(df: pd.DataFrame, label: str, ylim: Tuple[float, float], output_dir: Path) -> Path:
	fname = output_dir / f"{label}_power_soc.pdf"
	fig, (ax_power, ax_soc) = plt.subplots(
		2,
		1,
		figsize=(10, 6),
		sharex=True,
		gridspec_kw={"height_ratios": [2.5, 1.0]},
	)
	idx = df.index

	ax_power.plot(idx, df["Load"], color="black", marker="o", markersize=2, linewidth=1.0, label="Load")
	ax_power.plot(idx, df["Ppv"], color="#f2c94c", linewidth=1.5, label="PV")
	if "Ppv_available" in df:
		ax_power.fill_between(
			idx,
			df["Ppv"],
			df["Ppv_available"],
			color="#f2c94c",
			alpha=0.3,
			label="PV curtailment",
		)
	ax_power.plot(idx, df["Pgrid"], color="#2f80ed", linewidth=1.5, label="Grid")
	ax_power.bar(idx, df["PBESS"], width=0.01, color="#eb5757", label="BESS")
	ev_col = "Pev" if "Pev" in df.columns else "PEV"
	if ev_col in df:
		ev_series = df[ev_col]
		ax_power.bar(idx, ev_series, width=0.01, color="#27ae60", label="EV")
		ax_power.plot(idx, ev_series, color="#219653", linewidth=1.0, linestyle="--", label="EV power")
	ax_power.set_ylabel("Power (kW)")
	ax_power.set_ylim(ylim)
	ax_power.legend(loc="lower right")
	ax_power.grid(True, alpha=0.3)

	ax_soc.plot(idx, df["soc_bess"], label="SOC_BESS")
	ax_soc.plot(idx, df["soc_ev"], label="SOC_EV")
	ax_soc.set_ylabel("SOC")
	ax_soc.set_xlabel("Time")
	ax_soc.set_ylim(0.0, 1.05)
	ax_soc.legend(loc="upper right")
	ax_soc.grid(True, alpha=0.3)

	fig.autofmt_xdate()
	fig.tight_layout()
	fig.savefig(fname)
	plt.close(fig)
	return fname


def main() -> None:
	config, dataframe = load_config_and_data()
	teacher_df = solve_teacher(config, dataframe)

	teacher_reward, teacher_env_df, teacher_csv = rollout_env(config, dataframe, teacher_policy(teacher_df), "teacher")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = load_model(device)
	mlp_reward, mlp_env_df, mlp_csv = rollout_env(config, dataframe, mlp_policy(model, device), "mlp")

	delta_reward = mlp_reward - teacher_reward
	print("Teacher results -> reward: {:.3f}, csv: {}".format(teacher_reward, teacher_csv))
	print("MLP results     -> reward: {:.3f}, csv: {}".format(mlp_reward, mlp_csv))
	print("Reward delta (MLP - Teacher): {:.3f}".format(delta_reward))

	teacher_env_df = enrich_operation_df(teacher_env_df, config)
	mlp_env_df = enrich_operation_df(mlp_env_df, config)
	power_ylim = compute_power_limits(teacher_env_df, mlp_env_df)

	EXPORT_DIR.mkdir(parents=True, exist_ok=True)
	teacher_plot = plot_power_and_soc(teacher_env_df, "teacher", power_ylim, EXPORT_DIR)
	mlp_plot = plot_power_and_soc(mlp_env_df, "mlp", power_ylim, EXPORT_DIR)

	print("Power/SOC plots saved:", teacher_plot, mlp_plot)


if __name__ == "__main__":
	main()
