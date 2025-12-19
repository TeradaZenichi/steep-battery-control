from __future__ import annotations

import json
import re
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import calendar

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from env.environment import SmartHomeEnv
from hp import HyperParameters
from model import ImitationLSTM
from opt import Teacher

DATASETS = ["WY", "CY"]
START_DATES = [
    "01/03/2000 00:00",
    "01/06/2000 00:00",
    "01/09/2000 00:00",
    "01/12/2000 00:00",
]
CONFIG_PATH = Path("data/parameters.json")
DAYS = 10
SOLVER_NAME = "gurobi"
MODEL_JSON = Path(__file__).with_name("model.json")
RESULTS_BASE = Path("Results")
MODEL_SUBDIR = "2_LSTM_IL"
TARIFF_OVERRIDE: str | None = None
SUMMARY_NAME = "evaluation_summary.json"
SUMMARY_TEXT_NAME = "evaluation_summary_lines.json"


def _month_range_label(start_date_str: str, days: int) -> str:
    try:
        start = datetime.strptime(start_date_str, "%d/%m/%Y %H:%M")
    except Exception:
        start = datetime.strptime(start_date_str, "%d/%m/%Y")
    end = start + timedelta(days=max(0, days - 1))
    s = calendar.month_abbr[start.month].lower()
    e = calendar.month_abbr[end.month].lower()
    return f"{s}-{e}"


def _load_tariff_label(config_path: Path, override: str | None) -> str:
    label = override
    if label is None:
        with open(config_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
        label = config["Grid"]["tariff_column"]
    text = str(label).strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return cleaned or "default_tariff"


def _snapshot_config(target_dir: Path) -> None:
    shutil.copy2(CONFIG_PATH, target_dir / "parameters.json")


TARIFF_LABEL = _load_tariff_label(CONFIG_PATH, TARIFF_OVERRIDE)
RESULTS_DIR = (RESULTS_BASE / TARIFF_LABEL / MODEL_SUBDIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "best.pt"
EXPORT_DIR = RESULTS_DIR


def _export_dir_for(run_label: str) -> Path:
    return RESULTS_DIR / run_label

REWARD_COMPONENT_KEYS = [
    "bess_degradation",
    "bess_penalty",
    "ev_degradation",
    "ev_penalty",
    "grid_cost",
    "grid_revenue",
    "grid_penalty",
]
DELTA_ATTR_NAME = "\u0394t"


def zero_reward_components() -> dict:
    return {key: 0.0 for key in REWARD_COMPONENT_KEYS}


def get_sim_delta_t(sim) -> float:
    unicode_dt = getattr(sim, DELTA_ATTR_NAME, None)
    if unicode_dt is not None:
        return float(unicode_dt)
    timestep_minutes = float(getattr(sim, "timestep", 60.0))
    return timestep_minutes / 60.0


def extract_reward_components(env: SmartHomeEnv) -> dict:
    delta_t = get_sim_delta_t(env.sim)
    return {
        "bess_degradation": -float(env.bess._costdeg) * delta_t,
        "bess_penalty": -float(env.bess._penalty) * delta_t,
        "ev_degradation": -float(env.ev._costdeg) * delta_t,
        "ev_penalty": -float(env.ev._penalty) * delta_t,
        "grid_cost": -float(env.grid._cost) * delta_t,
        "grid_revenue": float(env.grid._revenue) * delta_t,
        "grid_penalty": -float(env.grid._penalty) * delta_t,
    }


def accumulate_reward_components(total: dict, step_values: dict) -> None:
    for key, value in step_values.items():
        total[key] = total.get(key, 0.0) + float(value)


def format_component_summary(label: str, components: dict) -> str:
    parts = [f"{key}={components.get(key, 0.0):.3f}" for key in REWARD_COMPONENT_KEYS]
    return f"{label} components -> " + ", ".join(parts)


def resolve_state_mask(payload: dict | None) -> np.ndarray | None:
    if not payload:
        return None
    vector = payload.get("vector")
    if vector is None:
        return None
    mask_array = np.asarray(vector, dtype=bool)
    return mask_array if mask_array.size else None


def load_config_and_data(data_path: Path) -> Tuple[dict, pd.DataFrame]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
        config = json.load(fp)
    dataframe = pd.read_csv(data_path, sep=";")
    return config, dataframe


def solve_teacher(config: dict, dataframe: pd.DataFrame, start_date: str) -> pd.DataFrame:
    teacher = Teacher(config, dataframe=dataframe, start_date=start_date, days=DAYS)
    teacher.build(start_soc=0.5)
    teacher.solve(solver_name=SOLVER_NAME)
    return teacher.results_df()


def load_model(device: torch.device) -> Tuple[ImitationLSTM, HyperParameters, dict | None]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    hparams_payload = checkpoint.get("hparams")
    if hparams_payload is not None:
        hparams = HyperParameters.from_dict(hparams_payload)
    else:
        hparams = HyperParameters.from_json(MODEL_JSON)
    model = ImitationLSTM(hparams).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, hparams, checkpoint.get("state_mask")


def rollout_env(
    config: dict,
    dataframe: pd.DataFrame,
    policy_fn: Callable,
    label: str,
    state_mask: np.ndarray | None,
    start_date: str,
) -> Tuple[float, dict, pd.DataFrame, Path]:
    env = SmartHomeEnv(config, dataframe=dataframe, days=DAYS, state_mask=state_mask, start_date=start_date)
    obs, _ = env.reset()
    env.bess.reset(soc_init=0.5)
    total_reward = 0.0
    components = zero_reward_components()
    while not env.done:
        action = policy_fn(env, obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        accumulate_reward_components(components, extract_reward_components(env))
    local_export = _export_dir_for(label)
    export_path = local_export / f"{label}_env_replay.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    df = env.build_operation_dataframe()
    df.to_csv(export_path)
    return total_reward, components, df, export_path


def teacher_policy(results_df: pd.DataFrame) -> Callable:
    def _policy(env: SmartHomeEnv, _obs: np.ndarray) -> np.ndarray:
        ts = env.sim.current_datetime
        row = results_df.loc[ts, ["PBESS", "Pev", "chi_pv"]]
        return row.to_numpy(dtype=np.float32)

    return _policy


def lstm_policy(model: ImitationLSTM, device: torch.device, seq_len: int) -> Callable:
    window = deque(maxlen=seq_len)
    checked_dim = {"value": False}

    def _policy(env: SmartHomeEnv, obs: np.ndarray) -> np.ndarray:
        if not checked_dim["value"]:
            if obs.shape[0] != model.hparams.input_dim:
                raise ValueError(
                    f"Observation dimension {obs.shape[0]} does not match model input {model.hparams.input_dim}."
                )
            checked_dim["value"] = True
        window.append(obs.astype(np.float32))
        sequence = np.stack(list(window), axis=0) if window else np.zeros((0, obs.shape[0]), dtype=np.float32)
        if sequence.shape[0] < seq_len:
            if sequence.shape[0] == 0:
                pad_row = np.zeros((1, model.hparams.input_dim), dtype=np.float32)
            else:
                pad_row = sequence[0:1]
            pad = np.repeat(pad_row, seq_len - sequence.shape[0], axis=0)
            sequence = np.concatenate([pad, sequence], axis=0)
        tensor = torch.as_tensor(sequence, dtype=torch.float32, device=device).unsqueeze(0)
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


def export_summary(
    output_dir: Path,
    hparams: HyperParameters,
    mask_payload: dict | None,
    metrics: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "hyperparameters": hparams.to_dict(),
        "metrics": metrics,
        "state_mask": None,
    }
    if mask_payload:
        vector = np.asarray(mask_payload.get("vector", []), dtype=bool)
        labels = [str(label) for label in mask_payload.get("labels", [])]
        enabled = [label for label, flag in zip(labels, vector) if flag]
        disabled = [label for label, flag in zip(labels, vector) if not flag]
        summary["state_mask"] = {
            "spec": mask_payload.get("spec", "unknown"),
            "vector": vector.astype(int).tolist(),
            "labels": labels,
            "enabled_features": enabled,
            "disabled_features": disabled,
        }
    summary_path = output_dir / SUMMARY_NAME
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Evaluation summary saved to {summary_path}")
    return summary_path


def save_json_summary(output_dir: Path, lines: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / SUMMARY_TEXT_NAME
    with open(text_path, "w", encoding="utf-8") as fp:
        json.dump({"lines": lines}, fp, indent=2)
    return text_path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, hparams, mask_payload = load_model(device)
    teacher_mask = resolve_state_mask(mask_payload)

    for dataset in DATASETS:
        data_path = Path(f"data/Simulation_{dataset}_Fut_HP__PV5000-HB5000.csv")
        config, dataframe = load_config_and_data(data_path)
        for start_date in START_DATES:
            run_label = f"{dataset}-{_month_range_label(start_date, DAYS)}"
            run_dir = _export_dir_for(run_label)
            run_dir.mkdir(parents=True, exist_ok=True)
            _snapshot_config(run_dir)
            teacher_df = solve_teacher(config, dataframe, start_date)

            teacher_reward, teacher_components, teacher_env_df, teacher_csv = rollout_env(
                config,
                dataframe,
                teacher_policy(teacher_df),
                "teacher",
                teacher_mask,
                start_date,
            )

            lstm_reward, lstm_components, lstm_env_df, lstm_csv = rollout_env(
                config,
                dataframe,
                lstm_policy(model, device, seq_len=hparams.seq_len),
                "lstm",
                teacher_mask,
                start_date,
            )

            delta_reward = lstm_reward - teacher_reward
            teacher_comp_line = format_component_summary("Teacher", teacher_components)
            lstm_comp_line = format_component_summary("LSTM", lstm_components)
            summary_lines = [
                "Teacher results -> reward: {:.3f}, csv: {}".format(teacher_reward, teacher_csv),
                "LSTM results   -> reward: {:.3f}, csv: {}".format(lstm_reward, lstm_csv),
                "Reward delta (LSTM - Teacher): {:.3f}".format(delta_reward),
                teacher_comp_line,
                lstm_comp_line,
            ]
            for line in summary_lines:
                print(line)

            teacher_env_df = enrich_operation_df(teacher_env_df, config)
            lstm_env_df = enrich_operation_df(lstm_env_df, config)
            power_ylim = compute_power_limits(teacher_env_df, lstm_env_df)

            teacher_out = run_dir
            lstm_out = run_dir
            teacher_plot = plot_power_and_soc(teacher_env_df, "teacher", power_ylim, teacher_out)
            lstm_plot = plot_power_and_soc(lstm_env_df, "lstm", power_ylim, lstm_out)

            plot_line = f"Power/SOC plots saved: {teacher_plot} {lstm_plot}"
            print(plot_line)
            summary_lines.append(plot_line)
            summary_json_path = export_summary(
                lstm_out,
                hparams,
                mask_payload,
                {
                    "teacher_reward": teacher_reward,
                    "lstm_reward": lstm_reward,
                    "delta_reward": delta_reward,
                    "teacher_components": teacher_components,
                    "lstm_components": lstm_components,
                },
            )
            summary_lines.append(f"Evaluation summary JSON: {summary_json_path}")
            text_summary_path = save_json_summary(lstm_out, summary_lines)
            print(f"JSON summary saved to {text_summary_path}")


if __name__ == "__main__":
    main()
