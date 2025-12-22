from __future__ import annotations

import calendar
from datetime import datetime, timedelta
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from env.environment import SmartHomeEnv  # type: ignore
from opt import Teacher  # type: ignore
from hp import HyperParameters  # type: ignore
from model import TimeDecisionTransformer  # type: ignore

CONFIG_PATH = Path("data/parameters.json")
RESULTS_BASE = Path("Results")
MODEL_SUBDIR = "6_TDT_FT_TD3"
SUMMARY_NAME = "evaluation_summary.json"
SUMMARY_TEXT_NAME = "evaluation_summary_lines.json"
TARIFF_OVERRIDE: str | None = None
TARIFFS: list[str] | None = ["tar_s", "tar_w", "tar_sw", "tar_flat", "tar_tou"]
DATASETS = ["WY", "CY"]
RUN_SCHEDULE: list[tuple[str, int]] = [("01/01/2000 00:00", 365)]
SOLVER_NAME = "gurobi"
MODEL_JSON = Path(__file__).with_name("model.json")


def _load_tariff_label(config_path: Path, override: str | None) -> str:
    label = override
    if label is None:
        with open(config_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
        label = config["Grid"].get("tariff_column")
    text = str(label).strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return cleaned or "default_tariff"


TARIFF_LABEL = _load_tariff_label(CONFIG_PATH, TARIFF_OVERRIDE)
RESULTS_ROOT = RESULTS_BASE / TARIFF_LABEL / MODEL_SUBDIR / "train"


def set_tariff_dirs(tariff_label: str) -> None:
    global TARIFF_LABEL, RESULTS_ROOT
    TARIFF_LABEL = tariff_label
    RESULTS_ROOT = RESULTS_BASE / TARIFF_LABEL / MODEL_SUBDIR / "train"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def resolve_state_mask(payload) -> np.ndarray | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        vector = payload.get("vector")
    else:
        vector = payload
    if vector is None:
        return None
    mask_array = np.asarray(vector, dtype=bool)
    return mask_array if mask_array.size else None


def load_config_and_data(config_path: Path, data_path: Path) -> Tuple[dict, pd.DataFrame]:
    with open(config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)
    dataframe = pd.read_csv(data_path, sep=";")
    return config, dataframe


REWARD_COMPONENT_KEYS = [
    "bess_degradation",
    "bess_penalty",
    "ev_degradation",
    "ev_penalty",
    "grid_cost",
    "grid_revenue",
    "grid_penalty",
]


def zero_reward_components() -> dict:
    return {key: 0.0 for key in REWARD_COMPONENT_KEYS}


def get_sim_delta_t(sim) -> float:
    if hasattr(sim, "dt"):
        return float(sim.dt)
    unicode_dt = getattr(sim, "\u0394t", None)
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


class Actor(torch.nn.Module):
    def __init__(self, hparams: HyperParameters, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.core = TimeDecisionTransformer(hparams)
        self.seq_len = hparams.seq_len
        self.action_dim = hparams.output_dim
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = state_seq.shape
        device = state_seq.device
        action_tokens = torch.zeros(bs, seq_len, self.action_dim, device=device, dtype=state_seq.dtype)
        returns_to_go = torch.zeros(bs, seq_len, device=device, dtype=state_seq.dtype)
        timesteps = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)
        pad_mask = torch.zeros(bs, seq_len, device=device, dtype=torch.bool)
        preds = self.core(state_seq, action_tokens, returns_to_go, timesteps, padding_mask=pad_mask)
        raw = preds[:, -1, :]
        return torch.min(torch.max(raw, self.action_low), self.action_high)


def actor_policy(actor: Actor, device: torch.device, seq_len: int) -> Callable:
    def _policy(env: SmartHomeEnv, window: np.ndarray) -> np.ndarray:
        window_t = torch.as_tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = actor(window_t).squeeze(0).cpu().numpy()
        return np.clip(action, env.action_space.low, env.action_space.high)

    return _policy


def teacher_policy(results_df: pd.DataFrame) -> Callable:
    def _policy(env: SmartHomeEnv, _obs: np.ndarray) -> np.ndarray:
        ts = env.sim.current_datetime
        row = results_df.loc[ts, ["PBESS", "Pev", "chi_pv"]]
        return row.to_numpy(dtype=np.float32)

    return _policy


def solve_teacher(config: dict, dataframe: pd.DataFrame, start_date: str, days: int) -> pd.DataFrame:
    teacher = Teacher(config, dataframe=dataframe, start_date=start_date, days=days)
    teacher.build(start_soc=0.5)
    teacher.solve(solver_name=SOLVER_NAME)
    return teacher.results_df()


def init_window(obs: np.ndarray, seq_len: int) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    return np.tile(obs, (seq_len, 1))


def roll_window(window: np.ndarray, new_obs: np.ndarray) -> np.ndarray:
    return np.concatenate([window[1:], np.asarray(new_obs, dtype=np.float32)[None, :]], axis=0)


def rollout_env(
    config: dict,
    dataframe: pd.DataFrame,
    policy_fn: Callable,
    label: str,
    state_mask: np.ndarray | None,
    start_date: str,
    days: int,
    seq_len: int,
) -> Tuple[float, dict, pd.DataFrame, Path]:
    env = SmartHomeEnv(config, dataframe=dataframe, days=days, state_mask=state_mask, start_date=start_date)
    obs, _ = env.reset()
    env.bess.reset(soc_init=0.5)
    window = init_window(obs, seq_len)
    total_reward = 0.0
    components = zero_reward_components()
    if hasattr(policy_fn, "reset"):
        policy_fn.reset()
    while not env.done:
        action = policy_fn(env, window)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        accumulate_reward_components(components, extract_reward_components(env))
        window = roll_window(window, obs)
        if hasattr(policy_fn, "on_step"):
            policy_fn.on_step(action, reward)
    export_path = Path("unused")
    df = env.build_operation_dataframe()
    return total_reward, components, df, export_path


def _month_range_label(start_date_str: str, days: int) -> str:
    try:
        start = datetime.strptime(start_date_str, "%d/%m/%Y %H:%M")
    except Exception:
        start = datetime.strptime(start_date_str, "%d/%m/%Y")
    end = start + timedelta(days=max(0, days - 1))
    s = calendar.month_abbr[start.month].lower()
    e = calendar.month_abbr[end.month].lower()
    return f"{s}-{e}"


def _export_dir_for(base_dir: Path, run_label: str) -> Path:
    return base_dir / "eval" / run_label


def enrich_operation_df(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    enriched = df.copy()
    if "ppv_curtailed" in enriched:
        enriched["Ppv_available"] = enriched["Ppv"].fillna(0.0) + enriched["ppv_curtailed"].fillna(0.0)
    elif "produced_electricity_rate_W" in enriched:
        enriched["Ppv_available"] = enriched["produced_electricity_rate_W"].astype(float) / 1000.0
    else:
        enriched["Ppv_available"] = enriched.get("Ppv", 0.0)

    bess_emax = float(config.get("BESS", {}).get("Emax", 1.0))
    ev_emax = float(config.get("EV", {}).get("Emax", 1.0))
    enriched["soc_bess"] = enriched.get("EBESS", 0.0) / max(bess_emax, 1e-6)
    enriched["soc_ev"] = enriched.get("Eev", 0.0) / max(ev_emax, 1e-6)
    return enriched


def compute_power_limits(df_teacher: pd.DataFrame, df_actor: pd.DataFrame) -> Tuple[float, float]:
    power_cols = ["Pgrid", "PBESS", "Load", "Ppv", "Ppv_available"]
    combined = []
    for df in (df_teacher, df_actor):
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


def export_summary(output_dir: Path, hparams: HyperParameters, mask_payload, metrics: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "hyperparameters": hparams.to_dict(),
        "metrics": metrics,
        "state_mask": None,
    }
    if mask_payload:
        if isinstance(mask_payload, dict):
            vector_raw = mask_payload.get("vector", [])
            labels = [str(label) for label in mask_payload.get("labels", [])]
            spec = mask_payload.get("spec", "unknown")
        else:
            vector_raw = mask_payload
            labels = []
            spec = "unknown"
        vector = np.asarray(vector_raw, dtype=bool)
        enabled = [label for label, flag in zip(labels, vector) if flag]
        disabled = [label for label, flag in zip(labels, vector) if not flag]
        summary["state_mask"] = {
            "spec": spec,
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
    cfg_config = CONFIG_PATH
    base_tariff = _load_tariff_label(cfg_config, TARIFF_OVERRIDE)
    tariffs = [str(t) for t in (TARIFFS if TARIFFS else [base_tariff])]

    for tariff in tariffs:
        set_tariff_dirs(tariff)
        actor_path = RESULTS_ROOT / "actor_finetuned.pt"

        if not actor_path.exists():
            raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")

        payload = torch.load(actor_path, map_location=device)
        hparams = HyperParameters.from_dict(payload["hparams"])
        mask_payload = payload.get("state_mask")
        state_mask = resolve_state_mask(mask_payload)
        state_dict = payload["actor_state_dict"]
        seq_len = int(getattr(hparams, "seq_len", 1))

        for dataset in DATASETS:
            for start_date, days in RUN_SCHEDULE:
                run_label = f"{dataset}-{_month_range_label(start_date, days)}"
                print(f"=== Evaluating {tariff} | {run_label} ===")
                data_path = Path(f"data/Simulation_{dataset}_Fut_HP__PV5000-HB5000.csv")
                config, dataframe = load_config_and_data(cfg_config, data_path)

                run_dir = _export_dir_for(RESULTS_ROOT, run_label)
                run_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cfg_config, run_dir / "parameters.json")

                teacher_df = solve_teacher(config, dataframe, start_date, days)

                env = SmartHomeEnv(config, dataframe=dataframe, days=days, state_mask=state_mask, start_date=start_date)
                actor = Actor(hparams, env.action_space.low, env.action_space.high).to(device)
                missing, unexpected = actor.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"Warning: missing keys when loading actor: {missing}")
                if unexpected:
                    print(f"Warning: unexpected keys when loading actor: {unexpected}")
                actor.eval()

                teacher_reward, teacher_components, teacher_env_df, _ = rollout_env(
                    config,
                    dataframe,
                    teacher_policy(teacher_df),
                    "teacher",
                    state_mask,
                    start_date,
                    days,
                    seq_len,
                )

                actor_reward, actor_components, actor_env_df, _ = rollout_env(
                    config,
                    dataframe,
                    actor_policy(actor, device, seq_len),
                    "actor",
                    state_mask,
                    start_date,
                    days,
                    seq_len,
                )

                teacher_env_df = enrich_operation_df(teacher_env_df, config)
                actor_env_df = enrich_operation_df(actor_env_df, config)
                teacher_csv = run_dir / f"{run_label}_teacher_env.csv"
                actor_csv = run_dir / f"{run_label}_actor_env.csv"
                teacher_env_df.to_csv(teacher_csv)
                actor_env_df.to_csv(actor_csv)

                power_ylim = compute_power_limits(teacher_env_df, actor_env_df)
                teacher_plot = plot_power_and_soc(teacher_env_df, "teacher", power_ylim, run_dir)
                actor_plot = plot_power_and_soc(actor_env_df, "actor", power_ylim, run_dir)

                delta_reward = actor_reward - teacher_reward
                teacher_comp_line = format_component_summary("Teacher", teacher_components)
                actor_comp_line = format_component_summary("Actor", actor_components)
                summary_lines = [
                    f"Teacher results -> reward: {teacher_reward:.3f}, csv: {teacher_csv}",
                    f"Actor results   -> reward: {actor_reward:.3f}, csv: {actor_csv}",
                    f"Reward delta (Actor - Teacher): {delta_reward:.3f}",
                    teacher_comp_line,
                    actor_comp_line,
                    f"Power/SOC plots saved: {teacher_plot} {actor_plot}",
                ]
                for line in summary_lines:
                    print(line)

                summary_json_path = export_summary(
                    run_dir,
                    hparams,
                    mask_payload,
                    {
                        "teacher_reward": teacher_reward,
                        "actor_reward": actor_reward,
                        "delta_reward": delta_reward,
                        "teacher_components": teacher_components,
                        "actor_components": actor_components,
                    },
                )
                summary_lines.append(f"Evaluation summary JSON: {summary_json_path}")
                text_summary_path = save_json_summary(run_dir, summary_lines)
                print(f"JSON summary saved to {text_summary_path}")


if __name__ == "__main__":
    main()
