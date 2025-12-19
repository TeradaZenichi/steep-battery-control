from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from env.environment import SmartHomeEnv  # type: ignore  # pylint: disable=wrong-import-position
from opt import Teacher  # type: ignore  # pylint: disable=wrong-import-position
from hp import HyperParameters  # type: ignore  # pylint: disable=wrong-import-position
from model import TimeDecisionTransformer  # type: ignore  # pylint: disable=wrong-import-position
from hpo import (  # type: ignore  # pylint: disable=wrong-import-position
    HPOConfig,
    apply_mask,
    register_mask_library,
    resolve_mask_vector,
    run_hpo_pipeline,
)

CONFIG_PATH = Path("data/parameters.json")
DATA_PATH = Path("data/Simulation_CY_Cur_HP__PV5000-HB5000.csv")
START_DATE = "01/01/2000 00:00"
DAYS = 365
SOLVER_NAME = "gurobi"
MODEL_JSON = Path(__file__).with_name("model.json")
RESULTS_BASE = Path("Results")
MODEL_SUBDIR = "3_TDT_IL"
TARIFF_OVERRIDE: str | None = None
TARIFFS: list[str] | None = ["tar_s", "tar_w", "tar_sw", "tar_flat", "tar_tou"]
HPO_SETTINGS = HPOConfig()


def _load_tariff_label(config_path: Path, override: str | None) -> str:
    label = override
    if label is None:
        with open(config_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
        label = config["Grid"]["tariff_column"]
    text = str(label).strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return cleaned or "default_tariff"


CURRENT_TARIFF_OVERRIDE: str | None = None
TARIFF_LABEL = _load_tariff_label(CONFIG_PATH, TARIFF_OVERRIDE)
TARIFF_DIR = RESULTS_BASE / TARIFF_LABEL
RESULTS_DIR = TARIFF_DIR / MODEL_SUBDIR
CACHE_DIR = TARIFF_DIR / "cache"
SAVE_PATH = RESULTS_DIR / "best.pt"


def set_tariff_dirs(tariff_label: str) -> None:
    global TARIFF_LABEL, TARIFF_DIR, RESULTS_DIR, CACHE_DIR, SAVE_PATH, CURRENT_TARIFF_OVERRIDE
    TARIFF_LABEL = tariff_label
    CURRENT_TARIFF_OVERRIDE = tariff_label
    TARIFF_DIR = RESULTS_BASE / TARIFF_LABEL
    RESULTS_DIR = TARIFF_DIR / MODEL_SUBDIR
    CACHE_DIR = TARIFF_DIR / "cache"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = RESULTS_DIR / "best.pt"


set_tariff_dirs(TARIFF_LABEL)


def _scenario_signature(payload: Dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


def _snapshot_config(target_dir: Path) -> None:
    shutil.copy2(CONFIG_PATH, target_dir / "parameters.json")


def _cache_path(label: str, signature: str) -> Path:
    return CACHE_DIR / f"{label.replace(' ', '_')}_{signature}.npz"


def _load_cached(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    states = data["states"].astype(np.float32, copy=False)
    targets = data["targets"].astype(np.float32, copy=False)
    rewards = data["rewards"].astype(np.float32, copy=False)
    labels = data["labels"].tolist()
    return states, targets, rewards, labels


def _save_cached(
    path: Path,
    states: np.ndarray,
    targets: np.ndarray,
    rewards: np.ndarray,
    labels: List[str],
    metadata: Dict[str, object],
) -> None:
    np.savez(
        path,
        states=states,
        targets=targets,
        rewards=rewards,
        labels=np.array(labels, dtype=object),
        metadata=json.dumps(metadata, sort_keys=True),
    )


TRAIN_SCENARIOS: List[Dict[str, object]] = [
    {
        "name": "base",
        "config_path": CONFIG_PATH,
        "data_path": Path("data/Simulation_CY_Cur_HP__PV5000-HB5000.csv"),
        "start_date": START_DATE,
        "days": DAYS,
        "solver_name": SOLVER_NAME,
        "start_soc": 0.5,
        "state_mask": "full",
        "cache": True,
    },
    {
        "name": "extended",
        "config_path": CONFIG_PATH,
        "data_path": Path("data/Simulation_WY_Cur_HP__PV5000-HB5000.csv"),
        "start_date": START_DATE,
        "days": DAYS,
        "solver_name": SOLVER_NAME,
        "start_soc": 0.5,
        "state_mask": "full",
        "cache": True,
    },
]


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_hparams_and_hpo(path: Path) -> Tuple[HyperParameters, HPOConfig]:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    hpo_payload = payload.pop("hpo", None)
    base_hparams = HyperParameters.from_dict(payload)
    if isinstance(hpo_payload, dict):
        allowed = set(HPOConfig.__dataclass_fields__.keys())
        filtered = {key: value for key, value in hpo_payload.items() if key in allowed}
        hpo_settings = HPOConfig(**filtered)
    else:
        hpo_settings = HPOConfig()
    return base_hparams, hpo_settings


def simulate_teacher_rewards(
    config: dict,
    dataframe: pd.DataFrame,
    start_date: str,
    days: int,
    start_soc: float,
    actions_df: pd.DataFrame,
) -> np.ndarray:
    env = SmartHomeEnv(config, dataframe=dataframe, days=days, start_date=start_date)
    obs, _ = env.reset()
    env.bess.reset(soc_init=start_soc)
    rewards: List[float] = []
    while not env.done:
        ts = env.sim.current_datetime
        if ts not in actions_df.index:
            raise KeyError(f"Timestamp {ts} missing in teacher actions")
        row = actions_df.loc[ts, ["PBESS", "Pev", "chi_pv"]]
        action = row.to_numpy(dtype=np.float32)
        obs, reward, done, _ = env.step(action)
        rewards.append(float(reward))
    return np.asarray(rewards, dtype=np.float32)


def _compute_returns_to_go(rewards: np.ndarray) -> np.ndarray:
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running += rewards[idx]
        rtg[idx] = running
    return rtg


def _build_sequence_windows(
    features: np.ndarray,
    targets: np.ndarray,
    rewards: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq_len = max(1, int(seq_len))
    state_dim = features.shape[1]
    action_dim = targets.shape[1]
    rtg = _compute_returns_to_go(rewards)
    pad = seq_len - 1
    zero_states = np.zeros((pad, state_dim), dtype=np.float32)
    zero_actions = np.zeros((pad, action_dim), dtype=np.float32)
    zero_returns = np.zeros(pad, dtype=np.float32)
    states_pad = np.concatenate([zero_states, features], axis=0)
    actions_pad = np.concatenate([zero_actions, targets], axis=0)
    rtg_pad = np.concatenate([zero_returns, rtg], axis=0)
    timesteps = np.arange(len(features), dtype=np.int64)
    time_pad = np.concatenate([np.zeros(pad, dtype=np.int64), timesteps], axis=0)
    mask_pad = np.concatenate([np.ones(pad, dtype=bool), np.zeros(len(features), dtype=bool)], axis=0)

    seq_states: List[np.ndarray] = []
    seq_actions: List[np.ndarray] = []
    seq_returns: List[np.ndarray] = []
    seq_timesteps: List[np.ndarray] = []
    seq_masks: List[np.ndarray] = []
    for idx in range(pad, pad + len(features)):
        start = idx - seq_len + 1
        end = idx + 1
        seq_states.append(states_pad[start:end])
        seq_actions.append(actions_pad[start:end])
        seq_returns.append(rtg_pad[start:end])
        seq_timesteps.append(time_pad[start:end])
        seq_masks.append(mask_pad[start:end])
    return (
        np.stack(seq_states, axis=0),
        np.stack(seq_actions, axis=0),
        np.stack(seq_returns, axis=0),
        np.stack(seq_timesteps, axis=0),
        np.stack(seq_masks, axis=0),
    )


def build_teacher_dataset(
    scenarios: List[Dict[str, object]],
    seq_len: int,
    mask_override: object | None = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    np.ndarray,
    str,
]:
    state_chunks: List[np.ndarray] = []
    action_chunks: List[np.ndarray] = []
    return_chunks: List[np.ndarray] = []
    timestep_chunks: List[np.ndarray] = []
    mask_chunks: List[np.ndarray] = []
    shared_labels: List[str] | None = None
    shared_mask_vector: np.ndarray | None = None
    shared_mask_repr: str | None = None
    full_label_template: List[str] | None = None

    for idx, scenario in enumerate(scenarios, start=1):
        config_path = Path(scenario.get("config_path", CONFIG_PATH))
        data_path = Path(scenario.get("data_path", DATA_PATH))
        start_date = str(scenario.get("start_date", START_DATE))
        days = int(scenario.get("days", DAYS))
        solver_name = str(scenario.get("solver_name", SOLVER_NAME))
        start_soc = float(scenario.get("start_soc", 0.5))
        label = str(scenario.get("name", f"scenario_{idx}"))
        mask_spec = mask_override if mask_override is not None else scenario.get("state_mask")
        mask_repr = register_mask_library(mask_spec)
        cache_enabled = bool(scenario.get("cache", True))
        signature = _scenario_signature(
            {
                "config_path": str(config_path.resolve()),
                "data_path": str(data_path.resolve()),
                "start_date": start_date,
                "days": days,
                "solver": solver_name,
                "start_soc": start_soc,
            }
        )
        cache_file = _cache_path(label, signature)
        states_full = targets = rewards = labels_full = None
        if cache_enabled and cache_file.exists():
            try:
                states_full, targets, rewards, labels_full = _load_cached(cache_file)
                print(
                    f"Loaded cached data for scenario '{label}' ({states_full.shape[0]} steps) from {cache_file}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to load cache for scenario '{label}' ({exc}); rebuilding")

        if states_full is None or targets is None or rewards is None or labels_full is None:
            print(
                f"Running teacher scenario '{label}' with data={data_path.name}, days={days}, start_soc={start_soc}"
            )
            config = _load_config(config_path)
            if CURRENT_TARIFF_OVERRIDE:
                config.setdefault("Grid", {})["tariff_column"] = CURRENT_TARIFF_OVERRIDE
            dataframe = pd.read_csv(data_path, sep=";")
            teacher = Teacher(config, dataframe=dataframe, start_date=start_date, days=days)
            teacher.build(start_soc=start_soc)
            teacher.solve(solver_name=solver_name)
            states_full, labels_full = teacher.get_masked_observations(state_mask=None)
            actions_df = teacher.results_df()[["PBESS", "Pev", "chi_pv"]]
            targets = actions_df.to_numpy(dtype=np.float32)
            rewards = simulate_teacher_rewards(config, dataframe, start_date, days, start_soc, actions_df)
            if cache_enabled:
                metadata = {
                    "label": label,
                    "signature": signature,
                    "num_samples": len(states_full),
                }
                _save_cached(
                    cache_file,
                    states_full.astype(np.float32),
                    targets.astype(np.float32),
                    rewards.astype(np.float32),
                    labels_full,
                    metadata,
                )
                print(f"Cached scenario '{label}' to {cache_file}")

        masked_states, masked_labels, mask_vector = apply_mask(states_full, labels_full, mask_spec)
        features = masked_states.astype(np.float32)
        target_block = targets.astype(np.float32)
        rewards_block = rewards.astype(np.float32)
        (
            seq_states,
            seq_actions,
            seq_returns,
            seq_timesteps,
            seq_masks,
        ) = _build_sequence_windows(features, target_block, rewards_block, seq_len)

        if seq_states.size == 0:
            print(f"Skipping scenario '{label}' due to insufficient samples for seq_len={seq_len}")
            continue

        if full_label_template is None:
            full_label_template = list(labels_full)
        if shared_labels is None:
            shared_labels = list(masked_labels)
        if shared_mask_vector is None:
            shared_mask_vector = mask_vector.astype(bool, copy=True)
        if shared_mask_repr is None:
            shared_mask_repr = mask_repr
        state_chunks.append(seq_states)
        action_chunks.append(seq_actions)
        return_chunks.append(seq_returns)
        timestep_chunks.append(seq_timesteps)
        mask_chunks.append(seq_masks)

    if not state_chunks:
        raise RuntimeError("No scenarios produced training data; try reducing seq_len or revisiting inputs.")

    features = np.concatenate(state_chunks, axis=0)
    targets = np.concatenate(action_chunks, axis=0)
    returns = np.concatenate(return_chunks, axis=0)
    timesteps = np.concatenate(timestep_chunks, axis=0)
    masks = np.concatenate(mask_chunks, axis=0)
    mask_vector = (
        shared_mask_vector
        if shared_mask_vector is not None
        else resolve_mask_vector(full_label_template or [], mask_override)
    )
    mask_repr = shared_mask_repr or "none"
    print(f"Aggregated dataset shape -> features: {features.shape}, targets: {targets.shape}")
    return features, targets, returns, timesteps, masks, shared_labels or [], mask_vector, mask_repr


def make_loaders(
    features: np.ndarray,
    targets: np.ndarray,
    returns: np.ndarray,
    timesteps: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
    val_split: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    total = len(features)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)
    split = int(total * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]

    def _build_dataset(idxs: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(features[idxs]),
            torch.from_numpy(targets[idxs]),
            torch.from_numpy(returns[idxs]),
            torch.from_numpy(timesteps[idxs]),
            torch.from_numpy(masks[idxs]),
        )

    train_loader = DataLoader(_build_dataset(train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_build_dataset(val_idx), batch_size=batch_size)
    return train_loader, val_loader


def masked_mse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (~mask).float().unsqueeze(-1)
    if valid.sum() <= 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    loss = (preds - targets) ** 2
    loss = (loss * valid).sum() / valid.sum().clamp_min(1.0)
    return loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        states, actions, returns, timesteps, masks = [tensor.to(device) for tensor in batch]
        optimizer.zero_grad()
        preds = model(states, actions, returns, timesteps, padding_mask=masks)
        loss = masked_mse(preds, actions, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * states.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            states, actions, returns, timesteps, masks = [tensor.to(device) for tensor in batch]
            preds = model(states, actions, returns, timesteps, padding_mask=masks)
            loss = masked_mse(preds, actions, masks)
            total_loss += loss.item() * states.size(0)
    return total_loss / len(loader.dataset)


def train_model(
    hparams: HyperParameters,
    mask_override: object | None = None,
    trial: optuna.Trial | None = None,
) -> Dict[str, object]:
    seq_len = max(1, int(hparams.seq_len))
    (
        features,
        targets,
        returns,
        timesteps,
        masks,
        feature_labels,
        mask_vector,
        mask_repr,
    ) = build_teacher_dataset(TRAIN_SCENARIOS, seq_len=seq_len, mask_override=mask_override)
    state_dim = features.shape[2]
    if state_dim != hparams.input_dim:
        print(f"Detected state_dim={state_dim}; updating hyperparameters (was {hparams.input_dim}).")
    hparams.input_dim = state_dim
    action_dim = targets.shape[2]
    if action_dim != hparams.output_dim:
        print(f"Detected action_dim={action_dim}; updating hyperparameters (was {hparams.output_dim}).")
    hparams.output_dim = action_dim
    if feature_labels:
        print(f"Feature columns ({len(feature_labels)}): {feature_labels}")

    val_split = getattr(hparams, "val_split", 0.2)
    train_loader, val_loader = make_loaders(
        features,
        targets,
        returns,
        timesteps,
        masks,
        hparams.batch_size,
        val_split,
        hparams.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeDecisionTransformer(hparams).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    patience = getattr(hparams, "patience", 0)
    improvement_tol = getattr(hparams, "improvement_tol", 1e-5)
    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0
    best_epoch = 0
    train_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(1, hparams.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
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
        if trial is not None and HPO_SETTINGS.report_pruning:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    state_to_save = best_state_dict if best_state_dict is not None else model.state_dict()
    return {
        "state_dict": state_to_save,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "train_history": train_history,
        "val_history": val_history,
        "feature_labels": feature_labels,
        "mask_vector": mask_vector.astype(bool, copy=True),
        "mask_repr": mask_repr,
        "seq_len": seq_len,
    }


def save_training_artifacts(hparams: HyperParameters, result: Dict[str, object]) -> None:
    torch.save(
        {
            "state_dict": result["state_dict"],
            "hparams": hparams.to_dict(),
            "best_val_loss": result["best_val_loss"],
            "state_mask": {
                "spec": result.get("mask_repr", "none"),
                "vector": np.asarray(result["mask_vector"], dtype=bool).tolist(),
                "labels": result.get("feature_labels", []),
            },
        },
        SAVE_PATH,
    )
    print(f"Model saved to {SAVE_PATH}")
    _snapshot_config(RESULTS_DIR)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(result["train_history"]) + 1), result["train_history"], label="train_loss")
    ax.plot(range(1, len(result["val_history"]) + 1), result["val_history"], label="val_loss")
    if result["best_epoch"]:
        ax.axvline(result["best_epoch"], color="green", linestyle="--", label=f"best_val@{result['best_epoch']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mse loss")
    ax.set_title("Time Decision Transformer imitation training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.pdf")
    plt.close(fig)
    print(f"Training curves saved to {RESULTS_DIR / 'training_curves.pdf'}")
    print(f"Best validation loss {result['best_val_loss']:.6f} at epoch {result['best_epoch']}")


def main() -> None:
    global HPO_SETTINGS
    base_hparams, HPO_SETTINGS = load_hparams_and_hpo(MODEL_JSON)
    base_tariff = _load_tariff_label(CONFIG_PATH, TARIFF_OVERRIDE)
    tariffs = [str(t) for t in (TARIFFS if TARIFFS else [base_tariff])]

    for tariff in tariffs:
        set_tariff_dirs(tariff)
        working_hparams = HyperParameters.from_dict(base_hparams.to_dict())
        mask_override: object | None = None

        if HPO_SETTINGS.enabled:
            working_hparams, mask_override = run_hpo_pipeline(
                base_hparams,
                HPO_SETTINGS,
                lambda h, m, tr: train_model(h, mask_override=m, trial=tr),
                SAVE_PATH.parent,
            )

        result = train_model(working_hparams, mask_override=mask_override)
        save_training_artifacts(working_hparams, result)


if __name__ == "__main__":
    main()
