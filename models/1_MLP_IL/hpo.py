from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import optuna
import matplotlib.pyplot as plt
import pandas as pd

from hp import HyperParameters


@dataclass
class HPOConfig:
    enabled: bool = False
    n_trials: int = 10
    timeout: int | None = None
    study_name: str = "mlp_mask_search"
    storage: str | None = None
    direction: str = "minimize"
    seed: int = 42
    mask_presets: Sequence[str] = field(
        default_factory=lambda: ["full", "no_climate", "power_core", "bess_ev_focus"]
    )
    allow_custom_mask: bool = True
    dropout_range: Tuple[float, float] = (0.0, 0.3)
    lr_log_range: Tuple[float, float] = (-4.5, -2.5)
    weight_decay_log_range: Tuple[float, float] = (-6.0, -3.0)
    batch_size_options: Sequence[int] = field(default_factory=lambda: [128, 256, 512])
    hidden_size_options: Sequence[Any] = field(
        default_factory=lambda: [(256, 128), (256, 256), (512, 256)]
    )
    activation_options: Sequence[str] | None = None
    epochs_limit: int | None = 20
    final_epochs: int | None = None
    report_pruning: bool = True


def create_study(config: HPOConfig) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=config.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5)
    return optuna.create_study(
        direction=config.direction,
        sampler=sampler,
        pruner=pruner,
        storage=config.storage,
        study_name=config.study_name,
        load_if_exists=bool(config.storage),
    )


def run_optimization(config: HPOConfig, objective: Callable[[optuna.Trial], float]) -> optuna.Study:
    study = create_study(config)
    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    return study


def suggest_mask(trial: optuna.Trial, config: HPOConfig, state_groups: Sequence[str]) -> object:
    preset_options = list(dict.fromkeys(config.mask_presets))
    allow_custom = config.allow_custom_mask and len(state_groups) > 0
    if allow_custom:
        preset_options.append("custom")
    choice = trial.suggest_categorical("mask_preset", preset_options)
    if choice != "custom":
        return choice
    overrides: Dict[str, bool] = {}
    for group in state_groups:
        overrides[group] = trial.suggest_categorical(f"mask_group_{group}", [True, False])
    if all(overrides.values()):
        return "full"
    return overrides


def _coerce_hidden_option(option: Any) -> Tuple[str, Tuple[int, ...]]:
    if isinstance(option, str):
        cleaned = option.strip()
        if not cleaned:
            raise ValueError("Received empty string for hidden layer option")
        tokens = [token for token in re.split(r"[x,;\s-]+", cleaned) if token]
        if not tokens:
            raise ValueError(f"Could not parse hidden layer option from '{option}'")
        dims = tuple(int(token) for token in tokens)
        return "x".join(str(dim) for dim in dims), dims
    if isinstance(option, Sequence):
        dims_list: List[int] = []
        for value in option:
            dims_list.append(int(value))
        if not dims_list:
            raise ValueError("Hidden layer option sequence is empty")
        dims = tuple(dims_list)
        return "x".join(str(dim) for dim in dims), dims
    raise TypeError(f"Unsupported hidden layer specification: {option!r}")


def suggest_hparams(trial: optuna.Trial, base: HyperParameters, config: HPOConfig) -> HyperParameters:
    payload = base.to_dict()
    hidden_mapping: Dict[str, Tuple[int, ...]] = {}
    for option in config.hidden_size_options:
        label, dims = _coerce_hidden_option(option)
        if label not in hidden_mapping:
            hidden_mapping[label] = dims
    if hidden_mapping:
        labels = list(hidden_mapping.keys())
        chosen_label = trial.suggest_categorical("hidden_sizes", labels)
        payload["hidden_sizes"] = hidden_mapping[chosen_label]
    else:
        payload["hidden_sizes"] = tuple(int(v) for v in base.hidden_sizes)
    activation_choices = [opt for opt in (config.activation_options or []) if isinstance(opt, str) and opt]
    if activation_choices:
        payload["activation"] = trial.suggest_categorical(
            "activation",
            list(dict.fromkeys(activation_choices)),
        )
    payload["dropout"] = trial.suggest_float("dropout", *config.dropout_range)
    payload["learning_rate"] = float(10 ** trial.suggest_float("log10_lr", *config.lr_log_range))
    payload["weight_decay"] = float(10 ** trial.suggest_float("log10_weight_decay", *config.weight_decay_log_range))
    payload["batch_size"] = trial.suggest_categorical("batch_size", list(config.batch_size_options))
    epochs_limit = config.epochs_limit
    if epochs_limit is not None:
        payload["epochs"] = min(int(payload.get("epochs", base.epochs)), epochs_limit)
    return HyperParameters.from_dict(payload)


def plot_hpo_results(study: optuna.Study, output_path: Path) -> None:
    records = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        payload = {"val_loss": float(trial.value)}
        payload.update(trial.params)
        records.append(payload)
    if len(records) < 2:
        return
    df = pd.DataFrame(records)
    param_cols = [col for col in df.columns if col != "val_loss" and df[col].notna().any()]
    if not param_cols:
        return
    n_cols = min(3, len(param_cols))
    n_rows = (len(param_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for idx, param in enumerate(param_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        y = df["val_loss"].to_numpy(dtype=float)
        series = df[param]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            ax.scatter(numeric.to_numpy(dtype=float), y, color="#2f80ed", alpha=0.8)
            ax.set_xlabel(param)
        else:
            labels = series.astype(str)
            codes, uniques = pd.factorize(labels)
            ax.scatter(codes, y, color="#eb5757", alpha=0.8)
            ax.set_xticks(range(len(uniques)))
            ax.set_xticklabels(uniques, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel(f"{param} (cat)")
        if col == 0:
            ax.set_ylabel("val_loss")
        ax.grid(True, alpha=0.2)
    for idx in range(len(param_cols), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")
    fig.suptitle("Optuna trials: val_loss vs hyperparameters", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

def plot_parallel_hparams(study: optuna.Study, output_path: Path, max_cat_ticks: int = 6) -> None:
    records = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        row = {"val_loss": float(trial.value)}
        row.update(trial.params)
        records.append(row)
    if len(records) < 2:
        return
    df = pd.DataFrame(records)
    param_cols = [col for col in df.columns if col != "val_loss" and df[col].notna().any()]
    if not param_cols:
        return
    columns = param_cols + ["val_loss"]
    scaled: Dict[str, np.ndarray] = {}
    tick_meta: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for col in columns:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            values = numeric.to_numpy(dtype=float)
            vmin = float(values.min())
            vmax = float(values.max())
            span = vmax - vmin
            denom = span if span > 0 else 1.0
            scaled_values = (values - vmin) / denom
            ticks = np.linspace(0.0, 1.0, num=5)
            tick_labels = [f"{vmin + t * span:.3g}" if span > 0 else f"{vmin:.3g}" for t in ticks]
        else:
            labels = series.astype(str)
            uniques = pd.unique(labels)
            mapping = {label: idx for idx, label in enumerate(uniques)}
            codes = np.array([mapping[label] for label in labels], dtype=float)
            denom = max(len(uniques) - 1, 1)
            scaled_values = codes / denom
            tick_count = min(len(uniques), max_cat_ticks)
            if tick_count == len(uniques):
                tick_idx = np.arange(len(uniques), dtype=float)
                ticks = tick_idx / denom
                tick_labels = [str(label) for label in uniques]
            else:
                tick_idx = np.linspace(0, len(uniques) - 1, tick_count)
                ticks = (tick_idx / denom).astype(float)
                tick_labels = [str(uniques[int(round(idx))]) for idx in tick_idx]
        scaled[col] = scaled_values
        tick_meta[col] = (np.asarray(ticks, dtype=float), list(tick_labels))

    loss_values = df["val_loss"].to_numpy(dtype=float)
    loss_min = float(loss_values.min())
    loss_max = float(loss_values.max())
    if loss_max == loss_min:
        loss_max += 1e-9
    x_positions = np.arange(len(columns), dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(columns)), 5))
    cmap = plt.cm.inferno_r
    norm = plt.Normalize(loss_min, loss_max)
    for row_idx in range(len(df)):
        color = cmap(norm(loss_values[row_idx]))
        y_coords = [scaled[col][row_idx] for col in columns]
        ax.plot(x_positions, y_coords, color=color, alpha=0.75, linewidth=1.5)

    for idx, col in enumerate(columns):
        ticks, labels = tick_meta[col]
        ax.axvline(idx, color="#d0d0d0", linewidth=1, zorder=0)
        for tick, label in zip(ticks, labels):
            ax.text(
                idx - 0.05,
                float(np.clip(tick, 0.0, 1.0)),
                label,
                ha="right",
                va="center",
                fontsize=7,
                color="#555555",
            )

    ax.set_xlim(0, len(columns) - 1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(columns, rotation=25, ha="right")
    ax.set_ylabel("normalized scale")
    ax.set_title("Optuna trials: parallel hyperparameter view")
    ax.grid(axis="y", alpha=0.2)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("val_loss")

    output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

STATE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "bess": ("bess_soc", "bess_soh"),
    "ev": ("ev_soc", "ev_soh", "ev_connected", "ev_shortfall"),
    "pv": ("pv_available",),
    "load": ("load", "net_load", "tariff"),
    "climate": (),
}


STATE_MASK_LIBRARY: Dict[str, Dict[str, bool]] = {
    "full": {},
    "no_climate": {"climate": False},
    "power_core": {"pv": False},
    "bess_ev_focus": {"load": False, "pv": False},
}


def _normalize_mask_spec(spec: object | None) -> Dict[str, bool]:
    if spec is None:
        return {}
    if isinstance(spec, str):
        return dict(STATE_MASK_LIBRARY.get(spec, {}))
    if isinstance(spec, dict):
        return {str(key): bool(value) for key, value in spec.items()}
    return {}


def register_mask_library(spec: object | None) -> str:
    if spec is None:
        return "full"
    if isinstance(spec, str):
        if spec not in STATE_MASK_LIBRARY:
            STATE_MASK_LIBRARY[spec] = {}
        return spec
    normalized = _normalize_mask_spec(spec)
    fingerprint = hashlib.md5(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    label = f"custom_{fingerprint}"
    STATE_MASK_LIBRARY[label] = normalized
    return label


def resolve_mask_vector(labels: List[str], spec: object | None) -> np.ndarray:
    overrides = _normalize_mask_spec(spec)
    vector = np.ones(len(labels), dtype=bool)
    group_flags = {group: overrides.get(group, True) for group in STATE_GROUPS}
    column_flags = {col: overrides[col] for col in overrides if col not in STATE_GROUPS}
    for idx, label in enumerate(labels):
        include = column_flags.get(label)
        if include is None:
            include = True
            for group, columns in STATE_GROUPS.items():
                if label in columns:
                    include = group_flags.get(group, True)
                    break
        vector[idx] = include
    return vector


def apply_mask(states: np.ndarray, labels: List[str], spec: object | None) -> Tuple[np.ndarray, List[str], np.ndarray]:
    mask_vector = resolve_mask_vector(labels, spec)
    selected = np.nonzero(mask_vector)[0]
    if len(selected) == 0:
        selected = np.arange(states.shape[1])
        mask_vector = np.ones(states.shape[1], dtype=bool)
    masked_states = states[:, selected]
    masked_labels = [labels[idx] for idx in selected]
    return masked_states, masked_labels, mask_vector


def run_hpo_pipeline(
    base_hparams: HyperParameters,
    config: HPOConfig,
    objective_fn: Callable[[HyperParameters, object | None, optuna.Trial | None], Dict[str, object]],
    artifacts_dir: Path,
) -> Tuple[HyperParameters, object | None]:
    artifacts_dir = Path(artifacts_dir)
    state_groups: Sequence[str] = tuple(STATE_GROUPS.keys())

    def objective(trial: optuna.Trial) -> float:
        mask_choice = suggest_mask(trial, config, state_groups)
        hparams = suggest_hparams(trial, base_hparams, config)
        trial.set_user_attr("mask_choice", mask_choice)
        trial.set_user_attr("hparams", hparams.to_dict())
        result = objective_fn(hparams, mask_choice, trial)
        if isinstance(result, dict):
            metric = result.get("best_val_loss") or result.get("val_loss")
            val_loss = float(metric if metric is not None else 0.0)
        else:
            val_loss = float(result)
        return val_loss

    study = run_optimization(config, objective)
    if study.trials:
        plot_hpo_results(study, artifacts_dir / "hpo_scatter.png")
        plot_parallel_hparams(study, artifacts_dir / "hpo_parallel.png")
    best_trial = study.best_trial
    hparams_payload = best_trial.user_attrs.get("hparams", base_hparams.to_dict())
    best_hparams = HyperParameters.from_dict(hparams_payload)
    if config.final_epochs is not None:
        best_hparams.epochs = config.final_epochs
    best_mask = best_trial.user_attrs.get("mask_choice")
    return best_hparams, best_mask


__all__ = [
    "STATE_GROUPS",
    "STATE_MASK_LIBRARY",
    "HPOConfig",
    "create_study",
    "run_optimization",
    "suggest_hparams",
    "suggest_mask",
    "apply_mask",
    "register_mask_library",
    "resolve_mask_vector",
    "run_hpo_pipeline",
    "plot_hpo_results",
    "plot_parallel_hparams",
]
