"""Imitation-learning helpers for architecture comparisons.

The teacher provides optimal actions for each timestep; observations are
collected from the active SmartHomeEnv so the dataset matches the controller's
input space. Optionally runs Optuna-based hyperparameter optimization that
includes the observation-history length L.
"""
from __future__ import annotations

import csv
import importlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from environment import SmartHomeEnv
from opt import Teacher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _chronological_split(x, y, eval_frac):
    n = len(x)
    cut = int(np.floor(n * (1.0 - float(eval_frac))))
    cut = max(1, min(n - 1, cut))
    return x[:cut], y[:cut], x[cut:], y[cut:]


def _make_sequences(x, y, history_len):
    history_len = max(1, int(history_len))
    if history_len == 1:
        return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)
    if len(x) < history_len:
        return (
            np.empty((0, history_len, x.shape[-1]), dtype=np.float32),
            np.empty((0, y.shape[-1]), dtype=np.float32),
        )
    xs = [x[i - history_len + 1:i + 1] for i in range(history_len - 1, len(x))]
    ys = [y[i] for i in range(history_len - 1, len(y))]
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _actor_output(actor, xb):
    out = actor(xb)
    return out[0] if isinstance(out, tuple) else out


def _load_run_df(run):
    return pd.read_csv(
        PROJECT_ROOT / run["dataset"],
        sep=";",
        parse_dates=["timestamp"],
        dayfirst=True,
        index_col="timestamp",
    )


def _teacher_env_dataset(run, tariff, parameters, solver):
    df = _load_run_df(run)
    start = datetime.strptime(run["date"], "%Y-%m-%d %H:%M:%S")

    teacher = Teacher(df, parameters, start, run["days"], run["soc"], tariff)
    teacher.build()
    teacher.solve(solver=solver)
    teacher.get_operation()

    env = SmartHomeEnv(
        df, parameters, start, run["days"], run["soc"], tariff,
        track_operation=False,
    )
    obs, _ = env.reset()

    xs, ys = [], []
    for t in teacher.model.Ωt:
        action = np.asarray(teacher.get_actions(t), dtype=np.float32)
        xs.append(np.asarray(obs, dtype=np.float32))
        ys.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _build_raw_dataset(cfg, tariff, parameters):
    """Run the MILP teacher once per scenario; return raw (x, y) split into
    chronological train/val. History windowing is applied later."""
    eval_frac = cfg["training"].get("eval", 0.2)
    solver = cfg.get("solver", "gurobi")
    raw_tr, raw_va = [], []
    raw_train = raw_val = 0
    for run in cfg["runs"]:
        x, y = _teacher_env_dataset(run, tariff, parameters, solver)
        x_tr, y_tr, x_va, y_va = _chronological_split(x, y, eval_frac)
        raw_train += len(x_tr)
        raw_val += len(x_va)
        raw_tr.append((x_tr, y_tr))
        raw_va.append((x_va, y_va))
    return raw_tr, raw_va, raw_train, raw_val


def _sequences_from_raw(raw_tr, raw_va, history_len):
    """Convert raw chronological data into fixed-length history sequences."""
    train_x, train_y, val_x, val_y = [], [], [], []
    for x, y in raw_tr:
        xs, ys = _make_sequences(x, y, history_len)
        if len(xs):
            train_x.append(xs)
            train_y.append(ys)
    for x, y in raw_va:
        xs, ys = _make_sequences(x, y, history_len)
        if len(xs):
            val_x.append(xs)
            val_y.append(ys)
    if not train_x or not val_x:
        raise RuntimeError(
            f"IL dataset is empty for history_len={history_len}. "
            "Check run length and history_len."
        )
    return {
        "train_x": np.concatenate(train_x, axis=0),
        "train_y": np.concatenate(train_y, axis=0),
        "val_x": np.concatenate(val_x, axis=0),
        "val_y": np.concatenate(val_y, axis=0),
    }


# ---------------------------------------------------------------------------
# Training inner loop (used by both the fixed-config path and HPO trials)
# ---------------------------------------------------------------------------

def _train_one(actor_cfg, module, raw_tr, raw_va, params,
               lr, batch_size, weight_decay, history_len,
               epochs, patience_max, grad_clip, loss_dims,
               trial=None, audit_writer=None, pbar_desc=None):
    """Train a single configuration. Returns (best_val_loss, best_state_dict)."""
    data = _sequences_from_raw(raw_tr, raw_va, history_len)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(data["train_x"]), torch.from_numpy(data["train_y"])),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(data["val_x"]), torch.from_numpy(data["val_y"])),
        batch_size=batch_size, shuffle=False,
    )
    actor = module.load_actor(actor_cfg, device=DEVICE)
    opt = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    best_val, patience, best_state = float("inf"), 0, None

    iterable = range(epochs)
    if pbar_desc is not None:
        iterable = tqdm(iterable, desc=pbar_desc, dynamic_ncols=True, leave=False)

    for epoch in iterable:
        actor.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = _actor_output(actor, xb)
            loss = mse(pred[:, :loss_dims], yb[:, :loss_dims])
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        actor.eval()
        val_losses, full_losses = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = _actor_output(actor, xb)
                val_losses.append(float(mse(pred[:, :loss_dims], yb[:, :loss_dims]).cpu()))
                full_losses.append(float(mse(pred, yb).cpu()))

        val_loss = float(np.mean(val_losses))
        if audit_writer is not None:
            audit_writer.writerow({
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "val_full_mse": float(np.mean(full_losses)),
            })

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        else:
            patience += 1
            if patience >= patience_max:
                break

        if trial is not None:
            import optuna
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val, best_state, data


# ---------------------------------------------------------------------------
# Optional HPO via Optuna
# ---------------------------------------------------------------------------

def _run_hpo(actor_cfg, module, raw_tr, raw_va, parameters, tcfg, hpo_cfg, arch_name, tariff):
    """Optuna search over lr, batch_size, weight_decay, and history_len."""
    import optuna

    ss = hpo_cfg["search_space"]
    n_trials = int(hpo_cfg.get("n_trials", 30))
    epochs = int(hpo_cfg.get("epochs", tcfg.get("epochs", 100)))
    patience_max = int(hpo_cfg.get("patience", tcfg.get("patience", 10)))
    grad_clip = float(tcfg.get("grad_clip_norm", 1.0))
    loss_dims = int(tcfg.get("loss_dims", 2))

    def objective(trial):
        lr = trial.suggest_float("lr", ss["lr"][0], ss["lr"][1], log=True)
        batch_size = trial.suggest_categorical("batch_size", ss["batch_size"])
        weight_decay = trial.suggest_float(
            "weight_decay", ss["weight_decay"][0], ss["weight_decay"][1]
        )
        history_len = trial.suggest_categorical("history_len", ss["history_len"])
        best_val, _, _ = _train_one(
            actor_cfg, module, raw_tr, raw_va, parameters,
            lr=lr, batch_size=int(batch_size), weight_decay=weight_decay,
            history_len=int(history_len),
            epochs=epochs, patience_max=patience_max,
            grad_clip=grad_clip, loss_dims=loss_dims,
            trial=trial,
        )
        return best_val

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(tcfg.get("seed", 42))),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train_il(algo_root: Path):
    algo_root = Path(algo_root)
    arch_name = algo_root.name
    model_root = PROJECT_ROOT / "models" / arch_name
    cfg = _read_json(algo_root / "config.json")
    model_cfg = _read_json(model_root / "model.json")
    parameters_path = PROJECT_ROOT / "data" / "parameters.json"
    parameters = _read_json(parameters_path)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    module = importlib.import_module(f"models.{arch_name}.model")
    actor_cfg = dict(model_cfg["actor"])
    actor_cfg["parameters"] = str(parameters_path)

    tcfg = cfg["training"]
    epochs = int(tcfg.get("epochs", 300))
    patience_max = int(tcfg.get("patience", 20))
    grad_clip = float(tcfg.get("grad_clip_norm", 1.0))
    loss_dims = int(tcfg.get("loss_dims", 2))
    hpo_cfg = cfg.get("hpo", {})
    hpo_enabled = bool(hpo_cfg.get("enabled", False))

    for tariff in cfg.get("tariffs", ["tar_s"]):
        folder = PROJECT_ROOT / "paper" / "train" / "supervised" / arch_name / tariff
        folder.mkdir(parents=True, exist_ok=True)

        raw_tr, raw_va, raw_train, raw_val = _build_raw_dataset(cfg, tariff, parameters)

        # ------ HPO stage (optional) ------
        if hpo_enabled:
            print(f"[hpo] {arch_name}/{tariff}: searching {hpo_cfg.get('n_trials', 30)} trials")
            best_params, best_val_hpo = _run_hpo(
                actor_cfg, module, raw_tr, raw_va, parameters,
                tcfg, hpo_cfg, arch_name, tariff,
            )
            lr = float(best_params["lr"])
            batch_size = int(best_params["batch_size"])
            weight_decay = float(best_params["weight_decay"])
            history_len = int(best_params["history_len"])
            print(f"[hpo] {arch_name}/{tariff}: best params = {best_params}, "
                  f"best val_loss = {best_val_hpo:.6f}")
        else:
            lr = float(tcfg.get("lr", 3e-4))
            batch_size = int(tcfg.get("batch_size", 256))
            weight_decay = float(tcfg.get("weight_decay", 0.0))
            history_len = int(tcfg.get("history_len", 96))

        # ------ Final training with the chosen configuration ------
        audit_path = folder / "audit_il.csv"
        with open(audit_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_full_mse"])
            writer.writeheader()
            best_val, best_state, data = _train_one(
                actor_cfg, module, raw_tr, raw_va, parameters,
                lr=lr, batch_size=batch_size, weight_decay=weight_decay,
                history_len=history_len,
                epochs=epochs, patience_max=patience_max,
                grad_clip=grad_clip, loss_dims=loss_dims,
                audit_writer=writer,
                pbar_desc=f"IL {arch_name} {tariff}",
            )

        if best_state is None:
            raise RuntimeError(f"IL training did not converge for {arch_name}/{tariff}")

        torch.save(best_state, folder / "best.pt")
        with open(folder / "best_params.json", "w", encoding="utf-8") as out:
            json.dump({
                "lr": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "history_len": history_len,
                "loss_dims": loss_dims,
                "val_loss": best_val,
                "hpo_enabled": hpo_enabled,
            }, out, indent=4)
        with open(folder / "actor_cfg.json", "w", encoding="utf-8") as out:
            saved_cfg = dict(actor_cfg)
            saved_cfg["history_len"] = history_len
            json.dump(saved_cfg, out, indent=4)
        with open(folder / "dataset_meta.json", "w", encoding="utf-8") as out:
            json.dump({
                "architecture": arch_name,
                "tariff": tariff,
                "raw_train": raw_train,
                "raw_val": raw_val,
                "train_sequences": int(len(data["train_x"])),
                "val_sequences": int(len(data["val_x"])),
                "observation_dim": int(data["train_x"].shape[-1]),
                "target_dim": int(data["train_y"].shape[-1]),
                "history_len": history_len,
                "note": "Loss trains BESS/EV dims only; PV is deterministic in the actor and logged via val_full_mse.",
            }, out, indent=4)
