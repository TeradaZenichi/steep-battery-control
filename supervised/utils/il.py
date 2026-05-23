"""Imitation-learning helpers for architecture comparisons.

The teacher optimizer still provides the optimal actions, but observations are
collected from the current SmartHomeEnv so the dataset matches the active
34-dimensional model input.
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


def _build_dataset(cfg, tariff, parameters):
    eval_frac = cfg["training"].get("eval", 0.2)
    history_len = cfg["training"].get("history_len", 96)
    solver = cfg.get("solver", "gurobi")
    train_x, train_y, val_x, val_y = [], [], [], []
    raw_train = raw_val = 0

    for run in cfg["runs"]:
        x, y = _teacher_env_dataset(run, tariff, parameters, solver)
        x_tr, y_tr, x_va, y_va = _chronological_split(x, y, eval_frac)
        raw_train += len(x_tr)
        raw_val += len(x_va)

        xs, ys = _make_sequences(x_tr, y_tr, history_len)
        if len(xs):
            train_x.append(xs)
            train_y.append(ys)

        xs, ys = _make_sequences(x_va, y_va, history_len)
        if len(xs):
            val_x.append(xs)
            val_y.append(ys)

    if not train_x or not val_x:
        raise RuntimeError("IL dataset is empty. Check run length and history_len.")

    return {
        "train_x": np.concatenate(train_x, axis=0),
        "train_y": np.concatenate(train_y, axis=0),
        "val_x": np.concatenate(val_x, axis=0),
        "val_y": np.concatenate(val_y, axis=0),
        "raw_train": raw_train,
        "raw_val": raw_val,
    }


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
    history_len = int(tcfg.get("history_len", 96))
    batch_size = int(tcfg.get("batch_size", 256))
    lr = float(tcfg.get("lr", 3e-4))
    weight_decay = float(tcfg.get("weight_decay", 0.0))
    epochs = int(tcfg.get("epochs", 300))
    patience_max = int(tcfg.get("patience", 20))
    grad_clip = float(tcfg.get("grad_clip_norm", 1.0))
    loss_dims = int(tcfg.get("loss_dims", 2))

    for tariff in cfg.get("tariffs", ["tar_s"]):
        folder = PROJECT_ROOT / "Results" / "train" / "supervised" / arch_name / tariff
        folder.mkdir(parents=True, exist_ok=True)

        data = _build_dataset(cfg, tariff, parameters)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(data["train_x"]), torch.from_numpy(data["train_y"])),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(data["val_x"]), torch.from_numpy(data["val_y"])),
            batch_size=batch_size,
            shuffle=False,
        )

        actor = module.load_actor(actor_cfg, device=DEVICE)
        opt = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=weight_decay)
        mse = nn.MSELoss()
        audit_path = folder / "audit_il.csv"
        best_val, patience = float("inf"), 0

        with open(audit_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_full_mse"])
            writer.writeheader()

            pbar = tqdm(range(epochs), desc=f"IL {arch_name} {tariff}", dynamic_ncols=True)
            for epoch in pbar:
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

                row = {
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)),
                    "val_loss": float(np.mean(val_losses)),
                    "val_full_mse": float(np.mean(full_losses)),
                }
                writer.writerow(row)
                f.flush()
                pbar.set_postfix({"val": f"{row['val_loss']:.5f}"})

                if row["val_loss"] < best_val:
                    best_val, patience = row["val_loss"], 0
                    torch.save(actor.state_dict(), folder / "best.pth")
                    with open(folder / "best_params.json", "w", encoding="utf-8") as out:
                        json.dump({
                            "lr": lr,
                            "batch_size": batch_size,
                            "weight_decay": weight_decay,
                            "history_len": history_len,
                            "loss_dims": loss_dims,
                            "val_loss": best_val,
                        }, out, indent=4)
                    with open(folder / "actor_cfg.json", "w", encoding="utf-8") as out:
                        saved_cfg = dict(actor_cfg)
                        saved_cfg["history_len"] = history_len
                        json.dump(saved_cfg, out, indent=4)
                else:
                    patience += 1
                    if patience >= patience_max:
                        break

        with open(folder / "dataset_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "architecture": arch_name,
                "tariff": tariff,
                "raw_train": data["raw_train"],
                "raw_val": data["raw_val"],
                "train_sequences": int(len(data["train_x"])),
                "val_sequences": int(len(data["val_x"])),
                "observation_dim": int(data["train_x"].shape[-1]),
                "target_dim": int(data["train_y"].shape[-1]),
                "note": "Loss trains BESS/EV dims only; PV is deterministic in the actor and logged via val_full_mse.",
            }, f, indent=4)
