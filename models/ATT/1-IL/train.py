from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../steep-battery-control
MODEL_ROOT   = Path(__file__).resolve().parents[2]  # .../models
ATT_ROOT     = Path(__file__).resolve().parents[1]  # .../models/ATT
ALGO_ROOT    = Path(__file__).resolve().parent      # .../models/ATT/1-IL
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ATT_ROOT))
sys.path.insert(0, str(MODEL_ROOT))
sys.path.append(str(ALGO_ROOT))

from hpo import HPO
from opt import Teacher
from model import load_actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _chronological_split(x_i: np.ndarray, y_i: np.ndarray, eval_frac: float):
    """Split a single run into train/val using the last eval_frac portion as validation."""
    n = len(x_i)
    if n < 2:
        return x_i, y_i, np.empty((0, x_i.shape[1]), dtype=np.float32), np.empty((0, y_i.shape[1]), dtype=np.float32)

    cut = int(np.floor(n * (1.0 - eval_frac)))
    cut = max(1, min(n - 1, cut))

    x_tr, y_tr = x_i[:cut], y_i[:cut]
    x_va, y_va = x_i[cut:], y_i[cut:]
    return x_tr, y_tr, x_va, y_va


def _make_sequences(x_i: np.ndarray, y_i: np.ndarray, history_len: int):
    history_len = max(1, int(history_len))
    if history_len == 1:
        return x_i, y_i

    n = len(x_i)
    if n < history_len:
        return np.empty((0, history_len, x_i.shape[1]), dtype=np.float32), np.empty((0, y_i.shape[1]), dtype=np.float32)

    xs = []
    ys = []
    for end in range(history_len - 1, n):
        start = end - history_len + 1
        xs.append(x_i[start:end + 1])
        ys.append(y_i[end])

    x_seq = np.asarray(xs, dtype=np.float32)
    y_seq = np.asarray(ys, dtype=np.float32)
    return x_seq, y_seq


def _actor_output(actor: torch.nn.Module, xb: torch.Tensor) -> torch.Tensor:
    out = actor(xb)
    if isinstance(out, tuple):
        return out[0]
    return out


def main():
    with open(ATT_ROOT / "model.json", encoding="utf-8") as f:
        model_cfg = json.load(f)

    with open(ALGO_ROOT / "config.json", encoding="utf-8") as f:
        train_cfg = json.load(f)

    with open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8") as f:
        parameters = json.load(f)

    model_cfg["actor"]["parameters"] = str(PROJECT_ROOT / "data" / "parameters.json")

    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    eval_frac = float(train_cfg["training"]["eval"])
    grad_clip_norm = float(train_cfg["training"].get("grad_clip_norm", 1.0))

    for tariff in ["tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat"]:
        folder = PROJECT_ROOT / "Results" / "train" / "ATT" / "1-IL" / tariff
        folder.mkdir(parents=True, exist_ok=True)

        hpo = HPO(train_cfg, model_cfg, Teacher, PROJECT_ROOT / "data" / "parameters.json")
        hpo.run(tariff)

        lr = hpo.best_params["lr"]
        batch_size = hpo.best_params["batch_size"]
        weight_decay = hpo.best_params["weight_decay"]
        history_len = int(hpo.best_params.get("history_len", train_cfg["training"].get("history_len", 1)))

        input_dim = int(model_cfg["actor"]["input_dim"])
        output_dim = int(model_cfg["actor"]["output_dim"])

        train_seq_x: list[np.ndarray] = []
        train_seq_y: list[np.ndarray] = []
        val_seq_x: list[np.ndarray] = []
        val_seq_y: list[np.ndarray] = []
        raw_train_count = 0
        raw_val_count = 0

        for run in train_cfg["runs"]:
            df = pd.read_csv(
                run["dataset"],
                sep=";",
                parse_dates=["timestamp"],
                dayfirst=True,
                index_col="timestamp",
            )
            date = datetime.strptime(run["date"], "%Y-%m-%d %H:%M:%S")

            teacher = Teacher(df, parameters, date, run["days"], run["soc"], tariff)
            teacher.build()
            teacher.solve()

            x_i, y_i = teacher.get_training_data()
            x_i = x_i.astype(np.float32, copy=False)
            y_i = y_i.astype(np.float32, copy=False)

            x_i_tr, y_i_tr, x_i_va, y_i_va = _chronological_split(x_i, y_i, eval_frac)
            raw_train_count += len(x_i_tr)
            raw_val_count += len(x_i_va)

            x_i_tr_seq, y_i_tr_seq = _make_sequences(x_i_tr, y_i_tr, history_len)
            x_i_va_seq, y_i_va_seq = _make_sequences(x_i_va, y_i_va, history_len)

            if len(x_i_tr_seq) > 0:
                train_seq_x.append(x_i_tr_seq)
                train_seq_y.append(y_i_tr_seq)
            if len(x_i_va_seq) > 0:
                val_seq_x.append(x_i_va_seq)
                val_seq_y.append(y_i_va_seq)

        if train_seq_x:
            X_tr_seq = np.concatenate(train_seq_x, axis=0)
            y_tr_seq = np.concatenate(train_seq_y, axis=0)
        else:
            X_tr_seq = np.empty((0, history_len, input_dim), dtype=np.float32)
            y_tr_seq = np.empty((0, output_dim), dtype=np.float32)

        if val_seq_x:
            X_va_seq = np.concatenate(val_seq_x, axis=0)
            y_va_seq = np.concatenate(val_seq_y, axis=0)
        else:
            X_va_seq = np.empty((0, history_len, input_dim), dtype=np.float32)
            y_va_seq = np.empty((0, output_dim), dtype=np.float32)

        print(f"Data loaded: {raw_train_count + raw_val_count} samples (train: {raw_train_count}, val: {raw_val_count})")

        train_dataset = TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq))
        val_dataset   = TensorDataset(torch.FloatTensor(X_va_seq), torch.FloatTensor(y_va_seq))

        print(f"Sequence data: history_len={history_len} (train: {len(train_dataset)}, val: {len(val_dataset)})")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = load_actor(model_cfg["actor"], device=DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val, patience = float("inf"), 0

        for epoch in (p_outer := tqdm(
            range(train_cfg["training"]["epochs"]),
            desc=f"Training Actor - {tariff}",
            position=0,
            dynamic_ncols=True
        )):
            model.train()
            for step_idx, (xb, yb) in enumerate((p_inner := tqdm(
                train_loader,
                desc="  Training",
                position=1,
                leave=False,
                dynamic_ncols=True
            )), start=1):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(_actor_output(model, xb), yb)

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite train loss detected at tariff={tariff}, epoch={epoch}, step={step_idx}."
                    )

                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

                p_inner.set_postfix({"train_loss": f"{loss.item():.4f}"})

            model.eval()
            val_loss = 0.0
            if len(val_loader) > 0:
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        val_loss += criterion(_actor_output(model, xb), yb).item()
                val_loss /= len(val_loader)

            if not np.isfinite(val_loss):
                raise RuntimeError(f"Non-finite val_loss detected at tariff={tariff}, epoch={epoch}.")

            p_outer.set_postfix({"val_loss": f"{val_loss:.4f}"})

            if val_loss < best_val:
                torch.save(model.state_dict(), folder / "best.pth")
                with open(folder / "best_params.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "lr": lr,
                        "batch_size": batch_size,
                        "weight_decay": weight_decay,
                        "history_len": history_len,
                        "val_loss": float(val_loss)
                    }, f, indent=4)

                best_val, patience = val_loss, 0
                with open(folder / "actor_cfg.json", "w", encoding="utf-8") as f:
                    actor_cfg_out = dict(model_cfg["actor"])
                    actor_cfg_out["history_len"] = history_len
                    json.dump(actor_cfg_out, f, indent=4)
            else:
                patience += 1
                if patience >= train_cfg["training"]["patience"]:
                    break

        if not (folder / "best.pth").exists():
            raise RuntimeError(f"Training finished without best checkpoint for tariff={tariff}.")


if __name__ == "__main__":
    main()
