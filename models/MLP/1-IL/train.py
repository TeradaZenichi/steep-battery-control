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
MLP_ROOT     = Path(__file__).resolve().parent.parent   # .../models/MLP
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MLP_ROOT))
sys.path.insert(0, str(MODEL_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from hpo import HPO
from opt import Teacher
from model import load_actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _chronological_split(x_i: np.ndarray, y_i: np.ndarray, eval_frac: float):
    """Split a single run into train/val using the last eval_frac portion as validation."""
    n = len(x_i)
    if n < 2:
        # Degenerate case: keep all in train (val empty)
        return x_i, y_i, np.empty((0, x_i.shape[1]), dtype=np.float32), np.empty((0, y_i.shape[1]), dtype=np.float32)

    cut = int(np.floor(n * (1.0 - eval_frac)))
    cut = max(1, min(n - 1, cut))  # ensure both splits are non-empty

    x_tr, y_tr = x_i[:cut], y_i[:cut]
    x_va, y_va = x_i[cut:], y_i[cut:]
    return x_tr, y_tr, x_va, y_va


def main():
    with open(Path(__file__).resolve().parent.parent / "model.json") as f:
        model_cfg = json.load(f)

    with open(Path(__file__).resolve().parent / "config.json") as f:
        train_cfg = json.load(f)

    with open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8") as f:
        parameters = json.load(f)

    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    eval_frac = float(train_cfg["training"]["eval"])
    grad_clip_norm = float(train_cfg["training"].get("grad_clip_norm", 1.0))

    for tariff in ["tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat"]:
        folder = PROJECT_ROOT / "Results" / "train" / "MLP" / "1-IL" / tariff
        folder.mkdir(parents=True, exist_ok=True)

        # HPO must use the SAME split logic (see hpo.py below)
        hpo = HPO(train_cfg, model_cfg, Teacher, PROJECT_ROOT / "data" / "parameters.json")
        hpo.run(tariff)

        X_tr = np.empty((0, 23), dtype=np.float32)
        y_tr = np.empty((0, 3), dtype=np.float32)
        X_va = np.empty((0, 23), dtype=np.float32)
        y_va = np.empty((0, 3), dtype=np.float32)

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

            X_tr = np.vstack([X_tr, x_i_tr])
            y_tr = np.vstack([y_tr, y_i_tr])
            X_va = np.vstack([X_va, x_i_va])
            y_va = np.vstack([y_va, y_i_va])

        train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        val_dataset   = TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(y_va))

        print(f"Data loaded: {len(X_tr)+len(X_va)} samples (train: {len(train_dataset)}, val: {len(val_dataset)})")

        # Actor hyperparameters
        lr = hpo.best_params["lr"]
        batch_size = hpo.best_params["batch_size"]
        weight_decay = hpo.best_params["weight_decay"]

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
                # BC with forward(): do NOT use projection here (as you requested)
                loss = criterion(model(xb), yb)

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
                        val_loss += criterion(model(xb), yb).item()
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
                        "val_loss": float(val_loss)
                    }, f, indent=4)

                best_val, patience = val_loss, 0
                with open(folder / "actor_cfg.json", "w", encoding="utf-8") as f:
                    json.dump(model_cfg["actor"], f, indent=4)
            else:
                patience += 1
                if patience >= train_cfg["training"]["patience"]:
                    break

        if not (folder / "best.pth").exists():
            raise RuntimeError(f"Training finished without best checkpoint for tariff={tariff}.")


if __name__ == "__main__":
    main()
