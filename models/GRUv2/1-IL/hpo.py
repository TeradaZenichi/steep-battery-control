import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import optuna
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from model import load_actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _actor_output(actor: torch.nn.Module, xb: torch.Tensor) -> torch.Tensor:
    out = actor(xb)
    if isinstance(out, tuple):
        return out[0]
    return out


def _chronological_split(x_i: np.ndarray, y_i: np.ndarray, eval_frac: float):
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


class HPO:
    def __init__(self, cfg, model_cfg, teacher_class, params_path):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.teacher_class = teacher_class
        self.params_path = params_path
        self.search_splits = []

        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    def load_data(self, tariff):
        """Load data from all searches and cache per-run chronological splits."""
        self.search_splits = []

        eval_frac = float(self.cfg["optuna"]["eval"])

        with open(self.params_path, encoding="utf-8") as f:
            params = json.load(f)

        for search in self.cfg["searches"]:
            df = pd.read_csv(
                search["dataset"],
                sep=";",
                parse_dates=["timestamp"],
                dayfirst=True,
                index_col="timestamp",
            )
            date = datetime.strptime(search["date"], "%Y-%m-%d %H:%M:%S")

            teacher = self.teacher_class(df, params, date, search["days"], search["soc"], tariff)
            teacher.build()
            teacher.solve()

            x_i, y_i = teacher.get_training_data()
            x_i = x_i.astype(np.float32, copy=False)
            y_i = y_i.astype(np.float32, copy=False)

            x_i_tr, y_i_tr, x_i_va, y_i_va = _chronological_split(x_i, y_i, eval_frac)
            self.search_splits.append((x_i_tr, y_i_tr, x_i_va, y_i_va))


    def _build_datasets(self, history_len: int):
        x_tr_list, y_tr_list = [], []
        x_va_list, y_va_list = [], []

        for x_i_tr, y_i_tr, x_i_va, y_i_va in self.search_splits:
            x_tr_seq, y_tr_seq = _make_sequences(x_i_tr, y_i_tr, history_len)
            x_va_seq, y_va_seq = _make_sequences(x_i_va, y_i_va, history_len)

            if len(x_tr_seq) > 0:
                x_tr_list.append(x_tr_seq)
                y_tr_list.append(y_tr_seq)
            if len(x_va_seq) > 0:
                x_va_list.append(x_va_seq)
                y_va_list.append(y_va_seq)

        if x_tr_list:
            X_tr = np.concatenate(x_tr_list, axis=0)
            y_tr = np.concatenate(y_tr_list, axis=0)
        else:
            X_tr = np.empty((0, history_len, self.model_cfg["actor"]["input_dim"]), dtype=np.float32)
            y_tr = np.empty((0, self.model_cfg["actor"]["output_dim"]), dtype=np.float32)

        if x_va_list:
            X_va = np.concatenate(x_va_list, axis=0)
            y_va = np.concatenate(y_va_list, axis=0)
        else:
            X_va = np.empty((0, history_len, self.model_cfg["actor"]["input_dim"]), dtype=np.float32)
            y_va = np.empty((0, self.model_cfg["actor"]["output_dim"]), dtype=np.float32)

        train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        val_dataset = TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(y_va))
        return train_dataset, val_dataset

    def objective(self, trial):
        """Optuna objective function."""
        ss = self.cfg["optuna"]["search_space"]
        lr = trial.suggest_float("lr", ss["lr"][0], ss["lr"][1], log=True)
        batch_size = trial.suggest_categorical("batch_size", ss["batch_size"])
        weight_decay = trial.suggest_float("weight_decay", ss["weight_decay"][0], ss["weight_decay"][1])
        history_len = trial.suggest_categorical("history_len", ss.get("history_len", [self.cfg["training"].get("history_len", 1)]))

        train_dataset, val_dataset = self._build_datasets(history_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = load_actor(self.model_cfg["actor"], device=DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val, patience = float("inf"), 0

        for epoch in range(self.cfg["optuna"]["epochs"]):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(_actor_output(model, xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            if len(val_loader) > 0:
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        val_loss += criterion(_actor_output(model, xb), yb).item()
                val_loss /= len(val_loader)

            if val_loss < best_val:
                best_val, patience = val_loss, 0
            else:
                patience += 1
                if patience >= self.cfg["optuna"]["patience"]:
                    break

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val

    def run(self, tariff):
        """Run HPO and return best parameters."""
        self.load_data(tariff)

        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        study.optimize(self.objective, n_trials=self.cfg["optuna"]["n_trials"], show_progress_bar=True)

        print(f"\nBest val_loss: {study.best_value:.6f}")
        print(f"Best params: {study.best_params}")
        self.best_params = study.best_params
        return study.best_params
