import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import optuna
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import load_actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HPO:
    def __init__(self, cfg, model_cfg, teacher_class, params_path):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.teacher_class = teacher_class
        self.params_path = params_path
        self.train_dataset = None
        self.val_dataset = None
        
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
    
    def load_data(self, tariff):
        """Load data from all searches and split into train/val."""
        X, y = np.empty((0, 23), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        
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
            X, y = np.vstack([X, x_i]), np.vstack([y, y_i])
        
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        val_size = int(len(dataset) * self.cfg["optuna"]["eval"])
        self.train_dataset, self.val_dataset = random_split(
            dataset, [len(dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(self.cfg["seed"])
        )
        print(f"Data loaded: {len(X)} samples (train: {len(self.train_dataset)}, val: {len(self.val_dataset)})")
    
    def objective(self, trial):
        """Optuna objective function."""
        ss = self.cfg["optuna"]["search_space"]
        lr = trial.suggest_float("lr", ss["lr"][0], ss["lr"][1], log=True)
        batch_size = trial.suggest_categorical("batch_size", ss["batch_size"])
        weight_decay = trial.suggest_float("weight_decay", ss["weight_decay"][0], ss["weight_decay"][1])
        
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)
        
        model = load_actor(self.model_cfg["actor"]).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val, patience = float("inf"), 0
        
        for epoch in range(self.cfg["optuna"]["epochs"]):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    val_loss += criterion(model(xb), yb).item()
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
        
        #save the found parameters in self
        #lr
        self.best_params = {
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay
        }
        return best_val
    
    def run(self, tariff):
        """Run HPO and return best parameters."""
        self.load_data(tariff)
        
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        study.optimize(self.objective, n_trials=self.cfg["optuna"]["n_trials"], show_progress_bar=True)
        
        print(f"\nBest val_loss: {study.best_value:.6f}")
        print(f"Best params: {study.best_params}")
        
        return study.best_params