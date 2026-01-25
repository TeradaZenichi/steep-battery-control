from torch.utils.data import DataLoader, TensorDataset, random_split
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import optuna
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



def main():
    with open(Path(__file__).resolve().parent.parent / "model.json") as f:
        model_cfg = json.load(f)
    
    with open(Path(__file__).resolve().parent / "config.json") as f:
        train_cfg = json.load(f)

    # PROJECT_ROOT / "data" / "parameters.json
    with open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8") as f:
        parameters = json.load(f)
   
   
    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    for tariff in ["tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat"]:
        folder = PROJECT_ROOT / "Results" / "train" / "MLP" / "1-IL" / tariff
        folder.mkdir(parents=True, exist_ok=True)
        hpo = HPO(train_cfg, model_cfg, Teacher, PROJECT_ROOT / "data" / "parameters.json")
        hpo.run(tariff)
        
        X, y = np.empty((0, 23), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
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
            X, y = np.vstack([X, x_i]), np.vstack([y, y_i])
        
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        val_size = int(len(dataset) * train_cfg["training"]["eval"])
        train_dataset, val_dataset = random_split(
            dataset, [len(dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(train_cfg["seed"])
        )
        print(f"Data loaded: {len(X)} samples (train: {len(train_dataset)}, val: {len(val_dataset)})")
        # 1. Define modelo Actor
        
        lr = hpo.best_params["lr"]
        batch_size = hpo.best_params["batch_size"]
        weight_decay = hpo.best_params["weight_decay"]
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        model = load_actor(model_cfg["actor"], device=DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val, patience = float("inf"), 0
        
        for epoch in (p_outer := tqdm(range(train_cfg["training"]["epochs"]), desc=f"Training Actor - {tariff}", position=0, dynamic_ncols=True)):
            model.train()
            for xb, yb in (p_inner := tqdm(train_loader, desc="  Training", position=1, leave=False, dynamic_ncols=True)):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

                # add loss to progress bar
                p_inner.set_postfix({"train_loss": f"{loss.item():.4f}"})
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    val_loss += criterion(model(xb), yb).item()
            val_loss /= len(val_loader)
            # add val_loss to progress bar
            p_outer.set_postfix({"val_loss": f"{val_loss:.4f}"})
            
            if val_loss < best_val:
                torch.save(model.state_dict(), folder / "best.pth")
                # save json with best results
                with open(folder / "best_params.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "lr": lr,
                        "batch_size": batch_size,
                        "weight_decay": weight_decay,
                        "val_loss": val_loss
                    }, f, indent=4)
                best_val, patience = val_loss, 0
                with open(folder / "actor_cfg.json", "w", encoding="utf-8") as f:
                    json.dump(model_cfg["actor"], f, indent=4)

            else:
                patience += 1
                if patience >= train_cfg["training"]["patience"]:
                    break


if __name__ == "__main__":
    main()