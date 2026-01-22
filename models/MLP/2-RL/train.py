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