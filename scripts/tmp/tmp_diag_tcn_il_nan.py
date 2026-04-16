from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models"
TCN_ROOT = MODEL_ROOT / "TCN"
ALGO_ROOT = TCN_ROOT / "1-IL"

import sys

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TCN_ROOT))
sys.path.insert(0, str(MODEL_ROOT))
sys.path.append(str(ALGO_ROOT))

from model import load_actor
from opt import Teacher
from train import _chronological_split, _make_sequences, _actor_output

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def finite_report(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    print(
        f"[{name}] shape={arr.shape} nan={np.isnan(arr).sum()} inf={np.isinf(arr).sum()} "
        f"max_abs={np.nanmax(np.abs(arr)) if arr.size else 0.0:.6f}"
    )


def tensor_stats(name: str, t: torch.Tensor) -> str:
    t_cpu = t.detach().float().cpu()
    finite = torch.isfinite(t_cpu)
    nan_count = int(torch.isnan(t_cpu).sum().item())
    inf_count = int(torch.isinf(t_cpu).sum().item())
    max_abs = float(torch.nan_to_num(t_cpu.abs(), nan=0.0, posinf=0.0, neginf=0.0).max().item()) if t_cpu.numel() else 0.0
    return f"{name}: shape={tuple(t_cpu.shape)} finite={bool(finite.all())} nan={nan_count} inf={inf_count} max_abs={max_abs:.6f}"


def named_param_finite_report(model: nn.Module) -> list[str]:
    lines: list[str] = []
    for name, p in model.named_parameters():
        p_det = p.detach()
        finite = bool(torch.isfinite(p_det).all())
        pmax = float(torch.nan_to_num(p_det.abs(), nan=0.0, posinf=0.0, neginf=0.0).max().item()) if p_det.numel() else 0.0
        lines.append(f"param[{name}] finite={finite} max_abs={pmax:.6f} shape={tuple(p_det.shape)}")
    return lines


def main() -> None:
    with open(TCN_ROOT / "model.json", encoding="utf-8") as f:
        model_cfg = json.load(f)

    with open(ALGO_ROOT / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    with open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8") as f:
        parameters = json.load(f)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Best params observed in prior run log
    tariff = "tar_s"
    lr = 3.539042265033424e-05
    batch_size = 32
    weight_decay = 3.930309320734253e-05
    history_len = 192
    eval_frac = float(cfg["training"]["eval"])
    grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 1.0))

    print("=== TCN IL NAN DIAG ===")
    print(f"tariff={tariff} lr={lr} batch_size={batch_size} wd={weight_decay} history_len={history_len}")

    input_dim = int(model_cfg["actor"]["input_dim"])
    output_dim = int(model_cfg["actor"]["output_dim"])

    train_seq_x: list[np.ndarray] = []
    train_seq_y: list[np.ndarray] = []
    val_seq_x: list[np.ndarray] = []
    val_seq_y: list[np.ndarray] = []

    for run in cfg["runs"]:
        print(f"[teacher] {run['name']} {run['date']} days={run['days']}")
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

    finite_report("X_tr_seq", X_tr_seq)
    finite_report("y_tr_seq", y_tr_seq)
    finite_report("X_va_seq", X_va_seq)
    finite_report("y_va_seq", y_va_seq)

    if not np.isfinite(X_tr_seq).all() or not np.isfinite(y_tr_seq).all():
        print("[diagnosis] dataset already contains non-finite values")
        return

    train_ds = TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model_cfg["actor"]["parameters"] = str(PROJECT_ROOT / "data" / "parameters.json")
    model = load_actor(model_cfg["actor"], device=DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    print(f"[train] batches={len(train_loader)}")

    for step_idx, (xb, yb) in enumerate(train_loader, start=1):
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        pred = _actor_output(model, xb)
        loss = criterion(pred, yb)

        if not torch.isfinite(loss):
            print("[diagnosis] NON-FINITE LOSS")
            print(f"step={step_idx}")
            print(tensor_stats("xb", xb))
            print(tensor_stats("yb", yb))
            print(tensor_stats("pred", pred))
            print(tensor_stats("loss", loss.reshape(1)))
            break

        loss.backward()

        grad_nan = False
        bad_grad_reports: list[str] = []
        grad_max_abs = 0.0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if not torch.isfinite(g).all():
                grad_nan = True
                g_nan = int(torch.isnan(g).sum().item())
                g_inf = int(torch.isinf(g).sum().item())
                gmax = float(torch.nan_to_num(g.abs(), nan=0.0, posinf=0.0, neginf=0.0).max().item()) if g.numel() else 0.0
                bad_grad_reports.append(
                    f"grad[{name}] finite=False nan={g_nan} inf={g_inf} max_abs={gmax:.6f} shape={tuple(g.shape)}"
                )
                continue
            gmax = float(g.abs().max().item())
            if gmax > grad_max_abs:
                grad_max_abs = gmax

        if grad_nan:
            print("[diagnosis] NON-FINITE GRADIENT")
            print(f"step={step_idx} loss={float(loss.detach().cpu()):.6f}")
            print(tensor_stats("xb", xb))
            print(tensor_stats("yb", yb))
            print(tensor_stats("pred", pred))
            for line in bad_grad_reports[:20]:
                print(line)
            for line in named_param_finite_report(model)[:20]:
                print(line)
            break

        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        with torch.no_grad():
            param_max_abs = 0.0
            for p in model.parameters():
                pmax = float(p.detach().abs().max().item())
                if pmax > param_max_abs:
                    param_max_abs = pmax

        if step_idx % 500 == 0 or step_idx == 1:
            print(
                f"[step {step_idx}] loss={float(loss.detach().cpu()):.6f} "
                f"grad_max_abs={grad_max_abs:.6f} param_max_abs={param_max_abs:.6f}"
            )

    else:
        print("[diagnosis] epoch finished without non-finite loss")


if __name__ == "__main__":
    main()
