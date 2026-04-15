from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models"
TCN_ROOT = MODEL_ROOT / "TCN"
ALGO_ROOT = TCN_ROOT / "1-IL"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TCN_ROOT))
sys.path.insert(0, str(MODEL_ROOT))
sys.path.append(str(ALGO_ROOT))

from opt import Teacher
from train import _chronological_split, _make_sequences


def finite_report(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    return {
        "shape": tuple(arr.shape),
        "all_finite": bool(finite.all()),
        "nan_count": int(np.isnan(arr).sum()),
        "posinf_count": int(np.isposinf(arr).sum()),
        "neginf_count": int(np.isneginf(arr).sum()),
        "max_abs": float(np.nanmax(np.abs(arr))) if arr.size else 0.0,
    }


def main() -> None:
    with open(ALGO_ROOT / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    with open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8") as f:
        par = json.load(f)

    tariff = "tar_s"
    eval_frac = float(cfg["training"]["eval"])
    history_len = 192

    print(f"[debug] tariff={tariff} eval_frac={eval_frac} history_len={history_len}")

    X_tr = np.empty((0, 23), dtype=np.float32)
    y_tr = np.empty((0, 3), dtype=np.float32)
    X_va = np.empty((0, 23), dtype=np.float32)
    y_va = np.empty((0, 3), dtype=np.float32)

    for idx, run in enumerate(cfg["runs"], start=1):
        print("\n" + "=" * 80)
        print(f"[run {idx}/{len(cfg['runs'])}] {run['name']} dataset={run['dataset']} date={run['date']} days={run['days']} soc={run['soc']}")

        df = pd.read_csv(
            run["dataset"],
            sep=";",
            parse_dates=["timestamp"],
            dayfirst=True,
            index_col="timestamp",
        )
        start = datetime.strptime(run["date"], "%Y-%m-%d %H:%M:%S")

        teacher = Teacher(df, par, start, run["days"], run["soc"], tariff)
        teacher.build()
        teacher.solve()

        op = teacher.get_operation()
        op_nan = op.isna().sum()
        op_inf = np.isinf(op.select_dtypes(include=[np.number]).to_numpy()).sum()
        print(f"[operation] rows={len(op)} cols={len(op.columns)} nan_total={int(op_nan.sum())} inf_total={int(op_inf)}")
        if int(op_nan.sum()) > 0:
            bad_cols = op_nan[op_nan > 0].sort_values(ascending=False)
            print("[operation] nan_by_col:")
            print(bad_cols.to_string())

        x_i, y_i = teacher.get_training_data()
        x_i = x_i.astype(np.float32, copy=False)
        y_i = y_i.astype(np.float32, copy=False)

        rep_x = finite_report(x_i)
        rep_y = finite_report(y_i)
        print(f"[x_i] {rep_x}")
        print(f"[y_i] {rep_y}")

        x_i_tr, y_i_tr, x_i_va, y_i_va = _chronological_split(x_i, y_i, eval_frac)
        X_tr = np.vstack([X_tr, x_i_tr])
        y_tr = np.vstack([y_tr, y_i_tr])
        X_va = np.vstack([X_va, x_i_va])
        y_va = np.vstack([y_va, y_i_va])

    print("\n" + "#" * 80)
    print("[aggregate raw]")
    print(f"X_tr: {finite_report(X_tr)}")
    print(f"y_tr: {finite_report(y_tr)}")
    print(f"X_va: {finite_report(X_va)}")
    print(f"y_va: {finite_report(y_va)}")

    X_tr_seq, y_tr_seq = _make_sequences(X_tr, y_tr, history_len)
    X_va_seq, y_va_seq = _make_sequences(X_va, y_va, history_len)

    print("\n[aggregate seq]")
    print(f"X_tr_seq: {finite_report(X_tr_seq)}")
    print(f"y_tr_seq: {finite_report(y_tr_seq)}")
    print(f"X_va_seq: {finite_report(X_va_seq)}")
    print(f"y_va_seq: {finite_report(y_va_seq)}")


if __name__ == "__main__":
    main()
