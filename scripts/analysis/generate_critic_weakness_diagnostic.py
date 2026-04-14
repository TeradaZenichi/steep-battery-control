from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = PROJECT_ROOT / "Results" / "train"
OUT_ROOT = PROJECT_ROOT / "Results" / "analysis" / "new"
APPROACH_DIR = "2-RL"
TAIL_FRAC = 0.20
MIN_TAIL_STEPS = 10


def _tail_slice(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    tail_n = max(MIN_TAIL_STEPS, int(np.ceil(n * TAIL_FRAC)))
    tail_n = min(tail_n, n)
    return df.tail(tail_n).copy()


def _slope(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _load_metrics(path: Path, model: str, tariff: str) -> dict[str, float | int | str]:
    df = pd.read_csv(path).sort_values("episode").reset_index(drop=True)
    tail = _tail_slice(df)

    q1 = tail["q1_mean"].to_numpy(dtype=float)
    q2 = tail["q2_mean"].to_numpy(dtype=float)
    q_avg = 0.5 * (q1 + q2)
    backup = tail["backup_mean"].to_numpy(dtype=float)
    critic_loss = tail["critic_loss"].to_numpy(dtype=float)

    q_backup_gap = q_avg - backup
    q_disagreement = q1 - q2

    critic_loss_mean = float(np.mean(critic_loss))
    critic_loss_std = float(np.std(critic_loss, ddof=0))

    metrics = {
        "model": model,
        "tariff": tariff,
        "n_rows": int(len(df)),
        "tail_rows": int(len(tail)),
        "critic_loss_tail_mean": critic_loss_mean,
        "critic_loss_tail_std": critic_loss_std,
        "critic_loss_tail_cv": float(critic_loss_std / (abs(critic_loss_mean) + 1e-12)),
        "critic_loss_slope_tail": _slope(critic_loss),
        "abs_critic_loss_slope_tail": float(abs(_slope(critic_loss))),
        "abs_q_backup_gap_tail": float(np.mean(np.abs(q_backup_gap))),
        "q_backup_bias_tail": float(np.mean(q_backup_gap)),
        "q_disagreement_tail": float(np.mean(np.abs(q_disagreement))),
        "tail_non_finite_rows": int((~np.isfinite(tail[["q1_mean", "q2_mean", "backup_mean", "critic_loss"]]).all(axis=1)).sum()),
        "end_no_improve_evals": int(df["no_improve_evals"].iloc[-1]),
        "end_no_improve_episodes": int(df["no_improve_episodes"].iloc[-1]),
    }
    return metrics


def _add_weakness_score(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "critic_loss_tail_mean",
        "abs_q_backup_gap_tail",
        "q_disagreement_tail",
        "critic_loss_tail_cv",
        "abs_critic_loss_slope_tail",
    ]

    out = df.copy()
    pct_cols: list[str] = []

    for col in score_cols:
        pct_col = f"{col}_pct"
        out[pct_col] = out[col].rank(pct=True, method="average")
        pct_cols.append(pct_col)

    out["critic_weakness_score"] = 100.0 * out[pct_cols].mean(axis=1)

    q_low = out["critic_weakness_score"].quantile(0.33)
    q_high = out["critic_weakness_score"].quantile(0.66)

    def _band(v: float) -> str:
        if v <= q_low:
            return "low"
        if v <= q_high:
            return "medium"
        return "high"

    out["critic_weakness_band"] = out["critic_weakness_score"].map(_band)
    return out


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    paths = sorted(TRAIN_ROOT.glob(f"*/{APPROACH_DIR}/*/audit_training.csv"))

    records: list[dict[str, float | int | str]] = []
    for path in paths:
        rel = path.relative_to(TRAIN_ROOT)
        model = rel.parts[0]
        tariff = rel.parts[2]
        records.append(_load_metrics(path, model=model, tariff=tariff))

    by_model_tariff = pd.DataFrame(records)
    by_model_tariff = _add_weakness_score(by_model_tariff)
    by_model_tariff = by_model_tariff.sort_values(["critic_weakness_score", "model", "tariff"], ascending=[False, True, True]).reset_index(drop=True)

    by_model = (
        by_model_tariff
        .groupby("model", as_index=False)
        .agg(
            n_tariffs=("tariff", "count"),
            critic_weakness_score_mean=("critic_weakness_score", "mean"),
            critic_weakness_score_median=("critic_weakness_score", "median"),
            critic_loss_tail_mean=("critic_loss_tail_mean", "mean"),
            abs_q_backup_gap_tail_mean=("abs_q_backup_gap_tail", "mean"),
            q_disagreement_tail_mean=("q_disagreement_tail", "mean"),
            critic_loss_tail_cv_mean=("critic_loss_tail_cv", "mean"),
            abs_critic_loss_slope_tail_mean=("abs_critic_loss_slope_tail", "mean"),
            high_band_count=("critic_weakness_band", lambda s: int((s == "high").sum())),
        )
        .sort_values("critic_weakness_score_mean", ascending=False)
        .reset_index(drop=True)
    )

    out_model_tariff = OUT_ROOT / "critic_weakness_by_model_tariff.csv"
    out_model = OUT_ROOT / "critic_weakness_by_model.csv"

    by_model_tariff.to_csv(out_model_tariff, index=False, encoding="utf-8")
    by_model.to_csv(out_model, index=False, encoding="utf-8")

    print(f"Saved: {out_model_tariff}")
    print(f"Saved: {out_model}")


if __name__ == "__main__":
    main()
