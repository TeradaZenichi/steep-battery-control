"""Vanilla audit CSV writer: collects per-update step metrics and aggregates per episode."""
from pathlib import Path
import pandas as pd
import numpy as np


def _finite(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def mean_or_nan(values):
    a = _finite(values)
    return float(np.mean(a)) if a.size else np.nan


class Audit:
    def __init__(self, csv_path, every=5):
        self.path = Path(csv_path)
        self.every = int(every)
        self.df = pd.DataFrame()
        self._pending = []
        self._steps = []

    def open(self):
        self._steps.clear()

    def step(self, **metrics):
        self._steps.append(metrics)

    def aggregate(self):
        sl = self._steps
        col = lambda k: [s.get(k, np.nan) for s in sl]
        n_critic = len(sl)
        n_actor = sum(1 for s in sl if np.isfinite(s.get("actor_loss", np.nan)))
        return {
            "n_critic_updates": n_critic,
            "n_actor_updates": n_actor,
            "critic_loss": mean_or_nan(col("critic_loss")),
            "critic_main_loss": mean_or_nan(col("critic_main_loss")),
            "actor_loss": mean_or_nan(col("actor_loss")),
            "alpha_loss": mean_or_nan(col("alpha_loss")),
            "q1_mean": mean_or_nan(col("q1_mean")),
            "q2_mean": mean_or_nan(col("q2_mean")),
            "q_return_corr": mean_or_nan(col("q_return_corr")),
            "backup_mean": mean_or_nan(col("backup_mean")),
            "q_pi_mean": mean_or_nan(col("q_pi_mean")),
            "q_pi_std_mean": mean_or_nan(col("q_pi_std_mean")),
            "logp_mean": mean_or_nan(col("logp_mean")),
            "violation_mean": mean_or_nan(col("viol_mean")),
            "proj_delta_mean": mean_or_nan(col("proj_delta_mean")),
            "kl": mean_or_nan(col("kl")),
            "kl_from_best": mean_or_nan(col("kl_from_best")),
            "kl_beta": mean_or_nan(col("kl_beta")),
            "alpha": mean_or_nan(col("alpha")),
            "lambda_value": mean_or_nan(col("lambda_value")),
            "actor_lr": mean_or_nan(col("actor_lr")),
            "critic_lr": mean_or_nan(col("critic_lr")),
        }

    def add(self, row):
        self._pending.append(row)

    def flush(self, force=False):
        if not self._pending:
            return
        if force or len(self._pending) >= self.every:
            self.df = pd.concat([self.df, pd.DataFrame(self._pending)], ignore_index=True)
            self._pending.clear()
            if "episode" in self.df.columns:
                self.df = self.df.sort_values("episode", ignore_index=True)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.df.to_csv(self.path, index=False)
            except PermissionError:
                self.df.to_csv(self.path.with_name(f"{self.path.stem}.recovery{self.path.suffix}"), index=False)
