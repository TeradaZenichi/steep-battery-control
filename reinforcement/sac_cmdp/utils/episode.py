"""Samples training episode start times from CY/WY datasets, avoiding eval windows."""
import pandas as pd
import numpy as np


class EpisodeGen:
    def __init__(self, config, data_path, seed=42):
        base = str(data_path).rstrip("/\\")
        load = lambda p: pd.read_csv(p, sep=";", parse_dates=["timestamp"], dayfirst=True, index_col="timestamp")
        self.df_cy = load(f"{base}/Simulation_CY_Cur_HP__PV5000-HB5000.csv")
        self.df_wy = load(f"{base}/Simulation_WY_Cur_HP__PV5000-HB5000.csv")
        self.days = int(config["train"]["days"])
        self.duration = pd.Timedelta(days=self.days)
        self.eval_windows = self._windows(config["val"])
        # Leave 7 days of warmup at the start for lag_7d features.
        self.warmup = pd.Timedelta(days=7)
        self.rng = np.random.default_rng(int(seed))

    @staticmethod
    def _windows(val_cfg):
        w = {"cy": [], "wy": []}
        for it in val_cfg:
            t0 = pd.to_datetime(it["date"])
            t1 = t0 + pd.Timedelta(days=it["days"])
            key = "cy" if "cy" in it["dataset"].lower() else "wy"
            w[key].append((t0, t1))
        return w

    def _valid(self, key):
        df = self.df_cy if key == "cy" else self.df_wy
        first = df.index.min() + self.warmup
        last = df.index.max() - self.duration
        cands = df.index[(df.index >= first) & (df.index <= last)]
        blocked = self.eval_windows[key]
        return pd.DatetimeIndex([t for t in cands if not any(t < e and t + self.duration > s for s, e in blocked)])

    @staticmethod
    def _key(dataset):
        ds = str(dataset).lower()
        if "cy" in ds: return "cy"
        if "wy" in ds: return "wy"
        raise ValueError(f"Unknown dataset: {dataset}")

    def sample(self, dataset):
        cands = self._valid(self._key(dataset))
        return cands[self.rng.integers(0, len(cands))]

    def load(self, dataset):
        return self.df_cy if self._key(dataset) == "cy" else self.df_wy
