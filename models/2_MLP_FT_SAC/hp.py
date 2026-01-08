# models/2_MLP_FT_SAC/hp.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


class HP:
    def __init__(self, d: dict[str, Any] | None = None):
        self._d = dict(d or {})
        for k, v in self._d.items():
            if isinstance(k, str) and k.isidentifier():
                setattr(self, k, v)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None):
        return cls(d)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._d)


@dataclass(frozen=True)
class RunConfig:
    d: dict[str, Any]

    @classmethod
    def from_json(cls, p: Path):
        with open(p, "r", encoding="utf-8") as fp:
            return cls(json.load(fp))

    def exp(self) -> dict[str, Any]:
        return self.d.get("experiment", {}) or {}

    def sac(self) -> dict[str, Any]:
        return self.d.get("sac", {}) or {}

    def eval(self) -> dict[str, Any]:
        return self.d.get("evaluation", {}) or {}

    def schedule(self) -> dict[str, Any]:
        return self.d.get("exploration_schedule", {}) or {}

    def name(self) -> str:
        return str(self.exp().get("name", "exp"))

    def seed(self) -> int:
        return int(self.exp().get("seed", 42))

    def device(self) -> str:
        return str(self.exp().get("device", "cuda"))

    def results_root(self) -> str:
        return str(self.exp().get("results_root", "Results"))

    def model_id(self) -> str:
        return str(self.exp().get("model_id", "2_MLP_FT_SAC"))

    def model_dir(self) -> str:
        return str(self.exp().get("model_dir", "models/2_MLP_FT_SAC"))

    def tariffs(self) -> list[str]:
        return list(self.exp().get("tariffs", []) or [])

    def paths_by_tariff(self, tariff: str) -> dict[str, Any]:
        return (self.d.get("paths_by_tariff", {}) or {}).get(tariff, {}) or {}

    def val_runs(self) -> list[dict[str, Any]]:
        return list(self.d.get("val_runs", []) or [])

    def train_runs(self) -> list[dict[str, Any]]:
        return list(self.d.get("train_runs", []) or [])

    def test_runs(self) -> list[dict[str, Any]]:
        return list(self.d.get("test_runs", []) or [])

    def base_cfg_path(self) -> str:
        return str(self.d.get("config", "data/parameters.json"))

    def sac_common(self) -> dict[str, Any]:
        return (self.sac().get("common", {}) or {})

    def sac_actor(self) -> dict[str, Any]:
        return (self.sac().get("actor", {}) or {})

    def sac_actor_opt(self) -> dict[str, Any]:
        return (self.sac().get("actor_optimizer", {}) or {})

    def sac_crit(self) -> dict[str, Any]:
        return (self.sac().get("critics", {}) or {})

    def sac_crit_arch(self) -> dict[str, Any]:
        return (self.sac_crit().get("architecture", {}) or {})

    def sac_crit_opt(self) -> dict[str, Any]:
        return (self.sac_crit().get("optimizer", {}) or {})

    def sac_target_update(self) -> dict[str, Any]:
        return (((self.sac_crit().get("training", {}) or {}).get("target_update", {}) or {}))

    def sac_entropy(self) -> dict[str, Any]:
        return (self.sac().get("entropy", {}) or {})

    def eval_every_steps(self) -> int:
        return int(self.eval().get("eval_every_steps", 5000))

    def deterministic_eval(self) -> bool:
        return bool(self.eval().get("deterministic_eval", True))

    def tol_rel(self) -> float:
        return float(((((self.eval().get("best_model_selection", {}) or {}).get("primary_constraint", {}) or {}).get("tolerance_rel", 0.0))))

    def min_impr_rel(self) -> float:
        return float(((self.eval().get("best_model_selection", {}) or {}).get("min_improvement_rel", 0.0)))

    def stages(self) -> list[dict[str, Any]]:
        return list((self.schedule().get("stages", []) or []))

    def stage(self, stage_id: int) -> dict[str, Any]:
        return next((s for s in self.stages() if int(s.get("stage_id", -1)) == int(stage_id)), {}) or {}

    def stage_alpha_max(self, stage_id: int, default_max: float) -> float:
        return float((((self.stage(stage_id).get("alpha_bounds", {}) or {}).get("alpha_max", default_max))))

    def sample_ep_days(self, stage_id: int, fallback_days: int) -> int:
        sam = (self.stage(stage_id).get("episode_days_sampler", {}) or {})
        vals = sam.get("values", [fallback_days])
        w = sam.get("weights", None)
        if w is None:
            return int(np.random.choice(vals))
        w = np.array(w, dtype=float)
        return int(np.random.choice(vals, p=w / max(w.sum(), 1e-12)))


class ReplayBuffer:
    def __init__(self, n: int, obs_dim: int, act_dim: int):
        self.n = int(n)
        self.i = 0
        self.full = False
        self.o = np.zeros((self.n, obs_dim), dtype=np.float32)
        self.a = np.zeros((self.n, act_dim), dtype=np.float32)
        self.r = np.zeros((self.n, 1), dtype=np.float32)
        self.no = np.zeros((self.n, obs_dim), dtype=np.float32)
        self.d = np.zeros((self.n, 1), dtype=np.float32)

    def add(self, o, a, r, no, d):
        self.o[self.i] = o
        self.a[self.i] = a
        self.r[self.i, 0] = r
        self.no[self.i] = no
        self.d[self.i, 0] = d
        self.i += 1
        if self.i >= self.n:
            self.i = 0
            self.full = True

    def size(self) -> int:
        return self.n if self.full else self.i

    def sample(self, bs: int, device: torch.device):
        idx = np.random.randint(0, max(self.size(), 1), size=int(bs))
        return (
            torch.as_tensor(self.o[idx], device=device),
            torch.as_tensor(self.a[idx], device=device),
            torch.as_tensor(self.r[idx], device=device),
            torch.as_tensor(self.no[idx], device=device),
            torch.as_tensor(self.d[idx], device=device),
        )
