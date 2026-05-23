"""Multiprocess evaluation. Worker is generic: takes the actor module path
(e.g. 'models.GRU.model'), dynamically imports it, and runs an episode.

The runner caches the actor + dataset per worker to avoid reload overhead.
"""
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import torch
import sys


def _run_episode(env, actor, history_len, deterministic):
    obs, _ = env.reset()
    hist = deque([obs.copy() for _ in range(history_len)], maxlen=history_len)
    total_reward = 0.0
    metrics = {"energy_cost": 0.0, "bess_cost": 0.0, "ev_cost": 0.0,
               "pv_cost": 0.0, "grid_penalty": 0.0, "bess_abs_power": 0.0,
               "ev_abs_power": 0.0, "pv_cmd": 0.0, "steps": 0}
    done = truncated = False
    while not (done or truncated):
        seq = np.stack(hist, axis=0)
        x = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            a = actor.act(x, deterministic=deterministic).squeeze(0).cpu().numpy()
        obs, reward, done, truncated, info = env.step(a)
        hist.append(obs.copy())
        total_reward += float(reward)
        metrics["energy_cost"] += float(info["energy_cost"])
        metrics["bess_cost"] += float(info["bess_cost"])
        metrics["ev_cost"] += float(info["ev_cost"])
        metrics["pv_cost"] += float(info["pv_cost"])
        metrics["grid_penalty"] += float(info["penalty"])
        metrics["bess_abs_power"] += abs(float(info["pbess"]))
        metrics["ev_abs_power"] += abs(float(info["pev"]))
        metrics["pv_cmd"] += float(a[2])
        metrics["steps"] += 1
    n = max(1, metrics["steps"])
    return {"reward": total_reward, "energy_cost": metrics["energy_cost"] / n,
            "bess_cost": metrics["bess_cost"] / n, "ev_cost": metrics["ev_cost"] / n,
            "pv_cost": metrics["pv_cost"] / n,
            "grid_penalty": metrics["grid_penalty"] / n,
            "bess_abs_power_mean": metrics["bess_abs_power"] / n,
            "ev_abs_power_mean": metrics["ev_abs_power"] / n,
            "pv_cmd_mean": metrics["pv_cmd"] / n}


_DF_CACHE: dict[str, pd.DataFrame] = {}
_ACTOR_CACHE: dict[int, torch.nn.Module] = {}


def _worker(run, parameters, tariff, actor_module, actor_cfg, actor_state, history_len, deterministic, actor_key, project_root):
    # Workers are CPU-bound and many of them run in parallel; let each use a
    # single thread so they don't fight over cores. Without this, eval is
    # 3-5× slower than v4 on a 12-worker pool.
    torch.set_num_threads(1)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from environment import SmartHomeEnv
    mod = importlib.import_module(actor_module)

    ds = Path(run["dataset"])
    if not ds.is_absolute():
        ds = Path(project_root) / ds
    key = str(ds.resolve())
    df = _DF_CACHE.get(key)
    if df is None:
        df = pd.read_csv(key, sep=";", parse_dates=["timestamp"], dayfirst=True, index_col="timestamp")
        _DF_CACHE[key] = df

    actor = _ACTOR_CACHE.get(actor_key)
    if actor is None:
        actor = mod.load_actor(actor_cfg)
        _ACTOR_CACHE[actor_key] = actor
    actor.load_state_dict(actor_state)
    actor.eval()

    env = SmartHomeEnv(df, parameters, start=pd.to_datetime(run["date"], format="%Y-%m-%d %H:%M:%S"),
                       days=int(run["days"]), BESS_SoC=float(run.get("soc", 0.5)),
                       tariff=tariff, track_operation=False)
    out = _run_episode(env, actor, history_len, deterministic)
    out["scenario"] = run.get("name", "")
    return out


class EvalRunner:
    def __init__(self, actor_module, actor_cfg, parameters, tariff, history_len, project_root, n_workers=4):
        self.actor_module = actor_module
        self.actor_cfg = actor_cfg
        self.parameters = parameters
        self.tariff = tariff
        self.history_len = int(history_len)
        self.project_root = str(project_root)
        self.pool = ProcessPoolExecutor(max_workers=int(n_workers))
        # Stable cache key (constant per architecture+tariff) so workers reuse
        # a single Actor instance instead of creating one per eval submission.
        self.actor_key = f"{actor_module}|{tariff}|{int(history_len)}"

    def submit(self, runs, actor_state, deterministic=True):
        return [self.pool.submit(_worker, run, self.parameters, self.tariff, self.actor_module,
                                  self.actor_cfg, actor_state, self.history_len, deterministic,
                                  self.actor_key, self.project_root) for run in runs]

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class AsyncEval:
    """Ordered async evaluation queue over an EvalRunner."""

    def __init__(self, runner, runs, max_pending=2):
        self.runner = runner
        self.runs = list(runs)
        self.max_pending = max(1, int(max_pending))
        self.pending = deque()

    def submit(self, episode, actor_state, snapshot, context):
        futures = self.runner.submit(self.runs, actor_state, deterministic=True)
        self.pending.append(
            {
                "episode": int(episode),
                "futures": futures,
                "snapshot": snapshot,
                "context": context,
            }
        )

    def collect(self, wait=False, max_items=None):
        done = []
        while self.pending:
            job = self.pending[0]
            futures = job["futures"]
            if not wait and not all(f.done() for f in futures):
                break
            self.pending.popleft()
            job["summary"] = summarize([f.result() for f in futures])
            done.append(job)
            if max_items is not None and len(done) >= int(max_items):
                break
        return done

    def drain(self):
        done = []
        while self.pending:
            done.extend(self.collect(wait=True, max_items=1))
        return done


def summarize(results):
    rewards = np.asarray([r["reward"] for r in results], dtype=np.float64)
    sorted_r = np.sort(rewards)
    tail = min(2, len(sorted_r))
    return {
        "mean_reward": float(np.mean(rewards)),
        "worst_reward": float(np.min(rewards)),
        "robust_reward": float(np.mean(sorted_r[:tail])),
        "energy_cost_mean": float(np.mean([r["energy_cost"] for r in results])),
        "bess_cost_mean": float(np.mean([r["bess_cost"] for r in results])),
        "ev_cost_mean": float(np.mean([r["ev_cost"] for r in results])),
        "pv_cost_mean": float(np.mean([r["pv_cost"] for r in results])),
        "grid_penalty_mean": float(np.mean([r["grid_penalty"] for r in results])),
        "bess_abs_power_mean": float(np.mean([r["bess_abs_power_mean"] for r in results])),
        "ev_abs_power_mean": float(np.mean([r["ev_abs_power_mean"] for r in results])),
        "pv_cmd_mean": float(np.mean([r["pv_cmd_mean"] for r in results])),
    }
