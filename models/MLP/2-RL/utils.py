from pathlib import Path
import pandas as pd
import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../steep-battery-control
MODEL_ROOT   = Path(__file__).resolve().parents[2]  # .../models
MLP_ROOT     = Path(__file__).resolve().parent.parent   # .../models/MLP
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MLP_ROOT))
sys.path.insert(0, str(MODEL_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))


from environment import SmartHomeEnv
from model import load_actor

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = device

        self.obs        = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs   = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.acts       = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.rews       = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones      = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool) -> None:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        act = np.asarray(act, dtype=np.float32).reshape(-1)

        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"obs_dim mismatch: expected {self.obs_dim}, got {obs.shape[0]}")
        if next_obs.shape[0] != self.obs_dim:
            raise ValueError(f"next_obs_dim mismatch: expected {self.obs_dim}, got {next_obs.shape[0]}")
        if act.shape[0] != self.act_dim:
            raise ValueError(f"act_dim mismatch: expected {self.act_dim}, got {act.shape[0]}")

        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.acts[self.ptr] = act
        self.rews[self.ptr, 0] = float(rew)
        self.dones[self.ptr, 0] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")
        batch_size = int(batch_size)
        idx = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "obs": torch.as_tensor(self.obs[idx], device=self.device),
            "acts": torch.as_tensor(self.acts[idx], device=self.device),
            "rews": torch.as_tensor(self.rews[idx], device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=self.device),
            "dones": torch.as_tensor(self.dones[idx], device=self.device),
        }
        return batch
    

class Hyperparameters:
    def __init__(self, config: dict):
        self.seed       = config["seed"]
        self.days       = config["days"]
        self.γ          = config["γ"]
        self.τ          = config["τ"]
        self.batch_size     = config["batch_size"]
        self.update_steps   = config["update_steps"]
        self.actor_lr       = config["actor_lr"]
        self.critic_lr      = config["critic_lr"]
        self.α_lr           = config["α_lr"]
        self.auto_α         = config["auto_entropy"]
        self.policy_delay   = config["policy_delay"]
        self.grad_clip      = config["grad_clip"]
        self.log_std_min    = config["log_std_min"]
        self.log_std_max    = config["log_std_max"]
        self.warmup_episodes = config["warmup_episodes"]
        self.train_episodes  = config["train_episodes"]
        self.eval_every      = config["evaluate_every"]
        self.batch_size      = config["batch_size"]
        self.buffer_size     = config["buffer_size"]
        self.reward_scale    = config["reward_scale"]
        self.target_entropy  = config["target_entropy"]


class Temperature(torch.nn.Module):
    """Learnable entropy temperature (log_alpha)."""

    def __init__(self, init_log_alpha: float, target_entropy: float):
        super().__init__()
        self.log_alpha = torch.nn.Parameter(torch.tensor(float(init_log_alpha), dtype=torch.float32))
        self.target_entropy = float(target_entropy)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def loss(self, log_prob: torch.Tensor) -> torch.Tensor:
        return (self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

class EpisodeGen:
    def __init__(self, config, data_path):
        base = str(data_path).rstrip("/\\")
        self.df_cy = pd.read_csv(f"{base}/Simulation_CY_Cur_HP__PV5000-HB5000.csv", sep=";", parse_dates=["timestamp"], dayfirst=True, index_col="timestamp")
        self.df_wy = pd.read_csv(f"{base}/Simulation_WY_Cur_HP__PV5000-HB5000.csv", sep=";", parse_dates=["timestamp"], dayfirst=True, index_col="timestamp")

        self.days = int(config["train"]["days"])
        self.val_cfg = config["val"]
        self.duration = pd.Timedelta(days=self.days)
        self.eval_windows = self._build_eval_windows()
        self.rng = np.random.default_rng(int(config["train"].get("seed", 0)))

    def _build_eval_windows(self):
        windows = {"cy": [], "wy": []}
        for item in self.val_cfg:
            start = pd.to_datetime(item["date"])
            end = start + pd.Timedelta(days=item["days"])
            key = "cy" if "CY" in item["dataset"] or "cy" in item["dataset"] else "wy"
            windows[key].append((start, end))
        return windows

    def _valid_starts(self, key):
        df = self.df_cy if key == "cy" else self.df_wy
        blocked = self.eval_windows[key]
        latest_start = df.index.max() - self.duration
        candidates = df.index[(df.index >= df.index.min()) & (df.index <= latest_start)]
        valid = []
        for ts in candidates:
            end = ts + self.duration
            if any((ts < b_end) and (end > b_start) for b_start, b_end in blocked):
                continue
            valid.append(ts)
        return pd.DatetimeIndex(valid)

    def _normalize_dataset(self, dataset: str) -> str:
        ds = str(dataset).lower()
        if "cy" in ds:
            return "cy"
        if "wy" in ds:
            return "wy"
        raise ValueError(f"Unknown dataset identifier: {dataset}")

    def sample(self, dataset: str):
        key = self._normalize_dataset(dataset)
        candidates = self._valid_starts(key)
        if len(candidates) == 0:
            raise RuntimeError("No valid start timestamps outside eval windows.")
        start = candidates[self.rng.integers(0, len(candidates))]
        return start


def _eval_worker(run, parameters, tariff, actor_cfg, actor_state_dict, episode_length):
    import torch
    import pandas as pd

    torch.set_num_threads(1)

    df = pd.read_csv(
        run["dataset"],
        sep=";",
        parse_dates=["timestamp"],
        dayfirst=True,
        index_col="timestamp",
    )
    date = pd.to_datetime(run["date"], format="%Y-%m-%d %H:%M:%S")

    env = SmartHomeEnv(
        df,
        parameters,
        start=date,
        days=run["days"],
        BESS_SoC=run["soc"],
        tariff=tariff,
    )

    actor = load_actor(actor_cfg, device=torch.device("cpu"))
    actor.load_state_dict(actor_state_dict, strict=True)
    actor.eval()

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        obs = obs["obs"] if "obs" in obs else (obs["observation"] if "observation" in obs else next(iter(obs.values())))

    done = False
    truncated = False
    episode_reward = 0.0
    steps = 0

    while (not done) and (not truncated) and steps < episode_length:
        obs_t = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action_t = actor(obs_t)
        action_np = action_t.squeeze(0).numpy()

        next_obs, rew, done, truncated, info = env.step(action_np)

        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        if isinstance(next_obs, dict):
            next_obs = next_obs["obs"] if "obs" in next_obs else (next_obs["observation"] if "observation" in next_obs else next(iter(next_obs.values())))

        episode_reward += float(rew)
        obs = next_obs
        steps += 1

    return episode_reward

