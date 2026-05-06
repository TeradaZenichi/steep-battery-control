from pathlib import Path
from collections import deque
import pandas as pd
import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../steep-battery-control
MODEL_ROOT   = Path(__file__).resolve().parents[2]  # .../models
TCN_ROOT     = Path(__file__).resolve().parents[1]  # .../models/TCN
ALGO_ROOT    = Path(__file__).resolve().parent      # .../models/TCN/2-RL
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TCN_ROOT))
sys.path.insert(0, str(MODEL_ROOT))
sys.path.append(str(ALGO_ROOT))


from environment import SmartHomeEnv
from model import load_actor, load_actor_state_dict_compat

_EVAL_DF_CACHE = {}
_EVAL_ACTOR_CACHE = {}

class _ReplayShard:
    """Lazy-frame replay buffer: stores one obs per transition and reconstructs
    sequences of ``history_len`` on the fly during ``sample()``.

    Memory footprint is dominated by *single* obs/next_obs arrays of shape
    ``(capacity, obs_dim)`` instead of ``(capacity, history_len, obs_dim)``,
    giving a ~2Ãƒâ€”history_len reduction (e.g. ~384Ãƒâ€” for history_len=192).
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device,
                 history_len: int = 1, n_step: int = 1, gamma: float = 0.995):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = device
        self.history_len = max(1, int(history_len))
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)

        # Flat storage: one obs vector per transition
        self.obs      = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.acts     = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.rews     = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones    = np.zeros((self.capacity, 1), dtype=np.float32)
        self.gamma_pows = np.zeros((self.capacity, 1), dtype=np.float32)

        # Episode boundary tracking
        self.episode_ids = np.full(self.capacity, -1, dtype=np.int64)
        self._current_episode_id = 0

        self.ptr = 0
        self.size = 0
        self.nstep_queues: dict[object, deque] = {}

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def _store_transition(self, obs: np.ndarray, act: np.ndarray, rew: float,
                          next_obs: np.ndarray, done: bool, gamma_pow: float,
                          episode_id: int) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32).reshape(-1)

        # Accept sequences (history_len, obs_dim) Ã¢â‚¬â€œ take last frame only
        if obs.ndim == 2:
            obs = obs[-1]
        if next_obs.ndim == 2:
            next_obs = next_obs[-1]

        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.acts[self.ptr]     = act
        self.rews[self.ptr, 0]  = float(rew)
        self.dones[self.ptr, 0] = 1.0 if done else 0.0
        self.gamma_pows[self.ptr, 0] = float(gamma_pow)
        self.episode_ids[self.ptr] = episode_id

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    # N-step aggregation
    # ------------------------------------------------------------------
    def _aggregate(self, q: deque):
        ret = 0.0
        gamma_pow = 1.0
        next_obs = q[-1][3]
        done_n = False

        for i, (_, _, rew_i, next_obs_i, done_i) in enumerate(q):
            ret += (self.gamma ** i) * float(rew_i)
            next_obs = next_obs_i
            gamma_pow = self.gamma ** (i + 1)
            if done_i:
                done_n = True
                break

        obs0, act0, _, _, _ = q[0]
        return obs0, act0, ret, next_obs, done_n, gamma_pow

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float,
            next_obs: np.ndarray, done: bool, stream_id=0) -> None:
        key = stream_id
        if key not in self.nstep_queues:
            self.nstep_queues[key] = deque()
        q = self.nstep_queues[key]

        q.append((obs, act, float(rew), next_obs, bool(done)))

        while len(q) >= self.n_step:
            obs0, act0, ret, next_obs_n, done_n, gamma_pow = self._aggregate(
                deque(list(q)[:self.n_step]))
            self._store_transition(obs0, act0, ret, next_obs_n, done_n, gamma_pow,
                                   self._current_episode_id)
            q.popleft()

        if done:
            while len(q) > 0:
                obs0, act0, ret, next_obs_n, done_n, gamma_pow = self._aggregate(q)
                self._store_transition(obs0, act0, ret, next_obs_n, done_n, gamma_pow,
                                       self._current_episode_id)
                q.popleft()
            self._current_episode_id += 1

    # ------------------------------------------------------------------
    # Sequence reconstruction
    # ------------------------------------------------------------------
    def _build_sequences(self, indices: np.ndarray, use_next: bool = False) -> np.ndarray:
        """Reconstruct (batch, history_len, obs_dim) from flat storage."""
        B = len(indices)
        H = self.history_len
        src = self.next_obs if use_next else self.obs

        seqs = np.empty((B, H, self.obs_dim), dtype=np.float32)
        ep_ids = self.episode_ids

        for b in range(B):
            idx = int(indices[b])
            ep = ep_ids[idx]
            frames = [src[idx]]

            cursor = idx
            for _ in range(H - 1):
                prev = (cursor - 1) % self.capacity
                if prev >= self.size and self.size < self.capacity:
                    break
                if ep_ids[prev] != ep:
                    break
                frames.append(src[prev])
                cursor = prev

            frames.reverse()

            while len(frames) < H:
                frames.insert(0, frames[0])

            seqs[b] = np.stack(frames, axis=0)

        return seqs

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        batch_size = int(batch_size)
        idx = np.random.randint(0, self.size, size=batch_size)

        obs_seq = self._build_sequences(idx, use_next=False)
        next_obs_seq = self._build_sequences(idx, use_next=True)

        obs_t      = torch.as_tensor(obs_seq, device=self.device)
        next_obs_t = torch.as_tensor(next_obs_seq, device=self.device)
        acts_t     = torch.as_tensor(self.acts[idx], device=self.device)
        rews_t     = torch.as_tensor(self.rews[idx], device=self.device)
        dones_t    = torch.as_tensor(self.dones[idx], device=self.device)
        gamma_pow_t = torch.as_tensor(self.gamma_pows[idx], device=self.device)

        return {
            "obs": obs_t,
            "act": acts_t,
            "rew": rews_t,
            "next_obs": next_obs_t,
            "done": dones_t,
            "gamma_pow": gamma_pow_t,
            "acts": acts_t,
            "rews": rews_t,
            "dones": dones_t,
        }
    


class ReplayBuffer:
    """Replay buffer split by data base/stream.

    Each stream (currently CY and WY) owns an independent n-step queue,
    episode counter, and circular storage. Sequence models therefore never
    reconstruct histories across bases, even when collection interleaves CY
    and WY steps inside the same training episode.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device,
                 history_len: int = 1, n_step: int = 1, gamma: float = 0.995):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = device
        self.history_len = max(1, int(history_len))
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.shard_capacity = max(1, self.capacity // 2)
        self._shards: dict[object, _ReplayShard] = {}

    @property
    def size(self) -> int:
        return int(sum(shard.size for shard in self._shards.values()))

    def __len__(self) -> int:
        return self.size

    def _new_shard(self) -> _ReplayShard:
        try:
            return _ReplayShard(
                capacity=self.shard_capacity,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                device=self.device,
                history_len=self.history_len,
                n_step=self.n_step,
                gamma=self.gamma,
            )
        except TypeError:
            return _ReplayShard(
                capacity=self.shard_capacity,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                device=self.device,
                n_step=self.n_step,
                gamma=self.gamma,
            )

    def _shard(self, stream_id=0) -> _ReplayShard:
        key = stream_id
        if key not in self._shards:
            self._shards[key] = self._new_shard()
        return self._shards[key]

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float,
            next_obs: np.ndarray, done: bool, stream_id=0) -> None:
        self._shard(stream_id).add(obs, act, rew, next_obs, done, stream_id=0)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        shards = [shard for shard in self._shards.values() if shard.size > 0]
        if not shards:
            raise RuntimeError("Cannot sample from an empty buffer.")

        batch_size = int(batch_size)
        base = batch_size // len(shards)
        rem = batch_size % len(shards)
        counts = [base + (1 if i < rem else 0) for i in range(len(shards))]
        counts = [count for count in counts if count > 0]

        batches = [shard.sample(count) for shard, count in zip(shards, counts)]
        out = {
            key: torch.cat([batch[key] for batch in batches], dim=0)
            for key in batches[0].keys()
        }
        perm = torch.randperm(out["obs"].shape[0], device=self.device)
        return {key: value[perm] for key, value in out.items()}


class Hyperparameters:
    def __init__(self, config: dict):
        self.seed = 42
        self.days = int(config["days"])
        self.gamma = float(config["gamma"])
        self.tau = float(config["tau"])
        self.batch_size = int(config["batch_size"])
        self.update_steps = 1
        self.actor_lr = 1e-4
        self.critic_lr = 1e-4
        self.alpha_lr = float(config["alpha_lr"])
        self.auto_entropy = True
        self.grad_clip = True
        self.log_std_min = -10
        self.log_std_max = float(config["log_std_max"])
        self.warmup_episodes = int(config["warmup_episodes"])
        self.train_episodes = int(config["train_episodes"])
        self.eval_every = int(config["evaluate_every"])
        self.buffer_size = int(config["buffer_size"])
        self.reward_scale = 1.0
        self.target_entropy = float(config["target_entropy"])
        self.n_step = int(config["n_step"])


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
        return -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

class EpisodeGen:
    def __init__(self, config, data_path):
        base = str(data_path).rstrip("/\\")
        self.df_cy = pd.read_csv(f"{base}/Simulation_CY_Cur_HP__PV5000-HB5000.csv", sep=";", parse_dates=["timestamp"], dayfirst=True, index_col="timestamp")
        self.df_wy = pd.read_csv(f"{base}/Simulation_WY_Cur_HP__PV5000-HB5000.csv", sep=";", parse_dates=["timestamp"], dayfirst=True, index_col="timestamp")

        self.days = int(config["train"]["days"])
        self.val_cfg = config["val"]
        self.duration = pd.Timedelta(days=self.days)
        self.eval_windows = self._build_eval_windows()
        self.rng = np.random.default_rng(42)

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

    def load(self, dataset: str):
        key = self._normalize_dataset(dataset)
        return self.df_cy if key == "cy" else self.df_wy


def _eval_worker(run, parameters, tariff, actor_cfg, actor_state_dict, episode_length, history_len=1, deterministic=True, actor_cache_tag=None):
    import torch
    import pandas as pd

    torch.set_num_threads(1)

    dataset_path = Path(run["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    cache_key = str(dataset_path.resolve())

    df = _EVAL_DF_CACHE.get(cache_key)
    if df is None:
        df = pd.read_csv(
            cache_key,
            sep=";",
            parse_dates=["timestamp"],
            dayfirst=True,
            index_col="timestamp",
        )
        _EVAL_DF_CACHE[cache_key] = df
    date = pd.to_datetime(run["date"], format="%Y-%m-%d %H:%M:%S")

    env = SmartHomeEnv(
        df,
        parameters,
        start=date,
        days=run["days"],
        BESS_SoC=run["soc"],
        tariff=tariff,
    )

    cache_tag = int(actor_cache_tag) if actor_cache_tag is not None else -1
    actor = _EVAL_ACTOR_CACHE.get(cache_tag)
    if actor is None:
        actor = load_actor(actor_cfg, device=torch.device("cpu"))
        load_actor_state_dict_compat(actor, actor_state_dict, strict=True)
        actor.eval()
        _EVAL_ACTOR_CACHE.clear()
        _EVAL_ACTOR_CACHE[cache_tag] = actor

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        obs = obs["obs"] if "obs" in obs else (obs["observation"] if "observation" in obs else next(iter(obs.values())))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)

    history_len = max(1, int(history_len))
    history = deque([obs.copy() for _ in range(history_len)], maxlen=history_len)

    done = False
    truncated = False
    episode_reward = 0.0
    steps = 0

    while (not done) and (not truncated) and steps < episode_length:
        obs_seq = np.stack(history, axis=0)
        obs_t = torch.as_tensor(obs_seq, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            if deterministic:
                _, _, action_t, _ = actor.sample(obs_t)
            else:
                action_t, _, _, _ = actor.sample(obs_t)
        action_np = action_t.squeeze(0).numpy()

        next_obs, rew, done, truncated, info = env.step(action_np)

        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        if isinstance(next_obs, dict):
            next_obs = next_obs["obs"] if "obs" in next_obs else (next_obs["observation"] if "observation" in next_obs else next(iter(next_obs.values())))
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        episode_reward += float(rew)
        history.append(next_obs.copy())
        steps += 1

    result = {"reward": float(episode_reward), "steps": int(steps)}
    op = getattr(env, "operation", None)
    if op is not None and len(op) > 0:
        for col in ["energy_cost", "bess_cost", "ev_cost", "pv_cost", "grid_penalty"]:
            if col in op:
                result[col] = float(pd.to_numeric(op[col], errors="coerce").fillna(0.0).sum())
        if "pv_cmd" in op:
            result["pv_cmd_mean"] = float(pd.to_numeric(op["pv_cmd"], errors="coerce").fillna(0.0).mean())
        if "PBESS" in op:
            result["bess_abs_power_mean"] = float(np.abs(pd.to_numeric(op["PBESS"], errors="coerce").fillna(0.0).to_numpy()).mean())
        if "PEV" in op:
            result["ev_abs_power_mean"] = float(np.abs(pd.to_numeric(op["PEV"], errors="coerce").fillna(0.0).to_numpy()).mean())
    return result
