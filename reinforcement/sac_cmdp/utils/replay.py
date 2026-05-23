"""N-step replay buffer sharded by stream, with optional n-step cost returns."""
from collections import deque
import numpy as np
import torch


class _Shard:
    def __init__(self, capacity, obs_dim, act_dim, device, history_len, n_step, gamma, cost_names=()):
        self.capacity, self.obs_dim, self.act_dim = int(capacity), int(obs_dim), int(act_dim)
        self.device = device
        self.history_len = max(1, int(history_len))
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.cost_names = tuple(cost_names or ())
        z = lambda *s: np.zeros(s, dtype=np.float32)
        self.obs = z(self.capacity, self.obs_dim)
        self.next_obs = z(self.capacity, self.obs_dim)
        self.acts = z(self.capacity, self.act_dim)
        self.rews = z(self.capacity, 1)
        self.costs = {name: z(self.capacity, 1) for name in self.cost_names}
        self.dones = z(self.capacity, 1)
        self.gamma_pows = z(self.capacity, 1)
        self.episode_ids = np.full(self.capacity, -1, dtype=np.int64)
        self._eid, self.ptr, self.size = 0, 0, 0
        self._q = deque()

    def _store(self, obs, act, rew, costs, next_obs, done, gp, eid):
        if obs.ndim == 2:
            obs = obs[-1]
        if next_obs.ndim == 2:
            next_obs = next_obs[-1]
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.acts[i] = act.reshape(-1)
        self.rews[i, 0] = float(rew)
        for name in self.cost_names:
            self.costs[name][i, 0] = float((costs or {}).get(name, 0.0))
        self.dones[i, 0] = 1.0 if done else 0.0
        self.gamma_pows[i, 0] = float(gp)
        self.episode_ids[i] = eid
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _aggregate(self, q):
        ret, gp, next_obs, done_n = 0.0, 1.0, q[-1][4], False
        cost_ret = {name: 0.0 for name in self.cost_names}
        for i, (_, _, r, costs, no, d) in enumerate(q):
            disc = self.gamma ** i
            ret += disc * float(r)
            for name in self.cost_names:
                cost_ret[name] += disc * float((costs or {}).get(name, 0.0))
            next_obs, gp = no, self.gamma ** (i + 1)
            if d:
                done_n = True
                break
        o0, a0, _, _, _, _ = q[0]
        return o0, a0, ret, cost_ret, next_obs, done_n, gp

    def add(self, obs, act, rew, next_obs, done, costs=None):
        self._q.append((np.asarray(obs, np.float32), np.asarray(act, np.float32),
                        float(rew), dict(costs or {}), np.asarray(next_obs, np.float32), bool(done)))
        while len(self._q) >= self.n_step:
            self._store(*self._aggregate(deque(list(self._q)[:self.n_step])), self._eid)
            self._q.popleft()
        if done:
            while self._q:
                self._store(*self._aggregate(self._q), self._eid)
                self._q.popleft()
            self._eid += 1

    def _seq(self, idx, use_next):
        H = self.history_len
        src = self.next_obs if use_next else self.obs
        idx_arr = np.asarray(idx, dtype=np.int64)
        h_arr = np.arange(H, dtype=np.int64)
        indices = (idx_arr[:, None] - h_arr[None, :]) % self.capacity
        target_ep = self.episode_ids[idx_arr][:, None]
        valid = self.episode_ids[indices] == target_ep
        if self.size < self.capacity:
            valid &= indices < self.size
        valid_cum = np.cumprod(valid, axis=1).astype(bool)
        last_valid_h = valid_cum.sum(axis=1) - 1
        h_clamped = np.minimum(h_arr[None, :], last_valid_h[:, None])
        final_indices = (idx_arr[:, None] - h_clamped[:, ::-1]) % self.capacity
        return src[final_indices]

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=int(batch_size))
        t = lambda a: torch.as_tensor(a, device=self.device)
        out = {
            "obs": t(self._seq(idx, False)),
            "next_obs": t(self._seq(idx, True)),
            "act": t(self.acts[idx]),
            "rew": t(self.rews[idx]),
            "done": t(self.dones[idx]),
            "gamma_pow": t(self.gamma_pows[idx]),
        }
        for name in self.cost_names:
            out[f"{name}_cost"] = t(self.costs[name][idx])
        return out

    def clone(self):
        other = _Shard(self.capacity, self.obs_dim, self.act_dim, self.device,
                       self.history_len, self.n_step, self.gamma, self.cost_names)
        other.obs[:] = self.obs
        other.next_obs[:] = self.next_obs
        other.acts[:] = self.acts
        other.rews[:] = self.rews
        for name in self.cost_names:
            other.costs[name][:] = self.costs[name]
        other.dones[:] = self.dones
        other.gamma_pows[:] = self.gamma_pows
        other.episode_ids[:] = self.episode_ids
        other._eid, other.ptr, other.size = self._eid, self.ptr, self.size
        return other


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device, history_len=1, n_step=1, gamma=0.99, cost_names=()):
        self.capacity, self.obs_dim, self.act_dim = int(capacity), int(obs_dim), int(act_dim)
        self.device, self.history_len, self.n_step, self.gamma = device, history_len, n_step, gamma
        self.cost_names = tuple(cost_names or ())
        self._cap = max(1, self.capacity // 2)
        self._shards: dict[object, _Shard] = {}

    @property
    def size(self):
        return int(sum(s.size for s in self._shards.values()))

    def __len__(self):
        return self.size

    def _shard(self, stream_id):
        if stream_id not in self._shards:
            self._shards[stream_id] = _Shard(self._cap, self.obs_dim, self.act_dim,
                                             self.device, self.history_len, self.n_step,
                                             self.gamma, self.cost_names)
        return self._shards[stream_id]

    def add(self, obs, act, rew, next_obs, done, stream_id=0, costs=None):
        self._shard(stream_id).add(obs, act, rew, next_obs, done, costs=costs)

    def sample(self, batch_size):
        shards = [s for s in self._shards.values() if s.size > 0]
        if not shards:
            raise RuntimeError("Empty buffer.")
        bs = int(batch_size)
        base, rem = bs // len(shards), bs % len(shards)
        counts = [c for c in [base + (1 if i < rem else 0) for i in range(len(shards))] if c > 0]
        batches = [s.sample(c) for s, c in zip(shards, counts)]
        out = {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}
        perm = torch.randperm(out["obs"].shape[0], device=self.device)
        return {k: v[perm] for k, v in out.items()}

    def clone(self):
        other = ReplayBuffer(self.capacity, self.obs_dim, self.act_dim, self.device,
                             self.history_len, self.n_step, self.gamma, self.cost_names)
        other._shards = {k: s.clone() for k, s in self._shards.items()}
        return other
