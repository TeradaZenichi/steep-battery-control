"""N-step replay buffer sharded by stream (CY/WY) so histories never cross datasets."""
from collections import deque
import numpy as np
import torch


class _Shard:
    def __init__(self, capacity, obs_dim, act_dim, device, history_len, n_step, gamma):
        self.capacity, self.obs_dim, self.act_dim = int(capacity), int(obs_dim), int(act_dim)
        self.device = device
        self.history_len = max(1, int(history_len))
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        z = lambda *s: np.zeros(s, dtype=np.float32)
        self.obs = z(self.capacity, self.obs_dim)
        self.next_obs = z(self.capacity, self.obs_dim)
        self.acts = z(self.capacity, self.act_dim)
        self.rews = z(self.capacity, 1)
        self.dones = z(self.capacity, 1)
        self.gamma_pows = z(self.capacity, 1)
        self.episode_ids = np.full(self.capacity, -1, dtype=np.int64)
        self._eid, self.ptr, self.size = 0, 0, 0
        self._q = deque()

    def _store(self, obs, act, rew, next_obs, done, gp, eid):
        if obs.ndim == 2: obs = obs[-1]
        if next_obs.ndim == 2: next_obs = next_obs[-1]
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.acts[i] = act.reshape(-1)
        self.rews[i, 0] = float(rew)
        self.dones[i, 0] = 1.0 if done else 0.0
        self.gamma_pows[i, 0] = float(gp)
        self.episode_ids[i] = eid
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _aggregate(self, q):
        ret, gp, next_obs, done_n = 0.0, 1.0, q[-1][3], False
        for i, (_, _, r, no, d) in enumerate(q):
            ret += (self.gamma ** i) * float(r)
            next_obs, gp = no, self.gamma ** (i + 1)
            if d: done_n = True; break
        o0, a0, _, _, _ = q[0]
        return o0, a0, ret, next_obs, done_n, gp

    def add(self, obs, act, rew, next_obs, done):
        self._q.append((np.asarray(obs, np.float32), np.asarray(act, np.float32),
                        float(rew), np.asarray(next_obs, np.float32), bool(done)))
        while len(self._q) >= self.n_step:
            self._store(*self._aggregate(deque(list(self._q)[:self.n_step])), self._eid)
            self._q.popleft()
        if done:
            while self._q:
                self._store(*self._aggregate(self._q), self._eid)
                self._q.popleft()
            self._eid += 1

    def _seq(self, idx, use_next):
        """Vectorized history reconstruction. For each sampled index, walks
        back history_len-1 storage positions, stopping at episode boundary or
        unwritten positions (when buffer not yet full). Pads the start with
        the oldest available frame. Output layout: [oldest, ..., newest].
        """
        H = self.history_len
        src = self.next_obs if use_next else self.obs
        idx_arr = np.asarray(idx, dtype=np.int64)
        h_arr = np.arange(H, dtype=np.int64)

        # storage position for each (batch, h): (idx - h) mod capacity, h=0 is newest
        indices = (idx_arr[:, None] - h_arr[None, :]) % self.capacity        # (n, H)

        # validity: same episode as the sampled idx; if buffer not full, also written
        target_ep = self.episode_ids[idx_arr][:, None]                       # (n, 1)
        valid = self.episode_ids[indices] == target_ep                       # (n, H)
        if self.size < self.capacity:
            valid &= indices < self.size

        # walk from h=0 outward; stop at first invalid h
        valid_cum = np.cumprod(valid, axis=1).astype(bool)
        last_valid_h = valid_cum.sum(axis=1) - 1                             # (n,) >= 0

        # clamp h, reverse to get [oldest, ..., newest] order, then fancy-index
        h_clamped = np.minimum(h_arr[None, :], last_valid_h[:, None])        # (n, H)
        final_indices = (idx_arr[:, None] - h_clamped[:, ::-1]) % self.capacity
        return src[final_indices]

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=int(batch_size))
        t = lambda a: torch.as_tensor(a, device=self.device)
        return {
            "obs": t(self._seq(idx, False)), "next_obs": t(self._seq(idx, True)),
            "act": t(self.acts[idx]), "rew": t(self.rews[idx]),
            "done": t(self.dones[idx]), "gamma_pow": t(self.gamma_pows[idx]),
        }

    def clone(self):
        other = _Shard(self.capacity, self.obs_dim, self.act_dim, self.device,
                       self.history_len, self.n_step, self.gamma)
        other.obs[:] = self.obs
        other.next_obs[:] = self.next_obs
        other.acts[:] = self.acts
        other.rews[:] = self.rews
        other.dones[:] = self.dones
        other.gamma_pows[:] = self.gamma_pows
        other.episode_ids[:] = self.episode_ids
        other._eid, other.ptr, other.size = self._eid, self.ptr, self.size
        return other


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device, history_len=1, n_step=1, gamma=0.99):
        self.capacity, self.obs_dim, self.act_dim = int(capacity), int(obs_dim), int(act_dim)
        self.device, self.history_len, self.n_step, self.gamma = device, history_len, n_step, gamma
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
                                             self.device, self.history_len, self.n_step, self.gamma)
        return self._shards[stream_id]

    def add(self, obs, act, rew, next_obs, done, stream_id=0):
        self._shard(stream_id).add(obs, act, rew, next_obs, done)

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
                             self.history_len, self.n_step, self.gamma)
        other._shards = {k: s.clone() for k, s in self._shards.items()}
        return other
