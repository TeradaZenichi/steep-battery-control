from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


def _ensure_seq_array(arr: np.ndarray, seq_len: int, state_dim: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != (seq_len, state_dim):
        raise ValueError(f"Expected sequence shape {(seq_len, state_dim)}, got {arr.shape}")
    return arr


@dataclass
class BufferSample:
    states: torch.Tensor  # (B, seq_len, state_dim)
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, seq_len: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.seq_len = int(seq_len)
        self.state_dim = int(state_dim)
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros((capacity, seq_len, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, seq_len, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def add(self, state_seq: np.ndarray, action: np.ndarray, reward: float, next_state_seq: np.ndarray, done: bool) -> None:
        state_seq = _ensure_seq_array(state_seq, self.seq_len, self.state_dim)
        next_state_seq = _ensure_seq_array(next_state_seq, self.seq_len, self.state_dim)
        idx = self.ptr % self.capacity
        self.states[idx] = torch.as_tensor(state_seq, dtype=torch.float32, device=self.device)
        self.actions[idx] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[idx] = torch.as_tensor([reward], dtype=torch.float32, device=self.device)
        self.next_states[idx] = torch.as_tensor(next_state_seq, dtype=torch.float32, device=self.device)
        self.dones[idx] = torch.as_tensor([done], dtype=torch.float32, device=self.device)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> BufferSample:
        idx = np.random.randint(0, self.size, size=batch_size)
        return BufferSample(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            dones=self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size
