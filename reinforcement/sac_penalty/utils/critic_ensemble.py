"""REDQ-style critic ensemble (Chen et al., 2021).

Wraps N twin critics. The target Q for the Bellman backup is computed as
min over a random subsample of m of the 2N target Qs, reducing overestimation
under narrow on-policy data.
"""
import torch.nn as nn
import torch


class CriticEnsemble(nn.Module):
    def __init__(self, build_critic, n=2):
        super().__init__()
        self.n = int(n)
        self.critics = nn.ModuleList([build_critic() for _ in range(self.n)])
        self.targets = nn.ModuleList([build_critic() for _ in range(self.n)])
        for c, t in zip(self.critics, self.targets):
            t.load_state_dict(c.state_dict())
            for p in t.parameters():
                p.requires_grad_(False)

    def all_q(self, obs, act):
        """All Q values from online critics. Shape (2N, B, 1)."""
        qs = []
        for c in self.critics:
            q1, q2 = c(obs, act)
            qs.append(q1); qs.append(q2)
        return torch.stack(qs, dim=0)

    @torch.no_grad()
    def target_q(self, obs, act, m=2):
        """min over m of 2N random target Qs. Shape (B, 1)."""
        qs = []
        for t in self.targets:
            q1, q2 = t(obs, act)
            qs.append(q1); qs.append(q2)
        stacked = torch.stack(qs, dim=0)  # (2N, B, 1)
        idx = torch.randperm(stacked.shape[0])[:int(m)]
        return stacked[idx].min(dim=0).values

    def soft_update(self, tau):
        with torch.no_grad():
            for c, t in zip(self.critics, self.targets):
                for p_c, p_t in zip(c.parameters(), t.parameters()):
                    p_t.data.mul_(1 - tau).add_(p_c.data, alpha=tau)
