"""Polyak-averaged copy of a model — useful for smoothed eval."""
import copy
import torch


class EMA:
    def __init__(self, model, tau=0.99):
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tau = float(tau)

    @torch.no_grad()
    def update(self, src):
        for p_src, p_ema in zip(src.parameters(), self.model.parameters()):
            p_ema.data.mul_(self.tau).add_(p_src.data, alpha=1 - self.tau)

    def state_dict(self):
        return self.model.state_dict()
