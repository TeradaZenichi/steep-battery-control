"""KL anchoring to a snapshot of the actor at peak. When activated, adds
β · KL(π_current ‖ π_anchor) to the actor loss to limit post-peak drift.
"""
import copy
import torch


class KLAnchor:
    def __init__(self, beta=0.1, decay=1.0):
        self.beta = float(beta)
        self.decay = float(decay)
        self.anchor = None

    def ready(self):
        return self.anchor is not None and self.beta > 0.0

    def snapshot(self, actor):
        self.anchor = copy.deepcopy(actor)
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.anchor.eval()

    def kl(self, obs, actor):
        """Closed-form KL(N(μ_c, σ_c) ‖ N(μ_a, σ_a)) for diagonal Gaussians.
        Operates on the (mu, log_std) heads — pre-tanh, pre-scale — which is
        where the SAC policy is parametrized as a Normal.
        """
        mu_c, ls_c = actor.policy(obs)
        with torch.no_grad():
            mu_a, ls_a = self.anchor.policy(obs)
        var_c, var_a = (2 * ls_c).exp(), (2 * ls_a).exp()
        kl_dim = ls_a - ls_c + (var_c + (mu_c - mu_a).pow(2)) / (2 * var_a) - 0.5
        return kl_dim.sum(dim=-1).mean()

    def decay_beta(self):
        self.beta *= self.decay
