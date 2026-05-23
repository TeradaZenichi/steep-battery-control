"""PID Lagrangian (Stooke, Achiam & Abbeel, 2020): replaces the standard
integral-only dual update with a PID controller. Reduces the oscillation and
unbounded growth seen with vanilla Lagrangian when the policy can't perfectly
satisfy the constraint.

lambda_t = max(0, K_P · V_t + integral + K_D · (V_t - V_{t-1}))
where `integral` accumulates K_I · V over time.
"""
import torch.nn as nn
import torch


class PIDLambda(nn.Module):
    def __init__(self, K_P=1.0, K_I=0.01, K_D=0.0, budget=0.0,
                 init_integral=0.1, max_lambda=None):
        super().__init__()
        self.K_P = float(K_P)
        self.K_I = float(K_I)
        self.K_D = float(K_D)
        self.budget = float(budget)
        self.max_lambda = None if max_lambda is None else float(max_lambda)
        self.register_buffer("integral", torch.tensor(float(init_integral)))
        self.register_buffer("prev_v", torch.tensor(0.0))
        self.register_buffer("_lam", torch.tensor(float(init_integral)))

    @property
    def value(self):
        return self._lam

    @torch.no_grad()
    def step(self, violation, opt=None):
        v = violation.detach().mean() - self.budget
        self.integral.add_(self.K_I * v).clamp_(min=0.0)
        d = v - self.prev_v
        self.prev_v.copy_(v)
        lam = (self.K_P * v + self.integral + self.K_D * d).clamp_(min=0.0)
        if self.max_lambda is not None:
            lam = lam.clamp(max=self.max_lambda)
        self._lam.copy_(lam)
        return {"lambda_value": float(self._lam),
                "lambda_p": float(self.K_P * v),
                "lambda_i": float(self.integral),
                "lambda_d": float(self.K_D * d)}
