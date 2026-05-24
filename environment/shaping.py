"""Potential-based reward shaping (Ng-Harada-Russell) for EV fast-charging cost.

Adds γ·Φ(s') − Φ(s) to the reward; preserves the optimal policy.
"""
from __future__ import annotations

# Obs indices — must match environment.SmartHomeEnv._get_observation().
OBS_BESS_SOC     = 12
OBS_EV_SOC       = 13
OBS_EV_CTRL      = 14
OBS_T_UNTIL_DEP  = 26
OBS_T_UNTIL_ARR  = 27


class AnticipatoryShapingWrapper:
    """Adds γ·Φ(s')−Φ(s) to the reward. Pass-through when omega == 0.

    Delegates unknown attributes to the wrapped env. Eval should use the bare
    SmartHomeEnv so reported metrics stay on raw reward.
    """

    def __init__(self, env, params, gamma: float, omega: float,
                 typical_drop: float = 0.137):
        self.env = env
        self.gamma = float(gamma)
        self.omega = float(omega)

        # Pre-compute constants from params for fast Φ evaluation.
        ev          = params["EV"]
        bess        = params["BESS"]
        general     = params["general"]
        self._target_soc   = float(ev["soc_min"])         # arrival target the env enforces
        self._ecrit        = float(ev["soc_critical"])    # FC kicks in below this
        self._p_fc         = float(ev["fast_tariff"])     # multiplier on tariff for FC
        self._emax_ev      = float(ev["Emax"])
        self._pmax_c_ev    = float(ev["Pmax_c"])
        self._emax_bess    = float(bess["Emax"])
        self._bess_floor   = 1.0 - float(bess["DoD"])     # min usable SoC
        self._typical_drop = float(typical_drop)
        # Conversion: t_until_* are normalized by 12h window; recover hours by *12.
        self._hours_per_unit = 12.0

        self._phi_prev = 0.0

    # ---- gym-like API ----------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi_prev = 0.0 if self.omega == 0.0 else self._phi(obs)
        return obs, info

    def step(self, action):
        obs, raw_r, done, truncated, info = self.env.step(action)
        if self.omega == 0.0:
            return obs, raw_r, done, truncated, info
        terminal = bool(done or truncated)
        phi_new = 0.0 if terminal else self._phi(obs)
        shaped = raw_r + self.gamma * phi_new - self._phi_prev
        self._phi_prev = phi_new
        info["raw_reward"] = float(raw_r)
        info["shaping_delta"] = float(shaped - raw_r)
        return obs, shaped, done, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _phi(self, obs) -> float:
        """Anticipatory potential — more negative means more upcoming FC risk."""
        last_soc      = float(obs[OBS_EV_SOC])
        ev_ctrl       = float(obs[OBS_EV_CTRL]) > 0.5
        bess_soc      = float(obs[OBS_BESS_SOC])
        t_until_dep   = float(obs[OBS_T_UNTIL_DEP])      # normalized
        t_until_arr   = float(obs[OBS_T_UNTIL_ARR])      # normalized

        if ev_ctrl:
            hours_to_dep = t_until_dep * self._hours_per_unit
            max_add_soc = (self._pmax_c_ev * hours_to_dep) / self._emax_ev
            deficit_soc = max(0.0, self._target_soc - last_soc - max_add_soc)
            cost = deficit_soc * self._emax_ev * self._p_fc
        else:
            # Ecrit floor handles mid-trip FC: large trips arrive at >= Ecrit.
            pred_arrival_soc = max(self._ecrit, last_soc - self._typical_drop)
            deficit_soc      = max(0.0, self._target_soc - pred_arrival_soc)
            deficit_kwh      = deficit_soc * self._emax_ev
            bess_help_kwh    = max(0.0, bess_soc - self._bess_floor) * self._emax_bess
            unmet            = max(0.0, deficit_kwh - bess_help_kwh)
            urgency          = 1.0 - t_until_arr
            cost = unmet * self._p_fc * urgency

        return -self.omega * cost
