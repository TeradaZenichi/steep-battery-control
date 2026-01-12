import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

CLIMATE_COLUMNS = [
    "drybulb_C", "relhum_percent", "Global Horizontal Radiation",
    "dni_Wm2", "dhi_Wm2", "Wind Speed (m/s)"
]


class Simulation:
    def __init__(self, config=None, dataframe=None):
        self.config = config or {}
        self.dataframe = dataframe.copy()
        if "timestamp" in self.dataframe:
            self.dataframe["timestamp"] = pd.to_datetime(
                self.dataframe["timestamp"],
                format="%d/%m/%Y %H:%M",
                dayfirst=True,
                errors="coerce",
            )
            self.timestamps = pd.DatetimeIndex(self.dataframe["timestamp"])
        else:
            self.timestamps = pd.DatetimeIndex([])
        self.duration = self.config.get("simulation_duration", 24)
        self.simulation_days = self.config.get("simulation_days", 1)
        self.timestep = self.config.get("timestep", 5)
        self.norm_cfg = self.config.get("state_normalization", {})
        self.pnorm = self.config.get("Pnorm", self.norm_cfg.get("Pmax", 1.0))
        self.tariff_base = self.norm_cfg.get("tariff_base", 1.0) or 1.0
        self.climate_columns = self.norm_cfg.get("climate_columns", CLIMATE_COLUMNS)
        self.state_history, self.action_history = [], []
        self.start_step = 0
        self.current_step = 0
        self.dt = self.timestep / 60.0
        self.num_steps = self._compute_num_steps()

    def _compute_num_steps(self):
        planned = int((self.duration * self.simulation_days * 60) / (self.timestep or 1)) or 1
        remaining = len(self.dataframe) - int(self.start_step)
        return min(planned, max(0, remaining))

    def reset(self):
        self.num_steps = self._compute_num_steps()
        self.current_step = int(self.start_step)
        self.final_step = min(self.current_step + self.num_steps - 1, len(self.dataframe) - 1)
        self.current_datetime = pd.to_datetime(self.dataframe.timestamp[self.current_step])
        self.state_history, self.action_history = [], []

    def step(self):
        if self.current_step >= self.final_step:
            return True
        self.current_step += 1
        self.current_datetime = pd.to_datetime(self.dataframe.timestamp[self.current_step])
        return False

    def get_value(self, column, default=0.0):
        return float(self.dataframe.iloc[self.current_step].get(column, default))

    def normalize_value(self, key, value):
        cfg = self.norm_cfg.get(key)
        if isinstance(cfg, dict):
            vmin, vmax = cfg.get("min"), cfg.get("max")
            if vmin is not None and vmax is not None and vmax != vmin:
                return float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
        return float(value)

    def get_climate_features(self, row):
        return [self.normalize_value(c, float(row.get(c, 0.0))) for c in self.climate_columns]

    def record_state(self, raw_state, masked_state=None, unnormalized_state=None):
        snap = {
            "step": self.current_step,
            "timestamp": self.current_datetime,
            "raw_state": np.array(raw_state, copy=True),
        }
        if masked_state is not None:
            snap["masked_state"] = np.array(masked_state, copy=True)
        if unnormalized_state is not None:
            snap["raw_state_unscaled"] = np.array(unnormalized_state, copy=True)
        self.state_history.append(snap)

    def record_action(
        self,
        pbess,
        pev,
        xpvc,
        pgrid,
        ppv_used,
        ppv_curtailed,
        pbess_cmd=None,
        pev_cmd=None,
        load_kw=None,
        soc_bess=None,
        soc_ev=None,
    ):
        d = {
            "step": self.current_step,
            "timestamp": self.current_datetime,
            "timestep_minutes": self.timestep,
            "PBESS": float(pbess),
            "PEV": float(pev),
            "XPV": float(xpvc),
            "pgrid": float(pgrid),
            "ppv_used": float(ppv_used),
            "ppv_curtailed": float(ppv_curtailed),
        }
        if pbess_cmd is not None:
            d["PBESS_cmd"] = float(pbess_cmd)
        if pev_cmd is not None:
            d["Pev_cmd"] = float(pev_cmd)
        if load_kw is not None:
            d["Pload"] = float(load_kw)
        if soc_bess is not None:
            d["soc_bess"] = float(soc_bess)
        if soc_ev is not None:
            d["soc_ev"] = float(soc_ev)
        self.action_history.append(d)

    def _pad(self, seq, n):
        if n is None:
            return list(seq)
        if not seq:
            return []
        if n <= len(seq):
            return list(seq[-n:])
        return [seq[0]] * (n - len(seq)) + list(seq)

    def get_action_history(self, last_n=None):
        return self._pad(self.action_history, last_n)

    def get_state_history(self, last_n=None):
        return self._pad(self.state_history, last_n)

    def get_unnormalized_state_history(self, last_n=None):
        unscaled = [s for s in self.state_history if "raw_state_unscaled" in s]
        unscaled = self._pad(unscaled, last_n)
        return [
            {
                "step": s["step"],
                "timestamp": s["timestamp"],
                "raw_state_unscaled": np.array(s["raw_state_unscaled"], copy=True),
            }
            for s in unscaled
        ]

    def set_start_step(self, start_step):
        self.start_step = int(start_step)
        self.current_step = int(start_step)

    def set_start_date(self, start_date):
        self.set_start_step(
            int(
                np.searchsorted(
                    self.timestamps.to_numpy(dtype="datetime64[ns]"),
                    np.datetime64(pd.to_datetime(start_date, dayfirst=True)),
                    side="left",
                )
            )
        )


class BESS:
    def __init__(self, config=None, sim=None):
        self.sim = sim
        self.Pmax = config.get("Pmax", 5.0)
        self.Emax = config.get("Emax", 10.0)
        self.η = config.get("η", 0.95)
        beta_raw = float(config.get("β", config.get("beta", 0.999)))
        self.beta_leak_hour = float(config.get("β_leak", config.get("beta_leak", beta_raw)))
        self.beta_deg = float(config.get("β_deg", config.get("beta_deg", beta_raw)))
        self.ramp_penalty = float(config.get("ramp_penalty", 0.0) or 0.0)
        self.DoD = config.get("DoD", 0.8)
        self.soc_min = config.get("soc_min", 0.1)
        self.soc_max = config.get("soc_max", 0.9)
        self.capex = config.get("capex", 100.0)
        self.ncycles = float(config.get("ncycles", 1.0)) or 1.0
        usable = (self.Emax * self.DoD) or 1.0
        self._deg_cost_per_kwh = self.capex / (self.ncycles * usable)

        self.soc0 = config.get("soc_init", 0.5)
        self.soh0 = config.get("soh_init", 1.0)

        sk = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90], np.float32)
        cc = np.array([3.1, 4.7, 5.3, 5.3, 5.3, 5.3, 5.3, 4.5, 3.2], np.float32)
        dc = np.array([4.1, 5.2, 5.3, 5.3, 5.3, 5.3, 5.1, 4.5, 3.8], np.float32)
        curve = (config.get("soc_power_curve") or {})
        self._soc_knots = np.array(curve.get("soc", sk), np.float32)
        self._pmax_charge_curve = np.array(curve.get("charge_kw", cc), np.float32)
        self._pmax_discharge_curve = np.array(curve.get("discharge_kw", dc), np.float32)

        self.soc, self.soh = self.soc0, self.soh0
        self._pbess = self._pbess_c = self._pbess_d = 0.0
        self._costdeg = self._penalty = 0.0

    def _pmax_charge(self, soc):
        return min(
            self.Pmax,
            max(
                0.0,
                float(
                    np.interp(
                        float(np.clip(soc, self.soc_min, self.soc_max)),
                        self._soc_knots,
                        self._pmax_charge_curve,
                    )
                ),
            ),
        )

    def _pmax_discharge(self, soc):
        return min(
            self.Pmax,
            max(
                0.0,
                float(
                    np.interp(
                        float(np.clip(soc, self.soc_min, self.soc_max)),
                        self._soc_knots,
                        self._pmax_discharge_curve,
                    )
                ),
            ),
        )

    def reset(self, soc_init=None):
        self.soc = self.soc0 if soc_init is None else soc_init
        self.soh = self.soh0
        self._pbess = self._pbess_c = self._pbess_d = self._costdeg = self._penalty = 0.0
        self._pbess_prev = 0.0
        return self.get_obs()

    def step(self, action):
        p_cmd = float(np.clip(action, -self._pmax_discharge(self.soc), self._pmax_charge(self.soc)))
        e_prev = self.soc * self.Emax
        dt_h = float(self.sim.dt)
        beta_step = float(np.clip(self.beta_leak_hour, 0.0, 1.0)) ** dt_h
        e_leak = beta_step * e_prev
        e_next = float(
            np.clip(
                e_leak
                + self.sim.dt
                * (max(p_cmd, 0.0) * self.η - max(-p_cmd, 0.0) / self.η),
                self.soc_min * self.Emax,
                self.soc_max * self.Emax,
            )
        )
        d_ctrl = e_next - e_leak  # only controlled energy affects power balance
        self.soc = e_next / self.Emax
        self._pbess = d_ctrl / self.sim.dt
        self._pbess_c, self._pbess_d = max(self._pbess, 0.0), max(-self._pbess, 0.0)
        thr = abs(d_ctrl)
        self.soh = max(0.0, self.soh - (1.0 - self.beta_deg) * thr / (self.Emax * self.DoD))
        self._costdeg = thr * self._deg_cost_per_kwh
        ramp = abs(self._pbess - getattr(self, "_pbess_prev", 0.0))
        self._penalty = self.ramp_penalty * ramp
        self._pbess_prev = self._pbess
        return self._pbess, self._pbess_c, self._pbess_d, {
            "soc": self.soc,
            "soh": self.soh,
            "p_cmd": p_cmd,
            "p_used": self._pbess,
            "energy_prev": e_prev,
            "energy_leak": e_leak,
            "energy_next": e_next,
            "deg_cost": self._costdeg,
            "ramp_penalty": self._penalty,
        }

    def get_obs(self):
        return np.array([self.soc, self.soh], np.float32)


class EV:
    """
    EV model aligned with Teacher logic:
    - Arrival: status[t] > 0.01 and status[t-1] <= 0.01 (t0 counts as arrival if status[t0] > 0.01)
    - Departure: status[t] <= 0.01 and status[t-1] > 0.01
    - Power is forced to 0 when disconnected and at arrival step
    - Energy: 0 when disconnected; status[t]*Emax at arrival; otherwise retention + delta(action)
    - SOH: initialized at soh_init; held constant while disconnected; degrades only when connected
    - Departure penalty uses thresholds/weights (Teacher-style)
    """

    def __init__(self, config=None, sim=None):
        self.sim = sim
        cfg = config or {}

        # Power and energy limits
        self.Pmax_c = float(cfg.get("Pmax_c", 7.0))
        self.Pmax_d = float(cfg.get("Pmax_d", 0.0))
        self.Emax = float(cfg.get("Emax", 50.0))

        # Efficiencies and beta
        self.η = float(cfg.get("η", cfg.get("eta", 0.95)))

        beta_raw = float(cfg.get("β", cfg.get("beta", 0.999)))
        self.β = beta_raw

        # Teacher uses beta for retention and degradation (unless you explicitly split them)
        self.beta_leak_hour = float(cfg.get("β_leak", cfg.get("beta_leak", self.β)))
        self.beta_deg = float(cfg.get("β_deg", cfg.get("beta_deg", self.β)))

        # SOC/SOH bounds and initial
        self.DoD = float(cfg.get("DoD", 0.8))
        self.soc_min = float(cfg.get("soc_min", 0.0))
        self.soc_max = float(cfg.get("soc_max", 1.0))
        self.soc0 = float(cfg.get("soc_init", 0.0))
        self.soh0 = float(cfg.get("soh_init", 1.0))

        # Ramp penalty (env-only, but keep consistent behavior)
        self.ramp_penalty = float(cfg.get("ramp_penalty", 0.0) or 0.0)

        # Use same key as Teacher: "column" (fallback to legacy env key "status_column")
        self.column_status = cfg.get("column", cfg.get("status_column", "ev_status"))

        # Degradation cost (kept as in env; teacher uses cost_per_kwh)
        self.capex = float(cfg.get("capex", 0.0))
        self.ncycles = float(cfg.get("ncycles", 1.0)) or 1.0
        usable_energy = max(self.Emax * self.DoD, 1e-6)
        self._deg_cost_per_kwh = (self.capex / (self.ncycles * usable_energy)) if self.capex else 0.0

        # Departure penalty (Teacher-style)
        # Your parameters.json uses:
        # "departure_penalty": {"thresholds":[...], "weights":[...]} and "penalty_coef": 1.0
        dep_cfg = cfg.get("departure_penalty", None)
        if isinstance(dep_cfg, dict):
            self.dep_thresholds = list(dep_cfg.get("thresholds", []))
            self.dep_weights = list(dep_cfg.get("weights", []))
        else:
            # Teacher legacy naming fallback
            self.dep_thresholds = list(cfg.get("dep_thresholds", cfg.get("departure_thresholds", [])))
            self.dep_weights = list(cfg.get("dep_weights", cfg.get("departure_weights", [])))

        # If nothing provided, default to "no penalty"
        if not self.dep_thresholds or not self.dep_weights:
            self.dep_thresholds = []
            self.dep_weights = []

        # Multipliers (Teacher has penalty_coef and penalty_departure)
        self.penalty_coef = float(cfg.get("penalty_coef", 1.0))
        # You cannot change parameters.json; teacher may encode the global multiplier elsewhere.
        # Here we accept either "penalty_departure" or legacy scalar "departure_penalty" if given as float.
        if isinstance(dep_cfg, (int, float)):
            self.penalty_departure = float(dep_cfg)
        else:
            self.penalty_departure = float(cfg.get("penalty_departure", 1.0))

        # Internal state
        self.connected = False
        self.soc = 0.0
        self.soh = self.soh0

        self._pev = self._pev_c = self._pev_d = 0.0
        self._costdeg = 0.0
        self._penalty = 0.0
        self._pev_prev_used = 0.0  # previous "used" power (Teacher compares against prev_val)
        self._was_connected = False  # kept for compatibility with existing env logic

    def _status_at_step(self, step: int) -> float:
        if self.sim is None or self.sim.dataframe is None or len(self.sim.dataframe) == 0:
            return 0.0
        step = int(np.clip(step, 0, len(self.sim.dataframe) - 1))
        return float(self.sim.dataframe.iloc[step].get(self.column_status, 0.0))

    def _get_status_signal(self) -> float:
        return self._status_at_step(int(self.sim.current_step))

    def _is_arrival(self, status_now: float, status_prev: float, is_first_step: bool) -> bool:
        if is_first_step:
            return status_now > 0.01
        return (status_now > 0.01) and (status_prev <= 0.01)

    def _is_departure(self, status_now: float, status_prev: float, is_first_step: bool) -> bool:
        if is_first_step:
            return False
        return (status_now <= 0.01) and (status_prev > 0.01)

    def reset(self):
        self.connected = False
        # Teacher forces Eev=0 when disconnected; keep soc at 0 baseline
        self.soc = 0.0
        # Teacher sets SOH_EV[t0]=soh_init and then holds it even while disconnected
        self.soh = self.soh0

        self._pev = self._pev_c = self._pev_d = 0.0
        self._costdeg = 0.0
        self._penalty = 0.0
        self._pev_prev_used = 0.0
        self._was_connected = False
        return self.get_obs()

    def _departure_penalty_from_soc(self, soc_dep: float) -> float:
        if not self.dep_thresholds or not self.dep_weights:
            return 0.0
        n = min(len(self.dep_thresholds), len(self.dep_weights))
        pen = 0.0
        for k in range(n):
            thr = float(self.dep_thresholds[k])
            w = float(self.dep_weights[k])
            pen += w * max(0.0, thr - float(soc_dep))
        return float(self.penalty_coef) * float(self.penalty_departure) * pen

    def step(self, action):
        dt_h = float(self.sim.dt) if self.sim is not None else 1.0
        cur_step = int(self.sim.current_step) if self.sim is not None else 0
        first_step = (cur_step <= int(getattr(self.sim, "start_step", 0)))

        status_now = self._status_at_step(cur_step)
        status_prev = self._status_at_step(cur_step - 1) if not first_step else 0.0

        arrival = self._is_arrival(status_now, status_prev, first_step)
        departure = self._is_departure(status_now, status_prev, first_step)

        self.connected = bool(status_now > 0.01)

        # Departure event (Teacher applies penalty on departure step using SOC from previous connected step)
        if departure:
            soc_dep = float(self.soc)  # SOC right before forcing E=0 (aligned with Teacher using t_prev)
            self._penalty = self._departure_penalty_from_soc(soc_dep)

            # Teacher: Eev[t]=0 when disconnected; SOH holds constant
            self.soc = 0.0
            # keep self.soh unchanged

            self._pev = self._pev_c = self._pev_d = 0.0
            self._costdeg = 0.0
            self._pev_prev_used = 0.0
            self._was_connected = self.connected
            return self._pev, self._pev_c, self._pev_d, {
                "connected": False,
                "event": "departed",
                "status": status_now,
                "soc_dep": soc_dep,
                "departure_penalty": self._penalty,
                "p_cmd": 0.0,
                "p_used": 0.0,
                "soc": self.soc,
                "soh": self.soh,
            }

        # Disconnected (absent): Teacher forces power=0 and E=0; SOH holds
        if not self.connected:
            self.soc = 0.0
            # keep self.soh unchanged
            self._pev = self._pev_c = self._pev_d = 0.0
            self._costdeg = 0.0
            self._penalty = 0.0
            self._pev_prev_used = 0.0
            self._was_connected = self.connected
            return self._pev, self._pev_c, self._pev_d, {
                "connected": False,
                "event": "absent",
                "status": status_now,
                "p_cmd": 0.0,
                "p_used": 0.0,
                "soc": self.soc,
                "soh": self.soh,
            }

        # Connected + arrival: Teacher sets Eev=status*Emax and enforces Pev=0 at arrival step
        if arrival:
            soc_arrival = float(np.clip(status_now, self.soc_min, self.soc_max))
            self.soc = soc_arrival
            # Teacher does NOT reset SOH at arrival
            # self.soh unchanged

            self._pev = self._pev_c = self._pev_d = 0.0
            self._costdeg = 0.0
            self._penalty = 0.0
            self._pev_prev_used = 0.0  # prev_val=0 when previously disconnected
            self._was_connected = self.connected
            return self._pev, self._pev_c, self._pev_d, {
                "connected": True,
                "event": "arrival",
                "status": status_now,
                "p_cmd": 0.0,
                "p_used": 0.0,
                "soc": self.soc,
                "soh": self.soh,
            }

        # Connected + not arrival: normal dynamics
        p_cmd = float(np.clip(action, -self.Pmax_d, self.Pmax_c))

        e_prev = float(self.soc) * float(self.Emax)
        beta_step = float(np.clip(self.beta_leak_hour, 0.0, 1.0)) ** float(dt_h)
        e_leak = beta_step * e_prev

        e_next = float(
            np.clip(
                e_leak + dt_h * (max(p_cmd, 0.0) * self.η - max(-p_cmd, 0.0) / self.η),
                self.soc_min * self.Emax,
                self.soc_max * self.Emax,
            )
        )

        # Teacher: only controlled energy affects power balance
        d_ctrl = e_next - e_leak
        p_used = d_ctrl / max(dt_h, 1e-9)

        self.soc = e_next / max(self.Emax, 1e-9)
        self._pev = float(p_used)
        self._pev_c, self._pev_d = max(self._pev, 0.0), max(-self._pev, 0.0)

        # SOH degradation aligned with Teacher: deg_factor=(1-beta)/(Emax*DoD) and thr=abs(d_ctrl)
        thr = abs(d_ctrl)
        deg_factor = (1.0 - float(self.beta_deg)) / max(self.Emax * self.DoD, 1e-9)
        self.soh = max(0.0, float(self.soh) - deg_factor * thr)

        self._costdeg = thr * float(self._deg_cost_per_kwh)

        # Ramp penalty aligned with Teacher: previous val is 0 if previous status was disconnected
        prev_val = 0.0 if status_prev <= 0.01 else float(self._pev_prev_used)
        ramp = abs(float(self._pev) - prev_val)
        self._penalty = float(self.ramp_penalty) * ramp

        self._pev_prev_used = float(self._pev)
        self._was_connected = self.connected

        return self._pev, self._pev_c, self._pev_d, {
            "connected": True,
            "event": "connected",
            "status": status_now,
            "p_cmd": p_cmd,
            "p_used": float(self._pev),
            "energy_prev": e_prev,
            "energy_leak": e_leak,
            "energy_next": e_next,
            "soc": float(self.soc),
            "soh": float(self.soh),
            "deg_cost": float(self._costdeg),
            "ramp_penalty": float(self._penalty),
        }

    def get_obs(self):
        return np.array([float(self.soc), float(self.soh), 1.0 if self.connected else 0.0], np.float32)


class Grid:
    def __init__(self, config=None, sim=None):
        self.sim = sim
        self.Pmax_import = config.get("Pmax_import", 15.0)
        self.Pmax_export = config.get("Pmax_export", 5.0)
        self.tariff_column = config.get("tariff_column", "tariff")
        self.penalty_power = float(config.get("penalty_power", 0.0) or 0.0)
        self._pgrid = 0.0
        self.tariff = 0.0
        self._cost = 0.0
        self._revenue = 0.0
        self._penalty = 0.0

    def _get_tariff_rate(self):
        return float(self.sim.get_value(self.tariff_column, 0.0))

    def reset(self):
        self._pgrid = 0.0
        self.tariff = self._get_tariff_rate()
        self._cost = 0.0
        self._revenue = 0.0
        self._penalty = 0.0

    def step(self, pgrid):
        p = float(pgrid)
        self.tariff = self._get_tariff_rate()
        p_clip = float(np.clip(p, -self.Pmax_export, self.Pmax_import))
        self._penalty = self.penalty_power * abs(p - p_clip)
        self._pgrid = p_clip
        e = self._pgrid * self.sim.dt
        if e >= 0:
            self._cost = e * self.tariff
            self._revenue = 0.0
        else:
            self._cost = 0.0
            self._revenue = (-e) * self.tariff
        return self._pgrid, {
            "pgrid": p,
            "pgrid_used": p_clip,
            "tariff": self.tariff,
            "energy_kWh": e,
            "import_cost": self._cost,
            "export_revenue": self._revenue,
            "penalty_power": self._penalty,
        }

    def get_obs(self):
        return np.array([self._pgrid, self.tariff], np.float32)


class PV:
    def __init__(self, config=None, sim=None):
        self.sim = sim
        self.Pmax = config.get("Pmax", 5.0)
        self.column = config.get("column", "produced_electricity_rate_W")
        self._ppv = self._curt = self._available_kw = 0.0

    def reset(self):
        self._ppv = self._curt = self._available_kw = 0.0

    def _get_available_kw(self):
        return self.sim.get_value(self.column, 0.0) / 1000.0

    def step(self, action):
        self._curt = float(np.clip(action, 0.0, 1.0))
        self._available_kw = self._get_available_kw()
        self._ppv = self._available_kw * (1.0 - self._curt)
        return self._ppv, {
            "pv_available_kw": self._available_kw,
            "pv_used_kw": self._ppv,
            "pv_curtailed_kw": self._available_kw - self._ppv,
            "curtailment": self._curt,
        }

    def get_obs(self):
        return np.array([self._get_available_kw() / (self.Pmax or 1.0)], np.float32)


class Load:
    def __init__(self, config=None, sim=None):
        self.sim = sim
        self.Pmax = config.get("Pmax", 10.0)
        self.column = config.get("column", "electricity_demand_rate_W")
        self._pload = 0.0

    def reset(self):
        self._pload = 0.0

    def _get_demand_kw(self):
        return self.sim.get_value(self.column, 0.0) / 1000.0

    def step(self):
        self._pload = self._get_demand_kw()
        return self._pload

    def get_obs(self):
        return np.array([self._pload], np.float32)


class SmartHomeEnv(gym.Env):
    def __init__(self, config=None, dataframe=None, render_mode=None, days=None, state_mask=None, start_date=None):
        super().__init__()
        self.config = config or {}
        g = dict(self.config.get("general", {}))
        if days is not None:
            g["simulation_days"] = days
        self.sim = Simulation(g, dataframe=dataframe)
        if start_date is not None:
            self.sim.set_start_date(pd.to_datetime(start_date, dayfirst=True))

        self.bess = BESS(self.config.get("BESS", {}), sim=self.sim)
        self.grid = Grid(self.config.get("Grid", {}), sim=self.sim)
        self.load = Load(self.config.get("Load", {}), sim=self.sim)
        self.pv = PV(self.config.get("PV", {}), sim=self.sim)
        self.ev = EV(self.config.get("EV", {}), sim=self.sim)

        self.pv_curtailment_penalty = float(g.get("pv_curtailment_penalty", 0.01))

        self.action_space = spaces.Box(
            low=np.array([-self.bess.Pmax, -self.ev.Pmax_d, 0.0], np.float32),
            high=np.array([self.bess.Pmax, self.ev.Pmax_c, 1.0], np.float32),
            dtype=np.float32,
        )

        c, t = len(self.sim.climate_columns), 10
        self._full_obs_len = c + t + 2 + 3 + 1 + 1 + 3
        self._feature_names = (
            [f"climate_{x}" for x in self.sim.climate_columns]
            + [
                "month_sin",
                "month_cos",
                "day_sin",
                "day_cos",
                "hour_sin",
                "hour_cos",
                "minute_sin",
                "minute_cos",
                "weekday_sin",
                "weekday_cos",
            ]
            + ["bess_soc", "bess_soh"]
            + ["ev_soc", "ev_soh", "ev_connected"]
            + ["ev_shortfall"]
            + ["pv_available"]
            + ["load", "tariff", "net_load"]
        )
        self.state_mask = np.array(state_mask, bool) if state_mask is not None else None
        self.state_feature_labels = (
            [n for n, k in zip(self._feature_names, self.state_mask)] if self.state_mask is not None else list(self._feature_names)
        )
        obs_len = int(self.state_mask.sum()) if self.state_mask is not None else self._full_obs_len
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        self.state = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.bess.reset()
        self.grid.reset()
        self.load.reset()
        self.pv.reset()
        self.ev.reset()

        self.load.step()
        self.pv.step(0.0)


        status0 = float(self.ev._get_status_signal())
        self.ev.connected = bool(status0 > 0.01)

        if self.ev.connected:
            self.ev.soc = float(np.clip(status0, self.ev.soc_min, self.ev.soc_max))
        else:
            self.ev.soc = 0.0

        self.ev.soh = float(self.ev.soh0)
        self.ev._was_connected = self.ev.connected

        self.ev._pev = self.ev._pev_c = self.ev._pev_d = 0.0
        self.ev._pev_prev = 0.0

        self.grid.tariff = self.grid._get_tariff_rate()

        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        pb, pe, x = action
        _, _, _, bi = self.bess.step(pb)
        _, _, _, ei = self.ev.step(pe)
        self.pv.step(x)
        self.load.step()

        pb_used = float(bi.get("p_used", bi.get("p_cmd", 0.0)))
        pe_used = float(ei.get("p_used", ei.get("p_cmd", 0.0)))
        pgrid = pb_used + pe_used + self.load._pload - self.pv._ppv
        ppv_curtailed = max(self.pv._available_kw - self.pv._ppv, 0.0)

        self.sim.record_action(
            pb_used,
            pe_used,
            self.pv._curt,
            pgrid,
            self.pv._ppv,
            ppv_curtailed,
            pbess_cmd=float(bi.get("p_cmd", 0.0)),
            pev_cmd=float(ei.get("p_cmd", 0.0)),
            load_kw=self.load._pload,
            soc_bess=self.bess.soc,
            soc_ev=self.ev.soc,
        )
        self.grid.step(pgrid)

        pv_curt_cost = ppv_curtailed * self.sim.dt * self.pv_curtailment_penalty
        r = -(
            self.bess._costdeg
            + self.bess._penalty
            + self.ev._costdeg
            + self.ev._penalty
            + self.grid._cost
            - self.grid._revenue
            + self.grid._penalty
            + pv_curt_cost
        )

        self.done = self.sim.step()
        obs = self._get_obs()
        return obs, r, self.done, {"terminated": False, "truncated": self.done, "pv_curt_cost": pv_curt_cost}

    def _build_time_features(self, ts):
        if pd.isna(ts):
            return [0.0] * 10

        def cyc(v, p):
            a = 2 * np.pi * (v / p)
            return [float(np.sin(a)), float(np.cos(a))]

        return cyc(ts.month - 1, 12) + cyc(ts.day - 1, ts.days_in_month) + cyc(ts.hour, 24) + cyc(ts.minute, 60) + cyc(ts.weekday(), 7)

    def _get_obs(self):
        row = self.sim.dataframe.iloc[self.sim.current_step]
        ts = pd.to_datetime(row.get("timestamp"))
        climate = self.sim.get_climate_features(row)
        climate_raw = [float(row.get(c, 0.0)) for c in self.sim.climate_columns]
        time = self._build_time_features(ts)

        b = self.bess.get_obs().tolist()
        e = self.ev.get_obs().tolist()
        shortfall = max(0.0, 1.0 - e[0]) if e[2] > 0.5 else 0.0

        pv_av = self.pv._get_available_kw()
        load_kw = self.load._get_demand_kw()
        tariff_now = self.grid._get_tariff_rate()
        net_kw = load_kw - pv_av

        feats = climate + time + b + e + [shortfall] + [
            pv_av / (self.sim.pnorm or 1.0),
            load_kw / (self.sim.pnorm or 1.0),
            tariff_now / (self.sim.tariff_base or 1.0),
            net_kw / (self.sim.pnorm or 1.0),
        ]
        unscaled = climate_raw + time + b + e + [shortfall] + [pv_av, load_kw, tariff_now, net_kw]

        raw = np.array(feats, np.float32)
        masked = raw[self.state_mask] if self.state_mask is not None else raw
        self.sim.record_state(raw, masked, np.array(unscaled, np.float32))
        self.state = masked
        return masked

    def build_operation_dataframe(self):
        seg = self.sim.dataframe.iloc[self.sim.start_step : self.sim.start_step + self.sim.num_steps].copy()
        seg["timestamp"] = pd.to_datetime(seg["timestamp"])
        base = seg.set_index("timestamp")

        act = pd.DataFrame(self.sim.get_action_history())
        act["timestamp"] = pd.to_datetime(act["timestamp"])
        act = act.set_index("timestamp")

        unscaled = pd.DataFrame(self.sim.get_unnormalized_state_history())
        unscaled["timestamp"] = pd.to_datetime(unscaled["timestamp"])
        unscaled = unscaled.set_index("timestamp").groupby(level=0).last()

        c, t, b, e = len(self.sim.climate_columns), 10, 2, 3
        idx_bess = c + t
        idx_ev = idx_bess + b
        idx_ev_status = idx_ev + 2
        idx_load = idx_ev + e + 2

        comp = unscaled["raw_state_unscaled"].apply(
            lambda a: (
                v := np.asarray(a, float),
                pd.Series({"soc_bess": v[idx_bess], "soc_ev": v[idx_ev], "ev_status": v[idx_ev_status], "load_kw": v[idx_load]}),
            )[1]
        )
        comb = act.join(comp, how="left")

        out = pd.DataFrame(
            {
                "Pgrid": comb["pgrid"],
                "Ppv": comb["ppv_used"],
                "Load": comb["load_kw"],
                "PBESS": comb["PBESS"],
                "EBESS": comb["soc_bess"] * self.bess.Emax,
                "Pev": comb["PEV"],
                "Eev": comb["soc_ev"] * self.ev.Emax,
                "ev_status": comb["ev_status"],
            }
        )
        out["PBESS_cmd"] = comb["PBESS_cmd"] if "PBESS_cmd" in comb else comb["PBESS"]
        out["Pev_cmd"] = comb["Pev_cmd"] if "Pev_cmd" in comb else comb["PEV"]
        return base.join(out, how="left", rsuffix="_env")
