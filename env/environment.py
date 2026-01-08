import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

CLIMATE_COLUMNS = [
    "drybulb_C", "relhum_percent", "Global Horizontal Radiation",
    # "dni_Wm2", "dhi_Wm2", "wdir_deg",  # NOTE: in provided CSVs, wdir_deg behaves as wind speed (m/s)
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
        target = np.datetime64(pd.to_datetime(start_date, dayfirst=True))
        self.set_start_step(
            int(
                np.searchsorted(
                    self.timestamps.to_numpy(dtype="datetime64[ns]"),
                    target,
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
        # Leakage retention factor is interpreted as an HOURLY retention ratio and rescaled to the current timestep.
        self.beta_leak_hour = float(config.get("β_leak", config.get("beta_leak", beta_raw)))
        # Degradation factor remains throughput-based (no timestep exponent).
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
        self._penalty = self.ramp_penalty * ramp * self.sim.dt
        self._pbess_prev = self._pbess
        return self._pbess, self._costdeg, self._penalty, {
            "p_cmd": p_cmd,
            "p_used": self._pbess,
            "pbess_charge_kw": self._pbess_c,
            "pbess_discharge_kw": self._pbess_d,
            "soc": self.soc,
            "soh": self.soh,
            "throughput_kWh": thr,
            "leakage_kWh": max(0.0, e_prev - e_leak),
        }

    def get_obs(self):
        return np.array([self.soc, self.soh], np.float32)


class EV:
    def __init__(self, config=None, sim=None):
        self.sim = sim
        self.Pmax_c = config.get("Pmax_c", 7.0)
        self.Pmax_d = config.get("Pmax_d", 7.0)
        self.Emax = config.get("Emax", 50.0)
        self.η = config.get("η", 0.9)
        self.β = config.get("β", 0.999)
        self.ramp_penalty = float(config.get("ramp_penalty", 0.0) or 0.0)
        self.penalty_coef = float(config.get("penalty_coef", 1.0) or 1.0)
        self.DoD = config.get("DoD", 0.8)
        self.soc_min = config.get("soc_min", 0.0)
        self.soc_max = config.get("soc_max", 1.0)
        self.soc0 = config.get("soc_init", 0.5)
        self.soh0 = config.get("soh_init", 1.0)
        self.capex = config.get("capex", 0.0)
        self.ncycles = float(config.get("ncycles", 1.0)) or 1.0
        self.penalty_departure = config.get("penalty_departure", 0.0) or 0.0
        self.status_column = config.get("status_column", config.get("column", "ev_status"))
        usable = (self.Emax * self.DoD) or 1.0
        self._deg_cost_per_kwh = (self.capex / (self.ncycles * usable)) if self.capex else 0.0

        self.soc, self.soh = self.soc0, self.soh0
        self.connected = False
        self._pev = self._pev_c = self._pev_d = 0.0
        self._costdeg = self._penalty = 0.0

    def reset(self, soc_init=None):
        self.soc = self.soc0 if soc_init is None else soc_init
        self.soh = self.soh0
        self.connected = False
        self._pev = self._pev_c = self._pev_d = self._costdeg = self._penalty = 0.0
        self._pev_prev = 0.0
        self._was_connected = False
        return self.get_obs()

    def _get_status_signal(self):
        r = self.sim.dataframe.iloc[self.sim.current_step]
        if self.status_column in r.index:
            return float(r[self.status_column])
        return 0.0

    def step(self, action):
        status = self._get_status_signal()
        prev = getattr(self, "_was_connected", False)
        self.connected = bool(status > 0.01)

        # If the vehicle disconnects at this timestep, enforce departure penalty proportional to SOC shortfall.
        if (not self.connected) and prev:
            shortfall = max(0.0, 1.0 - self.soc)
            self._penalty = self.penalty_coef * self.penalty_departure * shortfall
            self._was_connected = self.connected
            self._pev = self._pev_c = self._pev_d = 0.0
            self._costdeg = 0.0
            self._pev_prev = 0.0
            return self._pev, self._costdeg, self._penalty, {
                "event": "disconnected",
                "status_signal": status,
                "connected": False,
                "soc": self.soc,
                "soh": self.soh,
                "departure_shortfall": shortfall,
            }

        # If not connected, EV cannot exchange power.
        if not self.connected:
            self._pev = self._pev_c = self._pev_d = 0.0
            self._costdeg = 0.0
            self._penalty = 0.0
            self._pev_prev = 0.0
            self._was_connected = self.connected
            return self._pev, self._costdeg, self._penalty, {
                "event": "idle",
                "status_signal": status,
                "connected": False,
                "p_cmd": 0.0,
                "p_used": 0.0,
                "soc": self.soc,
                "soh": self.soh,
            }

        # Connected: apply action within bounds and SOC constraints.
        p_cmd = float(np.clip(action, -self.Pmax_d, self.Pmax_c))
        e_prev = self.soc * self.Emax
        e_leak = float(np.clip(self.β, 0.0, 1.0)) * e_prev  # kept as original behavior for EV
        e_next = float(
            np.clip(
                e_leak
                + self.sim.dt
                * (max(p_cmd, 0.0) * self.η - max(-p_cmd, 0.0) / self.η),
                self.soc_min * self.Emax,
                self.soc_max * self.Emax,
            )
        )
        d = e_next - e_leak
        self.soc = e_next / self.Emax
        self._pev = d / self.sim.dt
        self._pev_c, self._pev_d = max(self._pev, 0.0), max(-self._pev, 0.0)
        thr = abs(d)
        self.soh = max(0.0, self.soh - (1.0 - float(self.β)) * thr / (self.Emax * self.DoD))
        self._costdeg = thr * self._deg_cost_per_kwh
        ramp = abs(self._pev - getattr(self, "_pev_prev", 0.0))
        self._penalty = self.ramp_penalty * ramp * self.sim.dt
        self._pev_prev = self._pev
        self._was_connected = self.connected

        return self._pev, self._costdeg, self._penalty, {
            "event": "connected" if (self.connected and not prev) else "connected",
            "status_signal": status,
            "connected": True,
            "p_cmd": p_cmd,
            "p_used": self._pev,
            "pev_charge_kw": self._pev_c,
            "pev_discharge_kw": self._pev_d,
            "soc": self.soc,
            "soh": self.soh,
        }

    def get_obs(self):
        return np.array([self.soc, self.soh, 1.0 if self.connected else 0.0], np.float32)


class Grid:
    def __init__(self, config=None, sim=None):
        self.sim = sim
        self.Pmax_import = config.get("Pmax_import", 10.0)
        self.Pmax_export = config.get("Pmax_export", 10.0)
        self.net_penalty = config.get("net_penalty", 10.0)
        self.net_penalty_quadratic = config.get("net_penalty_quadratic", 0.0)
        self.tariff_column = config.get("tariff_column")
        self._pgrid = self._cost = self._revenue = self._penalty = 0.0
        self.tariff = 0.0

    def reset(self):
        self._pgrid = self._cost = self._revenue = self._penalty = self.tariff = 0.0

    def _get_tariff_rate(self):
        r = self.sim.dataframe.iloc[self.sim.current_step]
        return float(r[self.tariff_column]) if self.tariff_column in r.index else 0.0

    def step(self, action):
        p = float(action)
        ex = max(0.0, p - self.Pmax_import) + max(0.0, -p - self.Pmax_export)
        self._penalty = self.net_penalty * ex * self.sim.dt + (ex * ex) * self.sim.dt * self.net_penalty_quadratic
        self._pgrid = p
        e = p * self.sim.dt
        self.tariff = self._get_tariff_rate()
        self._cost, self._revenue = (self.tariff * e, 0.0) if e >= 0 else (0.0, -self.tariff * e)
        return self._cost, self._revenue, self._penalty, {
            "p_cmd": p,
            "p_used": p,
            "tariff": self.tariff,
            "energy_kWh": e,
            "import_cost": self._cost,
            "export_revenue": self._revenue,
            "penalty_power": ex,
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
        self._full_obs_len = c + t + 2 + 3 + 1 + 1 + 3  # climate + time + bess + ev + shortfall + pv + (load,tariff,net_load)
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

        # Synchronize exogenous signals (PV, load, tariff, EV status) with the initial timestamp.
        self.load.step()
        self.pv.step(0.0)  # default: no curtailment at episode start
        self.ev.step(0.0)  # sync connected flag and initial SoC from ev_status when available
        self.grid.tariff = self.grid._get_tariff_rate()

        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        pb, pe, x = action
        _, _, _, bi = self.bess.step(pb)
        _, _, _, ei = self.ev.step(pe)
        self.pv.step(x)
        self.load.step()

        pb_bus = float(bi.get("p_cmd", 0.0))
        pe_bus = float(ei.get("p_cmd", 0.0))

        pgrid = pb_bus + pe_bus + self.load._pload - self.pv._ppv
        ppv_curtailed = max(self.pv._available_kw - self.pv._ppv, 0.0)

        self.sim.record_action(
            pb_bus,
            pe_bus,
            self.pv._curt,
            pgrid,
            self.pv._ppv,
            ppv_curtailed,
            pbess_cmd=pb_bus,
            pev_cmd=pe_bus,
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
        pnorm = self.sim.pnorm or 1.0
        ts = pd.to_datetime(row.get("timestamp"))
        climate = self.sim.get_climate_features(row)
        climate_raw = [float(row.get(c, 0.0)) for c in self.sim.climate_columns]
        time = self._build_time_features(ts)

        b = self.bess.get_obs().tolist()
        e = self.ev.get_obs().tolist()
        shortfall = max(0.0, 1.0 - e[0])

        pv_av = self.pv._get_available_kw()
        load_kw = self.load._get_demand_kw()
        tariff_now = self.grid._get_tariff_rate()
        net_kw = load_kw - pv_av

        feats = climate + time + b + e + [shortfall] + [
            pv_av / pnorm,
            load_kw / pnorm,
            tariff_now / (self.sim.tariff_base or 1.0),
            net_kw / pnorm,
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
