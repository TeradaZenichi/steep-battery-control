from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo


CLIMATE_COLUMNS = [
    "drybulb_C",
    "relhum_percent",
    "Global Horizontal Radiation",
    "dni_Wm2",
    "dhi_Wm2",
    "Wind Speed (m/s)",
]


def _encode_cycle(value: float, period: float) -> list[float]:
    if period <= 0 or pd.isna(value):
        return [0.0, 0.0]
    angle = 2.0 * math.pi * (float(value) / float(period))
    return [float(math.sin(angle)), float(math.cos(angle))]


def _build_time_features(ts: pd.Timestamp) -> list[float]:
    if pd.isna(ts):
        return [0.0] * 10
    month_val = int(ts.month) - 1
    day_val = int(ts.day) - 1
    hour_val = int(ts.hour)
    minute_val = int(ts.minute)
    weekday_val = int(ts.weekday())

    days_in_month = int(pd.Period(ts, freq="M").days_in_month)

    features: list[float] = []
    features += _encode_cycle(month_val, 12)
    features += _encode_cycle(day_val, days_in_month)
    features += _encode_cycle(hour_val, 24)
    features += _encode_cycle(minute_val, 60)
    features += _encode_cycle(weekday_val, 7)
    return features


class BESS:
    def __init__(self, config: dict):
        self.Pmax = float(config.get("Pmax", 5.0))
        self.Emax = float(config.get("Emax", 10.0))
        self.eta = float(config.get("eta", config.get("η", 0.95)))

        beta_raw = float(config.get("beta", config.get("β", 0.999)))
        self.beta_leak_hour = float(config.get("beta_leak", config.get("β_leak", beta_raw)))
        self.beta_deg = float(config.get("beta_deg", config.get("β_deg", beta_raw)))

        self.DoD = float(config.get("DoD", 0.8))
        self.soc_min = float(config.get("soc_min", 0.1))
        self.soc_max = float(config.get("soc_max", 0.9))
        self.soc_init = float(config.get("soc_init", 0.5))
        self.soh_init = float(config.get("soh_init", 1.0))

        self.capex = float(config.get("capex", 100.0))
        self.ncycles = max(1.0, float(config.get("ncycles", 1.0)) or 1.0)
        self.ramp_penalty = float(config.get("ramp_penalty", 0.0) or 0.0)

        usable_energy = max(self.Emax * self.DoD, 1e-6)
        self.cost_per_kwh = (self.capex / (self.ncycles * usable_energy)) if self.capex else 0.0


class EV:
    def __init__(self, config: dict, dataframe: pd.DataFrame):
        self.Pmax_c = float(config.get("Pmax_c", 7.0))
        self.Pmax_d = float(config.get("Pmax_d", 7.0))
        self.Emax = float(config.get("Emax", 50.0))
        self.eta = float(config.get("eta", config.get("η", 0.9)))

        beta_raw = float(config.get("beta", config.get("β", 0.999)))
        self.beta = beta_raw
        self.beta_leak_hour = float(config.get("beta_leak", config.get("β_leak", beta_raw)))

        self.DoD = float(config.get("DoD", 0.8))
        self.soc_min = float(config.get("soc_min", 0.0))
        self.soc_max = float(config.get("soc_max", 1.0))
        self.soc_init = float(config.get("soc_init", 0.0))
        self.soh_init = float(config.get("soh_init", 1.0))

        self.capex = float(config.get("capex", 0.0))
        self.ncycles = max(1.0, float(config.get("ncycles", 1.0)) or 1.0)
        self.ramp_penalty = float(config.get("ramp_penalty", 0.0) or 0.0)

        # Departure penalty configuration
        self.dep_thresholds = list(config.get("dep_thresholds", [0.2, 0.5, 0.8]))
        self.dep_weights = list(config.get("dep_weights", [1.0] * len(self.dep_thresholds)))
        self.penalty_coef = float(config.get("penalty_coef", 1.0))
        self.penalty_departure = float(config.get("penalty_departure", 0.0))

        usable_energy = max(self.Emax * self.DoD, 1e-6)
        self.cost_per_kwh = (self.capex / (self.ncycles * usable_energy)) if self.capex else 0.0

        profile = dataframe.get("ev_status", pd.Series(dtype=float))
        index = pd.to_datetime(dataframe["timestamp"], format="%d/%m/%Y %H:%M", dayfirst=True)
        self.profile = profile.set_axis(index, copy=False)


class Load:
    def __init__(self, config: dict, dataframe: pd.DataFrame):
        self.Pmax = float(config.get("Pmax", 10.0))
        profile = dataframe.get("electricity_demand_rate_W", pd.Series(dtype=float)) / 1000.0
        index = pd.to_datetime(dataframe["timestamp"], format="%d/%m/%Y %H:%M", dayfirst=True)
        self.profile = profile.set_axis(index, copy=False)


class PV:
    def __init__(self, config: dict, dataframe: pd.DataFrame):
        self.Pmax = float(config.get("Pmax", 5.0))
        profile = dataframe.get("produced_electricity_rate_W", pd.Series(dtype=float)) / 1000.0
        index = pd.to_datetime(dataframe["timestamp"], format="%d/%m/%Y %H:%M", dayfirst=True)
        self.profile = profile.set_axis(index, copy=False)


class Grid:
    def __init__(self, config: dict, dataframe: pd.DataFrame):
        self.Pmax_import = float(config.get("Pmax_import", 10.0))
        self.Pmax_export = float(config.get("Pmax_export", 10.0))
        self.net_penalty = float(config.get("net_penalty", 10.0))
        self.net_penalty_quadratic = float(config.get("net_penalty_quadratic", 0.0))

        self.tariff_column = config.get("tariff_column", config.get("tariff", "tar_tou"))
        profile = dataframe.get(self.tariff_column, pd.Series(dtype=float))
        index = pd.to_datetime(dataframe["timestamp"], format="%d/%m/%Y %H:%M", dayfirst=True)
        self.tariff = profile.set_axis(index, copy=False)


class Teacher:
    def __init__(self, config, dataframe, start_date=None, days=5, state_mask=None):
        self.config = config or {}

        timestamps = pd.to_datetime(
            dataframe["timestamp"],
            format="%d/%m/%Y %H:%M",
            dayfirst=True,
            errors="coerce",
        )
        dataframe = dataframe.copy()
        dataframe["timestamp"] = timestamps
        self._timeline = pd.DatetimeIndex(timestamps)

        self.load = Load(self.config.get("Load", {}), dataframe)
        self.grid = Grid(self.config.get("Grid", {}), dataframe)
        self.pv = PV(self.config.get("PV", {}), dataframe)
        self.ev = EV(self.config.get("EV", {}), dataframe)
        self.bess = BESS(self.config.get("BESS", {}))

        general_cfg = self.config.get("general", {})
        self.norm_cfg = general_cfg.get("state_normalization", {})
        self.pnorm = float(general_cfg.get("Pnorm", self.norm_cfg.get("Pmax", 1.0)))
        self.tariff_base = max(float(self.norm_cfg.get("tariff_base", 1.0)), 1e-6)
        self.climate_columns = list(self.norm_cfg.get("climate_columns", CLIMATE_COLUMNS))
        self.curtailment_penalty = float(general_cfg.get("pv_curtailment_penalty", 0.01))

        self.dt_hours = float(general_cfg.get("timestep", 5.0)) / 60.0
        steps_needed = max(1, int(days * 24.0 / self.dt_hours))

        if start_date is None:
            start_idx = 0
        else:
            start_date = pd.to_datetime(start_date, dayfirst=True)
            diffs = (self._timeline - start_date).to_series().abs()
            start_idx = int(diffs.values.argmin()) if len(diffs) else 0

        end_idx = min(start_idx + steps_needed, len(dataframe))
        segment = dataframe.iloc[start_idx:end_idx].copy()
        segment = segment.dropna(subset=["timestamp"])
        self._segment = segment.set_index("timestamp")

        self.Ωt = list(pd.DatetimeIndex(self._segment.index))

        # Cache some exogenous lookups
        self._pv_available_lookup = {t: float(self.pv.profile.get(t, 0.0)) for t in self.Ωt}
        self._load_lookup = {t: float(self.load.profile.get(t, 0.0)) for t in self.Ωt}
        self._tariff_lookup = {t: float(self.grid.tariff.get(t, 0.0)) for t in self.Ωt}

        # Observation layout mirrors environment
        climate_len = len(self.climate_columns)
        time_len = 10
        bess_len = 2
        ev_len = 3
        pv_len = 1
        extra_terms = 3  # load, tariff, net_load (pv_available is pv_len)
        shortfall_terms = 1
        self._full_obs_len = climate_len + time_len + bess_len + ev_len + shortfall_terms + pv_len + extra_terms
        self._feature_names = (
            [f"climate_{col}" for col in self.climate_columns]
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

        self.state_mask = None
        self.state_feature_labels = list(self._feature_names)
        if state_mask is not None:
            resolved_mask = self._resolve_mask(state_mask)
            self.state_mask = resolved_mask
            self.state_feature_labels = [name for name, keep in zip(self._feature_names, resolved_mask) if keep]

        self._state_history: list[dict] = []
        self._results_cache: pd.DataFrame | None = None

    def build(self, start_soc: float):
        self._results_cache = None
        self._state_history = []

        self._bess_retention = float(np.clip(self.bess.beta_leak_hour, 0.0, 1.0)) ** float(self.dt_hours)
        self._ev_retention = float(np.clip(self.ev.beta, 0.0, 1.0)) ** float(self.dt_hours)

        m = pyo.ConcreteModel()

        m.T = pyo.Set(initialize=self.Ωt, ordered=True)

        m.Pgrid = pyo.Var(m.T, domain=pyo.Reals)
        m.chi_pv = pyo.Var(m.T, bounds=(0.0, 1.0))
        for t in self.Ωt:
            if self._pv_available_lookup.get(t, 0.0) <= 1e-9:
                m.chi_pv[t].fix(0.0)

        m.PBESS_c = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.PBESS_d = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.PBESS = pyo.Var(m.T, domain=pyo.Reals)
        m.gamma_BESS_c = pyo.Var(m.T, domain=pyo.Binary)
        m.gamma_BESS_d = pyo.Var(m.T, domain=pyo.Binary)
        m.EBESS = pyo.Var(
            m.T,
            bounds=(self.bess.soc_min * self.bess.Emax, self.bess.soc_max * self.bess.Emax),
        )
        m.SOH_BESS = pyo.Var(m.T, bounds=(0.0, 1.0))
        m.RampBESS = pyo.Var(m.T, domain=pyo.NonNegativeReals)

        m.Pev_c = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.Pev_d = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.Pev = pyo.Var(m.T, domain=pyo.Reals)
        m.gamma_EV_c = pyo.Var(m.T, domain=pyo.Binary)
        m.gamma_EV_d = pyo.Var(m.T, domain=pyo.Binary)
        m.Eev = pyo.Var(m.T, bounds=(0.0, self.ev.Emax))
        m.SOH_EV = pyo.Var(m.T, bounds=(0.0, 1.0))
        m.RampEV = pyo.Var(m.T, domain=pyo.NonNegativeReals)

        m.GridExImport = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.GridExExport = pyo.Var(m.T, domain=pyo.NonNegativeReals)

        K = list(range(len(self.ev.dep_thresholds)))
        m.K = pyo.Set(initialize=K, ordered=True)
        m.EV_DepShort = pyo.Var(m.T, m.K, domain=pyo.NonNegativeReals)
        m.EV_DepPenalty = pyo.Var(m.T, domain=pyo.NonNegativeReals)

        def power_balance_rule(mm, t):
            load_kw = self._load_lookup.get(t, 0.0)
            pv_av = self._pv_available_lookup.get(t, 0.0)
            return load_kw + mm.PBESS[t] + mm.Pev[t] == mm.Pgrid[t] + pv_av * (1.0 - mm.chi_pv[t])

        m.power_balance_constr = pyo.Constraint(m.T, rule=power_balance_rule)

        def bess_charge_discharge_rule(mm, t):
            return mm.PBESS[t] == mm.PBESS_c[t] - mm.PBESS_d[t]

        m.bess_charge_discharge_constr = pyo.Constraint(m.T, rule=bess_charge_discharge_rule)

        def bess_charge_rule(mm, t):
            return mm.PBESS_c[t] <= self.bess.Pmax * mm.gamma_BESS_c[t]

        m.bess_charge_constr = pyo.Constraint(m.T, rule=bess_charge_rule)

        def bess_discharge_rule(mm, t):
            return mm.PBESS_d[t] <= self.bess.Pmax * mm.gamma_BESS_d[t]

        m.bess_discharge_constr = pyo.Constraint(m.T, rule=bess_discharge_rule)

        def bess_binary_rule(mm, t):
            return mm.gamma_BESS_c[t] + mm.gamma_BESS_d[t] <= 1

        m.bess_binary_constr = pyo.Constraint(m.T, rule=bess_binary_rule)

        start_energy = float(start_soc) * float(self.bess.Emax)
        start_soh = float(self.bess.soh_init)

        def bess_energy_rule(mm, t):
            delta = (mm.PBESS_c[t] * self.bess.eta - mm.PBESS_d[t] / self.bess.eta) * self.dt_hours
            if t == self.Ωt[0]:
                return mm.EBESS[t] == self._bess_retention * start_energy + delta
            t_prev = self.Ωt[self.Ωt.index(t) - 1]
            return mm.EBESS[t] == self._bess_retention * mm.EBESS[t_prev] + delta

        m.bess_energy_constr = pyo.Constraint(m.T, rule=bess_energy_rule)

        def bess_soh_rule(mm, t):
            thr = (mm.PBESS_c[t] * self.bess.eta + mm.PBESS_d[t] / self.bess.eta) * self.dt_hours
            deg_factor = (1.0 - float(self.bess.beta_deg)) / max(self.bess.Emax * self.bess.DoD, 1e-6)
            if t == self.Ωt[0]:
                return mm.SOH_BESS[t] == start_soh - deg_factor * thr
            t_prev = self.Ωt[self.Ωt.index(t) - 1]
            return mm.SOH_BESS[t] == mm.SOH_BESS[t_prev] - deg_factor * thr

        m.bess_soh_constr = pyo.Constraint(m.T, rule=bess_soh_rule)

        def bess_ramp_pos_rule(mm, t):
            if t == self.Ωt[0]:
                return mm.RampBESS[t] >= mm.PBESS[t] - 0.0
            t_prev = self.Ωt[self.Ωt.index(t) - 1]
            return mm.RampBESS[t] >= mm.PBESS[t] - mm.PBESS[t_prev]

        def bess_ramp_neg_rule(mm, t):
            if t == self.Ωt[0]:
                return mm.RampBESS[t] >= -(mm.PBESS[t] - 0.0)
            t_prev = self.Ωt[self.Ωt.index(t) - 1]
            return mm.RampBESS[t] >= -(mm.PBESS[t] - mm.PBESS[t_prev])

        m.bess_ramp_pos_constr = pyo.Constraint(m.T, rule=bess_ramp_pos_rule)
        m.bess_ramp_neg_constr = pyo.Constraint(m.T, rule=bess_ramp_neg_rule)

        def ev_charge_discharge_rule(mm, t):
            return mm.Pev[t] == mm.Pev_c[t] - mm.Pev_d[t]

        m.ev_charge_discharge_constr = pyo.Constraint(m.T, rule=ev_charge_discharge_rule)

        def ev_charge_rule(mm, t):
            return mm.Pev_c[t] <= self.ev.Pmax_c * mm.gamma_EV_c[t]

        def ev_discharge_rule(mm, t):
            return mm.Pev_d[t] <= self.ev.Pmax_d * mm.gamma_EV_d[t]

        def ev_binary_rule(mm, t):
            return mm.gamma_EV_c[t] + mm.gamma_EV_d[t] <= 1

        m.ev_charge_constr = pyo.Constraint(m.T, rule=ev_charge_rule)
        m.ev_discharge_constr = pyo.Constraint(m.T, rule=ev_discharge_rule)
        m.ev_binary_constr = pyo.Constraint(m.T, rule=ev_binary_rule)

        status = {t: float(self.ev.profile.get(t, 0.0)) for t in self.Ωt}

        def is_arrival(t):
            if t == self.Ωt[0]:
                if status[t] > 0.01:
                    return True
                else:
                    return False
            else:
                t_prev = self.Ωt[self.Ωt.index(t) - 1]
                return (status[t] > 0.01) and (status[t_prev] <= 0.01)

        def ev_zero_when_disconnected_rule(mm, t):
            if status[t] <= 0.01:
                return mm.Pev[t] == 0.0
            if is_arrival(t):
                return mm.Pev[t] == 0.0
            return pyo.Constraint.Skip
        m.ev_zero_power_disconnected = pyo.Constraint(m.T, rule=ev_zero_when_disconnected_rule)

        
        def ev_energy_rule(mm, t):
            if status[t] <= 0.01:
                return mm.Eev[t] == 0.0
            if is_arrival(t):
                start_e = status[t] * float(self.ev.Emax)
                return mm.Eev[t] == start_e
            t_prev = self.Ωt[self.Ωt.index(t) - 1]
            delta = (mm.Pev_c[t] * self.ev.eta - mm.Pev_d[t] / self.ev.eta) * self.dt_hours
            return mm.Eev[t] == self._ev_retention * mm.Eev[t_prev] + delta
        m.ev_energy_constr = pyo.Constraint(m.T, rule=ev_energy_rule)

        def ev_soh_rule(mm, t):
            if t == self.Ωt[0]:
                return mm.SOH_EV[t] == float(self.ev.soh_init)
            t_prev = self.Ωt[self.Ωt.index(t) - 1]
            if status[t] <= 0.01:
                return mm.SOH_EV[t] == mm.SOH_EV[t_prev]
            thr = (mm.Pev_c[t] * self.ev.eta + mm.Pev_d[t] / self.ev.eta) * self.dt_hours
            deg_factor = (1.0 - float(self.ev.beta)) / max(self.ev.Emax * self.ev.DoD, 1e-6)
            return mm.SOH_EV[t] == mm.SOH_EV[t_prev] - deg_factor * thr
        m.ev_soh_constr = pyo.Constraint(m.T, rule=ev_soh_rule)


        def ev_ramp_pos_rule(mm, t):
            if status[t] <= 0.01:
                return mm.RampEV[t] == 0.0
            if t == self.Ωt[0]:
                return mm.RampEV[t] >= mm.Pev[t] - 0.0
            t_prev = self.Ωt[self.Ωt.index(t) - 1]
            prev_val = 0.0 if status[t_prev] <= 0.01 else mm.Pev[t_prev]
            return mm.RampEV[t] >= mm.Pev[t] - prev_val

        def ev_ramp_neg_rule(mm, t):
            if status[t] <= 0.01:
                return pyo.Constraint.Skip
            if t == self.Ωt[0]:
                return mm.RampEV[t] >= -(mm.Pev[t] - 0.0)
            t_prev = self.Ωt[self.Ωt.index(t) - 1]
            prev_val = 0.0 if status[t_prev] <= 0.01 else mm.Pev[t_prev]
            return mm.RampEV[t] >= -(mm.Pev[t] - prev_val)

        m.ev_ramp_pos_constr = pyo.Constraint(m.T, rule=ev_ramp_pos_rule)
        m.ev_ramp_neg_constr = pyo.Constraint(m.T, rule=ev_ramp_neg_rule)

        def grid_ex_import_rule(mm, t):
            return mm.GridExImport[t] >= mm.Pgrid[t] - self.grid.Pmax_import

        def grid_ex_export_rule(mm, t):
            return mm.GridExExport[t] >= -mm.Pgrid[t] - self.grid.Pmax_export

        m.grid_ex_import_constr = pyo.Constraint(m.T, rule=grid_ex_import_rule)
        m.grid_ex_export_constr = pyo.Constraint(m.T, rule=grid_ex_export_rule)

        def is_departure(t):
            if t == self.Ωt[0]:
                return False
            t_prev = self.Ωt[self.Ωt.index(t) - 1]
            return (status[t] <= 0.01) and (status[t_prev] > 0.01)
        
        def ev_dep_short_rule(mm, t, k):
            if not is_departure(t):
                return mm.EV_DepShort[t, k] == 0.0
            t_prev = self.Ωt[self.Ωt.index(t) - 1]
            soc_dep = mm.Eev[t_prev] / max(self.ev.Emax, 1e-6)
            thr = float(self.ev.dep_thresholds[k])
            return mm.EV_DepShort[t, k] >= thr - soc_dep
        m.ev_dep_short_constr = pyo.Constraint(m.T, m.K, rule=ev_dep_short_rule)
    
        def ev_dep_penalty_rule(mm, t):
            if not is_departure(t):
                return mm.EV_DepPenalty[t] == 0.0
            mult = float(self.ev.penalty_coef) * float(self.ev.penalty_departure)
            expr = sum(float(self.ev.dep_weights[k]) * mm.EV_DepShort[t, k] for k in K)
            return mm.EV_DepPenalty[t] == mult * expr
        m.ev_dep_penalty_constr = pyo.Constraint(m.T, rule=ev_dep_penalty_rule)

        def objective_rule(mm):
            energy_cost = sum(self._tariff_lookup.get(t, 0.0) * mm.Pgrid[t] for t in self.Ωt) * self.dt_hours
            curtailment_cost = (
                float(self.curtailment_penalty)
                * sum(self._pv_available_lookup.get(t, 0.0) * mm.chi_pv[t] for t in self.Ωt)
                * self.dt_hours
            )
            bess_throughput = sum(
                (mm.PBESS_c[t] * self.bess.eta + mm.PBESS_d[t] / self.bess.eta) * self.dt_hours
                for t in self.Ωt
            )
            ev_throughput = sum(
                (mm.Pev_c[t] * self.ev.eta + mm.Pev_d[t] / self.ev.eta) * self.dt_hours
                for t in self.Ωt
            )
            bess_degradation = float(self.bess.cost_per_kwh) * bess_throughput
            ev_degradation = float(self.ev.cost_per_kwh) * ev_throughput
            bess_ramp_cost = float(self.bess.ramp_penalty) * sum(mm.RampBESS[t] for t in self.Ωt) * self.dt_hours
            ev_ramp_cost = float(self.ev.ramp_penalty) * sum(mm.RampEV[t] for t in self.Ωt) * self.dt_hours
            grid_penalty_lin = (
                float(self.grid.net_penalty)
                * sum(mm.GridExImport[t] + mm.GridExExport[t] for t in self.Ωt)
                * self.dt_hours
            )

            if float(self.grid.net_penalty_quadratic) > 0:
                grid_penalty_quad = (
                    float(self.grid.net_penalty_quadratic)
                    * sum((mm.GridExImport[t] + mm.GridExExport[t]) ** 2 for t in self.Ωt)
                    * self.dt_hours
                )
            else:
                grid_penalty_quad = 0.0

            ev_departure_penalty = sum(mm.EV_DepPenalty[t] for t in self.Ωt)

            return (
                energy_cost
                + curtailment_cost
                + bess_degradation
                + ev_degradation
                + bess_ramp_cost
                + ev_ramp_cost
                + grid_penalty_lin
                + grid_penalty_quad
                + ev_departure_penalty
            )

        m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        self.model = m
        return self

    def solve(self, solver_name="gurobi"):
        solver = pyo.SolverFactory(solver_name)
        self.results = solver.solve(self.model, tee=True)
        return self.results

    def results_df(self):
        if not hasattr(self, "model"):
            raise RuntimeError("Call build() before requesting results.")
        if self._results_cache is not None:
            return self._results_cache.copy()

        m = self.model
        idx = pd.DatetimeIndex(self.Ωt)

        def _val(v):
            try:
                out = pyo.value(v, exception=False)
            except Exception:
                out = None
            if out is None or (isinstance(out, float) and math.isnan(out)):
                return 0.0
            return float(out)

        data = {
            "Pgrid": [_val(m.Pgrid[t]) for t in self.Ωt],
            "PBESS": [_val(m.PBESS[t]) for t in self.Ωt],
            "PBESS_c": [_val(m.PBESS_c[t]) for t in self.Ωt],
            "PBESS_d": [_val(m.PBESS_d[t]) for t in self.Ωt],
            "EBESS": [_val(m.EBESS[t]) for t in self.Ωt],
            "SOH_BESS": [_val(m.SOH_BESS[t]) for t in self.Ωt],
            "RampBESS": [_val(m.RampBESS[t]) for t in self.Ωt],
            "PEV": [_val(m.Pev[t]) for t in self.Ωt],
            "PEV_c": [_val(m.Pev_c[t]) for t in self.Ωt],
            "PEV_d": [_val(m.Pev_d[t]) for t in self.Ωt],
            "Eev": [_val(m.Eev[t]) for t in self.Ωt],
            "SOH_EV": [_val(m.SOH_EV[t]) for t in self.Ωt],
            "RampEV": [_val(m.RampEV[t]) for t in self.Ωt],
            "chi_pv": [_val(m.chi_pv[t]) for t in self.Ωt],
            "GridExImport": [_val(m.GridExImport[t]) for t in self.Ωt],
            "GridExExport": [_val(m.GridExExport[t]) for t in self.Ωt],
            "EV_DepPenalty": [_val(m.EV_DepPenalty[t]) for t in self.Ωt],
            "ev_status": [float(self.ev.profile.get(t, 0.0)) for t in self.Ωt],
        }

        df = pd.DataFrame(data, index=idx)

        df["tariff"] = [float(self.grid.tariff.get(t, 0.0)) for t in self.Ωt]
        df["Ppv"] = [float(self.pv.profile.get(t, 0.0)) for t in self.Ωt]
        df["Load"] = [float(self.load.profile.get(t, 0.0)) for t in self.Ωt]
        df["pv_available"] = df["Ppv"]
        df["pv_used"] = df["pv_available"] * (1.0 - df["chi_pv"])

        self._results_cache = df
        return df.copy()

    def _normalize_value(self, key, value):
        cfg = self.norm_cfg.get(key)
        if isinstance(cfg, dict):
            vmin = cfg.get("min")
            vmax = cfg.get("max")
            if vmin is not None and vmax is not None and vmax != vmin:
                norm = (value - vmin) / (vmax - vmin)
                return float(min(max(norm, 0.0), 1.0))
        return float(value)

    def _resolve_mask(self, state_mask):
        mask = state_mask if state_mask is not None else getattr(self, "state_mask", None)
        if mask is None:
            return None
        mask_array = np.array(mask, dtype=bool)
        return mask_array

    def _compute_state_snapshots(self):
        if self._state_history:
            return self._state_history

        results = self.results_df()
        base_segment = self._segment.reindex(pd.DatetimeIndex(self.Ωt)).ffill().bfill()

        snapshots = []
        for idx, ts in enumerate(self.Ωt):
            row = base_segment.loc[ts] if ts in base_segment.index else base_segment.iloc[idx]

            climate_feats = [
                self._normalize_value(col, float(row.get(col, 0.0))) for col in self.climate_columns
            ]
            climate_raw = [float(row.get(col, 0.0)) for col in self.climate_columns]

            timestamp = pd.to_datetime(ts)
            time_feats = _build_time_features(timestamp)

            soc_bess = float(results.loc[ts, "EBESS"] / max(self.bess.Emax, 1e-6))
            soh_bess = float(results.loc[ts, "SOH_BESS"])
            bess_obs = [soc_bess, soh_bess]

            ev_connected = 1.0 if float(results.loc[ts, "ev_status"]) > 0.01 else 0.0
            soc_ev = float(results.loc[ts, "Eev"] / max(self.ev.Emax, 1e-6))
            soh_ev = float(results.loc[ts, "SOH_EV"])
            ev_obs = [soc_ev, soh_ev, ev_connected]
            ev_shortfall = max(0.0, 1.0 - soc_ev) if ev_connected > 0.5 else 0.0

            pv_available_kw = float(results.loc[ts, "pv_available"])
            pv_obs = [pv_available_kw / max(self.pv.Pmax, 1e-6)]

            load_kw = float(results.loc[ts, "Load"])
            load_norm = load_kw / max(self.pnorm, 1e-6)
            pv_used_kw = float(results.loc[ts, "pv_used"])
            net_load_kw = load_kw - pv_used_kw
            net_load_norm = net_load_kw / max(self.pnorm, 1e-6)
            tariff_val = float(results.loc[ts, "tariff"])
            tariff_norm = tariff_val / self.tariff_base

            features = (
                climate_feats
                + time_feats
                + bess_obs
                + ev_obs
                + [ev_shortfall]
                + pv_obs
                + [load_norm, tariff_norm, net_load_norm]
            )

            total_steps = len(self.Ωt)
            steps_left = float(total_steps - idx)
            unnormalized = (
                climate_raw
                + time_feats
                + bess_obs
                + ev_obs
                + [ev_shortfall]
                + [pv_available_kw]
                + [load_kw, tariff_val, net_load_kw, steps_left]
            )

            snapshots.append(
                {
                    "step": idx,
                    "timestamp": timestamp,
                    "raw_state": np.array(features, dtype=np.float32),
                    "raw_state_unscaled": np.array(unnormalized, dtype=np.float32),
                }
            )

        self._state_history = snapshots
        return self._state_history

    def _snapshots_to_dataframe(self, snapshots):
        rows = []
        for snap in snapshots:
            row = {"timestamp": snap["timestamp"]}
            row.update({name: float(value) for name, value in zip(self._feature_names, snap["raw_state"])} )
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.set_index("timestamp")

    def _get_history_with_padding(self, sequence, last_n):
        if last_n is None or last_n <= 0:
            return list(sequence)
        if not sequence:
            return []
        if last_n <= len(sequence):
            return list(sequence[-last_n:])
        pad_count = last_n - len(sequence)
        first_entry = copy.deepcopy(sequence[0])
        padded_prefix = [copy.deepcopy(first_entry) for _ in range(pad_count)]
        return padded_prefix + list(sequence)

    def build_state_dataframe(self, state_mask=None, include_full_state=False):
        mask = self._resolve_mask(state_mask)
        snapshots = self._compute_state_snapshots()
        df = self._snapshots_to_dataframe(snapshots)
        if mask is None:
            return df
        masked_cols = [name for name, keep in zip(self._feature_names, mask) if keep]
        if include_full_state:
            return df, df.loc[:, masked_cols]
        return df.loc[:, masked_cols]

    def get_state_history(self, last_n=None, state_mask=None, include_unscaled=False):
        mask = self._resolve_mask(state_mask)
        snapshots = self._compute_state_snapshots()
        history = []
        for snap in snapshots:
            entry = {
                "step": snap["step"],
                "timestamp": snap["timestamp"],
                "raw_state": np.array(snap["raw_state"], copy=True),
            }
            masked_state = snap["raw_state"][mask] if mask is not None else snap["raw_state"]
            entry["masked_state"] = np.array(masked_state, copy=True)
            if include_unscaled and "raw_state_unscaled" in snap:
                entry["raw_state_unscaled"] = np.array(snap["raw_state_unscaled"], copy=True)
            history.append(entry)
        return self._get_history_with_padding(history, last_n)

    def get_masked_observations(self, state_mask=None):
        mask = self._resolve_mask(state_mask)
        snapshots = self._compute_state_snapshots()
        if mask is None:
            observations = [snap["raw_state"] for snap in snapshots]
            labels = list(self._feature_names)
        else:
            observations = [snap["raw_state"][mask] for snap in snapshots]
            labels = [name for name, keep in zip(self._feature_names, mask) if keep]
        array = np.vstack(observations) if observations else np.zeros((0, len(labels)))
        return array, labels

    def get_action_history(self, last_n=None):
        results = self.results_df()
        records = []
        for idx, ts in enumerate(results.index):
            records.append(
                {
                    "step": idx,
                    "timestamp": ts,
                    "PBESS": float(results.loc[ts, "PBESS"]),
                    "Pev": float(results.loc[ts, "PEV"]),
                    "Pgrid": float(results.loc[ts, "Pgrid"]),
                    "Ppv_used": float(results.loc[ts, "pv_used"]),
                    "Ppv_available": float(results.loc[ts, "pv_available"]),
                    "Load": float(results.loc[ts, "Load"]),
                }
            )
        return self._get_history_with_padding(records, last_n)

    def get_full_states(self) -> tuple[np.ndarray, list[str]]:
        snapshots = self._compute_state_snapshots()
        states = [snap["raw_state"] for snap in snapshots]
        return np.vstack(states), list(self._feature_names)

    def apply_state_mask(self, mask_spec=None) -> tuple[np.ndarray, list[str]]:
        states, labels = self.get_full_states()
        mask = self._resolve_mask(mask_spec)
        if mask is None:
            return states, labels
        masked = states[:, mask]
        masked_labels = [name for name, keep in zip(labels, mask) if keep]
        return masked, masked_labels
