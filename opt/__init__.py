import pyomo.environ as pyo
import pandas as pd
import numpy as np
import math
import copy


CLIMATE_COLUMNS = [
    'drybulb_C',
    'relhum_percent',
    'Global Horizontal Radiation',
    'dni_Wm2',
    'dhi_Wm2',
    'Wind Speed (m/s)',
]


def _encode_cycle(value, period):
    angle = 2 * math.pi * (value / period)
    return [math.sin(angle), math.cos(angle)]


def _build_time_features(timestamp):
    if pd.isna(timestamp):
        return [0.0] * 10

    month_val = max(timestamp.month - 1, 0)
    day_val = max(timestamp.day - 1, 0)
    hour_val = timestamp.hour
    minute_val = timestamp.minute
    weekday_val = timestamp.weekday()
    days_in_month = max(timestamp.days_in_month, 1)

    features = []
    features += _encode_cycle(month_val, 12)
    features += _encode_cycle(day_val, days_in_month)
    features += _encode_cycle(hour_val, 24)
    features += _encode_cycle(minute_val, 60)
    features += _encode_cycle(weekday_val, 7)
    return features

class BESS:
    def __init__(self, config):
        self.Pmax    = config.get('Pmax', 5.0)
        self.Emax    = config.get('Emax', 10.0)
        self.η       = config.get('η', 0.95)
        self.β       = config.get('β', 0.999)
        self.DoD     = config.get('DoD', 0.8)
        self.soc_min = config.get('soc_min', 0.1)
        self.soc_max = config.get('soc_max', 0.9)
        self.capex = config.get('capex', 100.0)
        self.ncycles = max(1.0, float(config.get('ncycles', 1.0)))
        self.ramp_penalty = float(config.get('ramp_penalty', 0.0))
        usable_energy = max(self.Emax * self.DoD, 1e-6)
        self.cost_per_kwh = self.capex / (self.ncycles * usable_energy)
        pass


class EV:
    def __init__(self, config, dataframe):
        self.Pmax_c  = config.get('Pmax_c', 7.0)
        self.Pmax_d  = config.get('Pmax_d', 7.0)
        self.Emax    = config.get('Emax', 50.0)
        self.η       = config.get('η', 0.9)
        self.β       = config.get('β', 0.999)
        self.DoD     = config.get('DoD', 0.8)
        self.soc_min = config.get('soc_min', 0.0)
        self.soc_max = config.get('soc_max', 1.0)
        self.capex   = config.get('capex', 0.0)
        self.ncycles = max(1.0, float(config.get('ncycles', 1.0)))
        self.ramp_penalty = float(config.get('ramp_penalty', 0.0))
        usable_energy = max(self.Emax * self.DoD, 1e-6)
        self.cost_per_kwh = self.capex / (self.ncycles * usable_energy)
        profile = dataframe.get('ev_status', pd.Series(dtype=float))
        index = pd.to_datetime(dataframe['timestamp'], format="%d/%m/%Y %H:%M", dayfirst=True)
        self.profile = profile.set_axis(index, copy=False)

class Load:
    def __init__(self, config, dataframe):
        self.Pmax = config.get('Pmax', 10.0)
        profile = dataframe.get('electricity_demand_rate_W', pd.Series(dtype=float))/1000
        index = pd.to_datetime(dataframe['timestamp'], format="%d/%m/%Y %H:%M", dayfirst=True)
        self.profile = profile.set_axis(index, copy=False)

class PV:
    def __init__(self, config, dataframe):
        self.Pmax = config.get('Pmax', 5.0)
        profile = dataframe.get('produced_electricity_rate_W', pd.Series(dtype=float))/1000
        index = pd.to_datetime(dataframe['timestamp'], format="%d/%m/%Y %H:%M", dayfirst=True)
        self.profile = profile.set_axis(index, copy=False)

class Grid:
    def __init__(self, config, dataframe):
        self.Pmax_import = config.get('Pmax_import', 10.0)
        self.Pmax_export = config.get('Pmax_export', 10.0)
        column_name = config.get('tariff_column', "tar_tou")
        profile = dataframe.get(column_name, pd.Series(dtype=float))
        index = pd.to_datetime(dataframe['timestamp'], format="%d/%m/%Y %H:%M", dayfirst=True)
        self.tariff = profile.set_axis(index, copy=False)
        pass

class Teacher:
    def __init__(self, config, dataframe, start_date = None, days = 5, state_mask=None):
        self.config = config or {}
        timestamps = pd.to_datetime(
            dataframe['timestamp'],
            format="%d/%m/%Y %H:%M",
            dayfirst=True,
            errors='coerce',
        )
        self._timeline = pd.DatetimeIndex(timestamps)

        self.load = Load(self.config.get('Load', {}), dataframe)
        self.grid = Grid(self.config.get('Grid', {}), dataframe)
        self.pv   = PV(self.config.get('PV', {}), dataframe)
        self.ev   = EV(self.config.get('EV', {}), dataframe)
        self.bess = BESS(self.config.get('BESS', {}))

        general_cfg = self.config.get('general', {})
        self.norm_cfg = general_cfg.get('state_normalization', {})
        self.pnorm = general_cfg.get('Pnorm', self.norm_cfg.get('Pmax', 1.0))
        self.tariff_base = max(self.norm_cfg.get('tariff_base', 1.0), 1e-6)
        self.climate_columns = self.norm_cfg.get('climate_columns', CLIMATE_COLUMNS)
        self.curtailment_penalty = float(general_cfg.get('pv_curtailment_penalty', 0.01))

        self.Δt = general_cfg.get("timestep", 5.0)/60
        steps_needed = max(1, int(days * 24 / self.Δt))

        if start_date is None:
            start_idx = 0
        else:
            requested_start = pd.to_datetime(start_date, dayfirst=True)
            start_idx = int(self._timeline.searchsorted(requested_start, side='left'))

        end_idx = min(len(self._timeline), start_idx + steps_needed)

        self.start_date = self._timeline[start_idx]
        self.Ωt = list(self._timeline[start_idx:end_idx])
        self._pv_available_lookup = {ts: float(self.pv.profile[ts]) for ts in self.Ωt}
        segment = dataframe.iloc[start_idx:end_idx].copy()
        segment['timestamp'] = pd.to_datetime(
            segment['timestamp'],
            format="%d/%m/%Y %H:%M",
            dayfirst=True,
            errors='coerce',
        )
        self._segment = segment.set_index('timestamp')
        self._state_history = []
        self._results_cache = None

        climate_len = len(self.climate_columns)
        time_len = 10
        bess_len = 2
        ev_len = 3
        pv_len = 1
        extra_terms = 3
        shortfall_terms = 1
        self._full_obs_len = climate_len + time_len + bess_len + ev_len + shortfall_terms + pv_len + extra_terms
        self._feature_names = (
            [f"climate_{col}" for col in self.climate_columns] +
            ["month_sin","month_cos","day_sin","day_cos","hour_sin","hour_cos",
             "minute_sin","minute_cos","weekday_sin","weekday_cos"] +
            ["bess_soc","bess_soh"] +
            ["ev_soc","ev_soh","ev_connected"] +
            ["ev_shortfall"] +
            ["pv_available"] +
            ["load","tariff","net_load"]
        )
        self.state_mask = None
        self.state_feature_labels = list(self._feature_names)
        if state_mask is not None:
            resolved_mask = self._resolve_mask(state_mask)
            self.state_mask = resolved_mask
            self.state_feature_labels = [
                name for name, keep in zip(self._feature_names, resolved_mask) if keep
            ]
        return


    def build(self, start_soc):
    
        # Variables for optimization
        self.model = pyo.ConcreteModel()
        self._bess_retention = max(min(self.bess.β, 1.0), 0.0)
        start_energy = start_soc * self.bess.Emax
        self._prev_time = {ts: (self.Ωt[idx - 1] if idx > 0 else None) for idx, ts in enumerate(self.Ωt)}
        
        self.model.Pgrid    = pyo.Var(self.Ωt, bounds=(-self.grid.Pmax_export, self.grid.Pmax_import))
        self.model.PBESS    = pyo.Var(self.Ωt, bounds=(-self.bess.Pmax, self.bess.Pmax))
        self.model.EBESS    = pyo.Var(self.Ωt, bounds=(self.bess.soc_min * self.bess.Emax, self.bess.soc_max * self.bess.Emax))
        self.model.PBESS_c  = pyo.Var(self.Ωt, bounds=(0, self.bess.Pmax))
        self.model.PBESS_d  = pyo.Var(self.Ωt, bounds=(0, self.bess.Pmax))
        self.model.γBESS_c  = pyo.Var(self.Ωt, domain=pyo.Binary)
        self.model.γBESS_d  = pyo.Var(self.Ωt, domain=pyo.Binary)
        

        self.model.χPV = pyo.Var(self.Ωt, bounds=(0, 1))


        self.model.Pev   = pyo.Var(self.Ωt, bounds=(-self.ev.Pmax_d, self.ev.Pmax_c))
        self.model.Pev_c = pyo.Var(self.Ωt, bounds=(0, self.ev.Pmax_c))
        self.model.Pev_d = pyo.Var(self.Ωt, bounds=(0, self.ev.Pmax_d))
        self.model.γev_c = pyo.Var(self.Ωt, pyo.Binary)
        self.model.γev_d = pyo.Var(self.Ωt, pyo.Binary)

        self.model.Eev = pyo.Var(self.Ωt, bounds=(0, self.ev.Emax))
        self.model.RampBESS = pyo.Var(self.Ωt, domain=pyo.NonNegativeReals)
        self.model.RampEV = pyo.Var(self.Ωt, domain=pyo.NonNegativeReals)

        # objective function
        def objective_rule(m):
            energy_cost         = sum(m.Pgrid[t] * self.grid.tariff[t] * self.Δt for t in self.Ωt)
            curtailment_cost    = sum(self.pv.profile[t] * m.χPV[t] * self.curtailment_penalty * self.Δt for t in self.Ωt)
            eta_bess = max(self.bess.η, 1e-6)
            eta_ev = max(self.ev.η, 1e-6)
            bess_throughput = sum(
                (m.PBESS_c[t] * eta_bess + m.PBESS_d[t] / eta_bess) * self.Δt
                for t in self.Ωt
            )
            ev_throughput = sum(
                (m.Pev_c[t] * eta_ev + m.Pev_d[t] / eta_ev) * self.Δt
                for t in self.Ωt
            )
            bess_degradation = self.bess.cost_per_kwh * bess_throughput
            ev_degradation = self.ev.cost_per_kwh * ev_throughput
            bess_ramp_cost = self.bess.ramp_penalty * sum(m.RampBESS[t] for t in self.Ωt)
            ev_ramp_cost = self.ev.ramp_penalty * sum(m.RampEV[t] for t in self.Ωt)
            return (
                energy_cost
                + curtailment_cost
                + bess_degradation
                + ev_degradation
                + bess_ramp_cost
                + ev_ramp_cost
            )
        self.model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


        def power_balance_rule(m, t):
            return (self.load.profile[t] + m.PBESS[t] + self.model.Pev[t] == m.Pgrid[t] + self.pv.profile[t] * (1 - m.χPV[t]))
        self.model.power_balance_constr = pyo.Constraint(self.Ωt, rule=power_balance_rule)

        def pv_zero_output_rule(m, t):
            if self._pv_available_lookup.get(t, 0.0) == 0.0:
                return m.χPV[t] == 0.0
            return pyo.Constraint.Skip
        self.model.pv_zero_output_constr = pyo.Constraint(self.Ωt, rule=pv_zero_output_rule)
        
        #### BESS constraints
        def bess_charge_discharge_rule(m, t):
            return m.PBESS[t] == m.PBESS_c[t] - m.PBESS_d[t]
        self.model.bess_charge_discharge_constr = pyo.Constraint(self.Ωt, rule=bess_charge_discharge_rule)

        def bess_charge_rule(m, t):
            return m.PBESS_c[t] <= self.bess.Pmax * m.γBESS_c[t]
        self.model.bess_charge_constr = pyo.Constraint(self.Ωt, rule=bess_charge_rule)

        def bess_discharge_rule(m, t):
            return m.PBESS_d[t] <= self.bess.Pmax * m.γBESS_d[t]
        self.model.bess_discharge_constr = pyo.Constraint(self.Ωt, rule=bess_discharge_rule)

        def bess_soc_rule(m, t):
            delta = (m.PBESS_c[t] * self.bess.η - m.PBESS_d[t] / self.bess.η) * self.Δt
            if t == self.Ωt[0]:
                return m.EBESS[t] == self._bess_retention * start_energy + delta
            else:
                t_prev = self.Ωt[self.Ωt.index(t) - 1]
                return m.EBESS[t] == self._bess_retention * m.EBESS[t_prev] + delta
        self.model.bess_soc_constr = pyo.Constraint(self.Ωt, rule=bess_soc_rule)

        def bess_binary_rule(m, t):
            return m.γBESS_c[t] + m.γBESS_d[t] <= 1
        self.model.bess_binary_constr = pyo.Constraint(self.Ωt, rule=bess_binary_rule)

        def bess_ramp_pos_rule(m, t):
            prev = self._prev_time[t]
            prev_power = 0.0 if prev is None else m.PBESS[prev]
            return m.PBESS[t] - prev_power <= m.RampBESS[t]
        self.model.bess_ramp_pos_constr = pyo.Constraint(self.Ωt, rule=bess_ramp_pos_rule)

        def bess_ramp_neg_rule(m, t):
            prev = self._prev_time[t]
            prev_power = 0.0 if prev is None else m.PBESS[prev]
            return -(m.PBESS[t] - prev_power) <= m.RampBESS[t]
        self.model.bess_ramp_neg_constr = pyo.Constraint(self.Ωt, rule=bess_ramp_neg_rule)


        #### EV constraints
        def ev_charge_discharge_rule(m, t):
            return m.Pev[t] == m.Pev_c[t] - m.Pev_d[t]
        self.model.ev_charge_discharge_constr = pyo.Constraint(self.Ωt, rule=ev_charge_discharge_rule)

        def ev_power_setting_rule(m, t):
            if self.ev.profile[t] == 0:
                return m.Pev[t] == 0
            else:
                return pyo.Constraint.Skip
        self.model.ev_power_setting_constr = pyo.Constraint(self.Ωt, rule=ev_power_setting_rule)


        def ev_soc_setting_rule(m, t):
            if self.ev.profile[t] == 0:
                if t != self.Ωt[0]:
                    if self.ev.profile[self.Ωt[self.Ωt.index(t) - 1]] > 0:
                        return m.Eev[t] == self.ev.Emax
                return m.Eev[t] == 0
            else:
                if t == self.Ωt[0]:
                    return m.Eev[t] == self.ev.profile[t] * self.ev.Emax
                elif self.ev.profile[self.Ωt[self.Ωt.index(t) - 1]] == 0:
                    return m.Eev[t] == self.ev.profile[t] * self.ev.Emax
                return pyo.Constraint.Skip
        self.model.ev_soc_setting_constr = pyo.Constraint(self.Ωt, rule=ev_soc_setting_rule)


        def ev_energy_rule(m, t):
            if self.ev.profile[t] == 0:
                if t != self.Ωt[0]:
                    if self.ev.profile[self.Ωt[self.Ωt.index(t) - 1]] > 0:
                        t_prev = self.Ωt[self.Ωt.index(t) - 1]
                        return m.Eev[t] == m.Eev[t_prev] + self.Δt * (m.Pev_c[t] * self.ev.η - m.Pev_d[t] / self.ev.η)
                return pyo.Constraint.Skip
            else:
                t_prev = self.Ωt[self.Ωt.index(t) - 1]
                if self.ev.profile[self.Ωt[self.Ωt.index(t) - 1]] == 0:
                    return pyo.Constraint.Skip
                return m.Eev[t] == m.Eev[t_prev] + self.Δt * (m.Pev_c[t] * self.ev.η - m.Pev_d[t] / self.ev.η)
        self.model.ev_energy_constr = pyo.Constraint(self.Ωt, rule=ev_energy_rule)

        def ev_ramp_pos_rule(m, t):
            prev = self._prev_time[t]
            prev_power = 0.0 if prev is None else m.Pev[prev]
            return m.Pev[t] - prev_power <= m.RampEV[t]
        self.model.ev_ramp_pos_constr = pyo.Constraint(self.Ωt, rule=ev_ramp_pos_rule)

        def ev_ramp_neg_rule(m, t):
            prev = self._prev_time[t]
            prev_power = 0.0 if prev is None else m.Pev[prev]
            return -(m.Pev[t] - prev_power) <= m.RampEV[t]
        self.model.ev_ramp_neg_constr = pyo.Constraint(self.Ωt, rule=ev_ramp_neg_rule)

        
        return
    

    def solve(self, solver_name="gurobi"):
        solver = pyo.SolverFactory(solver_name)
        self.results = solver.solve(self.model, tee=True)
        return self.results
    

    def results_df(self):
        if not hasattr(self, 'model'):
            raise RuntimeError("Call build() before requesting results.")
        if self._results_cache is not None:
            return self._results_cache.copy()

        data = {
            'Pgrid': [pyo.value(self.model.Pgrid[t]) for t in self.Ωt],
            'Ppv':   [self.pv.profile[t] for t in self.Ωt],
            'Load':  [self.load.profile[t] for t in self.Ωt],
            'PBESS': [pyo.value(self.model.PBESS[t]) for t in self.Ωt],
            'EBESS': [pyo.value(self.model.EBESS[t]) for t in self.Ωt],
            'Pev':   [pyo.value(self.model.Pev[t]) for t in self.Ωt],
            'Eev':   [pyo.value(self.model.Eev[t]) for t in self.Ωt],
            'ev_status': [self.ev.profile[t] for t in self.Ωt],
            'chi_pv': [pyo.value(self.model.χPV[t]) for t in self.Ωt],
        }
        index = pd.DatetimeIndex(self.Ωt)
        df = pd.DataFrame(data, index=index)
        df['tariff'] = [self.grid.tariff[t] for t in self.Ωt]
        df['pv_available'] = df['Ppv']
        df['pv_used'] = df['pv_available'] * (1.0 - df['chi_pv'])
        self._results_cache = df
        return df.copy()

    def _normalize_value(self, key, value):
        cfg = self.norm_cfg.get(key)
        if isinstance(cfg, dict):
            vmin = cfg.get('min')
            vmax = cfg.get('max')
            if vmin is not None and vmax is not None and vmax != vmin:
                norm = (value - vmin) / (vmax - vmin)
                return float(min(max(norm, 0.0), 1.0))
        return float(value)

    def _resolve_mask(self, state_mask):
        mask = state_mask if state_mask is not None else getattr(self, 'state_mask', None)
        if mask is None:
            return None
        mask_array = np.array(mask, dtype=bool)
        if mask_array.size != self._full_obs_len:
            raise ValueError("state_mask length does not match observation size")
        return mask_array

    def _compute_state_snapshots(self):
        if self._state_history:
            return self._state_history

        results = self.results_df()
        base_segment = self._segment.reindex(pd.DatetimeIndex(self.Ωt)).ffill().bfill()
        snapshots = []
        total_steps = len(self.Ωt)
        for idx, ts in enumerate(self.Ωt):
            row = base_segment.loc[ts] if ts in base_segment.index else base_segment.iloc[idx]
            climate_feats = [
                self._normalize_value(col, float(row.get(col, 0.0)))
                for col in self.climate_columns
            ]
            climate_raw = [float(row.get(col, 0.0)) for col in self.climate_columns]

            timestamp = pd.to_datetime(ts)
            time_feats = _build_time_features(timestamp)

            soc_bess = float(results.loc[ts, 'EBESS'] / max(self.bess.Emax, 1e-6))
            bess_obs = [soc_bess, 1.0]

            soc_ev = float(results.loc[ts, 'Eev'] / max(self.ev.Emax, 1e-6))
            ev_connected = 1.0 if float(results.loc[ts, 'ev_status']) > 0 else 0.0
            ev_obs = [soc_ev, 1.0, ev_connected]
            ev_shortfall = max(0.0, 1.0 - soc_ev)

            pv_available_kw = float(results.loc[ts, 'pv_available'])
            pv_obs = [pv_available_kw / max(self.pv.Pmax, 1e-6)]

            load_kw = float(results.loc[ts, 'Load'])
            load_norm = load_kw / max(self.pnorm, 1e-6)
            pv_used_kw = float(results.loc[ts, 'pv_used'])
            net_load_kw = load_kw - pv_used_kw
            net_load_norm = net_load_kw / max(self.pnorm, 1e-6)
            tariff_val = float(results.loc[ts, 'tariff'])
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

            snapshots.append({
                'step': idx,
                'timestamp': timestamp,
                'raw_state': np.array(features, dtype=np.float32),
                'raw_state_unscaled': np.array(unnormalized, dtype=np.float32),
            })

        self._state_history = snapshots
        return self._state_history

    def _snapshots_to_dataframe(self, snapshots):
        rows = []
        for snap in snapshots:
            row = {'timestamp': snap['timestamp']}
            row.update({name: float(value) for name, value in zip(self._feature_names, snap['raw_state'])})
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.set_index('timestamp')

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
                'step': snap['step'],
                'timestamp': snap['timestamp'],
                'raw_state': np.array(snap['raw_state'], copy=True),
            }
            masked_state = snap['raw_state'][mask] if mask is not None else snap['raw_state']
            entry['masked_state'] = np.array(masked_state, copy=True)
            if include_unscaled and 'raw_state_unscaled' in snap:
                entry['raw_state_unscaled'] = np.array(snap['raw_state_unscaled'], copy=True)
            history.append(entry)
        return self._get_history_with_padding(history, last_n)

    def get_masked_observations(self, state_mask=None):
        mask = self._resolve_mask(state_mask)
        snapshots = self._compute_state_snapshots()
        if mask is None:
            observations = [snap['raw_state'] for snap in snapshots]
            labels = list(self._feature_names)
        else:
            observations = [snap['raw_state'][mask] for snap in snapshots]
            labels = [name for name, keep in zip(self._feature_names, mask) if keep]
        array = np.vstack(observations) if observations else np.zeros((0, len(labels)))
        return array, labels

    def get_action_history(self, last_n=None):
        results = self.results_df()
        records = []
        for idx, ts in enumerate(results.index):
            records.append({
                'step': idx,
                'timestamp': ts,
                'PBESS': float(results.loc[ts, 'PBESS']),
                'Pev': float(results.loc[ts, 'Pev']),
                'Pgrid': float(results.loc[ts, 'Pgrid']),
                'Ppv_used': float(results.loc[ts, 'pv_used']),
                'Ppv_available': float(results.loc[ts, 'pv_available']),
                'Load': float(results.loc[ts, 'Load']),
            })
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