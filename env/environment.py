from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pandas as pd
import copy


CLIMATE_COLUMNS = [
    'drybulb_C',
    'relhum_percent',
    'Global Horizontal Radiation',
    'dni_Wm2',
    'dhi_Wm2',
    'Wind Speed (m/s)',
]


class Simulation:
    def __init__(self, config=None, dataframe=None):
        self.config          = config or {}
        self.dataframe       = dataframe.copy() if dataframe is not None else None
        if self.dataframe is not None and 'timestamp' in self.dataframe:
            self.dataframe['timestamp'] = pd.to_datetime(
                self.dataframe['timestamp'],
                format="%d/%m/%Y %H:%M",
                dayfirst=True,
                errors='coerce',
            )
            self.timestamps = pd.DatetimeIndex(self.dataframe['timestamp'])
        else:
            self.timestamps = pd.DatetimeIndex([])
        self.duration        = self.config.get('simulation_duration', 24)
        self.simulation_days = self.config.get('simulation_days', 1)
        self.timestep        = self.config.get('timestep', 5)
        self.norm_cfg        = self.config.get('state_normalization', {})
        self.pnorm           = self.config.get('Pnorm', self.norm_cfg.get('Pmax', 1.0))
        self.tariff_base     = max(self.norm_cfg.get('tariff_base', 1.0), 1e-6)
        self.climate_columns = self.norm_cfg.get('climate_columns', CLIMATE_COLUMNS)
        self.state_history   = []
        self.action_history  = []

        self.start_step = 0
        self.current_step = 0
        self.num_steps = self._compute_num_steps()
        self.dt = self.timestep / 60.0
        return

    def reset(self):
        self.num_steps      = self._compute_num_steps()
        start_idx = int(self.start_step)
        self.current_step   = start_idx
        self.final_step     = min(start_idx + self.num_steps - 1, max(len(self.dataframe) - 1, 0))
        self.current_datetime = pd.to_datetime(self.dataframe.timestamp[self.current_step])
        self.state_history  = []
        self.action_history = []
        return

    def step(self):
        final_idx = min(
            getattr(self, 'final_step', self.num_steps - 1),
            max(len(self.dataframe) - 1, 0),
        )
        if self.current_step >= final_idx:
            return True
        self.current_step += 1
        #get current datetime from dataframe using self.current_step
        self.current_datetime = pd.to_datetime(self.dataframe.timestamp[self.current_step])
        return False

    def _compute_num_steps(self):
        total_hours = self.duration * self.simulation_days
        total_minutes = total_hours * 60
        planned_steps = max(1, int(total_minutes / max(self.timestep, 1)))
        remaining = max(0, len(self.dataframe) - int(self.start_step))
        return min(planned_steps, remaining)

    def _get_normalized_state(self):
        #read the current step from dataframe and normalize it
        state = self.dataframe.iloc[self.current_step]
        return state

    def get_value(self, column, default=0.0):
        idx = min(max(self.current_step, 0), len(self.dataframe) - 1)
        value = self.dataframe.iloc[idx][column]
        return float(value)

    def normalize_value(self, key, value):
        cfg = self.norm_cfg.get(key)
        if isinstance(cfg, dict):
            vmin = cfg.get('min')
            vmax = cfg.get('max')
            if vmin is not None and vmax is not None and vmax != vmin:
                norm = (value - vmin) / (vmax - vmin)
                return float(np.clip(norm, 0.0, 1.0))
        return float(value)

    def get_climate_features(self, row):
        return [
            self.normalize_value(col, float(row.get(col, 0.0)))
            for col in self.climate_columns
        ]

    def record_state(self, raw_state, masked_state=None, unnormalized_state=None):
        timestamp = getattr(self, 'current_datetime', None) or pd.to_datetime(
            self.dataframe.iloc[self.current_step].get('timestamp')
            if self.dataframe is not None else None,
            errors='coerce',
        )
        snapshot = {
            'step': self.current_step,
            'timestamp': timestamp,
            'raw_state': np.array(raw_state, copy=True),
        }
        if masked_state is not None:
            snapshot['masked_state'] = np.array(masked_state, copy=True)
        if unnormalized_state is not None:
            snapshot['raw_state_unscaled'] = np.array(unnormalized_state, copy=True)
        self.state_history.append(snapshot)

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

    def get_state_history(self, last_n=None):
        if last_n is None:
            return list(self.state_history)
        return self._get_history_with_padding(self.state_history, last_n)

    def record_action(self, pbess, pev, xpvc, pgrid, ppv_used, ppv_curtailed, pbess_cmd=None, pev_cmd=None):
        timestamp = getattr(self, 'current_datetime', None)
        entry = {
            'step': self.current_step,
            'timestamp': timestamp,
            'timestep_minutes': self.timestep,
            'PBESS': float(pbess),
            'PEV': float(pev),
            'XPV': float(xpvc),
            'pgrid': float(pgrid),
            'ppv_used': float(ppv_used),
            'ppv_curtailed': float(ppv_curtailed),
        }
        if pbess_cmd is not None:
            entry['PBESS_cmd'] = float(pbess_cmd)
        if pev_cmd is not None:
            entry['Pev_cmd'] = float(pev_cmd)
        self.action_history.append(entry)

    def get_action_history(self, last_n=None):
        if last_n is None:
            return list(self.action_history)
        return self._get_history_with_padding(self.action_history, last_n)

    def get_unnormalized_state_history(self, last_n=None):
        unscaled = [snap for snap in self.state_history if 'raw_state_unscaled' in snap]
        if last_n is not None:
            unscaled = self._get_history_with_padding(unscaled, last_n)
        return [
            {
                'step': snap['step'],
                'timestamp': snap.get('timestamp'),
                'raw_state_unscaled': np.array(snap['raw_state_unscaled'], copy=True),
            }
            for snap in unscaled
        ]

    def set_start_step(self, start_step):
        if self.dataframe is None or len(self.dataframe) == 0:
            self.start_step = 0
            self.current_step = 0
            return
        self.start_step = int(start_step)
        self.current_step = int(start_step)

    def set_start_date(self, start_date):
        if self.timestamps.empty:
            self.set_start_step(0)
            return
        target = pd.to_datetime(start_date, dayfirst=True)
        ts_array = self.timestamps.to_numpy(dtype='datetime64[ns]')
        idx = int(np.searchsorted(ts_array, np.datetime64(target), side='left'))
        self.set_start_step(idx)


class BESS:
    def __init__(self, config=None, sim: Simulation = None):
        self.sim     = sim
        self.Pmax    = config.get('Pmax', 5.0)
        self.Emax    = config.get('Emax', 10.0)
        self.η       = config.get('η', 0.95)
        self.β       = config.get('β', 0.999)
        self.DoD     = config.get('DoD', 0.8)
        self.soc_min = config.get('soc_min', 0.1)
        self.soc_max = config.get('soc_max', 0.9)
        self.capex = config.get('capex', 100.0)
        self.ncycles = max(1.0, float(config.get('ncycles', 1.0)))
        self.penalty_coef = config.get('penalty_coef', 0.0)
        usable_energy = max(self.Emax * self.DoD, 1e-6)
        self._deg_cost_per_kwh = self.capex / (self.ncycles * usable_energy)

        self.soc0    = config.get('soc_init', 0.5)
        self.soh0    = config.get('soh_init', 1.0)

        # Default power-vs-SoC curves based on 16S LFP pack (EVE LF105-L datasheet)
        default_soc_knots = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90], dtype=np.float32)
        default_charge_curve = np.array([3.1, 4.7, 5.3, 5.3, 5.3, 5.3, 5.3, 4.5, 3.2], dtype=np.float32)
        default_discharge_curve = np.array([4.1, 5.2, 5.3, 5.3, 5.3, 5.3, 5.1, 4.5, 3.8], dtype=np.float32)
        curve_cfg = config.get('soc_power_curve', {}) if isinstance(config, dict) else {}
        self._soc_knots = self._load_curve(curve_cfg.get('soc'), default_soc_knots)
        self._pmax_charge_curve = self._load_curve(curve_cfg.get('charge_kw'), default_charge_curve)
        self._pmax_discharge_curve = self._load_curve(curve_cfg.get('discharge_kw'), default_discharge_curve)
        if self._pmax_charge_curve.shape != self._soc_knots.shape or self._pmax_discharge_curve.shape != self._soc_knots.shape:
            self._soc_knots = default_soc_knots.copy()
            self._pmax_charge_curve = default_charge_curve.copy()
            self._pmax_discharge_curve = default_discharge_curve.copy()

        # Initialize state variables
        self.soc = self.soc0
        self.soh = self.soh0
        self._pbess = 0.0
        self._pbess_c = 0.0
        self._pbess_d = 0.0
        self._costdeg = 0.0
        self._penalty = 0.0

        return

    @staticmethod
    def _load_curve(values, default):
        if values is None:
            return default.copy()
        array = np.array(values, dtype=np.float32)
        if array.ndim != 1 or array.size != default.size:
            return default.copy()
        return array

    def _pmax_charge(self, soc):
        soc_clamped = float(np.clip(soc, self.soc_min, self.soc_max))
        limit = float(np.interp(soc_clamped, self._soc_knots, self._pmax_charge_curve))
        return max(0.0, min(limit, self.Pmax))

    def _pmax_discharge(self, soc):
        soc_clamped = float(np.clip(soc, self.soc_min, self.soc_max))
        limit = float(np.interp(soc_clamped, self._soc_knots, self._pmax_discharge_curve))
        return max(0.0, min(limit, self.Pmax))

    def reset(self, soc_init=None):
        self.soc0 = soc_init if soc_init is not None else self.soc0
        self.soc = self.soc0
        self.soh = self.soh0
        self._pbess = 0.0
        self._pbess_c = 0.0
        self._pbess_d = 0.0
        self._costdeg = 0.0
        self._penalty = 0.0
        return self.get_obs()

    def step(self, action):
        pmax_charge = self._pmax_charge(self.soc)
        pmax_discharge = self._pmax_discharge(self.soc)
        p_cmd = float(np.clip(action, -pmax_discharge, pmax_charge))
        p_charge = max(p_cmd, 0.0)
        p_discharge = max(-p_cmd, 0.0)

        # dt = max(self.sim.dt, 1e-6)
        energy_prev = self.soc * self.Emax
        emin = self.soc_min * self.Emax
        emax = self.soc_max * self.Emax

        delta_cmd = self.sim.dt * (p_charge * self.η - p_discharge / max(self.η, 1e-6))
        retention = float(np.clip(self.β, 0.0, 1.0))
        energy_after_leak = retention * energy_prev
        energy_unclamped = energy_after_leak + delta_cmd
        energy_next = float(np.clip(energy_unclamped, emin, emax))

        delta_actual = energy_next - energy_prev
        self.soc = energy_next / self.Emax if self.Emax > 0 else self.soc
        self._pbess = delta_actual / self.sim.dt
        self._pbess_c = max(self._pbess, 0.0)
        self._pbess_d = max(-self._pbess, 0.0)
        throughput = abs(delta_actual)

        deg = 1.0 - self.β
        self.soh = max(0.0, self.soh - deg * throughput / max(self.Emax * self.DoD, 1e-6))
        self._costdeg = throughput * self._deg_cost_per_kwh
        self._penalty = 0.0

        leakage_loss = energy_next - energy_after_leak

        info = {
            'p_cmd': p_cmd,
            'p_used': self._pbess,
            'pbess_charge_kw': self._pbess_c,
            'pbess_discharge_kw': self._pbess_d,
            'soc': self.soc,
            'soh': self.soh,
            'throughput_kWh': throughput,
            'leakage_kWh': leakage_loss,
        }

        return self._pbess, self._costdeg, self._penalty, info

    def get_obs(self):
        return np.array([self.soc, self.soh], dtype=np.float32)


class EV:
    def __init__(self, config=None, sim: Simulation = None):
        self.sim = sim
        self.Pmax_c = config.get('Pmax_c', 7.0)
        self.Pmax_d = config.get('Pmax_d', 7.0)
        self.Emax = config.get('Emax', 50.0)
        self.η    = config.get('η', 0.9)
        self.β    = config.get('β', 0.999)
        self.DoD  = config.get('DoD', 0.8)
        self.soc_min = config.get('soc_min', 0.0)
        self.soc_max = config.get('soc_max', 1.0)
        self.soc0 = config.get('soc_init', 0.5)
        self.soh0 = config.get('soh_init', 1.0)
        self.capex = config.get('capex', 0.0)
        self.ncycles = max(1.0, float(config.get('ncycles', 1.0)))
        self.penalty_coef = config.get('penalty_coef', 0.0)
        self.penalty_departure = config.get('penalty_departure', 0.0)
        self.column = config.get('column', 'ev_status')
        self.status_column = config.get('status_column', self.column)
        self.soc_column = config.get('soc_column', self.column)

        self.soc = self.soc0
        self.soh = self.soh0
        self.connected = False
        self._pev = 0.0
        self._pev_c = 0.0
        self._pev_d = 0.0
        self._costdeg = 0.0
        self._penalty = 0.0
        self._pending_absence_reset = False
        usable_energy = max(self.Emax * self.DoD, 1e-6)
        self._deg_cost_per_kwh = self.capex / (self.ncycles * usable_energy)

    def reset(self, soc_init=None):
        self.connected = False
        if soc_init is not None:
            self.soc0 = soc_init
        self.soc = self.soc0
        self.soh = self.soh0
        self._pev = 0.0
        self._pev_c = 0.0
        self._pev_d = 0.0
        self._costdeg = 0.0
        self._penalty = 0.0
        self._pending_absence_reset = False



    def step(self, action):
        status_signal = self.sim.get_value(self.status_column, default=0.0)
        soc_signal = self.sim.get_value(self.soc_column, default=self.soc)
    
        self._pev = 0.0
        self._pev_c = 0.0
        self._pev_d = 0.0
        self._costdeg = 0.0
        self._penalty = 0.0

        previously_connected = self.connected
        self.connected = status_signal > 0
        just_arrived = False

        if self._pending_absence_reset and not self.connected:
            self.soc = 0.0
            self._pending_absence_reset = False

        if self.connected and not previously_connected:
            just_arrived = True
            self.soc = np.clip(status_signal, self.soc_min, self.soc_max)
            self.soh = self.soh0

        if not self.connected:
            if previously_connected:
                energy_deficit = max(0.0, 1.0 - float(self.soc))
                penalty_energy = energy_deficit * self.Emax
                coef = float(self.penalty_departure or 1.0)
                self._penalty = penalty_energy * coef
                self._pending_absence_reset = True
                event = 'departure'
            else:
                self.soc = 0.0
                event = 'absent'
            return self._build_return(0.0, status_signal, event)

        p_cmd = float(np.clip(action, -self.Pmax_d, self.Pmax_c))
        p_charge = max(p_cmd, 0.0)
        p_discharge = max(-p_cmd, 0.0)
        # dt = max(self.sim.dt, 1e-6)
        energy_prev = self.soc * self.Emax
        emin = self.soc_min * self.Emax
        emax = self.soc_max * self.Emax

        delta_cmd = self.sim.dt * (p_charge * self.η - p_discharge / max(self.η, 1e-6))
        energy_next = float(np.clip(energy_prev + delta_cmd, emin, emax))
        delta_actual = energy_next - energy_prev
        self.soc = energy_next / self.Emax if self.Emax > 0 else self.soc

        self._pev = delta_actual / self.sim.dt
        self._pev_c = max(self._pev, 0.0)
        self._pev_d = max(-self._pev, 0.0)

        deg = 1.0 - self.β
        throughput = abs(delta_actual)
        self.soh = max(0.0, self.soh - deg * throughput / max(self.Emax * self.DoD, 1e-6))
        self._costdeg = throughput * self._deg_cost_per_kwh
        self._penalty = 0.0

        event = 'arrival' if just_arrived else 'connected'
        return self._build_return(p_cmd, status_signal, event)

    def get_obs(self):
        return np.array([
            self.soc,
            self.soh,
            1.0 if self.connected else 0.0,
        ], dtype=np.float32)

    def _build_return(self, command, status, event):
        try:
            p_cmd_info = float(command)
        except (TypeError, ValueError):
            p_cmd_info = 0.0

        info = {
            'event': event,
            'status_signal': status,
            'connected': self.connected,
            'p_cmd': p_cmd_info,
            'p_used': self._pev,
            'pev_charge_kw': self._pev_c,
            'pev_discharge_kw': self._pev_d,
            'soc': self.soc,
            'soh': self.soh,
        }
        return self._pev, self._costdeg, self._penalty, info


class Grid:
    def __init__(self, config=None, sim: Simulation = None):
        self.sim = sim
        self.Pmax_import = config.get('Pmax_import', 10.0)
        self.Pmax_export = config.get('Pmax_export', 10.0)
        self.net_penalty = config.get('net_penalty', 10.0)
        self.net_penalty_quadratic = config.get('net_penalty_quadratic', 0.0)
        self.cost_type   = config.get('cost_type', 'flat')
        self.tariff_column = config.get('tariff_column')
        self._pgrid         = 0.0
        self._cost          = 0.0
        self._revenue       = 0.0
        self._penalty       = 0.0
        self.tariff         = 0.0

    def reset(self):
        self._pgrid         = 0.0
        self._cost          = 0.0
        self._revenue       = 0.0
        self._penalty       = 0.0
        self.tariff         = 0.0

    def step(self, action):
        p_cmd = float(action)
        p_used = p_cmd

        exceed_import = max(0.0, p_cmd - self.Pmax_import)
        exceed_export = max(0.0, -p_cmd - self.Pmax_export)
        penalty_power = exceed_import + exceed_export

        penalty_energy = penalty_power * self.sim.dt
        quad_penalty = (penalty_power ** 2) * self.sim.dt * self.net_penalty_quadratic
        self._penalty = self.net_penalty * penalty_energy + quad_penalty
        self._pgrid = p_used

        energy = p_used * self.sim.dt
        tariff = self._get_tariff_rate()
        self.tariff = tariff

        if energy >= 0:
            self._cost = tariff * energy
            self._revenue = 0.0
        else:
            self._cost = 0.0
            self._revenue = -tariff * energy  # energy negative -> revenue positive if tariff>0

        info = {
            'p_cmd': p_cmd,
            'p_used': p_used,
            'tariff': tariff,
            'energy_kWh': energy,
            'import_cost': self._cost,
            'export_revenue': self._revenue,
            'penalty_power': penalty_power,
        }

        return self._cost, self._revenue, self._penalty, info

    def get_obs(self):
        return np.array([
            self._pgrid,
            self.tariff,
        ], dtype=np.float32)

    def _get_tariff_rate(self):
        row = self.sim.dataframe.iloc[self.sim.current_step]
        column = self.tariff_column
        return float(row[column]) if column in row.index else 0.0


class PV:
    def __init__(self, config=None, sim: Simulation = None):
        self.sim = sim
        self.Pmax = config.get('Pmax', 5.0)
        self.column = config.get('column', 'produced_electricity_rate_W')
        self._ppv = 0.0
        self._curt = 0.0
        self._available_kw = 0.0

    def reset(self):
        self._ppv = 0.0
        self._curt = 0.0
        self._available_kw = 0.0

    def step(self, action):
        chi = float(np.clip(action, 0.0, 1.0))
        self._curt = chi
        available_kw = self._get_available_kw()
        self._available_kw = available_kw
        self._ppv = available_kw * (1.0 - self._curt)

        curtailed_kw = available_kw - self._ppv
        info = {
            'pv_available_kw': available_kw,
            'pv_used_kw': self._ppv,
            'pv_curtailed_kw': curtailed_kw,
            'curtailment': self._curt,
        }

        return self._ppv, info

    def get_obs(self):
        available_kw = self._get_available_kw()
        return np.array([
            available_kw / max(self.Pmax, 1e-6)#,
            #self._ppv / max(self.Pmax, 1e-6),
            #self._curt,
        ], dtype=np.float32)

    def _get_available_kw(self):
        return self.sim.get_value(self.column, default=0.0) / 1000.0


class Load:
    def __init__(self, config=None, sim: Simulation = None):
        self.sim = sim
        self.Pmax = config.get('Pmax', 10.0)
        self.column = config.get('column', 'electricity_demand_rate_W')
        self._pload = 0.0

    def reset(self):
        self._pload = 0.0

    def step(self):
        self._pload = self._get_demand_kw()
        return self._pload

    def get_obs(self):
        return np.array([self._pload], dtype=np.float32)

    def _get_demand_kw(self):
        return self.sim.get_value(self.column, default=0.0) / 1000.0


class SmartHomeEnv(gym.Env):

    def __init__(self, config=None, dataframe=None, render_mode=None, days=None, state_mask=None, start_date=None):
        super().__init__()

        self.config = config
        general_cfg = dict(config.get('general', {})) if config else {}
        if days is not None: general_cfg['simulation_days'] = days
        self.sim = Simulation(general_cfg, dataframe=dataframe)
        self.start_date = pd.to_datetime(start_date, dayfirst=True) if start_date is not None else None
        if self.start_date is not None:
            self.sim.set_start_date(self.start_date)

        # smart home components
        self.bess = BESS(self.config.get('BESS', {}), sim=self.sim)
        self.grid = Grid(self.config.get('Grid', {}), sim=self.sim)
        self.load = Load(self.config.get('Load', {}), sim=self.sim)
        self.pv   = PV(self.config.get('PV', {}), sim=self.sim)
        self.ev   = EV(self.config.get('EV', {}), sim=self.sim)
        

        self.state = None
        self.done = False

        action_low = np.array([
            -self.bess.Pmax,
            -self.ev.Pmax_d,
            0.0,
        ], dtype=np.float32)
        action_high = np.array([
            self.bess.Pmax,
            self.ev.Pmax_c,
            1.0,
        ], dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        climate_len = len(self.sim.climate_columns)
        time_len = 2 * len(["month", "day", "hour", "minute", "weekday"])
        bess_len = len(self.bess.get_obs())
        ev_len = len(self.ev.get_obs())
        pv_len = len(self.pv.get_obs())
        extra_terms = 3  # load_norm, tariff_norm, net_load_norm
        shortfall_terms = 1
        self._full_obs_len = climate_len + time_len + bess_len + ev_len + shortfall_terms + pv_len + extra_terms

        self._feature_names = (
            [f"climate_{col}" for col in self.sim.climate_columns] +
            ["month_sin","month_cos","day_sin","day_cos","hour_sin","hour_cos",
            "minute_sin","minute_cos","weekday_sin","weekday_cos"] +
            ["bess_soc","bess_soh"] +
            ["ev_soc","ev_soh","ev_connected"] +
            ["ev_shortfall"] +
            ["pv_available","pv_used"] + # ,"pv_curtailment"] +
            ["load","tariff","net_load"]#,"steps_left_ratio"]
        )
        self.state_mask = np.array(state_mask, dtype=bool) if state_mask is not None else None
        self.state_feature_labels = (
            [name for name, keep in zip(self._feature_names, self.state_mask)]
            if self.state_mask is not None else list(self._feature_names)
        )

        if state_mask is not None:
            mask_array = np.array(state_mask, dtype=bool)
            if mask_array.size != self._full_obs_len:
                raise ValueError("state_mask length does not match observation size")
            self.state_mask = mask_array
            obs_len = int(mask_array.sum())
        else:
            self.state_mask = None
            obs_len = self._full_obs_len
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_len,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.bess.reset()
        self.grid.reset()
        self.load.reset()
        self.pv.reset()
        self.ev.reset()
        self.done = False
        options = options or {}
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # action will be a vector with [Pbess (positive means charge and negative means discharge), Pev (same as battery), Pv curtailment (value between 0 and 1)]
        pbess, pev, pv_curtail = action
        _, _, _, bess_info = self.bess.step(pbess)
        _, _, _, ev_info = self.ev.step(pev)
        self.pv.step(pv_curtail)
        self.load.step()

        pgrid = self.bess._pbess + self.ev._pev + self.load._pload - self.pv._ppv
        ppv_used = self.pv._ppv
        ppv_curtailed = max(self.pv._available_kw - self.pv._ppv, 0.0)
        # Log the realized device powers so the replay reflects the physical saturation limits
        self.sim.record_action(
            self.bess._pbess,
            self.ev._pev,
            self.pv._curt,
            pgrid,
            ppv_used,
            ppv_curtailed,
            pbess_cmd=bess_info.get('p_cmd') if isinstance(bess_info, dict) else None,
            pev_cmd=ev_info.get('p_cmd') if isinstance(ev_info, dict) else None,
        )
        self.grid.step(pgrid)

        observation = self._get_obs()
        reward = -self.bess._costdeg - self.bess._penalty -\
                 self.ev._costdeg - self.ev._penalty -\
                 self.grid._cost + self.grid._revenue - self.grid._penalty
        reward = reward * self.sim.dt  # scale by timestep duration
        horizon_reached = self.sim.step()
        self.done = horizon_reached
        info = {
            'terminated': False,
            'truncated': horizon_reached,
        }

        return observation, reward, self.done, info

    def _get_obs(self):
        row = self.sim.dataframe.iloc[self.sim.current_step]
        pnorm = max(self.sim.pnorm, 1e-6)

        climate_feats = self.sim.get_climate_features(row)
        climate_raw = [float(row.get(col, 0.0)) for col in self.sim.climate_columns]

        timestamp = pd.to_datetime(row.get('timestamp'))
        time_feats = self._build_time_features(timestamp)

        bess_obs = self.bess.get_obs()
        ev_obs = self.ev.get_obs()
        ev_shortfall = max(0.0, 1.0 - ev_obs[0])
        pv_obs = self.pv.get_obs()
        pv_available_kw = self.pv._get_available_kw()
        #pv_used_kw = self.pv._ppv
        load_obs = self.load.get_obs()
        load_kw = float(load_obs[0])
        load_norm = load_kw / pnorm
        net_load_kw = load_kw - self.pv._ppv
        net_load_norm = net_load_kw / pnorm

        tariff_norm = self.grid.tariff / self.sim.tariff_base
        # steps_left_ratio = 1.0 - (self.sim.current_step / max(self.sim.num_steps, 1))
        steps_elapsed = max(self.sim.current_step - getattr(self.sim, 'start_step', 0), 0)
        steps_left_abs = max(self.sim.num_steps - steps_elapsed, 0)

        features = (
            climate_feats
            + time_feats
            + bess_obs.tolist()
            + ev_obs.tolist()
            + [ev_shortfall]
            + pv_obs.tolist()
            + [load_norm, tariff_norm, net_load_norm]#, steps_left_ratio]
        )

        unnormalized_features = (
            climate_raw
            + time_feats
            + bess_obs.tolist()
            + ev_obs.tolist()
            + [ev_shortfall]
            + [pv_available_kw] #, pv_used_kw, self.pv._curt]
            + [load_kw, self.grid.tariff, net_load_kw, float(steps_left_abs)]
        )

        raw_state = np.array(features, dtype=np.float32)
        unnormalized_state = np.array(unnormalized_features, dtype=np.float32)
        if self.state_mask is not None:
            masked_state = raw_state[self.state_mask]
        else:
            masked_state = raw_state

        self.sim.record_state(raw_state, masked_state, unnormalized_state)
        self.state = masked_state
        return self.state

    def _build_time_features(self, timestamp):
        if pd.isna(timestamp):
            return [0.0] * 10

        def encode_cycle(value, period):
            angle = 2 * np.pi * (value / period)
            return [np.sin(angle), np.cos(angle)]

        month_val = max(timestamp.month - 1, 0)
        day_val = max(timestamp.day - 1, 0)
        hour_val = timestamp.hour
        minute_val = timestamp.minute
        weekday_val = timestamp.weekday()
        days_in_month = max(timestamp.days_in_month, 1)

        features = []
        features += encode_cycle(month_val, 12)
        features += encode_cycle(day_val, days_in_month)
        features += encode_cycle(hour_val, 24)
        features += encode_cycle(minute_val, 60)
        features += encode_cycle(weekday_val, 7)
        return features

    def build_operation_dataframe(self):
        segment = self.sim.dataframe.iloc[
            self.sim.start_step:self.sim.start_step + self.sim.num_steps
        ].copy()
        segment['timestamp'] = pd.to_datetime(segment['timestamp'])
        base_df = segment.set_index('timestamp')

        actions_df = pd.DataFrame(self.sim.get_action_history())
        actions_df['timestamp'] = pd.to_datetime(actions_df['timestamp'])
        actions_df = actions_df.set_index('timestamp')

        unscaled_states = pd.DataFrame(self.sim.get_unnormalized_state_history())
        unscaled_states['timestamp'] = pd.to_datetime(unscaled_states['timestamp'])
        unscaled_states = unscaled_states.set_index('timestamp')
        unscaled_states = unscaled_states.groupby(level=0).last()

        climate_len = len(self.sim.climate_columns)
        time_len = 10
        bess_len = len(self.bess.get_obs())
        ev_len = len(self.ev.get_obs())
        idx_bess = climate_len + time_len
        idx_ev = idx_bess + bess_len
        idx_ev_status = idx_ev + 2
        idx_load = idx_ev + ev_len + 2

        state_components = unscaled_states['raw_state_unscaled'].apply(
            lambda arr_like: (
                (vec := np.asarray(arr_like, dtype=float)),
                pd.Series({
                    'soc_bess': vec[idx_bess],
                    'soc_ev': vec[idx_ev],
                    'ev_status': vec[idx_ev_status],
                    'load_kw': vec[idx_load],
                })
            )[1]
        )

        combined = actions_df.join(state_components, how='left')
        operation_metrics = pd.DataFrame({
            'Pgrid': combined['pgrid'],
            'Ppv': combined['ppv_used'],
            'Load': combined['load_kw'],
            'PBESS': combined['PBESS'],
            'EBESS': combined['soc_bess'] * self.bess.Emax,
            'Pev': combined['PEV'],
            'Eev': combined['soc_ev'] * self.ev.Emax,
            'ev_status': combined['ev_status'],
        })
        operation_metrics['PBESS_cmd'] = (
            combined['PBESS_cmd'] if 'PBESS_cmd' in combined else combined['PBESS']
        )
        operation_metrics['Pev_cmd'] = (
            combined['Pev_cmd'] if 'Pev_cmd' in combined else combined['PEV']
        )

        return base_df.join(operation_metrics, how='left', rsuffix='_env')
