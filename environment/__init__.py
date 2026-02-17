from datetime import datetime, timedelta
from gymnasium import spaces
import gymnasium as gym
import pandas as pd
import numpy as np


class Simulation:
    def __init__(self, start, days, parameters):
        self.timestep = parameters["timestep"]
        self.Δt = self.timestep / 60
        self.params = parameters
        self.end = start + timedelta(days=days)
        self.start = start
        self.step = start
        self.days = days
        return


class Weather:
    def __init__(self, parameters, df, simulation):
        self.parameters = parameters
        self.sim = simulation
        self.df = df
        return

    def _get_obs(self):
        row = self.df.loc[self.sim.step]
        obs = [
            np.clip(
                (row[col] - self.parameters[col]["min"]) /
                (self.parameters[col]["max"] - self.parameters[col]["min"]),
                0.0, 1.0
            )
            for col in self.df.columns
        ]
        return obs


class Grid:
    def __init__(self, parameters, df, simulation):
        self.sim = simulation
        self.df = df
        self.Pmax = parameters["Pmax_import"]
        self.Pmin = -parameters["Pmax_export"]
        self.penalty = parameters["net_penalty"]
        self.history = []

    def reset(self):
        self.history = []
        return

    def step(self, power):
        cost, penalty = power * self.df[self.sim.step] * self.sim.Δt, 0.0
        if power > self.Pmax:
            penalty = self.penalty * (power - self.Pmax) * self.sim.Δt
        if power < self.Pmin:
            penalty = self.penalty * (self.Pmin - power) * self.sim.Δt
        self.history.append((power, cost, penalty))
        return power, cost, penalty


class LoadEnv():
    def __init__(self, parameters, df, simulation):
        self.parameters = parameters
        self.sim = simulation
        self.history = []
        self.df = df
        return

    def reset(self):
        self.history = []
        return

    def step(self):
        Pload = self.df[self.sim.step] / 1000  # kW
        self.history.append(Pload)
        return Pload, 0

    def _get_obs(self):
        return self.df[self.sim.step]


class PVEnv():
    def __init__(self, parameters, df, simulation):
        self.parameters = parameters
        self.sim = simulation
        self.history = []
        self.df = df
        return

    def reset(self):
        self.history = []
        return

    def step(self, action):
        Ppv = self.df[self.sim.step] / 1000  # kW
        PPV = Ppv * (1 - action)

        # Teacher-like small curtailment penalty: 0.01 * χPV * Ppv * Δt
        pv_cost = 0.01 * action * Ppv * self.sim.Δt

        self.history.append((Ppv, PPV, action, pv_cost))
        return PPV, pv_cost

    def _get_obs(self):
        return self.df[self.sim.step]


class BatteryEnv():
    def __init__(self, parameters, SoC, simulation):
        self.sim = simulation
        self.Pmax = parameters["Pmax"]
        self.Emax = parameters["Emax"]
        self.DoD = parameters["DoD"]
        self.η = parameters["η"]
        self.β = parameters["β"]
        self.capex = parameters["capex"]
        self.soc0 = SoC
        self.soc = SoC
        self.E = self.soc * self.Emax
        self.ncycles = parameters["ncycles"]
        self.sat_penalty = parameters["sat_penalty"]
        self.ramp_penalty = parameters["ramp_penalty"]
        self.soc_power = parameters["soc_power_curve_pu"]
        self.history = []   #(command, action, soc, energy)
        return

    def reset(self):
        self.soc = self.soc0
        self.E = self.soc * self.Emax
        self.history = []
        return

    def step(self, action):
        idx = int(np.clip(np.searchsorted(self.soc_power["soc"], self.soc) - 1, 0, len(self.soc_power["soc"]) - 1))
        Pcmd = action * self.Pmax
        Emin = self.Emax * (1 - self.DoD)

        self.E = self.E * (1 - self.β * self.sim.Δt)

        if Pcmd >= 0:  # charge
            P = min(Pcmd, self.soc_power["charge_pu"][idx] * self.Pmax)
            P = min(P, (self.Emax - self.E) / (self.sim.Δt * self.η))
            self.E = min(self.E + P * self.sim.Δt * self.η, self.Emax)
        else:          # discharge
            P = max(Pcmd, -self.soc_power["discharge_pu"][idx] * self.Pmax)
            P = max(P, (Emin - self.E) * self.η / self.sim.Δt)
            self.E = max(self.E + P * self.sim.Δt / self.η, Emin)

        eps = 1e-3
        sat = max(0.0, abs(P - Pcmd) - eps)
        sat_penalty = sat * self.sat_penalty * self.sim.Δt

        self.soc = np.clip(self.E / self.Emax, 0.0, 1.0)
        self.history.append((Pcmd, P, self.soc, self.E))

        cost = (self.capex / (self.Emax * self.ncycles)) * abs(P) * self.sim.Δt + sat_penalty
        return P, cost


class EVEnv():
    def __init__(self, parameters, df, tariff, simulation):
        self.sim = simulation
        self.df = df  # expects columns: ev_conn, ev_arrival, ev_departure
        self.grid_tariff = tariff

        self.Pmax_c = parameters["Pmax_c"]
        self.Pmax_d = parameters["Pmax_d"]
        self.Emax = parameters["Emax"]
        self.DoD = parameters["DoD"]
        self.η = parameters["η"]
        self.β = parameters["β"]
        self.capex = parameters["capex"]
        self.ncycles = parameters["ncycles"]

        # Teacher-like SoC min penalty (SEV_min) and threshold
        self.penalty = parameters["penalty"]
        self.soc_min = parameters["soc_min"]
        self.soc_critical = parameters["soc_critical"]
        self.fast_tariff = parameters["fast_tariff"] # this a tariff multiplier to grid tariff during fast charging 

        # Soft departure penalty coefficient (for shortfall to ev_departure)
        self.departure_penalty = parameters["departure_penalty"]

        self.sat_penalty = parameters["sat_penalty"]
        self.soc_power = parameters["soc_power_curve_pu"]

        self.prev_conn = 0

        self.soc = 1.0
        self.E = self.soc * self.Emax
        self.status = "disconnected"

        self.history = []  #(command, P, soc, E, status)
        return

    def reset(self):
        # Teacher sets EEV[start] = Emax; observation masks EV SoC at start/arrival if desired
        self.soc = 1.0
        self.E = self.Emax
        self.status = "disconnected"
        self.prev_conn = 0
        self.history = []
        return

    def step(self, action):
        if action >= 0:
            command = action * self.Pmax_c
        else:
            command = action * self.Pmax_d

        cost = 0.0

        t = self.sim.step
        conn_t = int(self.df["ev_conn"].loc[t])

        connected_t = conn_t in (1, 2)
        connected_prev = int(self.prev_conn) in (1, 2)

        is_start = (t == self.sim.start)
        is_arrival = (not is_start) and connected_t and (not connected_prev)

        Emin = self.Emax * (1 - self.DoD)

        # Disconnected: HOLD energy (Teacher-like), P=0
        if conn_t == 0:
            P = 0.0
            self.status = "disconnected"

        # Start step: if connected, Teacher forces PEV=0 and does not apply trip jump at start
        elif is_start and connected_t:
            P = 0.0
            self.status = "arriving"

        # Arrival: apply trip consumption jump and force P=0
        elif is_arrival:
            P = 0.0
            trip = float(self.df["ev_arrival"].loc[t])
            E_trip = self.Emax * trip
            E_dep = float(self.E)  # HOLD: energia do departure anterior
            Ecrit = float(self.soc_critical) * self.Emax

            if E_trip <= E_dep:
                self.E = E_dep - E_trip

            else:
                E_leg = self.Emax - Ecrit
                if E_leg <= 1e-9:
                    # fallback
                    tariff_t = float(self.grid_tariff.loc[t])
                    deficit = E_trip - E_dep
                    cost += float(self.fast_tariff) * tariff_t * deficit
                    self.E = 0.0
                else:
                    E_pre = max(0.0, E_dep - Ecrit)
                    R = E_trip - E_pre  # > 0
                    n_fast = int(np.ceil(R / E_leg))
                    tariff_t = float(self.grid_tariff.loc[t])
                    cost += float(n_fast) * float(self.fast_tariff) * tariff_t * E_leg
                    rem_last = R - float(n_fast - 1) * E_leg
                    self.E = self.Emax - rem_last

            self.E = float(np.clip(self.E, 0.0, self.Emax))
            self.status = "arriving"


        # Departure step: force P=0, apply self-discharge if previously connected
        elif conn_t == 2:
            P = 0.0

            if connected_prev:
                self.E = self.E * (1 - self.β * self.sim.Δt)

            self.status = "departing"

        # Connected (normal): self-discharge + charge/discharge
        else:
            idx = int(np.clip(np.searchsorted(self.soc_power["soc"], self.soc) - 1, 0, len(self.soc_power["soc"]) - 1))

            self.E = self.E * (1 - self.β * self.sim.Δt)

            if command >= 0:
                P = min(command, self.soc_power["charge_pu"][idx] * self.Pmax_c)
                P = min(P, (self.Emax - self.E) / (self.sim.Δt * self.η))
                self.E = min(self.E + P * self.sim.Δt * self.η, self.Emax)
            else:
                P = max(command, -self.soc_power["discharge_pu"][idx] * self.Pmax_d)
                P = max(P, (Emin - self.E) * self.η / self.sim.Δt)
                self.E = max(self.E + P * self.sim.Δt / self.η, Emin)

            self.status = "connected"

        self.soc = float(np.clip(self.E / self.Emax, 0.0, 1.0))

        # Saturation penalty only when controllable (connected)
        if self.status == "connected":
            eps = 1e-3
            sat = max(0.0, abs(P - command) - eps)
            cost += sat * self.sat_penalty * self.sim.Δt

        # Teacher-like SoC min shortfall penalty when connected (including departure)
        if connected_t:
            sev = max(0.0, self.Emax * self.soc_min - self.E)
            cost += self.penalty * sev * self.sim.Δt

        # Degradation
        cost += (self.capex / (self.Emax * self.ncycles)) * abs(P) * self.sim.Δt

        self.history.append((command, P, self.soc, self.E, self.status))
        self.prev_conn = conn_t

        return P, cost


class SmartHomeEnv(gym.Env):
    def __init__(self, df, parameters, start, days, BESS_SoC, tariff, track_operation=True):
        super().__init__()

        self.track_operation = bool(track_operation)

        self.sim = Simulation(start, days, parameters["general"])

        self.bess = BatteryEnv(parameters["BESS"], BESS_SoC, simulation=self.sim)
        self.load = LoadEnv(parameters["Load"], df["electricity_demand_rate_W"], simulation=self.sim)
        self.pv = PVEnv(parameters["PV"], df["produced_electricity_rate_W"], simulation=self.sim)

        # EV now uses ev_conn, ev_arrival, ev_departure (Teacher-like)
        self.ev = EVEnv(parameters["EV"], df[["ev_conn", "ev_arrival", "ev_departure"]], df[tariff], simulation=self.sim)

        self.grid = Grid(parameters["Grid"], df[tariff], simulation=self.sim)

        weather_df = df[[
            "drybulb_C", "relhum_percent", "Global Horizontal Radiation",
            "dni_Wm2", "dhi_Wm2", "Wind Speed (m/s)", "wdir_deg"
        ]]
        self.weather = Weather(parameters["general"]["state_normalization"], weather_df, simulation=self.sim)

        self.Pnorm = parameters["general"]["Pnorm"]

        self.operation = pd.DataFrame(columns=[
            "bess_cmd", "ev_cmd", "pv_cmd",
            "PLoad", "PPV", "PBESS", "PEV", "PGrid",
            "EBESS", "EEV", "SoCBESS", "SoCEV",
            "χPV",
            "tariff", "reward",
            "energy_cost", "bess_cost", "ev_cost",
            "pv_cost",
            "grid_penalty"
        ])

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_sample = np.asarray(self._get_observation(), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_sample.shape,
            dtype=np.float32,
        )

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        PD, _ = self.load.step()
        PBESS, bess_cost = self.bess.step(action[0])
        PEV, ev_cost = self.ev.step(action[1])
        PPV, pv_cost = self.pv.step(action[2])

        PGrid, energy_cost, penalty = self.grid.step(PD + PBESS + PEV - PPV)

        reward = - (energy_cost + penalty + bess_cost + ev_cost + pv_cost)

        if self.track_operation:
            row = {
                "bess_cmd": float(action[0]),
                "ev_cmd": float(action[1]),
                "pv_cmd": float(action[2]),
                "PLoad": float(PD),
                "PPV": float(PPV),
                "PBESS": float(PBESS),
                "PEV": float(PEV),
                "PGrid": float(PGrid),
                "EBESS": float(self.bess.E),
                "EEV": float(self.ev.E),
                "SoCBESS": float(self.bess.soc),
                "SoCEV": float(self.ev.soc),
                "χPV": float(action[2]),
                "tariff": float(self.grid.df[self.sim.step]),
                "reward": float(reward),
                "energy_cost": float(energy_cost),
                "bess_cost": float(bess_cost),
                "ev_cost": float(ev_cost),
                "pv_cost": float(pv_cost),
                "grid_penalty": float(penalty),
            }
            self.operation.loc[self.sim.step] = row

        info = {
            "energy_cost": energy_cost,
            "penalty": penalty,
            "pv_cost": pv_cost,
            "pgrid": PGrid,
            "pbess": PBESS,
            "pev": PEV,
            "ppv": PPV,
            "timestep": self.sim.step,
        }

        next_step = self.sim.step + timedelta(minutes=self.sim.timestep)
        terminated = next_step >= self.sim.end
        truncated = False

        if not terminated:
            self.sim.step = next_step
            self.state = self._get_observation()

        return self.state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        if "start" in options:
            self.sim.start = options["start"]
            self.sim.end = self.sim.start + timedelta(days=self.sim.days)
            self.sim.step = self.sim.start
        else:
            self.sim.step = self.sim.start

        if "days" in options:
            self.sim.days = options["days"]
            self.sim.end = self.sim.start + timedelta(days=self.sim.days)

        if "bess_soc" in options:
            self.bess.soc0 = float(options["bess_soc"])

        self.bess.reset()
        self.load.reset()
        self.grid.reset()
        self.pv.reset()
        self.ev.reset()

        self.state = self._get_observation()
        return self.state, {}

    def close(self):
        self.load.history.clear()
        self.pv.history.clear()
        self.bess.history.clear()
        self.ev.history.clear()
        self.grid.history.clear()
        self.state = None
        return

    def _get_observation(self):
        observation = [
            np.sin(2 * np.pi * (self.sim.step.minute / 60.0)),
            np.cos(2 * np.pi * (self.sim.step.minute / 60.0)),
            np.sin(2 * np.pi * (self.sim.step.hour / 24.0)),
            np.cos(2 * np.pi * (self.sim.step.hour / 24.0)),
            np.sin(2 * np.pi * ((self.sim.step.day - 1) / 31.0)),
            np.cos(2 * np.pi * ((self.sim.step.day - 1) / 31.0)),
            np.sin(2 * np.pi * ((self.sim.step.month - 1) / 12.0)),
            np.cos(2 * np.pi * ((self.sim.step.month - 1) / 12.0)),
            np.sin(2 * np.pi * (self.sim.step.weekday() / 7.0)),
            np.cos(2 * np.pi * (self.sim.step.weekday() / 7.0)),
        ]

        conn_t = int(self.ev.df["ev_conn"].loc[self.sim.step])
        ev_connected = conn_t in (1, 2)

        # Observation masking (Teacher-like): hide SoC at start and arrival step
        if self.sim.step == self.sim.start:
            ev_soc_obs = 0.0
        else:
            is_arrival_obs = ev_connected and (int(self.ev.prev_conn) == 0)
            if (not ev_connected) or is_arrival_obs:
                ev_soc_obs = 0.0
            else:
                ev_soc_obs = self.ev.soc

        # Yes: also multiply by connected flag (extra safety)
        ev_soc_obs = float(ev_soc_obs) * int(ev_connected)

        power_obs = [
            (self.load.df[self.sim.step] / 1000) / self.Pnorm,
            (self.pv.df[self.sim.step] / 1000) / self.Pnorm,
            self.bess.soc,
            ev_soc_obs,
            1.0 if ev_connected else 0.0,
        ]

        tariff_obs = [self.grid.df[self.sim.step]]
        weather_obs = self.weather._get_obs()
        observations = observation + power_obs + tariff_obs + weather_obs

        return np.array(observations, dtype=np.float32)
