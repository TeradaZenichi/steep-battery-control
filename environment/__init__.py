from datetime import datetime, timedelta
from gymnasium import spaces
import gymnasium as gym
import pandas as pd
import numpy as np

class Simulation:
    def __init__(self, start, days, parameters):
        self.timestep   = parameters["timestep"]
        self.Δt         = self.timestep/60
        self.params     = parameters
        self.end        = start + timedelta(days=days)
        self.start      = start
        self.step       = start
        self.days       = days
        

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
            np.clip((row[col] - self.parameters[col]["min"]) / (self.parameters[col]["max"] - self.parameters[col]["min"]), 0.0, 1.0)
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
        Pload = self.df[self.sim.step]/1000  # kW
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
        Ppv = self.df[self.sim.step]/1000  # kW
        self.history.append((Ppv, Ppv * (1 - action)))
        reward = 0
        return Ppv * (1 - action), reward
    
    def _get_obs(self):
        return self.df[self.sim.step]



class BatteryEnv():
    def __init__(self, parameters, SoC, simulation):
        self.sim    = simulation
        self.Pmax   = parameters["Pmax"]
        self.Emax   = parameters["Emax"]
        self.DoD    = parameters["DoD"]
        self.η      = parameters["η"]
        self.β      = parameters["β"]
        self.capex  = parameters["capex"]
        self.soc0   = SoC
        self.soc    = SoC
        self.E      = self.soc * self.Emax
        self.ncycles        = parameters["ncycles"]
        self.ramp_penalty   = parameters["ramp_penalty"]
        self.soc_power      = parameters["soc_power_curve_pu"]
        self.history = []   #(command, action, soc, energy)
        
        return

    def reset(self):
        self.soc = self.soc0
        self.E   = self.soc * self.Emax
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

        self.soc = np.clip(self.E / self.Emax, 0.0, 1.0)
        self.history.append((Pcmd, P, self.soc, self.E))

        cost = (self.capex / (self.Emax * self.ncycles)) * abs(P) * self.sim.Δt
        return P, cost
    

class EVEnv():
    def __init__(self, parameters, df, simulation):
        self.sim = simulation
        self.df = df
        self.Pmax_c     = parameters["Pmax_c"]
        self.Pmax_d     = parameters["Pmax_d"]
        self.Emax       = parameters["Emax"]
        self.DoD        = parameters["DoD"]
        self.η          = parameters["η"]
        self.β          = parameters["β"]
        self.capex      = parameters["capex"]
        self.ncycles    = parameters["ncycles"]
        self.penalty    = parameters["departure_penalty"]
        self.soc_power  = parameters["soc_power_curve_pu"]
        self.soc        = 0.0
        self.E          = 0.0
        self.status     = "disconnected"
        
        self.departure_penalty  = parameters["departure_penalty"]
        self.soc_power_curve_pu = parameters["soc_power_curve_pu"]
        self.history = []  #(command, action, soc, energy, status)
        return

    def reset(self):
        self.soc = 0.0
        self.history = []
        self.E = 0.0        
        return 

    def step(self, action):
        if action >= 0:
            command = action * self.Pmax_c
        else:
            command = action * self.Pmax_d

        cost = 0.0
        status = self._get_status()
        Emin = self.Emax * (1 - self.DoD)

        if status == "disconnected":
            P, self.E, self.soc = 0.0, 0.0, 0.0

        elif status == "arriving":
            P = 0.0
            self.soc = float(np.clip(self.df[self.sim.step], 0.0, 1.0))
            self.E = self.soc * self.Emax
            self.status = "connected"

        elif status == "connected":
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
            self.soc = np.clip(self.E / self.Emax, 0.0, 1.0)

        elif status == "departing":
            P = 0.0
            idx = int(np.searchsorted(self.departure_penalty["thresholds"], self.soc, side="right"))
            idx = int(np.clip(idx, 0, len(self.departure_penalty["weights"]) - 1))
            cost = self.departure_penalty["weights"][idx] * (1 - self.soc)
            self.status = "disconnected"
            self.E, self.soc = 0.0, 0.0

        cost += (self.capex / (self.Emax * self.ncycles)) * abs(P) * self.sim.Δt
        self.history.append((command, P, self.soc, self.E, self.status))

        return P, cost

    def _get_status(self):
        status = self.df[self.sim.step]
        if status < 0.01 and self.status == "disconnected":
            return "disconnected"
        elif status  > 0.01 and self.status == "disconnected":
            return "arriving"
        elif status  > 0.01 and self.status == "arriving":
            return "connected"
        elif status  > 0.01 and self.status == "connected":
            return "connected"
        elif status < 0.01 and self.status == "connected":
            return "departing"
        elif status < 0.01 and self.status == "departing":
            return "disconnected"
            

class SmartHomeEnv(gym.Env):
    def __init__(self, df, parameters, start, days, BESS_SoC, tariff):
        super().__init__()
        self.sim = Simulation(start, days, parameters["general"])
        self.bess = BatteryEnv(parameters["BESS"], BESS_SoC, simulation = self.sim)
        self.load = LoadEnv(parameters["Load"], df["electricity_demand_rate_W"], simulation = self.sim)
        self.pv = PVEnv(parameters["PV"], df["produced_electricity_rate_W"], simulation = self.sim)
        self.ev = EVEnv(parameters["EV"], df["ev_status"], simulation = self.sim)
        self.grid = Grid(parameters["Grid"], df[tariff], simulation = self.sim)
    
        weather_df = df[["drybulb_C", "relhum_percent", "Global Horizontal Radiation", 
                           "dni_Wm2", "dhi_Wm2", "Wind Speed (m/s)", "wdir_deg"]]
        self.weather = Weather(parameters["general"]["state_normalization"], weather_df, simulation = self.sim)
        self.Pnorm = parameters["general"]["Pnorm"]
        
        # Use datetime index for operations to avoid storing timestamp as a column
        self.operation = pd.DataFrame(columns=[
            "bess_cmd", "ev_cmd", "pv_cmd", "PLoad", "PPV", "PBESS", "PEV", "PGrid",
            "EBESS", "EEV", "SoCBESS", "SoCEV", "χPV",
            "tariff", "reward", "energy_cost", "bess_cost", "ev_cost", "grid_penalty"
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

        PD, _               = self.load.step()
        PBESS, bess_cost    = self.bess.step(action[0])
        PEV, ev_cost        = self.ev.step(action[1])
        PPV, _              = self.pv.step(action[2])

        PGrid, energy_cost, penalty = self.grid.step(PD + PBESS + PEV - PPV)
        reward = - (energy_cost + penalty + bess_cost + ev_cost)

        self.operation.loc[self.sim.step] = {
            "bess_cmd": action[0],
            "ev_cmd": action[1],
            "pv_cmd": action[2],
            "PLoad": PD,
            "PPV": PPV,
            "PBESS": PBESS,
            "PEV": PEV,
            "PGrid": PGrid,
            "EBESS": self.bess.E,
            "EEV": self.ev.E,
            "SoCBESS": self.bess.soc,
            "SoCEV": self.ev.soc,
            "χPV": action[2],
            "tariff": self.grid.df[self.sim.step],
            "reward": reward,
            "energy_cost": energy_cost,
            "bess_cost": bess_cost,
            "ev_cost": ev_cost,
            "grid_penalty": penalty,
        }

        info = {
            "energy_cost": energy_cost,
            "penalty": penalty,
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

        power_obs = [
            (self.load.df[self.sim.step] / 1000) / self.Pnorm,
            (self.pv.df[self.sim.step] / 1000) / self.Pnorm,
            self.bess.soc,
            self.ev.soc,
            1.0 if self.ev.df[self.sim.step] > 0.01 else 0.0,
        ]

        tariff_obs  = [self.grid.df[self.sim.step]]
        weather_obs = self.weather._get_obs()
        observations = observation + power_obs + tariff_obs + weather_obs

        return np.array(observations, dtype=np.float32)