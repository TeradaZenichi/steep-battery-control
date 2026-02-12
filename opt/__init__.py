from datetime import datetime, timedelta
import pyomo.environ as pyo
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

    def _get_obs(self, t):
        row = self.df.loc[t]
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

    def cost(self, t):
        return self.df[t]


class Load():
    def __init__(self, parameters, df, simulation):
        self.parameters = parameters
        self.sim = simulation
        self.history = []
        self.df = df
        return

    def power(self, t):
        return self.df[t]/1000  # kW


class PV():
    def __init__(self, parameters, df, simulation):
        self.parameters = parameters
        self.sim = simulation
        self.history = []
        self.df = df
        return

    def power(self, t):
        return self.df[t]/1000  # kW


class Battery():
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


class EV():
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


class Teacher:
    def __init__(self, df, parameters, start, days, BESS_SoC, tariff):
        self.sim    = Simulation(start, days, parameters["general"])
        self.bess   = Battery(parameters["BESS"], BESS_SoC, simulation = self.sim)
        self.load   = Load(parameters["Load"], df["electricity_demand_rate_W"], simulation = self.sim)
        self.pv     = PV(parameters["PV"], df["produced_electricity_rate_W"], simulation = self.sim)
        self.ev     = EV(parameters["EV"], df["ev_status"], simulation = self.sim)
        self.grid   = Grid(parameters["Grid"], df[tariff], simulation = self.sim)

        weather_df  = df[["drybulb_C", "relhum_percent", "Global Horizontal Radiation",
                          "dni_Wm2", "dhi_Wm2", "Wind Speed (m/s)", "wdir_deg"]]
        self.weather = Weather(parameters["general"]["state_normalization"], weather_df, simulation = self.sim)
        self.Pnorm = parameters["general"]["Pnorm"]

        # Use datetime index for operations to avoid storing timestamp as a column
        self.operation = pd.DataFrame(columns=[
            "bess_cmd", "ev_cmd", "pv_cmd", "PLoad", "PPV", "PBESS", "PEV", "PGrid",
            "tariff", "reward", "energy_cost", "bess_cost", "ev_cost", "grid_penalty"
        ])

        self.Ωt = [start + timedelta(minutes=i*self.sim.timestep) for i in range(int((self.sim.end - start).total_seconds() / 60 / self.sim.timestep))]


    def build(self):
        m = pyo.ConcreteModel()
        m.Ωt = pyo.Set(initialize=self.Ωt, ordered=True)

        # Variables
        m.PGrid     = pyo.Var(m.Ωt, bounds=(self.grid.Pmin, self.grid.Pmax))
        m.PBESS     = pyo.Var(m.Ωt, bounds=(-self.bess.Pmax, self.bess.Pmax))
        m.PBESS_c   = pyo.Var(m.Ωt, bounds=(0, self.bess.Pmax))
        m.PBESS_d   = pyo.Var(m.Ωt, bounds=(0, self.bess.Pmax))
        m.EBESS     = pyo.Var(m.Ωt, bounds=(self.bess.Emax*(1 - self.bess.DoD), self.bess.Emax))
        m.γBESS_c   = pyo.Var(m.Ωt, within=pyo.Binary)
        m.γBESS_d   = pyo.Var(m.Ωt, within=pyo.Binary)
        m.PEV       = pyo.Var(m.Ωt, bounds=(-self.ev.Pmax_d, self.ev.Pmax_c))
        m.PEV_c     = pyo.Var(m.Ωt, bounds=(0, self.ev.Pmax_c))
        m.PEV_d     = pyo.Var(m.Ωt, bounds=(0, self.ev.Pmax_d))
        m.EEV       = pyo.Var(m.Ωt, bounds=(0, self.ev.Emax))
        m.γEV_c     = pyo.Var(m.Ωt, within=pyo.Binary)
        m.γEV_d     = pyo.Var(m.Ωt, within=pyo.Binary)
        m.χPV       = pyo.Var(m.Ωt, bounds=(0, 1))

        for t in self.Ωt:
            if self.pv.power(t) <= 1e-9:
                m.χPV[t].fix(0.0)

        # Objective Function
        def objective_rule(model):
            energy_cost = sum(self.grid.cost(t) * model.PGrid[t] * self.sim.Δt for t in model.Ωt)
            bess_degradation = sum((self.bess.capex / self.bess.ncycles) * ((model.PBESS_d[t] + model.PBESS_c[t]) * self.sim.Δt) / self.bess.Emax for t in model.Ωt)
            ev_degradation = sum((self.ev.capex / self.ev.ncycles) * ((model.PEV_d[t] + model.PEV_c[t]) * self.sim.Δt) / self.ev.Emax for t in model.Ωt)
            pv_penalty = sum(0.01 * model.χPV[t] * self.pv.power(t) * self.sim.Δt for t in model.Ωt)
            return energy_cost + bess_degradation + ev_degradation + pv_penalty
        m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Power balance
        def power_balance_rule(model, t):
            PD = self.load.power(t)
            PPV = self.pv.power(t) * (1 - model.χPV[t])
            return PD + model.PBESS[t] + model.PEV[t] == model.PGrid[t] + PPV
        m.power_balance = pyo.Constraint(m.Ωt, rule=power_balance_rule)

        # Battery energy update
        def bess_energy_update_rule(model, t):
            if t == self.sim.start:
                return model.EBESS[t] == self.bess.soc0 * self.bess.Emax + (model.PBESS_c[t] * self.bess.η - model.PBESS_d[t] / self.bess.η) * self.sim.Δt - self.bess.E * self.bess.β * self.sim.Δt
            else:
                t_prev = model.Ωt.prev(t)
                return model.EBESS[t] == model.EBESS[t_prev] + (model.PBESS_c[t] * self.bess.η - model.PBESS_d[t] / self.bess.η) * self.sim.Δt - model.EBESS[t_prev] * self.bess.β * self.sim.Δt
        m.bess_energy_update = pyo.Constraint(m.Ωt, rule=bess_energy_update_rule)

        # Battery charge/discharge exclusivity
        def bess_charge_discharge_rule(model, t):
            return model.γBESS_c[t] + model.γBESS_d[t] <= 1
        m.bess_charge_discharge = pyo.Constraint(m.Ωt, rule=bess_charge_discharge_rule)

        def bess_power(model, t):
            return model.PBESS[t] == model.PBESS_c[t] - model.PBESS_d[t]
        m.bess_power_def = pyo.Constraint(m.Ωt, rule=bess_power)

        def bess_charge_power_limit_rule(model, t):
            return model.PBESS_c[t] <= model.γBESS_c[t] * self.bess.Pmax
        m.bess_charge_power_limit = pyo.Constraint(m.Ωt, rule=bess_charge_power_limit_rule)

        def bess_discharge_power_limit_rule(model, t):
            return model.PBESS_d[t] <= model.γBESS_d[t] * self.bess.Pmax
        m.bess_discharge_power_limit = pyo.Constraint(m.Ωt, rule=bess_discharge_power_limit_rule)

        # EV energy update
        def ev_energy_update_rule(model, t):
            if t == self.sim.start:
                if self.ev.df[t] > 0.01:
                    return model.EEV[t] == self.ev.Emax * self.ev.df[t]
                return m.EEV[t] == 0
            elif self.ev.df[t] > 0.01 and self.ev.df[t - timedelta(minutes=self.sim.timestep)] < 0.01:
                return m.PEV[t] == 0
            elif self.ev.df[t] > 0.01:
                t_prev = model.Ωt.prev(t)
                return model.EEV[t] == model.EEV[t_prev] + (model.PEV_c[t] * self.ev.η - model.PEV_d[t] / self.ev.η) * self.sim.Δt - model.EEV[t_prev] * self.ev.β * self.sim.Δt
            else:
                return model.EEV[t] == 0
        m.ev_energy_update = pyo.Constraint(m.Ωt, rule=ev_energy_update_rule)

        def ev_soc_rule(model, t):
            if t == self.sim.start:
                if self.ev.df[t] > 0.01:
                    return model.EEV[t] == self.ev.Emax * self.ev.df[t]
                return pyo.Constraint.Skip
            elif self.ev.df[t] > 0.01 and self.ev.df[t - timedelta(minutes=self.sim.timestep)] < 0.01:
                return model.EEV[t] == self.ev.Emax * self.ev.df[t]
            elif self.ev.df[t] < 0.01 and self.ev.df[t - timedelta(minutes=self.sim.timestep)] > 0.01:
                return model.EEV[t] == 0
            elif t == self.sim.end:
                return pyo.Constraint.Skip
            elif self.ev.df[t] > 0.01 and self.ev.df[t + timedelta(minutes=self.sim.timestep)] < 0.01:
                return model.EEV[t] == self.ev.Emax
            else:
                return pyo.Constraint.Skip
        m.ev_soc_constraint = pyo.Constraint(m.Ωt, rule=ev_soc_rule)

        # Enforce DoD floor only when connected
        def ev_dod_rule(model, t):
            if self.ev.df[t] > 0.01:
                return model.EEV[t] >= self.ev.Emax * (1 - self.ev.DoD)
            return pyo.Constraint.Skip
        m.ev_dod_constraint = pyo.Constraint(m.Ωt, rule=ev_dod_rule)

        def power_arrival_departure_rule(model, t):
            if t == self.sim.start:
                if self.ev.df[t] > 0.01:
                    return model.PEV[t] == 0
                return pyo.Constraint.Skip
            elif self.ev.df[t] > 0.01 and self.ev.df[t - timedelta(minutes=self.sim.timestep)] < 0.01:
                return model.PEV[t] == 0
            elif self.ev.df[t] < 0.01 and self.ev.df[t - timedelta(minutes=self.sim.timestep)] > 0.01:
                return model.PEV[t] == 0
            elif self.ev.df[t] < 0.01:
                return model.PEV[t] == 0
            else:
                return pyo.Constraint.Skip
        m.power_arrival_departure_constraint = pyo.Constraint(m.Ωt, rule=power_arrival_departure_rule)

        # EV charge/discharge exclusivity
        def ev_charge_discharge_rule(model, t):
            return model.γEV_c[t] + model.γEV_d[t] <= 1
        m.ev_charge_discharge = pyo.Constraint(m.Ωt, rule=ev_charge_discharge_rule)

        def ev_power(model, t):
            return model.PEV[t] == model.PEV_c[t] - model.PEV_d[t]
        m.ev_power_def = pyo.Constraint(m.Ωt, rule=ev_power)

        def ev_charge_power_limit_rule(model, t):
            return model.PEV_c[t] <= model.γEV_c[t] * self.ev.Pmax_c
        m.ev_charge_power_limit = pyo.Constraint(m.Ωt, rule=ev_charge_power_limit_rule)

        def ev_discharge_power_limit_rule(model, t):
            return model.PEV_d[t] <= model.γEV_d[t] * self.ev.Pmax_d
        m.ev_discharge_power_limit = pyo.Constraint(m.Ωt, rule=ev_discharge_power_limit_rule)

        self.model = m

        return


    def solve(self, solver="gurobi"):
        solve = pyo.SolverFactory(solver)
        self.results = solve.solve(self.model, tee=True)
        return  self.results


    def get_operation(self):
        self.operation = pd.DataFrame(columns=[
            "PLoad", "PPV", "PBESS", "PEV", "PGrid", "EBESS", "EEV", "SoCBESS", "SoCEV", "χPV", "tariff", "energy_cost", "bess_cost", "ev_cost"
        ])

        for t in self.model.Ωt:
            self.operation.loc[t] = {
                "PLoad": self.load.power(t),
                "PPV": self.pv.power(t) * (1 - pyo.value(self.model.χPV[t])),
                "PBESS": pyo.value(self.model.PBESS[t]),
                "PEV": pyo.value(self.model.PEV[t]),
                "PGrid": pyo.value(self.model.PGrid[t]),
                "EBESS": pyo.value(self.model.EBESS[t]),
                "EEV": pyo.value(self.model.EEV[t]),
                "SoCBESS": pyo.value(self.model.EBESS[t]) / self.bess.Emax,
                "SoCEV": pyo.value(self.model.EEV[t]) / self.ev.Emax,
                "χPV": pyo.value(self.model.χPV[t]),
                "tariff": self.grid.df[t],
                "energy_cost": self.grid.cost(t) * pyo.value(self.model.PGrid[t]) * self.sim.Δt,
                "bess_cost" : (self.bess.capex / self.bess.ncycles) * (pyo.value(self.model.PBESS_d[t] + self.model.PBESS_c[t]) * self.sim.Δt) / self.bess.Emax,
                "ev_cost" : (self.ev.capex / self.ev.ncycles) * (pyo.value(self.model.PEV_d[t] + self.model.PEV_c[t]) * self.sim.Δt) / self.ev.Emax,
            }
        return self.operation


    def get_obs(self, t):
        # Same time encoding as SmartHomeEnv._get_observation()
        observation = [
            np.sin(2 * np.pi * (t.minute / 60.0)),
            np.cos(2 * np.pi * (t.minute / 60.0)),
            np.sin(2 * np.pi * (t.hour / 24.0)),
            np.cos(2 * np.pi * (t.hour / 24.0)),
            np.sin(2 * np.pi * ((t.day - 1) / 31.0)),
            np.cos(2 * np.pi * ((t.day - 1) / 31.0)),
            np.sin(2 * np.pi * ((t.month - 1) / 12.0)),
            np.cos(2 * np.pi * ((t.month - 1) / 12.0)),
            np.sin(2 * np.pi * (t.weekday() / 7.0)),
            np.cos(2 * np.pi * (t.weekday() / 7.0)),
        ]

        # State-at-decision (env-style): SoC BEFORE action at t
        if t == self.sim.start:
            bess_soc_state = float(self.bess.soc0)
            ev_prev_connected = False
            ev_soc_state = 0.0
        else:
            t_prev = t - timedelta(minutes=self.sim.timestep)
            bess_soc_state = float(self.operation.loc[t_prev, "SoCBESS"])
            ev_prev_connected = bool(self.ev.df[t_prev] > 0.01)

            ev_connected = bool(self.ev.df[t] > 0.01)
            if not ev_connected:
                ev_soc_state = 0.0
            elif not ev_prev_connected:
                # "arriving" step: env observation still has ev.soc = 0.0
                ev_soc_state = 0.0
            else:
                ev_soc_state = float(self.operation.loc[t_prev, "SoCEV"])

        ev_connected = bool(self.ev.df[t] > 0.01)

        power_obs = [
            self.load.power(t)/self.Pnorm,
            self.pv.power(t)/self.Pnorm,
            bess_soc_state,
            ev_soc_state,
            1.0 if ev_connected else 0.0,
        ]

        tariff_obs  = [self.grid.df[t]]
        weather_obs = self.weather._get_obs(t)
        observations = observation + power_obs + tariff_obs + weather_obs
        return np.array(observations, dtype=np.float32)


    def get_actions(self, t):
        # Actions WITHOUT projection: only normalization to match SmartHomeEnv action space
        PBESS_des = float(self.operation.loc[t, "PBESS"])
        PEV_des   = float(self.operation.loc[t, "PEV"])
        χPV       = float(self.operation.loc[t, "χPV"])

        # BESS normalized command (symmetric limits)
        bess_cmd = float(np.clip(PBESS_des / self.bess.Pmax, -1.0, 1.0))

        # EV normalized command (asymmetric limits) + arrival/disconnected rules consistent with env
        ev_connected = bool(self.ev.df[t] > 0.01)

        if not ev_connected:
            ev_cmd = 0.0
        else:
            if t == self.sim.start:
                ev_prev_connected = False
            else:
                t_prev = t - timedelta(minutes=self.sim.timestep)
                ev_prev_connected = bool(self.ev.df[t_prev] > 0.01)

            # Arriving step: env forces P=0
            if not ev_prev_connected:
                ev_cmd = 0.0
            else:
                if PEV_des >= 0.0:
                    ev_cmd = PEV_des / self.ev.Pmax_c
                else:
                    ev_cmd = PEV_des / self.ev.Pmax_d
                ev_cmd = float(np.clip(ev_cmd, -1.0, 1.0))

        χPV = float(np.clip(χPV, 0.0, 1.0))

        actions = [bess_cmd, ev_cmd, χPV]
        return actions


    def get_training_data(self):
        self.get_operation()
        X_train = np.asarray([self.get_obs(t) for t in self.model.Ωt], dtype=np.float32)
        y_train = np.asarray([self.get_actions(t) for t in self.model.Ωt], dtype=np.float32)
        return X_train, y_train
