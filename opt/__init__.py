from datetime import timedelta
import pyomo.environ as pyo
import pandas as pd
import numpy as np


def _storage_capex(parameters):
    if "capex_per_kwh" in parameters:
        return float(parameters["capex_per_kwh"]) * float(parameters["Emax"])
    return float(parameters["capex"])


class Simulation:
    def __init__(self, start, days, parameters):
        self.timestep = parameters["timestep"]
        self.Δt = self.timestep / 60  # hours
        self.start = start
        self.end = start + timedelta(days=days)
        self.days = days
        return


class Weather:
    def __init__(self, parameters, df):
        self.parameters = parameters
        self.df = df
        return

    def _get_obs(self, t):
        row = self.df.loc[t]
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
    def __init__(self, parameters, df):
        self.df = df
        self.Pmax = parameters["Pmax_import"]
        self.Pmin = -parameters["Pmax_export"]
        self.export_tariff_factor = parameters.get("export_tariff_factor", 1.0)
        return

    def cost(self, t):
        return self.df[t]


class Load:
    def __init__(self, df):
        self.df = df
        return

    def power(self, t):
        return self.df[t] / 1000  # kW


class PV:
    def __init__(self, df):
        self.df = df
        return

    def power(self, t):
        return self.df[t] / 1000  # kW


class Battery:
    def __init__(self, parameters, SoC):
        self.Pmax = parameters["Pmax"]
        self.Emax = parameters["Emax"]
        self.DoD = parameters["DoD"]
        self.η = parameters["η"]
        self.β = parameters["β"]
        self.capex = _storage_capex(parameters)
        self.ncycles = parameters["ncycles"]
        self.soc0 = SoC
        return


class EV:
    def __init__(self, parameters, df):
        self.df = df
        self.Pmax_c = parameters["Pmax_c"]
        self.Pmax_d = parameters["Pmax_d"]
        self.Emax = parameters["Emax"]
        self.DoD = parameters["DoD"]
        self.η = parameters["η"]
        self.β = parameters["β"]
        self.capex = _storage_capex(parameters)
        self.ncycles = parameters["ncycles"]
        self.penalty = parameters["penalty"]
        self.soc_min = parameters["soc_min"]
        return


class Teacher:
    def __init__(self, df, parameters, start, days, BESS_SoC, tariff):
        self.sim = Simulation(start, days, parameters["general"])

        self.bess = Battery(parameters["BESS"], BESS_SoC)
        self.load = Load(df["electricity_demand_rate_W"])
        self.pv = PV(df["produced_electricity_rate_W"])

        self.ev = EV(parameters["EV"], df[["ev_conn", "ev_arrival", "ev_departure"]])
        self.grid = Grid(parameters["Grid"], df[tariff])

        weather_df = df[
            ["drybulb_C", "relhum_percent", "Global Horizontal Radiation",
             "dni_Wm2", "dhi_Wm2", "Wind Speed (m/s)", "wdir_deg"]
        ]
        self.weather = Weather(parameters["general"]["state_normalization"], weather_df)

        self.Pnorm = parameters["general"]["Pnorm"]

        self.Ωt = [
            start + timedelta(minutes=i * self.sim.timestep)
            for i in range(int((self.sim.end - start).total_seconds() / 60 / self.sim.timestep))
        ]

        self.operation = pd.DataFrame()
        self.model = None
        self.results = None
        return

    def build(self):
        m = pyo.ConcreteModel()
        m.Ωt = pyo.Set(initialize=self.Ωt, ordered=True)

        # Variables
        m.PGrid = pyo.Var(m.Ωt, bounds=(self.grid.Pmin, self.grid.Pmax))
        m.PGrid_import = pyo.Var(m.Ωt, bounds=(0, self.grid.Pmax))
        m.PGrid_export = pyo.Var(m.Ωt, bounds=(0, -self.grid.Pmin))

        m.PBESS = pyo.Var(m.Ωt, bounds=(-self.bess.Pmax, self.bess.Pmax))
        m.PBESS_c = pyo.Var(m.Ωt, bounds=(0, self.bess.Pmax))
        m.PBESS_d = pyo.Var(m.Ωt, bounds=(0, self.bess.Pmax))
        m.EBESS = pyo.Var(m.Ωt, bounds=(self.bess.Emax * (1 - self.bess.DoD), self.bess.Emax))
        m.γBESS_c = pyo.Var(m.Ωt, within=pyo.Binary)
        m.γBESS_d = pyo.Var(m.Ωt, within=pyo.Binary)

        m.PEV = pyo.Var(m.Ωt, bounds=(-self.ev.Pmax_d, self.ev.Pmax_c))
        m.PEV_c = pyo.Var(m.Ωt, bounds=(0, self.ev.Pmax_c))
        m.PEV_d = pyo.Var(m.Ωt, bounds=(0, self.ev.Pmax_d))
        m.EEV = pyo.Var(m.Ωt, bounds=(0, self.ev.Emax))
        m.γEV_c = pyo.Var(m.Ωt, within=pyo.Binary)
        m.γEV_d = pyo.Var(m.Ωt, within=pyo.Binary)

        m.χPV = pyo.Var(m.Ωt, bounds=(0, 1))

        # EV SoC soft penalty helper
        m.SEV_min = pyo.Var(m.Ωt, bounds=(0, None))

        # Fix PV curtailment variable when PV=0
        for t in self.Ωt:
            if self.pv.power(t) <= 1e-9:
                m.χPV[t].fix(0.0)

        def _is_connected(v):
            return int(v) in (1, 2)

        # Objective
        def objective_rule(model):
            energy_cost = sum(
                self.grid.cost(t) *
                (model.PGrid_import[t] - self.grid.export_tariff_factor * model.PGrid_export[t]) *
                self.sim.Δt
                for t in model.Ωt
            )

            bess_degradation = sum(
                (self.bess.capex / self.bess.ncycles) *
                ((model.PBESS_d[t] + model.PBESS_c[t]) * self.sim.Δt) / self.bess.Emax
                for t in model.Ωt
            )

            ev_degradation = sum(
                (self.ev.capex / self.ev.ncycles) *
                ((model.PEV_d[t] + model.PEV_c[t]) * self.sim.Δt) / self.ev.Emax
                for t in model.Ωt
            )

            pv_penalty = sum(0.01 * model.χPV[t] * self.pv.power(t) * self.sim.Δt for t in model.Ωt)

            ev_penalty = self.ev.penalty * sum(model.SEV_min[t] * self.sim.Δt for t in model.Ωt)

            return energy_cost + bess_degradation + ev_degradation + pv_penalty + ev_penalty

        m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        def grid_import_export_rule(model, t):
            return model.PGrid[t] == model.PGrid_import[t] - model.PGrid_export[t]

        m.grid_import_export = pyo.Constraint(m.Ωt, rule=grid_import_export_rule)

        # Power balance
        def power_balance_rule(model, t):
            PD = self.load.power(t)
            PPV = self.pv.power(t) * (1 - model.χPV[t])
            return PD + model.PBESS[t] + model.PEV[t] == model.PGrid[t] + PPV

        m.power_balance = pyo.Constraint(m.Ωt, rule=power_balance_rule)

        # BESS dynamics
        def bess_energy_update_rule(model, t):
            if t == self.sim.start:
                return model.EBESS[t] == self.bess.soc0 * self.bess.Emax + \
                    (model.PBESS_c[t] * self.bess.η - model.PBESS_d[t] / self.bess.η) * self.sim.Δt - \
                    (self.bess.soc0 * self.bess.Emax) * self.bess.β * self.sim.Δt
            t_prev = model.Ωt.prev(t)
            return model.EBESS[t] == model.EBESS[t_prev] + \
                (model.PBESS_c[t] * self.bess.η - model.PBESS_d[t] / self.bess.η) * self.sim.Δt - \
                model.EBESS[t_prev] * self.bess.β * self.sim.Δt

        m.bess_energy_update = pyo.Constraint(m.Ωt, rule=bess_energy_update_rule)

        def bess_charge_discharge_rule(model, t):
            return model.γBESS_c[t] + model.γBESS_d[t] <= 1

        m.bess_charge_discharge = pyo.Constraint(m.Ωt, rule=bess_charge_discharge_rule)

        def bess_power_def_rule(model, t):
            return model.PBESS[t] == model.PBESS_c[t] - model.PBESS_d[t]

        m.bess_power_def = pyo.Constraint(m.Ωt, rule=bess_power_def_rule)

        def bess_charge_power_limit_rule(model, t):
            return model.PBESS_c[t] <= model.γBESS_c[t] * self.bess.Pmax

        m.bess_charge_power_limit = pyo.Constraint(m.Ωt, rule=bess_charge_power_limit_rule)

        def bess_discharge_power_limit_rule(model, t):
            return model.PBESS_d[t] <= model.γBESS_d[t] * self.bess.Pmax

        m.bess_discharge_power_limit = pyo.Constraint(m.Ωt, rule=bess_discharge_power_limit_rule)

        # EV start energy
        def ev_start_energy_rule(model, t):
            if t == self.sim.start:
                return model.EEV[t] == self.ev.Emax
            return pyo.Constraint.Skip

        m.ev_start_energy = pyo.Constraint(m.Ωt, rule=ev_start_energy_rule)

        # EV arrival: jump at first connected step after disconnection
        def ev_arrival_rule(model, t):
            if t == self.sim.start:
                return pyo.Constraint.Skip

            t_prev = model.Ωt.prev(t)
            conn_t = int(self.ev.df["ev_conn"].loc[t])
            conn_prev = int(self.ev.df["ev_conn"].loc[t_prev])

            if _is_connected(conn_t) and (not _is_connected(conn_prev)):
                return model.EEV[t] == model.EEV[t_prev] - self.ev.Emax * float(self.ev.df["ev_arrival"].loc[t])

            return pyo.Constraint.Skip

        m.ev_arrival = pyo.Constraint(m.Ωt, rule=ev_arrival_rule)

        # EV hold when disconnected
        def ev_hold_disconnected_rule(model, t):
            if t == self.sim.start:
                return pyo.Constraint.Skip

            t_prev = model.Ωt.prev(t)
            conn_t = int(self.ev.df["ev_conn"].loc[t])

            if conn_t == 0:
                return model.EEV[t] == model.EEV[t_prev]

            return pyo.Constraint.Skip

        m.ev_hold_disconnected = pyo.Constraint(m.Ωt, rule=ev_hold_disconnected_rule)

        # EV energy update when connected (excluding arrival)
        def ev_energy_update_rule(model, t):
            if t == self.sim.start:
                return pyo.Constraint.Skip

            t_prev = model.Ωt.prev(t)
            conn_t = int(self.ev.df["ev_conn"].loc[t])
            conn_prev = int(self.ev.df["ev_conn"].loc[t_prev])

            if _is_connected(conn_t) and _is_connected(conn_prev):
                return model.EEV[t] == model.EEV[t_prev] + \
                    (model.PEV_c[t] * self.ev.η - model.PEV_d[t] / self.ev.η) * self.sim.Δt - \
                    model.EEV[t_prev] * self.ev.β * self.sim.Δt

            return pyo.Constraint.Skip

        m.ev_energy_update = pyo.Constraint(m.Ωt, rule=ev_energy_update_rule)

        # EV DoD only when connected
        def ev_dod_rule(model, t):
            conn_t = int(self.ev.df["ev_conn"].loc[t])
            if _is_connected(conn_t):
                return model.EEV[t] >= self.ev.Emax * (1 - self.ev.DoD)
            return pyo.Constraint.Skip

        m.ev_dod_constraint = pyo.Constraint(m.Ωt, rule=ev_dod_rule)

        # Departure requirement at ev_conn == 2
        def ev_departure_rule(model, t):
            conn_t = int(self.ev.df["ev_conn"].loc[t])
            if conn_t == 2:
                return model.EEV[t] >= self.ev.Emax * float(self.ev.df["ev_departure"].loc[t])
            return pyo.Constraint.Skip

        m.ev_departure = pyo.Constraint(m.Ωt, rule=ev_departure_rule)

        # Force PEV=0 when disconnected, at arrival, and at departure step
        def power_arrival_departure_rule(model, t):
            conn_t = int(self.ev.df["ev_conn"].loc[t])

            if conn_t == 0 or conn_t == 2:
                return model.PEV[t] == 0

            if t == self.sim.start:
                return model.PEV[t] == 0

            t_prev = model.Ωt.prev(t)
            conn_prev = int(self.ev.df["ev_conn"].loc[t_prev])

            if _is_connected(conn_t) and (not _is_connected(conn_prev)):
                return model.PEV[t] == 0

            return pyo.Constraint.Skip

        m.power_arrival_departure_constraint = pyo.Constraint(m.Ωt, rule=power_arrival_departure_rule)

        # EV SoC soft min penalty (only when connected)
        def ev_soc_min_shortfall_rule(model, t):
            conn_t = int(self.ev.df["ev_conn"].loc[t])

            if conn_t in (1, 2):
                return model.SEV_min[t] >= self.ev.Emax * self.ev.soc_min - model.EEV[t]

            if conn_t == 0:
                return model.SEV_min[t] == 0

            return pyo.Constraint.Skip

        m.ev_soc_min_shortfall = pyo.Constraint(m.Ωt, rule=ev_soc_min_shortfall_rule)

        # EV charge/discharge exclusivity and decomposition
        def ev_charge_discharge_rule(model, t):
            return model.γEV_c[t] + model.γEV_d[t] <= 1

        m.ev_charge_discharge = pyo.Constraint(m.Ωt, rule=ev_charge_discharge_rule)

        def ev_power_def_rule(model, t):
            return model.PEV[t] == model.PEV_c[t] - model.PEV_d[t]

        m.ev_power_def = pyo.Constraint(m.Ωt, rule=ev_power_def_rule)

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
        return self.results

    def get_operation(self):
        self.operation = pd.DataFrame(columns=[
            "PLoad", "PPV", "PBESS", "PEV", "PGrid",
            "EBESS", "EEV", "SoCBESS", "SoCEV",
            "χPV", "tariff", "energy_cost", "bess_cost", "ev_cost"
        ])

        for t in self.model.Ωt:
            conn_t = int(self.ev.df["ev_conn"].loc[t])
            ev_connected = conn_t in (1, 2)
            if t == self.sim.start:
                ev_arrival = False
            else:
                t_prev = self.model.Ωt.prev(t)
                conn_prev = int(self.ev.df["ev_conn"].loc[t_prev])
                ev_arrival = ev_connected and (conn_prev not in (1, 2))

            ev_mask = 1 if ev_connected and (not ev_arrival) else 0
            self.operation.loc[t] = {
                "PLoad": self.load.power(t),
                "PPV": self.pv.power(t) * (1 - pyo.value(self.model.χPV[t])),
                "PBESS": pyo.value(self.model.PBESS[t]),
                "PEV": pyo.value(self.model.PEV[t]),
                "PGrid": pyo.value(self.model.PGrid[t]),
                "EBESS": pyo.value(self.model.EBESS[t]),
                "EEV": pyo.value(self.model.EEV[t]) * ev_mask,
                "SoCBESS": pyo.value(self.model.EBESS[t]) / self.bess.Emax,
                "SoCEV": (pyo.value(self.model.EEV[t]) / self.ev.Emax) * ev_mask,
                "χPV": pyo.value(self.model.χPV[t]),
                "tariff": self.grid.df[t],
                "energy_cost": self.grid.cost(t) *
                               (pyo.value(self.model.PGrid_import[t]) -
                                self.grid.export_tariff_factor * pyo.value(self.model.PGrid_export[t])) *
                               self.sim.Δt,
                "bess_cost": (self.bess.capex / self.bess.ncycles) *
                             (pyo.value(self.model.PBESS_d[t] + self.model.PBESS_c[t]) * self.sim.Δt) / self.bess.Emax,
                "ev_cost": (self.ev.capex / self.ev.ncycles) *
                           (pyo.value(self.model.PEV_d[t] + self.model.PEV_c[t]) * self.sim.Δt) / self.ev.Emax,
            }

        return self.operation

    def get_obs(self, t):
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

        def _is_connected(v):
            return int(v) in (1, 2)

        conn_t = int(self.ev.df["ev_conn"].loc[t])
        ev_connected = _is_connected(conn_t)
        ev_controllable = conn_t == 1

        if t == self.sim.start:
            bess_soc_state = float(self.bess.soc0)
            ev_soc_state = 0.0
        else:
            t_prev = self.model.Ωt.prev(t)
            bess_soc_state = float(self.operation.loc[t_prev, "SoCBESS"])

            conn_prev = int(self.ev.df["ev_conn"].loc[t_prev])
            ev_prev_connected = _is_connected(conn_prev)

            if not ev_controllable:
                ev_soc_state = 0.0
            elif not ev_prev_connected:
                ev_soc_state = 0.0
            else:
                ev_soc_state = float(self.operation.loc[t_prev, "SoCEV"])

        power_obs = [
            self.load.power(t) / self.Pnorm,
            self.pv.power(t) / self.Pnorm,
            bess_soc_state,
            ev_soc_state * int(ev_controllable),
            1.0 if ev_controllable else 0.0,
        ]

        tariff_obs = [self.grid.df[t]]
        weather_obs = self.weather._get_obs(t)
        observations = observation + power_obs + tariff_obs + weather_obs
        return np.array(observations, dtype=np.float32)

    def get_actions(self, t):
        PBESS_des = float(self.operation.loc[t, "PBESS"])
        PEV_des = float(self.operation.loc[t, "PEV"])
        χPV = float(self.operation.loc[t, "χPV"])

        def _is_connected(v):
            return int(v) in (1, 2)

        conn_t = int(self.ev.df["ev_conn"].loc[t])
        ev_connected = _is_connected(conn_t)

        bess_cmd = float(np.clip(PBESS_des / self.bess.Pmax, -1.0, 1.0))

        # EV: disconnected, arrival, or departure step => 0
        if (not ev_connected) or (conn_t == 2):
            ev_cmd = 0.0
        else:
            # here conn_t == 1
            if t == self.sim.start:
                ev_prev_connected = False
            else:
                t_prev = self.model.Ωt.prev(t)
                conn_prev = int(self.ev.df["ev_conn"].loc[t_prev])
                ev_prev_connected = _is_connected(conn_prev)

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
