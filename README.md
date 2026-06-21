# steep-battery-control

Research code for residential home energy management with photovoltaic generation
(PV), a stationary battery (BESS), an electric vehicle (EV), and grid exchange
under dynamic electricity tariffs.

The current paper studies whether Soft Actor-Critic (SAC) can be trained for
this setting in a stable, safe, and economically competitive way without relying
on expert demonstrations or behavior-cloning warm starts.

> **For new methods, start with the environment.** The main reusable artifact in
> this repository is `environment.SmartHomeEnv`: a method-agnostic, Gymnasium-style
> simulator of a home with PV, BESS, EV, and grid exchange under five tariffs. You
> can plug in any controller (RL, MPC, offline RL, evolutionary search, or a
> hand-written rule) without needing the rest of the codebase. See
> [Using The Environment Directly](#using-the-environment-directly) for a minimal
> rollout. Everything else here (`sac/`, `opt/`, `supervised/`, `baselines/`) is one
> set of controllers built on top of the simulator, not a prerequisite for using it.

## Scope

This repository provides the simulation environment, a set of controllers, and an
evaluation harness for the residential PV + BESS + EV + grid setting under dynamic
tariffs. The controllers include rule-based baselines, a behavior-cloning reference
distilled from a MILP teacher, and several Soft Actor-Critic variants (penalty-based
and CMDP-Lagrangian, with optional reward shaping and anti-overestimation critics).
They are run across three sequence encoders (GRU, attention, TCN) and five tariff
structures under a shared train/validation/test protocol.

The associated study is in progress; this README documents how to run the code and
how the pieces fit together, not its results.

## What Is Versioned

The repository tracks code, configurations, and input data needed to reproduce
the experiments.

Important tracked paths:

- `data/`: simulation data and system parameters.
- `environment/`: residential energy-management simulator.
- `models/`: neural sequence encoders used by supervised and SAC controllers.
- `opt/`: MILP teacher used to generate behavior-cloning references.
- `supervised/`: behavior-cloning training used only for the FT comparator.
- `sac/`: current unified SAC experiment harness.
- `baselines/`: rule-based baselines.
- `reinforcement/`: legacy SAC implementations kept for reference.
- `scripts/`: auxiliary experiment and analysis scripts.

Generated experiment artifacts are intentionally ignored by git. In particular,
the `.gitignore` excludes:

- `paper/`: local training, testing, ablation, checkpoint, and operation outputs.
- `Results/`: legacy generated outputs.
- `.venv/`, logs, caches, backups, and `__pycache__/`.

This means `paper/train/...`, `paper/test/...`, `paper/ablation/...`, model
checkpoints, audit CSVs, and operation traces are local products of a run. They
are not expected to be present after cloning the repository.

## Using The Environment Directly

The simulator is a Gymnasium-style environment implemented as
`environment.SmartHomeEnv`, and it is the recommended primary entry point for
anyone building on this work. It runs without the SAC harness, so a new
methodology (PPO, TD3, online MPC, offline RL, evolutionary search, or a
hand-written controller) can be plugged in directly. For comparable results,
evaluate on the same monthly windows and tariff columns (see the comparison note
at the end of this section).

Minimal rollout:

```python
import json
import pandas as pd

from environment import SmartHomeEnv

df = pd.read_csv(
    "data/Simulation_CY_Cur_HP__PV5000-HB5000.csv",
    sep=";",
    parse_dates=["timestamp"],
    dayfirst=True,
    index_col="timestamp",
)

with open("data/parameters.json", encoding="utf-8") as f:
    params = json.load(f)

env = SmartHomeEnv(
    df=df,
    parameters=params,
    start=pd.Timestamp("2000-07-01 00:00:00"),
    days=2,
    BESS_SoC=0.5,
    tariff="tar_sw",
    track_operation=True,
)

obs, info = env.reset()
done = False
episode_reward = 0.0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    done = terminated or truncated

print(episode_reward)
print(env.operation.tail())
```

The environment is not registered through `gym.make`; instantiate
`SmartHomeEnv` directly.

### Inputs

`SmartHomeEnv` expects a timestamp-indexed dataframe and the parameter dictionary
from `data/parameters.json`.

Required dataframe columns for direct closed-loop simulation:

- `electricity_demand_rate_W`
- `produced_electricity_rate_W`
- `ev_conn`
- `ev_arrival`
- the selected tariff column, for example `tar_sw`
- weather columns used by normalization:
  - `drybulb_C`
  - `relhum_percent`
  - `Global Horizontal Radiation`
  - `dni_Wm2`
  - `dhi_Wm2`
  - `Wind Speed (m/s)`
  - `wdir_deg`

The bundled datasets also include `ev_departure`, which is useful for teacher
and analysis workflows, but the closed-loop environment uses `ev_conn` and
precomputed arrival/departure indices internally.

### Actions

The action is a three-dimensional continuous vector:

```text
[a_bess, a_ev, a_pv]
```

with bounds:

- `a_bess in [-1, 1]`: negative discharges the stationary battery, positive
  charges it.
- `a_ev in [-1, 1]`: negative discharges the EV when available, positive charges
  it.
- `a_pv in [0, 1]`: photovoltaic curtailment fraction.

The environment clips actions to this box and then applies device-level physics:
state-of-charge limits, power limits, availability, efficiencies,
self-discharge, battery degradation, EV trip energy, and EV fast-charging
fallback costs.

The bare environment does not guarantee grid-feasible actions. If grid import or
export limits are exceeded, the violation is charged through the grid penalty in
the reward. For deployment-style comparisons, use a safety guard during
evaluation or reproduce the safety-projected evaluation path used by the SAC
tests.

### Observations

The observation vector currently has 34 features:

- 10 cyclic time features for minute, hour, day, month, and weekday.
- normalized load and available PV.
- BESS SoC, EV SoC, and EV controllability flag.
- current tariff.
- 7 normalized weather features.
- tariff percentile over recent history.
- short-term PV moving average.
- EV timing features: time since arrival, time until departure, and time until
  next arrival.
- one-day and one-week lags of tariff, PV, and load.

The exact construction is in `SmartHomeEnv._get_observation()`. A new method
should treat `env.observation_space.shape` as the source of truth rather than
hard-coding the dimension.

### Rewards And Diagnostics

The reward is the negative operating cost:

```text
reward = -(grid_energy_cost + grid_penalty + bess_cost + ev_cost + pv_cost)
```

The `info` dictionary returned by `step()` includes:

- `energy_cost`
- `bess_cost`
- `ev_cost`
- `pv_cost`
- `penalty`
- `pgrid`
- `pbess`
- `pev`
- `ppv`
- `timestep`

For fast training, use `track_operation=False`. For debugging and plots, use
`track_operation=True`; the full step-by-step trace is stored in
`env.operation`.

### Resetting Scenarios

The same environment instance can be reused with different start dates, horizon
lengths, and initial BESS SoC:

```python
obs, info = env.reset(options={
    "start": "2000-08-01 00:00:00",
    "days": 7,
    "bess_soc": 0.8,
})
```

For fair comparison with the paper grid, evaluate methods on the same monthly
validation/test windows and tariff columns used by `sac/run_final_grid.py` and
the SAC test scripts.

## Current Experiment Matrix

The final SAC grid is defined in `sac/run_final_grid.py`.

Methods in the final grid:

- `u_penalty`: penalty-based SAC.
- `u_cmdp`: CMDP-Lagrangian SAC with separate EV cost critics.
- `u_cmdp_shaped`: CMDP SAC plus potential-based BESS shaping.
- `u_cmdp_ft`: CMDP SAC initialized from a behavior-cloning actor.
- `u_cmdp_droq`: CMDP SAC with DroQ-style critic dropout.
- `u_cmdp_redq`: CMDP SAC with a REDQ-style randomized critic ensemble.

Architectures:

- `GRU`
- `MHA`
- `TCN`

Tariffs:

- `tar_flat`
- `tar_s`
- `tar_w`
- `tar_sw`
- `tar_tou`

Seeds:

- Seed `42` for every method, architecture, and tariff.
- Extra seeds `7` and `13` on the lead tariff `tar_sw`.

The lean final grid therefore has:

- `6` methods
- `3` architectures
- `5` tariffs
- `3` seeds only for `tar_sw`
- `126` train/test cells in total

## SAC Harness

The unified SAC implementation lives under `sac/`.

Key files:

- `sac/common/trainer.py`: shared train loop, validation, checkpointing.
- `sac/common/updates.py`: SAC, CMDP, DroQ, and REDQ update logic.
- `sac/run_final_grid.py`: generic grid orchestrator with skip-if-done markers.
- `sac/run_final_gru.py`: hardcoded GRU machine runner.
- `sac/run_final_mha.py`: hardcoded MHA machine runner.
- `sac/run_final_tcn.py`: hardcoded TCN machine runner.
- `sac/make_configs.py`: config generator. See the warning below before using it.

Each experiment directory follows the same layout:

```text
sac/<method>/<arch>/
    config.json
    train.py
    test.py
```

Training writes to:

```text
paper/train/<method>/<arch>/<tariff>[-s<seed>]/
```

Testing writes to:

```text
paper/test/<method>/<arch>/<tariff>[-s<seed>]/
```

The orchestrator skips completed work using:

- `.train_done`
- `.test_done`

If a directory exists without the corresponding marker, treat it as an
in-progress or interrupted run.

## Behavior Cloning Comparator

The main SAC methods are demonstration-free. Behavior cloning is used only for:

- the stand-alone supervised reference, and
- the `u_cmdp_ft` fine-tuning comparator.

Supervised checkpoints are expected at:

```text
paper/train/supervised/<arch>/<tariff>/best.pt
```

They can be generated with:

```powershell
.\.venv\Scripts\python.exe supervised\run_supervised_gru.py
.\.venv\Scripts\python.exe supervised\run_supervised_mha.py
.\.venv\Scripts\python.exe supervised\run_supervised_tcn.py
```

The architecture-specific supervised runners are idempotent and train the five
tariffs for that architecture. The GRU runner forces CPU use so it can run
without competing with an active SAC grid on the GPU.

## Running The Final Grid

All commands should be launched from the repository root.

To preview the current plan and local completion markers:

```powershell
.\.venv\Scripts\python.exe sac\run_final_grid.py --plan-only
```

Preview only one architecture:

```powershell
.\.venv\Scripts\python.exe sac\run_final_grid.py --archs GRU --plan-only
```

Run the hardcoded architecture runners:

```powershell
.\.venv\Scripts\python.exe sac\run_final_gru.py
.\.venv\Scripts\python.exe sac\run_final_mha.py
.\.venv\Scripts\python.exe sac\run_final_tcn.py
```

Run a distributed split across three machines:

```powershell
.\.venv\Scripts\python.exe sac\run_final_grid.py --machine 1/3
.\.venv\Scripts\python.exe sac\run_final_grid.py --machine 2/3
.\.venv\Scripts\python.exe sac\run_final_grid.py --machine 3/3
```

Run only a method subset:

```powershell
.\.venv\Scripts\python.exe sac\run_final_grid.py --methods u_cmdp,u_cmdp_shaped --archs GRU
```

Run only seed `42` without extra lead-tariff seeds:

```powershell
.\.venv\Scripts\python.exe sac\run_final_grid.py --seeds 42 --extra-seeds none
```

Train without annual testing:

```powershell
.\.venv\Scripts\python.exe sac\run_final_grid.py --skip-test
```

## Monitoring Runs

Use the plan view for the most reliable live status:

```powershell
.\.venv\Scripts\python.exe sac\run_final_grid.py --archs GRU --plan-only
```

Training diagnostics are written incrementally to:

```text
paper/train/<method>/<arch>/<tariff>[-s<seed>]/audit_training.csv
```

Example:

```powershell
Get-Content paper\train\u_cmdp\GRU\tar_sw-s13\audit_training.csv -Tail 5
```

A completed training cell contains checkpoints:

- `best.pt`
- `best_robust.pt`
- `best_worst.pt`
- `last.pt`

and metadata files:

- `best_meta.json`
- `best_robust_meta.json`
- `best_worst_meta.json`
- `last_meta.json`

Annual testing evaluates the checkpoints in raw and safety-projected modes and
writes summaries plus operation traces under `paper/test/...`.

## Important Config Warning

During the current campaign, the active `sac/<method>/<arch>/config.json` files
are the source of truth.

Do not regenerate configs blindly with:

```powershell
.\.venv\Scripts\python.exe sac\make_configs.py
```

unless `sac/make_configs.py` has first been checked against the active configs.
A previous unstable setting used `target_entropy = -1.0`; the patched active
final-grid configs use `target_entropy = -2.0` to avoid entropy-temperature
explosion caused by the narrow feasible BESS action interval.

## Ablation Campaign

An ablation campaign lives under `sac/ablation/`; it was used during development to
study training behavior before the final grid. Its artifacts are written under
`paper/ablation/`, which is ignored by git, and can be regenerated locally when
needed.

## Baselines

Rule-based baselines live under:

- `baselines/RB/`
- `baselines/RBS/`

`RBS` is the safer rule-based reference used as the practical baseline for the
cost comparison. It helps interpret whether a controller is cheaper because it
controls better, rather than because it violates constraints that a deployable
controller would respect.

## Legacy Code

The older `models/` and `reinforcement/` training paths are kept because they
document the project history and still contain reusable components. The current
paper grid, however, is the unified path under `sac/` plus the supervised BC
comparators under `supervised/`.

For new final-paper runs, prefer `sac/run_final_grid.py` or the architecture
runners in `sac/run_final_*.py`.
