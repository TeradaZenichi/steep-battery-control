# steep-battery-control

Research repository for residential energy management with battery energy storage (BESS), electric vehicle (EV), photovoltaic generation (PV), and dynamic electricity tariffs.

The project uses a two-stage learning pipeline:

1. `1-IL` (Imitation Learning): train actors to imitate a MILP teacher (`Pyomo`).
2. `2-RL` (Reinforcement Learning): refine policies with SAC plus feasibility-aware projection and dual penalty.

## Technical Overview

- **Environment**: `environment/__init__.py` (`SmartHomeEnv`) models grid, load, PV, BESS, EV, weather, and tariffs.
- **Teacher/oracle**: `opt/__init__.py` (`Teacher`) solves dispatch by mathematical optimization.
- **Model families** in `models/`:
  - `ATT`, `ATTv2`
  - `ATT_MEM`, `ATT_MEMv2`
  - `GRU`, `GRUv2`
  - `MLP`, `MLPv2`
  - `TCN`, `TCNv2`
- **Tariffs** evaluated in train/test loops: `tar_s`, `tar_w`, `tar_sw`, `tar_tou`, `tar_flat`.
- **Datasets**: `CY/WY` with `Cur` (training/validation) and `Fut` (test/generalization).

## Repository Structure

- `data/`: CSV datasets and global parameters (`parameters.json`).
- `environment/`: Gymnasium environment.
- `models/<family>/1-IL/`: IL train/test/HPO/plot scripts.
- `models/<family>/2-RL/`: RL train/test scripts and RL utilities.
- `models/test_utils/`: shared test-time teacher and summary utilities.
- `opt/`: MILP teacher.
- `scripts/analysis/`: post-processing, statistical tests, and report tables.
- `scripts/distributed/`: split training and merge utilities.
- `Results/`: generated artifacts (checkpoints, summaries, logs, analytics).

## Environment Details

The environment is implemented in `SmartHomeEnv` (`environment/__init__.py`).

### Core Dynamics

- Simulation step is controlled by `data/parameters.json` (`general.timestep`, currently 5 minutes).
- Episode length is `days * 24h`, discretized by timestep.
- Grid power is:
  - `PGrid = PLoad + PBESS + PEV - PPV`
- Reward per step is:
  - `reward = -(energy_cost + grid_penalty + bess_cost + ev_cost + pv_cost)`

### Action Space

The action is 3D:

- `a[0]`: BESS command in `[-1, 1]` (discharge/charge).
- `a[1]`: EV command in `[-1, 1]` (V2G/charge, subject to EV availability and limits).
- `a[2]`: PV curtailment in `[0, 1]`.

### Observation Space

The observation has 23 features and includes:

1. Time encodings (sin/cos for minute, hour, day, month, weekday).
2. Power/system terms:
   - normalized load,
   - normalized PV,
   - BESS SoC,
   - EV SoC (masked when EV is not controllable),
   - EV controllability flag.
3. Current tariff value.
4. Weather features:
   - `drybulb_C`,
   - `relhum_percent`,
   - `Global Horizontal Radiation`,
   - `dni_Wm2`,
   - `dhi_Wm2`,
   - `Wind Speed (m/s)`,
   - `wdir_deg`.

### EV Logic

The EV sub-environment uses `ev_conn`, `ev_arrival`, and `ev_departure`:

- `ev_conn == 0`: disconnected.
- `ev_conn == 1`: connected/controllable.
- `ev_conn == 2`: departure step (power forced to zero).

Arrival and departure are modeled explicitly, including:

- trip-energy jumps,
- fast-charging cost proxy,
- SoC-min penalties while connected,
- degradation terms.

### Operation Logging

When `track_operation=True`, each step is logged to `env.operation` with:

- commands (`bess_cmd`, `ev_cmd`, `pv_cmd`),
- physical powers/energies (`PLoad`, `PPV`, `PBESS`, `PEV`, `PGrid`, `EBESS`, `EEV`),
- SoCs (`SoCBESS`, `SoCEV`),
- costs and reward components.

### Required Dataset Columns

Training/testing CSV files are expected to include at least:

- `timestamp`
- `electricity_demand_rate_W`
- `produced_electricity_rate_W`
- `ev_conn`, `ev_arrival`, `ev_departure`
- tariff columns (`tar_s`, `tar_w`, `tar_sw`, `tar_tou`, `tar_flat`)
- weather columns used in state normalization.

## RL Scripts (Detailed)

RL scripts live under `models/<family>/2-RL/`.

### `train.py`

Main behavior:

- Trains one SAC agent per tariff (`tar_s`, `tar_w`, `tar_sw`, `tar_tou`, `tar_flat`).
- Uses two training streams (`CY` and `WY`) sampled by `EpisodeGen`.
- Runs warmup episodes with random actions before gradient updates.
- Uses a lazy-frame replay buffer with:
  - history stacking (`history_len`),
  - n-step returns (`n_step`),
  - lower memory usage by storing flat transitions and reconstructing sequences on sample.
- Supports multi-thread stepping for train environments (`train_env_workers`).
- Supports process-pool parallel evaluation (`eval_workers`).

SAC update logic includes:

- critic backup with entropy term,
- actor loss with three terms:
  - entropy term,
  - Q term,
  - dual feasibility term (`lambda * cost`) when enabled.
- automatic entropy temperature update (`alpha`) when configured.
- target critic soft update (`tau`).

Feasibility/constraint handling:

- Actor output is projected to action-feasible bounds (model-level projection).
- Projection residual induces a cost.
- Dual variable `lambda` is updated with:
  - `lambda <- clamp(lambda + lr * (E[cost] - cost_limit), 0, lambda_max)`.

Evaluation and checkpoint policy:

- Periodic mini-eval (`evaluate_every`) on a subset of validation runs.
- Optional full eval when checkpoint score improves.
- Tracks deterministic and stochastic metrics.
- Saves 3 actor variants:
  - `best_actor_eval.pt` (combo score),
  - `best_actor_eval_det.pt`,
  - `best_actor_eval_stoch.pt`.
- Saves full checkpoint files and metadata JSON.
- Writes `audit_training.csv` with episode-level diagnostics:
  - rewards,
  - Q/backup statistics,
  - alpha/lambda,
  - cost violation metrics,
  - actor/critic/alpha losses,
  - no-improvement counters,
  - timing.

Early stop:

- Controlled by `early_stop_patience` and `min_episodes_before_early_stop`.

IL-to-RL inheritance:

- RL can load IL HPO artifacts (`Results/train/<family>/1-IL/<tariff>/best_params.json`).
- `il_inherit_mode` controls what gets inherited (for example `history_len`, optionally LR/batch/weight decay).

### `test.py`

Main behavior:

- Evaluates trained RL actors against the cached teacher summaries.
- Supports checkpoint selection with:
  - `--actor-variant combo|det|stoch`
- For each tariff and configured test run:
  - rolls out actor in `SmartHomeEnv`,
  - computes actor reward,
  - compares to teacher reward,
  - exports optional operation and breakdown CSVs.
- Writes:
  - `summary_<variant>.json`,
  - legacy `summary.json` for `combo`,
  - per-variant operation files,
  - cross-variant comparison table `actor_variant_comparison.csv`.

### `utils.py`

Contains reusable RL components:

- `ReplayBuffer` (lazy-frame, n-step, sequence reconstruction),
- `EpisodeGen` (CY/WY sampling outside validation windows),
- `Hyperparameters`,
- `Temperature` (learnable entropy coefficient),
- `_eval_worker` (parallel evaluation rollout helper).

## Requirements

- Python `3.13` (project currently uses this version).
- Dependencies from `requirements.txt`.
- MILP solver available through Pyomo (current default in teacher is `gurobi`).
- CUDA GPU is optional but strongly recommended for full experiments.
- `scipy` for Wilcoxon tests in `scripts/analysis/generate_bootstrap_wilcoxon.py`.

## Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install scipy
```

## How To Run

Run all commands from the repository root.

### 1) Build teacher baseline for test runs

```powershell
.\.venv\Scripts\python.exe .\generate_teacher_test_baseline.py --family all --stage all
```

This creates/reuses shared teacher cache and writes `teacher_summary.json` per family/stage/tariff.

### 2) Train and test one family (example: MLP)

```powershell
# IL
.\.venv\Scripts\python.exe .\models\MLP\1-IL\train.py
.\.venv\Scripts\python.exe .\models\MLP\1-IL\test.py

# RL
.\.venv\Scripts\python.exe .\models\MLP\2-RL\train.py
.\.venv\Scripts\python.exe .\models\MLP\2-RL\test.py --actor-variant combo
.\.venv\Scripts\python.exe .\models\MLP\2-RL\test.py --actor-variant det
.\.venv\Scripts\python.exe .\models\MLP\2-RL\test.py --actor-variant stoch
```

### 3) Run all families sequentially

```powershell
$families = @("ATT","ATTv2","ATT_MEM","ATT_MEMv2","GRU","GRUv2","MLP","MLPv2","TCN","TCNv2")
foreach ($f in $families) {
  .\.venv\Scripts\python.exe ".\models\$f\1-IL\train.py"
  .\.venv\Scripts\python.exe ".\models\$f\1-IL\test.py"
  .\.venv\Scripts\python.exe ".\models\$f\2-RL\train.py"
  .\.venv\Scripts\python.exe ".\models\$f\2-RL\test.py --actor-variant combo"
}
```

## Distributed Training (A/B/C)

Recommended for large experimental campaigns.

### Run split training

```powershell
# Machine A
.\.venv\Scripts\python.exe .\scripts\distributed\run_machine_A.py

# Machine B
.\.venv\Scripts\python.exe .\scripts\distributed\run_machine_B.py

# Machine C
.\.venv\Scripts\python.exe .\scripts\distributed\run_machine_C.py
```

You can also run:

```powershell
.\.venv\Scripts\python.exe .\scripts\distributed\run_split_training.py --machine A --stage all
```

### Package and merge results

```powershell
# On remote machine (packager currently accepts A or B)
.\.venv\Scripts\python.exe .\scripts\distributed\package_split_results.py --machine B --stage all

# On main machine
.\.venv\Scripts\python.exe .\scripts\distributed\merge_split_results.py --source .\Results\analysis\package_machine_B_all_YYYYMMDD_HHMMSS.zip
```

## Analysis and Reporting

After `Results/test` is available:

```powershell
.\.venv\Scripts\python.exe .\scripts\analysis\generate_tariff_spreadsheets.py
.\.venv\Scripts\python.exe .\scripts\analysis\generate_bootstrap_wilcoxon.py
.\.venv\Scripts\python.exe .\scripts\analysis\generate_statistical_tables.py
.\.venv\Scripts\python.exe .\scripts\analysis\generate_operational_stats.py
.\.venv\Scripts\python.exe .\scripts\analysis\generate_critic_weakness_diagnostic.py
```

Main outputs:

- `Results/analysis/`: per-tariff comparisons and distributed logs.
- `Results/statistical_tests/`: bootstrap CI + paired Wilcoxon outputs (CSV/TEX).
- `Results/figures/analysis/operational_stats/`: operational comparison plots.

## Expected Artifacts

- `Results/train/<MODEL>/1-IL/<TARIFF>/`
  - `best.pth`
  - `best_params.json`
  - `actor_cfg.json`
- `Results/train/<MODEL>/2-RL/<TARIFF>/`
  - `best_actor_eval.pt`, `best_actor_eval_det.pt`, `best_actor_eval_stoch.pt`
  - `best_checkpoint_eval*.pt`
  - `best_eval*_meta.json`
  - `audit_training.csv`
  - `final_full_eval.json` (when enabled)
- `Results/test/<MODEL>/<STAGE>/<TARIFF>/`
  - `teacher_summary.json`
  - `summary.json` and/or `summary_<variant>.json`
  - actor/teacher operation CSVs
  - breakdown CSVs

## Academic Contributions

1. Hybrid optimization-learning methodology (MILP teacher plus IL/RL student policies).
2. Controlled benchmarking across architectures, tariffs, and training stages.
3. Reproducible experiment setup through JSON configs, fixed seeds, and teacher caching.
4. Fair IL vs RL comparisons on shared test definitions and identical teacher baseline.
5. Statistical rigor using bootstrap confidence intervals and paired Wilcoxon testing.
6. Interpretable cost decomposition (energy, penalties, BESS/EV degradation, fast-charge events).
7. Scalable workflow via distributed training and result consolidation.

## Notes

- The teacher uses `gurobi` by default in `opt.Teacher.solve()`.
- Full training is computationally expensive; distributed execution is recommended.
- Some analysis scripts intentionally focus on model subsets, depending on experiment design.
