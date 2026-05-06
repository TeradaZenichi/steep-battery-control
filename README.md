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

## RL Training Protocol

The `2-RL` stage refines each actor with Soft Actor-Critic (SAC) after IL pretraining/architecture selection. Each model family has its own `models/<family>/2-RL/train.py`, but the training protocol is shared across families.

### Training Rollout

- One SAC agent is trained per tariff: `tar_s`, `tar_w`, `tar_sw`, `tar_tou`, and `tar_flat`.
- Each training episode uses two parallel data streams:
  - `CY` from `Simulation_CY_Cur_HP__PV5000-HB5000.csv`,
  - `WY` from `Simulation_WY_Cur_HP__PV5000-HB5000.csv`.
- The default RL horizon is `7` days at the global 5-minute timestep.
- With two streams, this gives `4032` environment steps per episode:
  - `2016` steps for CY,
  - `2016` steps for WY.
- Warmup uses random actions projected through the same safety layer used during policy rollouts.
- After warmup, the actor samples stochastic actions for exploration and stores projected executable actions in replay.

### SAC Update

Each update samples lazy history sequences from the replay buffer and applies:

- twin critics with target critics,
- n-step returns,
- automatic entropy temperature (`alpha`),
- actor loss composed of:
  - entropy term,
  - Q-value term,
  - dual feasibility term.

The main RL hyperparameters are configured in `models/<family>/2-RL/config.json`, including:

- `batch_size`,
- `history_len`,
- `n_step`,
- `update_every_steps`,
- `lambda_lr`,
- `early_stop_patience`,
- `gamma`,
- `tau`,
- `alpha_lr`.

Other training choices are intentionally fixed in code for reproducibility and simpler experiment management:

- `days = 7`,
- `train_episodes = 300`,
- `warmup_episodes = 10`,
- `evaluate_every = 1`,
- `target_entropy = -2.0`,
- `log_std_max = 0`,
- `buffer_size = 200000`,
- `train_env_workers = 1`,
- `eval_workers = 12`.

### Safety Layer and Dual Penalty

The actor proposes a raw action, and the model-level safety layer projects it to feasible BESS, EV, and PV bounds before the action is executed in `SmartHomeEnv`.

This design keeps the executed policy physically feasible while still teaching the actor to internalize constraints:

1. The projected action is sent to the environment.
2. The projection residual is converted into a feasibility cost.
3. A Lagrangian dual variable `lambda` penalizes repeated dependence on projection.

The current dual policy is:

- `cost_limit = 0.05`,
- `cost_limit_high = 0.05`,
- `cost_limit_low = 0.02`,
- `dual_warmup_episodes = 10`,
- `lambda_max = 1.0`,
- `lambda_lr` from the RL config.

This means the safety layer is used as an execution guard, while the dual term pressures the stochastic actor to generate actions that are feasible before projection.

### Validation and Early Stopping

Validation runs every episode. The validation suite has 24 scenarios:

- 12 monthly CY scenarios,
- 12 monthly WY scenarios,
- 7 days per scenario,
- initial BESS SoC cycling through `0.2`, `0.5`, and `0.8`.

Validation uses a process pool with `eval_workers = 12`, so the 24 scenarios run in two parallel batches. Early stopping is delayed until at least 100 episodes and then controlled by `early_stop_patience`.

### Checkpoints

The trainer writes two deterministic checkpoint families:

- reward checkpoint:
  - `best_actor_eval.pt`,
  - `best_actor_eval_det.pt`,
  - `best_checkpoint_eval.pt`,
  - `best_checkpoint_eval_det.pt`,
  - `best_eval_meta.json`,
  - `best_eval_det_meta.json`.
- safety-aware checkpoint:
  - `best_actor_eval_safe.pt`,
  - `best_checkpoint_eval_safe.pt`,
  - `best_eval_safe_meta.json`.

The reward checkpoint selects the highest deterministic validation reward.

The safety-aware checkpoint first prefers policies with:

- `cost_mean <= 0.04`,
- `cost_p95 <= 0.18`.

Among feasible policies, it selects the highest validation reward. If no feasible policy has appeared yet, it selects the lowest mean projection cost, then lowest p95 projection cost, then lowest violation fraction, then highest reward. The violation fraction is kept as a diagnostic and tie-breaker because it can be high even when the projection magnitude is already small.

### Training Audit

Every 5 episodes, and also at the end of training or early stopping, the trainer flushes `audit_training.csv` with:

- training reward,
- deterministic validation reward,
- best checkpoint scores,
- Q-value and backup statistics,
- `alpha` and `lambda`,
- projection cost metrics,
- violation fraction,
- actor/critic/alpha losses,
- no-improvement counters,
- episode timing.

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

- Deterministic validation every episode on the configured validation suite.
- Saves reward-selected deterministic checkpoints:
  - `best_actor_eval.pt`,
  - `best_actor_eval_det.pt`.
- Saves safety-aware deterministic checkpoints:
  - `best_actor_eval_safe.pt`.
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

### `test.py`

Main behavior:

- Evaluates trained RL actors against the cached teacher summaries.
- Supports checkpoint selection with:
  - `--actor-variant combo|det`
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
  - `best_actor_eval.pt`, `best_actor_eval_det.pt`, `best_actor_eval_safe.pt`
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
