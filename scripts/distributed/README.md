# Three-Machine Training Plan

This folder contains an operational workflow to split model training across three machines, without passing parser arguments in day-to-day execution.

## Model split

- Machine A: `MLP`, `MLPv2`, `GRU`, `ATT_MEM`
- Machine B: `ATT`, `ATTv2`, `GRUv2`
- Machine C: `ATT_MEMv2`, `TCN`, `TCNv2`

Each machine runs models from simpler to more complex, and each model runs in-order: `1-IL` then `2-RL`.

## 1) Run training jobs on each machine

From project root (Python scripts, no parser flags required):

```powershell
# Machine A
.\.venv\Scripts\python.exe .\scripts\distributed\run_machine_A.py

# Machine B
.\.venv\Scripts\python.exe .\scripts\distributed\run_machine_B.py

# Machine C
.\.venv\Scripts\python.exe .\scripts\distributed\run_machine_C.py
```

Shell wrappers are also available:

```bash
./scripts/distributed/run_machine_A.sh
./scripts/distributed/run_machine_B.sh
./scripts/distributed/run_machine_C.sh
```

Outputs are written under `Results/analysis/distributed/`.

## 2) Package results from remote machine

On the machine you want to export from:

```powershell
.\.venv\Scripts\python.exe .\scripts\distributed\package_split_results.py --machine B --stage all
```

This creates a zip package under `Results/analysis/`.

## 3) Merge into main machine

Copy the package file to the main machine, then run:

```powershell
.\.venv\Scripts\python.exe .\scripts\distributed\merge_split_results.py --source .\Results\analysis\package_machine_B_all_YYYYMMDD_HHMMSS.zip
```

You can also pass an extracted folder containing `Results/`.

## Suggested safety checklist

- Use the same git commit hash on all machines.
- Use same Python/CUDA/PyTorch versions.
- Keep `data/` identical on all machines.
- Do not run same `model/stage/tariff` simultaneously on multiple machines.
- Merge only after all runs finish.
- Keep distributed logs from all machines in `Results/analysis/distributed/`.
