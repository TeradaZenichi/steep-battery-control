# Two-Machine Training Plan

This folder contains an operational workflow to split model training across two machines.

## Model split

- Machine A models: `TCN`, `ATT_MEM`, `ATT_MEMv2`, `ATTv2`, `MLPv2`
- Machine B models: `TCNv2`, `ATT`, `GRU`, `MLP`, `GRUv2`

The split keeps one heavy TCN family on each side and balances medium models.

## 1) Run training jobs on each machine

From project root:

```powershell
# Machine A
.\.venv\Scripts\python.exe .\scripts\distributed\run_split_training.py --machine A --stage all

# Machine B
.\.venv\Scripts\python.exe .\scripts\distributed\run_split_training.py --machine B --stage all
```

Optional stage filters:

```powershell
# RL only
.\.venv\Scripts\python.exe .\scripts\distributed\run_split_training.py --machine A --stage rl

# IL only
.\.venv\Scripts\python.exe .\scripts\distributed\run_split_training.py --machine B --stage il
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

- Use the same git commit hash on both machines.
- Use same Python/CUDA/PyTorch versions.
- Keep `data/` identical in both machines.
- Do not run same `model/stage/tariff` simultaneously on both machines.
- Merge only after both runs finish.
- Keep distributed logs from both machines in `Results/analysis/distributed/`.
