# steep-battery-control

## Repository layout

- `models/`: model families and training/evaluation stages (`1-IL`, `2-RL`).
- `data/`: datasets and shared parameter files.
- `Results/`: generated outputs (train/test/statistics/figures).
- `scripts/analysis/`: project-level analysis/report generation scripts.
- `scripts/distributed/`: two-machine orchestration and merge helpers.
- `scripts/tmp/`: temporary/diagnostic scripts kept outside the repository root.

## Common script entrypoints

Run from repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe .\scripts\analysis\generate_tariff_spreadsheets.py
.\.venv\Scripts\python.exe .\scripts\analysis\generate_bootstrap_wilcoxon.py
.\.venv\Scripts\python.exe .\scripts\analysis\generate_statistical_tables.py
```

Distributed training helpers:

```powershell
.\.venv\Scripts\python.exe .\scripts\distributed\run_split_training.py --machine A --stage all
.\.venv\Scripts\python.exe .\scripts\distributed\run_split_training.py --machine B --stage all
```

Teacher-only test baseline (always runs all tariffs and the same `test` set from each `config.json`; teacher results are cached with readable run-based file names and reused across architectures, then `teacher_summary.json` is written per suite/tariff):

```powershell
.\.venv\Scripts\python.exe .\generate_teacher_test_baseline.py --family all --stage all
```

Then run model `test.py` scripts to evaluate only actor checkpoints against saved teacher summaries.