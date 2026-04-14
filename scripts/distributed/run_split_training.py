from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "Results" / "analysis" / "distributed"

ASSIGNMENTS = {
    "A": ["TCN", "ATT_MEM", "ATT_MEMv2", "ATTv2", "MLPv2"],
    "B": ["TCNv2", "ATT", "GRU", "MLP", "GRUv2"],
}

STAGE_MAP = {
    "all": ["1-IL", "2-RL"],
    "il": ["1-IL"],
    "rl": ["2-RL"],
}


def build_jobs(machine: str, stage_mode: str) -> list[dict]:
    jobs = []
    models = ASSIGNMENTS[machine]
    stages = STAGE_MAP[stage_mode]

    for model in models:
        for stage in stages:
            train_path = ROOT / "models" / model / stage / "train.py"
            if not train_path.exists():
                continue
            jobs.append(
                {
                    "model": model,
                    "stage": stage,
                    "train_path": train_path,
                }
            )
    return jobs


def run_job(job: dict, python_exe: str, log_dir: Path, dry_run: bool) -> dict:
    cmd = [python_exe, str(job["train_path"])]
    log_file = log_dir / f"{job['model']}__{job['stage']}__train.log"

    if dry_run:
        return {
            "model": job["model"],
            "stage": job["stage"],
            "status": "dry-run",
            "returncode": 0,
            "elapsed_sec": 0.0,
            "command": cmd,
            "log_file": str(log_file),
        }

    t0 = time.perf_counter()
    with open(log_file, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc = proc.wait()
    elapsed = time.perf_counter() - t0

    return {
        "model": job["model"],
        "stage": job["stage"],
        "status": "ok" if rc == 0 else "error",
        "returncode": rc,
        "elapsed_sec": float(elapsed),
        "command": cmd,
        "log_file": str(log_file),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "model",
        "stage",
        "status",
        "returncode",
        "elapsed_sec",
        "log_file",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run split training jobs for machine A or B")
    parser.add_argument("--machine", choices=["A", "B"], required=True)
    parser.add_argument("--stage", choices=["all", "il", "rl"], default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = ANALYSIS_DIR / f"logs_machine_{args.machine}_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.machine, args.stage)
    if not jobs:
        print("No jobs found for selection.")
        return

    print(f"Machine {args.machine} | stage={args.stage} | jobs={len(jobs)}")
    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {job['model']} {job['stage']} -> {job['train_path']}")

    results: list[dict] = []
    t0 = time.perf_counter()

    for idx, job in enumerate(jobs, start=1):
        print(f"[RUN {idx}/{len(jobs)}] {job['model']} {job['stage']}")
        out = run_job(job, args.python, log_dir, args.dry_run)
        results.append(out)
        print(
            f"[DONE] {job['model']} {job['stage']} status={out['status']} "
            f"elapsed={out['elapsed_sec']:.1f}s rc={out['returncode']}"
        )

        if args.stop_on_error and out["status"] == "error":
            print("Stopping on first error (--stop-on-error).")
            break

    total_elapsed = time.perf_counter() - t0

    summary = {
        "machine": args.machine,
        "stage": args.stage,
        "python": args.python,
        "run_id": run_id,
        "total_elapsed_sec": float(total_elapsed),
        "total_elapsed_min": float(total_elapsed / 60.0),
        "ok_count": int(sum(1 for r in results if r["status"] in {"ok", "dry-run"})),
        "error_count": int(sum(1 for r in results if r["status"] == "error")),
        "jobs": results,
    }

    json_path = ANALYSIS_DIR / f"train_split_machine_{args.machine}_{args.stage}_{run_id}.json"
    csv_path = ANALYSIS_DIR / f"train_split_machine_{args.machine}_{args.stage}_{run_id}.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, results)

    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}")
    print(
        f"[TOTAL] elapsed={summary['total_elapsed_sec']:.1f}s "
        f"({summary['total_elapsed_min']:.2f} min) ok={summary['ok_count']} err={summary['error_count']}"
    )


if __name__ == "__main__":
    main()
