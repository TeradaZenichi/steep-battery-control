from __future__ import annotations

import argparse
import codecs
import csv
import json
import locale
import os
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

SUCCESS_STATUSES = {"ok", "dry-run"}
IL_TARIFFS = ("tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat")


def job_key(job: dict) -> tuple[str, str]:
    return job["model"], job["stage"]


def checkpoint_path(machine: str, stage_mode: str) -> Path:
    return ANALYSIS_DIR / f"train_split_machine_{machine}_{stage_mode}_latest.json"


def load_checkpoint(path: Path, machine: str, stage_mode: str, jobs: list[dict]) -> dict | None:
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if data.get("machine") != machine or data.get("stage") != stage_mode:
        return None

    plan = data.get("jobs_plan") or []
    current_plan = [{"model": j["model"], "stage": j["stage"]} for j in jobs]
    if plan != current_plan:
        return None

    if not isinstance(data.get("results"), list):
        return None

    return data


def save_checkpoint(
    path: Path,
    machine: str,
    stage_mode: str,
    python_exe: str,
    jobs: list[dict],
    results: list[dict],
    finished: bool,
) -> None:
    payload = {
        "machine": machine,
        "stage": stage_mode,
        "python": python_exe,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "finished": finished,
        "jobs_plan": [{"model": j["model"], "stage": j["stage"]} for j in jobs],
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _latest_result(results: list[dict], model: str, stage: str) -> dict | None:
    for row in reversed(results):
        if row.get("model") == model and row.get("stage") == stage:
            return row
    return None


def _has_il_artifacts(model: str) -> bool:
    base = ROOT / "Results" / "train" / model / "1-IL"
    for tariff in IL_TARIFFS:
        folder = base / tariff
        if not (folder / "best.pth").exists():
            return False
        if not (folder / "best_params.json").exists():
            return False
    return True


def check_il_prerequisite(model: str, stage_mode: str, results: list[dict]) -> tuple[bool, str]:
    il_row = _latest_result(results, model, "1-IL")
    if il_row is not None:
        status = str(il_row.get("status"))
        if status in SUCCESS_STATUSES:
            return True, f"1-IL already successful (status={status}) in this plan"
        return False, f"1-IL exists with non-success status={status}"

    if stage_mode == "rl":
        if _has_il_artifacts(model):
            return True, "1-IL artifacts found on disk"
        return False, "1-IL artifacts not found on disk (best.pth/best_params.json per tariff)"

    return False, "1-IL has not run yet in this plan"


def run_job(job: dict, python_exe: str, log_dir: Path, dry_run: bool, live_output: bool) -> dict:
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
        if live_output:
            env = os.environ.copy()
            # Keep child process output unbuffered so tqdm/prints appear immediately.
            env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=env,
            )

            if proc.stdout is None:
                raise RuntimeError("Failed to capture child process output stream.")

            stream_encoding = locale.getpreferredencoding(False) or "utf-8"
            decoder = codecs.getincrementaldecoder(stream_encoding)(errors="replace")

            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    log.write(text)
                    log.flush()

            tail = decoder.decode(b"", final=True)
            if tail:
                sys.stdout.write(tail)
                sys.stdout.flush()
                log.write(tail)
                log.flush()

            rc = proc.wait()
        else:
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
        "note",
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
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from latest incomplete run for the selected machine/stage (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore latest checkpoint and run all jobs from scratch.",
    )
    parser.add_argument(
        "--live-output",
        dest="live_output",
        action="store_true",
        help="Stream each child training output to console while also writing logs (default).",
    )
    parser.add_argument(
        "--no-live-output",
        dest="live_output",
        action="store_false",
        help="Disable live child output and write only to log files.",
    )
    parser.set_defaults(live_output=True, resume=True)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = ANALYSIS_DIR / f"logs_machine_{args.machine}_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.machine, args.stage)
    if not jobs:
        print("No jobs found for selection.")
        return

    ckpt_path = checkpoint_path(args.machine, args.stage)
    prior_state = load_checkpoint(ckpt_path, args.machine, args.stage, jobs) if args.resume else None

    completed_results: list[dict] = []
    completed_keys: set[tuple[str, str]] = set()
    resumed_count = 0

    if prior_state:
        for row in prior_state["results"]:
            if row.get("status") in SUCCESS_STATUSES:
                key = (row.get("model"), row.get("stage"))
                if key not in completed_keys:
                    completed_keys.add(key)
                    completed_results.append(row)
        resumed_count = len(completed_results)

    pending_jobs = [job for job in jobs if job_key(job) not in completed_keys]

    print(
        f"Machine {args.machine} | stage={args.stage} | jobs={len(jobs)} "
        f"| live_output={args.live_output} | resume={args.resume}"
    )
    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {job['model']} {job['stage']} -> {job['train_path']}")

    if resumed_count:
        print(f"[RESUME] Loaded {resumed_count} completed job(s) from {ckpt_path}")
    if not pending_jobs:
        print("All jobs for this selection are already complete in latest checkpoint.")
        print("Use --no-resume to run all jobs again from scratch.")
        return

    results: list[dict] = list(completed_results)
    save_checkpoint(
        ckpt_path,
        args.machine,
        args.stage,
        args.python,
        jobs,
        results,
        finished=False,
    )

    t0 = time.perf_counter()

    for idx, job in enumerate(pending_jobs, start=1):
        print(f"[RUN {idx}/{len(pending_jobs)}] {job['model']} {job['stage']}")

        if job["stage"] == "2-RL":
            il_ok, il_note = check_il_prerequisite(job["model"], args.stage, results)
            if not il_ok:
                out = {
                    "model": job["model"],
                    "stage": job["stage"],
                    "status": "blocked-il-prereq",
                    "returncode": None,
                    "elapsed_sec": 0.0,
                    "command": [args.python, str(job["train_path"])],
                    "log_file": str(log_dir / f"{job['model']}__{job['stage']}__train.log"),
                    "note": il_note,
                }
                results.append(out)
                print(f"[BLOCKED] {job['model']} {job['stage']} reason={il_note}")
                save_checkpoint(
                    ckpt_path,
                    args.machine,
                    args.stage,
                    args.python,
                    jobs,
                    results,
                    finished=False,
                )
                if args.stop_on_error:
                    print("Stopping on first blocked/error result (--stop-on-error).")
                    break
                continue

        out = run_job(job, args.python, log_dir, args.dry_run, args.live_output)
        results.append(out)
        print(
            f"[DONE] {job['model']} {job['stage']} status={out['status']} "
            f"elapsed={out['elapsed_sec']:.1f}s rc={out['returncode']}"
        )

        save_checkpoint(
            ckpt_path,
            args.machine,
            args.stage,
            args.python,
            jobs,
            results,
            finished=False,
        )

        if args.stop_on_error and out["status"] not in SUCCESS_STATUSES:
            print("Stopping on first blocked/error result (--stop-on-error).")
            break

    total_elapsed = time.perf_counter() - t0

    summary = {
        "machine": args.machine,
        "stage": args.stage,
        "python": args.python,
        "run_id": run_id,
        "resumed_completed_count": resumed_count,
        "total_elapsed_sec": float(total_elapsed),
        "total_elapsed_min": float(total_elapsed / 60.0),
        "ok_count": int(sum(1 for r in results if r["status"] in SUCCESS_STATUSES)),
        "error_count": int(sum(1 for r in results if r["status"] not in SUCCESS_STATUSES)),
        "jobs": results,
    }

    json_path = ANALYSIS_DIR / f"train_split_machine_{args.machine}_{args.stage}_{run_id}.json"
    csv_path = ANALYSIS_DIR / f"train_split_machine_{args.machine}_{args.stage}_{run_id}.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, results)

    all_complete = all(job_key(job) in {(r.get("model"), r.get("stage")) for r in results if r.get("status") in SUCCESS_STATUSES} for job in jobs)
    save_checkpoint(
        ckpt_path,
        args.machine,
        args.stage,
        args.python,
        jobs,
        results,
        finished=all_complete,
    )

    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}")
    print(
        f"[TOTAL] elapsed={summary['total_elapsed_sec']:.1f}s "
        f"({summary['total_elapsed_min']:.2f} min) ok={summary['ok_count']} err={summary['error_count']}"
    )


if __name__ == "__main__":
    main()
