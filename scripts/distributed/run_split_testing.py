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

# Keep split plan aligned with training script.
from run_split_training import ASSIGNMENTS, STAGE_MAP

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "Results" / "analysis" / "distributed"

SUCCESS_STATUSES = {"ok", "dry-run"}
TARIFFS = ("tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat")
RL_VARIANT_TO_CKPT = {
    "combo": "best_actor_eval.pt",
    "det": "best_actor_eval_det.pt",
    "stoch": "best_actor_eval_stoch.pt",
}


def job_key(job: dict) -> tuple[str, str, str]:
    return job["model"], job["stage"], job.get("variant") or ""


def checkpoint_path(machine: str, stage_mode: str) -> Path:
    return ANALYSIS_DIR / f"test_split_machine_{machine}_{stage_mode}_latest.json"


def _jobs_plan(jobs: list[dict]) -> list[dict]:
    return [
        {
            "model": j["model"],
            "stage": j["stage"],
            "variant": j.get("variant"),
        }
        for j in jobs
    ]


def load_checkpoint(path: Path, machine: str, stage_mode: str, jobs: list[dict]) -> dict | None:
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if data.get("machine") != machine or data.get("stage") != stage_mode:
        return None

    if data.get("jobs_plan") != _jobs_plan(jobs):
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
        "jobs_plan": _jobs_plan(jobs),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_jobs(machine: str, stage_mode: str, rl_variants: tuple[str, ...]) -> list[dict]:
    jobs: list[dict] = []
    models = ASSIGNMENTS[machine]
    stages = STAGE_MAP[stage_mode]

    for model in models:
        for stage in stages:
            test_path = ROOT / "models" / model / stage / "test.py"
            if not test_path.exists():
                continue

            if stage == "2-RL":
                for variant in rl_variants:
                    jobs.append(
                        {
                            "model": model,
                            "stage": stage,
                            "variant": variant,
                            "test_path": test_path,
                        }
                    )
            else:
                jobs.append(
                    {
                        "model": model,
                        "stage": stage,
                        "variant": "",
                        "test_path": test_path,
                    }
                )

    return jobs


def _has_teacher_summary(model: str, stage: str) -> tuple[bool, str]:
    base = ROOT / "Results" / "test" / model / stage
    missing: list[str] = []
    for tariff in TARIFFS:
        p = base / tariff / "teacher_summary.json"
        if not p.exists():
            missing.append(tariff)

    if missing:
        return False, f"missing teacher_summary for tariffs={missing}"
    return True, "teacher_summary present for all tariffs"


def _has_il_artifacts(model: str) -> tuple[bool, str]:
    base = ROOT / "Results" / "train" / model / "1-IL"
    missing: list[str] = []
    for tariff in TARIFFS:
        p = base / tariff / "best.pth"
        if not p.exists():
            missing.append(tariff)

    if missing:
        return False, f"missing IL best.pth for tariffs={missing}"
    return True, "IL artifacts present for all tariffs"


def _has_rl_artifacts(model: str, variant: str) -> tuple[bool, str]:
    ckpt_name = RL_VARIANT_TO_CKPT[variant]
    base = ROOT / "Results" / "train" / model / "2-RL"
    missing: list[str] = []
    for tariff in TARIFFS:
        p = base / tariff / ckpt_name
        if not p.exists():
            missing.append(tariff)

    if missing:
        return False, f"missing RL {ckpt_name} for tariffs={missing}"
    return True, f"RL artifacts present ({ckpt_name})"


def check_test_prerequisite(model: str, stage: str, variant: str) -> tuple[bool, str]:
    teacher_ok, teacher_note = _has_teacher_summary(model, stage)
    if not teacher_ok:
        return False, teacher_note

    if stage == "1-IL":
        return _has_il_artifacts(model)

    if stage == "2-RL":
        return _has_rl_artifacts(model, variant)

    return False, f"unknown stage={stage}"


def run_job(job: dict, python_exe: str, log_dir: Path, dry_run: bool, live_output: bool) -> dict:
    cmd = [python_exe, str(job["test_path"])]
    variant = str(job.get("variant") or "")
    if job["stage"] == "2-RL" and variant:
        cmd += ["--actor-variant", variant]

    variant_part = variant if variant else "default"
    log_file = log_dir / f"{job['model']}__{job['stage']}__{variant_part}__test.log"

    if dry_run:
        return {
            "model": job["model"],
            "stage": job["stage"],
            "variant": variant,
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
        "variant": variant,
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
        "variant",
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


def run_split_tests(
    machine: str,
    stage: str = "all",
    python_exe: str = sys.executable,
    dry_run: bool = False,
    stop_on_error: bool = False,
    resume: bool = True,
    live_output: bool = True,
    rl_variants: tuple[str, ...] = ("combo", "det", "stoch"),
) -> dict | None:
    if machine not in ASSIGNMENTS:
        raise ValueError(f"Unknown machine '{machine}'. Expected one of: {sorted(ASSIGNMENTS)}")
    if stage not in STAGE_MAP:
        raise ValueError(f"Unknown stage '{stage}'. Expected one of: {sorted(STAGE_MAP)}")

    bad_variants = [v for v in rl_variants if v not in RL_VARIANT_TO_CKPT]
    if bad_variants:
        raise ValueError(f"Unknown RL variants: {bad_variants}. Expected subset of {sorted(RL_VARIANT_TO_CKPT)}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = ANALYSIS_DIR / f"logs_test_machine_{machine}_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(machine, stage, rl_variants)
    if not jobs:
        print("No test jobs found for selection.")
        return None

    ckpt_path = checkpoint_path(machine, stage)
    prior_state = load_checkpoint(ckpt_path, machine, stage, jobs) if resume else None

    completed_results: list[dict] = []
    completed_keys: set[tuple[str, str, str]] = set()
    resumed_count = 0

    if prior_state:
        for row in prior_state["results"]:
            if row.get("status") in SUCCESS_STATUSES:
                key = (row.get("model"), row.get("stage"), row.get("variant") or "")
                if key not in completed_keys:
                    completed_keys.add(key)
                    completed_results.append(row)
        resumed_count = len(completed_results)

    pending_jobs = [job for job in jobs if job_key(job) not in completed_keys]

    print(
        f"Machine {machine} | stage={stage} | jobs={len(jobs)} | "
        f"live_output={live_output} | resume={resume} | rl_variants={','.join(rl_variants)}"
    )
    for idx, job in enumerate(jobs, start=1):
        suffix = f" variant={job['variant']}" if job.get("variant") else ""
        print(f"[{idx}/{len(jobs)}] {job['model']} {job['stage']}{suffix} -> {job['test_path']}")

    if resumed_count:
        print(f"[RESUME] Loaded {resumed_count} completed test job(s) from {ckpt_path}")
    if not pending_jobs:
        print("All test jobs for this selection are already complete in latest checkpoint.")
        print("Use --no-resume to run all tests again from scratch.")
        return None

    results: list[dict] = list(completed_results)
    save_checkpoint(
        ckpt_path,
        machine,
        stage,
        python_exe,
        jobs,
        results,
        finished=False,
    )

    t0 = time.perf_counter()

    for idx, job in enumerate(pending_jobs, start=1):
        variant = str(job.get("variant") or "")
        suffix = f" variant={variant}" if variant else ""
        print(f"[RUN {idx}/{len(pending_jobs)}] {job['model']} {job['stage']}{suffix}")

        ok, note = check_test_prerequisite(job["model"], job["stage"], variant)
        if not ok:
            out = {
                "model": job["model"],
                "stage": job["stage"],
                "variant": variant,
                "status": "blocked-prereq",
                "returncode": None,
                "elapsed_sec": 0.0,
                "command": [python_exe, str(job["test_path"])],
                "log_file": str(log_dir / f"{job['model']}__{job['stage']}__{variant or 'default'}__test.log"),
                "note": note,
            }
            results.append(out)
            print(f"[BLOCKED] {job['model']} {job['stage']}{suffix} reason={note}")
            save_checkpoint(
                ckpt_path,
                machine,
                stage,
                python_exe,
                jobs,
                results,
                finished=False,
            )
            if stop_on_error:
                print("Stopping on first blocked/error result (--stop-on-error).")
                break
            continue

        out = run_job(job, python_exe, log_dir, dry_run, live_output)
        results.append(out)
        print(
            f"[DONE] {job['model']} {job['stage']}{suffix} status={out['status']} "
            f"elapsed={out['elapsed_sec']:.1f}s rc={out['returncode']}"
        )

        save_checkpoint(
            ckpt_path,
            machine,
            stage,
            python_exe,
            jobs,
            results,
            finished=False,
        )

        if stop_on_error and out["status"] not in SUCCESS_STATUSES:
            print("Stopping on first blocked/error result (--stop-on-error).")
            break

    total_elapsed = time.perf_counter() - t0

    summary = {
        "machine": machine,
        "stage": stage,
        "python": python_exe,
        "run_id": run_id,
        "resumed_completed_count": resumed_count,
        "total_elapsed_sec": float(total_elapsed),
        "total_elapsed_min": float(total_elapsed / 60.0),
        "ok_count": int(sum(1 for r in results if r["status"] in SUCCESS_STATUSES)),
        "error_count": int(sum(1 for r in results if r["status"] not in SUCCESS_STATUSES)),
        "jobs": results,
    }

    json_path = ANALYSIS_DIR / f"test_split_machine_{machine}_{stage}_{run_id}.json"
    csv_path = ANALYSIS_DIR / f"test_split_machine_{machine}_{stage}_{run_id}.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, results)

    success_keys = {(r.get("model"), r.get("stage"), r.get("variant") or "") for r in results if r.get("status") in SUCCESS_STATUSES}
    all_complete = all(job_key(job) in success_keys for job in jobs)
    save_checkpoint(
        ckpt_path,
        machine,
        stage,
        python_exe,
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

    return summary


def _parse_rl_variants(raw: str) -> tuple[str, ...]:
    parts = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return ("combo", "det", "stoch")
    dedup: list[str] = []
    for p in parts:
        if p not in dedup:
            dedup.append(p)
    return tuple(dedup)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run split test jobs for machine A, B, or C")
    parser.add_argument("--machine", choices=["A", "B", "C"], required=True)
    parser.add_argument("--stage", choices=["all", "il", "rl"], default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from latest incomplete run for selected machine/stage (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore latest checkpoint and run all tests from scratch.",
    )
    parser.add_argument(
        "--live-output",
        dest="live_output",
        action="store_true",
        help="Stream child test.py output live to console and file (default).",
    )
    parser.add_argument(
        "--no-live-output",
        dest="live_output",
        action="store_false",
        help="Disable live output; write child output to log file only.",
    )
    parser.add_argument(
        "--rl-variants",
        default="combo,det,stoch",
        help="Comma-separated RL actor variants for 2-RL tests (e.g. combo or combo,det,stoch).",
    )
    parser.set_defaults(resume=True, live_output=True)
    args = parser.parse_args()

    run_split_tests(
        machine=args.machine,
        stage=args.stage,
        python_exe=args.python,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
        resume=args.resume,
        live_output=args.live_output,
        rl_variants=_parse_rl_variants(args.rl_variants),
    )


if __name__ == "__main__":
    main()
