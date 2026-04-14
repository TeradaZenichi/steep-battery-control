from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
TARIFF = "tar_s"
FORCE_BATCH_SIZE = 64
FORCE_UPDATE_EVERY_STEPS = 1
MODELS = [
    "ATT",
    "ATT_MEM",
    "GRU",
    "MLP",
    "TCN",
    "ATTv2",
    "ATT_MEMv2",
    "GRUv2",
    "MLPv2",
    "TCNv2",
]


def run_model(model: str) -> dict:
    out_file = ROOT / "Results" / "analysis" / f"deep_single_{model}.json"
    cmd = [
        str(PY),
        str(THIS_DIR / "tmp_deep_real_eval_single.py"),
        "--model",
        model,
        "--tariff",
        TARIFF,
        "--force-batch-size",
        str(FORCE_BATCH_SIZE),
        "--force-update-every-steps",
        str(FORCE_UPDATE_EVERY_STEPS),
        "--out",
        str(out_file),
    ]

    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - start

    result = {
        "model": model,
        "tariff": TARIFF,
        "elapsed_sec": elapsed,
        "returncode": proc.returncode,
    }

    if proc.returncode != 0:
        result["status"] = "error"
        result["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-80:])
        result["stdout_tail"] = "\n".join(proc.stdout.splitlines()[-80:])
        return result

    if not out_file.exists():
        result["status"] = "error"
        result["error"] = "missing_output_file"
        result["stdout_tail"] = "\n".join(proc.stdout.splitlines()[-80:])
        return result

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    result["status"] = payload.get("status", "unknown")
    result["payload"] = payload
    return result


def main() -> None:
    summary = {
        "tariff": TARIFF,
        "forced_settings": {
            "batch_size": FORCE_BATCH_SIZE,
            "update_every_steps": FORCE_UPDATE_EVERY_STEPS,
        },
        "started_at_epoch": time.time(),
        "runs": [],
    }

    t0 = time.perf_counter()
    for idx, model in enumerate(MODELS, start=1):
        print(f"[RUN {idx}/{len(MODELS)}] {model}")
        one = run_model(model)
        summary["runs"].append(one)
        print(
            f"[DONE] {model} status={one.get('status')} elapsed_sec={one.get('elapsed_sec'):.2f} rc={one.get('returncode')}"
        )

    total_elapsed = time.perf_counter() - t0
    summary["total_elapsed_sec"] = total_elapsed
    summary["total_elapsed_min"] = total_elapsed / 60.0

    ok_count = sum(1 for r in summary["runs"] if r.get("status") == "ok")
    summary["ok_count"] = ok_count
    summary["error_count"] = len(summary["runs"]) - ok_count

    out = ROOT / "Results" / "analysis" / "deep_real_eval_all10_forced_backward_timing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVED] {out}")
    print(f"[TOTAL] {total_elapsed:.2f}s ({total_elapsed/60.0:.2f} min), ok={ok_count}/{len(MODELS)}")


if __name__ == "__main__":
    main()
