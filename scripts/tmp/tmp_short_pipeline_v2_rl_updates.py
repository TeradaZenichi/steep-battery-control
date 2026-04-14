from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
MODELS = ["ATTv2", "ATT_MEMv2", "GRUv2", "MLPv2", "TCNv2"]
TARIFF = "tar_tou"


def build_child_code(model_name: str) -> str:
    return f'''
from pathlib import Path
import importlib.util
import json
import pandas as pd

ROOT = Path(r"{ROOT}")
train_path = ROOT / "models" / "{model_name}" / "2-RL" / "train.py"

spec = importlib.util.spec_from_file_location("{model_name}_train_mod", train_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

trainer = mod.Train("{TARIFF}")

# Short but with enough data for updates
trainer.hp.batch_size = 64
trainer.hp.warmup_episodes = 1
trainer.hp.train_episodes = 1
trainer.hp.eval_every = 1
trainer.episode_length = 96
trainer.eval_workers = 1
trainer.train_env_workers = 1
trainer.audit_every_episodes = 1
trainer.log_every_steps = 10_000

trainer.train()

audit = pd.read_csv(trainer.audit_csv)
last = audit.iloc[-1].to_dict()

keys = [
    "episode",
    "train_reward_total",
    "eval_reward_det",
    "eval_reward_stoch",
    "checkpoint_score",
    "alpha",
    "lambda",
    "cost_mean",
    "cost_p95",
    "frac_violation",
    "critic_loss",
    "actor_loss",
    "n_updates",
    "buffer_size",
]

out = {{"model": "{model_name}", "tariff": "{TARIFF}", "status": "ok"}}
for k in keys:
    v = last.get(k)
    if pd.isna(v):
        out[k] = None
    elif isinstance(v, (int, float)):
        out[k] = float(v)
    else:
        out[k] = v

print("__RESULT__" + json.dumps(out, ensure_ascii=False))
'''


def run_model(model_name: str) -> dict:
    code = build_child_code(model_name)
    proc = subprocess.run(
        [PYTHON, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    result_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            result_line = line[len("__RESULT__") :]

    if proc.returncode != 0:
        return {
            "model": model_name,
            "tariff": TARIFF,
            "status": "error",
            "returncode": proc.returncode,
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-50:]),
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-50:]),
        }

    if result_line is None:
        return {
            "model": model_name,
            "tariff": TARIFF,
            "status": "error",
            "returncode": proc.returncode,
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-50:]),
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-50:]),
            "reason": "No __RESULT__ line found",
        }

    return json.loads(result_line)


def main() -> None:
    results = []
    for model in MODELS:
        print(f"[RUN] {model} ({TARIFF})")
        res = run_model(model)
        results.append(res)
        print(json.dumps(res, ensure_ascii=False))

    out_path = ROOT / "Results" / "analysis" / "short_pipeline_v2_rl_updates_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
