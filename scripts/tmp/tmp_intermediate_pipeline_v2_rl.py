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
from collections import deque
from datetime import datetime
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path(r"{ROOT}")
train_path = ROOT / "models" / "{model_name}" / "2-RL" / "train.py"
config_path = ROOT / "models" / "{model_name}" / "2-RL" / "config.json"

spec = importlib.util.spec_from_file_location("{model_name}_train_mod", train_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

trainer = mod.Train("{TARIFF}")

# Intermediate run settings
trainer.hp.batch_size = 128
trainer.hp.warmup_episodes = 1
trainer.hp.train_episodes = 10
trainer.hp.eval_every = 2
trainer.episode_length = 96  # 1 day at 15-minute resolution
trainer.eval_workers = 1
trainer.train_env_workers = 1
trainer.audit_every_episodes = 1
trainer.log_every_steps = 10_000
trainer.early_stop_patience = 0

trainer.train()

audit = pd.read_csv(trainer.audit_csv)
last = audit.iloc[-1].to_dict()

# ---- one full test scenario (first entry in config['test']) ----
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
scenario = cfg["test"][0]  # usually test_cy_01

start = datetime.strptime(scenario["date"], "%Y-%m-%d %H:%M:%S")
days = int(scenario["days"])
soc = float(scenario["soc"])

df = pd.read_csv(
    ROOT / scenario["dataset"],
    sep=";",
    parse_dates=["timestamp"],
    dayfirst=True,
    index_col="timestamp",
)

device = torch.device("cpu")
actor = mod.load_actor(trainer.actor_cfg, device=device)
state_dict = torch.load(trainer.best_actor_path, map_location=device)
actor.load_state_dict(state_dict, strict=True)
actor.eval()

env = mod.SmartHomeEnv(
    df,
    trainer.parameters,
    start=start,
    days=days,
    BESS_SoC=soc,
    tariff=trainer.tariff,
)

obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]
if isinstance(obs, dict):
    obs = obs.get("obs", obs.get("observation", next(iter(obs.values()))))
obs = np.asarray(obs, dtype=np.float32).reshape(-1)

history_len = int(getattr(trainer, "history_len", 1))
history = deque([obs.copy() for _ in range(max(1, history_len))], maxlen=max(1, history_len))

max_steps = int((24 * 60 * days) / float(trainer.parameters["general"]["timestep"]))

done = False
truncated = False
scenario_reward = 0.0
steps = 0

while (not done) and (not truncated) and steps < max_steps:
    if history_len > 1:
        obs_in = np.stack(history, axis=0)
        obs_t = torch.as_tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        obs_t = torch.as_tensor(history[-1], dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        _, _, action_t, _ = actor.sample(obs_t)
    action = action_t.squeeze(0).cpu().numpy()

    next_obs, rew, done, truncated, info = env.step(action)

    if isinstance(next_obs, tuple):
        next_obs = next_obs[0]
    if isinstance(next_obs, dict):
        next_obs = next_obs.get("obs", next_obs.get("observation", next(iter(next_obs.values()))))
    next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
    history.append(next_obs.copy())

    scenario_reward += float(rew)
    steps += 1

op = env.operation.copy()

dt_h = float(trainer.parameters["general"]["timestep"]) / 60.0

chi_col = "χPV" if "χPV" in op.columns else ("chi_pv" if "chi_pv" in op.columns else None)
if chi_col is not None and "PPV" in op.columns:
    chi = op[chi_col].astype(float)
    ppv = op["PPV"].astype(float)
    denom = (1.0 - chi).replace(0.0, np.nan)
    pv_avail = (ppv / denom).fillna(ppv)
    pv_curtail_kwh = float(((pv_avail - ppv).clip(lower=0.0) * dt_h).sum())
    mean_chi = float(chi.mean())
else:
    pv_curtail_kwh = None
    mean_chi = None

if "PBESS" in op.columns:
    pb = op["PBESS"].astype(float)
    bess_pos_kwh = float((pb.clip(lower=0.0) * dt_h).sum())
    bess_neg_kwh = float(((-pb).clip(lower=0.0) * dt_h).sum())
else:
    bess_pos_kwh = None
    bess_neg_kwh = None

result = {{
    "model": "{model_name}",
    "tariff": "{TARIFF}",
    "status": "ok",
    "train": {{
        "last_episode": float(last.get("episode", 0.0)),
        "train_reward_total": None if pd.isna(last.get("train_reward_total")) else float(last.get("train_reward_total")),
        "eval_reward_det": None if pd.isna(last.get("eval_reward_det")) else float(last.get("eval_reward_det")),
        "eval_reward_stoch": None if pd.isna(last.get("eval_reward_stoch")) else float(last.get("eval_reward_stoch")),
        "checkpoint_score": None if pd.isna(last.get("checkpoint_score")) else float(last.get("checkpoint_score")),
        "alpha": None if pd.isna(last.get("alpha")) else float(last.get("alpha")),
        "lambda": None if pd.isna(last.get("lambda")) else float(last.get("lambda")),
        "cost_mean": None if pd.isna(last.get("cost_mean")) else float(last.get("cost_mean")),
        "frac_violation": None if pd.isna(last.get("frac_violation")) else float(last.get("frac_violation")),
        "n_updates": None if pd.isna(last.get("n_updates")) else int(last.get("n_updates")),
        "buffer_size": None if pd.isna(last.get("buffer_size")) else int(last.get("buffer_size")),
    }},
    "test_one_scenario": {{
        "scenario_name": scenario["name"],
        "days": days,
        "steps": steps,
        "actor_reward": float(scenario_reward),
        "mean_chiPV": mean_chi,
        "pv_curtail_kwh": pv_curtail_kwh,
        "bess_pos_kwh": bess_pos_kwh,
        "bess_neg_kwh": bess_neg_kwh,
    }},
}}

print("__RESULT__" + json.dumps(result, ensure_ascii=False))
'''


def run_model(model_name: str) -> dict:
    code = build_child_code(model_name)
    proc = subprocess.run(
        [PYTHON, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=7200,
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
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-80:]),
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-80:]),
        }

    if result_line is None:
        return {
            "model": model_name,
            "tariff": TARIFF,
            "status": "error",
            "returncode": proc.returncode,
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-80:]),
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-80:]),
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

    out_json = ROOT / "Results" / "analysis" / "intermediate_pipeline_v2_rl_summary.json"
    out_csv = ROOT / "Results" / "analysis" / "intermediate_pipeline_v2_rl_summary.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # flat CSV for quick view
    rows = []
    for r in results:
        if r.get("status") != "ok":
            rows.append({"model": r.get("model"), "status": r.get("status"), "returncode": r.get("returncode")})
            continue
        row = {
            "model": r["model"],
            "tariff": r["tariff"],
            "status": r["status"],
            "train_checkpoint_score": r["train"]["checkpoint_score"],
            "train_eval_reward_det": r["train"]["eval_reward_det"],
            "train_lambda": r["train"]["lambda"],
            "train_frac_violation": r["train"]["frac_violation"],
            "train_n_updates": r["train"]["n_updates"],
            "test_scenario": r["test_one_scenario"]["scenario_name"],
            "test_actor_reward": r["test_one_scenario"]["actor_reward"],
            "test_mean_chiPV": r["test_one_scenario"]["mean_chiPV"],
            "test_pv_curtail_kwh": r["test_one_scenario"]["pv_curtail_kwh"],
            "test_bess_pos_kwh": r["test_one_scenario"]["bess_pos_kwh"],
            "test_bess_neg_kwh": r["test_one_scenario"]["bess_neg_kwh"],
        }
        rows.append(row)

    import pandas as pd

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[SAVED] {out_json}")
    print(f"[SAVED] {out_csv}")


if __name__ == "__main__":
    main()
