"""Annual test protocol for rule-based baselines."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment import SmartHomeEnv
from models._physics import Battery, EV
from reinforcement.sac_penalty.utils.safe_eval import _run_episode, summarize_safe
from reinforcement.sac_penalty.utils.safety import SafetyLayer
from reinforcement.sac_penalty.utils.test_eval import _default_monthly_runs


TARIFFS = ["tar_s", "tar_w", "tar_sw", "tar_flat", "tar_tou"]
SAC_CONFIG = PROJECT_ROOT / "reinforcement" / "sac_penalty" / "GRU" / "config.json"
_DF_CACHE: dict[str, pd.DataFrame] = {}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def eval_baseline_runs(runs: list[dict], actor, params: dict, tariff: str, history_len: int):
    dt = float(params["general"]["timestep"]) / 60.0
    safety = SafetyLayer(Battery(params["BESS"], dt), EV(params["EV"], dt), params).eval()
    actor.eval()
    results = []
    for run in runs:
        ds = Path(run["dataset"])
        if not ds.is_absolute():
            ds = PROJECT_ROOT / ds
        key = str(ds.resolve())
        df = _DF_CACHE.get(key)
        if df is None:
            df = pd.read_csv(ds, sep=";", parse_dates=["timestamp"], dayfirst=True, index_col="timestamp")
            _DF_CACHE[key] = df
        env = SmartHomeEnv(
            df,
            params,
            start=pd.to_datetime(run["date"], format="%Y-%m-%d %H:%M:%S"),
            days=int(run["days"]),
            BESS_SoC=float(run.get("soc", 0.5)),
            tariff=tariff,
            track_operation=False,
        )
        with torch.inference_mode():
            result = _run_episode(env, actor, safety, history_len, deterministic=True)
        result["scenario"] = run.get("name", "")
        results.append(result)
    return results


def run_baseline_eval(name: str, load_actor) -> None:
    cfg = json.load(open(SAC_CONFIG, encoding="utf-8"))
    params = json.load(open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8"))
    history_len = int(cfg["train"].get("history_len", 96))
    out_dir = PROJECT_ROOT / "Results" / "eval" / "models" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    actor = load_actor({})

    for tariff in TARIFFS:
        print(f"\n=== {tariff} ===")
        payload = {"summary": {}, "per_run": {}}
        for split in ("val", "test"):
            runs = cfg.get(split, [])
            if not runs:
                payload["summary"][split] = None
                payload["per_run"][split] = []
                continue
            results = eval_baseline_runs(runs, actor, params, tariff, history_len)
            summary = summarize_safe(results)
            print(f"  {split}: mean={summary['mean_reward']:.2f} worst={summary['worst_reward']:.2f}")
            rows.append({"tariff": tariff, "split": split, **summary})
            payload["summary"][split] = summary
            payload["per_run"][split] = results
        with open(out_dir / f"{name.lower()}_{tariff}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    _write_csv(out_dir / f"{name.lower()}_summary.csv", rows)
    print(f"\nWrote {out_dir / f'{name.lower()}_summary.csv'}")


def run_baseline_test(name: str, actor_module: str, load_actor) -> None:
    cfg = json.load(open(SAC_CONFIG, encoding="utf-8"))
    params = json.load(open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8"))
    history_len = int(cfg["train"].get("history_len", 96))
    tariffs = [t.strip() for t in os.environ.get("RUN_TARIFFS", ",".join(TARIFFS)).split(",") if t.strip()]
    runs = _default_monthly_runs()
    actor = load_actor({})

    for tariff in tariffs:
        out_dir = PROJECT_ROOT / "paper" / "test" / "baselines" / name / tariff
        out_dir.mkdir(parents=True, exist_ok=True)
        results = eval_baseline_runs(runs, actor, params, tariff, history_len)

        monthly_rows = []
        for run, result in zip(runs, results):
            monthly_rows.append({
                "architecture": name,
                "tariff": tariff,
                "checkpoint": "rule",
                "mode": "safe",
                "scenario": run["name"],
                **result,
            })
        summary = {
            "architecture": name,
            "tariff": tariff,
            "checkpoint": "rule",
            "mode": "safe",
            **summarize_safe(results),
        }
        summary["months"] = len(results)

        _write_csv(out_dir / "summary_monthly.csv", monthly_rows)
        _write_csv(out_dir / "summary_by_checkpoint.csv", [summary])
        with open(out_dir / "summary_overall.json", "w", encoding="utf-8") as f:
            json.dump([summary], f, indent=2)
        (out_dir / ".test_done").touch()
        print(f"{name} {tariff}: mean={summary['mean_reward']:.2f} worst={summary['worst_reward']:.2f}")
