"""Evaluate the RBS controller with SafetyLayer projection by default.

Run:
    python models/RBS/eval.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.RBS.model import load_actor
from reinforcement.sac_penalty.utils.safe_eval import SafeEvalRunner, summarize_safe

TARIFFS = ["tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat"]
SAC_CONFIG = PROJECT_ROOT / "reinforcement" / "sac_penalty" / "GRU" / "config.json"


def eval_split(runner, scenarios, actor_state):
    if not scenarios:
        return None, []
    futures = runner.submit(scenarios, actor_state, deterministic=True)
    results = [f.result() for f in futures]
    return summarize_safe(results), results


def main():
    cfg = json.load(open(SAC_CONFIG, encoding="utf-8"))
    params = json.load(open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8"))
    history_len = int(cfg["train"]["history_len"])

    out_dir = PROJECT_ROOT / "Results" / "eval" / "models" / "RBS"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    actor_cfg = {}
    actor_state = load_actor(actor_cfg).state_dict()

    for tariff in TARIFFS:
        print(f"\n=== {tariff} ===")
        runner = SafeEvalRunner(
            actor_module="models.RBS.model",
            actor_cfg=actor_cfg,
            parameters=params,
            tariff=tariff,
            history_len=history_len,
            project_root=str(PROJECT_ROOT),
            n_workers=cfg["train"].get("eval_workers", 4),
        )
        try:
            val_summary, val_runs = eval_split(runner, cfg.get("val", []), actor_state)
            test_summary, test_runs = eval_split(runner, cfg.get("test", []), actor_state)
        finally:
            runner.close()

        if val_summary is not None:
            print(f"  val:  mean={val_summary['mean_reward']:.2f}  "
                  f"worst={val_summary['worst_reward']:.2f}  "
                  f"grid={val_summary['grid_penalty_mean']:.5f}")
            rows.append({"tariff": tariff, "split": "val", **val_summary})
        if test_summary is not None:
            print(f"  test: mean={test_summary['mean_reward']:.2f}  "
                  f"worst={test_summary['worst_reward']:.2f}  "
                  f"grid={test_summary['grid_penalty_mean']:.5f}")
            rows.append({"tariff": tariff, "split": "test", **test_summary})

        with open(out_dir / f"rbs_{tariff}.json", "w") as f:
            json.dump({"summary": {"val": val_summary, "test": test_summary},
                       "per_run": {"val": val_runs, "test": test_runs}}, f, indent=2)

    pd.DataFrame(rows).to_csv(out_dir / "rbs_summary.csv", index=False)
    print(f"\nWrote {out_dir / 'rbs_summary.csv'}")


if __name__ == "__main__":
    main()
