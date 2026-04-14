from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import traceback
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def call_eval_worker(mod, trainer, run_cfg: dict, actor_state: dict, deterministic: bool) -> float:
    fn = mod._eval_worker
    sig = inspect.signature(fn)

    if "history_len" in sig.parameters:
        return float(
            fn(
                run_cfg,
                trainer.parameters,
                trainer.tariff,
                trainer.actor_cfg,
                actor_state,
                48,
                int(getattr(trainer, "history_len", 1)),
                deterministic,
            )
        )

    return float(
        fn(
            run_cfg,
            trainer.parameters,
            trainer.tariff,
            trainer.actor_cfg,
            actor_state,
            48,
            deterministic,
        )
    )


def run_case(model_name: str, tariff: str, force_batch_size: int, force_update_every_steps: int) -> dict:
    train_path = ROOT / "models" / model_name / "2-RL" / "train.py"
    result = {"model": model_name, "tariff": tariff}

    try:
        mod = load_module(f"deep_single_{model_name}", train_path)
        trainer = mod.Train(tariff)

        trainer.hp.warmup_episodes = 1
        trainer.hp.train_episodes = 3
        trainer.hp.eval_every = 1
        trainer.episode_length = 48
        trainer.eval_workers = 2
        trainer.train_env_workers = 1
        trainer.hp.batch_size = int(force_batch_size)
        trainer.update_every_steps = int(force_update_every_steps)
        trainer.audit_every_episodes = 1
        trainer.log_every_steps = 1_000_000
        trainer.early_stop_patience = 0
        trainer.min_episodes_before_early_stop = 10_000

        if isinstance(trainer.train_cfg.get("val"), list) and trainer.train_cfg["val"]:
            trainer.train_cfg["val"] = trainer.train_cfg["val"][:2]

        trainer.train()

        expected = {
            "best_actor_combo": trainer.best_actor_path,
            "best_actor_det": trainer.best_actor_det_path,
            "best_actor_stoch": trainer.best_actor_stoch_path,
            "best_meta_combo": trainer.best_meta_path,
            "best_meta_det": trainer.best_meta_det_path,
            "best_meta_stoch": trainer.best_meta_stoch_path,
            "audit_csv": trainer.audit_csv,
        }
        missing_artifacts = [name for name, path in expected.items() if not Path(path).exists()]

        audit_df = pd.read_csv(trainer.audit_csv)
        required_audit_cols = [
            "no_improve_evals_det",
            "no_improve_evals_stoch",
            "no_improve_evals_combo",
            "iteration_time_sec",
            "checkpoint_score",
            "eval_reward_det",
            "eval_reward_stoch",
        ]
        missing_audit_cols = [c for c in required_audit_cols if c not in audit_df.columns]

        latest = audit_df.iloc[-1].to_dict() if not audit_df.empty else {}

        variant_rewards = {}
        run_cfg = trainer.train_cfg["val"][0] if trainer.train_cfg.get("val") else None
        if run_cfg is not None:
            for label, ckpt_path in {
                "combo": trainer.best_actor_path,
                "det": trainer.best_actor_det_path,
                "stoch": trainer.best_actor_stoch_path,
            }.items():
                ckpt_path = Path(ckpt_path)
                if not ckpt_path.exists():
                    variant_rewards[label] = {"error": "checkpoint_missing"}
                    continue

                actor_state = torch.load(ckpt_path, map_location=torch.device("cpu"))
                det_reward = call_eval_worker(mod, trainer, run_cfg, actor_state, deterministic=True)
                stoch_reward = call_eval_worker(mod, trainer, run_cfg, actor_state, deterministic=False)
                variant_rewards[label] = {
                    "det_reward": det_reward,
                    "stoch_reward": stoch_reward,
                }

        result.update(
            {
                "status": "ok",
                "forced_settings": {
                    "batch_size": int(force_batch_size),
                    "update_every_steps": int(force_update_every_steps),
                },
                "artifacts_missing": missing_artifacts,
                "audit_missing_columns": missing_audit_cols,
                "audit_rows": int(len(audit_df)),
                "latest_metrics": {
                    "checkpoint_score": float(latest.get("checkpoint_score")) if "checkpoint_score" in latest else None,
                    "eval_reward_det": float(latest.get("eval_reward_det")) if "eval_reward_det" in latest else None,
                    "eval_reward_stoch": float(latest.get("eval_reward_stoch")) if "eval_reward_stoch" in latest else None,
                    "lambda": float(latest.get("lambda")) if "lambda" in latest else None,
                    "frac_violation": float(latest.get("frac_violation")) if "frac_violation" in latest else None,
                    "n_updates": int(latest.get("n_updates")) if "n_updates" in latest else None,
                },
                "had_backward": bool(int(latest.get("n_updates", 0)) > 0),
                "variant_rewards": variant_rewards,
            }
        )

    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tariff", default="tar_s")
    parser.add_argument("--force-batch-size", type=int, default=64)
    parser.add_argument("--force-update-every-steps", type=int, default=1)
    parser.add_argument("--out", default=str(ROOT / "Results" / "analysis" / "deep_real_eval_single.json"))
    args = parser.parse_args()

    result = run_case(args.model, args.tariff, args.force_batch_size, args.force_update_every_steps)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
