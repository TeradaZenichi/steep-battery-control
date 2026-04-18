import gc
import inspect
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ARCHS = [
    "ATT",
    "ATTv2",
    "ATT_MEM",
    "ATT_MEMv2",
    "GRU",
    "GRUv2",
    "MLP",
    "MLPv2",
    "TCN",
    "TCNv2",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARAMETERS_PATH = str((ROOT / "data" / "parameters.json").resolve())


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _clear_conflicting_modules():
    for mod_name in ["model", "utils", "train", "hpo", "opt"]:
        sys.modules.pop(mod_name, None)


def _actor_output(actor, x):
    out = actor(x)
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, torch.Tensor) and out.dim() == 3:
        out = out[:, -1, :]
    return out


def _infer_history_len(il_cfg: dict) -> int:
    tr = il_cfg.get("training", {})
    if "history_len" in tr:
        try:
            return max(1, int(tr["history_len"]))
        except Exception:
            pass

    ss = il_cfg.get("optuna", {}).get("search_space", {})
    if "history_len" in ss:
        raw = ss["history_len"]
        if isinstance(raw, list) and raw:
            try:
                return max(1, int(raw[0]))
            except Exception:
                pass

    return 1


def _safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return float(default)


def benchmark_il(arch: str) -> dict:
    out = {
        "il_update_sec": float("nan"),
        "il_val_sec": float("nan"),
        "il_ok": False,
        "il_error": "",
        "il_history_len": 1,
    }
    try:
        model_path = ROOT / "models" / arch / "model.py"
        model_json = ROOT / "models" / arch / "model.json"
        il_cfg_path = ROOT / "models" / arch / "1-IL" / "config.json"

        model_mod = _load_module(model_path, f"bench_{arch}_model_il")
        model_cfg = json.loads(model_json.read_text(encoding="utf-8"))
        il_cfg = json.loads(il_cfg_path.read_text(encoding="utf-8"))

        actor_cfg = dict(model_cfg["actor"])
        actor_cfg["parameters"] = PARAMETERS_PATH

        hist = _infer_history_len(il_cfg)
        out["il_history_len"] = hist

        actor = model_mod.load_actor(actor_cfg, device=DEVICE)
        actor.train()

        batch = 64
        in_dim = int(actor_cfg["input_dim"])
        out_dim = int(actor_cfg.get("output_dim", 3))

        if hist > 1:
            xb = torch.randn(batch, hist, in_dim, device=DEVICE)
        else:
            xb = torch.randn(batch, in_dim, device=DEVICE)
        yb = torch.randn(batch, out_dim, device=DEVICE)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)

        _sync()
        t0 = time.perf_counter()
        pred = _actor_output(actor, xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _sync()
        t1 = time.perf_counter()

        actor.eval()
        _sync()
        t2 = time.perf_counter()
        with torch.no_grad():
            pred_val = _actor_output(actor, xb)
            _ = criterion(pred_val, yb)
        _sync()
        t3 = time.perf_counter()

        out["il_update_sec"] = float(t1 - t0)
        out["il_val_sec"] = float(t3 - t2)
        out["il_ok"] = True

        del actor, xb, yb
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        out["il_error"] = f"{type(exc).__name__}: {exc}"

    return out


def benchmark_rl(arch: str) -> dict:
    out = {
        "rl_collect_step_sec": float("nan"),
        "rl_update_sec": float("nan"),
        "rl_val_det_1run_1day_sec": float("nan"),
        "rl_val_stoch_1run_1day_sec": float("nan"),
        "rl_est_per_tariff_hours": float("nan"),
        "rl_est_all_tariffs_hours": float("nan"),
        "rl_ok": False,
        "rl_error": "",
        "rl_history_len": None,
    }

    trainer = None
    try:
        _clear_conflicting_modules()
        train_path = ROOT / "models" / arch / "2-RL" / "train.py"
        train_mod = _load_module(train_path, f"bench_{arch}_train_rl")

        trainer = train_mod.Train("tar_s")

        orig_episode_length = int(trainer.episode_length)
        env_count = len(getattr(trainer, "envs", {})) or 1
        orig_update_steps = int(getattr(trainer.hp, "update_steps", 1))
        orig_update_every = int(getattr(trainer, "update_every_steps", 1))
        orig_batch_size = int(getattr(trainer.hp, "batch_size", 8))
        orig_warmup_episodes = int(getattr(trainer.hp, "warmup_episodes", 0))
        orig_train_episodes = int(getattr(trainer.hp, "train_episodes", 0))
        orig_eval_every = int(getattr(trainer.hp, "eval_every", 1))
        orig_eval_workers = int(getattr(trainer, "eval_workers", 1))
        orig_val_runs = len(trainer.train_cfg.get("val", []))
        orig_days = int(getattr(trainer.hp, "days", 1))

        out["rl_history_len"] = int(getattr(trainer, "history_len", trainer.train_cfg.get("train", {}).get("history_len", 1)))

        # 1) Collect timing (no updates)
        trainer.episode_length = 4
        trainer.train_env_workers = 1
        trainer.log_every_steps = 10**9
        trainer.hp.update_steps = 0
        trainer.update_every_steps = max(1, orig_update_every)
        trainer.hp.batch_size = max(32, orig_batch_size)

        _sync()
        t0 = time.perf_counter()
        _, steps = trainer._collect_training_episode(0)
        _sync()
        t1 = time.perf_counter()

        steps = max(1, int(steps))
        per_step = (t1 - t0) / steps
        out["rl_collect_step_sec"] = float(per_step)

        # 2) Ensure buffer and time one update
        trainer.hp.update_steps = orig_update_steps
        trainer.hp.batch_size = min(8, orig_batch_size)
        trainer.episode_length = 8
        trainer.hp.warmup_episodes = 1
        if hasattr(trainer.buffer, "n_step"):
            try:
                trainer.buffer.n_step = 1
            except Exception:
                pass

        trainer._run_warmup()

        if int(getattr(trainer.buffer, "size", 0)) < int(trainer.hp.batch_size):
            trainer.hp.update_steps = 0
            trainer._collect_training_episode(0)
            trainer.hp.update_steps = orig_update_steps

        if int(getattr(trainer.buffer, "size", 0)) < int(trainer.hp.batch_size):
            raise RuntimeError("Buffer not filled enough for one update")

        _sync()
        t2 = time.perf_counter()
        trainer.update()
        _sync()
        t3 = time.perf_counter()
        update_sec = t3 - t2
        out["rl_update_sec"] = float(update_sec)

        # 3) One-run validation timing (forced 1 day), deterministic + stochastic
        val_runs = trainer.train_cfg.get("val", [])
        if not val_runs:
            raise RuntimeError("No validation runs configured")

        run0 = dict(val_runs[0])
        original_run_days = int(run0.get("days", 1))
        run0["days"] = 1

        day_episode_length = max(1, int(round(orig_episode_length / max(1, orig_days))))

        actor_state_cpu = {k: v.detach().cpu() for k, v in trainer.actor.state_dict().items()}
        eval_worker = train_mod._eval_worker
        sig = inspect.signature(eval_worker)

        common_args = [
            run0,
            trainer.parameters,
            trainer.tariff,
            trainer.actor_cfg,
            actor_state_cpu,
            day_episode_length,
        ]

        if "history_len" in sig.parameters:
            history_len = int(getattr(trainer, "history_len", 1))
            det_args = common_args + [history_len, True]
            stoch_args = common_args + [history_len, False]
        else:
            det_args = common_args + [True]
            stoch_args = common_args + [False]

        _sync()
        t4 = time.perf_counter()
        _ = eval_worker(*det_args)
        _sync()
        t5 = time.perf_counter()

        _sync()
        t6 = time.perf_counter()
        _ = eval_worker(*stoch_args)
        _sync()
        t7 = time.perf_counter()

        det_1day = t5 - t4
        stoch_1day = t7 - t6

        out["rl_val_det_1run_1day_sec"] = float(det_1day)
        out["rl_val_stoch_1run_1day_sec"] = float(stoch_1day)

        # 4) Coarse full-train estimate (based on measured primitives)
        episode_steps_train = orig_episode_length * env_count
        collect_per_episode = per_step * episode_steps_train

        updates_per_episode = (episode_steps_train // max(1, orig_update_every)) * max(1, orig_update_steps)
        update_per_episode = updates_per_episode * update_sec

        warmup_total = per_step * orig_episode_length * max(0, orig_warmup_episodes)

        # Scale 1-day validation to configured days/run and total val runs, adjusted by eval workers.
        det_per_run = det_1day * max(1, original_run_days)
        stoch_per_run = stoch_1day * max(1, original_run_days)
        eval_event = (orig_val_runs / max(1, orig_eval_workers)) * (det_per_run + stoch_per_run)

        eval_events = ((max(1, orig_train_episodes) - 1) // max(1, orig_eval_every)) + 1

        est_per_tariff_sec = warmup_total + (orig_train_episodes * (collect_per_episode + update_per_episode)) + (eval_events * eval_event)
        est_all_tariffs_sec = est_per_tariff_sec * 5

        out["rl_est_per_tariff_hours"] = float(est_per_tariff_sec / 3600.0)
        out["rl_est_all_tariffs_hours"] = float(est_all_tariffs_sec / 3600.0)
        out["rl_ok"] = True

    except Exception as exc:
        out["rl_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            del trainer
        except Exception:
            pass
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        _clear_conflicting_modules()

    return out


def _fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return "-"
        if abs(x) >= 100:
            return f"{x:.1f}"
        if abs(x) >= 10:
            return f"{x:.2f}"
        return f"{x:.3f}"
    return str(x)


def main():
    print(f"[bench] device={DEVICE}")
    rows = []

    for arch in ARCHS:
        print(f"[bench] {arch} ...")
        il = benchmark_il(arch)
        rl = benchmark_rl(arch)

        row = {
            "arch": arch,
            **il,
            **rl,
        }
        rows.append(row)

    out_path = ROOT / "Results" / "tmp" / "benchmark_il_rl_one_update_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== Benchmark (1 update + validation) ===")
    header = [
        "arch",
        "il_update_s",
        "il_val_s",
        "rl_update_s",
        "rl_val_det_1run_1day_s",
        "rl_val_stoch_1run_1day_s",
        "rl_est_per_tariff_h",
        "rl_est_all_tariffs_h",
    ]
    print("\t".join(header))
    for r in rows:
        print("\t".join([
            r["arch"],
            _fmt(r["il_update_sec"]),
            _fmt(r["il_val_sec"]),
            _fmt(r["rl_update_sec"]),
            _fmt(r["rl_val_det_1run_1day_sec"]),
            _fmt(r["rl_val_stoch_1run_1day_sec"]),
            _fmt(r["rl_est_per_tariff_hours"]),
            _fmt(r["rl_est_all_tariffs_hours"]),
        ]))

    ok_rows = [r for r in rows if r.get("rl_ok")]
    if ok_rows:
        total_all_arch_h = sum(r.get("rl_est_all_tariffs_hours", 0.0) for r in ok_rows)
        avg_all_arch_h = total_all_arch_h / len(ok_rows)
        print(f"\n[bench] RL total estimate across {len(ok_rows)} architectures: {total_all_arch_h:.2f} h ({total_all_arch_h/24.0:.2f} days)")
        print(f"[bench] RL average estimate per architecture (all 5 tariffs): {avg_all_arch_h:.2f} h")

    failed = [r for r in rows if (not r.get("il_ok")) or (not r.get("rl_ok"))]
    if failed:
        print("\n=== Failures ===")
        for r in failed:
            if not r.get("il_ok"):
                print(f"[IL] {r['arch']}: {r.get('il_error','')} ")
            if not r.get("rl_ok"):
                print(f"[RL] {r['arch']}: {r.get('rl_error','')} ")

    print(f"\n[bench] saved: {out_path}")


if __name__ == "__main__":
    main()
