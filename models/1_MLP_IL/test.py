from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import pyomo.environ as pyo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from env.environment import SmartHomeEnv  # type: ignore
from hp import HyperParameters  # type: ignore
from model import Actor  # type: ignore
from opt import Teacher  # type: ignore

CONFIG_PATH = Path("data/parameters.json")
RESULTS_ROOT = Path("Results")
MODEL_SUBDIR = "1_MLP_IL"
RUN_CONFIG_PATH = Path(__file__).with_name("run_config.json")

SUMMARY_NAME = "evaluation_summary.json"
SUMMARY_TEXT_NAME = "evaluation_summary_lines.json"


def resolve_project_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataframe(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep=";")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    return df


def get_env_feature_names(cfg: dict, df: pd.DataFrame, start_date: str, days: int) -> list[str]:
    env = SmartHomeEnv(cfg, dataframe=df, start_date=start_date, days=days, state_mask=None)
    obs, _ = env.reset()
    if hasattr(env, "state_feature_labels"):
        return list(getattr(env, "state_feature_labels"))
    if hasattr(env, "_feature_names"):
        return list(getattr(env, "_feature_names"))
    return [f"f_{i}" for i in range(len(obs))]


def resolve_state_mask(mask_payload: Any, feature_names: list[str]) -> np.ndarray | None:
    if mask_payload is None:
        return None
    if isinstance(mask_payload, (list, tuple, np.ndarray)):
        m = np.array(mask_payload, dtype=bool)
        return m if m.size == len(feature_names) else None
    if isinstance(mask_payload, dict):
        # legacy schema: {"mask": [...], "feature_names": [...]}
        m = mask_payload.get("mask")
        fn = mask_payload.get("feature_names")
        if m is not None:
            m = np.array(m, dtype=bool)
            if fn is None:
                return m if m.size == len(feature_names) else None
            if isinstance(fn, (list, tuple)) and len(fn) == len(m):
                idx = {str(n): i for i, n in enumerate(fn)}
                out = np.zeros(len(feature_names), dtype=bool)
                for j, name in enumerate(feature_names):
                    if name in idx:
                        out[j] = bool(m[idx[name]])
                return out

        # IL schema: {"vector": [...], "labels": [...]}
        v = mask_payload.get("vector")
        lab = mask_payload.get("labels")
        if v is not None:
            v = np.array(v, dtype=bool)
            if lab is not None and isinstance(lab, (list, tuple)) and len(lab) == len(v):
                idx = {str(n): i for i, n in enumerate(lab)}
                out = np.zeros(len(feature_names), dtype=bool)
                for j, name in enumerate(feature_names):
                    if name in idx:
                        out[j] = bool(v[idx[name]])
                return out
            return v if v.size == len(feature_names) else None
    return None


def build_action_log(env: SmartHomeEnv) -> pd.DataFrame:
    cols = [
        "timestamp",
        "PPV",
        "Pload",
        "PGRID",
        "EBESS",
        "SoCBESS",
        "EEV",
        "SoCEV",
        "PBESS",
        "PEV",
        "XPV",
    ]

    acts = env.sim.get_action_history()
    if not acts:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, float]] = []
    pmax_ev = max(getattr(env.ev, "Pmax_c", 0.0), getattr(env.ev, "Pmax_d", 0.0), 1e-9)

    for a in acts:
        soc_bess = a.get("soc_bess")
        soc_ev = a.get("soc_ev")
        pload = a.get("Pload")
        step_idx = a.get("step")

        pbess_val = float(a.get("PBESS", np.nan))
        pev_val = float(a.get("PEV", np.nan))
        ppv_val = float(a.get("ppv_used", np.nan))

        if pload is None and step_idx is not None:
            try:
                row = env.sim.dataframe.iloc[int(step_idx)]
                pload = float(row.get(env.load.column, np.nan) / 1000.0)
            except Exception:
                pload = np.nan

        rows.append(
            {
                "timestamp": pd.to_datetime(a.get("timestamp")),
                "PPV": ppv_val,
                "Pload": float(pload) if pload is not None else np.nan,
                "PGRID": float(a.get("pgrid", np.nan)),
                "EBESS": float(soc_bess * env.bess.Emax) if soc_bess is not None else np.nan,
                "SoCBESS": float(soc_bess) if soc_bess is not None else np.nan,
                "EEV": float(soc_ev * env.ev.Emax) if soc_ev is not None else np.nan,
                "SoCEV": float(soc_ev) if soc_ev is not None else np.nan,
                "PBESS": pbess_val,
                "PEV": pev_val,
                "XPV": float(a.get("XPV", np.nan)),
            }
        )

    df = pd.DataFrame(rows, columns=cols)
    num_cols = df.select_dtypes(include=["float", "int"]).columns
    df[num_cols] = df[num_cols].round(4)
    return df


def safe_step(env: SmartHomeEnv, action: np.ndarray):
    out = env.step(action)
    if len(out) == 5:
        o, r, done, trunc, info = out
        return o, float(r), bool(done or trunc), info
    if len(out) == 4:
        o, r, done, info = out
        trunc = bool(info.get("truncated", False)) if isinstance(info, dict) else False
        return o, float(r), bool(done or trunc), info
    raise ValueError(f"Unexpected env.step output length: {len(out)}")


def rollout_policy(env: SmartHomeEnv, actor: Actor, device: str) -> tuple[float, list[dict[str, float]]]:
    obs, _ = env.reset()
    done = False
    total_r = 0.0
    comps: list[dict[str, float]] = []

    with torch.no_grad():
        while not done:
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = actor(x).detach().cpu().numpy().squeeze(0)

            obs, r, done, info = safe_step(env, a)
            total_r += r

            comps.append(
                {
                    "bess_degradation": float(getattr(env.bess, "_costdeg", 0.0)),
                    "bess_penalty": float(getattr(env.bess, "_penalty", 0.0)),
                    "ev_degradation": float(getattr(env.ev, "_costdeg", 0.0)),
                    "ev_penalty": float(getattr(env.ev, "_penalty", 0.0)),
                    "grid_cost": float(getattr(env.grid, "_cost", 0.0)),
                    "grid_revenue": float(getattr(env.grid, "_revenue", 0.0)),
                    "grid_penalty": float(getattr(env.grid, "_penalty", 0.0)),
                    "pv_curt_cost": float(info.get("pv_curt_cost", 0.0)),
                    "reward": float(r),
                }
            )

    return float(total_r), comps


def load_actor_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    mask_payload = None
    hp_payload = None
    state_dict = None

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        mask_payload = ckpt.get("state_mask_payload") or ckpt.get("state_mask")
        hp_payload = ckpt.get("hyperparameters") or ckpt.get("hp")
    else:
        state_dict = ckpt

    if hp_payload is None:
        raise KeyError("Checkpoint não contém hyperparameters para construir o Actor.")

    hp = HyperParameters.from_dict(hp_payload)

    actor = Actor(hp)
    actor.load_state_dict(state_dict, strict=False)
    actor.to(device)
    actor.eval()
    return actor, mask_payload


def default_ckpt_for(tariff: str, results_root: Path) -> Path:
    """
    Resolve checkpoint path without requiring JSON changes.

    Priority:
    1) Results/<tariff>/<MODEL_SUBDIR>/best.pt   (your layout)
    2) Results/<tariff>/<MODEL_SUBDIR>/last.pt
    3) Results/<MODEL_SUBDIR>/<tariff>/best.pt  (older layout)
    4) Results/<MODEL_SUBDIR>/<tariff>/last.pt
    """
    candidates = [
        results_root / tariff / MODEL_SUBDIR / "best.pt",
        # results_root / tariff / MODEL_SUBDIR / "last.pt",
        # results_root / MODEL_SUBDIR / tariff / "best.pt",
        # results_root / MODEL_SUBDIR / tariff / "last.pt",
    ]
    for p in candidates:
        if p.exists():
            return p

    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No checkpoint found for tariff='{tariff}'. Tried:\n{tried}")



def _read_eval_runs(run_cfg: dict, cfg: dict) -> list[dict]:
    # Prefer run_config.json (eval -> runs fallback), else parameters.json
    ev = run_cfg.get("eval") if isinstance(run_cfg.get("eval"), list) else None
    if ev:
        return list(ev)
    if isinstance(run_cfg.get("runs"), list):
        return list(run_cfg["runs"])
    ev_cfg = cfg.get("eval") or {}
    runs = ev_cfg.get("runs") if isinstance(ev_cfg, dict) else None
    if runs:
        return list(runs)
    return list(cfg.get("runs", []))


def _pick_available_solver() -> str:
    return "gurobi"


def solve_teacher_and_extract(cfg: dict, df: pd.DataFrame, start_date: str, days: int, tariff: str):
    teacher = Teacher(cfg, df, start_date=start_date, days=days, state_mask=None)
    teacher.build(start_soc=float(cfg.get("BESS", {}).get("soc_init", 0.5)))
    solver_name = _pick_available_solver()
    teacher.solve(solver_name=solver_name)
    teacher_df = teacher.results_df()
    teacher_obj = float(pyo.value(teacher.model.objective))
    return teacher_df, teacher_obj, solver_name


def rollout_teacher_actions(env: SmartHomeEnv, teacher_df: pd.DataFrame) -> tuple[float, list[dict[str, float]]]:
    obs, _ = env.reset()
    done = False
    total_r = 0.0
    comps: list[dict[str, float]] = []

    for _, row in teacher_df.iterrows():
        if done:
            break
        pb = float(row.get("PBESS", 0.0))
        pe = float(row.get("Pev", 0.0))
        x = float(row.get("chi_pv", 0.0))
        obs, r, done, info = safe_step(env, np.array([pb, pe, x], dtype=np.float32))
        total_r += r

        comps.append(
            {
                "bess_degradation": float(getattr(env.bess, "_costdeg", 0.0)),
                "bess_penalty": float(getattr(env.bess, "_penalty", 0.0)),
                "ev_degradation": float(getattr(env.ev, "_costdeg", 0.0)),
                "ev_penalty": float(getattr(env.ev, "_penalty", 0.0)),
                "grid_cost": float(getattr(env.grid, "_cost", 0.0)),
                "grid_revenue": float(getattr(env.grid, "_revenue", 0.0)),
                "grid_penalty": float(getattr(env.grid, "_penalty", 0.0)),
                "pv_curt_cost": float(info.get("pv_curt_cost", 0.0)),
                "reward": float(r),
            }
        )

    return float(total_r), comps


def main():
    run_cfg = json.load(open(resolve_project_path(RUN_CONFIG_PATH), "r", encoding="utf-8")) if RUN_CONFIG_PATH.exists() else {}

    config = load_config(resolve_project_path(run_cfg.get("config", CONFIG_PATH)))
    exp = config.get("experiment", {}) or {}

    device = str(run_cfg.get("device", exp.get("device", "cpu")))
    results_root = resolve_project_path(run_cfg.get("results_root", exp.get("results_root", RESULTS_ROOT)))

    tariffs = run_cfg.get("tariffs") or exp.get("tariffs") or ["tar_s", "tar_w", "tar_sw", "tar_flat", "tar_tou"]
    runs = _read_eval_runs(run_cfg, config)

    # SEM subpasta evaluation
    out_dir = results_root / MODEL_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"runs": []}
    lines: list[str] = []

    for tariff in [str(t) for t in tariffs]:
        ckpt_path = resolve_project_path(exp.get("checkpoint", default_ckpt_for(tariff, results_root)))
        actor, mask_payload = load_actor_checkpoint(ckpt_path, device=device)

        for r in runs:
            data_path = resolve_project_path(r["data"])
            df = load_dataframe(data_path)
            start_date = str(r["start_date"])
            days = int(r["days"])
            run_label = str(r.get("run_label", Path(r["data"]).stem))

            cfg = json.loads(json.dumps(config))
            cfg.setdefault("Grid", {})
            cfg["Grid"]["tariff_column"] = tariff

            teacher_csv = out_dir / f"teacher_{tariff}_{run_label}.csv"

            teacher_df, teacher_obj, teacher_solver = solve_teacher_and_extract(cfg, df, start_date, days, tariff)

            env_teacher = SmartHomeEnv(cfg, dataframe=df, start_date=start_date, days=days, state_mask=None)
            teacher_R, teacher_comps = rollout_teacher_actions(env_teacher, teacher_df)

            teacher_log = build_action_log(env_teacher)
            teacher_log.to_csv(teacher_csv, index=False)

            fn = get_env_feature_names(config, df, start_date, days)
            sm = resolve_state_mask(mask_payload, fn)
            env_actor = SmartHomeEnv(cfg, dataframe=df, start_date=start_date, days=days, state_mask=sm)
            actor_R, actor_comps = rollout_policy(env_actor, actor, device=device)

            actor_csv = out_dir / f"actor_{tariff}_{run_label}.csv"
            actor_log = build_action_log(env_actor)
            actor_log.to_csv(actor_csv, index=False)

            delta = float(actor_R - teacher_R)
            better = "tie" if abs(delta) < 1e-9 else ("actor" if delta > 0 else "teacher")

            rec = {
                "tariff": tariff,
                "run_label": run_label,
                "start_date": start_date,
                "days": days,
                "data": str(data_path),
                "checkpoint": str(ckpt_path),
                "actor_actions_path": str(actor_csv),
                "actor_total_reward": float(actor_R),
                "actor_components_sum": {
                    k: float(np.sum([c.get(k, 0.0) for c in actor_comps]))
                    for k in [
                        "bess_degradation",
                        "bess_penalty",
                        "ev_degradation",
                        "ev_penalty",
                        "pv_curt_cost",
                        "grid_cost",
                        "grid_revenue",
                        "grid_penalty",
                        "reward",
                    ]
                },
                "teacher_objective": float(teacher_obj),
                "teacher_reward_equiv": float(-teacher_obj),
                "teacher_solver": teacher_solver,
                "teacher_decisions_path": str(teacher_csv),
                "teacher_total_reward_env": float(teacher_R),
                "teacher_components_sum": {
                    k: float(np.sum([c.get(k, 0.0) for c in teacher_comps]))
                    for k in [
                        "bess_degradation",
                        "bess_penalty",
                        "ev_degradation",
                        "ev_penalty",
                        "pv_curt_cost",
                        "grid_cost",
                        "grid_revenue",
                        "grid_penalty",
                        "reward",
                    ]
                },
                "delta_reward_actor_minus_teacher": float(delta),
                "better": better,
            }
            summary["runs"].append(rec)

            line = (
                f"{tariff} {run_label} start={start_date} days={days} "
                f"ActorR={actor_R:.3f} TeacherR={teacher_R:.3f} "
                f"TeacherObj={teacher_obj:.3f} better={better} ΔR={delta:.3f} ckpt={ckpt_path.name}"
            )
            lines.append(line)
            print(line)

    with open(out_dir / SUMMARY_NAME, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(out_dir / SUMMARY_TEXT_NAME, "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=2, ensure_ascii=False)

    print(f"Saved summary to: {out_dir}")


if __name__ == "__main__":
    main()
