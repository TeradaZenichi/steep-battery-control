# models/2_MLP_FT_SAC/test.py
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from env.environment import SmartHomeEnv  # type: ignore
from hp import HP  # type: ignore
from model import ActorGaussian  # type: ignore
from opt import Teacher  # type: ignore


CONFIG_PATH = PROJECT_ROOT / "data" / "parameters.json"
RESULTS_ROOT = PROJECT_ROOT / "Results"
MODEL_SUBDIR = "2_MLP_FT_SAC"
SUMMARY_NAME = "evaluation_summary.json"
SUMMARY_TEXT_NAME = "evaluation_summary_lines.json"
RUN_CONFIG_PATH = HERE / "run_config.json"


def resolve_project_path(p: str | Path, root: Path = PROJECT_ROOT) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    return df.reset_index(drop=True)


def get_env_feature_names(config: dict, df: pd.DataFrame, start_date: str, days: int, tariff: str) -> list[str]:
    cfg = json.loads(json.dumps(config))
    cfg.setdefault("Grid", {})
    cfg["Grid"]["tariff_column"] = tariff
    e = SmartHomeEnv(cfg, dataframe=df, start_date=start_date, days=days, state_mask=None)
    return list(getattr(e, "_feature_names", getattr(e, "state_feature_labels", [])))


def resolve_state_mask(payload: Any, feature_names: list[str] | None = None) -> np.ndarray | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        v = payload.get("vector", None)
        lab = payload.get("labels", None)
        if v is None:
            return None
        v = np.asarray(v, dtype=bool)
        if feature_names is not None and lab is not None and len(lab) == len(v):
            m = {lab[i]: bool(v[i]) for i in range(len(lab))}
            v = np.asarray([m.get(n, False) for n in feature_names], dtype=bool)
        return v if v.any() else None
    v = np.asarray(payload, dtype=bool)
    return v if v.any() else None


def extract_reward_components(env: SmartHomeEnv) -> dict:
    return {
        "bess_degradation": -float(getattr(env.bess, "_costdeg", 0.0)),
        "bess_penalty": -float(getattr(env.bess, "_penalty", 0.0)),
        "ev_degradation": -float(getattr(env.ev, "_costdeg", 0.0)),
        "ev_penalty": -float(getattr(env.ev, "_penalty", 0.0)),
        "grid_cost": -float(getattr(env.grid, "_cost", 0.0)),
        "grid_revenue": float(getattr(env.grid, "_revenue", 0.0)),
        "grid_penalty": -float(getattr(env.grid, "_penalty", 0.0)),
        "grid_net_cost": float(getattr(env.grid, "_cost", 0.0))
        - float(getattr(env.grid, "_revenue", 0.0))
        + float(getattr(env.grid, "_penalty", 0.0)),
    }


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
        o, r, te, tr, _ = out
        return o, float(r), bool(te), bool(tr)
    o, r, d, _ = out
    return o, float(r), bool(d), False


@torch.no_grad()
def rollout_policy(
    env: SmartHomeEnv,
    actor: ActorGaussian,
    device: torch.device,
    deterministic: bool,
) -> tuple[float, float, list[dict]]:
    o, _ = env.reset()
    te = tr = False
    R, grid_net = 0.0, 0.0
    comps: list[dict] = []

    while not (te or tr):
        a = (
            actor.act(torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0), deterministic=deterministic)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        o, r, te, tr = safe_step(env, a)
        R += r
        c = extract_reward_components(env)
        grid_net += float(c["grid_net_cost"])
        c["reward"] = r
        c["terminated"] = te
        c["truncated"] = tr
        c["timestamp"] = getattr(env.sim, "current_datetime", None)
        comps.append(c)

    return R, grid_net, comps


def load_actor_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[ActorGaussian, dict | None]:
    p = torch.load(ckpt_path, map_location="cpu")
    hp = HP.from_dict(p.get("hyperparameters", p.get("hp", {})))
    a = ActorGaussian(hp).to(device)
    if "actor_gaussian_state_dict" in p:
        a.load_state_dict(p["actor_gaussian_state_dict"], strict=True)
    elif "model_state_dict" in p:
        a.mean_net.load_state_dict(p["model_state_dict"], strict=True)
    else:
        a.load_state_dict(p, strict=False)
    return a, p.get("state_mask", None)


def ckpt_for_tariff(tariff: str, results_root: Path) -> Path:
    p = results_root / tariff / MODEL_SUBDIR / "best.pt"
    if p.exists():
        return p
    raise FileNotFoundError(f"Missing checkpoint for tariff={tariff}: {p}")


def _read_test_runs(run_cfg: dict) -> list[dict]:
    exp = run_cfg.get("experiment", {}) if isinstance(run_cfg.get("experiment", {}), dict) else {}
    if isinstance(exp.get("test_runs", None), list):
        return list(exp["test_runs"])
    if isinstance(run_cfg.get("test_runs", None), list):
        return list(run_cfg["test_runs"])
    e = run_cfg.get("eval", None)
    if isinstance(e, dict) and isinstance(e.get("runs", None), list):
        return list(e["runs"])
    if isinstance(e, list):
        return list(e)
    raise KeyError("run_config.json must define experiment.test_runs (preferred) or test_runs/eval.")


def _read_experiment_tariffs(run_cfg: dict, config: dict) -> list[str]:
    exp = run_cfg.get("experiment", {}) if isinstance(run_cfg.get("experiment", {}), dict) else {}
    t = exp.get("tariffs", None)
    if isinstance(t, list) and t:
        return [str(x) for x in t]
    if isinstance(run_cfg.get("tariffs", None), list) and run_cfg["tariffs"]:
        return [str(x) for x in run_cfg["tariffs"]]
    return [str(config.get("Grid", {}).get("tariff_column", "tar_tou"))]


def _pick_available_solver() -> str:
    import pyomo.environ as pyo  # type: ignore

    for s in ["gurobi", "cplex", "highs", "cbc", "glpk"]:
        if pyo.SolverFactory(s).available(exception_flag=False):
            return s
    raise RuntimeError("No MILP solver available for Pyomo.")


def solve_teacher_and_extract(
    cfg: dict,
    df: pd.DataFrame,
    start_date: str,
    days: int,
    state_mask,
) -> tuple[float, pd.DataFrame, str]:
    import pyomo.environ as pyo  # type: ignore

    s = _pick_available_solver()
    t = Teacher(cfg, df, start_date=start_date, days=days, state_mask=state_mask)
    t.build(start_soc=float(cfg.get("BESS", {}).get("soc_init", 0.5)))
    pyo.SolverFactory(s).solve(t.model, tee=False)
    return float(pyo.value(t.model.objective)), t.results_df(), s


def rollout_teacher_actions(env: SmartHomeEnv, tdf: pd.DataFrame) -> tuple[float, float, list[dict]]:
    o, _ = env.reset()
    te = tr = False
    R, grid_net, i = 0.0, 0.0, 0
    comps: list[dict] = []

    while not (te or tr):
        ts = getattr(env.sim, "current_datetime", None)
        row = tdf.loc[pd.Timestamp(ts)] if ts is not None and pd.Timestamp(ts) in tdf.index else tdf.iloc[i]
        a = np.array([float(row["PBESS"]), float(row["Pev"]), float(row["chi_pv"])], dtype=np.float32)
        o, r, te, tr = safe_step(env, a)
        R += r
        c = extract_reward_components(env)
        grid_net += float(c["grid_net_cost"])
        c["reward"] = r
        c["terminated"] = te
        c["truncated"] = tr
        c["timestamp"] = getattr(env.sim, "current_datetime", None)
        comps.append(c)
        i += 1

    return R, grid_net, comps



def main() -> None:
    run_cfg = json.load(open(RUN_CONFIG_PATH, "r", encoding="utf-8")) if RUN_CONFIG_PATH.exists() else {}
    config = load_config(resolve_project_path(run_cfg.get("config", CONFIG_PATH)))
    results_root = resolve_project_path(run_cfg.get("results_root", RESULTS_ROOT))

    tariffs = _read_experiment_tariffs(run_cfg, config)
    runs = _read_test_runs(run_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det_eval = bool(run_cfg.get("deterministic_eval", True))
    use_teacher = bool(run_cfg.get("use_teacher", True))

    # NOTE: kept as-is from your SAC script; change to results_root / MODEL_SUBDIR if you also want to remove "evaluation"
    out_dir = results_root / MODEL_SUBDIR 
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[EVAL] device={device} model={MODEL_SUBDIR} out={out_dir} tariffs={tariffs} n_runs={len(runs)} "
        f"teacher={'ON' if use_teacher else 'OFF'} det_eval={det_eval}"
    )

    summary: dict = {"model": MODEL_SUBDIR, "runs": []}
    lines: list[str] = []

    for tariff in tariffs:
        ckpt_path = resolve_project_path(run_cfg.get("checkpoint", ckpt_for_tariff(tariff, results_root)))
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

            fn = get_env_feature_names(config, df, start_date, days, tariff)
            sm = resolve_state_mask(mask_payload, fn)

            env_actor = SmartHomeEnv(cfg, dataframe=df, start_date=start_date, days=days, state_mask=sm)
            actor_R, actor_grid, actor_comps = rollout_policy(env_actor, actor, device=device, deterministic=det_eval)
            actor_csv = out_dir / f"actor_{tariff}_{run_label}.csv"
            actor_log = build_action_log(env_actor)
            actor_log.to_csv(actor_csv, index=False)

            teacher_obj = teacher_R = teacher_grid = None
            teacher_solver = None
            teacher_csv = None
            teacher_comps = None

            if use_teacher:
                teacher_obj, teacher_df, teacher_solver = solve_teacher_and_extract(cfg, df, start_date, days, state_mask=sm)
                teacher_csv = out_dir / f"teacher_{tariff}_{run_label}.csv"
                env_teacher = SmartHomeEnv(cfg, dataframe=df, start_date=start_date, days=days, state_mask=sm)
                teacher_R, teacher_grid, teacher_comps = rollout_teacher_actions(env_teacher, teacher_df)

                teacher_log = build_action_log(env_teacher)
                teacher_log.to_csv(teacher_csv, index=False)

            delta = None if teacher_R is None else float(actor_R - float(teacher_R))
            better = None if teacher_R is None else ("tie" if abs(delta) < 1e-9 else ("actor" if delta > 0 else "teacher"))

            rec = {
                "tariff": tariff,
                "run_label": run_label,
                "start_date": start_date,
                "days": days,
                "data": str(data_path),
                "checkpoint": str(ckpt_path),
                "deterministic_eval": det_eval,
                "actor_actions_path": str(actor_csv),
                "actor_total_reward": float(actor_R),
                "actor_reward_per_day": float(actor_R / max(days, 1e-12)),
                "actor_grid_net_cost_total": float(actor_grid),
                "actor_grid_net_cost_per_day": float(actor_grid / max(days, 1e-12)),
                "actor_components_sum": {
                    k: float(np.sum([c.get(k, 0.0) for c in actor_comps]))
                    for k in [
                        "bess_degradation",
                        "bess_penalty",
                        "ev_degradation",
                        "ev_penalty",
                        "grid_cost",
                        "grid_revenue",
                        "grid_penalty",
                        "grid_net_cost",
                        "reward",
                    ]
                },
                "teacher_objective": None if teacher_obj is None else float(teacher_obj),
                "teacher_reward_equiv": None if teacher_obj is None else float(-teacher_obj),
                "teacher_solver": teacher_solver,
                "teacher_decisions_path": None if teacher_csv is None else str(teacher_csv),
                "teacher_total_reward_env": None if teacher_R is None else float(teacher_R),
                "teacher_grid_net_cost_total": None if teacher_grid is None else float(teacher_grid),
                "teacher_grid_net_cost_per_day": None if teacher_grid is None else float(teacher_grid / max(days, 1e-12)),
                "teacher_components_sum": None
                if teacher_comps is None
                else {
                    k: float(np.sum([c.get(k, 0.0) for c in teacher_comps]))
                    for k in [
                        "bess_degradation",
                        "bess_penalty",
                        "ev_degradation",
                        "ev_penalty",
                        "grid_cost",
                        "grid_revenue",
                        "grid_penalty",
                        "grid_net_cost",
                        "reward",
                    ]
                },
                "delta_reward_actor_minus_teacher": delta,
                "better": better,
            }
            summary["runs"].append(rec)

            line = (
                f"{tariff} {run_label} start={start_date} days={days} "
                f"ActorR={actor_R:.3f} grid/day={actor_grid/max(days,1e-12):.3f} "
                + (
                    f"TeacherR={float(teacher_R):.3f} TeacherObj={float(teacher_obj):.3f} better={better} ΔR={float(delta):.3f}"
                    if teacher_R is not None and teacher_obj is not None
                    else "Teacher=OFF"
                )
                + f" ckpt={ckpt_path}"
            )
            lines.append(line)
            print(line)

    with open(out_dir / SUMMARY_NAME, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(out_dir / SUMMARY_TEXT_NAME, "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=2, ensure_ascii=False)

    print(f"[DONE] saved: {out_dir / SUMMARY_NAME}")


if __name__ == "__main__":
    main()
