from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from env.environment import SmartHomeEnv  # type: ignore
from opt import Teacher  # type: ignore

CONFIG_PATH = Path("data/parameters.json")
RESULTS_ROOT = Path("Results")
RUN_CONFIG_PATH = Path(__file__).with_name("run_config.json")

OUT_SUBDIR = "teacher_vs_env"
SUMMARY_NAME = "comparison_summary.json"
SUMMARY_TEXT_NAME = "comparison_summary_lines.json"


# ------------------------- I/O helpers -------------------------


def resolve_project_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_ev_departure(cfg: dict) -> None:
    """Ensure EV departure penalty fields are numeric for env/teacher."""
    ev = cfg.get("EV")
    if not isinstance(ev, dict):
        return
    dep = ev.get("departure_penalty")
    if isinstance(dep, dict):
        ev.setdefault("dep_thresholds", dep.get("thresholds", []))
        ev.setdefault("dep_weights", dep.get("weights", []))
        ev.setdefault("penalty_departure", 1.0)
        weights = dep.get("weights", [])
        ev["departure_penalty"] = float(weights[0]) if weights else 0.0
    else:
        ev.setdefault("dep_thresholds", ev.get("dep_thresholds", []))
        ev.setdefault("dep_weights", ev.get("dep_weights", []))
        ev.setdefault("penalty_departure", ev.get("penalty_departure", 0.0))


def load_dataframe(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep=";")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    return df


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


# ------------------------- Env logging -------------------------


def build_action_log(env: SmartHomeEnv) -> pd.DataFrame:
    """
    Pulls the env internal trajectory from env.sim action history.

    IMPORTANT:
    - PGRID here is whatever the env logged as 'pgrid' at each step.
    - EBESS/EEV are derived from logged SoC * Emax (consistent with env).
    """
    cols = [
        "timestamp",
        "PGRID_env",
        "EBESS_env",
        "EEV_env",
        "SoCBESS_env",
        "SoCEV_env",
        "PBESS_applied",
        "PEV_applied",
        "XPV_applied",
        "PPV_used",
        "Pload",
    ]

    acts = env.sim.get_action_history()
    if not acts:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, float]] = []
    for a in acts:
        ts = pd.to_datetime(a.get("timestamp"))
        soc_bess = a.get("soc_bess")
        soc_ev = a.get("soc_ev")

        pbess_val = float(a.get("PBESS", np.nan))
        pev_val = float(a.get("PEV", np.nan))
        xpv_val = float(a.get("XPV", np.nan))
        ppv_used = float(a.get("ppv_used", np.nan))
        pload = a.get("Pload")
        if pload is None:
            pload = np.nan

        rows.append(
            {
                "timestamp": ts,
                "PGRID_env": float(a.get("pgrid", np.nan)),
                "EBESS_env": float(soc_bess * env.bess.Emax) if soc_bess is not None else np.nan,
                "EEV_env": float(soc_ev * env.ev.Emax) if soc_ev is not None else np.nan,
                "SoCBESS_env": float(soc_bess) if soc_bess is not None else np.nan,
                "SoCEV_env": float(soc_ev) if soc_ev is not None else np.nan,
                "PBESS_applied": pbess_val,
                "PEV_applied": pev_val,
                "XPV_applied": xpv_val,
                "PPV_used": ppv_used,
                "Pload": float(pload) if pload is not None else np.nan,
            }
        )

    df = pd.DataFrame(rows, columns=cols)
    num_cols = df.select_dtypes(include=["float", "int"]).columns
    df[num_cols] = df[num_cols].round(8)
    return df


def safe_step(env: SmartHomeEnv, action: np.ndarray):
    """
    Keeps compatibility with your current env.step signature.
    """
    out = env.step(action)
    if len(out) == 5:
        o, r, done, trunc, info = out
        return o, float(r), bool(done or trunc), info
    if len(out) == 4:
        o, r, done, info = out
        trunc = bool(info.get("truncated", False)) if isinstance(info, dict) else False
        return o, float(r), bool(done or trunc), info
    raise ValueError(f"Unexpected env.step output length: {len(out)}")


# ------------------------- Optimization + rollout -------------------------


def _pick_available_solver(preferred: str | None = None) -> str:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates += ["gurobi", "cplex", "cbc", "highs", "glpk"]

    for s in candidates:
        try:
            opt = pyo.SolverFactory(s)
            if opt is not None and opt.available():
                return s
        except Exception:
            continue

    return "gurobi"


def solve_teacher(cfg: dict, df: pd.DataFrame, start_date: str, days: int):
    teacher = Teacher(cfg, df, start_date=start_date, days=days, state_mask=None)
    start_soc = float(cfg.get("BESS", {}).get("soc_init", 0.5))
    teacher.build(start_soc=start_soc)

    solver_name = _pick_available_solver()
    teacher.solve(solver_name=solver_name)

    teacher_df = teacher.results_df().copy()
    teacher_obj = float(pyo.value(teacher.model.objective))
    return teacher_df, teacher_obj, solver_name


def rollout_teacher_in_env(cfg: dict, df: pd.DataFrame, start_date: str, days: int, teacher_df: pd.DataFrame):
    """
    Applies teacher actions [PBESS, Pev, chi_pv] into the env and returns the env log.
    """
    env = SmartHomeEnv(cfg, dataframe=df, start_date=start_date, days=days, state_mask=None)
    obs, _ = env.reset()
    done = False

    # Iterate in teacher index order; stop if env horizon ends first
    for ts, row in teacher_df.iterrows():
        if done:
            break
        pb = float(row.get("PBESS", 0.0))
        pe = float(row.get("Pev", 0.0))
        x = float(row.get("chi_pv", 0.0))
        obs, r, done, info = safe_step(env, np.array([pb, pe, x], dtype=np.float32))

    env_log = build_action_log(env)
    return env, env_log


# ------------------------- Comparison -------------------------


def _mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x))) if x.size else float("nan")


def _rmse(x: np.ndarray) -> float:
    return float(math.sqrt(np.mean(x * x))) if x.size else float("nan")


def _maxabs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else float("nan")


def compare_teacher_vs_env(teacher_df: pd.DataFrame, env_log: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Compares internal trajectories:
      Teacher: Pgrid, EBESS, Eev
      Env:     PGRID_env, EBESS_env, EEV_env

    Returns:
      merged_df with diffs
      metrics dict
    """
    td = teacher_df.copy()

    # Teacher df indexed by timestamp; normalize to column
    if "timestamp" not in td.columns:
        td = td.reset_index().rename(columns={"index": "timestamp"})
    td["timestamp"] = pd.to_datetime(td["timestamp"], errors="coerce")
    td = td.dropna(subset=["timestamp"])

    # Select and standardize teacher columns
    rename_teacher = {}
    if "Pgrid" in td.columns:
        rename_teacher["Pgrid"] = "PGRID_teacher"
    if "EBESS" in td.columns:
        rename_teacher["EBESS"] = "EBESS_teacher"
    if "Eev" in td.columns:
        rename_teacher["Eev"] = "EEV_teacher"

    td = td.rename(columns=rename_teacher)

    keep_teacher = ["timestamp", "PGRID_teacher", "EBESS_teacher", "EEV_teacher"]
    for k in keep_teacher:
        if k not in td.columns:
            td[k] = np.nan
    td = td[keep_teacher]

    # Env log already has timestamp and *_env
    el = env_log.copy()
    el["timestamp"] = pd.to_datetime(el["timestamp"], errors="coerce")
    el = el.dropna(subset=["timestamp"])

    keep_env = ["timestamp", "PGRID_env", "EBESS_env", "EEV_env", "SoCBESS_env", "SoCEV_env"]
    for k in keep_env:
        if k not in el.columns:
            el[k] = np.nan
    el = el[keep_env]

    # Align on timestamps (inner join on overlap)
    merged = pd.merge(td, el, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)

    # Compute diffs
    for var in ["PGRID", "EBESS", "EEV"]:
        merged[f"{var}_diff"] = merged[f"{var}_env"] - merged[f"{var}_teacher"]

    # Metrics on overlap
    metrics = {"n_overlap": int(len(merged)), "vars": {}}
    for var in ["PGRID", "EBESS", "EEV"]:
        diff = merged[f"{var}_diff"].to_numpy(dtype=float)
        diff = diff[np.isfinite(diff)]
        metrics["vars"][var] = {
            "mae": _mae(diff),
            "rmse": _rmse(diff),
            "max_abs": _maxabs(diff),
        }

    return merged, metrics

# ------------------------- Plotting (Teacher vs Env time series) -------------------------
# Add this block to your existing script (same file). It only depends on matplotlib.

import matplotlib.pyplot as plt


def _downsample_df_time(df: pd.DataFrame, max_points: int = 6000) -> pd.DataFrame:
    if max_points is None or max_points <= 0:
        return df
    n = len(df)
    if n <= max_points:
        return df
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return df.iloc[idx].reset_index(drop=True)


def _maybe_resample(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if rule is None:
        return df
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    out = out.resample(str(rule)).mean(numeric_only=True).reset_index()
    return out


def plot_teacher_vs_env_timeseries(
    merged_df: pd.DataFrame,
    out_dir: Path,
    tariff: str,
    run_label: str,
    *,
    save_formats: list[str] = ["pdf"],
    dpi: int = 200,
    max_points: int = 6000,
    resample_rule: str | None = None,  # e.g., "15min", "1H", "1D"
) -> list[Path]:
    """
    Generates time-series plots comparing Teacher vs Env for:
      - PGRID, EBESS, EEV (overlay)
      - PGRID_diff, EBESS_diff, EEV_diff (diff = env - teacher)

    Expects columns from compare_teacher_vs_env():
      timestamp,
      PGRID_teacher, PGRID_env, PGRID_diff,
      EBESS_teacher, EBESS_env, EBESS_diff,
      EEV_teacher, EEV_env, EEV_diff
    """
    out_paths: list[Path] = []

    df = merged_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df = _maybe_resample(df, resample_rule)
    df = _downsample_df_time(df, max_points=max_points)

    figs_dir = out_dir / "Figures" / tariff / run_label
    figs_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig: plt.Figure, name: str):
        for fmt in save_formats:
            p = figs_dir / f"{name}.{fmt}"
            if fmt.lower() in {"png", "jpg", "jpeg", "webp"}:
                fig.savefig(p, dpi=dpi)
            else:
                fig.savefig(p)
            out_paths.append(p)

    for var in ["PGRID", "EBESS", "EEV"]:
        tcol = f"{var}_teacher"
        ecol = f"{var}_env"
        dcol = f"{var}_diff"

        if not all(c in df.columns for c in [tcol, ecol, dcol]):
            continue

        # Overlay plot
        fig1 = plt.figure()
        plt.plot(df["timestamp"], df[tcol], label=f"{var} (teacher)")
        plt.plot(df["timestamp"], df[ecol], label=f"{var} (env)")
        plt.title(f"{tariff} | {run_label} | {var}: teacher vs env")
        plt.xlabel("Time")
        plt.ylabel(var)
        plt.legend()
        plt.tight_layout()
        _save(fig1, f"{var}_overlay")
        plt.close(fig1)

        # Diff plot
        fig2 = plt.figure()
        plt.plot(df["timestamp"], df[dcol], label=f"{var}_diff (env - teacher)")
        plt.title(f"{tariff} | {run_label} | {var}_diff = env - teacher")
        plt.xlabel("Time")
        plt.ylabel(f"{var}_diff")
        plt.legend()
        plt.tight_layout()
        _save(fig2, f"{var}_diff")
        plt.close(fig2)

    return out_paths



# ------------------------- Main -------------------------


def main():
    run_cfg = (
        json.load(open(resolve_project_path(RUN_CONFIG_PATH), "r", encoding="utf-8"))
        if RUN_CONFIG_PATH.exists()
        else {}
    )

    config = load_config(resolve_project_path(run_cfg.get("config", CONFIG_PATH)))
    _normalize_ev_departure(config)
    exp = config.get("experiment", {}) or {}

    results_root = resolve_project_path(run_cfg.get("results_root", exp.get("results_root", RESULTS_ROOT)))
    tariffs = run_cfg.get("tariffs") or exp.get("tariffs") or ["tar_s", "tar_w", "tar_sw", "tar_flat", "tar_tou"]
    runs = _read_eval_runs(run_cfg, config)

    out_dir = results_root / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"runs": []}
    lines: list[str] = []

    for tariff in [str(t) for t in tariffs]:
        for r in runs:
            data_path = resolve_project_path(r["data"])
            df = load_dataframe(data_path)

            start_date = str(r["start_date"])
            days = int(r["days"])
            run_label = str(r.get("run_label", Path(r["data"]).stem))

            # Per-tariff cfg
            cfg = json.loads(json.dumps(config))
            _normalize_ev_departure(cfg)
            cfg.setdefault("Grid", {})
            cfg["Grid"]["tariff_column"] = tariff

            # 1) Solve teacher and save decisions
            teacher_df, teacher_obj, solver_name = solve_teacher(cfg, df, start_date, days)
            teacher_decisions_csv = out_dir / f"teacher_decisions_{tariff}_{run_label}.csv"
            teacher_df.to_csv(teacher_decisions_csv, index=True)

            # 2) Roll teacher decisions into env and log env internal trajectory
            _, env_log = rollout_teacher_in_env(cfg, df, start_date, days, teacher_df)
            env_roll_csv = out_dir / f"env_rollout_{tariff}_{run_label}.csv"
            env_log.to_csv(env_roll_csv, index=False)

            # 3) Compare internal trajectories (Pgrid, EBESS, EEV)
            merged, metrics = compare_teacher_vs_env(teacher_df, env_log)
            compare_csv = out_dir / f"compare_{tariff}_{run_label}.csv"
            merged.to_csv(compare_csv, index=False)

            rec = {
                "tariff": tariff,
                "run_label": run_label,
                "start_date": start_date,
                "days": days,
                "data": str(data_path),
                "solver": solver_name,
                "teacher_objective": float(teacher_obj),
                "teacher_decisions_path": str(teacher_decisions_csv),
                "env_rollout_path": str(env_roll_csv),
                "comparison_path": str(compare_csv),
                "metrics": metrics,
            }
            summary["runs"].append(rec)

            mP = metrics["vars"]["PGRID"]
            mB = metrics["vars"]["EBESS"]
            mE = metrics["vars"]["EEV"]
            line = (
                f"{tariff} {run_label} start={start_date} days={days} solver={solver_name} "
                f"overlap={metrics['n_overlap']} | "
                f"PGRID(mae={mP['mae']:.4f}, rmse={mP['rmse']:.4f}, max={mP['max_abs']:.4f}) | "
                f"EBESS(mae={mB['mae']:.4f}, rmse={mB['rmse']:.4f}, max={mB['max_abs']:.4f}) | "
                f"EEV(mae={mE['mae']:.4f}, rmse={mE['rmse']:.4f}, max={mE['max_abs']:.4f})"
            )
            lines.append(line)
            print(line)

    with open(out_dir / SUMMARY_NAME, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(out_dir / SUMMARY_TEXT_NAME, "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=2, ensure_ascii=False)

    print(f"Saved outputs to: {out_dir}")

    # ------------------------- Main (add inside your existing main loop) -------------------------
    # Insert these lines right after you create `merged, metrics = compare_teacher_vs_env(...)`
    # and right after you save `compare_csv` (or before, either is fine).

    # 4) Plots (Teacher vs Env over time)
    try:
        _ = plot_teacher_vs_env_timeseries(
            merged_df=merged,
            out_dir=out_dir,              # this is Results/teacher_vs_env
            tariff=tariff,
            run_label=run_label,
            save_formats=["pdf"],         # add "png" if you want
            dpi=200,
            max_points=6000,              # reduce if files are too heavy
            resample_rule=None,           # e.g., "1H" for 5-min data over 365 days
        )
    except Exception as e:
        print(f"[warn] plotting failed for {tariff} {run_label}: {e}")


if __name__ == "__main__":
    main()
