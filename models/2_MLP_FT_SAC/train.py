# models/2_MLP_FT_SAC/train.py
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = (Path.cwd() if (Path.cwd() / "data" / "parameters.json").exists() else HERE.parents[2])

sys.path.insert(0, str(HERE))
sys.path.insert(1, str(PROJECT_ROOT))

from env.environment import SmartHomeEnv  # type: ignore
from model import ActorGaussian, TwinCritic  # type: ignore
from hp import ReplayBuffer, HP  # type: ignore

RUN_CONFIG_PATH = HERE / "run_config.json"


# -----------------------------
# IO helpers
# -----------------------------
def rpath(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as fp:
        return json.load(fp)


def dump_json(p: Path, obj: dict):
    with open(p, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)


def load_df(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, sep=";")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    return df.reset_index(drop=True)


# -----------------------------
# Environment helpers
# -----------------------------
def safe_step(env: SmartHomeEnv, action: np.ndarray):
    out = env.step(action)
    if len(out) == 5:
        o, r, te, tr, _ = out
        return o, float(r), bool(te), bool(tr)
    o, r, d, _ = out
    return o, float(r), bool(d), False


def make_env(cfg: dict, df: pd.DataFrame, start_date: str, days: int, tariff: str, state_mask):
    c = json.loads(json.dumps(cfg))
    c.setdefault("Grid", {})
    c["Grid"]["tariff_column"] = tariff
    return SmartHomeEnv(c, dataframe=df, start_date=start_date, days=days, state_mask=state_mask)


# -----------------------------
# JSON-aligned utilities
# -----------------------------
def _req(d: dict, key: str):
    if key not in d:
        raise KeyError(f"run_config.json missing key: {key}")
    return d[key]


def _norm_probs(w: list[float]) -> np.ndarray:
    arr = np.asarray(w, dtype=float)
    s = float(arr.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.ones_like(arr) / max(len(arr), 1)
    return arr / s


def sample_episode_days(stage_cfg: dict, max_days: int) -> int:
    sampler = stage_cfg.get("episode_days_sampler", {}) or {}
    vals = list(sampler.get("values", [1]) or [1])
    wts = list(sampler.get("weights", [1.0] * len(vals)) or [1.0] * len(vals))
    if len(vals) != len(wts):
        raise ValueError("episode_days_sampler.values and weights must have the same length.")
    ep = int(np.random.choice(vals, p=_norm_probs([float(x) for x in wts])))
    return max(1, min(int(max_days), int(ep)))


def init_linear_weights(module: torch.nn.Module, init_type: str, bias_init: str):
    init_type = (init_type or "").lower()
    bias_init = (bias_init or "").lower()
    for m in module.modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if init_type in ("xavier_uniform", "xavier_uniform_"):
            torch.nn.init.xavier_uniform_(m.weight)
        elif init_type in ("xavier_normal", "xavier_normal_"):
            torch.nn.init.xavier_normal_(m.weight)
        elif init_type in ("kaiming_uniform", "kaiming_uniform_"):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif init_type in ("kaiming_normal", "kaiming_normal_"):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        # else: keep defaults

        if m.bias is not None:
            if bias_init in ("zeros", "zero"):
                torch.nn.init.zeros_(m.bias)
            elif bias_init in ("ones", "one"):
                torch.nn.init.ones_(m.bias)


def _get_attr_any(obj: Any, names: list[str]) -> Optional[float]:
    for n in names:
        if hasattr(obj, n):
            try:
                v = getattr(obj, n)
                if callable(v):
                    v = v()
                return float(v)
            except Exception:
                continue
    return None


def get_dt_hours(env: Any) -> float:
    for name in ("dt_hours", "delta_t_hours", "step_hours"):
        v = _get_attr_any(env, [name])
        if v is not None:
            return float(v)

    for name in ("dt", "delta_t", "step_size"):
        v = _get_attr_any(env, [name])
        if v is not None:
            # Heuristic: seconds vs hours
            return float(v / 3600.0) if v > 24.0 else float(v)

    return 1.0


def get_grid_power_kw(env: Any) -> Optional[float]:
    grid = getattr(env, "grid", None)
    if grid is not None:
        v = _get_attr_any(
            grid,
            ["p_kw", "p", "P", "power_kw", "power", "_p", "_p_kw", "p_grid_kw", "p_grid", "p_net_kw", "p_net"],
        )
        if v is not None:
            return float(v)
    return _get_attr_any(env, ["p_grid_kw", "p_grid", "grid_power_kw", "grid_power"])


def get_soc(env: Any) -> Optional[float]:
    bess = getattr(env, "bess", None)
    if bess is not None:
        v = _get_attr_any(bess, ["soc", "SoC", "_soc", "state_of_charge", "soc_frac", "soc_percent"])
        if v is not None:
            return float(v)
    return _get_attr_any(env, ["soc", "SoC", "_soc"])


@torch.no_grad()
def eval_runs(
    base_cfg: dict,
    runs: list[dict],
    tariff: str,
    actor: ActorGaussian,
    device: torch.device,
    state_mask,
    deterministic: bool,
    reward_scale: float,
) -> dict:
    reward_total = 0.0
    total_days = 0.0

    grid_import_total_kwh = 0.0
    grid_import_peak_kw = 0.0

    soc_min = float("inf")
    soc_max = -float("inf")
    soc_end_vals: list[float] = []

    for rr in runs:
        env = make_env(base_cfg, load_df(rpath(rr["data"])), rr["start_date"], int(rr["days"]), tariff, state_mask)
        dt_h = get_dt_hours(env)

        o = env.reset()[0]
        te = tr = False
        last_soc: Optional[float] = None

        while not (te or tr):
            ot = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            a = actor.act(ot, deterministic=deterministic).squeeze(0).detach().cpu().numpy()
            o, r, te, tr = safe_step(env, a)
            reward_total += float(reward_scale) * float(r)

            p_grid = get_grid_power_kw(env)
            if p_grid is not None:
                p_imp = max(float(p_grid), 0.0)
                grid_import_total_kwh += p_imp * float(dt_h)
                grid_import_peak_kw = max(grid_import_peak_kw, p_imp)

            soc = get_soc(env)
            if soc is not None and np.isfinite(soc):
                soc_min = min(soc_min, float(soc))
                soc_max = max(soc_max, float(soc))
                last_soc = float(soc)

        if last_soc is not None:
            soc_end_vals.append(last_soc)

        total_days += float(rr["days"])

    if not np.isfinite(soc_min):
        soc_min = float("nan")
    if not np.isfinite(soc_max):
        soc_max = float("nan")

    return {
        "reward_total": float(reward_total),
        "reward_per_day": float(reward_total / max(total_days, 1e-12)),
        "grid_import_total_kwh": float(grid_import_total_kwh),
        "grid_import_peak_kw": float(grid_import_peak_kw),
        "soc_min": float(soc_min),
        "soc_max": float(soc_max),
        "soc_end": float(np.mean(soc_end_vals)) if soc_end_vals else float("nan"),
        "days": float(total_days),
    }


def is_better_lexicographic(cur: dict, best: dict, sel: dict) -> bool:
    prim = sel.get("primary_constraint", {}) or {}
    sec = sel.get("secondary_objective", {}) or {}

    p_metric = str(prim.get("metric", "")).strip()
    s_metric = str(sec.get("metric", "")).strip()
    if not p_metric or not s_metric:
        raise ValueError("best_model_selection requires primary_constraint.metric and secondary_objective.metric")

    p_dir = str(prim.get("direction", "minimize")).lower()
    s_dir = str(sec.get("direction", "maximize")).lower()

    tol_rel = float(prim.get("tolerance_rel", 0.0))
    min_impr_rel = float(sel.get("min_improvement_rel", 0.0))

    cp, bp = float(cur[p_metric]), float(best[p_metric])
    cs, bs = float(cur[s_metric]), float(best[s_metric])

    if p_dir == "minimize":
        improves_primary = cp < bp * (1.0 - tol_rel)
        within_tol = cp <= bp * (1.0 + tol_rel)
    elif p_dir == "maximize":
        improves_primary = cp > bp * (1.0 + tol_rel)
        within_tol = cp >= bp * (1.0 - tol_rel)
    else:
        raise ValueError(f"Unsupported primary direction: {p_dir}")

    if improves_primary:
        return True
    if not within_tol:
        return False

    ref = max(abs(bs), 1e-12)
    if s_dir == "maximize":
        return cs > bs + ref * min_impr_rel
    if s_dir == "minimize":
        return cs < bs - ref * min_impr_rel
    raise ValueError(f"Unsupported secondary direction: {s_dir}")


def main():
    rc = load_json(RUN_CONFIG_PATH)

    # Base env parameters file (kept consistent with the repository layout)
    base_cfg = load_json(PROJECT_ROOT / "data" / "parameters.json")

    exp = _req(rc, "experiment")
    seed = int(exp.get("seed", 0))
    dev = str(exp.get("device", "cpu")).lower()
    device = torch.device("cuda" if dev.startswith("cuda") and torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_id = str(exp.get("model_id", "2_MLP_FT_SAC"))
    results_root = str(exp.get("results_root", "Results"))
    tariffs = list(exp.get("tariffs", []) or [])
    if not tariffs:
        raise ValueError("experiment.tariffs is empty.")

    sac = _req(rc, "sac")
    actor_cfg = sac.get("actor", {}) or {}
    if not bool(actor_cfg.get("init_from_il", True)):
        raise ValueError("This script expects sac.actor.init_from_il = true (actor initialized from IL checkpoint).")

    common = sac.get("common", {}) or {}
    gamma = float(common.get("gamma", 0.99))
    batch_size = int(common.get("batch_size", 256))
    replay_buffer_size = int(common.get("replay_buffer_size", 2_000_000))
    learning_starts = int(common.get("learning_starts", 5000))
    updates_per_env_step = int(common.get("updates_per_env_step", 1))
    reward_scale = float(sac.get("reward_scale", 1.0))

    actor_opt = sac.get("actor_optimizer", {}) or {}
    critic_cfg = sac.get("critics", {}) or {}
    if not bool(critic_cfg.get("twin_q", True)):
        raise ValueError("run_config.json expects sac.critics.twin_q = true (TwinCritic).")

    critic_arch = critic_cfg.get("architecture", {}) or {}
    critic_opt = critic_cfg.get("optimizer", {}) or {}
    critic_train = critic_cfg.get("training", {}) or {}
    target_update = (critic_train.get("target_update", {}) or {})
    init_cfg = critic_cfg.get("initialization", {}) or {}

    actor_lr = float(actor_opt.get("lr", 3e-4))
    actor_wd = float(actor_opt.get("weight_decay", 0.0))
    actor_betas = tuple(actor_opt.get("betas", [0.9, 0.999]))
    actor_eps = float(actor_opt.get("eps", 1e-8))

    critic_lr = float(critic_opt.get("lr", 3e-4))
    critic_wd = float(critic_opt.get("weight_decay", 0.0))
    critic_betas = tuple(critic_opt.get("betas", [0.9, 0.999]))
    critic_eps = float(critic_opt.get("eps", 1e-8))

    tau = float(target_update.get("tau", 0.005))
    target_update_every = int(target_update.get("update_every_steps", 1))
    grad_clip_norm = float(critic_train.get("grad_clip_norm", 0.0))

    init_type = str(init_cfg.get("init_type", ""))
    bias_init = str(init_cfg.get("bias_init", ""))

    ent = sac.get("entropy", {}) or {}
    target_entropy_cfg = ent.get("target_entropy", "auto")
    alpha_cfg = ent.get("alpha", {}) or {}
    alpha_mode = str(alpha_cfg.get("mode", "auto")).lower()
    alpha_init = float(alpha_cfg.get("alpha_init", 0.05))
    alpha_min = float(alpha_cfg.get("alpha_min", 0.001))
    alpha_max_global = float(alpha_cfg.get("alpha_max", 0.5))
    alpha_lr = float(alpha_cfg.get("alpha_lr", 3e-4))

    schedule = _req(rc, "exploration_schedule")
    stages = list(schedule.get("stages", []) or [])
    if not stages:
        raise ValueError("exploration_schedule.stages is empty.")
    stage_by_id = {int(s["stage_id"]): s for s in stages}
    stage_order = [int(s["stage_id"]) for s in stages]

    train_runs = list(_req(rc, "train_runs"))
    val_runs = list(_req(rc, "val_runs"))
    paths_by_tariff = _req(rc, "paths_by_tariff")

    eval_cfg = _req(rc, "evaluation")
    eval_every_steps = int(eval_cfg.get("eval_every_steps", 5000))
    deterministic_eval = bool(eval_cfg.get("deterministic_eval", True))
    report_metrics = list(eval_cfg.get("report_metrics", []) or [])
    best_sel = eval_cfg.get("best_model_selection", {}) or {}
    best_sel_enabled = bool(best_sel.get("enabled", True))

    print(f"[TRAIN] root={PROJECT_ROOT} cfg={RUN_CONFIG_PATH} device={device} seed={seed} model_id={model_id}")
    print(f"[TRAIN] tariffs={tariffs} train_runs={len(train_runs)} val_runs={len(val_runs)} eval_every={eval_every_steps}")
    if best_sel_enabled:
        prim = best_sel.get("primary_constraint", {}) or {}
        sec = best_sel.get("secondary_objective", {}) or {}
        print(
            f"[SELECT] primary={prim.get('metric')}({prim.get('direction')}) tol_rel={prim.get('tolerance_rel')} "
            f"secondary={sec.get('metric')}({sec.get('direction')}) min_improvement_rel={best_sel.get('min_improvement_rel')}"
        )

    for tariff in tariffs:
        pbt = paths_by_tariff.get(tariff, {}) or {}
        resume_from_il = rpath(pbt.get("resume_from_il", Path(results_root) / tariff / "1_MLP_IL" / "best.pt"))
        out_dir = rpath(pbt.get("output_dir", Path(results_root) / tariff / model_id))
        out_dir.mkdir(parents=True, exist_ok=True)

        il = torch.load(resume_from_il, map_location="cpu")
        hp = HP.from_dict(il.get("hyperparameters", {}))
        state_mask = (il.get("state_mask") or {}).get("vector", None)

        # Apply critic architecture exactly from JSON (actor must remain compatible with IL checkpoint)
        hp.critic_hidden_sizes = list(critic_arch.get("hidden_layers", [256, 256]))
        hp.critic_activation = str(critic_arch.get("activation", "relu"))
        hp.critic_dropout = float(critic_arch.get("dropout", 0.0))
        try:
            hp._d["critic_hidden_sizes"] = hp.critic_hidden_sizes
            hp._d["critic_activation"] = hp.critic_activation
            hp._d["critic_dropout"] = hp.critic_dropout
        except Exception:
            pass

        actor = ActorGaussian(hp).to(device)
        actor.mean_net.load_state_dict(il["model_state_dict"], strict=True)

        critics = TwinCritic(hp).to(device)
        init_linear_weights(critics, init_type=init_type, bias_init=bias_init)
        tcritics = TwinCritic(hp).to(device)
        tcritics.load_state_dict(critics.state_dict(), strict=True)

        o_actor = torch.optim.Adam(
            actor.parameters(), lr=actor_lr, weight_decay=actor_wd, betas=actor_betas, eps=actor_eps
        )
        o_crit = torch.optim.Adam(
            critics.parameters(), lr=critic_lr, weight_decay=critic_wd, betas=critic_betas, eps=critic_eps
        )

        if target_entropy_cfg == "auto":
            target_entropy = -float(getattr(hp, "output_dim", 1))
        else:
            target_entropy = float(target_entropy_cfg)

        if alpha_mode == "auto":
            log_alpha = torch.tensor(math.log(max(alpha_init, 1e-12)), device=device, requires_grad=True)
            o_alpha = torch.optim.Adam([log_alpha], lr=alpha_lr)
        elif alpha_mode == "fixed":
            log_alpha = None
            o_alpha = None
        else:
            raise ValueError(f"Unsupported sac.entropy.alpha.mode: {alpha_mode}")

        buf = ReplayBuffer(
            replay_buffer_size,
            int(getattr(hp, "input_dim", 0)),
            int(getattr(hp, "output_dim", 0)),
        )

        hist_path = out_dir / "train_history.json"
        hist = {"tariff": tariff, "eval": [], "episodes": []}

        best = eval_runs(base_cfg, val_runs, tariff, actor, device, state_mask, deterministic_eval, reward_scale) if val_runs else {}
        for m in report_metrics:
            if m not in best:
                raise KeyError(
                    f"Evaluation metric '{m}' requested in evaluation.report_metrics is not produced by eval_runs()."
                )

        torch.save(
            {
                "model_state_dict": actor.mean_net.state_dict(),
                "actor_state_dict": actor.state_dict(),
                "critics_state_dict": critics.state_dict(),
                "target_critics_state_dict": tcritics.state_dict(),
                "hyperparameters": hp.to_dict(),
                "state_mask": il.get("state_mask", None),
                "log_alpha": (None if log_alpha is None else float(log_alpha.detach().cpu().item())),
                "steps": 0,
                "best_val": best,
                "run_config": rc,
            },
            out_dir / "best.pt",
        )

        print(f"\n[TARIFF] {tariff} out={out_dir}")
        if val_runs:
            print(
                f"[BASE] reward/day={best['reward_per_day']:.6f} "
                f"grid_import_peak_kw={best['grid_import_peak_kw']:.6f} "
                f"grid_import_total_kwh={best['grid_import_total_kwh']:.6f}"
            )

        steps = 0
        target_update_counter = 0

        # IMPORTANT CHANGE:
        # - Stage steps are determined by train_runs (sum of steps_to_run).
        # - The 'stages[*].steps' field is treated as informative only.
        for stage_id in stage_order:
            st = stage_by_id[stage_id]

            run_list = [rr for rr in train_runs if int(rr.get("stage_id", 1)) == stage_id]
            if not run_list:
                print(f"[STAGE {stage_id}] skipped (no train_runs for this stage_id)")
                continue

            stage_steps_runs = sum(int(rr.get("steps_to_run", 0)) for rr in run_list)
            stage_steps_cfg = int(st.get("steps", 0))

            if stage_steps_cfg != stage_steps_runs:
                print(f"[STAGE {stage_id}] overriding steps: cfg={stage_steps_cfg} -> runs_sum={stage_steps_runs}")

            stage_steps = stage_steps_runs

            collect_det = bool(st.get("collect_deterministic", stage_id in (1, 4)))

            stage_alpha_max = float((st.get("alpha_bounds", {}) or {}).get("alpha_max", alpha_max_global))
            stage_alpha_max = float(min(alpha_max_global, max(alpha_min, stage_alpha_max)))

            bur = (st.get("bursting", {}) or {})
            bursting_enabled = bool(bur.get("enabled", False))
            burst_steps = int(bur.get("burst_steps", 0) or 0)
            cooldown_steps = int(bur.get("cooldown_steps", 0) or 0)
            cycle_steps = max(burst_steps + cooldown_steps, 1)
            stage_step = 0

            print(
                f"\n[STAGE {stage_id}] name={st.get('name','')} steps={stage_steps} "
                f"collect_deterministic={collect_det} alpha_max={stage_alpha_max} bursting={bursting_enabled}"
            )

            pbar = tqdm(total=stage_steps, desc=f"{tariff} | stage {stage_id}", leave=False)

            for rr in run_list:
                df = load_df(rpath(rr["data"]))
                base = pd.to_datetime(rr["start_date"], dayfirst=True)
                days = int(rr["days"])
                steps_to_run = int(rr.get("steps_to_run", 0))
                run_label = str(rr.get("run_label", "RUN"))

                run_steps = 0
                while run_steps < steps_to_run:
                    ep_days = sample_episode_days(st, max_days=days)
                    off = 0 if days <= ep_days else int(np.random.randint(0, days - ep_days + 1))
                    sdate = (base + pd.Timedelta(days=int(off))).strftime("%d/%m/%Y %H:%M")

                    env = make_env(base_cfg, df, sdate, int(ep_days), tariff, state_mask)
                    o = env.reset()[0]
                    te = tr = False

                    hist["episodes"].append({"stage_id": stage_id, "run_label": run_label, "start_date": sdate, "days": ep_days})

                    while not (te or tr) and run_steps < steps_to_run:
                        # Bursting schedule (if enabled) overrides deterministic collection
                        if bursting_enabled and burst_steps > 0 and cooldown_steps > 0:
                            in_burst = (stage_step % cycle_steps) < burst_steps
                            collect_det_step = False if in_burst else True
                        else:
                            collect_det_step = collect_det

                        ot = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                        a = actor.act(ot, deterministic=collect_det_step).squeeze(0).detach().cpu().numpy()
                        no, r, te, tr = safe_step(env, a)
                        r_scaled = float(reward_scale) * float(r)

                        buf.add(o, a, r_scaled, no, float(te or tr))
                        o = no

                        steps += 1
                        run_steps += 1
                        stage_step += 1
                        pbar.update(1)

                        # Learning
                        if steps >= learning_starts and buf.size() >= batch_size:
                            for _ in range(updates_per_env_step):
                                o_t, a_t, r_t, no_t, d_t = buf.sample(batch_size, device)

                                na_t, nlp = actor.sample(no_t)
                                q1t, q2t = tcritics(no_t, na_t)

                                if log_alpha is not None:
                                    alpha_val = float(torch.exp(log_alpha).detach().cpu().item())
                                else:
                                    alpha_val = float(alpha_init)

                                alpha_val = float(np.clip(alpha_val, alpha_min, stage_alpha_max))
                                alpha_t = torch.as_tensor(alpha_val, device=device)

                                y = r_t + (1.0 - d_t) * gamma * (torch.min(q1t, q2t) - alpha_t * nlp)

                                q1, q2 = critics(o_t, a_t)
                                loss_crit = torch.nn.functional.mse_loss(q1, y.detach()) + torch.nn.functional.mse_loss(q2, y.detach())
                                o_crit.zero_grad(set_to_none=True)
                                loss_crit.backward()
                                if grad_clip_norm and grad_clip_norm > 0.0:
                                    torch.nn.utils.clip_grad_norm_(critics.parameters(), grad_clip_norm)
                                o_crit.step()

                                a2, lp2 = actor.sample(o_t)
                                loss_actor = (-(torch.min(*critics(o_t, a2)) - alpha_t * lp2).mean())
                                o_actor.zero_grad(set_to_none=True)
                                loss_actor.backward()
                                if grad_clip_norm and grad_clip_norm > 0.0:
                                    torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip_norm)
                                o_actor.step()

                                if log_alpha is not None and o_alpha is not None:
                                    loss_alpha = (-(log_alpha * (lp2.detach() + target_entropy)).mean())
                                    o_alpha.zero_grad(set_to_none=True)
                                    loss_alpha.backward()
                                    o_alpha.step()

                                target_update_counter += 1
                                if target_update_every <= 1 or (target_update_counter % target_update_every == 0):
                                    with torch.no_grad():
                                        for p, tp in zip(critics.parameters(), tcritics.parameters()):
                                            tp.data.mul_(1.0 - tau).add_(tau * p.data)

                        # Evaluation + best model selection (strictly follows JSON)
                        if val_runs and (steps % eval_every_steps == 0):
                            cur = eval_runs(base_cfg, val_runs, tariff, actor, device, state_mask, deterministic_eval, reward_scale)
                            for m in report_metrics:
                                if m not in cur:
                                    raise KeyError(
                                        f"Evaluation metric '{m}' requested in evaluation.report_metrics is not produced by eval_runs()."
                                    )
                            cur.update({"steps": steps, "stage_id": stage_id})
                            hist["eval"].append(cur)
                            dump_json(hist_path, hist)

                            improved = is_better_lexicographic(cur, best, best_sel) if best_sel_enabled and best else True
                            if improved:
                                best = dict(cur)
                                torch.save(
                                    {
                                        "model_state_dict": actor.mean_net.state_dict(),
                                        "actor_state_dict": actor.state_dict(),
                                        "critics_state_dict": critics.state_dict(),
                                        "target_critics_state_dict": tcritics.state_dict(),
                                        "hyperparameters": hp.to_dict(),
                                        "state_mask": il.get("state_mask", None),
                                        "log_alpha": (None if log_alpha is None else float(log_alpha.detach().cpu().item())),
                                        "steps": steps,
                                        "best_val": best,
                                        "run_config": rc,
                                    },
                                    out_dir / "best.pt",
                                )
                                print(
                                    f"[BEST] steps={steps} stage={stage_id} "
                                    f"grid_import_peak_kw={best['grid_import_peak_kw']:.6f} reward/day={best['reward_per_day']:.6f}"
                                )

            pbar.close()

        torch.save(
            {
                "model_state_dict": actor.mean_net.state_dict(),
                "actor_state_dict": actor.state_dict(),
                "critics_state_dict": critics.state_dict(),
                "target_critics_state_dict": tcritics.state_dict(),
                "hyperparameters": hp.to_dict(),
                "state_mask": il.get("state_mask", None),
                "log_alpha": (None if log_alpha is None else float(log_alpha.detach().cpu().item())),
                "steps": steps,
                "best_val": best,
                "run_config": rc,
            },
            out_dir / "last.pt",
        )

        print(
            f"[DONE] {tariff} steps={steps} "
            f"best_grid_import_peak_kw={best.get('grid_import_peak_kw', float('nan')):.6f} "
            f"best_reward/day={best.get('reward_per_day', float('nan')):.6f}"
        )
        print(f"[HIST] {hist_path}")


if __name__ == "__main__":
    main()
