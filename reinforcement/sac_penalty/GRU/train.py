"""GRU SAC training: local exploration around a moving KL anchor.

Differences vs v9d:
- Dynamic anchor updates use a small improvement margin plus confirmation.
- Phase 2 uses a smaller actor LR and a slightly higher target entropy.

Inherited from v9d:
- `anchor.decay_beta()` is called once per episode, not once per actor update.
- KL/projection audit columns are enabled through `reinforcement.sac_penalty.utils.Audit`.

Inherited from v9b:
- `safe_rollout=False` (env penalty teaches natural safety; aligned w/ deploy)
- Transition is patience-based (not fixed ep): triggers `anchor_patience` evals
  after the last *qualifying* improvement, only after `phase1_min_episodes`.
- Anchor target requires BOTH `eval_mean_reward ≥ anchor_min_reward` AND
  `eval_bess_abs_power_mean ≥ anchor_min_bess_power`.
- Optimizer state of the actor is preserved on transition (no Adam spike).
- Anchor is updated dynamically post-transition if a strictly better joint
  improvement is found (`anchor_dynamic=True`).

Run:
    python models/GRU/2-RL/train.py
"""
import json
import sys
import importlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALGO_ROOT = Path(__file__).resolve().parent
ARCH_NAME = ALGO_ROOT.name
MODEL_ROOT = PROJECT_ROOT / "models" / ARCH_NAME
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ALGO_ROOT))

from environment import SmartHomeEnv
from reinforcement.sac_penalty.utils import (
    SafetyLayer, Hyperparameters, ReplayBuffer,
    Temperature, soft_update, alpha_step,
    critic_step_ens, actor_step_ens,
    EpisodeGen, Audit, AsyncEval, EvalRunner, collect_streams,
    EMA, EarlyStop, CriticEnsemble, PIDLambda, KLAnchor,
)
from reinforcement.sac_penalty.utils.safe_eval import SafeEvalRunner
from tools import EpisodeBar, UpdateBar, episode_bars, update_train_postfix

MODEL_MODULE = f"models.{ARCH_NAME}.model"
_model = importlib.import_module(MODEL_MODULE)
load_actor = _model.load_actor
load_critic = _model.load_critic

TARIFFS = ["tar_s", "tar_w", "tar_sw", "tar_flat", "tar_tou"]


def _qualifies(es, min_reward, min_bess):
    """Joint criterion: needs BOTH economic reward and BESS activity."""
    r = float(es.get("mean_reward", -1e9))
    b = float(es.get("bess_abs_power_mean", 0.0))
    return r >= min_reward and b >= min_bess


def train_tariff(tariff: str):
    with open(ALGO_ROOT / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(MODEL_ROOT / "model.json", encoding="utf-8") as f:
        model_cfg = json.load(f)
    model_cfg["actor"]["parameters"] = str(PROJECT_ROOT / "data" / "parameters.json")
    with open(model_cfg["actor"]["parameters"], encoding="utf-8") as f:
        env_params = json.load(f)

    hp = Hyperparameters(cfg["train"])
    sota = cfg["train"].get("sota", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(hp.seed); np.random.seed(hp.seed)

    # ---- Modules ----
    actor = load_actor(model_cfg["actor"], device=device)
    ensemble = CriticEnsemble(
        lambda: load_critic(model_cfg["critic"], device=device),
        n=sota.get("ensemble_size", 2),
    ).to(device)
    safety = SafetyLayer(actor.bess, actor.ev, env_params).to(device)
    safety.eval()
    safe_rollout = bool(sota.get("safe_rollout", False))
    rollout_safety = safety if safe_rollout else None
    project_actor_q = bool(sota.get("project_actor_q", False))
    project_target_q = bool(sota.get("project_target_q", False))
    proj_penalty_coef_cfg = float(sota.get("projection_penalty_coef", 0.0))

    temp = Temperature(init_alpha=hp.init_alpha).to(device)
    lam = PIDLambda(
        K_P=sota.get("pid_K_P", 0.0), K_I=sota.get("pid_K_I", 0.0),
        K_D=sota.get("pid_K_D", 0.0), budget=hp.violation_budget,
        init_integral=hp.init_lambda,
        max_lambda=sota.get("lambda_max", None),
    ).to(device)
    ema_actor = EMA(actor, tau=sota.get("ema_tau", 0.99))
    early_stop = EarlyStop(
        patience=hp.early_stop_patience,
        min_episodes=hp.min_episodes_before_early_stop,
    )
    anchor = KLAnchor(beta=0.0,  # starts disabled; turned on at transition
                       decay=sota.get("kl_decay", 0.998))

    opt_actor = torch.optim.Adam(actor.parameters(), lr=hp.actor_lr)
    opt_critic = torch.optim.Adam(ensemble.parameters(), lr=hp.critic_lr)
    opt_alpha = torch.optim.Adam([temp.log_alpha], lr=hp.alpha_lr)

    buffer = ReplayBuffer(
        hp.buffer_size, obs_dim=model_cfg["actor"]["input_dim"], act_dim=3, device=device,
        history_len=hp.history_len, n_step=hp.n_step, gamma=hp.gamma,
    )

    episodes = EpisodeGen(cfg, str(PROJECT_ROOT / "data"), seed=hp.seed)

    out_dir = PROJECT_ROOT / "Results" / "train" / "reinforcement" / "sac_penalty" / ARCH_NAME / tariff
    eval_runner = EvalRunner(
        actor_module=MODEL_MODULE, actor_cfg=model_cfg["actor"],
        parameters=env_params, tariff=tariff, history_len=hp.history_len,
        project_root=str(PROJECT_ROOT), n_workers=cfg["train"].get("eval_workers", 4),
    )
    async_eval = AsyncEval(eval_runner, cfg["val"],
                            max_pending=cfg["train"].get("max_pending_evals", 2))
    safe_eval_every = max(1, int(sota.get("safe_eval_every", 5)))
    safe_eval_runner = SafeEvalRunner(
        actor_module=MODEL_MODULE, actor_cfg=model_cfg["actor"],
        parameters=env_params, tariff=tariff, history_len=hp.history_len,
        project_root=str(PROJECT_ROOT), n_workers=cfg["train"].get("eval_workers", 4),
    )
    async_safe_eval = AsyncEval(safe_eval_runner, cfg["val"],
                                max_pending=cfg["train"].get("max_pending_evals", 2))
    audit = Audit(out_dir / "audit_training.csv", every=5)
    safe_audit = Audit(out_dir / "audit_safe_eval.csv", every=1)

    # ---- v9e phase / anchor state ----
    min_reward = float(sota.get("anchor_min_reward", -52.0))
    min_bess = float(sota.get("anchor_min_bess_power", 0.15))
    patience = int(sota.get("anchor_patience", 8))
    phase1_min_episodes = int(sota.get("phase1_min_episodes", 12))
    phase1_max_episodes = int(sota.get("phase1_max_episodes", 60))
    reset_pid = bool(sota.get("phase_transition_reset_pid", True))
    reset_actor_opt = bool(sota.get("phase_transition_reset_actor_opt", False))
    anchor_dynamic = bool(sota.get("anchor_dynamic", True))
    phase2_lambda_max = float(sota.get("phase2_lambda_max", sota.get("lambda_max", 0.0)))
    phase2_kl_beta = float(sota.get("phase2_kl_beta", sota.get("kl_beta", 0.03)))
    phase2_actor_lr_factor = float(sota.get("phase2_actor_lr_factor", 0.5))
    anchor_margin = float(sota.get("anchor_improvement_margin", 0.1))
    anchor_confirm_evals = max(1, int(sota.get("anchor_confirm_evals", 2)))
    phase2_target_entropy = hp.target_entropy + float(sota.get("phase2_target_entropy_offset", 0.0))
    target_entropy = [hp.target_entropy]

    state = {
        "phase": 1,
        "best_qualifying_reward": -1e9,
        "best_qualifying_state": None,
        "best_qualifying_ep": -1,
        "evals_since_best": 0,
        "transition_ep": None,
        "post_transition_best_reward": -1e9,  # for anchor_dynamic
        "anchor_candidate_state": None,
        "anchor_candidate_reward": -1e9,
        "anchor_candidate_ep": -1,
        "anchor_candidate_count": 0,
        "anchor_updates": 0,
    }
    last_eval_summary = None
    checkpoint_scores = {
        "best": -float("inf"),
        "best_robust": -float("inf"),
        "best_worst": -float("inf"),
    }
    step_count = [0]
    m_target = sota.get("redq_m", 2)
    tqdm.write(
        f"[setup] {tariff} GRU safe_rollout={safe_rollout} "
        f"anchor[min_reward={min_reward}, min_bess={min_bess}, "
        f"patience={patience}, min_ep={phase1_min_episodes}, "
        f"max_ep={phase1_max_episodes}, lr_factor={phase2_actor_lr_factor}, "
        f"margin={anchor_margin}, confirm={anchor_confirm_evals}]"
    )

    def snapshot():
        return {
            "actor": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
            "critic": {k: v.detach().cpu().clone() for k, v in ensemble.state_dict().items()},
            "alpha": {k: v.detach().cpu().clone() for k, v in temp.state_dict().items()},
            "lambda": {k: v.detach().cpu().clone() for k, v in lam.state_dict().items()},
            "ema": {k: v.detach().cpu().clone() for k, v in ema_actor.state_dict().items()},
        }

    def save_checkpoint(label, job, summary=None, metric=None, value=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(job["snapshot"], out_dir / f"{label}.pt")
        meta = {
            "architecture": ARCH_NAME,
            "tariff": tariff,
            "episode": int(job["episode"]),
            "metric": metric,
            "value": None if value is None else float(value),
            "train_reward": float(job["context"].get("train_reward", np.nan)),
            "phase": int(state["phase"]),
            "summary": summary or {},
        }
        with open(out_dir / f"{label}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def transition_to_phase2(ep, reason=""):
        """Reset PID, set KL anchor from best qualifying EMA, keep opt_actor.

        Per design analysis: opt_actor.state is NOT cleared (avoids Adam
        first-step spike from m1/m2 reset). lam integral is reset because
        phase 2 starts a new constraint regime.
        """
        state["phase"] = 2
        state["transition_ep"] = ep
        state["post_transition_best_reward"] = state["best_qualifying_reward"]
        target_entropy[0] = phase2_target_entropy
        lam.max_lambda = phase2_lambda_max
        if reset_pid:
            with torch.no_grad():
                lam.integral.zero_(); lam.prev_v.zero_(); lam._lam.zero_()
        if reset_actor_opt:
            opt_actor.state.clear()
        for group in opt_actor.param_groups:
            group["lr"] *= phase2_actor_lr_factor
        if state["best_qualifying_state"] is not None:
            actor.load_state_dict(state["best_qualifying_state"])
            ema_actor.model.load_state_dict(state["best_qualifying_state"])
            anchor.snapshot(actor)
            anchor.beta = phase2_kl_beta
            tqdm.write(
                f"[phase2] {tariff} ep {ep} -> anchor set "
                f"(reward={state['best_qualifying_reward']:.2f}, "
                f"from ep {state['best_qualifying_ep']}, β={anchor.beta}, {reason})"
            )
        else:
            anchor.beta = 0.0  # no qualifying policy → falls back to v7-like behavior
            tqdm.write(
                f"[phase2] {tariff} ep {ep} -> NO qualifying anchor found, "
                f"continuing without KL ({reason})"
            )

    def maybe_update_anchor_dynamic(ep, es, snap):
        """Post-phase-2: if a strictly better joint-qualifying eval appears,
        update the anchor (still soft via KL, not a hard policy reset)."""
        if not anchor_dynamic or state["phase"] != 2 or anchor.anchor is None:
            return
        if not _qualifies(es, min_reward, min_bess):
            state["anchor_candidate_state"] = None
            state["anchor_candidate_count"] = 0
            return
        if state["anchor_candidate_state"] is None:
            if es["mean_reward"] <= state["post_transition_best_reward"] + anchor_margin:
                return
            state["anchor_candidate_state"] = {k: v.clone() for k, v in snap["ema"].items()}
            state["anchor_candidate_reward"] = es["mean_reward"]
            state["anchor_candidate_ep"] = ep
            state["anchor_candidate_count"] = 1
            tqdm.write(
                f"[anchor?] {tariff} ep {ep} -> candidate "
                f"(reward={es['mean_reward']:.2f}, bess={es['bess_abs_power_mean']:.3f})"
            )
            return
        if es["mean_reward"] > state["anchor_candidate_reward"]:
            state["anchor_candidate_state"] = {k: v.clone() for k, v in snap["ema"].items()}
            state["anchor_candidate_reward"] = es["mean_reward"]
            state["anchor_candidate_ep"] = ep
            state["anchor_candidate_count"] = 1
            return
        state["anchor_candidate_count"] += 1
        if state["anchor_candidate_count"] < anchor_confirm_evals:
            return  # require meaningful improvement
        state["post_transition_best_reward"] = state["anchor_candidate_reward"]
        ema_state = state["anchor_candidate_state"]
        ema_actor.model.load_state_dict(ema_state)
        anchor.snapshot(ema_actor.model)
        state["anchor_updates"] += 1
        tqdm.write(
            f"[anchor++] {tariff} ep {ep} -> anchor refreshed from ep "
            f"{state['anchor_candidate_ep']} (reward={state['anchor_candidate_reward']:.2f}, "
            f"confirm={state['anchor_candidate_count']})"
        )
        state["anchor_candidate_state"] = None
        state["anchor_candidate_count"] = 0

    def maybe_promote_phase1(ep, es, snap):
        """Track best joint-qualifying policy in phase 1. If new candidate
        improves: store and reset patience counter."""
        if state["phase"] != 1:
            return
        if not _qualifies(es, min_reward, min_bess):
            state["evals_since_best"] += 1
            return
        if es["mean_reward"] > state["best_qualifying_reward"]:
            state["best_qualifying_reward"] = es["mean_reward"]
            state["best_qualifying_ep"] = ep
            state["best_qualifying_state"] = {k: v.clone() for k, v in snap["ema"].items()}
            state["evals_since_best"] = 0
            tqdm.write(
                f"[phase1] {tariff} ep {ep} -> candidate updated "
                f"(reward={es['mean_reward']:.2f}, bess={es['bess_abs_power_mean']:.3f})"
            )
        else:
            state["evals_since_best"] += 1

    def maybe_trigger_transition(ep):
        """Patience-based trigger or hard cutoff at phase1_max_episodes."""
        if state["phase"] != 1:
            return
        if ep < phase1_min_episodes:
            return
        if state["best_qualifying_state"] is not None and state["evals_since_best"] >= patience:
            transition_to_phase2(ep, reason=f"patience={state['evals_since_best']}")
            return
        if ep >= phase1_max_episodes:
            transition_to_phase2(ep, reason=f"max_episodes={phase1_max_episodes}")

    def finish_evals(jobs):
        nonlocal last_eval_summary
        for job in jobs:
            es = job["summary"]
            last_eval_summary = es
            improved = early_stop.update(job["episode"], es["mean_reward"])
            for label, metric in (
                ("best", "mean_reward"),
                ("best_robust", "robust_reward"),
                ("best_worst", "worst_reward"),
            ):
                value = float(es.get(metric, -float("inf")))
                if value > checkpoint_scores[label]:
                    checkpoint_scores[label] = value
                    save_checkpoint(label, job, es, metric, value)
            save_checkpoint("last", job, es, "last_eval", es.get("mean_reward"))
            maybe_promote_phase1(job["episode"], es, job["snapshot"])
            maybe_update_anchor_dynamic(job["episode"], es, job["snapshot"])
            tqdm.write(
                f"[eval] {tariff} ep {job['episode']:4d} -> "
                f"mean={es['mean_reward']:.2f}  best={early_stop.best:.2f}  "
                f"bess={es.get('bess_abs_power_mean', float('nan')):.3f}  "
                f"phase={state['phase']}  patience={state['evals_since_best']}"
                + ("  ★" if improved else "")
            )
            row = {
                "episode": job["episode"],
                "train_reward": job["context"]["train_reward"],
                "best_eval_reward": early_stop.best,
                "phase": state["phase"],
                "evals_since_best": state["evals_since_best"],
                "anchor_candidate_count": state["anchor_candidate_count"],
                "anchor_updates": state["anchor_updates"],
                **job["context"]["agg"],
                **{f"eval_{k}": v for k, v in es.items()},
            }
            audit.add(row); audit.flush()

    def finish_safe_evals(jobs):
        for job in jobs:
            es = job["summary"]
            tqdm.write(
                f"[eval_safe] {tariff} ep {job['episode']:4d} -> mean={es['mean_reward']:.2f}"
            )
            row = {
                "episode": job["episode"],
                "train_reward": job["context"]["train_reward"],
                "best_eval_reward": early_stop.best,
                **job["context"]["agg"],
                **{f"eval_safe_{k}": v for k, v in es.items()},
            }
            safe_audit.add(row); safe_audit.flush()

    stopped = False
    for ep, pbar in episode_bars(hp.train_episodes, hp.warmup_episodes, tariff):
        finish_evals(async_eval.collect())
        finish_safe_evals(async_safe_eval.collect())
        maybe_trigger_transition(ep)
        if early_stop.should_stop(ep):
            tqdm.write(f"[stop] {tariff} early stop at ep {ep} (best={early_stop.best:.2f})")
            stopped = True
            break

        envs = {
            stream: SmartHomeEnv(episodes.load(stream), env_params,
                                  start=episodes.sample(stream), days=hp.days,
                                  BESS_SoC=np.random.uniform(0.1, 0.9),
                                  tariff=tariff, track_operation=False)
            for stream in ("cy", "wy")
        }
        audit.open()

        if ep >= hp.warmup_episodes:
            total_steps = sum(env.sim.end_idx - env.sim.t_idx for env in envs.values())
            update_bar = UpdateBar(total_steps // hp.update_every_steps, ep)
        else:
            update_bar = None

        def on_step(transition, key):
            o, a, r, no, d = transition
            buffer.add(o, a, r * hp.reward_scale, no, d, stream_id=key)
            if ep < hp.warmup_episodes or buffer.size < hp.batch_size:
                return
            step_count[0] += 1
            if step_count[0] % hp.update_every_steps != 0:
                return
            batch = buffer.sample(hp.batch_size)
            row = critic_step_ens(ensemble, actor, batch, temp, opt_critic,
                                   grad_clip=hp.grad_clip, m=m_target,
                                   safety=safety if project_target_q else None)
            if ep >= hp.actor_update_start_episode and step_count[0] % hp.actor_update_every == 0:
                ainfo = actor_step_ens(actor, ensemble, safety, batch, temp, lam,
                                        opt_actor, kl_anchor=anchor,
                                        grad_clip=hp.grad_clip,
                                        project_q=project_actor_q,
                                        proj_penalty_coef=proj_penalty_coef_cfg)
                al_info = alpha_step(temp, ainfo.pop("logp"), target_entropy[0], opt_alpha)
                ld_info = lam.step(ainfo.pop("viol"))
                ema_actor.update(actor)
                row.update({**ainfo, **al_info, **ld_info,
                            "actor_lr": opt_actor.param_groups[0]["lr"]})
            ensemble.soft_update(hp.tau)
            audit.step(**row)
            if update_bar is not None:
                update_bar.step(row)

        try:
            with EpisodeBar(envs, ep, hp.train_episodes) as episode_bar:
                transitions, rewards = collect_streams(
                    envs, actor, hp.history_len, device,
                    deterministic=False, progress=episode_bar, on_step=on_step,
                    safety=rollout_safety,
                )
        finally:
            if update_bar is not None:
                update_bar._flush(); update_bar.pbar.close()

        train_reward = float(sum(rewards.values()))

        eval_submitted = False
        if (ep + 1) % hp.evaluate_every == 0:
            while len(async_eval.pending) >= async_eval.max_pending:
                finish_evals(async_eval.collect(wait=True, max_items=1))
            ema_state = {k: v.detach().cpu().clone() for k, v in ema_actor.state_dict().items()}
            ctx = {"train_reward": train_reward, "agg": audit.aggregate()}
            async_eval.submit(ep, ema_state, snapshot(), ctx)
            eval_submitted = True
            tqdm.write(f"[submit] {tariff} ep {ep:4d} (pending={len(async_eval.pending)})")
            if (ep + 1) % safe_eval_every == 0:
                while len(async_safe_eval.pending) >= async_safe_eval.max_pending:
                    finish_safe_evals(async_safe_eval.collect(wait=True, max_items=1))
                async_safe_eval.submit(ep, ema_state, snapshot(), ctx)
                tqdm.write(f"[submit_safe] {tariff} ep {ep:4d} (pending={len(async_safe_eval.pending)})")

        agg = audit.aggregate()
        if not eval_submitted:
            row = {"episode": ep, "train_reward": train_reward,
                    "best_eval_reward": early_stop.best, **agg}
            audit.add(row); audit.flush()
        finish_evals(async_eval.collect())
        finish_safe_evals(async_safe_eval.collect())
        update_train_postfix(pbar, train_reward, last_eval_summary,
                              early_stop.best, agg,
                              len(async_eval.pending) + len(async_safe_eval.pending))
        if state["phase"] == 2 and anchor.ready():
            anchor.decay_beta()

    finish_evals(async_eval.drain())
    finish_safe_evals(async_safe_eval.drain())
    audit.flush(force=True)
    safe_audit.flush(force=True)
    eval_runner.close()
    safe_eval_runner.close()


def main():
    for tariff in TARIFFS:
        train_tariff(tariff)


if __name__ == "__main__":
    main()
