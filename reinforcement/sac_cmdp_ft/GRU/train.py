"""GRU SAC-CMDP training.

Clean baseline inspired by Lagrangian SAC with separate reward and cost
critics. No elite replay, KL anchor, actor BC, canary, or phase transition.

Run:
    python reinforcement/sac_cmdp/GRU/train.py
"""
import json
import os
import sys
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
from environment.shaping import AnticipatoryShapingWrapper
from reinforcement.sac_cmdp.utils import (
    SafetyLayer, Hyperparameters, ReplayBuffer, Temperature, CostLambda,
    soft_update, reward_critic_step, cost_critic_step, actor_step,
    alpha_step, lambda_step, EpisodeGen, Audit, AsyncEval, EvalRunner,
    collect_streams, cost_tensor, EMA,
)
from models.GRU.model import load_actor, load_critic
from tools import EpisodeBar, UpdateBar, episode_bars, update_train_postfix

RUN_SUFFIX = os.environ.get("RUN_SUFFIX", "")
_DEFAULT_TARIFFS = ["tar_s", "tar_w", "tar_sw", "tar_flat", "tar_tou"]
TARIFFS = [t.strip() for t in os.environ.get("RUN_TARIFFS", ",".join(_DEFAULT_TARIFFS)).split(",") if t.strip()]


def _active_costs(cmdp_cfg):
    costs = cmdp_cfg.get("costs", {})
    return {name: cfg for name, cfg in costs.items() if bool(cfg.get("enabled", False))}


def train_tariff(tariff: str):
    with open(ALGO_ROOT / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(MODEL_ROOT / "model.json", encoding="utf-8") as f:
        model_cfg = json.load(f)
    model_cfg["actor"]["parameters"] = str(PROJECT_ROOT / "data" / "parameters.json")
    with open(model_cfg["actor"]["parameters"], encoding="utf-8") as f:
        env_params = json.load(f)

    hp = Hyperparameters(cfg["train"])
    cmdp = cfg["train"].get("cmdp", {})
    costs_cfg = _active_costs(cmdp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)

    actor = load_actor(model_cfg["actor"], device=device)
    # Optional behavior cloning initialization for FT mode.
    bc_init_path = cfg["train"].get("bc_init_checkpoint")
    if bc_init_path:
        bc_full = PROJECT_ROOT / bc_init_path
        bc_state = torch.load(bc_full, map_location=device)
        actor_state = bc_state["actor"] if isinstance(bc_state, dict) and "actor" in bc_state else bc_state
        actor.load_state_dict(actor_state)
        tqdm.write(f"[init] loaded BC actor from {bc_full}")
    ema_actor = EMA(actor, tau=cfg["train"].get("ema_tau", 0.99))
    reward_critic = load_critic(model_cfg["critic"], device=device)
    reward_target = load_critic(model_cfg["critic"], device=device)
    reward_target.load_state_dict(reward_critic.state_dict())

    cost_critics = {name: load_critic(model_cfg["critic"], device=device) for name in costs_cfg}
    cost_targets = {name: load_critic(model_cfg["critic"], device=device) for name in costs_cfg}
    for name in costs_cfg:
        cost_targets[name].load_state_dict(cost_critics[name].state_dict())

    temp = Temperature(init_alpha=hp.init_alpha).to(device)
    safety = SafetyLayer(actor.bess, actor.ev, env_params).to(device).eval()
    lambdas = {
        name: CostLambda(
            init_lambda=float(ccfg.get("lambda_init", 0.1)),
            budget=float(ccfg.get("budget", 0.0)),
            max_lambda=ccfg.get("lambda_max", None),
        ).to(device)
        for name, ccfg in costs_cfg.items()
    }

    opt_actor = torch.optim.Adam(actor.parameters(), lr=hp.actor_lr)
    opt_reward = torch.optim.Adam(reward_critic.parameters(), lr=hp.critic_lr)
    opt_cost = {name: torch.optim.Adam(cost_critics[name].parameters(), lr=hp.critic_lr) for name in costs_cfg}
    opt_alpha = torch.optim.Adam([temp.log_alpha], lr=hp.alpha_lr)
    opt_lambda = {
        name: torch.optim.Adam([lambdas[name].log_lambda], lr=float(costs_cfg[name].get("lambda_lr", 3e-4)))
        for name in costs_cfg
    }

    buffer = ReplayBuffer(
        hp.buffer_size, obs_dim=model_cfg["actor"]["input_dim"], act_dim=3,
        device=device, history_len=hp.history_len, n_step=hp.n_step, gamma=hp.gamma,
        cost_names=list(costs_cfg),
    )

    episodes = EpisodeGen(cfg, str(PROJECT_ROOT / "data"), seed=hp.seed)
    out_dir = PROJECT_ROOT / "paper" / "train" / "sac_cmdp_ft" / ARCH_NAME / f"{tariff}{RUN_SUFFIX}"
    eval_runner = EvalRunner(
        actor_module="models.GRU.model", actor_cfg=model_cfg["actor"],
        parameters=env_params, tariff=tariff, history_len=hp.history_len,
        project_root=str(PROJECT_ROOT), n_workers=cfg["train"].get("eval_workers", 4),
    )
    async_eval = AsyncEval(eval_runner, cfg["val"], max_pending=cfg["train"].get("max_pending_evals", 2))
    audit = Audit(out_dir / "audit_training.csv", every=5)

    safe_rollout = bool(cmdp.get("safe_rollout", False))
    rollout_safety = safety if safe_rollout else None
    checkpoint_scores = {
        "best": -float("inf"),
        "best_robust": -float("inf"),
        "best_worst": -float("inf"),
    }
    last_eval_summary = None
    step_count = 0
    tqdm.write(f"[setup] CMDP {tariff}: costs={list(costs_cfg)} safe_rollout={safe_rollout}")

    def immediate_costs(obs, act):
        if not costs_cfg:
            return {}
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        act_t = torch.as_tensor(act, dtype=torch.float32, device=device).view(1, -1)
        with torch.inference_mode():
            return {
                name: float(cost_tensor(name, obs_t, act_t, safety, env_params, ccfg).detach().cpu())
                for name, ccfg in costs_cfg.items()
            }

    def snapshot():
        return {
            "actor": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
            "ema": {k: v.detach().cpu().clone() for k, v in ema_actor.state_dict().items()},
            "reward_critic": {k: v.detach().cpu().clone() for k, v in reward_critic.state_dict().items()},
            "cost_critics": {
                name: {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}
                for name, critic in cost_critics.items()
            },
            "alpha": {k: v.detach().cpu().clone() for k, v in temp.state_dict().items()},
            "lambdas": {
                name: {k: v.detach().cpu().clone() for k, v in lam.state_dict().items()}
                for name, lam in lambdas.items()
            },
        }

    def save_checkpoint(label, job, summary, metric, value):
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(job["snapshot"], out_dir / f"{label}.pt")
        meta = {
            "tariff": tariff,
            "episode": int(job["episode"]),
            "metric": metric,
            "value": None if value is None else float(value),
            "train_reward": float(job["context"].get("train_reward", float("nan"))),
            "summary": summary or {},
        }
        with open(out_dir / f"{label}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def finish_evals(jobs):
        nonlocal last_eval_summary
        for job in jobs:
            eval_summary = job["summary"]
            last_eval_summary = eval_summary
            improvements = []
            for label, metric in (
                ("best", "mean_reward"),
                ("best_robust", "robust_reward"),
                ("best_worst", "worst_reward"),
            ):
                value = float(eval_summary.get(metric, -float("inf")))
                if value > checkpoint_scores[label]:
                    checkpoint_scores[label] = value
                    save_checkpoint(label, job, eval_summary, metric, value)
                    improvements.append(label)
            save_checkpoint("last", job, eval_summary, "last_eval", eval_summary.get("mean_reward"))
            tqdm.write(
                f"[eval] {tariff} ep {job['episode']:4d} -> "
                f"mean={eval_summary['mean_reward']:.2f} "
                f"robust={eval_summary.get('robust_reward', float('nan')):.2f} "
                f"worst={eval_summary.get('worst_reward', float('nan')):.2f} "
                f"best={checkpoint_scores['best']:.2f}"
                + (f"  ★ {','.join(improvements)}" if improvements else "")
            )
            row = {
                "episode": job["episode"],
                "train_reward": job["context"]["train_reward"],
                "best_eval_reward": checkpoint_scores["best"],
                "best_robust_reward": checkpoint_scores["best_robust"],
                "best_worst_reward": checkpoint_scores["best_worst"],
                **job["context"]["agg"],
                **{f"eval_{k}": v for k, v in eval_summary.items()},
            }
            audit.add(row)
            audit.flush()

    for ep, pbar in episode_bars(hp.train_episodes, hp.warmup_episodes, tariff):
        finish_evals(async_eval.collect())
        envs = {
            stream: AnticipatoryShapingWrapper(
                SmartHomeEnv(
                    episodes.load(stream), env_params, start=episodes.sample(stream),
                    days=hp.days, BESS_SoC=np.random.uniform(0.1, 0.9),
                    tariff=tariff, track_operation=False,
                ),
                params=env_params, gamma=hp.gamma,
                omega=cfg["train"].get("shaping_omega", 0.0),
                typical_drop=cfg["train"].get("shaping_typical_drop", 0.137),
            )
            for stream in ("cy", "wy")
        }
        with EpisodeBar(envs, ep, hp.train_episodes) as episode_bar:
            transitions, rewards = collect_streams(
                envs, actor, hp.history_len, device, deterministic=False,
                progress=episode_bar, safety=rollout_safety,
            )
        train_reward = float(sum(rewards.values()))
        for stream, stream_transitions in transitions.items():
            for o, a, r, no, d in stream_transitions:
                costs = immediate_costs(o, a)
                buffer.add(o, a, r * hp.reward_scale, no, d, stream_id=stream, costs=costs)

        audit.open()
        if ep >= hp.warmup_episodes:
            n_transitions = sum(len(stream_transitions) for stream_transitions in transitions.values())
            n_updates = n_transitions // hp.update_every_steps
            with UpdateBar(n_updates, ep) as update_bar:
                for _ in range(n_updates):
                    step_count += 1
                    batch = buffer.sample(hp.batch_size)
                    row = reward_critic_step(
                        reward_critic, reward_target, actor, batch, temp,
                        opt_reward, grad_clip=hp.grad_clip,
                    )
                    for name, ccfg in costs_cfg.items():
                        row.update(cost_critic_step(
                            name, cost_critics[name], cost_targets[name], actor, batch,
                            opt_cost[name], safety, env_params, ccfg, grad_clip=hp.grad_clip,
                        ))
                    if ep >= hp.actor_update_start_episode and (step_count % hp.actor_update_every) == 0:
                        ainfo = actor_step(
                            actor, reward_critic, cost_critics, lambdas, batch, temp,
                            opt_actor, grad_clip=hp.grad_clip,
                        )
                        q_costs = ainfo.pop("q_costs")
                        al_info = alpha_step(temp, ainfo.pop("logp"), hp.target_entropy, opt_alpha)
                        row.update(ainfo)
                        row.update(al_info)
                        for name, q_cost in q_costs.items():
                            linfo = lambda_step(lambdas[name], q_cost, opt_lambda[name])
                            row[f"{name}_lambda_loss"] = linfo["lambda_loss"]
                            row[f"{name}_lambda_value"] = linfo["lambda_value"]
                        ema_actor.update(actor)
                    soft_update(reward_critic, reward_target, hp.tau)
                    for name in costs_cfg:
                        soft_update(cost_critics[name], cost_targets[name], hp.tau)
                    audit.step(**row)
                    update_bar.step(row)

        eval_submitted = False
        if (ep + 1) % hp.evaluate_every == 0:
            while len(async_eval.pending) >= async_eval.max_pending:
                finish_evals(async_eval.collect(wait=True, max_items=1))
            state = snapshot()
            # Ablation: evaluate the live actor. EMA is still tracked/saved in
            # checkpoints, but eval no longer mixes gamma effects with EMA lag.
            async_eval.submit(ep, state["actor"], state, {"train_reward": train_reward, "agg": audit.aggregate()})
            eval_submitted = True
            tqdm.write(f"[submit] {tariff} ep {ep:4d} (pending={len(async_eval.pending)})")

        agg = audit.aggregate()
        if not eval_submitted:
            audit.add({
                "episode": ep,
                "train_reward": train_reward,
                "best_eval_reward": checkpoint_scores["best"],
                "best_robust_reward": checkpoint_scores["best_robust"],
                "best_worst_reward": checkpoint_scores["best_worst"],
                **agg,
            })
            audit.flush()
        finish_evals(async_eval.collect())
        update_train_postfix(pbar, train_reward, last_eval_summary, checkpoint_scores["best"], agg, len(async_eval.pending))

    finish_evals(async_eval.drain())
    audit.flush(force=True)
    eval_runner.close()
    (out_dir / ".train_done").touch()


def main():
    for tariff in TARIFFS:
        train_tariff(tariff)


if __name__ == "__main__":
    main()
