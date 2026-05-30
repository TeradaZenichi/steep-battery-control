"""Unified SAC trainer (one harness, config-driven switches).

All experiments share this trainer; an experiment is the base config plus a few
switches, so penalty vs CMDP differ ONLY in the safety formulation and every
other knob (online update loop, UTD, lrs, tau, entropy, n_step, episodes) is
identical. Switches in config["train"]:

  safety            : "cmdp" (cost critics + Lagrangian dual) | "penalty"
                      (feasibility stays in the scalar reward, no cost critics)
  ema_backup        : bool — use a lagged EMA actor for the Bellman backup action
  rlpd              : {enabled, prior_ratio, sources, demos_root} — 50/50 mixed
                      sampling from a static prior buffer of RB/RBS demos
  shaping_omega     : float — anticipatory EV potential (env wrapper)
  shaping_bess_omega: float — anticipatory BESS-arbitrage potential (env wrapper)
  bc_init           : path | {tariff: path} | null — warm-start the actor (FT)

Update schedule is ONLINE/interleaved (updates happen during the rollout, not
batched at episode end): every `update_every_steps` env steps a critic update;
every `actor_update_every` env steps an actor/alpha/(lambda) update.

Output: paper/train/<exp>/<arch>/<tariff>, where <exp> = algo_root.parent.name.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import contextlib
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment import SmartHomeEnv
from environment.shaping import AnticipatoryShapingWrapper
from reinforcement.sac_cmdp.utils import (
    SafetyLayer, Hyperparameters, ReplayBuffer, Temperature, CostLambda,
    soft_update, alpha_step, lambda_step,
    EpisodeGen, Audit, AsyncEval, EvalRunner, collect_streams, cost_tensor, EMA,
    load_prior_buffer, merge_batches,
)
from sac.common.updates import reward_critic_step, cost_critic_step, actor_step
from sac.common.tools import EpisodeBar, UpdateBar, episode_bars, update_train_postfix

_DEFAULT_TARIFFS = ["tar_s", "tar_w", "tar_sw", "tar_flat", "tar_tou"]


def _active_costs(train_cfg):
    if train_cfg.get("safety", "cmdp") != "cmdp":
        return {}
    costs = train_cfg.get("cmdp", {}).get("costs", {})
    return {name: c for name, c in costs.items() if bool(c.get("enabled", False))}


def _resolve_bc_init(bc_init, arch_name, tariff):
    if not bc_init:
        return None
    path = bc_init[tariff] if isinstance(bc_init, dict) else bc_init
    path = str(path).format(arch=arch_name, tariff=tariff)
    p = (PROJECT_ROOT / path) if not os.path.isabs(path) else Path(path)
    return p if p.exists() else None


def train_tariff(algo_root: Path, arch_name: str, exp_name: str, tariff: str, run_suffix: str = ""):
    load_actor = __import__(f"models.{arch_name}.model", fromlist=["load_actor"]).load_actor
    load_critic = __import__(f"models.{arch_name}.model", fromlist=["load_critic"]).load_critic

    with open(algo_root / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    model_cfg = json.load(open(PROJECT_ROOT / "models" / arch_name / "model.json", encoding="utf-8"))
    droq_dropout = float(cfg["train"].get("droq_dropout", 0.0))
    if droq_dropout > 0:
        model_cfg["critic"]["dropout_rate"] = droq_dropout
    n_critics = int(cfg["train"].get("n_critics", 2))
    m_target = cfg["train"].get("m_target")   # None = use all; integer = REDQ subset min
    m_target = int(m_target) if m_target else None
    if n_critics != 2:
        model_cfg["critic"]["n_critics"] = n_critics
    model_cfg["actor"]["parameters"] = str(PROJECT_ROOT / "data" / "parameters.json")
    env_params = json.load(open(model_cfg["actor"]["parameters"], encoding="utf-8"))

    tcfg = cfg["train"]
    hp = Hyperparameters(tcfg)
    safety_mode = tcfg.get("safety", "cmdp")
    ema_backup = bool(tcfg.get("ema_backup", False))
    rlpd_cfg = tcfg.get("rlpd", {})
    rlpd_on = bool(rlpd_cfg.get("enabled", False))
    costs_cfg = _active_costs(tcfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(tcfg.get("precision", "fp32")).lower()
    if precision == "bf16" and device.type == "cuda":
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif precision == "fp16" and device.type == "cuda":
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        amp_ctx = contextlib.nullcontext
        precision = "fp32"
    seed = int(os.environ.get("RUN_SEED", hp.seed))
    torch.manual_seed(seed)
    np.random.seed(seed)

    actor = load_actor(model_cfg["actor"], device=device)
    bc_path = _resolve_bc_init(tcfg.get("bc_init"), arch_name, tariff)
    if bc_path is not None:
        state = torch.load(bc_path, map_location=device)
        actor.load_state_dict(state["actor"] if isinstance(state, dict) and "actor" in state else state)
        tqdm.write(f"[bc_init] {tariff}: warm-started actor from {bc_path}")

    ema_actor = EMA(actor, tau=tcfg.get("ema_tau", 0.99))
    backup_actor = ema_actor.model if ema_backup else actor

    reward_critic = load_critic(model_cfg["critic"], device=device)
    reward_target = load_critic(model_cfg["critic"], device=device)
    reward_target.load_state_dict(reward_critic.state_dict())
    cost_critics = {n: load_critic(model_cfg["critic"], device=device) for n in costs_cfg}
    cost_targets = {n: load_critic(model_cfg["critic"], device=device) for n in costs_cfg}
    for n in costs_cfg:
        cost_targets[n].load_state_dict(cost_critics[n].state_dict())

    temp = Temperature(init_alpha=hp.init_alpha).to(device)
    safety = SafetyLayer(actor.bess, actor.ev, env_params).to(device).eval()
    lambdas = {
        n: CostLambda(init_lambda=float(c.get("lambda_init", 0.1)),
                      budget=float(c.get("budget", 0.0)),
                      max_lambda=c.get("lambda_max")).to(device)
        for n, c in costs_cfg.items()
    }

    opt_actor = torch.optim.Adam(actor.parameters(), lr=hp.actor_lr)
    opt_reward = torch.optim.Adam(reward_critic.parameters(), lr=hp.critic_lr)
    opt_cost = {n: torch.optim.Adam(cost_critics[n].parameters(), lr=hp.critic_lr) for n in costs_cfg}
    opt_alpha = torch.optim.Adam([temp.log_alpha], lr=hp.alpha_lr)
    opt_lambda = {n: torch.optim.Adam([lambdas[n].log_lambda], lr=float(costs_cfg[n].get("lambda_lr", 3e-4))) for n in costs_cfg}

    buffer = ReplayBuffer(hp.buffer_size, obs_dim=model_cfg["actor"]["input_dim"], act_dim=3,
                          device=device, history_len=hp.history_len, n_step=hp.n_step,
                          gamma=hp.gamma, cost_names=list(costs_cfg))

    prior_buffer = None
    bs_prior = 0
    bs_online = hp.batch_size
    if rlpd_on:
        def _prior_cost_fn(o, a):
            if not costs_cfg:
                return None
            ot = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            at = torch.as_tensor(a, dtype=torch.float32, device=device).view(1, -1)
            with torch.inference_mode():
                return {n: float(cost_tensor(n, ot, at, safety, env_params, c).detach().cpu()) for n, c in costs_cfg.items()}
        prior_buffer = load_prior_buffer(
            demos_root=str(PROJECT_ROOT / rlpd_cfg.get("demos_root", "paper/demos")),
            sources=list(rlpd_cfg.get("sources", ["RB", "RBS"])), tariff=tariff,
            obs_dim=model_cfg["actor"]["input_dim"], act_dim=3, device=device,
            history_len=hp.history_len, n_step=hp.n_step, gamma=hp.gamma,
            reward_scale=hp.reward_scale, cost_names=list(costs_cfg),
            cost_compute_fn=_prior_cost_fn, capacity=rlpd_cfg.get("prior_capacity"))
        prior_ratio = float(rlpd_cfg.get("prior_ratio", 0.5))
        bs_prior = int(round(hp.batch_size * prior_ratio))
        bs_online = hp.batch_size - bs_prior
        tqdm.write(f"[rlpd] {tariff}: prior={prior_buffer.size} split online {bs_online}/prior {bs_prior}")

    episodes = EpisodeGen(cfg, str(PROJECT_ROOT / "data"), seed=seed)
    out_dir = PROJECT_ROOT / "paper" / "train" / exp_name / arch_name / f"{tariff}{run_suffix}"
    eval_runner = EvalRunner(actor_module=f"models.{arch_name}.model", actor_cfg=model_cfg["actor"],
                             parameters=env_params, tariff=tariff, history_len=hp.history_len,
                             project_root=str(PROJECT_ROOT), n_workers=tcfg.get("eval_workers", 4))
    async_eval = AsyncEval(eval_runner, cfg["val"], max_pending=tcfg.get("max_pending_evals", 2))
    audit = Audit(out_dir / "audit_training.csv", every=5)

    shaping_omega = float(tcfg.get("shaping_omega", 0.0))
    shaping_bess_omega = float(tcfg.get("shaping_bess_omega", 0.0))
    safe_rollout = bool(tcfg.get("cmdp", {}).get("safe_rollout", False))
    rollout_safety = safety if safe_rollout else None
    checkpoint_scores = {"best": -float("inf"), "best_robust": -float("inf"), "best_worst": -float("inf")}
    last_eval_summary = None
    step_count = [0]
    tqdm.write(f"[setup] {exp_name}/{arch_name} {tariff}: safety={safety_mode} costs={list(costs_cfg)} "
               f"ema_backup={ema_backup} rlpd={rlpd_on} shaping_bess={shaping_bess_omega} "
               f"UTD=1/{hp.update_every_steps} bc_init={'yes' if bc_path else 'no'} precision={precision} "
               f"droq_dropout={droq_dropout} n_critics={n_critics} m_target={m_target}")

    def immediate_costs(o, a):
        if not costs_cfg:
            return {}
        ot = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
        at = torch.as_tensor(a, dtype=torch.float32, device=device).view(1, -1)
        with torch.inference_mode():
            return {n: float(cost_tensor(n, ot, at, safety, env_params, c).detach().cpu()) for n, c in costs_cfg.items()}

    def sample_batch():
        if prior_buffer is None:
            return buffer.sample(hp.batch_size)
        b_on = buffer.sample(bs_online) if (bs_online > 0 and buffer.size > 0) else None
        b_pr = prior_buffer.sample(bs_prior) if (bs_prior > 0 and prior_buffer.size > 0) else None
        if b_on is None:
            return b_pr
        if b_pr is None:
            return b_on
        return merge_batches(b_on, b_pr)

    def snapshot():
        return {
            "actor": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
            "ema": {k: v.detach().cpu().clone() for k, v in ema_actor.state_dict().items()},
            "reward_critic": {k: v.detach().cpu().clone() for k, v in reward_critic.state_dict().items()},
            "cost_critics": {n: {k: v.detach().cpu().clone() for k, v in c.state_dict().items()} for n, c in cost_critics.items()},
            "alpha": {k: v.detach().cpu().clone() for k, v in temp.state_dict().items()},
            "lambdas": {n: {k: v.detach().cpu().clone() for k, v in l.state_dict().items()} for n, l in lambdas.items()},
        }

    def save_checkpoint(label, job, summary, metric, value):
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(job["snapshot"], out_dir / f"{label}.pt")
        meta = {"tariff": tariff, "episode": int(job["episode"]), "metric": metric,
                "value": None if value is None else float(value),
                "train_reward": float(job["context"].get("train_reward", float("nan"))),
                "summary": summary or {}}
        with open(out_dir / f"{label}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def finish_evals(jobs):
        nonlocal last_eval_summary
        for job in jobs:
            es = job["summary"]
            last_eval_summary = es
            improvements = []
            for label, metric in (("best", "mean_reward"), ("best_robust", "robust_reward"), ("best_worst", "worst_reward")):
                value = float(es.get(metric, -float("inf")))
                if value > checkpoint_scores[label]:
                    checkpoint_scores[label] = value
                    save_checkpoint(label, job, es, metric, value)
                    improvements.append(label)
            save_checkpoint("last", job, es, "last_eval", es.get("mean_reward"))
            tqdm.write(f"[eval] {tariff} ep {job['episode']:4d} -> mean={es['mean_reward']:.2f} "
                       f"best={checkpoint_scores['best']:.2f}" + (f"  * {','.join(improvements)}" if improvements else ""))
            audit.add({"episode": job["episode"], "train_reward": job["context"]["train_reward"],
                       "best_eval_reward": checkpoint_scores["best"], **job["context"]["agg"],
                       **{f"eval_{k}": v for k, v in es.items()}})
            audit.flush()

    for ep, pbar in episode_bars(hp.train_episodes, hp.warmup_episodes, tariff):
        finish_evals(async_eval.collect())
        envs = {
            stream: AnticipatoryShapingWrapper(
                SmartHomeEnv(episodes.load(stream), env_params, start=episodes.sample(stream),
                             days=hp.days, BESS_SoC=np.random.uniform(0.1, 0.9), tariff=tariff,
                             track_operation=False),
                params=env_params, gamma=hp.gamma, omega=shaping_omega,
                typical_drop=tcfg.get("shaping_typical_drop", 0.137), bess_omega=shaping_bess_omega)
            for stream in ("cy", "wy")
        }
        audit.open()
        total_steps = sum(env.sim.end_idx - env.sim.t_idx for env in envs.values())
        update_bar = UpdateBar(total_steps // hp.update_every_steps, ep) if ep >= hp.warmup_episodes else None

        def on_step(transition, key):
            o, a, r, no, d = transition
            buffer.add(o, a, r * hp.reward_scale, no, d, stream_id=key, costs=immediate_costs(o, a))
            if ep < hp.warmup_episodes or buffer.size < hp.batch_size:
                return
            step_count[0] += 1
            if step_count[0] % hp.update_every_steps != 0:
                return
            batch = sample_batch()
            if batch is None:
                return
            with amp_ctx():
                row = reward_critic_step(reward_critic, reward_target, backup_actor, batch, temp, opt_reward, grad_clip=hp.grad_clip, m_target=m_target)
                for n in costs_cfg:
                    row.update(cost_critic_step(n, cost_critics[n], cost_targets[n], backup_actor, batch, opt_cost[n], grad_clip=hp.grad_clip, m_target=m_target))
                if ep >= hp.actor_update_start_episode and step_count[0] % hp.actor_update_every == 0:
                    ainfo = actor_step(actor, reward_critic, cost_critics, lambdas, batch, temp, opt_actor, grad_clip=hp.grad_clip, m_actor=m_target)
                    q_costs = ainfo.pop("q_costs")
                    row.update(ainfo)
                    row.update(alpha_step(temp, ainfo.pop("logp"), hp.target_entropy, opt_alpha))
                    for n, qc in q_costs.items():
                        li = lambda_step(lambdas[n], qc, opt_lambda[n])
                        row[f"{n}_lambda_value"] = li["lambda_value"]
                    ema_actor.update(actor)
            soft_update(reward_critic, reward_target, hp.tau)
            for n in costs_cfg:
                soft_update(cost_critics[n], cost_targets[n], hp.tau)
            audit.step(**row)
            if update_bar is not None:
                update_bar.step(row)

        try:
            with EpisodeBar(envs, ep, hp.train_episodes) as episode_bar:
                _, rewards = collect_streams(envs, actor, hp.history_len, device, deterministic=False,
                                             progress=episode_bar, on_step=on_step, safety=rollout_safety)
        finally:
            if update_bar is not None:
                update_bar._flush(); update_bar.pbar.close()
        train_reward = float(sum(rewards.values()))

        eval_submitted = False
        if (ep + 1) % hp.evaluate_every == 0:
            while len(async_eval.pending) >= async_eval.max_pending:
                finish_evals(async_eval.collect(wait=True, max_items=1))
            state = snapshot()
            async_eval.submit(ep, state["actor"], state, {"train_reward": train_reward, "agg": audit.aggregate()})
            eval_submitted = True

        agg = audit.aggregate()
        if not eval_submitted:
            audit.add({"episode": ep, "train_reward": train_reward, "best_eval_reward": checkpoint_scores["best"], **agg})
            audit.flush()
        finish_evals(async_eval.collect())
        update_train_postfix(pbar, train_reward, last_eval_summary, checkpoint_scores["best"], agg, len(async_eval.pending))

    finish_evals(async_eval.drain())
    audit.flush(force=True)
    eval_runner.close()
    (out_dir / ".train_done").touch()


def run_train(algo_root: Path):
    algo_root = Path(algo_root).resolve()
    arch_name = algo_root.name
    exp_name = algo_root.parent.name
    run_suffix = os.environ.get("RUN_SUFFIX", "")
    tariffs = [t.strip() for t in os.environ.get("RUN_TARIFFS", ",".join(_DEFAULT_TARIFFS)).split(",") if t.strip()]
    for tariff in tariffs:
        train_tariff(algo_root, arch_name, exp_name, tariff, run_suffix=run_suffix)
