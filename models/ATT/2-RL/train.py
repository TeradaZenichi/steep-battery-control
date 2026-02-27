from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import deque
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import json
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../steep-battery-control
MODEL_ROOT   = Path(__file__).resolve().parents[2]  # .../models
ATT_ROOT     = Path(__file__).resolve().parents[1]  # .../models/ATT
ALGO_ROOT    = Path(__file__).resolve().parent      # .../models/ATT/2-RL
sys.path.insert(0, str(ALGO_ROOT))
sys.path.insert(1, str(ATT_ROOT))
sys.path.insert(2, str(MODEL_ROOT))
sys.path.insert(3, str(PROJECT_ROOT))

from utils import ReplayBuffer, EpisodeGen, Hyperparameters, Temperature, _eval_worker
from model import load_actor, load_critic
from environment import SmartHomeEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train:
    def __init__(self, tariff: str):
        self.tariff = tariff

        with open(ATT_ROOT / "model.json", encoding="utf-8") as f:
            self.model_cfg = json.load(f)

        with open(Path(__file__).resolve().parent / "config.json", encoding="utf-8") as f:
            self.train_cfg = json.load(f)

        with open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8") as f:
            self.parameters = json.load(f)

        self.episodegen = EpisodeGen(self.train_cfg, PROJECT_ROOT / "data")
        self.hp = Hyperparameters(self.train_cfg["train"])
        self.history_len = self._resolve_history_len(tariff)
        self.log_every_steps = int(self.train_cfg["train"].get("log_every_steps", 50))
        self.audit_every_episodes = int(self.train_cfg["train"].get("audit_every_episodes", 5))
        self.update_every_steps = int(self.train_cfg["train"].get("update_every_steps", 1))
        self.eval_workers = int(self.train_cfg["train"].get("eval_workers", 1))
        self.train_env_workers = int(self.train_cfg["train"].get("train_env_workers", 1))
        self.eval_ma_window = int(self.train_cfg["train"].get("eval_ma_window", 10))
        self.early_stop_patience = int(self.train_cfg["train"].get("early_stop_patience", 120))
        self.min_episodes_before_early_stop = int(self.train_cfg["train"].get("min_episodes_before_early_stop", 150))
        self.checkpoint_min_delta = float(self.train_cfg["train"].get("checkpoint_min_delta", 0.0))
        self.checkpoint_use_combined_score = bool(self.train_cfg["train"].get("checkpoint_use_combined_score", True))
        self.checkpoint_weight_det = float(self.train_cfg["train"].get("checkpoint_weight_det", 0.4))
        self.checkpoint_weight_stoch = float(self.train_cfg["train"].get("checkpoint_weight_stoch", 0.6))
        default_checkpoint_metric = "mean" if self.checkpoint_use_combined_score else "det"
        self.checkpoint_metric = str(self.train_cfg["train"].get("checkpoint_metric", default_checkpoint_metric)).lower()
        if self.checkpoint_metric not in {"det", "stoch", "mean"}:
            self.checkpoint_metric = default_checkpoint_metric
        w_sum = self.checkpoint_weight_det + self.checkpoint_weight_stoch
        if w_sum <= 0.0:
            self.checkpoint_weight_det = 0.5
            self.checkpoint_weight_stoch = 0.5
            w_sum = 1.0
        self.checkpoint_weight_det /= w_sum
        self.checkpoint_weight_stoch /= w_sum

        self.env_cy = SmartHomeEnv(
            self.episodegen.df_cy,
            self.parameters,
            start=self.episodegen.sample("cy"),
            days=self.hp.days,
            BESS_SoC=0.5,
            tariff=self.tariff,
            track_operation=False,
        )
        self.env_wy = SmartHomeEnv(
            self.episodegen.df_wy,
            self.parameters,
            start=self.episodegen.sample("wy"),
            days=self.hp.days,
            BESS_SoC=0.5,
            tariff=self.tariff,
            track_operation=False,
        )
        self.envs = {"cy": self.env_cy, "wy": self.env_wy}

        self.episode_length = int(24 * 60 // self.env_cy.sim.timestep * self.hp.days)

        torch.manual_seed(self.hp.seed)
        np.random.seed(self.hp.seed)
        random.seed(self.hp.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.hp.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.folder = PROJECT_ROOT / "Results" / "train" / "ATT" / "2-RL" / self.tariff
        self.folder.mkdir(parents=True, exist_ok=True)

        self.best_actor_path = self.folder / "best_actor_eval.pt"
        self.best_ckpt_path  = self.folder / "best_checkpoint_eval.pt"
        self.best_meta_path  = self.folder / "best_eval_meta.json"

        self.best_actor_ma_path = self.folder / "best_actor_eval_ma.pt"
        self.best_ckpt_ma_path  = self.folder / "best_checkpoint_eval_ma.pt"
        self.best_meta_ma_path  = self.folder / "best_eval_ma_meta.json"

        self.buffer = ReplayBuffer(
            capacity=self.hp.buffer_size,
            obs_dim=self.model_cfg["actor"]["input_dim"],
            act_dim=self.model_cfg["actor"]["output_dim"],
            device=DEVICE,
            history_len=self.history_len,
            n_step=self.hp.n_step,
            gamma=self.hp.γ,
        )

        self.actor_cfg = dict(self.model_cfg["actor"])
        self.actor_cfg["log_std_min"] = float(self.hp.log_std_min)
        self.actor_cfg["log_std_max"] = float(self.hp.log_std_max)
        self.actor_cfg["parameters"] = str((PROJECT_ROOT / "data" / "parameters.json").resolve())

        self.actor = load_actor(self.actor_cfg, device=DEVICE)
        self.critics = load_critic(self.model_cfg["critic"], device=DEVICE)
        self.critics_target = load_critic(self.model_cfg["critic"], device=DEVICE)
        self.critics_target.load_state_dict(self.critics.state_dict(), strict=True)

        self.temperature = Temperature(
            init_log_alpha=0.0,
            target_entropy=self.hp.target_entropy
        ).to(DEVICE)

        self.opt_alpha = torch.optim.Adam([self.temperature.log_alpha], lr=self.hp.α_lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.hp.actor_lr)
        self.opt_critic = torch.optim.Adam(self.critics.parameters(), lr=self.hp.critic_lr)

        self.lmbda = torch.zeros(1, device=DEVICE)
        self.lmbda_lr = float(self.train_cfg["train"].get("lambda_lr", 1e-3))
        self.cost_limit = float(self.train_cfg["train"].get("cost_limit", 1e-4))
        self.lmbda_max = float(self.train_cfg["train"].get("lambda_max", 100.0))
        self.lambda_deadzone = float(self.train_cfg["train"].get("lambda_deadzone", 0.0))
        self.dual_enabled = bool(self.train_cfg["train"].get("dual_enabled", True))

        self.best_eval_reward  = -float("inf")
        self.best_eval_episode = -1
        self.best_eval_ma = -float("inf")
        self.best_eval_ma_episode = -1
        self.best_checkpoint_score = -float("inf")
        self.best_checkpoint_episode = -1
        self.best_train_reward = -float("inf")
        self.last_improvement_episode = -1
        self.last_improvement_eval_count = -1
        self.eval_count = 0

        self.eval_rewards = []
        self.eval_ma_rewards = []
        self.eval_window = deque(maxlen=max(1, self.eval_ma_window))
        self.train_rewards = []
        self.q1_values = []
        self.q2_values = []
        self.total_episode_rewards = []

        self.backup_means = []
        self.backup_abs_maxs = []
        self.q_next_means = []
        self.logp_means = []
        self.logp_mins = []

        self.cost_means = []
        self.cost_p95s = []
        self.frac_violations = []
        self.actor_losses = []
        self.actor_term_entropies = []
        self.actor_term_qs = []
        self.actor_term_duals = []
        self.critic_losses = []
        self.alpha_losses = []
        self.violation_eps = float(self.train_cfg["train"].get("violation_eps", 1e-6))

        self.audit_df = pd.DataFrame(columns=[
            "episode",
            "train_reward_total",
            "eval_reward_det",
            "eval_reward",
            "eval_reward_ma",
            "eval_reward_stoch",
            "checkpoint_score",
            "checkpoint_metric",
            "best_train_reward",
            "best_eval_reward",
            "best_eval_ma",
            "best_checkpoint_score",
            "q1_mean",
            "q2_mean",
            "backup_mean",
            "backup_abs_max",
            "q_next_mean",
            "logp_mean",
            "logp_min",
            "n_updates",
            "steps",
            "buffer_size",
            "alpha",
            "lambda",
            "dual_enabled",
            "cost_mean",
            "cost_p95",
            "frac_violation",
            "critic_loss",
            "actor_loss",
            "actor_term_entropy",
            "actor_term_q",
            "actor_term_dual",
            "alpha_loss",
            "no_improve_episodes",
            "no_improve_evals",
        ])
        self.audit_csv = self.folder / "audit_training.csv"


    def save_checkpoint(self, filepath: Path) -> None:
        ckpt = {
            "tariff": self.tariff,
            "actor_cfg": self.actor_cfg,
            "train_cfg": self.train_cfg,
            "parameters_hash_hint": "data/parameters.json",
            "actor_state_dict": self.actor.state_dict(),
            "critics_state_dict": self.critics.state_dict(),
            "critics_target_state_dict": self.critics_target.state_dict(),
            "temperature_state_dict": self.temperature.state_dict(),
            "opt_actor_state_dict": self.opt_actor.state_dict(),
            "opt_critic_state_dict": self.opt_critic.state_dict(),
            "opt_alpha_state_dict": self.opt_alpha.state_dict(),
            "lambda_value": float(self.lmbda.detach().cpu().item()),
        }
        torch.save(ckpt, filepath)


    def _save_best_eval(self, eval_reward_value: float, episode: int, checkpoint_score: float, eval_reward_stoch: float) -> None:
        torch.save(self.actor.state_dict(), self.best_actor_path)
        self.save_checkpoint(self.best_ckpt_path)
        meta = {
            "best_eval_reward": float(eval_reward_value),
            "best_eval_episode": int(episode),
            "best_eval_reward_stoch": float(eval_reward_stoch),
            "best_checkpoint_score": float(checkpoint_score),
            "best_checkpoint_episode": int(episode),
            "checkpoint_metric": str(self.checkpoint_metric),
            "checkpoint_use_combined_score": bool(self.checkpoint_use_combined_score),
            "checkpoint_weight_det": float(self.checkpoint_weight_det),
            "checkpoint_weight_stoch": float(self.checkpoint_weight_stoch),
            "tariff": self.tariff
        }
        with open(self.best_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


    def _save_best_eval_ma(self, eval_reward_ma: float, episode: int) -> None:
        torch.save(self.actor.state_dict(), self.best_actor_ma_path)
        self.save_checkpoint(self.best_ckpt_ma_path)
        meta = {
            "best_eval_ma": float(eval_reward_ma),
            "best_eval_ma_episode": int(episode),
            "eval_ma_window": int(self.eval_ma_window),
            "tariff": self.tariff
        }
        with open(self.best_meta_ma_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


    def eval(self, deterministic: bool = True) -> float:
        eval_desc = "Eval Det" if deterministic else "Eval Stoch"
        runs = self.train_cfg["val"]
        eval_workers = max(1, min(self.eval_workers, len(runs)))
        actor_state_cpu = {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}
        rewards = []
        with tqdm(total=len(runs), desc=eval_desc, position=2, dynamic_ncols=True, leave=False) as p_eval:
            with ProcessPoolExecutor(max_workers=eval_workers) as executor:
                futures = [
                    executor.submit(
                        _eval_worker,
                        run,
                        self.parameters,
                        self.tariff,
                        self.actor_cfg,
                        actor_state_cpu,
                        self.episode_length,
                        self.history_len,
                        deterministic,
                    )
                    for run in runs
                ]

                for idx, fut in enumerate(as_completed(futures), start=1):
                    total_reward = float(fut.result())
                    rewards.append(total_reward)
                    running_mean = float(np.mean(rewards))
                    p_eval.set_postfix({
                        "scenario": f"{idx}/{len(runs)}",
                        "reward": f"{total_reward:.2f}",
                        "avg": f"{running_mean:.2f}",
                        "w": int(eval_workers),
                    })
                    p_eval.update(1)

        return float(np.mean(rewards))


    def update(self):
        batch = self.buffer.sample(self.hp.batch_size)

        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        gamma_pow = batch["gamma_pow"]

        obs_critic = obs[:, -1, :] if obs.dim() == 3 else obs
        next_obs_critic = next_obs[:, -1, :] if next_obs.dim() == 3 else next_obs

        with torch.no_grad():
            next_action, logp_next, _, cost_next = self.actor.sample(next_obs)

            q1_next, q2_next = self.critics_target(next_obs_critic, next_action)
            q_next = torch.min(q1_next, q2_next)

            alpha = self.temperature.alpha

            backup = rew + gamma_pow * (1.0 - done) * (q_next - alpha * logp_next)

        q1, q2 = self.critics(obs_critic, act)
        critic_loss = torch.mean((q1 - backup) ** 2) + torch.mean((q2 - backup) ** 2)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        if self.hp.grad_clip:
            nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=1.0)
        self.opt_critic.step()

        action_pi, logp_pi, _, cost = self.actor.sample(obs)
        q1_pi, q2_pi = self.critics(obs_critic, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        alpha = self.temperature.alpha.detach()

        actor_term_entropy = alpha * logp_pi
        actor_term_q = -q_pi
        actor_term_dual = self.lmbda * cost if self.dual_enabled else torch.zeros_like(cost)
        actor_loss = torch.mean(actor_term_entropy + actor_term_q + actor_term_dual)

        self.opt_actor.zero_grad()
        actor_loss.backward()
        if self.hp.grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.opt_actor.step()

        if self.hp.auto_entropy:
            alpha_loss = torch.mean(-self.temperature.log_alpha * (logp_pi + self.hp.target_entropy).detach())
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()
        else:
            alpha_loss = torch.tensor(0.0, device=DEVICE)

        for param, target_param in zip(self.critics.parameters(), self.critics_target.parameters()):
            target_param.data.copy_(self.hp.τ * param.data + (1 - self.hp.τ) * target_param.data)

        with torch.no_grad():
            if self.dual_enabled and self.lmbda_lr > 0.0:
                target = cost.new_tensor(self.cost_limit)
                grad = torch.mean(cost) - target
                if abs(float(grad.detach().cpu())) > self.lambda_deadzone:
                    self.lmbda += self.lmbda_lr * grad
                self.lmbda = torch.clamp(self.lmbda, min=0.0, max=self.lmbda_max)
            else:
                self.lmbda.zero_()

        self.q1_values.append(float(q1.mean().detach().cpu()))
        self.q2_values.append(float(q2.mean().detach().cpu()))
        self.backup_means.append(float(backup.mean().detach().cpu()))
        self.backup_abs_maxs.append(float(torch.max(torch.abs(backup)).detach().cpu()))
        self.q_next_means.append(float(q_next.mean().detach().cpu()))
        self.logp_means.append(float(logp_pi.mean().detach().cpu()))
        self.logp_mins.append(float(logp_pi.min().detach().cpu()))

        cost_cpu = cost.detach().view(-1).cpu().numpy()
        self.cost_means.append(float(np.mean(cost_cpu)))
        self.cost_p95s.append(float(np.percentile(cost_cpu, 95)))
        frac_violation = float((cost_cpu > self.violation_eps).mean())
        self.frac_violations.append(frac_violation)
        self.critic_losses.append(float(critic_loss.detach().cpu()))
        self.actor_losses.append(float(actor_loss.detach().cpu()))
        self.actor_term_entropies.append(float(torch.mean(actor_term_entropy).detach().cpu()))
        self.actor_term_qs.append(float(torch.mean(actor_term_q).detach().cpu()))
        self.actor_term_duals.append(float(torch.mean(actor_term_dual).detach().cpu()))
        self.alpha_losses.append(float(alpha_loss.detach().cpu()))


    def _reset_train_envs(self):
        for key, env in self.envs.items():
            env.reset(options={"start": self.episodegen.sample(key), "bess_soc": np.random.uniform(0.1, 0.9)})


    @staticmethod
    def _obs_vector(obs) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).reshape(-1)


    def _init_histories(self) -> dict:
        histories = {}
        for key, env in self.envs.items():
            obs0 = self._obs_vector(env.state)
            histories[key] = deque([obs0.copy() for _ in range(self.history_len)], maxlen=self.history_len)
        return histories


    def _run_warmup(self):
        print("Starting warmup episodes...")
        for episode in tqdm(range(self.hp.warmup_episodes), desc="Warmup Episodes", position=0, dynamic_ncols=True):
            self._reset_train_envs()
            histories = self._init_histories()
            env_dones = {"cy": False, "wy": False}
            steps = 0

            with tqdm(total=self.episode_length, desc="Warmup Steps", position=1, dynamic_ncols=True, leave=False) as pbar:
                while not all(env_dones.values()) and steps < self.episode_length:
                    for key, env in self.envs.items():
                        if env_dones[key]:
                            continue

                        obs_seq = np.stack(histories[key], axis=0)
                        action = env.action_space.sample()
                        next_obs, rew, done, truncated, info = env.step(action)
                        next_obs_vec = self._obs_vector(next_obs)
                        histories[key].append(next_obs_vec.copy())
                        next_obs_seq = np.stack(histories[key], axis=0)

                        self.buffer.add(obs_seq, action, rew * self.hp.reward_scale, next_obs_seq, done or truncated, stream_id=key)

                        env_dones[key] = done or truncated
                        steps += 1

                        pbar.update(1)

    def _collect_training_episode(self, episode: int) -> tuple[float, int]:
        self._reset_train_envs()
        histories = self._init_histories()
        env_dones = {"cy": False, "wy": False}
        reward = {"cy": 0.0, "wy": 0.0, "total": 0.0}

        steps = 0
        updates_in_episode = 0
        episode_total_steps = self.episode_length * len(self.envs)
        env_workers = max(1, min(self.train_env_workers, len(self.envs)))

        with ThreadPoolExecutor(max_workers=env_workers) as step_executor:
            with tqdm(total=episode_total_steps, desc=f"Ep {episode + 1}/{self.hp.train_episodes} Steps", position=1, dynamic_ncols=True, leave=False) as pbar:
                while not all(env_dones.values()) and steps < episode_total_steps:
                    active = [(key, env) for key, env in self.envs.items() if not env_dones[key]]

                    obs_map = {key: np.stack(histories[key], axis=0) for key, env in active}
                    obs_batch = np.stack([obs_map[key] for key, _ in active], axis=0)
                    obs_t = torch.as_tensor(obs_batch, device=DEVICE)

                    with torch.no_grad():
                        action_t, _, _, _ = self.actor.sample(obs_t)

                    action_batch = action_t.detach().cpu().numpy()
                    job_payload = []
                    for i, (key, env) in enumerate(active):
                        action_exec = np.clip(action_batch[i], env.action_space.low, env.action_space.high)
                        job_payload.append((key, env, obs_map[key], action_exec))

                    futures = {
                        step_executor.submit(env.step, action_exec): (key, obs_np, action_exec)
                        for key, env, obs_np, action_exec in job_payload
                    }
                    completed = [
                        (futures[fut], fut.result())
                        for fut in as_completed(futures)
                    ]

                    for (key, obs_seq, action_exec), (next_obs, rew, done, truncated, info) in completed:
                        next_obs_vec = self._obs_vector(next_obs)
                        histories[key].append(next_obs_vec.copy())
                        next_obs_seq = np.stack(histories[key], axis=0)
                        self.buffer.add(obs_seq, action_exec, rew * self.hp.reward_scale, next_obs_seq, done or truncated, stream_id=key)

                        reward[key] += rew
                        reward["total"] += rew
                        env_dones[key] = done or truncated
                        steps += 1

                        progress_pct = 100.0 * steps / episode_total_steps
                        if (steps % self.log_every_steps == 0) or (steps == episode_total_steps):
                            pbar.set_postfix({
                                "step": f"{steps}/{episode_total_steps}",
                                "%": f"{progress_pct:.1f}",
                                "rcy": f"{reward['cy']:.2f}",
                                "rwy": f"{reward['wy']:.2f}",
                                "rtotal": f"{reward['total']:.2f}",
                                "upd": int(updates_in_episode),
                                "buf": int(self.buffer.size),
                                "w": int(env_workers),
                            })
                        pbar.update(1)

                        if self.buffer.size >= self.hp.batch_size and (steps % self.update_every_steps == 0):
                            for _ in range(self.hp.update_steps):
                                self.update()
                                updates_in_episode += 1

        return float(reward["total"]), int(steps)


    def _run_eval_and_checkpoint(self, episode: int) -> tuple[float, float, float, float, int, int]:
        eval_reward_det = np.nan
        eval_reward_ma = np.nan
        eval_reward_stoch = np.nan
        checkpoint_score = np.nan
        improved = False

        if episode % self.hp.eval_every == 0:
            eval_reward_det = float(self.eval(deterministic=True))
            eval_reward_stoch = float(self.eval(deterministic=False))
            self.eval_rewards.append(eval_reward_det)

            if self.checkpoint_metric == "det":
                checkpoint_score = float(eval_reward_det)
            elif self.checkpoint_metric == "stoch":
                checkpoint_score = float(eval_reward_stoch)
            elif self.checkpoint_metric == "mean":
                checkpoint_score = float(
                    self.checkpoint_weight_det * eval_reward_det
                    + self.checkpoint_weight_stoch * eval_reward_stoch
                )
            else:
                checkpoint_score = float(
                    self.checkpoint_weight_det * eval_reward_det
                    + self.checkpoint_weight_stoch * eval_reward_stoch
                )

            self.eval_count += 1

            if eval_reward_det > self.best_eval_reward:
                self.best_eval_reward = eval_reward_det
                self.best_eval_episode = int(episode)

            if checkpoint_score > self.best_checkpoint_score + self.checkpoint_min_delta:
                self.best_checkpoint_score = checkpoint_score
                self.best_checkpoint_episode = int(episode)
                self._save_best_eval(eval_reward_det, episode, checkpoint_score, eval_reward_stoch)
                improved = True

            self.eval_window.append(checkpoint_score)
            eval_reward_ma = float(np.mean(self.eval_window))
            self.eval_ma_rewards.append(eval_reward_ma)

            if eval_reward_ma > self.best_eval_ma + self.checkpoint_min_delta:
                self.best_eval_ma = eval_reward_ma
                self.best_eval_ma_episode = int(episode)
                self._save_best_eval_ma(eval_reward_ma, episode)
                improved = True

        if improved:
            self.last_improvement_episode = int(episode)
            self.last_improvement_eval_count = int(self.eval_count)

        no_improve_evals = 0 if self.last_improvement_eval_count < 0 else int(self.eval_count - self.last_improvement_eval_count)
        no_improve_episodes = 0 if self.last_improvement_episode < 0 else int(episode - self.last_improvement_episode)
        return eval_reward_det, eval_reward_ma, eval_reward_stoch, checkpoint_score, no_improve_episodes, no_improve_evals


    def _aggregate_episode_update_metrics(self, q_start: int) -> dict:
        q_end = len(self.q1_values)
        n_updates = int(q_end - q_start)

        if n_updates > 0:
            return {
                "n_updates": n_updates,
                "q1_mean": float(np.mean(self.q1_values[q_start:q_end])),
                "q2_mean": float(np.mean(self.q2_values[q_start:q_end])),
                "backup_mean": float(np.mean(self.backup_means[q_start:q_end])),
                "backup_abs_max": float(np.max(self.backup_abs_maxs[q_start:q_end])),
                "q_next_mean": float(np.mean(self.q_next_means[q_start:q_end])),
                "logp_mean": float(np.mean(self.logp_means[q_start:q_end])),
                "logp_min": float(np.min(self.logp_mins[q_start:q_end])),
                "cost_mean_ep": float(np.mean(self.cost_means[q_start:q_end])),
                "cost_p95_ep": float(np.mean(self.cost_p95s[q_start:q_end])),
                "frac_violation_ep": float(np.mean(self.frac_violations[q_start:q_end])),
                "critic_loss_ep": float(np.mean(self.critic_losses[q_start:q_end])),
                "actor_loss_ep": float(np.mean(self.actor_losses[q_start:q_end])),
                "actor_term_entropy_ep": float(np.mean(self.actor_term_entropies[q_start:q_end])),
                "actor_term_q_ep": float(np.mean(self.actor_term_qs[q_start:q_end])),
                "actor_term_dual_ep": float(np.mean(self.actor_term_duals[q_start:q_end])),
                "alpha_loss_ep": float(np.mean(self.alpha_losses[q_start:q_end])),
            }

        return {
            "n_updates": n_updates,
            "q1_mean": np.nan,
            "q2_mean": np.nan,
            "backup_mean": np.nan,
            "backup_abs_max": np.nan,
            "q_next_mean": np.nan,
            "logp_mean": np.nan,
            "logp_min": np.nan,
            "cost_mean_ep": np.nan,
            "cost_p95_ep": np.nan,
            "frac_violation_ep": np.nan,
            "critic_loss_ep": np.nan,
            "actor_loss_ep": np.nan,
            "actor_term_entropy_ep": np.nan,
            "actor_term_q_ep": np.nan,
            "actor_term_dual_ep": np.nan,
            "alpha_loss_ep": np.nan,
        }


    def _resolve_history_len(self, tariff: str) -> int:
        """Load history_len from IL HPO results for this tariff."""
        il_path = PROJECT_ROOT / "Results" / "train" / "ATT" / "1-IL" / tariff / "best_params.json"
        with open(il_path, encoding="utf-8") as f:
            hl = int(json.load(f)["history_len"])
        print(f"[history_len] {tariff} = {hl} (from IL HPO)")
        return hl

    def _build_audit_row(self, episode: int, train_total: float, eval_reward_det: float, eval_reward_ma: float, eval_reward_stoch: float, checkpoint_score: float, metrics: dict, steps: int, no_improve_episodes: int, no_improve_evals: int = 0) -> dict:
        alpha_val = float(self.temperature.alpha.detach().cpu())

        return {
            "episode": int(episode),
            "train_reward_total": train_total,
            "eval_reward_det": float(eval_reward_det) if not np.isnan(eval_reward_det) else np.nan,
            "eval_reward": float(eval_reward_det) if not np.isnan(eval_reward_det) else np.nan,
            "eval_reward_ma": float(eval_reward_ma) if not np.isnan(eval_reward_ma) else np.nan,
            "eval_reward_stoch": float(eval_reward_stoch) if not np.isnan(eval_reward_stoch) else np.nan,
            "checkpoint_score": float(checkpoint_score) if not np.isnan(checkpoint_score) else np.nan,
            "checkpoint_metric": str(self.checkpoint_metric),
            "best_train_reward": float(self.best_train_reward),
            "best_eval_reward": float(self.best_eval_reward),
            "best_eval_ma": float(self.best_eval_ma),
            "best_checkpoint_score": float(self.best_checkpoint_score),
            "q1_mean": float(metrics["q1_mean"]) if not np.isnan(metrics["q1_mean"]) else np.nan,
            "q2_mean": float(metrics["q2_mean"]) if not np.isnan(metrics["q2_mean"]) else np.nan,
            "backup_mean": float(metrics["backup_mean"]) if not np.isnan(metrics["backup_mean"]) else np.nan,
            "backup_abs_max": float(metrics["backup_abs_max"]) if not np.isnan(metrics["backup_abs_max"]) else np.nan,
            "q_next_mean": float(metrics["q_next_mean"]) if not np.isnan(metrics["q_next_mean"]) else np.nan,
            "logp_mean": float(metrics["logp_mean"]) if not np.isnan(metrics["logp_mean"]) else np.nan,
            "logp_min": float(metrics["logp_min"]) if not np.isnan(metrics["logp_min"]) else np.nan,
            "n_updates": int(metrics["n_updates"]),
            "steps": int(steps),
            "buffer_size": int(self.buffer.size),
            "alpha": alpha_val,
            "lambda": float(self.lmbda.detach().cpu().item()),
            "dual_enabled": int(self.dual_enabled),
            "cost_mean": float(metrics["cost_mean_ep"]) if not np.isnan(metrics["cost_mean_ep"]) else np.nan,
            "cost_p95": float(metrics["cost_p95_ep"]) if not np.isnan(metrics["cost_p95_ep"]) else np.nan,
            "frac_violation": float(metrics["frac_violation_ep"]) if not np.isnan(metrics["frac_violation_ep"]) else np.nan,
            "critic_loss": float(metrics["critic_loss_ep"]) if not np.isnan(metrics["critic_loss_ep"]) else np.nan,
            "actor_loss": float(metrics["actor_loss_ep"]) if not np.isnan(metrics["actor_loss_ep"]) else np.nan,
            "actor_term_entropy": float(metrics["actor_term_entropy_ep"]) if not np.isnan(metrics["actor_term_entropy_ep"]) else np.nan,
            "actor_term_q": float(metrics["actor_term_q_ep"]) if not np.isnan(metrics["actor_term_q_ep"]) else np.nan,
            "actor_term_dual": float(metrics["actor_term_dual_ep"]) if not np.isnan(metrics["actor_term_dual_ep"]) else np.nan,
            "alpha_loss": float(metrics["alpha_loss_ep"]) if not np.isnan(metrics["alpha_loss_ep"]) else np.nan,
            "no_improve_episodes": int(no_improve_episodes),
            "no_improve_evals": int(no_improve_evals),
        }


    def _update_train_postfix(self, p_outer, train_total: float, eval_reward_det: float, eval_reward_ma: float, eval_reward_stoch: float, checkpoint_score: float, metrics: dict, no_improve_episodes: int, no_improve_evals: int = 0):
        p_outer.set_postfix({
            "train_total": f"{train_total:.2f}",
            "eval": f"{eval_reward_det:.2f}" if not np.isnan(eval_reward_det) else "-",
            "eval_ma": f"{eval_reward_ma:.2f}" if not np.isnan(eval_reward_ma) else "-",
            "eval_stoch": f"{eval_reward_stoch:.2f}" if not np.isnan(eval_reward_stoch) else "-",
            "ckpt": f"{checkpoint_score:.2f}" if not np.isnan(checkpoint_score) else "-",
            "ckpt_m": str(self.checkpoint_metric),
            "best_eval": f"{self.best_eval_reward:.2f}",
            "best_eval_ma": f"{self.best_eval_ma:.2f}",
            "best_ckpt": f"{self.best_checkpoint_score:.2f}",
            "alpha": f"{float(self.temperature.alpha.detach().cpu()):.3f}",
            "lambda": f"{float(self.lmbda.detach().cpu().item()):.3f}",
            "dual": int(self.dual_enabled),
            "frac_viol": f"{float(metrics['frac_violation_ep']):.3f}" if not np.isnan(metrics["frac_violation_ep"]) else "nan",
            "no_imp_ep": int(no_improve_episodes),
            "no_imp_ev": int(no_improve_evals),
        })


    def _should_early_stop(self, episode: int, no_improve_evals: int) -> bool:
        return (
            self.early_stop_patience > 0
            and episode >= self.min_episodes_before_early_stop
            and self.last_improvement_eval_count >= 0
            and no_improve_evals >= self.early_stop_patience
        )


    def train(self):
        print(f"Training SAC with tariff: {self.tariff}")
        print(f"Using device: {DEVICE}")

        self._run_warmup()

        print("Starting training episodes...")
        for episode in (p_outer := tqdm(range(self.hp.train_episodes), desc="Training Episodes", position=0, dynamic_ncols=True)):
            q_start = len(self.q1_values)
            train_total, steps = self._collect_training_episode(episode)
            self.train_rewards.append(train_total)
            self.total_episode_rewards.append(train_total)

            if train_total > self.best_train_reward:
                self.best_train_reward = train_total

            eval_reward_det, eval_reward_ma, eval_reward_stoch, checkpoint_score, no_improve_episodes, no_improve_evals = self._run_eval_and_checkpoint(episode)
            metrics = self._aggregate_episode_update_metrics(q_start)
            row = self._build_audit_row(
                episode=episode,
                train_total=train_total,
                eval_reward_det=eval_reward_det,
                eval_reward_ma=eval_reward_ma,
                eval_reward_stoch=eval_reward_stoch,
                checkpoint_score=checkpoint_score,
                metrics=metrics,
                steps=steps,
                no_improve_episodes=no_improve_episodes,
                no_improve_evals=no_improve_evals,
            )

            self.audit_df.loc[len(self.audit_df)] = row
            if ((episode + 1) % self.audit_every_episodes == 0) or ((episode + 1) == self.hp.train_episodes):
                self.audit_df.to_csv(self.audit_csv, index=False)

            self._update_train_postfix(p_outer, train_total, eval_reward_det, eval_reward_ma, eval_reward_stoch, checkpoint_score, metrics, no_improve_episodes, no_improve_evals)

            if self._should_early_stop(episode, no_improve_evals):
                print(
                    f"Early stopping at episode {episode}: no improvement for {no_improve_evals} evals "
                    f"(patience={self.early_stop_patience}, min_start={self.min_episodes_before_early_stop})."
                )
                self.audit_df.to_csv(self.audit_csv, index=False)
                break


def main():
    for tariff in ["tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat"]:
        trainer = Train(tariff)
        trainer.train()


if __name__ == "__main__":
    main()
