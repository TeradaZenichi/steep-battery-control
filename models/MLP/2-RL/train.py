from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../steep-battery-control
MODEL_ROOT   = Path(__file__).resolve().parents[2]  # .../models
MLP_ROOT     = Path(__file__).resolve().parent.parent   # .../models/MLP
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MLP_ROOT))
sys.path.insert(0, str(MODEL_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from utils import ReplayBuffer, EpisodeGen, Hyperparameters, Temperature, _eval_worker
from model import load_actor, load_critic
from environment import SmartHomeEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train:
    def __init__(self, tariff: str):
        self.tariff = tariff

        with open(Path(__file__).resolve().parent.parent / "model.json") as f:
            self.model_cfg = json.load(f)

        with open(Path(__file__).resolve().parent / "config.json", encoding="utf-8") as f:
            self.train_cfg = json.load(f)

        with open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8") as f:
            self.parameters = json.load(f)

        self.episodegen = EpisodeGen(self.train_cfg, PROJECT_ROOT / "data")
        self.hp = Hyperparameters(self.train_cfg["train"])
        self.log_every_steps = int(self.train_cfg["train"].get("log_every_steps", 50))
        self.audit_every_episodes = int(self.train_cfg["train"].get("audit_every_episodes", 5))
        self.update_every_steps = max(1, int(self.train_cfg["train"].get("update_every_steps", 1)))

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

        # Defina o tamanho do episódio (número de steps por episódio)
        self.episode_length = int(24 * 60 // self.env_cy.sim.timestep * self.hp.days)

        torch.manual_seed(self.hp.seed)
        np.random.seed(self.hp.seed)

        self.folder = PROJECT_ROOT / "Results" / "train" / "MLP" / "2-RL" / self.tariff
        self.folder.mkdir(parents=True, exist_ok=True)

        # Arquivos únicos (sobrescrevem)
        self.best_actor_path = self.folder / "best_actor_eval.pt"
        self.best_ckpt_path  = self.folder / "best_checkpoint_eval.pt"
        self.best_meta_path  = self.folder / "best_eval_meta.json"

        self.buffer = ReplayBuffer(
            capacity=self.hp.buffer_size,
            obs_dim=self.model_cfg["actor"]["input_dim"],
            act_dim=self.model_cfg["actor"]["output_dim"],
            device=DEVICE,
        )

        # Merge actor architecture (model.json) with RL-specific stochastic settings (config.json).
        self.actor_cfg = dict(self.model_cfg["actor"])
        self.actor_cfg["log_std_min"] = float(self.hp.log_std_min)
        self.actor_cfg["log_std_max"] = float(self.hp.log_std_max)

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

        # --- CHANGE (Lagrangian): initialize dual variable (lambda) and its hyperparameters ---
        # This lambda penalizes infeasible raw actions via a cost produced by the projected sample().
        self.lmbda = torch.zeros(1, device=DEVICE)  # lambda >= 0
        self.lmbda_lr = float(self.train_cfg["train"].get("lambda_lr", 1e-3))
        self.cost_limit = float(self.train_cfg["train"].get("cost_limit", 1e-4))  # target cost (0 means no violation)
        self.lmbda_max = float(self.train_cfg["train"].get("lambda_max", 100.0))
        # --- END CHANGE ---

        # eval and reward tracking
        self.best_eval_reward  = -float("inf")
        self.best_eval_episode = -1
        self.best_train_reward = -float("inf")  # rastrear apenas (não salvar arquivos)

        # tracking lists
        self.eval_rewards = []
        self.train_rewards = []
        self.q1_values = []
        self.q2_values = []
        self.total_episode_rewards = []

        # additional metrics for audit (per update)
        self.backup_means = []
        self.backup_abs_maxs = []
        self.q_next_means = []
        self.logp_means = []
        self.logp_mins = []

        # --- CHANGE (Projection/Loss audit): extra per-update metrics ---
        # These are appended once per update, and sliced per-episode using q_start:q_end.
        self.cost_means = []
        self.cost_p95s = []
        self.frac_violations = []
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.violation_eps = float(self.train_cfg["train"].get("violation_eps", 1e-6))
        # --- END CHANGE ---

        # audit dataframe (updated at end of each episode)
        self.audit_df = pd.DataFrame(columns=[
            "episode",
            "train_reward_total",
            "eval_reward",
            "eval_reward_stoch",
            "best_train_reward",
            "best_eval_reward",
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
            # --- CHANGE (Lagrangian): log lambda value for audit ---
            "lambda",
            # --- END CHANGE ---
            # --- CHANGE (Projection/Loss audit): aggregated per-episode from per-update values ---
            "cost_mean",
            "cost_p95",
            "frac_violation",
            "critic_loss",
            "actor_loss",
            "alpha_loss",
            # --- END CHANGE ---
        ])
        self.audit_csv = self.folder / "audit_training.csv"


    def save_checkpoint(self, filepath: Path) -> None:
        """Save a full checkpoint for reproducibility and training resumption."""
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
            # --- CHANGE (Lagrangian): persist lambda for reproducibility ---
            "lambda_value": float(self.lmbda.detach().cpu().item()),
            # --- END CHANGE ---
        }
        torch.save(ckpt, filepath)


    def _save_best_eval(self, eval_reward_value: float, episode: int) -> None:
        """Save ONLY the best model/checkpoint according to eval, using fixed filenames."""
        # 1) Actor weights
        torch.save(self.actor.state_dict(), self.best_actor_path)

        # 2) Full checkpoint
        self.save_checkpoint(self.best_ckpt_path)

        # 3) Meta
        meta = {
            "best_eval_reward": float(eval_reward_value),
            "best_eval_episode": int(episode),
            "tariff": self.tariff
        }
        with open(self.best_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


    def create_eval_envs(self):
        self.eval_envs = []
        for val in self.train_cfg["val"]:
            df = self.episodegen.load(val["dataset"])
            start = pd.to_datetime(val["date"])
            env = SmartHomeEnv(
                df,
                self.parameters,
                start=start,
                days=val["days"],
                BESS_SoC=val["soc"],
                tariff=self.tariff,
                track_operation=False,
            )
            self.eval_envs.append(env)


    def eval(self, deterministic: bool = True) -> float:
        """Evaluate current policy over the configured validation scenarios."""
        if not hasattr(self, "eval_envs"):
            self.create_eval_envs()

        rewards = []
        eval_desc = "Eval Det" if deterministic else "Eval Stoch"
        with tqdm(total=len(self.eval_envs), desc=eval_desc, position=2, dynamic_ncols=True, leave=False) as p_eval:
            for idx, env in enumerate(self.eval_envs, start=1):
                obs, _ = env.reset()
                done = False
                steps = 0
                total_reward = 0.0

                while not done and steps < self.episode_length:
                    obs_t = torch.as_tensor(obs, device=DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        if deterministic:
                            # mu action (projected)
                            _, _, mu_action_proj, _ = self.actor.sample(obs_t)
                            action_t = mu_action_proj
                        else:
                            # sampled action (projected)
                            action_t, _, _, _ = self.actor.sample(obs_t)

                    action_np = action_t.squeeze(0).cpu().numpy()
                    action_exec = np.clip(action_np, env.action_space.low, env.action_space.high)
                    obs, rew, done, truncated, _ = env.step(action_exec)
                    total_reward += rew
                    steps += 1
                    done = done or truncated

                rewards.append(total_reward)
                running_mean = float(np.mean(rewards))
                p_eval.set_postfix({
                    "scenario": f"{idx}/{len(self.eval_envs)}",
                    "reward": f"{total_reward:.2f}",
                    "avg": f"{running_mean:.2f}",
                })
                p_eval.update(1)

        return float(np.mean(rewards))


    def update(self):
        """Single SAC update step."""
        batch = self.buffer.sample(self.hp.batch_size)

        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            # Next action sampled from current policy
            next_action, logp_next, _, cost_next = self.actor.sample(next_obs)

            q1_next, q2_next = self.critics_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next)

            alpha = self.temperature.alpha

            backup = rew + self.hp.γ * (1.0 - done) * (q_next - alpha * logp_next)

        q1, q2 = self.critics(obs, act)
        critic_loss = torch.mean((q1 - backup) ** 2) + torch.mean((q2 - backup) ** 2)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        if self.hp.grad_clip:
            nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=1.0)
        self.opt_critic.step()

        # Actor update
        action_pi, logp_pi, _, cost = self.actor.sample(obs)
        q1_pi, q2_pi = self.critics(obs, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        alpha = self.temperature.alpha.detach()

        actor_loss = torch.mean(alpha * logp_pi - q_pi + self.lmbda * cost)

        self.opt_actor.zero_grad()
        actor_loss.backward()
        if self.hp.grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.opt_actor.step()

        # Temperature update
        if self.hp.auto_entropy:
            alpha_loss = torch.mean(-self.temperature.log_alpha * (logp_pi + self.hp.target_entropy).detach())
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()
        else:
            alpha_loss = torch.tensor(0.0, device=DEVICE)

        # Target critics soft update
        for param, target_param in zip(self.critics.parameters(), self.critics_target.parameters()):
            target_param.data.copy_(self.hp.τ * param.data + (1 - self.hp.τ) * target_param.data)

        # --- CHANGE (Lagrangian): lambda dual update ---
        # lambda <- max(0, min(lambda_max, lambda + lr*(E[cost] - cost_limit)))
        with torch.no_grad():
            target = cost.new_tensor(self.cost_limit)
            grad = torch.mean(cost) - target
            self.lmbda += self.lmbda_lr * grad
            self.lmbda = torch.clamp(self.lmbda, min=0.0, max=self.lmbda_max)
        # --- END CHANGE ---

        # Logging for audit
        self.q1_values.append(float(q1.mean().detach().cpu()))
        self.q2_values.append(float(q2.mean().detach().cpu()))
        self.backup_means.append(float(backup.mean().detach().cpu()))
        self.backup_abs_maxs.append(float(torch.max(torch.abs(backup)).detach().cpu()))
        self.q_next_means.append(float(q_next.mean().detach().cpu()))
        self.logp_means.append(float(logp_pi.mean().detach().cpu()))
        self.logp_mins.append(float(logp_pi.min().detach().cpu()))

        # --- CHANGE (Projection/Loss audit): cost stats and losses ---
        cost_cpu = cost.detach().view(-1).cpu().numpy()
        self.cost_means.append(float(np.mean(cost_cpu)))
        self.cost_p95s.append(float(np.percentile(cost_cpu, 95)))
        frac_violation = float((cost_cpu > self.violation_eps).mean())
        self.frac_violations.append(frac_violation)
        self.critic_losses.append(float(critic_loss.detach().cpu()))
        self.actor_losses.append(float(actor_loss.detach().cpu()))
        self.alpha_losses.append(float(alpha_loss.detach().cpu()))
        # --- END CHANGE ---


    def train(self):
        print(f"Training SAC with tariff: {self.tariff}")
        print(f"Using device: {DEVICE}")

        # Warmup
        print("Starting warmup episodes...")
        for episode in tqdm(range(self.hp.warmup_episodes), desc="Warmup Episodes", position=0, dynamic_ncols=True):
            env_dones = {"cy": False, "wy": False}
            steps = 0

            with tqdm(total=self.episode_length, desc="Warmup Steps", position=1, dynamic_ncols=True, leave=False) as pbar:
                while not all(env_dones.values()) and steps < self.episode_length:
                    for key, env in self.envs.items():
                        if env_dones[key]:
                            continue

                        obs_np = env._get_observation()
                        action = env.action_space.sample()
                        next_obs, rew, done, truncated, info = env.step(action)

                        self.buffer.add(obs_np, action, rew * self.hp.reward_scale, next_obs, done or truncated)

                        env_dones[key] = done or truncated
                        steps += 1

                        pbar.update(1)

            for key, env in self.envs.items():
                env.reset(options={"start": self.episodegen.sample(key), "bess_soc": np.random.uniform(0.1, 0.9)})

        # Training
        self.create_eval_envs()
        print("Starting training episodes...")
        for episode in (p_outer := tqdm(range(self.hp.train_episodes), desc="Training Episodes", position=0, dynamic_ncols=True)):
            env_dones = {"cy": False, "wy": False}
            reward = {"cy": 0.0, "wy": 0.0, "total": 0.0}

            q_start = len(self.q1_values)
            steps = 0
            updates_in_episode = 0
            episode_total_steps = self.episode_length * len(self.envs)

            with tqdm(total=episode_total_steps, desc=f"Ep {episode + 1}/{self.hp.train_episodes} Steps", position=1, dynamic_ncols=True, leave=False) as pbar:
                while not all(env_dones.values()) and steps < episode_total_steps:
                    for key, env in self.envs.items():
                        if env_dones[key]:
                            continue

                        obs_np = env._get_observation()
                        obs_t = torch.as_tensor(obs_np, device=DEVICE).unsqueeze(0)

                        with torch.no_grad():
                            # --- CHANGE (new sample signature): use projected action directly ---
                            action_t, _, _, _ = self.actor.sample(obs_t)
                            # --- END CHANGE ---

                        action_np = action_t.squeeze(0).cpu().numpy()
                        action_exec = np.clip(action_np, env.action_space.low, env.action_space.high)
                        next_obs, rew, done, truncated, info = env.step(action_exec)

                        self.buffer.add(obs_np, action_exec, rew * self.hp.reward_scale, next_obs, done or truncated)

                        reward[key] += rew
                        reward["total"] += rew
                        env_dones[key] = done or truncated
                        steps += 1

                        progress_pct = 100.0 * steps / max(1, episode_total_steps)
                        if (steps % self.log_every_steps == 0) or (steps == episode_total_steps):
                            pbar.set_postfix({
                                "step": f"{steps}/{episode_total_steps}",
                                "%": f"{progress_pct:.1f}",
                                "rcy": f"{reward['cy']:.2f}",
                                "rwy": f"{reward['wy']:.2f}",
                                "rtotal": f"{reward['total']:.2f}",
                                "upd": int(updates_in_episode),
                                "buf": int(self.buffer.size),
                            })
                        pbar.update(1)

                        if self.buffer.size >= self.hp.batch_size and (steps % self.update_every_steps == 0):
                            for _ in range(self.hp.update_steps):
                                self.update()
                                updates_in_episode += 1

            for key, env in self.envs.items():
                env.reset(options={"start": self.episodegen.sample(key), "bess_soc": np.random.uniform(0.1, 0.9)})

            train_total = float(reward["total"])
            self.train_rewards.append(train_total)
            self.total_episode_rewards.append(train_total)

            # Atualiza o "best_train_reward" apenas para auditoria (SEM salvar arquivos)
            if train_total > self.best_train_reward:
                self.best_train_reward = train_total

            eval_reward_value = np.nan
            eval_reward_stoch = "skipped"
            if episode % self.hp.eval_every == 0:
                eval_reward_value = float(self.eval(deterministic=True))
                self.eval_rewards.append(eval_reward_value)

                # Salva SOMENTE o melhor do ponto de vista do eval, com nomes fixos
                if eval_reward_value > self.best_eval_reward:
                    self.best_eval_reward = eval_reward_value
                    self.best_eval_episode = int(episode)
                    self._save_best_eval(eval_reward_value, episode)

            q_end = len(self.q1_values)
            n_updates = int(q_end - q_start)

            if n_updates > 0:
                q1_mean = float(np.mean(self.q1_values[q_start:q_end]))
                q2_mean = float(np.mean(self.q2_values[q_start:q_end]))

                backup_mean = float(np.mean(self.backup_means[q_start:q_end]))
                backup_abs_max = float(np.max(self.backup_abs_maxs[q_start:q_end]))
                q_next_mean = float(np.mean(self.q_next_means[q_start:q_end]))
                logp_mean = float(np.mean(self.logp_means[q_start:q_end]))
                logp_min = float(np.min(self.logp_mins[q_start:q_end]))

                # --- CHANGE (Projection/Loss audit): aggregate per-episode from per-update values ---
                cost_mean_ep = float(np.mean(self.cost_means[q_start:q_end]))
                cost_p95_ep = float(np.mean(self.cost_p95s[q_start:q_end]))
                frac_violation_ep = float(np.mean(self.frac_violations[q_start:q_end]))
                critic_loss_ep = float(np.mean(self.critic_losses[q_start:q_end]))
                actor_loss_ep = float(np.mean(self.actor_losses[q_start:q_end]))
                alpha_loss_ep = float(np.mean(self.alpha_losses[q_start:q_end]))
                # --- END CHANGE ---
            else:
                q1_mean = np.nan
                q2_mean = np.nan
                backup_mean = np.nan
                backup_abs_max = np.nan
                q_next_mean = np.nan
                logp_mean = np.nan
                logp_min = np.nan

                # --- CHANGE (Projection/Loss audit) ---
                cost_mean_ep = np.nan
                cost_p95_ep = np.nan
                frac_violation_ep = np.nan
                critic_loss_ep = np.nan
                actor_loss_ep = np.nan
                alpha_loss_ep = np.nan
                # --- END CHANGE ---

            alpha_val = float(self.temperature.alpha.detach().cpu())

            row = {
                "episode": int(episode),
                "train_reward_total": train_total,
                "eval_reward": float(eval_reward_value) if not np.isnan(eval_reward_value) else np.nan,
                "eval_reward_stoch": eval_reward_stoch,
                "best_train_reward": float(self.best_train_reward),
                "best_eval_reward": float(self.best_eval_reward),
                "q1_mean": float(q1_mean) if not np.isnan(q1_mean) else np.nan,
                "q2_mean": float(q2_mean) if not np.isnan(q2_mean) else np.nan,
                "backup_mean": float(backup_mean) if not np.isnan(backup_mean) else np.nan,
                "backup_abs_max": float(backup_abs_max) if not np.isnan(backup_abs_max) else np.nan,
                "q_next_mean": float(q_next_mean) if not np.isnan(q_next_mean) else np.nan,
                "logp_mean": float(logp_mean) if not np.isnan(logp_mean) else np.nan,
                "logp_min": float(logp_min) if not np.isnan(logp_min) else np.nan,
                "n_updates": n_updates,
                "steps": int(steps),
                "buffer_size": int(self.buffer.size),
                "alpha": alpha_val,
                # --- CHANGE (Lagrangian): log lambda ---
                "lambda": float(self.lmbda.detach().cpu().item()),
                # --- END CHANGE ---
                # --- CHANGE (Projection/Loss audit): log aggregated per-episode values ---
                "cost_mean": float(cost_mean_ep) if not np.isnan(cost_mean_ep) else np.nan,
                "cost_p95": float(cost_p95_ep) if not np.isnan(cost_p95_ep) else np.nan,
                "frac_violation": float(frac_violation_ep) if not np.isnan(frac_violation_ep) else np.nan,
                "critic_loss": float(critic_loss_ep) if not np.isnan(critic_loss_ep) else np.nan,
                "actor_loss": float(actor_loss_ep) if not np.isnan(actor_loss_ep) else np.nan,
                "alpha_loss": float(alpha_loss_ep) if not np.isnan(alpha_loss_ep) else np.nan,
                # --- END CHANGE ---
            }

            # Evita FutureWarning do concat com DF vazio
            self.audit_df.loc[len(self.audit_df)] = row
            if ((episode + 1) % self.audit_every_episodes == 0) or ((episode + 1) == self.hp.train_episodes):
                self.audit_df.to_csv(self.audit_csv, index=False)

            # console log
            p_outer.set_postfix({
                "train_total": f"{train_total:.2f}",
                "eval": f"{eval_reward_value:.2f}" if not np.isnan(eval_reward_value) else "-",
                "eval_stoch": eval_reward_stoch,
                "best_eval": f"{self.best_eval_reward:.2f}",
                "alpha": f"{alpha_val:.3f}",
                "lambda": f"{float(self.lmbda.detach().cpu().item()):.3f}",
                "frac_viol": f"{float(frac_violation_ep):.3f}" if not np.isnan(frac_violation_ep) else "nan",
            })


def main():
    tariff = "tar_tou"
    trainer = Train(tariff)
    trainer.train()


if __name__ == "__main__":
    main()
