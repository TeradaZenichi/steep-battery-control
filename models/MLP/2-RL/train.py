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

        self.env_cy = SmartHomeEnv(
            self.episodegen.df_cy,
            self.parameters,
            start=self.episodegen.sample("cy"),
            days=self.hp.days,
            BESS_SoC=0.5,
            tariff=self.tariff,
        )
        self.env_wy = SmartHomeEnv(
            self.episodegen.df_wy,
            self.parameters,
            start=self.episodegen.sample("wy"),
            days=self.hp.days,
            BESS_SoC=0.5,
            tariff=self.tariff,
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
        self.cost_limit = float(self.train_cfg["train"].get("cost_limit", 0.0))  # target cost (0 means no violation)
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
        self.violation_eps = float(self.train_cfg["train"].get("violation_eps", 1e-8))
        # --- END CHANGE ---

        # audit dataframe (updated at end of each episode)
        self.audit_df = pd.DataFrame(columns=[
            "episode",
            "train_reward_total",
            "eval_reward",
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

        # 3) Meta (handy for quick inspection)
        meta = {
            "best_eval_reward": float(eval_reward_value),
            "best_eval_episode": int(episode),
            "tariff": self.tariff,
        }
        with open(self.best_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


    def create_eval_envs(self):
        runs = self.train_cfg["val"]
        self.eval_envs = []
        for run in runs:
            df = pd.read_csv(
                run["dataset"],
                sep=";",
                parse_dates=["timestamp"],
                dayfirst=True,
                index_col="timestamp",
            )
            date = pd.to_datetime(run["date"], format="%Y-%m-%d %H:%M:%S")
            env = SmartHomeEnv(
                df,
                self.parameters,
                start=date,
                days=run["days"],
                BESS_SoC=run["soc"],
                tariff=self.tariff,
            )
            self.eval_envs.append(env)


    def eval(self):
        total_reward = 0.0

        for env in self.eval_envs:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            done = False
            truncated = False
            episode_reward = 0.0
            steps = 0

            while (not done) and (not truncated) and steps < self.episode_length:
                obs_t = torch.as_tensor(obs, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    # --- CHANGE (Projected deterministic eval): use projected mean action from sample() ---
                    # sample() returns (action_proj, logp, mu_action_proj, cost)
                    _, _, action_t, _ = self.actor.sample(obs_t)
                    # --- END CHANGE ---
                action_np = action_t.squeeze(0).cpu().numpy()
                action_exec = np.clip(action_np, env.action_space.low, env.action_space.high)

                next_obs, rew, done, truncated, info = env.step(action_exec)

                episode_reward += float(rew)
                obs = next_obs
                steps += 1

            total_reward += episode_reward

        avg_reward = total_reward / len(self.eval_envs)
        return avg_reward


    def update(self):
        batch = self.buffer.sample(self.hp.batch_size)
        observations, actions, rewards, dones, next_obs = (
            batch["obs"],
            batch["acts"],
            batch["rews"],
            batch["dones"].float(),
            batch["next_obs"],
        )

        # Critic target
        with torch.no_grad():
            # --- CHANGE (new sample signature): ignore mu_action and cost here ---
            next_action, log_prob_next, _, _ = self.actor.sample(next_obs)
            # --- END CHANGE ---
            q1_next, q2_next = self.critics_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next)
            backup = rewards + self.hp.γ * (1 - dones) * (
                q_next - self.temperature.alpha.detach() * log_prob_next
            )

            self.backup_means.append(float(backup.detach().mean().cpu()))
            self.backup_abs_maxs.append(float(backup.detach().abs().max().cpu()))
            self.q_next_means.append(float(q_next.detach().mean().cpu()))

        # Critic update
        q1 = self.critics.q1(observations, actions)
        q2 = self.critics.q2(observations, actions)

        self.q1_values.append(float(q1.detach().mean().cpu()))
        self.q2_values.append(float(q2.detach().mean().cpu()))

        critics_loss = nn.MSELoss()(q1, backup) + nn.MSELoss()(q2, backup)

        # --- CHANGE (Loss audit): log critic loss per update (detached) ---
        self.critic_losses.append(float(critics_loss.detach().cpu()))
        # --- END CHANGE ---

        self.opt_critic.zero_grad()
        critics_loss.backward()
        if self.hp.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 10.0)
        self.opt_critic.step()

        # actor update
        for param in self.critics.parameters():
            param.requires_grad_(False)

        # --- CHANGE (new sample signature): get cost for Lagrangian penalty ---
        π_action, log_prob, _, cost = self.actor.sample(observations)
        # --- END CHANGE ---

        self.logp_means.append(float(log_prob.detach().mean().cpu()))
        self.logp_mins.append(float(log_prob.detach().min().cpu()))

        q1_π = self.critics.q1(observations, π_action)
        q2_π = self.critics.q2(observations, π_action)
        q_π = torch.min(q1_π, q2_π)

        # --- CHANGE (Lagrangian): add lambda * cost penalty to actor objective ---
        # Lagrangian penalty: encourages the policy to avoid infeasible raw actions
        # by penalizing the projected violation cost returned by sample().
        actor_loss = (self.temperature.alpha.detach() * log_prob - q_π + self.lmbda.detach() * cost).mean()
        # --- END CHANGE ---

        # --- CHANGE (Projection/Loss audit): log cost stats and actor loss per update ---
        # Note: cost is expected to be per-sample (batch-wise). If it is scalar, p95==mean.
        with torch.no_grad():
            cost_cpu = cost.detach().view(-1).cpu().numpy()
            cost_mean = float(cost_cpu.mean())
            cost_p95 = float(np.percentile(cost_cpu, 95)) if cost_cpu.size > 1 else float(cost_mean)
            frac_violation = float((cost_cpu > self.violation_eps).mean())
        self.cost_means.append(cost_mean)
        self.cost_p95s.append(cost_p95)
        self.frac_violations.append(frac_violation)
        self.actor_losses.append(float(actor_loss.detach().cpu()))
        # --- END CHANGE ---

        self.opt_actor.zero_grad()
        actor_loss.backward()
        if self.hp.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_actor.step()

        # --- CHANGE (Lagrangian): dual ascent update for lambda (keep lambda >= 0) ---
        # Dual update (projected gradient ascent): lambda <- [lambda + lr*(E[cost] - cost_limit)]_+
        with torch.no_grad():
            target = cost.new_tensor(self.cost_limit)
            self.lmbda += self.lmbda_lr * (cost.mean() - target)
            self.lmbda.clamp_(0.0, self.lmbda_max)
        # --- END CHANGE ---

        for param in self.critics.parameters():
            param.requires_grad_(True)

        # Temperature (entropy) update
        alpha_loss = self.temperature.loss(log_prob.detach())

        # --- CHANGE (Loss audit): log alpha loss per update (detached) ---
        self.alpha_losses.append(float(alpha_loss.detach().cpu()))
        # --- END CHANGE ---

        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # Soft update of target networks
        with torch.no_grad():
            for param, target_param in zip(self.critics.parameters(), self.critics_target.parameters()):
                target_param.copy_(self.hp.τ * param + (1 - self.hp.τ) * target_param)


    def warmup(self):
        print("Starting warmup episodes...")
        for episode in tqdm(range(self.hp.warmup_episodes), desc="Warmup Episodes"):
            env_dones = {"cy": False, "wy": False}
            steps = 0
            with tqdm(total=self.episode_length, desc="Warmup Steps", position=1, dynamic_ncols=True, leave=False) as pbar:
                while not all(env_dones.values()) and steps < self.episode_length:
                    for key, env in self.envs.items():
                        if env_dones[key]:
                            continue

                        obs_np = env._get_observation()
                        action_np = np.asarray(env.action_space.sample(), dtype=np.float32)
                        action_exec = np.clip(action_np, env.action_space.low, env.action_space.high)

                        next_obs, rew, done, truncated, info = env.step(action_exec)

                        self.buffer.add(obs_np, action_exec, rew * self.hp.reward_scale, next_obs, done or truncated)
                        env_dones[key] = done or truncated
                        steps += 1
                        pbar.update(1)

            for key, env in self.envs.items():
                env.reset(options={"start": self.episodegen.sample(key), "bess_soc": np.random.uniform(0.1, 0.9)})


    def train(self):
        self.create_eval_envs()
        print("Starting training episodes...")
        for episode in (p_outer := tqdm(range(self.hp.train_episodes), desc="Training Episodes", position=0, dynamic_ncols=True)):
            env_dones = {"cy": False, "wy": False}
            reward = {"cy": 0.0, "wy": 0.0, "total": 0.0}

            q_start = len(self.q1_values)
            steps = 0

            with tqdm(total=self.episode_length, desc="Train Steps", position=1, dynamic_ncols=True, leave=False) as pbar:
                while not all(env_dones.values()) and steps < self.episode_length:
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

                        pbar.set_postfix({"rcy": f"{reward['cy']:.2f}", "rwy": f"{reward['wy']:.2f}", "rtotal": f"{reward['total']:.2f}"})
                        pbar.update(1)

                        if self.buffer.size >= self.hp.batch_size:
                            for _ in range(self.hp.update_steps):
                                self.update()

            for key, env in self.envs.items():
                env.reset(options={"start": self.episodegen.sample(key), "bess_soc": np.random.uniform(0.1, 0.9)})

            train_total = float(reward["total"])
            self.train_rewards.append(train_total)
            self.total_episode_rewards.append(train_total)

            # Atualiza o "best_train_reward" apenas para auditoria (SEM salvar arquivos)
            if train_total > self.best_train_reward:
                self.best_train_reward = train_total

            eval_reward_value = np.nan
            if episode % self.hp.eval_every == 0:
                eval_reward_value = float(self.eval())
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
            self.audit_df.to_csv(self.audit_csv, index=False)

            p_outer.set_postfix({"best_eval": f"{self.best_eval_reward:.2f}", "train": f"{train_total:.2f}"})


    def run(self):
        self.warmup()
        self.train()


def main():
    for tariff in ["tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat"]:
        train = Train(tariff)
        train.run()

    a = 1


if __name__ == "__main__":
    main()
