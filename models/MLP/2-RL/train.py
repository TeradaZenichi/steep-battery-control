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

        self.buffer = ReplayBuffer(
            capacity=self.hp.buffer_size,
            obs_dim=self.model_cfg["actor"]["input_dim"],
            act_dim=self.model_cfg["actor"]["output_dim"],
            device=DEVICE,
        )

        self.actor = load_actor(self.model_cfg["actor"], device=DEVICE)
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

        # eval and reward tracking
        self.best_eval_reward  = -float("inf")
        self.best_train_reward = -float("inf")

        # tracking lists
        self.eval_rewards = []
        self.train_rewards = []
        self.q1_values = []
        self.q2_values = []
        self.total_episode_rewards = []

        # audit dataframe (updated at end of each episode)
        self.audit_df = pd.DataFrame(columns=[
            "episode",
            "train_reward_total",
            "eval_reward",
            "best_train_reward",
            "best_eval_reward",
            "q1_mean",
            "q2_mean",
            "n_updates",
            "steps",
            "buffer_size",
            "alpha",
        ])
        self.audit_csv = self.folder / "audit_training.csv"


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
            if isinstance(obs, dict):
                obs = obs["obs"] if "obs" in obs else (obs["observation"] if "observation" in obs else next(iter(obs.values())))

            done = False
            truncated = False
            episode_reward = 0.0
            steps = 0

            while (not done) and (not truncated) and steps < self.episode_length:
                obs_t = torch.as_tensor(obs, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    action_t = self.actor(obs_t)
                action_np = action_t.squeeze(0).cpu().numpy()

                next_obs, rew, done, truncated, info = env.step(action_np)

                if isinstance(next_obs, tuple):
                    next_obs = next_obs[0]
                if isinstance(next_obs, dict):
                    next_obs = next_obs["obs"] if "obs" in next_obs else (next_obs["observation"] if "observation" in next_obs else next(iter(next_obs.values())))

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
            next_action, log_prob_next, μ_action = self.actor.sample(next_obs)
            q1_next, q2_next = self.critics_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next)
            backup = rewards + self.hp.γ * (1 - dones) * (
                q_next - self.temperature.alpha.detach() * log_prob_next
            )

        # Critic update
        q1 = self.critics.q1(observations, actions)
        q2 = self.critics.q2(observations, actions)

        self.q1_values.append(float(q1.detach().mean().cpu()))
        self.q2_values.append(float(q2.detach().mean().cpu()))

        critics_loss = nn.MSELoss()(q1, backup) + nn.MSELoss()(q2, backup)
        self.opt_critic.zero_grad()
        critics_loss.backward()
        if self.hp.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 10.0)
        self.opt_critic.step()

        # actor update (freeze critic grads to avoid unnecessary accumulation)
        for param in self.critics.parameters():
            param.requires_grad_(False)

        π_action, log_prob, μ_action = self.actor.sample(observations)
        q1_π = self.critics.q1(observations, π_action)
        q2_π = self.critics.q2(observations, π_action)
        q_π = torch.min(q1_π, q2_π)
        actor_loss = (self.temperature.alpha.detach() * log_prob - q_π).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        if self.hp.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_actor.step()

        for param in self.critics.parameters():
            param.requires_grad_(True)

        # Temperature (entropy) update
        alpha_loss = self.temperature.loss(log_prob.detach())
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
                        action_np = np.random.uniform(
                            -1.0, 1.0, size=self.model_cfg["actor"]["output_dim"]
                        )

                        next_obs, rew, done, truncated, info = env.step(action_np)

                        if isinstance(next_obs, tuple):
                            next_obs = next_obs[0]
                        if isinstance(next_obs, dict):
                            next_obs = next_obs["obs"] if "obs" in next_obs else (next_obs["observation"] if "observation" in next_obs else next(iter(next_obs.values())))

                        self.buffer.add(obs_np, action_np, rew * self.hp.reward_scale, next_obs, done or truncated)
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
                            action_t = self.actor(obs_t)

                        action_np = action_t.squeeze(0).cpu().numpy()
                        next_obs, rew, done, truncated, info = env.step(action_np)

                        if isinstance(next_obs, tuple):
                            next_obs = next_obs[0]
                        if isinstance(next_obs, dict):
                            next_obs = next_obs["obs"] if "obs" in next_obs else (next_obs["observation"] if "observation" in next_obs else next(iter(next_obs.values())))

                        self.buffer.add(obs_np, action_np, rew * self.hp.reward_scale, next_obs, done or truncated)

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

            if train_total > self.best_train_reward:
                self.best_train_reward = train_total
                torch.save(
                    self.actor.state_dict(),
                    self.folder / f"best_actor_train_reward_{self.best_train_reward:.2f}.pt",
                )

            eval_reward_value = np.nan
            if episode % self.hp.eval_every == 0:
                eval_reward_value = float(self.eval())
                self.eval_rewards.append(eval_reward_value)

                if eval_reward_value > self.best_eval_reward:
                    self.best_eval_reward = eval_reward_value
                    torch.save(
                        self.actor.state_dict(),
                        self.folder / f"best_actor_eval_reward_{self.best_eval_reward:.2f}.pt",
                    )

            q_end = len(self.q1_values)
            n_updates = int(q_end - q_start)
            if n_updates > 0:
                q1_mean = float(np.mean(self.q1_values[q_start:q_end]))
                q2_mean = float(np.mean(self.q2_values[q_start:q_end]))
            else:
                q1_mean = np.nan
                q2_mean = np.nan

            alpha_val = float(self.temperature.alpha.detach().cpu())

            row = {
                "episode": int(episode),
                "train_reward_total": train_total,
                "eval_reward": float(eval_reward_value) if not np.isnan(eval_reward_value) else np.nan,
                "best_train_reward": float(self.best_train_reward),
                "best_eval_reward": float(self.best_eval_reward),
                "q1_mean": float(q1_mean) if not np.isnan(q1_mean) else np.nan,
                "q2_mean": float(q2_mean) if not np.isnan(q2_mean) else np.nan,
                "n_updates": n_updates,
                "steps": int(steps),
                "buffer_size": int(self.buffer.size),
                "alpha": alpha_val,
            }

            self.audit_df = pd.concat([self.audit_df, pd.DataFrame([row])], ignore_index=True)
            self.audit_df.to_csv(self.audit_csv, index=False)

            p_outer.set_postfix({"best_eval": f"{self.best_eval_reward:.2f}", "best_train": f"{self.best_train_reward:.2f}"})


    def run(self):
        self.warmup()
        self.train()



def main():
    tariff = "tar_tou"
    train = Train(tariff)
    train.run()

    a = 1


if __name__ == "__main__":
    main()
