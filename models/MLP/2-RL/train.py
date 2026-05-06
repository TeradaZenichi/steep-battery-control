from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import json
import random
import time
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../steep-battery-control
MODEL_ROOT   = Path(__file__).resolve().parents[2]  # .../models
MLP_ROOT   = Path(__file__).resolve().parent.parent   # .../models/MLP
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

        with open(MLP_ROOT / "model.json", encoding="utf-8") as f:
            self.model_cfg = json.load(f)

        with open(Path(__file__).resolve().parent / "config.json", encoding="utf-8") as f:
            self.train_cfg = json.load(f)

        with open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8") as f:
            self.parameters = json.load(f)

        self.episodegen = EpisodeGen(self.train_cfg, PROJECT_ROOT / "data")
        self.hp = Hyperparameters(self.train_cfg["train"])
        self.actor_weight_decay = 0.0
        self.history_len = int(self.train_cfg["train"].get("history_len", 1))
        print(f"[history_len] {tariff} = {self.history_len} (from RL config)")
        self.log_every_steps = 50
        self.audit_every_episodes = 5
        self.update_every_steps = int(self.train_cfg["train"]["update_every_steps"])
        self.eval_workers = 12
        self.train_env_workers = 1
        self.early_stop_patience = int(self.train_cfg["train"]["early_stop_patience"])
        self.min_episodes_before_early_stop = 100
        self.checkpoint_min_delta = 0.0
        self.checkpoint_metric = "det"
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

        # Defina o tamanho do episÃƒÂ³dio (nÃƒÂºmero de steps por episÃƒÂ³dio)
        self.episode_length = int(24 * 60 // self.env_cy.sim.timestep * self.hp.days)

        torch.manual_seed(self.hp.seed)
        np.random.seed(self.hp.seed)
        random.seed(self.hp.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.hp.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.folder = PROJECT_ROOT / "Results" / "train" / "MLP" / "2-RL" / self.tariff
        self.folder.mkdir(parents=True, exist_ok=True)

        # Arquivos ÃƒÂºnicos (sobrescrevem)
        self.best_actor_path = self.folder / "best_actor_eval.pt"
        self.best_ckpt_path  = self.folder / "best_checkpoint_eval.pt"
        self.best_meta_path  = self.folder / "best_eval_meta.json"


        self.best_actor_det_path = self.folder / "best_actor_eval_det.pt"
        self.best_ckpt_det_path  = self.folder / "best_checkpoint_eval_det.pt"
        self.best_meta_det_path  = self.folder / "best_eval_det_meta.json"
        self.best_actor_robust_path = self.folder / "best_actor_eval_robust.pt"
        self.best_ckpt_robust_path  = self.folder / "best_checkpoint_eval_robust.pt"
        self.best_meta_robust_path  = self.folder / "best_eval_robust_meta.json"
        self.best_actor_operational_path = self.folder / "best_actor_eval_operational.pt"
        self.best_ckpt_operational_path  = self.folder / "best_checkpoint_eval_operational.pt"
        self.best_meta_operational_path  = self.folder / "best_eval_operational_meta.json"
        self.final_actor_path = self.folder / "final_actor.pt"
        self.final_ckpt_path  = self.folder / "final_checkpoint.pt"
        self.final_meta_path  = self.folder / "final_eval_meta.json"

        self.buffer = ReplayBuffer(
            capacity=self.hp.buffer_size,
            obs_dim=self.model_cfg["actor"]["input_dim"],
            act_dim=self.model_cfg["actor"]["output_dim"],
            device=DEVICE,
            n_step=self.hp.n_step,
            gamma=self.hp.gamma,
        )

        # Merge actor architecture (model.json) with RL-specific stochastic settings (config.json).
        self.actor_cfg = dict(self.model_cfg["actor"])
        self.actor_cfg["log_std_min"] = float(self.hp.log_std_min)
        self.actor_cfg["log_std_max"] = float(self.hp.log_std_max)
        self.actor_cfg["parameters"] = str((PROJECT_ROOT / "data" / "parameters.json").resolve())

        self.actor = load_actor(self.actor_cfg, device=DEVICE)
        self.critic_cfg = dict(self.model_cfg["critic"])
        self.critic_cfg.setdefault("state_dim", int(self.model_cfg["actor"]["input_dim"]))
        self.critic_cfg.setdefault("action_dim", int(self.model_cfg["actor"]["output_dim"]))
        self.critics = load_critic(self.critic_cfg, device=DEVICE)
        self.critics_target = load_critic(self.critic_cfg, device=DEVICE)
        self.critics_target.load_state_dict(self.critics.state_dict(), strict=True)

        self.temperature = Temperature(
            init_log_alpha=0.0,
            target_entropy=self.hp.target_entropy
        ).to(DEVICE)

        self.opt_alpha = torch.optim.Adam([self.temperature.log_alpha], lr=self.hp.alpha_lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.hp.actor_lr, weight_decay=self.actor_weight_decay)
        self.opt_critic = torch.optim.Adam(self.critics.parameters(), lr=self.hp.critic_lr)

        self.best_eval_reward = -float("inf")
        self.best_eval_episode = -1
        self.best_checkpoint_score = -float("inf")
        self.best_checkpoint_episode = -1
        self.best_robust_score = -float("inf")
        self.best_robust_episode = -1
        self.best_operational_score = -float("inf")
        self.best_operational_key = (-float("inf"), -float("inf"), -float("inf"))
        self.best_operational_episode = -1
        self.operational_reward_tolerance = 5.0
        self.best_train_reward = -float("inf")
        self.last_improvement_episode = -1
        self.last_improvement_eval_count = -1
        self.eval_count = 0
        self.no_improve_evals = 0


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

        # --- CHANGE (Bounded-action audit): extra per-update metrics ---
        # These are appended once per update, and sliced per-episode using q_start:q_end.
        self.cost_means = []
        self.cost_p95s = []
        self.frac_violations = []
        self.actor_losses = []
        self.actor_term_entropies = []
        self.actor_term_qs = []
        self.critic_losses = []
        self.alpha_losses = []
        self.violation_eps = 1e-6
        # --- END CHANGE ---

        # audit dataframe (updated at end of each episode)
        self.audit_df = pd.DataFrame(columns=[
            "episode",
            "train_reward_total",
            "eval_reward_det",
            "eval_reward",
            "eval_reward_worst",
            "eval_reward_robust",
            "eval_operational_score",
            "checkpoint_score",
            "best_train_reward",
            "best_eval_reward",
            "best_checkpoint_score",
            "best_robust_score",
            "best_operational_score",
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
            "cost_mean",
            "cost_p95",
            "frac_violation",
            "critic_loss",
            "actor_loss",
            "actor_term_entropy",
            "actor_term_q",
            "alpha_loss",
            "no_improve_episodes",
            "no_improve_evals",
            "iteration_time_sec",
        ])
        self.audit_csv = self.folder / "audit_training.csv"

        self._audit_pending_rows = []
        self._eval_executor = None
        self._eval_executor_workers = 0
        self._eval_parallel_enabled = True
        self._eval_cache_tag = 0
        self.eval_runs_full = list(self.train_cfg["val"])
        self.eval_runs_train = list(self.eval_runs_full)
        self.final_full_eval = {}
        self.last_eval_stats = {}



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
        }
        torch.save(ckpt, filepath)



    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): Train._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Train._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return Train._json_safe(value.tolist())
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value


    def _checkpoint_meta(self, metric: str, stats: dict, episode: int, score: float) -> dict:
        return {
            "best_eval_reward": float(stats.get("mean_reward", np.nan)),
            "best_eval_episode": int(episode),
            "checkpoint_score": float(score),
            "checkpoint_metric": metric,
            "tariff": self.tariff,
            "eval_stats": self._json_safe(stats),
        }


    def _save_checkpoint_bundle(self, actor_path: Path, ckpt_path: Path, meta_path: Path, meta: dict, mirror_legacy: bool = False) -> None:
        torch.save(self.actor.state_dict(), actor_path)
        self.save_checkpoint(ckpt_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if mirror_legacy:
            torch.save(self.actor.state_dict(), self.best_actor_path)
            self.save_checkpoint(self.best_ckpt_path)
            with open(self.best_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)


    def _save_best_eval_det(self, stats: dict, episode: int) -> None:
        score = float(stats["mean_reward"])
        meta = self._checkpoint_meta("mean_reward_det", stats, episode, score)
        self._save_checkpoint_bundle(self.best_actor_det_path, self.best_ckpt_det_path, self.best_meta_det_path, meta, mirror_legacy=True)


    def _save_best_eval_robust(self, stats: dict, episode: int) -> None:
        score = float(stats["robust_score"])
        meta = self._checkpoint_meta("worst_two_mean_reward", stats, episode, score)
        self._save_checkpoint_bundle(self.best_actor_robust_path, self.best_ckpt_robust_path, self.best_meta_robust_path, meta)


    def _save_best_eval_operational(self, stats: dict, episode: int, key: tuple[float, float, float]) -> None:
        score = float(stats["operational_score"])
        meta = self._checkpoint_meta("operational_cost_with_reward_tolerance", stats, episode, score)
        meta["operational_reward_tolerance"] = float(self.operational_reward_tolerance)
        meta["operational_key"] = [float(x) for x in key]
        self._save_checkpoint_bundle(self.best_actor_operational_path, self.best_ckpt_operational_path, self.best_meta_operational_path, meta)


    def _save_final_checkpoint(self, episode: int) -> None:
        torch.save(self.actor.state_dict(), self.final_actor_path)
        self.save_checkpoint(self.final_ckpt_path)
        meta = {"episode": int(episode), "checkpoint_metric": "final_episode", "tariff": self.tariff, "eval_stats": self._json_safe(self.final_full_eval)}
        with open(self.final_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


    def _maybe_save_mean_checkpoint(self, stats: dict, episode: int) -> bool:
        score = float(stats.get("mean_reward", np.nan))
        if np.isnan(score) or score <= self.best_eval_reward + self.checkpoint_min_delta:
            return False
        self.best_eval_reward = score
        self.best_eval_episode = int(episode)
        self.best_checkpoint_score = score
        self.best_checkpoint_episode = int(episode)
        self._save_best_eval_det(stats, episode)
        return True


    def _maybe_save_robust_checkpoint(self, stats: dict, episode: int) -> None:
        score = float(stats.get("robust_score", np.nan))
        if np.isnan(score) or score <= self.best_robust_score + self.checkpoint_min_delta:
            return
        self.best_robust_score = score
        self.best_robust_episode = int(episode)
        self._save_best_eval_robust(stats, episode)


    def _maybe_save_operational_checkpoint(self, stats: dict, episode: int) -> None:
        mean_reward = float(stats.get("mean_reward", np.nan))
        operational_cost = float(stats.get("operational_cost", np.nan))
        if np.isnan(mean_reward) or np.isnan(operational_cost):
            return
        reward_ok = mean_reward >= self.best_eval_reward - self.operational_reward_tolerance
        key = (1.0 if reward_ok else 0.0, -operational_cost, mean_reward)
        if key <= self.best_operational_key:
            return
        self.best_operational_key = key
        self.best_operational_score = float(stats.get("operational_score", mean_reward - operational_cost))
        self.best_operational_episode = int(episode)
        self._save_best_eval_operational(stats, episode, key)


    def _next_eval_cache_tag(self) -> int:
        tag = int(self._eval_cache_tag)
        self._eval_cache_tag += 1
        return tag


    def _close_eval_executor(self) -> None:
        if self._eval_executor is not None:
            self._eval_executor.shutdown(wait=True, cancel_futures=False)
            self._eval_executor = None
            self._eval_executor_workers = 0


    def _get_eval_executor(self, eval_workers: int) -> ProcessPoolExecutor:
        if self._eval_executor is None or self._eval_executor_workers != int(eval_workers):
            self._close_eval_executor()
            self._eval_executor = ProcessPoolExecutor(max_workers=int(eval_workers))
            self._eval_executor_workers = int(eval_workers)
        return self._eval_executor


    @staticmethod
    def _summarize_eval_results(results: list[dict]) -> dict:
        if not results:
            return {"mean_reward": np.nan, "worst_reward": np.nan, "robust_score": np.nan, "std_reward": np.nan, "operational_cost": np.nan, "operational_score": np.nan, "rewards": []}
        rewards = np.asarray([float(r["reward"]) for r in results], dtype=np.float64)
        sorted_rewards = np.sort(rewards)
        tail_k = min(2, len(sorted_rewards))
        def mean_metric(name: str) -> float:
            vals = [float(r.get(name, 0.0)) for r in results]
            return float(np.mean(vals)) if vals else 0.0
        grid_penalty = mean_metric("grid_penalty")
        ev_cost = mean_metric("ev_cost")
        pv_cost = mean_metric("pv_cost")
        operational_cost = grid_penalty + ev_cost + pv_cost
        return {
            "mean_reward": float(np.mean(rewards)),
            "worst_reward": float(np.min(rewards)),
            "robust_score": float(np.mean(sorted_rewards[:tail_k])),
            "std_reward": float(np.std(rewards)),
            "operational_cost": float(operational_cost),
            "operational_score": float(np.mean(rewards) - operational_cost),
            "grid_penalty_mean": grid_penalty,
            "ev_cost_mean": ev_cost,
            "pv_cost_mean": pv_cost,
            "energy_cost_mean": mean_metric("energy_cost"),
            "bess_cost_mean": mean_metric("bess_cost"),
            "pv_cmd_mean": mean_metric("pv_cmd_mean"),
            "bess_abs_power_mean": mean_metric("bess_abs_power_mean"),
            "ev_abs_power_mean": mean_metric("ev_abs_power_mean"),
            "rewards": [float(x) for x in rewards],
        }


    def eval(self, deterministic: bool = True, runs: list | None = None, eval_desc: str | None = None) -> float:
        runs = self.eval_runs_full if runs is None else list(runs)
        if not runs:
            self.last_eval_stats = self._summarize_eval_results([])
            return float("nan")
        eval_desc = eval_desc or ("Eval Det" if deterministic else "Eval Stoch")
        eval_workers = max(1, min(self.eval_workers, len(runs)))
        actor_state_cpu = {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}
        eval_cache_tag = self._next_eval_cache_tag()
        worker_args_suffix = [deterministic, eval_cache_tag]

        results = []
        with tqdm(total=len(runs), desc=eval_desc, position=2, dynamic_ncols=True, leave=False) as p_eval:
            futures = None
            if eval_workers > 1 and self._eval_parallel_enabled:
                try:
                    executor = self._get_eval_executor(eval_workers)
                    futures = [executor.submit(_eval_worker, run, self.parameters, self.tariff, self.actor_cfg, actor_state_cpu, self.episode_length, *worker_args_suffix) for run in runs]
                except (PermissionError, OSError) as exc:
                    print(f"[eval] process pool unavailable ({exc}); falling back to sequential evaluation.")
                    self._eval_parallel_enabled = False
                    self._close_eval_executor()
            iterator = as_completed(futures) if futures is not None else runs
            for idx, item in enumerate(iterator, start=1):
                result = item.result() if futures is not None else _eval_worker(item, self.parameters, self.tariff, self.actor_cfg, actor_state_cpu, self.episode_length, *worker_args_suffix)
                if not isinstance(result, dict):
                    result = {"reward": float(result)}
                result["reward"] = float(result["reward"])
                results.append(result)
                rewards = [float(r["reward"]) for r in results]
                p_eval.set_postfix({"scenario": f"{idx}/{len(runs)}", "reward": f"{result['reward']:.2f}", "avg": f"{float(np.mean(rewards)):.2f}", "w": int(eval_workers) if futures is not None else 1})
                p_eval.update(1)
        self.last_eval_stats = self._summarize_eval_results(results)
        return float(self.last_eval_stats["mean_reward"])


    @staticmethod
    def _ensure_finite(name: str, tensor: torch.Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite tensor detected in update(): {name}")


    def update(self, episode=None):
        """Single SAC update step."""
        batch = self.buffer.sample(self.hp.batch_size)

        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        gamma_pow = batch["gamma_pow"]

        with torch.inference_mode():
            # Next action sampled from current policy
            next_action, logp_next, _, _ = self.actor.sample(next_obs)

            q1_next, q2_next = self.critics_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next)

            alpha = self.temperature.alpha

            backup = rew + gamma_pow * (1.0 - done) * (q_next - alpha * logp_next)
            self._ensure_finite("q_next", q_next)
            self._ensure_finite("logp_next", logp_next)
            self._ensure_finite("backup", backup)

        q1, q2 = self.critics(obs, act)
        critic_loss = torch.mean((q1 - backup) ** 2) + torch.mean((q2 - backup) ** 2)
        self._ensure_finite("q1", q1)
        self._ensure_finite("q2", q2)
        self._ensure_finite("critic_loss", critic_loss)

        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.hp.grad_clip:
            nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=1.0)
        self.opt_critic.step()

        # Actor update
        action_pi, logp_pi, _, cost = self.actor.sample(obs)
        q1_pi, q2_pi = self.critics(obs, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        alpha = self.temperature.alpha.detach()

        actor_term_entropy = alpha * logp_pi
        actor_term_q = -q_pi
        actor_loss = torch.mean(actor_term_entropy + actor_term_q)
        self._ensure_finite("logp_pi", logp_pi)
        self._ensure_finite("q_pi", q_pi)
        self._ensure_finite("cost", cost)
        self._ensure_finite("actor_loss", actor_loss)

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.hp.grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.opt_actor.step()

        # Temperature update
        if self.hp.auto_entropy:
            alpha_loss = torch.mean(-self.temperature.log_alpha * (logp_pi + self.hp.target_entropy).detach())
            self.opt_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.opt_alpha.step()
        else:
            alpha_loss = torch.tensor(0.0, device=DEVICE)

        self._ensure_finite("alpha_loss", alpha_loss)

        # Target critics soft update
        tau = self.hp.tau
        for param, target_param in zip(self.critics.parameters(), self.critics_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Logging for audit
        self.q1_values.append(float(q1.mean().detach().cpu()))
        self.q2_values.append(float(q2.mean().detach().cpu()))
        self.backup_means.append(float(backup.mean().detach().cpu()))
        self.backup_abs_maxs.append(float(torch.max(torch.abs(backup)).detach().cpu()))
        self.q_next_means.append(float(q_next.mean().detach().cpu()))
        self.logp_means.append(float(logp_pi.mean().detach().cpu()))
        self.logp_mins.append(float(logp_pi.min().detach().cpu()))

        # --- CHANGE (Bounded-action audit): cost stats and losses ---
        cost_cpu = cost.detach().view(-1).cpu().numpy()
        self.cost_means.append(float(np.mean(cost_cpu)))
        self.cost_p95s.append(float(np.percentile(cost_cpu, 95)))
        frac_violation = float((cost_cpu > self.violation_eps).mean())
        self.frac_violations.append(frac_violation)
        self.critic_losses.append(float(critic_loss.detach().cpu()))
        self.actor_losses.append(float(actor_loss.detach().cpu()))
        self.actor_term_entropies.append(float(torch.mean(actor_term_entropy).detach().cpu()))
        self.actor_term_qs.append(float(torch.mean(actor_term_q).detach().cpu()))
        self.alpha_losses.append(float(alpha_loss.detach().cpu()))
        # --- END CHANGE ---


    def _reset_train_envs(self):
        for key, env in self.envs.items():
            env.reset(options={"start": self.episodegen.sample(key), "bess_soc": np.random.uniform(0.1, 0.9)})


    def _sample_warmup_action(self, obs) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32)
        obs_t = torch.as_tensor(obs_arr[None, ...], device=DEVICE)
        with torch.inference_mode():
            amin_b, amax_b, amin_e, amax_e = self.actor._action_bounds(obs_t)
            a_bess = amin_b + torch.rand_like(amin_b) * (amax_b - amin_b)
            a_ev = amin_e + torch.rand_like(amin_e) * (amax_e - amin_e)
            a_pv = torch.rand_like(amin_b)
            action_t = torch.cat([a_bess, a_ev, a_pv], dim=-1)
            action_t, _ = self.actor._project(obs_t, action_t)
        return action_t[0].detach().cpu().numpy()


    def _run_warmup(self):
        print("Starting warmup episodes...")
        for episode in tqdm(range(self.hp.warmup_episodes), desc="Warmup Episodes", position=0, dynamic_ncols=True):
            env_dones = {"cy": False, "wy": False}
            steps = 0
            episode_total_steps = self.episode_length * len(self.envs)

            with tqdm(total=episode_total_steps, desc="Warmup Steps", position=1, dynamic_ncols=True, leave=False) as pbar:
                while not all(env_dones.values()) and steps < episode_total_steps:
                    for key, env in self.envs.items():
                        if env_dones[key]:
                            continue

                        obs_np = np.asarray(env.state, dtype=np.float32).copy()

                        action = self._sample_warmup_action(obs_np)

                        next_obs, rew, done, truncated, info = env.step(action)

                        self.buffer.add(obs_np, action, rew * self.hp.reward_scale, next_obs, done or truncated, stream_id=key)

                        env_dones[key] = done or truncated
                        steps += 1

                        pbar.update(1)

            self._reset_train_envs()


    def _collect_training_episode(self, episode: int) -> tuple[float, int]:
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

                    obs_map = {key: np.asarray(env.state, dtype=np.float32).copy() for key, env in active}
                    obs_batch = np.stack([obs_map[key] for key, _ in active], axis=0)
                    obs_t = torch.as_tensor(obs_batch, device=DEVICE)

                    with torch.inference_mode():
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

                    for (key, obs_np, action_exec), (next_obs, rew, done, truncated, info) in completed:
                        self.buffer.add(obs_np, action_exec, rew * self.hp.reward_scale, next_obs, done or truncated, stream_id=key)

                        reward[key] += rew
                        reward["total"] += rew
                        env_dones[key] = done or truncated
                        steps += 1

                        if (steps % self.log_every_steps == 0) or (steps == episode_total_steps):
                            progress_pct = 100.0 * steps / episode_total_steps
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
                                self.update(episode)
                                updates_in_episode += 1

        self._reset_train_envs()
        return float(reward["total"]), int(steps)



    def _run_eval_and_checkpoint(self, episode: int) -> tuple[float, float, int, int]:
        eval_reward_det = np.nan
        checkpoint_score = np.nan
        if episode % self.hp.eval_every == 0:
            eval_reward_det = float(self.eval(deterministic=True, runs=self.eval_runs_train, eval_desc="Eval Det"))
            stats = dict(self.last_eval_stats)
            checkpoint_score = float(stats.get("mean_reward", eval_reward_det))
            self.eval_rewards.append(eval_reward_det)
            self.eval_count += 1
            improved = self._maybe_save_mean_checkpoint(stats, episode)
            self._maybe_save_robust_checkpoint(stats, episode)
            self._maybe_save_operational_checkpoint(stats, episode)
            if improved:
                self.last_improvement_episode = int(episode)
                self.last_improvement_eval_count = int(self.eval_count)
        no_improve_evals = int(self.eval_count if self.last_improvement_eval_count < 0 else self.eval_count - self.last_improvement_eval_count)
        no_improve_episodes = int((episode + 1) if self.last_improvement_episode < 0 else episode - self.last_improvement_episode)
        self.no_improve_evals = no_improve_evals
        return eval_reward_det, checkpoint_score, no_improve_episodes, no_improve_evals


    def _run_final_full_eval(self, episode: int) -> None:
        if episode < 0:
            return
        self.eval(deterministic=True, runs=self.eval_runs_full, eval_desc="Final Eval Det")
        self.final_full_eval = dict(self.last_eval_stats)
        self._save_final_checkpoint(episode)


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
            "alpha_loss_ep": np.nan,
        }


    def _build_audit_row(self, episode: int, train_total: float, eval_reward_det: float, checkpoint_score: float, metrics: dict, steps: int, iteration_time_sec: float, no_improve_episodes: int, no_improve_evals: int = 0) -> dict:
        alpha_val = float(self.temperature.alpha.detach().cpu())
        eval_stats = self.last_eval_stats if not np.isnan(eval_reward_det) else {}

        return {
            "episode": int(episode),
            "train_reward_total": train_total,
            "eval_reward_det": float(eval_reward_det) if not np.isnan(eval_reward_det) else np.nan,
            "eval_reward": float(eval_reward_det) if not np.isnan(eval_reward_det) else np.nan,
            "eval_reward_worst": float(eval_stats.get("worst_reward", np.nan)),
            "eval_reward_robust": float(eval_stats.get("robust_score", np.nan)),
            "eval_operational_score": float(eval_stats.get("operational_score", np.nan)),
            "checkpoint_score": float(checkpoint_score) if not np.isnan(checkpoint_score) else np.nan,
            "best_train_reward": float(self.best_train_reward),
            "best_eval_reward": float(self.best_eval_reward),
            "best_checkpoint_score": float(self.best_checkpoint_score),
            "best_robust_score": float(self.best_robust_score),
            "best_operational_score": float(self.best_operational_score),
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
            "cost_mean": float(metrics["cost_mean_ep"]) if not np.isnan(metrics["cost_mean_ep"]) else np.nan,
            "cost_p95": float(metrics["cost_p95_ep"]) if not np.isnan(metrics["cost_p95_ep"]) else np.nan,
            "frac_violation": float(metrics["frac_violation_ep"]) if not np.isnan(metrics["frac_violation_ep"]) else np.nan,
            "critic_loss": float(metrics["critic_loss_ep"]) if not np.isnan(metrics["critic_loss_ep"]) else np.nan,
            "actor_loss": float(metrics["actor_loss_ep"]) if not np.isnan(metrics["actor_loss_ep"]) else np.nan,
            "actor_term_entropy": float(metrics["actor_term_entropy_ep"]) if not np.isnan(metrics["actor_term_entropy_ep"]) else np.nan,
            "actor_term_q": float(metrics["actor_term_q_ep"]) if not np.isnan(metrics["actor_term_q_ep"]) else np.nan,
            "alpha_loss": float(metrics["alpha_loss_ep"]) if not np.isnan(metrics["alpha_loss_ep"]) else np.nan,
            "no_improve_episodes": int(no_improve_episodes),
            "no_improve_evals": int(no_improve_evals),
            "iteration_time_sec": float(iteration_time_sec),
        }


    def _flush_audit(self, force: bool = False) -> None:
        if not self._audit_pending_rows:
            return
        if force or len(self._audit_pending_rows) >= self.audit_every_episodes:
            self.audit_df = pd.concat([self.audit_df, pd.DataFrame(self._audit_pending_rows)], ignore_index=True)
            self._audit_pending_rows.clear()
            self.audit_df.to_csv(self.audit_csv, index=False)


    def _update_train_postfix(self, p_outer, train_total: float, eval_reward_det: float, checkpoint_score: float, metrics: dict, no_improve_episodes: int, no_improve_evals: int = 0):
        p_outer.set_postfix({
            "train_total": f"{train_total:.2f}",
            "eval": f"{eval_reward_det:.2f}" if not np.isnan(eval_reward_det) else "-",
            "ckpt": f"{checkpoint_score:.2f}" if not np.isnan(checkpoint_score) else "-",
            "best": f"{self.best_eval_reward:.2f}",
            "alpha": f"{float(self.temperature.alpha.detach().cpu()):.3f}",
            "frac_viol": f"{float(metrics['frac_violation_ep']):.3f}" if not np.isnan(metrics["frac_violation_ep"]) else "nan",
            "no_imp_ep": int(no_improve_episodes),
            "no_imp_ev": int(no_improve_evals),
        })


    def _should_early_stop(self, episode: int, no_improve_evals: int) -> bool:
        return (
            self.early_stop_patience > 0
            and episode >= self.min_episodes_before_early_stop
            and no_improve_evals >= self.early_stop_patience
        )


    def train(self):
        print(f"Training SAC with tariff: {self.tariff}")
        print(f"Using device: {DEVICE}")

        self._run_warmup()

        # Training
        print("Starting training episodes...")
        last_episode = -1
        for episode in (p_outer := tqdm(range(self.hp.train_episodes), desc="Training Episodes", position=0, dynamic_ncols=True)):
            episode_start_time = time.perf_counter()
            last_episode = episode
            q_start = len(self.q1_values)
            train_total, steps = self._collect_training_episode(episode)
            self.train_rewards.append(train_total)
            self.total_episode_rewards.append(train_total)

            if train_total > self.best_train_reward:
                self.best_train_reward = train_total

            eval_reward_det, checkpoint_score, no_improve_episodes, no_improve_evals = self._run_eval_and_checkpoint(episode)
            metrics = self._aggregate_episode_update_metrics(q_start)
            iteration_time_sec = float(time.perf_counter() - episode_start_time)
            row = self._build_audit_row(
                episode=episode,
                train_total=train_total,
                eval_reward_det=eval_reward_det,
                checkpoint_score=checkpoint_score,
                metrics=metrics,
                steps=steps,
                iteration_time_sec=iteration_time_sec,
                no_improve_episodes=no_improve_episodes,
                no_improve_evals=no_improve_evals,
            )

            self._audit_pending_rows.append(row)
            flush_due = ((episode + 1) % self.audit_every_episodes == 0) or ((episode + 1) == self.hp.train_episodes)
            if flush_due:
                self._flush_audit(force=True)

            self._update_train_postfix(p_outer, train_total, eval_reward_det, checkpoint_score, metrics, no_improve_episodes, no_improve_evals)

            if self._should_early_stop(episode, no_improve_evals):
                print(
                    f"Early stopping at episode {episode}: no improvement for {no_improve_evals} evals "
                    f"(patience={self.early_stop_patience}, min_start={self.min_episodes_before_early_stop})."
                )
                self._flush_audit(force=True)
                break

        self._run_final_full_eval(last_episode)
        self._flush_audit(force=True)
        self._close_eval_executor()


def main():
    # for tariff in ["tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat"]:
    for tariff in ["tar_s"]:
        trainer = Train(tariff)
        trainer.train()


if __name__ == "__main__":
    main()







