from __future__ import annotations

import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else range(int(total or 0))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env.environment import SmartHomeEnv  # type: ignore
try:
    from .hp import HyperParameters
    from .model import Critic, TimeDecisionTransformer
    from .replay_buffer import BufferSample, ReplayBuffer
except ImportError:  # pragma: no cover
    from hp import HyperParameters
    from model import Critic, TimeDecisionTransformer
    from replay_buffer import BufferSample, ReplayBuffer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _load_tariff_label(config_path: Path, override: str | None) -> str:
    label = override
    if label is None:
        with open(config_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
        label = config["Grid"]["tariff_column"]
    return str(label).replace(" ", "_")


def make_grad_scaler(amp_enabled: bool, device: torch.device) -> GradScaler:
    try:
        return GradScaler(device_type="cuda", enabled=amp_enabled and device.type == "cuda")
    except TypeError:
        return GradScaler(enabled=amp_enabled and device.type == "cuda")


def autocast_cuda(amp_enabled: bool, device: torch.device):
    try:
        return autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda")
    except TypeError:
        return autocast(enabled=amp_enabled and device.type == "cuda")


def init_window(obs: np.ndarray, seq_len: int) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    return np.tile(obs, (seq_len, 1))


def roll_window(window: np.ndarray, new_obs: np.ndarray) -> np.ndarray:
    return np.concatenate([window[1:], np.asarray(new_obs, dtype=np.float32)[None, :]], axis=0)


RUN_CONFIG_PATH = Path(__file__).with_name("run_config.json")
def load_run_cfg() -> dict:
    if not RUN_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Expected run config JSON at {RUN_CONFIG_PATH}")
    with open(RUN_CONFIG_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)


def expand_runs(cfg: dict) -> list[dict]:
    runs = cfg.get("runs")
    base = {k: v for k, v in cfg.items() if k != "runs"}
    if not runs:
        return [base]
    expanded: list[dict] = []
    for idx, run in enumerate(runs):
        merged = dict(base)
        if run:
            merged.update(run)
        label = merged.get("run_label") or merged.get("name")
        if not label:
            base_label = base.get("run_label", "rl-finetune-tdt-td3")
            label = f"{base_label}-{idx + 1}"
        merged["run_label"] = label
        merged["seed"] = int(merged.get("seed", base.get("seed", 0)))
        expanded.append(merged)
    return expanded


class Actor(nn.Module):
    def __init__(self, hparams: HyperParameters, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.core = TimeDecisionTransformer(hparams)
        self.seq_len = hparams.seq_len
        self.action_dim = hparams.output_dim
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = state_seq.shape
        device = state_seq.device
        action_tokens = torch.zeros(bs, seq_len, self.action_dim, device=device, dtype=state_seq.dtype)
        returns_to_go = torch.zeros(bs, seq_len, device=device, dtype=state_seq.dtype)
        timesteps = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)
        pad_mask = torch.zeros(bs, seq_len, device=device, dtype=torch.bool)
        preds = self.core(state_seq, action_tokens, returns_to_go, timesteps, padding_mask=pad_mask)
        raw = preds[:, -1, :]
        return torch.min(torch.max(raw, self.action_low), self.action_high)


def load_imitation_checkpoint(path: Path, device: torch.device) -> tuple[dict, HyperParameters, dict | None]:
    payload = torch.load(path, map_location=device)
    state_dict = payload["state_dict"]
    if "hparams" not in payload:
        raise KeyError("Checkpoint is missing 'hparams'; please export them with the actor .pt")
    hparams = HyperParameters.from_dict(payload["hparams"])
    state_mask = payload.get("state_mask")
    return state_dict, hparams, state_mask


def pretrain_critics(
    cfg_config: Path,
    datasets: list[str],
    actor: Actor,
    critic: Critic,
    critic_lr: float,
    steps: int,
    batch_size: int,
    gamma: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    buffer_size: int,
    mode: str,
    amp_enabled: bool,
    checkpoint_path: Path,
    tariff_override: str | None,
) -> None:
    if steps <= 0:
        return

    device = next(actor.parameters()).device
    buffer: ReplayBuffer | None = None
    critic_opt = Adam(critic.parameters(), lr=critic_lr)
    critic_scaler = make_grad_scaler(amp_enabled, device)

    actor_target: Actor | None = None
    critic_target: Critic | None = None

    step_counter = 0
    pbar = tqdm(total=steps, desc="critic pretrain")
    ds_idx = 0
    while step_counter < steps:
        data_path = datasets[ds_idx % len(datasets)]
        env = build_env(
            cfg_config,
            Path(data_path),
            "01/01/1900 00:00",
            days=1,
            state_mask=None,
            tariff_override=tariff_override,
        )
        nrows = len(env.sim.dataframe)
        obs, _ = env.reset()
        window = init_window(obs, actor.core.hparams.seq_len)
        if buffer is None:
            buffer = ReplayBuffer(buffer_size, actor.core.hparams.seq_len, env.observation_space.shape[0], env.action_space.shape[0], device)

        if actor_target is None:
            actor_target = Actor(actor.core.hparams, env.action_space.low, env.action_space.high).to(device)
            critic_target = Critic(actor.core.hparams, env.action_space.shape[0], [256, 256]).to(device)
            actor_target.load_state_dict(actor.state_dict())
            critic_target.load_state_dict(critic.state_dict())

        env.sim.set_start_step(0)
        env.sim.current_step = 0
        env.sim.final_step = nrows - 1
        env.bess.reset()
        env.grid.reset()
        env.load.reset()
        env.pv.reset()
        env.ev.reset()

        obs = env._get_obs()
        window = init_window(obs, actor.core.hparams.seq_len)
        for _ in range(nrows):
            if step_counter >= steps:
                break
            actor.eval()
            with torch.no_grad(), autocast_cuda(amp_enabled, device):
                obs_seq_t = torch.as_tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
                action = actor(obs_seq_t).cpu().numpy()[0]
                noise = np.random.normal(0, policy_noise, size=action.shape)
                action = np.clip(action + noise, env.action_space.low, env.action_space.high)

            next_obs, reward, done, _ = env.step(action)
            next_window = roll_window(window, next_obs)
            done_flag = bool(done)
            buffer.add(window, action, reward, next_window, done_flag)
            window = next_window
            obs = next_obs
            step_counter += 1
            if pbar:
                pbar.update(1)

            if buffer.size >= batch_size:
                sample = buffer.sample(batch_size)
                with torch.no_grad(), autocast_cuda(amp_enabled, device):
                    noise_t = (torch.randn_like(sample.actions) * policy_noise).clamp(-noise_clip, noise_clip)
                    next_actions = actor_target(sample.next_states) + noise_t
                    next_actions = next_actions.clamp(
                        torch.as_tensor(env.action_space.low, device=device),
                        torch.as_tensor(env.action_space.high, device=device),
                    )
                    q1_target, q2_target = critic_target(sample.next_states, next_actions)
                    q_target = torch.min(q1_target, q2_target)
                    target = sample.rewards + (1.0 - sample.dones) * gamma * q_target

                critic_opt.zero_grad()
                with autocast_cuda(amp_enabled, device):
                    q1, q2 = critic(sample.states, sample.actions)
                    loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
                critic_scaler.scale(loss).backward()
                critic_scaler.step(critic_opt)
                critic_scaler.update()
                soft_update(critic, critic_target, tau)

            if step_counter % 1000 == 0:
                print(f"[critic pretrain] steps={step_counter}/{steps} buffer={buffer.size}")

        ds_idx += 1

    print(f"Finished critic pretraining: total steps={step_counter}, buffer_size={buffer.size if buffer else 0}")
    if pbar:
        pbar.close()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"critic_state_dict": critic.state_dict(), "hparams": actor.core.hparams.to_dict()}, checkpoint_path)
    print(f"Saved critic pretrain checkpoint to {checkpoint_path}")


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tgt, src in zip(target.parameters(), source.parameters()):
            tgt.data.mul_(1.0 - tau)
            tgt.data.add_(tau * src.data)


def buffer_state_dict(buffer: ReplayBuffer) -> dict:
    return {
        "ptr": buffer.ptr,
        "size": buffer.size,
        "states": buffer.states.cpu(),
        "actions": buffer.actions.cpu(),
        "rewards": buffer.rewards.cpu(),
        "next_states": buffer.next_states.cpu(),
        "dones": buffer.dones.cpu(),
    }


def load_buffer_state(buffer: ReplayBuffer, state: dict, device: torch.device) -> None:
    buffer.ptr = int(state["ptr"])
    buffer.size = int(state["size"])
    buffer.states = state["states"].to(device)
    buffer.actions = state["actions"].to(device)
    buffer.rewards = state["rewards"].to(device)
    buffer.next_states = state["next_states"].to(device)
    buffer.dones = state["dones"].to(device)


def save_training_checkpoint(
    path: Path,
    step: int,
    actor: Actor,
    critic: Critic,
    actor_target: Actor,
    critic_target: Critic,
    actor_opt: Adam,
    critic_opt: Adam,
    actor_scaler: GradScaler,
    critic_scaler: GradScaler,
    buffer: ReplayBuffer,
    rewards_log: list[tuple[int, float]],
    best_actor_state: dict,
    best_avg: float,
    bad_epochs: int,
) -> None:
    payload = {
        "step": int(step),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_target": actor_target.state_dict(),
        "critic_target": critic_target.state_dict(),
        "actor_opt": actor_opt.state_dict(),
        "critic_opt": critic_opt.state_dict(),
        "actor_scaler": actor_scaler.state_dict(),
        "critic_scaler": critic_scaler.state_dict(),
        "buffer": buffer_state_dict(buffer),
        "rewards_log": rewards_log,
        "best_actor_state": best_actor_state,
        "best_avg": best_avg,
        "bad_epochs": bad_epochs,
    }
    torch.save(payload, path)


def load_training_checkpoint(
    path: Path,
    device: torch.device,
    actor: Actor,
    critic: Critic,
    actor_target: Actor,
    critic_target: Critic,
    actor_opt: Adam,
    critic_opt: Adam,
    actor_scaler: GradScaler,
    critic_scaler: GradScaler,
    buffer: ReplayBuffer,
) -> tuple[int, list[tuple[int, float]], dict, float, int]:
    ckpt = torch.load(path, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]))
    critic_target.load_state_dict(ckpt.get("critic_target", ckpt["critic"]))
    actor_opt.load_state_dict(ckpt["actor_opt"])
    critic_opt.load_state_dict(ckpt["critic_opt"])
    actor_scaler.load_state_dict(ckpt.get("actor_scaler", {}))
    critic_scaler.load_state_dict(ckpt.get("critic_scaler", {}))
    if "buffer" in ckpt:
        load_buffer_state(buffer, ckpt["buffer"], device)
    step = int(ckpt.get("step", 1))
    rewards_log = ckpt.get("rewards_log", [])
    best_actor_state = ckpt.get("best_actor_state", deepcopy(actor.state_dict()))
    best_avg = float(ckpt.get("best_avg", -float("inf")))
    bad_epochs = int(ckpt.get("bad_epochs", 0))
    return step, rewards_log, best_actor_state, best_avg, bad_epochs


def train_td3(
    env: SmartHomeEnv,
    actor: Actor,
    critic: Critic,
    actor_lr: float,
    critic_lr: float,
    total_steps: int,
    warmup_steps: int,
    batch_size: int,
    gamma: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    policy_delay: int,
    buffer_size: int,
    seed: int,
    state_mask: list[bool] | None,
    bc_coef: float,
    bc_start_step: int,
    bc_decay_steps: int,
    behavior_actor: Actor | None,
    early_stop_window: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    amp_enabled: bool,
    checkpoint_dir: Path,
    checkpoint_interval: int,
    resume_checkpoint: Path | None,
) -> list[tuple[int, float]]:
    device = next(actor.parameters()).device
    seq_len = actor.core.hparams.seq_len
    buffer = ReplayBuffer(buffer_size, seq_len, env.observation_space.shape[0], env.action_space.shape[0], device)
    actor_target = Actor(actor.core.hparams, env.action_space.low, env.action_space.high).to(device)
    critic_target = Critic(actor.core.hparams, env.action_space.shape[0], [256, 256]).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = Adam(actor.parameters(), lr=actor_lr)
    critic_opt = Adam(critic.parameters(), lr=critic_lr)
    actor_scaler = make_grad_scaler(amp_enabled, device)
    critic_scaler = make_grad_scaler(amp_enabled, device)

    actor.train()
    critic.train()
    actor_target.eval()
    critic_target.eval()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset()
    window = init_window(obs, seq_len)
    episode_reward = 0.0
    episode_len = 0
    rewards_log: list[tuple[int, float]] = []

    best_avg = -float("inf")
    best_actor_state = deepcopy(actor.state_dict())
    bad_epochs = 0
    start_step = 1

    if resume_checkpoint and resume_checkpoint.exists():
        start_step, rewards_log, best_actor_state, best_avg, bad_epochs = load_training_checkpoint(
            resume_checkpoint,
            device,
            actor,
            critic,
            actor_target,
            critic_target,
            actor_opt,
            critic_opt,
            actor_scaler,
            critic_scaler,
            buffer,
        )
        print(f"Resuming from checkpoint {resume_checkpoint} at step {start_step}")

    last_step = start_step - 1

    for step in tqdm(range(start_step, total_steps + 1), total=total_steps, desc="td3 train"):
        if step <= warmup_steps:
            action = env.action_space.sample()
        else:
            was_training = actor.training
            actor.eval()
            with torch.no_grad(), autocast_cuda(amp_enabled, device):
                obs_seq_t = torch.as_tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
                action = actor(obs_seq_t).cpu().numpy()[0]
                noise = np.random.normal(0, policy_noise, size=action.shape)
                action = np.clip(action + noise, env.action_space.low, env.action_space.high)
            if was_training:
                actor.train()
        next_obs, reward, done, info = env.step(action)
        next_window = roll_window(window, next_obs)
        done_flag = bool(done)
        buffer.add(window, action, reward, next_window, done_flag)

        window = next_window
        obs = next_obs
        episode_reward += reward
        episode_len += 1

        if done_flag:
            rewards_log.append((step, episode_reward))
            print(f"Step {step:06d} | episode_reward={episode_reward:.3f} | len={episode_len}")
            obs, _ = env.reset()
            window = init_window(obs, seq_len)
            episode_reward = 0.0
            episode_len = 0

            if len(rewards_log) >= max(1, early_stop_window):
                recent = [r for _, r in rewards_log[-early_stop_window:]]
                mean_recent = float(np.mean(recent))
                if mean_recent > best_avg + early_stop_min_delta:
                    best_avg = mean_recent
                    best_actor_state = deepcopy(actor.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    print(f"Early stopping triggered: mean_recent={mean_recent:.3f}, best={best_avg:.3f}")
                    break

        if buffer.size < batch_size or step <= warmup_steps:
            continue

        sample = buffer.sample(batch_size)
        with torch.no_grad(), autocast_cuda(amp_enabled, device):
            noise = (torch.randn_like(sample.actions) * policy_noise).clamp(-noise_clip, noise_clip)
            next_actions = actor_target(sample.next_states) + noise
            next_actions = next_actions.clamp(
                torch.as_tensor(env.action_space.low, device=device),
                torch.as_tensor(env.action_space.high, device=device),
            )
            q1_target, q2_target = critic_target(sample.next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            target = sample.rewards + (1.0 - sample.dones) * gamma * q_target

        critic_opt.zero_grad()
        with autocast_cuda(amp_enabled, device):
            q1, q2 = critic(sample.states, sample.actions)
            critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        critic_scaler.scale(critic_loss).backward()
        critic_scaler.step(critic_opt)
        critic_scaler.update()

        if step % policy_delay == 0:
            actor_opt.zero_grad()
            with autocast_cuda(amp_enabled, device):
                actor_loss = -critic(sample.states, actor(sample.states))[0].mean()
                if behavior_actor is not None and bc_coef > 0.0 and step >= bc_start_step:
                    if bc_decay_steps > 0:
                        decay_ratio = max(0.0, 1.0 - (step - bc_start_step) / float(bc_decay_steps))
                        bc_weight = bc_coef * decay_ratio
                    else:
                        bc_weight = bc_coef
                    with torch.no_grad():
                        behavior_actions = behavior_actor(sample.states)
                    bc_loss = nn.functional.mse_loss(actor(sample.states), behavior_actions)
                    actor_loss = actor_loss + bc_weight * bc_loss
            actor_scaler.scale(actor_loss).backward()
            actor_scaler.step(actor_opt)
            actor_scaler.update()
            soft_update(actor, actor_target, tau)
            soft_update(critic, critic_target, tau)

        last_step = step

        if checkpoint_interval > 0 and step % checkpoint_interval == 0:
            ckpt_path = checkpoint_dir / f"step_{step}.pt"
            save_training_checkpoint(
                ckpt_path,
                step,
                actor,
                critic,
                actor_target,
                critic_target,
                actor_opt,
                critic_opt,
                actor_scaler,
                critic_scaler,
                buffer,
                rewards_log,
                best_actor_state,
                best_avg,
                bad_epochs,
            )

    if best_actor_state:
        actor.load_state_dict(best_actor_state)

    latest_ckpt = checkpoint_dir / "latest.pt"
    save_training_checkpoint(
        latest_ckpt,
        last_step,
        actor,
        critic,
        actor_target,
        critic_target,
        actor_opt,
        critic_opt,
        actor_scaler,
        critic_scaler,
        buffer,
        rewards_log,
        best_actor_state,
        best_avg,
        bad_epochs,
    )
    return rewards_log


def build_env(
    config_path: Path,
    data_path: Path,
    start_date: str,
    days: int,
    state_mask: list[bool] | None,
    tariff_override: str | None = None,
) -> SmartHomeEnv:
    with open(config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)
    if tariff_override:
        config.setdefault("Grid", {})["tariff_column"] = tariff_override
    import pandas as pd

    df = pd.read_csv(data_path, sep=";")
    return SmartHomeEnv(config, dataframe=df, days=days, start_date=start_date, state_mask=state_mask)


def main() -> None:
    cfg = load_run_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = expand_runs(cfg)
    cfg_config = Path(cfg["config"])
    base_tariff = _load_tariff_label(cfg_config, cfg.get("tariff_override"))
    tariffs_cfg = cfg.get("tariffs")
    tariffs = [str(t).strip() for t in tariffs_cfg] if tariffs_cfg else [base_tariff]

    for tariff in tariffs:
        tariff_label = str(tariff).replace(" ", "_")
        results_base = Path("RESULTS") / tariff_label / "6_TDT_FT_TD3" / "train"
        run_dir = results_base

        checkpoint = Path(cfg.get("checkpoint")) if cfg.get("checkpoint") else Path("Results") / tariff_label / "3_TDT_IL" / "best.pt"
        state_dict, hparams, state_mask_payload = load_imitation_checkpoint(checkpoint, device)
    mask_vector = None
    if isinstance(state_mask_payload, dict) and "vector" in state_mask_payload:
        mask_vector = [bool(x) for x in state_mask_payload.get("vector", [])]

        actor: Actor | None = None
        critic: Critic | None = None
        behavior_actor: Actor | None = None

        for idx, run_cfg in enumerate(runs, start=1):
            print(f"=== Run {idx}/{len(runs)}: {run_cfg['run_label']} | tariff={tariff_label} ===")
            set_seed(int(run_cfg["seed"]))

            cfg_data = Path(run_cfg["data"])
            env = build_env(
                cfg_config,
                cfg_data,
                str(run_cfg["start_date"]),
                int(run_cfg["days"]),
                state_mask=mask_vector,
                tariff_override=tariff,
            )

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        if actor is None:
            if obs_dim != hparams.input_dim:
                hparams.input_dim = obs_dim
            if action_dim != hparams.output_dim:
                hparams.output_dim = action_dim
            actor = Actor(hparams, env.action_space.low, env.action_space.high).to(device)
            missing, unexpected = actor.core.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Warning: missing keys when loading actor: {missing}")
            if unexpected:
                print(f"Warning: unexpected keys when loading actor: {unexpected}")
            behavior_actor = Actor(hparams, env.action_space.low, env.action_space.high).to(device)
            behavior_actor.load_state_dict(actor.state_dict())
            behavior_actor.eval()
            for p in behavior_actor.parameters():
                p.requires_grad = False
            critic = Critic(hparams, action_dim, [256, 256]).to(device)
        else:
            if obs_dim != actor.core.hparams.input_dim:
                actor.core.hparams.input_dim = obs_dim
            if action_dim != actor.core.hparams.output_dim:
                actor.core.hparams.output_dim = action_dim

        if critic is None:
            critic = Critic(hparams, action_dim, [256, 256]).to(device)

        if idx == 1:
            critic_ckpt = cfg.get("critic_checkpoint")
            if critic_ckpt:
                payload = torch.load(Path(critic_ckpt), map_location=device)
                critic_state = payload.get("critic_state_dict", payload)
                critic.load_state_dict(critic_state)
                print(f"Loaded critic checkpoint from {critic_ckpt}")
            elif bool(cfg.get("pretrain_critics", False)) and int(cfg.get("critic_pretrain_steps", 0)) > 0:
                pretrain_datasets = cfg.get("pretrain_datasets") or [cfg.get("data")]
                pretrain_mode = cfg.get("pretrain_mode", "offline")
                pretrain_ckpt = results_base / "pretrain" / "critic_pretrain.pt"
                pretrain_critics(
                    cfg_config=cfg_config,
                    datasets=pretrain_datasets,
                    actor=actor,
                    critic=critic,
                    critic_lr=float(cfg["critic_lr"]),
                    steps=int(cfg.get("critic_pretrain_steps", 0)),
                    batch_size=int(cfg.get("critic_pretrain_batch_size", cfg["batch_size"])),
                    gamma=float(cfg["gamma"]),
                    tau=float(cfg["tau"]),
                    policy_noise=float(cfg["policy_noise"]),
                    noise_clip=float(cfg["noise_clip"]),
                    buffer_size=int(cfg["buffer_size"]),
                    mode=pretrain_mode,
                    amp_enabled=bool(cfg.get("amp_enabled", True)),
                    checkpoint_path=pretrain_ckpt,
                    tariff_override=tariff,
                )

        subdir = run_dir / run_cfg["run_label"]
        subdir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = subdir / "checkpoints"
        resume_checkpoint = Path(run_cfg["resume_checkpoint"]) if run_cfg.get("resume_checkpoint") else None
        latest_ckpt = checkpoint_dir / "latest.pt"
        if resume_checkpoint is None and latest_ckpt.exists():
            resume_checkpoint = latest_ckpt
            print(f"Auto-resume from {resume_checkpoint}")
        with open(subdir / "finetune_config.json", "w", encoding="utf-8") as fp:
            json.dump({
                **run_cfg,
                "config": str(cfg_config),
                "data": str(cfg_data),
                "tariff": tariff,
                "checkpoint": str(checkpoint),
                "state_mask": mask_vector,
            }, fp, indent=2)

        rewards_log = train_td3(
            env=env,
            actor=actor,
            critic=critic,
            actor_lr=float(run_cfg["actor_lr"]),
            critic_lr=float(run_cfg["critic_lr"]),
            total_steps=int(run_cfg["total_steps"]),
            warmup_steps=int(run_cfg["warmup_steps"]),
            batch_size=int(run_cfg["batch_size"]),
            gamma=float(run_cfg["gamma"]),
            tau=float(run_cfg["tau"]),
            policy_noise=float(run_cfg["policy_noise"]),
            noise_clip=float(run_cfg["noise_clip"]),
            policy_delay=int(run_cfg["policy_delay"]),
            buffer_size=int(run_cfg["buffer_size"]),
            seed=int(run_cfg["seed"]),
            state_mask=mask_vector,
            bc_coef=float(run_cfg.get("bc_coef", cfg.get("bc_coef", 0.0))),
            bc_start_step=int(run_cfg.get("bc_start_step", cfg.get("bc_start_step", 0))),
            bc_decay_steps=int(run_cfg.get("bc_decay_steps", cfg.get("bc_decay_steps", 0))),
            behavior_actor=behavior_actor,
            early_stop_window=int(run_cfg.get("early_stop_window", cfg.get("early_stop_window", 6))),
            early_stop_patience=int(run_cfg.get("early_stop_patience", cfg.get("early_stop_patience", 3))),
            early_stop_min_delta=float(run_cfg.get("early_stop_min_delta", cfg.get("early_stop_min_delta", 0.0))),
            amp_enabled=bool(run_cfg.get("amp_enabled", cfg.get("amp_enabled", True))),
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=int(run_cfg.get("checkpoint_interval", cfg.get("checkpoint_interval", 0))),
            resume_checkpoint=resume_checkpoint,
        )

        np.savetxt(subdir / "episode_rewards.csv", np.array(rewards_log), delimiter=",", header="step,reward", comments="")

        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": actor.state_dict(),
                "hparams": actor.core.hparams.to_dict(),
                "state_mask": mask_vector,
            },
            run_dir / "actor_finetuned.pt",
        )


if __name__ == "__main__":
    main()
