"""Generic episode rollout used by the trainer's collection phase.

For multiprocess eval we already have the corresponding utils eval._run_episode; this one
runs in the main process, uses the live actor, and returns per-step
(obs, action, reward, next_obs, done) so the trainer pushes into the replay
buffer with full control over stream IDs and reward scaling.
"""
from collections import deque
import numpy as np
import torch


def collect_episode(env, actor, history_len, device, deterministic=False, progress=None, safety=None):
    """If `safety` is given, the actor's raw action is projected onto the
    feasible (box ∩ slab) set before being passed to env.step and stored in
    the transition. The buffer then always contains safe actions; critic
    learns Q-values over feasible behaviour only.
    """
    obs, _ = env.reset()
    hist = deque([obs.copy() for _ in range(history_len)], maxlen=history_len)
    transitions = []
    total_reward = 0.0
    done = truncated = False
    while not (done or truncated):
        seq = np.stack(hist, axis=0)
        x = torch.as_tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.inference_mode():
            raw = actor.act(x, deterministic=deterministic)
            if safety is not None:
                raw, _ = safety.project(x, raw)
            a = raw.squeeze(0).cpu().numpy()
        next_obs, reward, done, truncated, _ = env.step(a)
        hist_after = list(hist)[1:] + [next_obs.copy()]
        transitions.append((np.stack(list(hist), axis=0), a, float(reward),
                            np.stack(hist_after, axis=0), done or truncated))
        hist.append(next_obs.copy())
        total_reward += float(reward)
        if progress is not None:
            progress.step(reward)
    return transitions, total_reward


def collect_streams(envs, actor, history_len, device, deterministic=False, progress=None,
                     on_step=None, safety=None):
    """Collect one synchronized episode from each named environment.

    If `on_step` is given, it is called as `on_step(transition, stream_key)`
    after each per-stream env step. Lets the trainer interleave SAC updates
    inside the rollout (v4-style), while keeping the function agnostic of the
    training machinery (replay, optimizers, duals).

    If `safety` is given, the actor's batched raw action is projected onto
    the feasible (box ∩ slab) set before env.step. The transition stored in
    the buffer carries the PROJECTED action, so the critic learns Q on safe
    behavior exclusively (no grid-penalty contamination of the bootstrap).
    """
    hist = {}
    transitions = {key: [] for key in envs}
    rewards = {key: 0.0 for key in envs}
    done = {key: False for key in envs}

    for key, env in envs.items():
        obs, _ = env.reset()
        hist[key] = deque([obs.copy() for _ in range(history_len)], maxlen=history_len)

    while not all(done.values()):
        active = [key for key in envs if not done[key]]
        seqs = [np.stack(hist[key], axis=0) for key in active]
        x = torch.as_tensor(np.stack(seqs, axis=0), dtype=torch.float32, device=device)
        with torch.inference_mode():
            raw = actor.act(x, deterministic=deterministic)
            if safety is not None:
                raw, _ = safety.project(x, raw)
            acts = raw.cpu().numpy()

        for key, seq, act in zip(active, seqs, acts):
            next_obs, reward, terminated, truncated, _ = envs[key].step(act)
            hist_after = list(hist[key])[1:] + [next_obs.copy()]
            transition = (seq, act, float(reward), np.stack(hist_after, axis=0), terminated or truncated)
            transitions[key].append(transition)
            hist[key].append(next_obs.copy())
            rewards[key] += float(reward)
            done[key] = terminated or truncated
            if progress is not None:
                progress.step(key, reward)
            if on_step is not None:
                on_step(transition, key)

    return transitions, rewards
