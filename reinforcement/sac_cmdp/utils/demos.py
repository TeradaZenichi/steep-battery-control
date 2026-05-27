"""Prior-data demonstrations for RLPD-style training.

Layout on disk:
    paper/demos/<source>/<tariff>/<stream>_<run>.npz
        with keys: obs, act, rew, next_obs, done

Where:
    source  in {RB, RBS, ...}
    tariff  in {tar_s, tar_w, tar_sw, tar_flat, tar_tou}
    stream  in {cy, wy}
    run     zero-padded index, e.g. 00, 01, ...

The buffer loaded here uses the SAME ReplayBuffer class as the online buffer,
so n-step aggregation, history slicing, and cost annotation behave identically.
Costs are recomputed at load time using the trainer's cost configuration, so
the demo file format itself is config-agnostic.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .replay import ReplayBuffer


def save_episode(path, obs, act, rew, next_obs, done):
    """Save a single episode of transitions to a compressed .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=np.asarray(obs, dtype=np.float32),
        act=np.asarray(act, dtype=np.float32),
        rew=np.asarray(rew, dtype=np.float32),
        next_obs=np.asarray(next_obs, dtype=np.float32),
        done=np.asarray(done, dtype=np.bool_),
    )


def iter_demo_files(demos_root, sources, tariff):
    """Yield (source, stream, npz_path) for every demo episode matching the
    given sources and tariff. Streams are inferred from filename prefix."""
    root = Path(demos_root)
    for src in sources:
        tdir = root / src / tariff
        if not tdir.is_dir():
            continue
        for npz in sorted(tdir.glob("*.npz")):
            name = npz.stem
            stream = "cy" if name.startswith("cy") else "wy" if name.startswith("wy") else "any"
            yield src, stream, npz


def load_prior_buffer(
    demos_root,
    sources,
    tariff,
    obs_dim,
    act_dim,
    device,
    history_len,
    n_step,
    gamma,
    reward_scale,
    cost_names=(),
    cost_compute_fn=None,
    capacity=None,
):
    """Build a ReplayBuffer populated from demo .npz files.

    `cost_compute_fn(obs, act) -> dict[name, float]` recomputes costs from
    the current trainer's CMDP config. If None, costs default to 0.

    `capacity` defaults to total transitions counted across the npz files.
    Use a stream_id per (source, stream) so the buffer's internal shards
    sample proportionally to source × stream coverage.
    """
    files = list(iter_demo_files(demos_root, sources, tariff))
    if not files:
        raise FileNotFoundError(
            f"No demo files for tariff={tariff} under {demos_root} "
            f"(sources={list(sources)})"
        )

    # Estimate capacity: number of transitions across all files. Round up so
    # the internal per-shard cap (capacity // 2) does not truncate any source.
    if capacity is None:
        total = 0
        for _, _, npz in files:
            with np.load(npz) as data:
                total += int(data["rew"].shape[0])
        capacity = max(2, total * 2 + 2)  # buffer halves capacity per shard

    buf = ReplayBuffer(
        capacity=capacity,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        history_len=history_len,
        n_step=n_step,
        gamma=gamma,
        cost_names=cost_names,
    )

    for src, stream, npz in files:
        with np.load(npz) as data:
            obs = data["obs"]
            act = data["act"]
            rew = data["rew"]
            next_obs = data["next_obs"]
            done = data["done"]
        sid = f"{src}::{stream}"
        T = rew.shape[0]
        for t in range(T):
            costs = None
            if cost_names and cost_compute_fn is not None:
                costs = cost_compute_fn(obs[t], act[t])
            buf.add(
                obs=obs[t],
                act=act[t],
                rew=float(rew[t]) * float(reward_scale),
                next_obs=next_obs[t],
                done=bool(done[t]),
                stream_id=sid,
                costs=costs,
            )
    return buf


def merge_batches(b_online, b_prior, device=None):
    """Concatenate two batch dicts along dim 0 in matching key order.

    Both dicts must have the same keys (which is guaranteed when both are
    sampled from a ReplayBuffer constructed with the same cost_names).
    """
    import torch
    if not b_online and not b_prior:
        raise ValueError("Both batches empty.")
    if not b_online:
        return b_prior
    if not b_prior:
        return b_online
    out = {}
    for k in b_online:
        out[k] = torch.cat([b_online[k], b_prior[k]], dim=0)
    return out
