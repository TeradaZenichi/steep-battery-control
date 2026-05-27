"""Collect rule-based demonstrations for RLPD-style prior data.

Runs RB and RBS in closed-loop over many training episodes, saving raw
transitions per episode to .npz files under paper/demos/<source>/<tariff>/.

Output layout:
    paper/demos/RB/tar_sw/cy_00.npz
    paper/demos/RB/tar_sw/cy_01.npz
    ...
    paper/demos/RBS/tar_sw/wy_47.npz

Each .npz contains keys obs, act, rew, next_obs, done sliced per timestep
(shape [T, obs_dim], [T, act_dim], [T], [T, obs_dim], [T] respectively).

Reuses EpisodeGen + collect_episode + the rule-based actor implementations,
so the rollouts are identical (modulo deterministic vs stochastic) to what the
trainer would see if it used the rule-based policy as its actor.

Run:
    python scripts/collect_demos.py            # default: 24 eps × 2 streams × 5 tariffs × 2 sources
    python scripts/collect_demos.py --n_eps 48
    python scripts/collect_demos.py --tariffs tar_s,tar_sw --sources RBS
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environment import SmartHomeEnv
from reinforcement.sac_cmdp.utils import EpisodeGen, collect_episode, save_episode

DEFAULT_TARIFFS = ["tar_s", "tar_w", "tar_sw", "tar_flat", "tar_tou"]
DEFAULT_SOURCES = ["RB", "RBS"]
STREAMS = ["cy", "wy"]
DAYS_PER_EPISODE = 7
HISTORY_LEN = 1  # rule-based actors only look at the last obs


def _load_actor(source: str, device):
    if source == "RB":
        from baselines.RB.model import load_actor
    elif source == "RBS":
        from baselines.RBS.model import load_actor
    else:
        raise ValueError(f"Unknown source: {source}")
    return load_actor({}, device=device)


def _stack_transitions(transitions):
    """`collect_episode` returns (hist_seq, act, reward, next_seq, done) per step.
    Reduce hist_seq/next_seq to their last frame for compact storage; the
    ReplayBuffer reconstructs history at sample time using episode_ids."""
    obs = np.stack([t[0][-1] if t[0].ndim == 2 else t[0] for t in transitions], axis=0).astype(np.float32)
    act = np.stack([t[1] for t in transitions], axis=0).astype(np.float32)
    rew = np.array([t[2] for t in transitions], dtype=np.float32)
    next_obs = np.stack([t[3][-1] if t[3].ndim == 2 else t[3] for t in transitions], axis=0).astype(np.float32)
    done = np.array([t[4] for t in transitions], dtype=np.bool_)
    return obs, act, rew, next_obs, done


def collect(source: str, tariff: str, n_eps: int, seed: int, device, env_params, episodes, out_root: Path, overwrite: bool):
    actor = _load_actor(source, device)
    tariff_dir = out_root / source / tariff
    tariff_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_per_stream = max(1, n_eps // len(STREAMS))
    pbar = tqdm(total=n_per_stream * len(STREAMS), desc=f"{source}/{tariff}", leave=False)
    total_transitions = 0
    for stream in STREAMS:
        df = episodes.load(stream)
        for i in range(n_per_stream):
            out_path = tariff_dir / f"{stream}_{i:02d}.npz"
            if out_path.exists() and not overwrite:
                # Quick count for the summary
                with np.load(out_path) as data:
                    total_transitions += int(data["rew"].shape[0])
                pbar.update(1)
                continue
            start = episodes.sample(stream)
            soc0 = float(rng.uniform(0.1, 0.9))
            env = SmartHomeEnv(df, env_params, start=start, days=DAYS_PER_EPISODE,
                                BESS_SoC=soc0, tariff=tariff, track_operation=False)
            transitions, _ = collect_episode(env, actor, HISTORY_LEN, device, deterministic=True)
            # Ensure last transition has done=True so the buffer flushes its n-step queue.
            if transitions and not transitions[-1][-1]:
                last = list(transitions[-1])
                last[-1] = True
                transitions[-1] = tuple(last)
            obs, act, rew, next_obs, done = _stack_transitions(transitions)
            save_episode(out_path, obs, act, rew, next_obs, done)
            total_transitions += len(transitions)
            pbar.update(1)
    pbar.close()
    return total_transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eps", type=int, default=24,
                        help="Episodes per (source, tariff). Split across cy/wy streams.")
    parser.add_argument("--tariffs", type=str, default=",".join(DEFAULT_TARIFFS))
    parser.add_argument("--sources", type=str, default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out_root", type=str, default="paper/demos")
    args = parser.parse_args()

    tariffs = [t.strip() for t in args.tariffs.split(",") if t.strip()]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    out_root = PROJECT_ROOT / args.out_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8") as f:
        env_params = json.load(f)

    # The validation config for episode windows can come from any sac_cmdp arch
    # config (they share the same val schedule). Use GRU as the reference.
    with open(PROJECT_ROOT / "reinforcement" / "sac_cmdp" / "GRU" / "config.json", encoding="utf-8") as f:
        ref_cfg = json.load(f)

    summary = {}
    for tariff in tariffs:
        episodes = EpisodeGen(ref_cfg, str(PROJECT_ROOT / "data"), seed=args.seed)
        for source in sources:
            n_tr = collect(source, tariff, args.n_eps, args.seed,
                           device, env_params, episodes, out_root, args.overwrite)
            key = f"{source}/{tariff}"
            summary[key] = n_tr
            print(f"  {key:24s}  {n_tr:>8d} transitions")

    print("\nSummary:")
    grand_total = sum(summary.values())
    print(f"  total transitions: {grand_total}")
    print(f"  output root:       {out_root}")


if __name__ == "__main__":
    main()
