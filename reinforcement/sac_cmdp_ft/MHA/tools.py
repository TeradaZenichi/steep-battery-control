"""Small presentation helpers for MHA 2-RL training."""
import math

from tqdm import tqdm


def episode_bars(total: int, warmup: int, tariff: str):
    warmup = min(max(0, int(warmup)), int(total))
    if warmup:
        with tqdm(
            range(warmup),
            desc=f"{tariff} warmup",
            position=0,
            dynamic_ncols=True,
        ) as pbar:
            for episode in pbar:
                yield episode, pbar

    with tqdm(
        range(warmup, total),
        desc=f"{tariff} training",
        position=0,
        dynamic_ncols=True,
    ) as pbar:
        for episode in pbar:
            yield episode, pbar


class EpisodeBar:
    def __init__(self, envs, episode: int, total_episodes: int, refresh_every: int = 64):
        self.rewards = {key: 0.0 for key in envs}
        self.refresh_every = max(1, int(refresh_every))
        self._counter = 0
        self.pbar = tqdm(
            total=sum(int(env.sim.end_idx - env.sim.t_idx) for env in envs.values()),
            desc=f"ep {episode + 1}/{total_episodes}",
            position=1,
            dynamic_ncols=True,
            leave=False,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._flush()
        self.pbar.close()

    def step(self, stream, reward: float | None = None) -> None:
        if reward is None:
            reward = float(stream)
            stream = next(iter(self.rewards))
        self.rewards[stream] += float(reward)
        self._counter += 1
        if self._counter % self.refresh_every == 0:
            self._flush()

    def _flush(self) -> None:
        n = self._counter - self.pbar.n
        if n > 0:
            self.pbar.update(n)
        total = sum(self.rewards.values())
        postfix = {"reward": f"{total:.2f}"}
        postfix.update({key: f"{value:.2f}" for key, value in self.rewards.items()})
        self.pbar.set_postfix(postfix)


class UpdateBar:
    """Progress bar for the SAC update loop within an episode."""

    def __init__(self, total: int, episode: int, refresh_every: int = 16):
        self.refresh_every = max(1, int(refresh_every))
        self._counter = 0
        self._last_row = {}
        self.pbar = tqdm(
            total=int(total),
            desc=f"ep {episode + 1} updates",
            position=2,
            dynamic_ncols=True,
            leave=False,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._flush()
        self.pbar.close()

    def step(self, row) -> None:
        self._counter += 1
        self._last_row = row
        if self._counter % self.refresh_every == 0:
            self._flush()

    def _flush(self) -> None:
        n = self._counter - self.pbar.n
        if n > 0:
            self.pbar.update(n)
        if self._last_row:
            self.pbar.set_postfix(
                {
                    "critic": _fmt(self._last_row.get("critic_loss")),
                    "rcrit": _fmt(self._last_row.get("reward_critic_loss")),
                    "actor": _fmt(self._last_row.get("actor_loss")),
                    "alpha": _fmt(self._last_row.get("alpha")),
                    "ev_lam": _fmt(self._last_row.get("ev_lambda_value")),
                }
            )


def update_train_postfix(pbar, train_reward, eval_summary, best_eval, agg, pending_evals=0) -> None:
    eval_reward = float("nan") if eval_summary is None else float(eval_summary["mean_reward"])
    pbar.set_postfix(
        {
            "train": f"{train_reward:.2f}",
            "eval": f"{eval_reward:.2f}" if math.isfinite(eval_reward) else "-",
            "best": f"{best_eval:.2f}" if math.isfinite(best_eval) else "-",
            "alpha": _fmt(agg.get("alpha")),
            "ev_lam": _fmt(agg.get("ev_lambda_value")),
            "ev_q": _fmt(agg.get("ev_q_cost_pi_mean")),
            "pending_eval": int(pending_evals),
        }
    )


def _fmt(value) -> str:
    if value is None:
        return "nan"
    value = float(value)
    return f"{value:.3f}" if math.isfinite(value) else "nan"
