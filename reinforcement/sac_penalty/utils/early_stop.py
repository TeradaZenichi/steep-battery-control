"""Stop training when the best metric stops improving for `patience` evals."""


class EarlyStop:
    def __init__(self, patience=70, min_episodes=180):
        self.patience = int(patience)
        self.min_episodes = int(min_episodes)
        self.best = -float("inf")
        self.last_best_ep = 0

    def update(self, episode, metric):
        """Returns True if metric improved (and resets the counter)."""
        if metric > self.best:
            self.best = metric
            self.last_best_ep = int(episode)
            return True
        return False

    def should_stop(self, episode):
        if episode < self.min_episodes:
            return False
        return (episode - self.last_best_ep) >= self.patience
