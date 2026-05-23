"""Vanilla SAC hyperparameters: auto-alpha + 1 grid Lagrangian dual. No solar/EV duals."""


class Hyperparameters:
    def __init__(self, cfg):
        self.seed = int(cfg.get("seed", 42))
        self.days = int(cfg["days"])
        self.gamma = float(cfg["gamma"])
        self.tau = float(cfg["tau"])
        self.batch_size = int(cfg["batch_size"])
        self.history_len = int(cfg.get("history_len", 1))
        self.n_step = int(cfg.get("n_step", 1))
        self.buffer_size = int(cfg["buffer_size"])
        self.warmup_episodes = int(cfg["warmup_episodes"])
        self.train_episodes = int(cfg["train_episodes"])
        self.evaluate_every = int(cfg["evaluate_every"])
        self.update_every_steps = int(cfg.get("update_every_steps", 1))
        self.actor_update_every = int(cfg.get("actor_update_every", 1))
        self.actor_update_start_episode = int(cfg.get("actor_update_start_episode", 0))
        self.reward_scale = float(cfg.get("reward_scale", 1.0))
        self.grad_clip = bool(cfg.get("grad_clip", True))

        self.actor_lr = float(cfg.get("actor_lr", 1e-4))
        self.critic_lr = float(cfg.get("critic_lr", 1e-4))
        self.alpha_lr = float(cfg.get("alpha_lr", 3e-4))
        self.init_alpha = float(cfg.get("init_alpha", 0.1))
        self.target_entropy = float(cfg.get("target_entropy", -1.0))
        self.log_std_min = float(cfg.get("log_std_min", -10.0))
        self.log_std_max = float(cfg.get("log_std_max", 2.0))

        self.lambda_lr = float(cfg.get("lambda_lr", 3e-4))
        self.init_lambda = float(cfg.get("init_lambda", 0.1))
        self.violation_budget = float(cfg.get("violation_budget", 0.0))

        self.early_stop_patience = int(cfg.get("early_stop_patience", 0))
        self.min_episodes_before_early_stop = int(cfg.get("min_episodes_before_early_stop", 0))
