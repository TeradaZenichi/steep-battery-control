import json
import torch
import torch.nn as nn


class Actor(nn.Module):    
    def __init__(self, input_dim, hidden_dims, head_dim):
        super().__init__()
        
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU()]
            prev = dim
        
        self.backbone   = nn.Sequential(*layers)
        self.bess_head  = nn.Sequential(nn.Linear(prev, head_dim), nn.ReLU(), nn.Linear(head_dim, 1), nn.Tanh())
        self.ev_head    = nn.Sequential(nn.Linear(prev, head_dim), nn.ReLU(), nn.Linear(head_dim, 1), nn.Tanh())
        self.pv_head    = nn.Sequential(nn.Linear(prev, head_dim), nn.ReLU(), nn.Linear(head_dim, 1), nn.Sigmoid())
    
    def forward(self, x):
        h = self.backbone(x)
        return torch.cat([self.bess_head(h), self.ev_head(h), self.pv_head(h)], dim=-1)
    
    def predict(self, obs):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0) if obs.ndim == 1 else torch.FloatTensor(obs)
            return self.forward(x).numpy().squeeze()


class Critic(nn.Module):    
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        
        layers = []
        prev = state_dim + action_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU()]
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class DoubleCritic(nn.Module):    
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        self.q1 = Critic(state_dim, action_dim, hidden_dims)
        self.q2 = Critic(state_dim, action_dim, hidden_dims)
    
    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)
    
    def q_min(self, state, action):
        return torch.min(*self.forward(state, action))


def load_actor(config, weights_path=None):
    model = Actor(config["input_dim"], config["hidden_dims"], config["head_dim"])
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model


def load_critic(config, weights_path=None):
    model = DoubleCritic(config["state_dim"], config["action_dim"], config["hidden_dims"])
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model