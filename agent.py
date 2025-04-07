import torch
from torch import optim

from model import VPG
import torch.nn.functional as F
from torch.distributions import Normal

gamma = 0.99


class Agent():
    def __init__(self):
        # edit as needed
        self.model = VPG()
        self.rewards = []
        self.log_probabilities=[]
        
    def decide_action(self, state):
        # edit as needed
        state = torch.tensor(state, dtype=torch.float32)
        action_mean, act_std = self.model(state).chunk(2, dim=-1)
        action_std = F.softplus(act_std) + 5e-2  # increase variance to stimulate exploration
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        self.log_probabilities.append(dist.log_probabilities(action).sum())
        #action = torch.tanh(action)
        return action.detach().numpy()


    def update_model(self, state):
        G = 0
        policy_loss = []
        returns = []
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0,G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        for log_prob, G in zip(self.log_probabilities, returns):
            policy_loss.append(-log_prob * G)
        

    def add_reward(self, reward):
        self.rewards.append(reward)
        
