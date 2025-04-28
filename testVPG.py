import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
# Hyperparameters
learning_rate = 1e-4
gamma = 0.99
episodes = 1000000

# Simple Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(512, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # Trainable std
        self.std_const = 5e-2  # Constant std for exploration

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_head(x)
        std = self.log_std.exp() + self.std_const  # Ensure std is positive
        return mean, std

def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


if __name__ == "__main__":
    env = gym.make("Pusher-v5", render_mode="human", max_episode_steps=500)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = PolicyNetwork(obs_dim, act_dim)
    policy.load_state_dict(torch.load("VPG/policy_vpg2_276000.pth"))
    policy.std_const = 0
    policy.eval()
    
    for episode in range(10):
        state = env.reset()
        state = state[0]  # Unwrap the tuple
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            mean, std = policy(state_tensor)

            dist = Normal(mean, std)
            action = dist.sample()
            action_clipped = torch.clamp(action, float(env.action_space.low[0]), float(env.action_space.high[0]))

            next_state, reward, terminated, truncated, _ = env.step(action_clipped.detach().numpy())
            done = terminated or truncated

            total_reward += reward
            state = next_state

        print(f"Total Reward in Test: {total_reward:.2f}")
    env.close()