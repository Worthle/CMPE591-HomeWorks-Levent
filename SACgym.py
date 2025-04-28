import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from torch.utils.data import Dataset,DataLoader
import torch.distributions as distributions
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback








# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        #print(f"Filling Buffer {len(self.buffer)}/100 ")

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Actor (Policy) Network
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, act_dim)
        self.fc_logstd = nn.Linear(hidden_dim, act_dim)
        

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logstd = self.fc_logstd(x)
        std = torch.exp(logstd)
        return mu, std

# Critic Network (Q-value)
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# SAC Agent
class SACAgent:
    def __init__(self, obs_dim, act_dim):
        self.actor = Actor(obs_dim, act_dim)
        self.actor_target = Actor(obs_dim, act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(obs_dim, act_dim)
        self.critic1_target = Critic(obs_dim, act_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(obs_dim, act_dim)
        self.critic2_target = Critic(obs_dim, act_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
       
        self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=20000, eta_min=1e-5)
        self.critic1_scheduler = CosineAnnealingLR(self.critic1_optimizer, T_max=20000, eta_min=1e-5)
        self.critic2_scheduler = CosineAnnealingLR(self.critic2_optimizer, T_max=20000, eta_min=1e-5)
        #self.alpha_scheduler = CosineAnnealingLR(self.alpha_optimizer, T_max=20000, eta_min=1e-5)

        
        self.alpha = alpha  # Entropy coefficient
        self.target_entropy = -np.prod(act_dim).item()  # Target entropy for policy
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer()
        self.std_const = 5e-2
    def save_model(self,idx,n):
        torch.save(self.actor.state_dict(), f"SAC/SACactor_model{n}_{idx}.pt")
        torch.save(self.critic1.state_dict(), f"SAC/SACcritic1_model{n}_{idx}.pt")
        torch.save(self.critic2.state_dict(), f"SAC/SACcritic2_model{n}_{idx}.pt")


    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        #state = torch.from_numpy(state).type(torch.FloatTensor)
        mu, std = self.actor(state)
        std = F.softplus(std) + self.std_const
        dist = distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # For multi-dimensional actions
        return action, log_prob

    def update(self):
        if self.replay_buffer.size() < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)

        # Update Critic
        with torch.no_grad():
            next_mu, next_std = self.actor_target(next_states)
            next_dist = distributions.Normal(next_mu, next_std)
            next_action = next_dist.sample()
            next_log_prob = next_dist.log_prob(next_action).sum(dim=-1)
            target_q = rewards + gamma * (1 - dones) * (torch.min(self.critic1_target(next_states, next_action), self.critic2_target(next_states, next_action)) - self.alpha * min(next_log_prob))

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update Actor
        mu, std = self.actor(states)
        dist = distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        q1 = self.critic1(states, action)
        q2 = self.critic2(states, action)
        actor_loss = (self.alpha * log_prob - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Alpha (Temperature)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update Target Networks (Soft Update)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)



def mov_avg(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    pad = window_size // 2
    data_padded = np.pad(data, pad_width=pad,mode='edge')
    return np.convolve(data_padded,window,'valid')



# Hyperparameters
gamma = 0.9  # Discount factor
alpha = 0.2  # Entropy coefficient (temperature)
tau = 0.005  # Target network update rate
lr = 1e-4  # Learning rate
batch_size = 128
num_episodes = 100000


# Environment Interaction Loop
if __name__ == "__main__":
    # Initialize environment and agent
    env = gym.make("Pusher-v5", max_episode_steps=200)
    #vec_env = make_vec_env(lambda:env, n_envs=1)
    agent = SACAgent(obs_dim=23, act_dim=7)

    episode_rewards = []
    RPS = []

    #prefilling

    for episode in range(num_episodes):
        env.reset()
        state = env.observation_space.sample()
        cumulative_reward = 0.0
        done = False
        episode_steps = 0
        
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, is_terminal, is_truncated, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            cumulative_reward += reward
            state = next_state
            done = is_terminal or is_truncated
            episode_steps += 1
        agent.update()
        episode_rewards.append(cumulative_reward)
        RPS.append(cumulative_reward/episode_steps)
        if episode % 100 == 0:
            print(f"Episode={episode+1}/{num_episodes}, reward={cumulative_reward:.4f}, RPS={cumulative_reward/episode_steps:.4f}")
        if episode % 1000 == 0:
            agent.save_model(episode,7*num_episodes+9030)
            np.save(f"SAC/rewardsSAC{7*num_episodes+9030}_{episode}.npy", np.array(episode_rewards))
            np.save(f"SAC/RPSSAC{7*num_episodes+9030}_{episode}.npy", np.array(RPS))

        if episode % 1000 == 0:
            plt.close()
            plt.subplot(1,2,1)
            plt.plot(episode_rewards, '-', color='gray', linewidth=0.1)
            plt.plot(mov_avg(episode_rewards, min(episode+1,500)), 'r', linewidth=1)
            plt.title("Rewards")
            plt.subplot(1,2,2)
            plt.plot(RPS, '-', color='gray', linewidth=0.1)
            plt.plot(mov_avg(RPS, min(episode+1,500)), 'r', linewidth=1)
            plt.title("Rewards per Episode Steps")
            plt.draw()
            plt.pause(1)
    
    

    agent.save_model(1000,7*num_episodes+9010)
    np.save(f"SAC/rewardsSAC{7*num_episodes+9010}_{1000}.npy", np.array(episode_rewards))
    np.save(f"SAC/RPSSAC{7*num_episodes+9010}_{1000}.npy", np.array(RPS))