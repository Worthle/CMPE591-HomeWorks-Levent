import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from collections import deque
from homework3 import Hw3Env

import environment


class VPG(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hl=[16, 32, 32, 16]) -> None:#old 256 512 256
        super(VPG, self).__init__()
        layers = []
        layers.append(nn.Linear(obs_dim, hl[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hl)):
            layers.append(nn.Linear(hl[i-1], hl[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hl[-1], act_dim*2))  # act_dim * (1 for mean + 1 for std)
        self.model = nn.Sequential(*layers)
        self.rewards_history = []
        self.epi_reset()
    def forward(self, x):
        return self.model(x)
    def epi_reset(self):
        self.episode_actions = torch.Tensor([])
        self.episode_rewards = []



class Agent():
    def __init__(self):
        # edit as needed
        self.model = VPG()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3) #1e-4 old
        self.rewards = []

    def save_model(self,idx,n):
        torch.save(self.model.state_dict(), f"VPG/VPG_model{n}_{idx}.pt")


    def decide_action(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_mu, action_std = self.model(state).chunk(2, dim=-1)
        action_std = F.softplus(action_std) + 5e-2
        dist = torch.distributions.Normal(action_mu, action_std)
        action = dist.sample()
        #action = torch.tanh(action)
        self.model.episode_actions = torch.cat([self.model.episode_actions, dist.log_prob(action)])
        return action
    

    def update_model(self):
        R = 0
        rewards = []
        loss = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            rewards.insert(0,R)

        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        for episode_action, G in zip(self.model.episode_actions, rewards):
            loss.append(-episode_action * G)

        #loss = (torch.sum(torch.mul(self.model.episode_actions, rewards).mul(-1), -1))
        self.optimizer.zero_grad()
        torch.stack(loss).sum().backward()
        self.optimizer.step()
        self.model.rewards_history.append(np.sum(self.model.episode_rewards))
        self.model.epi_reset()

    def add_reward(self, reward):
        self.rewards.append(reward)

def mov_avg(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    pad = window_size // 2
    data_padded = np.pad(data, pad_width=pad,mode='edge')
    return np.convolve(data_padded,window,'valid')

episode_rewards = []
gamma=0.99

if __name__ == "__main__":
    env = Hw3Env(render_mode="offscreen")
    agent = Agent() 
    
    num_episodes = 1000
    rewards = []
    RPS = []
    for episode in range(num_episodes):
        env.reset()
        state = env.high_level_state()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        while not done:
            #print(state)
            action = agent.decide_action(state)
            
            next_state, reward, is_terminal, is_truncated = env.step(action)
            agent.add_reward(reward)
            cumulative_reward += reward
            done = is_terminal or is_truncated
            state = next_state
            episode_steps += 1
        
        if cumulative_reward > 0:
            print(f"Episode={episode+1}/{num_episodes}, reward={cumulative_reward:.4f}, RPS={cumulative_reward/episode_steps:.4f} - High Reward")
        else:
            print(f"Episode={episode+1}/{num_episodes}, reward={cumulative_reward:.4f}, RPS={cumulative_reward/episode_steps:.4f}")
        rewards.append(cumulative_reward)
        RPS.append(cumulative_reward/episode_steps)

        if episode % 10 == 0:
            agent.save_model(episode,num_episodes+9020)
            np.save(f"VPG/rewardsVPG{num_episodes+9020}_{episode}.npy", np.array(rewards))
            np.save(f"VPG/RPSVPG{num_episodes+9020}_{episode}.npy", np.array(RPS))
        if episode % 10 == 0:
            plt.close()
            plt.subplot(1,2,1)
            plt.plot(rewards, '-', color='gray', linewidth=0.1)
            plt.plot(mov_avg(rewards, min(episode+1,100)), 'r', linewidth=1)
            plt.title("Rewards")
            plt.subplot(1,2,2)
            plt.plot(RPS, '-', color='gray', linewidth=0.1)
            plt.plot(mov_avg(RPS, min(episode+1,100)), 'r', linewidth=1)
            plt.title("Rewards per Episode Steps")
            plt.draw()
            plt.pause(1)
        

        agent.update_model()


    agent.save_model(1000,num_episodes+9010)
    np.save(f"VPG/rewardsVPG{num_episodes+9010}_{1000}.npy", np.array(rewards))
    np.save(f"VPG/RPSVPG{num_episodes+9010}_{1000}.npy", np.array(RPS))