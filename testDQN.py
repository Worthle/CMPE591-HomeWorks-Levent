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
from homework2 import Hw2Env

import environment

#os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, hidden_activation=F.relu, output_activation=F.relu,hidden_size=500):
        super(DQN, self).__init__()
        self.hidden = nn.Linear(6, hidden_size)
        self.output = nn.Linear(hidden_size, 8)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def forward(self, x):
        x = self.hidden_activation(self.hidden(x))
        x = self.output(x)
        if self.output_activation:
        	x = self.output_activation(x)
        return x

def select_action(state, epsilon, dqn, N_ACTIONS):
    if random.random() < epsilon:
        return np.random.randint(N_ACTIONS)
    else:
        with torch.no_grad():
            
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            return torch.argmax(dqn(state)).item()


episode_rewards = []

per_episode_rewards = []
# dont forget i set the max steps to 100.
num_episodes = 10

epsilon = 0.0


if __name__ == "__main__":
    
    N_ACTIONS = 8
    
    env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")

    dqn_test = DQN(hidden_activation=F.relu,output_activation=None,hidden_size=500)
    dqn_test.load_state_dict(torch.load("DQN_model.pth"))
    dqn_test.eval()

    

    for episode in range(num_episodes):
        env.reset()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        while not done:
            state = env.high_level_state()
            action = select_action(state, epsilon, dqn_test, N_ACTIONS)
            next_state, reward, is_terminal, is_truncated = env.step(action)
            state = next_state
            cumulative_reward += reward
            done = is_terminal or is_truncated
        episode_steps += 1
        episode_rewards.append(cumulative_reward)
        per_episode_rewards.append(cumulative_reward / episode_steps)
        
        if episode % 2 == 0:
            print(f"Episode={episode}: Reward={cumulative_reward:.4f}, RPS={cumulative_reward/episode_steps:.4f}, Epsilon={epsilon:.2f}")

    plt.subplot(1,2,1)
    plt.plot(episode_rewards)
    plt.title("Rewards")
    plt.subplot(1,2,2)
    plt.plot(per_episode_rewards)
    plt.title("Rewards per Episode Steps")


    plt.show()
