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

epsilon_decay_plot = []

# dont forget i set the max steps to 100.
learning_rate = 0.0001
num_episodes = 100
update_frequency = 10
target_update_frequency = 200
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_decay_high_reward = 0.8
epsilon_increase = 1.001
epsilon_min = 0.05
epsilon_max = 1.0
batch_size=64
gamma=0.9
buffer_size=1000
reward_threshold = 10.0
high_reward_threshold = 50.0

if __name__ == "__main__":
    
    N_ACTIONS = 8
    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN(hidden_activation=F.relu,output_activation=None,hidden_size=500)
    #dqn.to(device)
    target_dqn = DQN(hidden_activation=F.relu,output_activation=None,hidden_size=500)
    #target_dqn.to(device)
    target_dqn.load_state_dict(dqn.state_dict())

    #optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    optimizer = optim.RMSprop(dqn.parameters(), lr=learning_rate, alpha=0.95, eps=1e-8)
    #loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(buffer_size)





    for episode in range(num_episodes):
        env.reset()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        
        if len(replay_buffer) < 100:
            print("Prefilling")
            while len(replay_buffer) < 100:
                state = env.high_level_state()
                action = select_action(state, epsilon, dqn, N_ACTIONS)
                next_state, reward, is_terminal, is_truncated = env.step(action)
                done = is_terminal or is_truncated
                cumulative_reward += reward
                replay_buffer.add(state, action, reward, next_state, done)
                if done:
                    env.reset()
                print(f"Buffer: ({len(replay_buffer)}/1000)")
            print("Finished prefilling")
        while not done:
            state = env.high_level_state()
            action = select_action(state, epsilon, dqn, N_ACTIONS)
            next_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            cumulative_reward += reward
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            if len(replay_buffer) > 1000 and episode_steps % update_frequency == 0:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones = torch.tensor(np.array(dones), dtype=torch.float32)
                with torch.no_grad():
                    target_q_values = target_dqn(next_states).max(1)[0]
                    targets = rewards + gamma * target_q_values * (1 - dones)
                q_values = dqn(states).gather(1,actions).squeeze()
                loss = loss_fn(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if episode % target_update_frequency == 0:
            target_dqn.load_state_dict(dqn.state_dict())
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        #if cumulative_reward > high_reward_threshold:
        #   epsilon = max(epsilon * epsilon_decay_high_reward, epsilon_min)
        #elif cumulative_reward > reward_threshold:
        #    epsilon = max(epsilon * epsilon_decay, epsilon_min)
        #else:
        #    epsilon = min(epsilon * epsilon_increase, epsilon_max)

        episode_steps += 1
        
        
        episode_rewards.append(cumulative_reward)
        per_episode_rewards.append(cumulative_reward / episode_steps)
        epsilon_decay_plot.append(epsilon)
        if episode % 5 == 0:
            print(f"Episode={episode}: Reward={cumulative_reward:.4f}, RPS={cumulative_reward/episode_steps:.4f}, Epsilon={epsilon:.4f}")

    torch.save(dqn.state_dict(),"DQN_model.pth")
    plt.subplot(1,3,1)
    plt.plot(episode_rewards)
    plt.title("Rewards")
    plt.subplot(1,3,2)
    plt.plot(per_episode_rewards)
    plt.title("Rewards per Episode Steps")
    plt.subplot(1,3,3)
    plt.plot(epsilon_decay_plot)
    plt.title("Epsilon")


    plt.show()
