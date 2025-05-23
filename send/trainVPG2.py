import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from collections import deque

from homework3 import Hw3Env

import environment

env = Hw3Env(render_mode="offscreen")

# Training hyperparameters
lr_p = 0.01
lr_v = 0.0005
gamma = 0.99
batch_size = 4


# Policy network
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.state_space = env.high_level_state().shape[0]
        

        self.fc1 = nn.Linear(self.state_space, 128)
        self.fc2 = nn.Linear(128, 2)

        self.gamma = gamma
        self.policy_hist = Variable(torch.Tensor())  # policy history for traj
        self.traj_reward = []
        self.loss_hist = Variable(
            torch.Tensor()
        )  # loss history for each traj in episode

    def forward(self, x):
        model = nn.Sequential(
            self.fc1, nn.Dropout(p=0.6), nn.ReLU(), self.fc2,
            nn.Softmax(dim=-1)
        )

        return model(x)


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()

        self.state_space = env.high_level_state().shape[0]

        self.fc1 = nn.Linear(self.state_space, 32)
        self.fc2 = nn.Linear(32, 1)

        self.value_hist = Variable(torch.Tensor())  # policy history for traj
        self.loss_hist = Variable(
            torch.Tensor()
        )  # loss history for each traj in episode

    def forward(self, x):
        model = nn.Sequential(self.fc1, nn.ReLU(), nn.Dropout(p=0.6), self.fc2)

        return model(x)


# Instantiate networks and optimizers
policy = PolicyNet()
value = ValueNet()
optimizer_policy = opt.Adam(policy.parameters(), lr=lr_p)
optimizer_value = opt.Adam(value.parameters(), lr=lr_v)
tb = True  # use tensorboard?
if tb:
    writer = SummaryWriter()


# stochastic policy
def select_action(s):
    state = torch.from_numpy(s).type(torch.FloatTensor)
    probs = policy(Variable(state))
    val = value(Variable(state))
    c = torch.distributions.Normal(probs, val)
    action = c.sample()  # sample action from distribution

    # store log probabilities and value function
    policy.policy_hist = torch.cat(
        [policy.policy_hist, c.log_prob(action).unsqueeze(0)]
    )
    value.value_hist = torch.cat([value.value_hist, val])

    return action


# compute losses for current trajectories
def get_traj_loss():
    returns = []
    R = 0

    # calculate discounted returns
    for r in policy.traj_reward[::-1]:
        R = r + policy.gamma * R
        returns.insert(0, R)

    # scale returns
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (
        returns.std() + 1e-8)

    # calculate trajectory losses
    loss_policy = torch.sum(
        torch.mul(
            policy.policy_hist, Variable(returns) - Variable(value.value_hist)
        ).mul(-1),
        -1,
    ).unsqueeze(0)

    loss_value = nn.MSELoss()(value.value_hist, Variable(returns)).unsqueeze(0)

    # store loss values
    policy.loss_hist = torch.cat([policy.loss_hist, loss_policy])

    value.loss_hist = torch.cat([value.loss_hist, loss_value])

    # clear traj_reward and policy and value histories
    policy.traj_reward = []
    policy.policy_hist = Variable(torch.Tensor())
    value.value_hist = Variable(torch.Tensor())


# network update after every batch of trajectories (end of episode)
def update_policy(ep):
    # compute episode loss from traj losses
    loss_policy = torch.mean(policy.loss_hist)
    loss_value = torch.mean(value.loss_hist)

    # tensorboard book keeping
    if tb:
        writer.add_scalar("loss/policy", loss_policy, ep)
        writer.add_scalar("loss/value", loss_value, ep)

    # take gradient steps
    optimizer_policy.zero_grad()
    loss_policy.backward()
    optimizer_policy.step()

    optimizer_value.zero_grad()
    loss_value.backward()
    optimizer_value.step()

    # re-initialize loss histories
    policy.loss_hist = Variable(torch.Tensor())
    value.loss_hist = Variable(torch.Tensor())


if __name__ == "__main__":
    for ep in range(1000):  # episodes
        episode_reward = 0
        for i in range(batch_size):  # batch of trajectories
            s = env.reset()
            done = False
            for t in range(1000):  # trajectory
                a = select_action(s)
                s, r, is_terminal, is_truncated = env.step(a)  # take step in env
                done = is_terminal or is_truncated
                policy.traj_reward.append(r)  # store traj reward

                if done:
                    break

            episode_reward += (
                np.sum(policy.traj_reward) / batch_size
            )  # add to average episode reward
            get_traj_loss()  # compute traj losses

        update_policy(ep)  # one step with computed losses

        if ep % 5 == 0:
            print("Episode: {}, reward: {}".format(ep, episode_reward))
            if tb:
                writer.add_scalar("reward", episode_reward, ep)


    #torch.save(agent.model.state_dict(), "VPG_model.pt")
    #torch.save(agent.model.state_dict(), "VPG_model.pth")
    #np.save("rewards.npy", np.array(rewards))
    #np.save("RPS.npy", np.array(RPS))

    plt.plot(episode_reward, '-', color='gray', linewidth=0.1)
    plt.plot(mov_avg(episode_reward, 100), 'r', linewidth=1)
    plt.title("Rewards")
    plt.show()