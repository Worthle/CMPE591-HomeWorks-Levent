import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import torch.nn.functional as F

# Hyperparameters
learning_rate = 3e-4
gamma = 0.99
tau = 0.005
buffer_size = 100000
batch_size = 128
target_entropy_scale = 1.0  # Target entropy = -action_dim * scale
episodes = 100000

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        u = dist.rsample()
        a = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return a, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.q(torch.cat([state, action], dim=-1))

class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.target_critic_1 = Critic(state_dim, action_dim)
        self.target_critic_2 = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=learning_rate)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.replay_buffer = deque(maxlen=buffer_size)

        self.target_entropy = -action_dim * target_entropy_scale
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

    def act(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action, _ = self.actor.sample(state)
        return action.squeeze(0).numpy()

    def store(self, transition):
        self.replay_buffer.append(transition)

    def soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

    def learn(self):
        if len(self.replay_buffer) < batch_size:
            return

        batch = list(zip(*random.sample(self.replay_buffer, batch_size)))
        states = torch.tensor(np.vstack(batch[0]), dtype=torch.float32)
        actions = torch.tensor(np.vstack(batch[1]), dtype=torch.float32)
        rewards = torch.tensor(np.vstack(batch[2]), dtype=torch.float32)
        next_states = torch.tensor(np.vstack(batch[3]), dtype=torch.float32)
        dones = torch.tensor(np.vstack(batch[4]), dtype=torch.float32)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.target_critic_1(next_states, next_actions)
            q2_next = self.target_critic_2(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - torch.exp(self.log_alpha) * next_log_probs
            target_q = rewards + gamma * (1 - dones) * min_q_next

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        loss_critic_1 = F.mse_loss(q1, target_q)
        loss_critic_2 = F.mse_loss(q2, target_q)

        self.critic_1_optimizer.zero_grad()
        loss_critic_1.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        loss_critic_2.backward()
        self.critic_2_optimizer.step()

        actions_new, log_probs_new = self.actor.sample(states)
        q1_new = self.critic_1(states, actions_new)
        q2_new = self.critic_2(states, actions_new)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (torch.exp(self.log_alpha) * log_probs_new - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs_new + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update(self.target_critic_1, self.critic_1)
        self.soft_update(self.target_critic_2, self.critic_2)

if __name__ == "__main__":
    env = gym.make("Pusher-v5", max_episode_steps=500)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim=obs_dim, action_dim=act_dim)

    episode_rewards = []   # To store total reward each episode
    mean_rewards = []      # To store sliding window mean reward
    window_size = 300       # Window size for mean reward computation
    episode_steps = 0

    # Set up interactive plotting
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ep_line, = ax.plot([], [], '-', label="Episode Reward", color='gray', linewidth=0.1)
    mean_line, = ax.plot([], [], 'r', label=f"Mean Reward (window size = {window_size})", linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Progress - SAC")
    ax.legend()

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        cumulative_reward = 0.0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = agent.act(state_tensor)
            
            next_state, reward, is_terminal, is_truncated, _ = env.step(action)
            reward = reward / 10.0  # Scale rewards to a reasonable range
            cumulative_reward += reward
            done = is_terminal or is_truncated

            agent.store((state, action, reward, next_state, done))

            state = next_state
            episode_steps += 1
            if episode_steps % 10 == 0:
                agent.learn()
            
            if episode_steps % 100 == 0:
                #agent.update_model()

                agent.soft_update(agent.target_critic_1, agent.critic_1)
                agent.soft_update(agent.target_critic_2, agent.critic_2)

        episode_rewards.append(cumulative_reward)
        
        # Compute sliding mean reward
        if len(episode_rewards) >= window_size:
            mean_reward = np.mean(episode_rewards[-window_size:])
        else:
            mean_reward = np.mean(episode_rewards)
        mean_rewards.append(mean_reward)

        if episode % 50 == 0:
            print(f"Episode {episode}: Total Reward = {cumulative_reward:.2f}, Mean Reward = {mean_reward:.2f}")
        if episode % 500 == 0:
            # Update the dynamic plot after every episode
            ep_line.set_data(range(len(episode_rewards)), episode_rewards)
            mean_line.set_data(range(len(mean_rewards)), mean_rewards)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
            

        if episode % 1000 == 0:
            # Save the model every 100 episodes
            torch.save(agent.actor.state_dict(), f"SAC/actor_sac2_{episode}.pth")
            print(f"Model saved as actor_sac_{episode}.pth")

        if episode % 1000 == 0:
            fig.savefig(f"SAC/training_progress_sac_{episode}.png")

    # Turn off interactive mode and display the final plot
    plt.ioff()
    plt.show()

    # Save the final model
    torch.save(agent.actor.state_dict(), "actor_sac.pth")
    print("Model saved as actor_sac.pth")

    # Save the final plot
    fig.savefig("training_progress_sac.png")

    env.close()