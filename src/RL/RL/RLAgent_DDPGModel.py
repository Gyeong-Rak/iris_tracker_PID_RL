# RLAgent_DDPGModel.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.max_action = torch.tensor(max_action, dtype=torch.float32)

    def forward(self, state):
        return self.net(state) * self.max_action.to(state.device)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*samples)
        return (np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d))

    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, state_dim=3, action_dim=4, max_action=[2.0, 2.0, 1.0, 0.5]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = np.array(max_action)

        self.actor = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.buffer = ReplayBuffer()
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.noise_std = 0.1

    def get_action(self, state, add_noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().detach().numpy().flatten()
        if add_noise:
            action += np.random.normal(0, self.noise_std, size=self.action_dim)
        return np.clip(action, -1.0, 1.0) * self.max_action

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_a = self.actor_target(ns)
            target_q = r + (1 - d) * self.gamma * self.critic_target(ns, next_a)
        q = self.critic(s, a)
        critic_loss = nn.MSELoss()(q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def save_model(self, path="ddpg_actor.pth"):
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path="ddpg_actor.pth"):
        if torch.cuda.is_available():
            self.actor.load_state_dict(torch.load(path))
        else:
            self.actor.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def compute_reward(self, lateral_error, vertical_error, forward_error):
        """Compute reward based on tracking errors and control actions"""
        # Target region - higher reward for getting close to target
        target_reward = 0

        if abs(lateral_error) < 64 and abs(vertical_error) < 0.4 and abs(forward_error) < 2.0:
            target_reward = 10.0
            
        # Error penalties - penalize being far from target
        lateral_penalty = -0.05 * min(abs(lateral_error), 300)
        vertical_penalty = -0.2 * min(abs(vertical_error), 300)
        forward_penalty = -0.2 * min(abs(forward_error), 300)
        
        # Total reward
        reward = target_reward + lateral_penalty + vertical_penalty + forward_penalty
        
        return reward