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
        self.fc1 = nn.Linear(state_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, action_dim)
        self.tanh = nn.Tanh()
        self.max_action = torch.tensor(max_action, dtype=torch.float32)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.tanh(self.fc2(x)) * self.max_action.to(x.device)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

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

class OUActionNoise:
    def __init__(self, mu, sigma=0.1, theta=0.15, dt=0.05, x0=None):
        self.mu     = mu
        self.sigma  = sigma
        self.theta  = theta
        self.dt     = dt
        self.x_prev = x0 if x0 is not None else np.zeros_like(self.mu)

    def __call__(self, dt=0.05):
        dt = dt if dt is not None else self.dt
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * dt
            + self.sigma * np.sqrt(dt) * np.random.randn(*self.mu.shape)
        )
        self.x_prev = x
        return x

class DDPGAgent:
    def __init__(self, batch_size = 128, gamma = 0.99, tau = 0.001, noise_std = 0.05,
                  actor_lr = 2e-4, critic_lr = 2e-3,
                  max_action=[1.0, 2.0, 0.5], mode='all'):
        # state : [error_vertical, error_forward, error_lateral]    /    action : [correction_vertical, correction_forward, correction_yaw]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mode = mode
        self.max_action = np.array(max_action)
        if mode == 'lateral':
            self.state_dim, self.action_dim = 2, 1
            self.max_action = np.array([self.max_action[2]])
        elif mode == 'vertical':
            self.state_dim, self.action_dim = 3, 1
            self.max_action = np.array([self.max_action[0]])
        elif mode == 'forward':
            self.state_dim, self.action_dim = 3, 1
            self.max_action = np.array([self.max_action[1]])
        else:  # 'all'
            print("################## Please check the mode")

        self.max_action_tensor = torch.tensor(self.max_action, dtype=torch.float32, device=self.device).view(1, -1)

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-3)

        self.critic2 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr, weight_decay=1e-3)

        self.buffer = ReplayBuffer()
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.ou_noise = OUActionNoise(mu=np.zeros(self.action_dim), sigma=self.noise_std, theta=0.15, dt=0.05)

    def get_action(self, state, add_noise=True, dt=0.05):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().detach().numpy().flatten()
        if add_noise: action += self.ou_noise(dt=dt)
        action = np.clip(action, -1.0, 1.0) * self.max_action

        return action

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device).unsqueeze(1) / self.max_action_tensor
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        # double critic update
        with torch.no_grad():
            next_a = self.actor_target(ns) / self.max_action_tensor
            tq1 = self.critic_target(ns, next_a)
            tq2 = self.critic2_target(ns, next_a)
            ratio = torch.min(tq1, tq2) / (torch.max(tq1, tq2) + 1e-6) # 1e-6 for numerial stability
            target_q = r + (1 - d) * self.gamma * ratio * torch.min(tq1, tq2)

        q1 = self.critic(s, a)
        loss1 = nn.MSELoss()(q1, target_q)
        self.critic_optimizer.zero_grad()
        loss1.backward()
        self.critic_optimizer.step()

        q2 = self.critic2(s, a)
        loss2 = nn.MSELoss()(q2, target_q)
        self.critic2_optimizer.zero_grad()
        loss2.backward()
        self.critic2_optimizer.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
        for t, s in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def save_model(self, path="ddpg_checkpoint.pth"):
        torch.save({
            'actor_state_dict'        : self.actor.state_dict(),
            'actor_target_state_dict' : self.actor_target.state_dict(),
            'critic_state_dict'       : self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'critic2_state_dict':       self.critic2.state_dict(),
            'critic2_target_state_dict':self.critic2_target.state_dict(),

            'actor_optimizer'         : self.actor_optimizer.state_dict(),
            'critic_optimizer'        : self.critic_optimizer.state_dict(),
            'critic2_optimizer':       self.critic2_optimizer.state_dict(),

            'noise_std'               : self.noise_std,
            'gamma'                   : self.gamma,
            'tau'                     : self.tau
        }, path)

    def load_model(self, path="ddpg_checkpoint.pth"):
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])

        self.noise_std = checkpoint.get('noise_std', self.noise_std)
        self.gamma     = checkpoint.get('gamma',     self.gamma)
        self.tau       = checkpoint.get('tau',       self.tau)

    def compute_reward(self, error):
        return -np.linalg.norm(error)