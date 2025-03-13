import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os

# Define neural network for Q-learning
class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class RLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.learning_rate = learning_rate
        
        # Define action space (discretized)
        self.forward_actions = np.linspace(-3.0, 3.0, 7)  # [-3, -2, -1, 0, 1, 2, 3]
        self.lateral_actions = np.linspace(-0.5, 0.5, 5)  # [-0.5, -0.25, 0, 0.25, 0.5]
        self.vertical_actions = np.linspace(-0.5, 0.5, 5)  # [-0.5, -0.25, 0, 0.25, 0.5]
        
        # Initialize networks
        self.model = DQNModel(state_size, action_size)
        self.target_model = DQNModel(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(10000)
        
        # Training parameters
        self.update_target_every = 100
        self.train_counter = 0
        
    def get_action(self, state):
        """Select action according to epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            # Random action
            action_idx = random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action_idx = torch.argmax(q_values).item()
        
        # Convert action index to actual control values
        action_space = self.action_size // 3  # Assuming equal divisions for each control
        forward_idx = action_idx % 7
        lateral_idx = (action_idx // 7) % 5
        vertical_idx = (action_idx // 35) % 5
        
        return {
            'forward': self.forward_actions[forward_idx],
            'lateral': self.lateral_actions[lateral_idx],
            'vertical': self.vertical_actions[vertical_idx],
            'action_idx': action_idx
        }
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.push(state, action, reward, next_state, done)
        
    def train(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and backpropagate
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network if needed
        self.train_counter += 1
        if self.train_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, filepath):
        """Save model weights"""
        torch.save(self.model.state_dict(), filepath)
        
    def load_model(self, filepath):
        """Load model weights"""
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(self.model.state_dict())


# Helper functions for reward computation
def compute_reward(errors, actions):
    """Compute reward based on tracking errors and control actions"""
    lateral_error, vertical_error, forward_error = errors
    
    # Target region - higher reward for getting close to target
    target_reward = 0
    if abs(lateral_error) < 30 and abs(vertical_error) < 30 and abs(forward_error) < 0.5:
        target_reward = 10.0
    elif abs(lateral_error) < 60 and abs(vertical_error) < 60 and abs(forward_error) < 1.0:
        target_reward = 5.0
        
    # Error penalties - penalize being far from target
    lateral_penalty = -0.01 * abs(lateral_error)
    vertical_penalty = -0.01 * abs(vertical_error)
    forward_penalty = -1.0 * abs(forward_error)
    
    # Action penalties - penalize large control inputs
    action_penalty = -0.1 * (abs(actions['forward']) + abs(actions['lateral']) + abs(actions['vertical']))
    
    # Total reward
    reward = target_reward + lateral_penalty + vertical_penalty + forward_penalty + action_penalty
    
    return reward


# Example usage to train a model offline
def train_model_offline(train_data, num_episodes=1000):
    """
    Train model offline using collected data
    
    Args:
        train_data: List of dictionaries containing state, action, reward, next_state, done
        num_episodes: Number of episodes to train
    """
    state_size = 6  # [lateral_error, vertical_error, forward_error, vel_x, vel_y, vel_z]
    action_size = 7 * 5 * 5  # Discretized actions for forward, lateral, vertical
    
    agent = RLAgent(state_size, action_size)
    
    # Add data to replay buffer
    for sample in train_data:
        agent.memory.push(
            sample['state'], 
            sample['action_idx'], 
            sample['reward'], 
            sample['next_state'], 
            sample['done']
        )
    
    # Train for specified episodes
    for episode in range(num_episodes):
        agent.train()
        
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}")
    
    # Save trained model
    agent.save_model("drone_rl_model_offline.pth")
    print("Offline training complete. Model saved.")
    
    return agent


# Example code to simulate and train with synthetic data
def generate_synthetic_data(num_samples=10000):
    """
    Generate synthetic training data for offline training
    """
    synthetic_data = []
    
    for _ in range(num_samples):
        # Generate random state
        state = np.random.uniform(-100, 100, 6)  # Random errors and velocities
        
        # Generate random action
        action_idx = np.random.randint(0, 7*5*5)
        
        # Calculate forward, lateral, vertical actions
        forward_idx = action_idx % 7
        lateral_idx = (action_idx // 7) % 5
        vertical_idx = (action_idx // 35) % 5
        
        forward_action = np.linspace(-3.0, 3.0, 7)[forward_idx]
        lateral_action = np.linspace(-0.5, 0.5, 5)[lateral_idx]
        vertical_action = np.linspace(-0.5, 0.5, 5)[vertical_idx]
        
        actions = {
            'forward': forward_action,
            'lateral': lateral_action,
            'vertical': vertical_action
        }
        
        # Generate next state based on simple dynamics
        next_state = state.copy()
        next_state[0] -= lateral_action * 20  # Lateral error changes based on lateral action
        next_state[1] -= vertical_action * 20  # Vertical error changes based on vertical action
        next_state[2] -= forward_action * 0.5  # Forward error changes based on forward action
        
        # Add some noise
        next_state += np.random.normal(0, 5, 6)
        
        # Compute reward
        errors = [state[0], state[1], state[2]]
        reward = compute_reward(errors, actions)
        
        # Add to synthetic data
        synthetic_data.append({
            'state': state,
            'action_idx': action_idx,
            'reward': reward,
            'next_state': next_state,
            'done': False
        })
    
    return synthetic_data


if __name__ == '__main__':
    # Example: Train with synthetic data
    print("Generating synthetic training data...")
    data = generate_synthetic_data(10000)
    
    print("Training model with synthetic data...")
    agent = train_model_offline(data, num_episodes=500)
    
    # Optional: Test trained model
    test_state = np.array([50.0, 30.0, 2.0, 0.0, 0.0, 0.0])
    action = agent.get_action(test_state)
    print(f"Test state: {test_state}")
    print(f"Recommended action: {action}")