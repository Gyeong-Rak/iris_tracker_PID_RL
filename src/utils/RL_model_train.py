import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os

# Define neural network to predect Q values
# Input: current state  (size: state_size)
# Output: Q values for all actions  (size: action_size)
class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 64) # fully connected layer, state_size -> 64 neurons
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) # forward using Rectified Linear Unit
        x = F.relu(self.fc2(x))
        return self.fc3(x) # there is no activation function at output layer. because outputs are not possibilities but expected values

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) # first-in-first-out format
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done)) # save data
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size) # sample data
    
    def __len__(self):
        return len(self.buffer)

class RLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.batch_size = 32
        self.learning_rate = learning_rate
        
        # Define action space
        self.motor0 = np.concatenate(np.linspace(-1.0, 0.0, 5), np.linspace(0.1, 1.0, 10)) # [-1.0, -0.75, -0.5, -0.25, 0.0, 0.1, 0.2, 0.3, ..., 1.0]
        self.motor1 = np.concatenate(np.linspace(-1.0, 0.0, 5), np.linspace(0.1, 1.0, 10))
        self.motor2 = np.concatenate(np.linspace(-1.0, 0.0, 5), np.linspace(0.1, 1.0, 10))
        self.motor3 = np.concatenate(np.linspace(-1.0, 0.0, 5), np.linspace(0.1, 1.0, 10))
        
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
        # Local minima에 빠지지 않고 overfitting을 막기 위해 epsilon을 사용해서 랜덤성을 보장
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            # state를 pytorch tensor로 변환하고 size를 (state_size, )에서 (batch_size(=1), state_size)로 변환
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # inference 과정에서는 backpropagation을 안하기 때문에 grad계산 안하기
            with torch.no_grad():
                # model에서 각 state에 대한 Q 값을 계산
                q_values = self.model(state_tensor)
            # 최대 Q 값의 index를 action_idx로 설정
            action_idx = torch.argmax(q_values).item()
        
        # Convert action index to actual control values
        # action 총 개수가 15 * 15 * 15 * 15개 일 때, action_idx하나가 정해지면 아래 과정을 통해 action을 정한다
        motor0_idx = action_idx % 15 # result: 0~14
        motor1_idx = (action_idx // 15) % 15
        motor2_idx = (action_idx // 225) % 15
        motor3_idx = (action_idx // 3375) % 15
        
        return {
            'motor0': self.motor0[motor0_idx],
            'motor1': self.motor1[motor1_idx],
            'motor2': self.motor2[motor2_idx],
            'motor3': self.motor3[motor3_idx],
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
        # data 상관관계를 제거하여 Independent and Identically Distributed 데이터를 보장하여 Local minima를 방지하기 위해 memory에서 랜덤 샘플링
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        # states에 대한 모든 행동의 Q값을 신경망(model)에서 예측해서, 예측된 Q 값 중 실제로 수행한 action에 해당하는 Q 값만 추출(.gather())하고, 차원을 줄여서 1D 텐서로 변환 (.squeeze(1))
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        # inference 과정에서는 backpropagation을 안하기 때문에 grad계산 안하기
        with torch.no_grad():
            # self.target_model(next_states)의 결과 (batch_size, action_size)모양의 Q 값이 나오고, 각 batch에서의 최대 Q 값(.max(1))의 최대값([0])
            # Bellman Equation에서 next step의 Q 값은 그냥 model이 아닌 target_model에서 계산함.
            # 만약 그냥 model을 사용한다면 훈련과 업데이트가 겹치면서 Q 값이 불안정해지고 발산 가능성
            next_q = self.target_model(next_states).max(1)[0]
        # Bellman Equation: Q(s,a) = rewards + gamma * maxQ(s',a')
        # 현재의 보상 뿐 아니라 미래의 보상까지 고려한 Q 값 -> 이상적인 값 -> target_q
        # dones == 1이면 next_q의 영향이 없어진다.
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and backpropagate
        # 현재 신경망이 예측한 Q 값 (current_q)와 Bellman Equation으로 예측한 Q 값의 차이(Mean Square Error)계산
        loss = F.mse_loss(current_q, target_q)
        # pytorch는 grad를 accumulate하므로 기존 값 제거
        self.optimizer.zero_grad()
        # backpropagation을 이용하여 신경망의 결과가 Bellman Equation과 같아지도록 학습
        loss.backward()
        # update weights, optimizer is defined at init step
        self.optimizer.step()
        
        # Update target network if needed
        self.train_counter += 1
        # target_model도 가끔씩 update
        if self.train_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        # 강화학습에서는 탐색(Exploration)과 활용(Exploitation)의 균형을 맞추는 것이 중요함.
        # 초반에는 random하게 환경을 탐색하는 것에 집중. 후반에는 학습이 많이 진행됐다고 판단하고 최적 정책을 따라가도록 설정
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
def compute_reward(errors, roll, pitch):
    """Compute reward based on tracking errors and control actions"""
    lateral_error, vertical_error, forward_error = errors
    
    # Target region - higher reward for getting close to target
    target_reward = 0
    if abs(lateral_error) < 30 and abs(vertical_error) < 30 and abs(forward_error) < 0.5:
        target_reward = 10.0
    elif abs(lateral_error) < 60 and abs(vertical_error) < 60 and abs(forward_error) < 1.0:
        target_reward = 5.0
        
    # Error penalties - penalize being far from target
    lateral_penalty = -0.01 * min(abs(lateral_error), 200)
    vertical_penalty = -0.01 * min(abs(vertical_error), 200)
    forward_penalty = -0.01 * min(abs(forward_error), 500)
    
    # Attitude penalties - penalize unstable attitude
    attitude_penalty = -0.01 * (abs(roll) + abs(pitch))
    
    # Total reward
    reward = target_reward + lateral_penalty + vertical_penalty + forward_penalty + attitude_penalty
    
    return reward


# Example usage to train a model offline
def train_model_offline(train_data, num_episodes=1000):
    """
    Train model offline using collected data
    
    Args:
        train_data: List of dictionaries containing state, action, reward, next_state, done
        num_episodes: Number of episodes to train
    """
    state_size = 5  # [lateral_error, vertical_error, forward_error, roll, pitch]
    action_size = 15 * 15 * 15 * 15 # Discretized actions for motors
    
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

if __name__ == '__main__':
    data = []
    
    print("Training model with synthetic data...")
    agent = train_model_offline(data, num_episodes=500)
    
    # Optional: Test trained model
    test_state = np.array([50.0, 30.0, 2.0, 0.0, 0.0, 0.0])
    action = agent.get_action(test_state)
    print(f"Test state: {test_state}")
    print(f"Recommended action: {action}")