import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os, pickle

# Define neural network to predect Q values
# Input: current state  (size: state_size)
# Output: Q values for all actions  (size: action_size)
class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        
    def forward(self, x):
        return self.net(x)

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
    def __init__(self, learning_rate=0.0001, gamma=0.99, epsilon_decay = 0.999):
        self.state_size = 3

        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.batch_size = 32
        self.learning_rate = learning_rate
        
        # Define action space
        # self.motor_action_space = np.linspace(0.6, 0.8, 21) # [0.6 0.61 ... 0.8]
        # self.action_size_per_motor = len(self.motor_action_space)
        # self.action_size = (self.action_size_per_motor)**4
        self.vertical_action_space = np.linspace(-0.5, 0.5, 21) # [-5.0 -4.5 ... 5.0]
        self.forward_action_space = np.linspace(-3.0, 3.0, 21) # [-3.0 -2.7 ... 3.0]
        self.yaw_action_space = np.linspace(-0.5, 0.5, 21) # [-5.0 -4.5 ... 5.0]
        self.action_size = len(self.vertical_action_space) * len(self.forward_action_space) * len(self.yaw_action_space)
        
        # Initialize networks
        self.model = DQNModel(self.state_size, self.action_size)
        self.target_model = DQNModel(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(20000)
        
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
        
        # # Convert action index to actual control values
        # # action 총 개수가 x * x * x * x개 일 때, action_idx하나가 정해지면 아래 과정을 통해 action을 정한다
        # motor0_idx = action_idx % self.action_size_per_motor # result: 0 ~ (self.action_size_per_motor - 1)
        # motor1_idx = (action_idx // self.action_size_per_motor) % self.action_size_per_motor
        # motor2_idx = (action_idx // (self.action_size_per_motor)**2) % self.action_size_per_motor
        # motor3_idx = (action_idx // (self.action_size_per_motor)**3) % self.action_size_per_motor
        
        # return {
        #     'motor0': self.motor_action_space[motor0_idx],
        #     'motor1': self.motor_action_space[motor1_idx],
        #     'motor2': self.motor_action_space[motor2_idx],
        #     'motor3': self.motor_action_space[motor3_idx],
        #     'action_idx': action_idx
        # }

        # Convert action index to setpoints
        cor_ver_idx = action_idx % len(self.vertical_action_space)
        cor_for_idx = (action_idx // len(self.vertical_action_space)) % len(self.forward_action_space)
        cor_yaw_idx = (action_idx // (len(self.vertical_action_space) * len(self.forward_action_space))) % len(self.yaw_action_space)

        return {
            'cor_ver' : self.vertical_action_space[cor_ver_idx],
            'cor_for' : self.forward_action_space[cor_for_idx],
            'cor_yaw' : self.yaw_action_space[cor_yaw_idx],
            'action_idx' : action_idx
        }

    def remember(self, state, action, reward, next_state, done_episode):
        """Store experience in memory"""
        self.memory.push(state, action, reward, next_state, done_episode)
        
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
        # states에 대한 모든 행동의 Q값을 신경망(model)에서 avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0예측해서, 예측된 Q 값 중 실제로 수행한 action에 해당하는 Q 값만 추출(.gather())하고, 차원을 줄여서 1D 텐서로 변환 (.squeeze(1))
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
            # print("epsilon decayed")

    def save_model(self, path="agent_state.pth"):
        torch.save({
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }, path)

    def load_model(self, path="agent_state.pth"):
        if os.path.exists(path):
            data = torch.load(path)
            self.model.load_state_dict(data["model"])
            self.target_model.load_state_dict(data["target_model"])
            self.optimizer.load_state_dict(data["optimizer"])
            self.epsilon = data.get("epsilon", 1.0)

            print(f"Epsilon: {self.epsilon:.4f}")

    def compute_reward(self, lateral_error, vertical_error, forward_error):
        """Compute reward based on tracking errors and control actions"""
        # Target region - higher reward for getting close to target
        target_reward = 0
        if abs(lateral_error) < 32 and abs(vertical_error) < 0.2 and abs(forward_error) < 1.0:
            target_reward = 20.0
        elif abs(lateral_error) < 64 and abs(vertical_error) < 0.4 and abs(forward_error) < 2.0:
            target_reward = 15.0
        elif abs(lateral_error) < 128 and abs(vertical_error) < 0.8 and abs(forward_error) < 4.0:
            target_reward = 10.0
            
        # Error penalties - penalize being far from target
        lateral_penalty = -0.05 * min(abs(lateral_error), 300)
        vertical_penalty = -0.1 * min(abs(vertical_error), 300)
        forward_penalty = -0.1 * min(abs(forward_error), 300)
        
        # Total reward
        reward = target_reward + lateral_penalty + vertical_penalty + forward_penalty
        
        return reward

def offline_train_from_pickle(pickle_path, max_iteration=200):
    with open(pickle_path, "rb") as f:
        expert_data = pickle.load(f)

    agent = RLAgent(learning_rate=0.01)

    for sample in expert_data:
        if sample["state"] is None or sample["next_state"] is None:
            continue
        if not isinstance(sample["state"], (list, np.ndarray)) or not isinstance(sample["next_state"], (list, np.ndarray)):
            continue

        state = np.array(sample['state']).flatten().tolist()
        next_state = np.array(sample['next_state']).flatten().tolist()
        agent.remember(
            state,
            sample['action_idx'],
            sample['reward'],
            next_state,
            sample['done']
        )

    loss_log = []
    best_loss = float("inf")
    best_model_state = None

    for iteration in range(max_iteration):
        if len(agent.memory) >= agent.batch_size:
            batch = agent.memory.sample(agent.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)

            current_q = agent.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = agent.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * agent.gamma * next_q
            loss = F.mse_loss(current_q, target_q)
            loss_log.append(loss.item())

            # best loss 저장
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = {
                    "model": agent.model.state_dict(),
                    "target_model": agent.target_model.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "epsilon": agent.epsilon
                }

            agent.train()

        if iteration % 10 == 0:
            recent_losses = loss_log[-5:]
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            print(f"iteration: {iteration}/{max_iteration}, avg_loss: {avg_loss:.4f}")

    if best_model_state is not None:
        torch.save(best_model_state, f"{best_loss:.4f}.pth")
        print(f"Offline training complete. Best model saved with loss {best_loss:.4f}.")
    else:
        print("Training did not produce any valid model.")

if __name__ == "__main__":
    print("Training model with expert data...")
    offline_train_from_pickle("/home/gr/iris_tracker_PID_RL/data/expert_data0402.pkl", max_iteration=120)