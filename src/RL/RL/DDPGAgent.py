# RLAgent_DDPGModel.py
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_p=0.5):
        super().__init__()
        self.fc1   = nn.Linear(in_features, out_features)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(p=dropout_p)
        self.fc2   = nn.Linear(out_features, out_features)
        self.skip  = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x + identity
        return self.relu(x)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.block1 = ResidualBlock(state_dim, 128, dropout_p=0.5)
        self.block2 = ResidualBlock(128, 256, dropout_p=0.5)
        self.block3 = ResidualBlock(256, 128, dropout_p=0.5)
        self.out    = nn.Linear(128, action_dim)
        self.tanh   = nn.Tanh()
        self.max_action = torch.tensor(max_action, dtype=torch.float32)

        self.apply(self._init_weights)
        nn.init.uniform_(self.out.weight, -1e-3, 1e-3)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, state):
        x = self.block1(state)
        x = self.block2(x)
        x = self.block3(x)
        return self.tanh(self.out(x)) * self.max_action.to(x.device)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        in_dim = state_dim + action_dim
        self.block1 = ResidualBlock(in_dim, 128, dropout_p=0.5)
        self.block2 = ResidualBlock(128, 256, dropout_p=0.5)
        self.block3 = ResidualBlock(256, 128, dropout_p=0.5)
        self.out    = nn.Linear(128, 1)

        self.apply(self._init_weights)
        nn.init.uniform_(self.out.weight, -1e-3, 1e-3)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.out(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

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
    def __call__(self, dt=0.02):
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
                  state_dim=3, action_dim=3, max_action=[1.0, 5.0, 0.5]):
        # state : [error_vertical, error_forward, error_lateral]    /    action : [correction_vertical, correction_forward, correction_yaw]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = np.array(max_action)
        self.max_action_tensor = torch.tensor(self.max_action, dtype=torch.float32, device=self.device).view(1, -1) # for action normalization

        self.actor = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-3)

        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
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
        if add_noise:
            action += self.ou_noise(dt=dt)
        return np.clip(action, -1.0, 1.0) * self.max_action

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device) / self.max_action_tensor
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

        # self.critic_lr_scheduler.step()
        # self.critic2_lr_scheduler.step()

        # Actor update
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor_lr_scheduler.step()

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

    def compute_reward(self, vertical_error, forward_error, lateral_error, speed, pitch):

        vertical_error = abs(vertical_error)
        forward_error = abs(forward_error)
        lateral_error = abs(lateral_error)
        speed = abs(speed)
        pitch = abs(pitch)

        # if vertical_error < 0.1: R1 = 5.0 - 50 * vertical_error
        # elif vertical_error < 0.3: R1 = -5 * math.ceil(vertical_error * 10)
        # else: R1 = -5 -5 * math.ceil(vertical_error * 5)

        # if forward_error < 0.1: R2 = 5.0 - 50 * forward_error
        # elif forward_error < 0.3: R2 = -5 * math.ceil(forward_error * 10)
        # else: R2 = -5 -5 * math.ceil(forward_error * 5)

        # if lateral_error < 0.1: R3 = 5.0 - 50 * lateral_error
        # elif lateral_error < 0.3: R3 = -5 * math.ceil(lateral_error * 10)
        # else: R3 = -5 -5 * math.ceil(lateral_error * 5)

        # if speed < 1.0: R4 = 1.0
        # elif speed < 2.0: R4 = -3 * math.floor(speed)
        # else: R4 = -3 -3 * math.floor(speed)
    
        R1 = -vertical_error
        R2 = -forward_error
        R3 = -lateral_error
        R5 = -pitch

        reward = (0.4 * R1 + 0.5 * R2 + 0.4 * R3 + 0.2 * R5) * 10

        return reward
    
def offline_train_from_pickle_ddpg(pickle_path):
    import pickle
    import numpy as np
    import torch
    import copy
    
    with open(pickle_path, "rb") as f:
        expert_data = pickle.load(f)

    max_iteration = 500
    batch_size = 100
    gamma = 0.95
    tau = 0.001
    noise_std = 0.05
    actor_lr = 0.0006
    critic_lr = 0.001
    lr_decay_rate = 0.995

    agent = DDPGAgent(batch_size=batch_size, gamma=gamma, tau=tau, noise_std=noise_std, actor_lr=actor_lr, critic_lr=critic_lr, lr_decay_rate=lr_decay_rate)

    for sample in expert_data:
        if sample["state"] is None or sample["next_state"] is None:
            continue
        if not isinstance(sample["state"], (list, np.ndarray)) or not isinstance(sample["next_state"], (list, np.ndarray)):
            continue

        state = np.array(sample['state']).flatten()
        next_state = np.array(sample['next_state']).flatten()
        action = np.array(sample['action']).flatten()  # DDPG는 continuous action
        reward = sample['reward']
        done = sample['done']

        agent.remember(state, action, reward, next_state, done)

    print("Training model with expert data...")
    print(f"actor_lr: {actor_lr}, critic_lr: {critic_lr}, lr_decay_rate: {lr_decay_rate}, gamma: {gamma}, tau: {tau}, noise_std: {noise_std}")
    print(f"Buffer size: {len(agent.buffer)}, batch size: {agent.batch_size}, max_iteration: {max_iteration}")

    # 4. offline training
    loss_log = []
    best_loss = float("inf")
    best_iteration = 0
    best_model_path = None
    stack = 0  # early stopping을 위한 변수 추가
    
    # 최적 모델 상태 저장을 위한 변수
    best_actor_state = None
    best_critic_state = None
    best_actor_target_state = None
    best_critic_target_state = None

    for iteration in range(max_iteration):
        agent.train()

        if iteration % 10 == 0 and len(agent.buffer) >= agent.batch_size:
            # 손실 측정
            s, a, r, ns, d = agent.buffer.sample(agent.batch_size)

            s = torch.FloatTensor(s).to(agent.device)
            a = torch.FloatTensor(a).to(agent.device)
            r = torch.FloatTensor(r).unsqueeze(1).to(agent.device)
            ns = torch.FloatTensor(ns).to(agent.device)
            d = torch.FloatTensor(d).unsqueeze(1).to(agent.device)

            with torch.no_grad():
                next_a = agent.actor_target(ns)
                target_q = r + (1 - d) * agent.gamma * agent.critic_target(ns, next_a)
            q = agent.critic(s, a)
            loss = torch.nn.functional.mse_loss(q, target_q)

            avg_loss = loss.item()
            loss_log.append(avg_loss)

            print(f"[{iteration}/{max_iteration}] Avg Critic Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_iteration = iteration
                best_model_path = f"ddpg_best_{best_loss:.4f}_iter_{best_iteration}.pth"
                stack = 0  # 손실이 개선되면 stack 초기화
                
                # 최적 모델 상태 저장
                best_actor_state = copy.deepcopy(agent.actor.state_dict())
                best_critic_state = copy.deepcopy(agent.critic.state_dict())
                best_actor_target_state = copy.deepcopy(agent.actor_target.state_dict())
                best_critic_target_state = copy.deepcopy(agent.critic_target.state_dict())
            else:
                stack += 1
                if stack > 5:
                    print(f"Early stopping at iteration {iteration} after 3 evaluations without improvement")
                    break

    # 최적 상태로 모델 복원
    if best_actor_state is not None:
        agent.actor.load_state_dict(best_actor_state)
        agent.critic.load_state_dict(best_critic_state)
        agent.actor_target.load_state_dict(best_actor_target_state)
        agent.critic_target.load_state_dict(best_critic_target_state)

    if best_model_path:
        agent.save_model(best_model_path)
        print(f"Offline training complete. Best model saved: {best_model_path} (iteration {best_iteration}/{max_iteration}, loss: {best_loss:.6f})")
        print(f"Training was {'early stopped' if stack > 3 else 'completed'} after {iteration+1} iterations")
    else:
        print("Training finished without saving best model.")

def offline_train_grid_search_ddpg(pickle_path, save_dir="./models"):
    import pickle
    import os
    import itertools
    import numpy as np
    import torch
    import time
    from datetime import datetime
    import json
    import copy
    
    # 결과 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 전문가 데이터 로드
    with open(pickle_path, "rb") as f:
        expert_data = pickle.load(f)
    
    # 그리드 서치를 위한 파라미터 설정
    param_grid = {
        'batch_size': [32],
        'gamma': [0.95, 0.99],
        'tau': [0.001, 0.002, 0.003, 0.004, 0.005],
        'noise_std': [0.05, 0.1],
        'actor_lr': [0.0001, 0.0003, 0.0006, 0.001],
        'critic_lr': [0.0001, 0.0003, 0.0006, 0.001],
        'lr_decay_rate': [0.995, 1.0],
        'max_iteration': [500]
    }
    
    # 모든 하이퍼파라미터 조합 생성
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    # 결과 기록을 위한 리스트
    results = []
    
    # 각 하이퍼파라미터 조합으로 학습 진행
    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        print(f"\n===== Combination {idx+1}/{len(combinations)} =====")
        print(f"Parameters: {params}")
        
        # 하이퍼파라미터 설정
        batch_size = params['batch_size']
        gamma = params['gamma']
        tau = params['tau']
        noise_std = params['noise_std']
        actor_lr = params['actor_lr']
        critic_lr = params['critic_lr']
        lr_decay_rate = params['lr_decay_rate']
        max_iteration = params['max_iteration']
        
        # 에이전트 초기화
        agent = DDPGAgent(
            batch_size=batch_size, 
            gamma=gamma, 
            tau=tau, 
            noise_std=noise_std, 
            actor_lr=actor_lr, 
            critic_lr=critic_lr, 
            lr_decay_rate=lr_decay_rate
        )
        
        # 전문가 데이터로 버퍼 채우기
        for sample in expert_data:
            if sample["state"] is None or sample["next_state"] is None:
                continue
            if not isinstance(sample["state"], (list, np.ndarray)) or not isinstance(sample["next_state"], (list, np.ndarray)):
                continue

            state = np.array(sample['state']).flatten()
            next_state = np.array(sample['next_state']).flatten()
            action = np.array(sample['action']).flatten()
            reward = sample['reward']
            done = sample['done']

            agent.remember(state, action, reward, next_state, done)
        
        # 학습 시작
        start_time = time.time()
        loss_log = []
        best_loss = float("inf")
        best_iteration = 0
        
        # 최적 모델 상태 저장을 위한 변수
        best_actor_state = None
        best_critic_state = None
        best_actor_target_state = None
        best_critic_target_state = None
        
        for iteration in range(max_iteration):
            agent.train()
            
            if iteration % 10 == 0 and len(agent.buffer) >= agent.batch_size:
                # 손실 측정
                s, a, r, ns, d = agent.buffer.sample(agent.batch_size)
                
                s = torch.FloatTensor(s).to(agent.device)
                a = torch.FloatTensor(a).to(agent.device)
                r = torch.FloatTensor(r).unsqueeze(1).to(agent.device)
                ns = torch.FloatTensor(ns).to(agent.device)
                d = torch.FloatTensor(d).unsqueeze(1).to(agent.device)
                
                with torch.no_grad():
                    next_a = agent.actor_target(ns)
                    target_q = r + (1 - d) * agent.gamma * agent.critic_target(ns, next_a)
                q = agent.critic(s, a)
                loss = torch.nn.functional.mse_loss(q, target_q)
                
                avg_loss = loss.item()
                loss_log.append(avg_loss)
                
                # best loss 업데이트 및 모델 상태 저장
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_iteration = iteration
                    
                    # 최적 모델 상태 저장
                    best_actor_state = copy.deepcopy(agent.actor.state_dict())
                    best_critic_state = copy.deepcopy(agent.critic.state_dict())
                    best_actor_target_state = copy.deepcopy(agent.actor_target.state_dict())
                    best_critic_target_state = copy.deepcopy(agent.critic_target.state_dict())
                
                if iteration % 100 == 0:
                    print(f"[{iteration}/{max_iteration}] Critic Loss: {avg_loss:.6f}")
        
        training_time = time.time() - start_time
        
        # 학습 결과 기록 (최적 모델 상태 포함)
        run_result = {
            'parameters': {p: float(v) if isinstance(v, (int, float)) else v for p, v in params.items()},
            'best_loss': float(best_loss),
            'best_iteration': best_iteration,
            'final_loss': float(loss_log[-1]) if loss_log else None,
            'loss_history': [float(l) for l in loss_log],
            'training_time': training_time,
            # 최적 모델 상태 저장
            'best_model_states': {
                'actor': best_actor_state,
                'critic': best_critic_state,
                'actor_target': best_actor_target_state,
                'critic_target': best_critic_target_state
            }
        }
        
        results.append(run_result)
        print(f"Run complete. Best loss: {best_loss:.6f} at iteration {best_iteration}, training time: {training_time:.2f} seconds")
    
    # 최종 결과 분석
    print("\n===== Grid Search Complete =====")
    
    # 손실 기준으로 정렬
    sorted_results = sorted(results, key=lambda x: x['best_loss'])
    
    # 결과 저장을 위한 버전 (모델 상태는 제외)
    results_to_save = []
    for result in sorted_results:
        result_copy = result.copy()
        result_copy.pop('best_model_states', None)  # 모델 상태 제외
        results_to_save.append(result_copy)
    
    # 결과 JSON으로 저장
    results_path = os.path.join(save_dir, "grid_search_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    
    # 상위 10개 모델 저장
    top_10_results = sorted_results[:10]
    for idx, result in enumerate(top_10_results):
        print(f"\nTop {idx+1}: Best Loss: {result['best_loss']:.6f} at iteration {result['best_iteration']}")
        for param, value in result['parameters'].items():
            print(f"   {param}: {value}")
            
        # 모델 객체 생성
        params = result['parameters']
        agent = DDPGAgent(
            batch_size=int(params['batch_size']), 
            gamma=params['gamma'], 
            tau=params['tau'], 
            noise_std=params['noise_std'], 
            actor_lr=params['actor_lr'], 
            critic_lr=params['critic_lr'], 
            lr_decay_rate=params['lr_decay_rate']
        )
        
        # 최적 상태 불러오기
        best_states = result['best_model_states']
        agent.actor.load_state_dict(best_states['actor'])
        agent.critic.load_state_dict(best_states['critic'])
        agent.actor_target.load_state_dict(best_states['actor_target'])
        agent.critic_target.load_state_dict(best_states['critic_target'])
        
        # 모델 저장
        model_filename = f"ddpg_top{idx+1}_loss_{result['best_loss']:.4f}.pth"
        model_path = os.path.join(save_dir, model_filename)
        agent.save_model(model_path)
        print(f"   Model saved to: {model_filename}")
    
    print(f"\n모든 결과가 {results_path}에 저장되었습니다.")

if __name__ == "__main__":
    pickle_path = "/home/gr/iris_tracker_PID_RL/expert_data_0429_100000sample.pkl"
    save_dir = "./ddpg_grid_search"     # 결과 저장 경로

    offline_train_from_pickle_ddpg(pickle_path)
    # offline_train_grid_search_ddpg(pickle_path, save_dir)