# weight_importance.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.append('/home/gr/iris_tracker_PID_RL/src/RL/RL')
from Decoupled_DDPGAgent import DDPGAgent

def plot_input_importance(agent, state_names, title):
    """rm 등을 사용할 수 있음
    importances = np.abs(W).sum(axis=0)
    agent: DDPGAgent 인스턴스 (load_model() 이미 호출된 상태)
    state_names: 입력 변수 이름 리스트 (len == agent.state_dim)
    title: 그래프 타이틀
    """
    # fc1.weight shape: (hidden_units, state_dim)
    W = agent.actor.fc1.weight.data.cpu().numpy()
    # 각 입력 특징의 중요도: 절댓값 합 (L1 norm) 또는 L2 norm 등을 사용할 수 있음
    importances = np.abs(W).sum(axis=0)
    importances /= importances.sum()

    plt.figure(figsize=(6,4))
    plt.bar(state_names, importances)
    plt.title(title)
    plt.ylabel('Normalized Importance')
    plt.xlabel('State Variable')
    plt.tight_layout()
    plt.show()

def generate_state_names(mode, history_length):
    if mode == 'vertical':
        names = [f"error_ver_t-{i}" for i in reversed(range(history_length))] + ["last_action_fwd"]
    elif mode == 'forward':
        names = [f"error_fwd_t-{i}" for i in reversed(range(history_length))] + ["last_action_ver"]
    elif mode == 'lateral':
        names = [f"error_lat_t-{i}" for i in reversed(range(history_length))]
    else:
        names = [f"state_t-{i}" for i in reversed(range(history_length))]
    return names

if __name__ == '__main__':
    # 모델 파일이 저장된 디렉토리 경로를 설정하세요
    history_length = 2
    checkpoint_dir = '/home/gr/iris_tracker_PID_RL/results/0606_2107_history2'

    # 1) Vertical Agent
    ver_agent = DDPGAgent(mode='vertical', history_length=history_length)
    ver_agent.load_model(os.path.join(checkpoint_dir, 'RL_model_timeout_vertical.pth'))
    state_names_ver = generate_state_names('vertical', history_length)
    plot_input_importance(ver_agent, state_names_ver, 'Vertical Agent Input Importance')

    # 2) Forward Agent
    fwd_agent = DDPGAgent(mode='forward', history_length=history_length)
    fwd_agent.load_model(os.path.join(checkpoint_dir, 'RL_model_timeout_forward.pth'))
    state_names_fwd = generate_state_names('forward', history_length)
    plot_input_importance(fwd_agent, state_names_fwd, 'Forward Agent Input Importance')

    # 3) Lateral Agent
    lat_agent = DDPGAgent(mode='lateral', history_length=history_length)
    lat_agent.load_model(os.path.join(checkpoint_dir, 'RL_model_timeout_lateral.pth'))
    state_names_lat = generate_state_names('lateral', history_length)
    plot_input_importance(lat_agent, state_names_lat, 'Lateral Agent Input Importance')
