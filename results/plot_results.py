import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 경로
csv_path = '/home/gr/iris_tracker_PID_RL/results/0529_2053/step_log.csv'
df = pd.read_csv(csv_path)

# Episode별 누적 step 계산
episodes = df['episode'].unique()
accumulated_env_steps = []
accumulated_train_steps = []
current_env_step = 0
current_train_step = 0

for episode in episodes:
    episode_data = df[df['episode'] == episode]
    episode_env_steps = episode_data['env_step'].values
    episode_train_steps = episode_data['train_step'].values

    accumulated_episode_env_steps = episode_env_steps + current_env_step
    accumulated_episode_train_steps = episode_train_steps + current_train_step

    accumulated_env_steps.extend(accumulated_episode_env_steps)
    accumulated_train_steps.extend(accumulated_episode_train_steps)

    current_env_step = accumulated_episode_env_steps[-1]
    current_train_step = accumulated_episode_train_steps[-1]

df['accumulated_env_step'] = accumulated_env_steps
df['accumulated_train_step'] = accumulated_train_steps

# Plot 설정
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
directions = ['ver_reward', 'fwd_reward', 'lat_reward']

for i, direction in enumerate(directions):
    # env_step 기준
    axs[i, 0].plot(df['accumulated_env_step'], df[f'{direction}(avg100env)'], label=f'{direction} (avg100env)')
    axs[i, 0].plot(df['accumulated_env_step'], df[f'{direction}(episode)'], label=f'{direction} (episode)')
    axs[i, 0].set_title(f'{direction} - Environment Step')
    axs[i, 0].set_xlabel('Accumulated Environment Steps')
    axs[i, 0].set_ylabel('Reward')
    axs[i, 0].legend()
    axs[i, 0].grid(alpha=0.3)

    # train_step 기준
    axs[i, 1].plot(df['accumulated_train_step'], df[f'{direction}(avg100env)'], label=f'{direction} (avg100env)')
    axs[i, 1].plot(df['accumulated_train_step'], df[f'{direction}(episode)'], label=f'{direction} (episode)')
    axs[i, 1].set_title(f'{direction} - Training Step')
    axs[i, 1].set_xlabel('Accumulated Training Steps')
    axs[i, 1].set_ylabel('Reward')
    axs[i, 1].legend()
    axs[i, 1].grid(alpha=0.3)

    # 에피소드 끝마다 세로선 추가
    for episode in episodes:
        episode_end_env = df[df['episode'] == episode]['accumulated_env_step'].max()
        episode_end_train = df[df['episode'] == episode]['accumulated_train_step'].max()

        axs[i, 0].axvline(x=episode_end_env, color='gray', linestyle='--', alpha=0.5)
        axs[i, 1].axvline(x=episode_end_train, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

