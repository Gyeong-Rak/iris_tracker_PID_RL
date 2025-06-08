#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
n-개의 DDPG 로그(step_log.csv) 를 한 번에 비교 플롯하는 스크립트
- avg100env         : 최근 100 env-step 평균 리워드
- total_avg         : 0 시점부터 지금까지 누적 리워드 / 누적 env-step
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1) 비교할 실험 로그 폴더 (step_log.csv 가 각 폴더 안에 있어야 함)
# ---------------------------------------------------------------------------
log_paths = [
    # '/home/gr/iris_tracker_PID_RL/results/0530_1702_same',
    # '/home/gr/iris_tracker_PID_RL/results/0530_1855_history4_lr_small',
    # '/home/gr/iris_tracker_PID_RL/results/0530_2117_history3',
    # '/home/gr/iris_tracker_PID_RL/results/0602_1340_3_dir_lr_same_1e-3',
    # '/home/gr/iris_tracker_PID_RL/results/0602_1802_lr_decay_5e-4',
    # '/home/gr/iris_tracker_PID_RL/results/0602_1958_lr_decay_1e-4',
    # '/home/gr/iris_tracker_PID_RL/results/0603_1522_lr_decay_0',
    # '/home/gr/iris_tracker_PID_RL/results/0603_1727_no_ver_fwd_interaction',
    # '/home/gr/iris_tracker_PID_RL/results/0603_1923_same_with_1727',
    # '/home/gr/iris_tracker_PID_RL/results/0603_2119_same_with_1802',
    # '/home/gr/iris_tracker_PID_RL/results/0604_1358_same_with_1802',
    # '/home/gr/iris_tracker_PID_RL/results/0604_1702_fwd_std_0.1',
    # '/home/gr/iris_tracker_PID_RL/results/0604_2138_new_reward',
    # '/home/gr/iris_tracker_PID_RL/results/training_logs/0605_1646_his3',
    # '/home/gr/iris_tracker_PID_RL/results/training_logs/0605_1833_wo_inter',
    # '/home/gr/iris_tracker_PID_RL/results/training_logs/0605_2035_his3_wo_inter',
    '/home/gr/iris_tracker_PID_RL/results/training_logs/0605_2237_his3',
    # '/home/gr/iris_tracker_PID_RL/results/training_logs/0606_1355_his3_wo_inter',
    # '/home/gr/iris_tracker_PID_RL/results/training_logs/0606_1739_his3_wo_fwd_inter',
    '/home/gr/iris_tracker_PID_RL/results/training_logs/0606_2107_his2',
    '/home/gr/iris_tracker_PID_RL/results/training_logs/0607_1506_his1',
    '/home/gr/iris_tracker_PID_RL/results/training_logs/0607_1807_his2_wo_inter',
    '/home/gr/iris_tracker_PID_RL/results/training_logs/0607_2251_his2_wo_fwd_inter'
]

# ---------------------------------------------------------------------------
# 2) 로그 로드 (파일 없는 폴더는 자동 스킵)
# ---------------------------------------------------------------------------
dfs          = []   # DataFrame 목록
valid_paths  = []   # 실존하는 경로(=플롯 레이블)
for p in log_paths:
    csv_path = os.path.join(p, 'step_log.csv')
    if os.path.isfile(csv_path):
        dfs.append(pd.read_csv(csv_path))
        valid_paths.append(p)
    else:
        print(f'⚠️  {csv_path} not found → skip')

if not dfs:
    raise RuntimeError("step_log.csv 를 찾지 못했습니다.")

# ---------------------------------------------------------------------------
# 3) 누적 env/train step 컬럼 추가 함수
# ---------------------------------------------------------------------------
def add_accumulated_step(df, src_col: str, dst_col: str) -> None:
    episodes   = df['episode'].unique()
    accumulated, cur = [], 0
    for ep in episodes:
        seg      = df[df['episode'] == ep][src_col] + cur
        accumulated.extend(seg)
        cur = seg.iloc[-1]
    df[dst_col] = accumulated

for df in dfs:
    add_accumulated_step(df, 'train_step', 'accumulated_train_step')
    add_accumulated_step(df, 'env_step',   'accumulated_env_step')

# ---------------------------------------------------------------------------
# 4) 보상 누적 합 / 전체 평균 계산
# ---------------------------------------------------------------------------
directions = ['ver_reward', 'fwd_reward', 'lat_reward']
for df in dfs:
    env_step_delta = df['accumulated_env_step'].diff().fillna(df['accumulated_env_step'])
    for d in directions:
        col_avg = f'{d}(avg100env)'
        df[f'{d}_reward_cumsum'] = (df[col_avg] * env_step_delta).cumsum()
        df[f'{d}_total_avg']     = df[f'{d}_reward_cumsum'] / df['accumulated_env_step']

# ---------------------------------------------------------------------------
# 5) 플롯
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(len(directions), 2, figsize=(16, 16))

for i, d in enumerate(directions):
    # (a) 최근 100 env-step 평균
    for df, lbl in zip(dfs, valid_paths):
        axs[i, 0].plot(df['accumulated_train_step'], df[f'{d}(avg100env)'],
                       label=lbl, alpha=0.7)
    axs[i, 0].set(title=f'{d} – avg100env',
                  xlabel='Accumulated Training Steps',
                  ylabel='Reward')
    axs[i, 0].grid(alpha=0.6)

    # (b) 전체 평균(total_avg)
    for df, lbl in zip(dfs, valid_paths):
        axs[i, 1].plot(df['accumulated_train_step'], df[f'{d}_total_avg'],
                       label=lbl, alpha=0.7)
    axs[i, 1].set(title=f'{d} – total avg',
                  xlabel='Accumulated Training Steps',
                  ylabel='Reward')
    axs[i, 1].grid(alpha=0.3)

# 범례 한 번에 추가
for ax_pair in axs:
    ax_pair[0].legend(fontsize=8)
    ax_pair[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
