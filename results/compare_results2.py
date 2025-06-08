#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
n-개의 DDPG 로그(step_log.csv)를 한 번에 비교 플롯하는 스크립트
- avg100env         : 연한 선 (먼저 그려짐)
- total_avg         : 진한 선 (앞에 표시됨)
- legend는 실험 단위로 한 번만
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1) 비교할 실험 로그 폴더
# ---------------------------------------------------------------------------
log_paths = [
    '/home/gr/iris_tracker_PID_RL/results/0606_2107_his2',
    '/home/gr/iris_tracker_PID_RL/results/0607_1807_his2_wo_inter',
    '/home/gr/iris_tracker_PID_RL/results/0607_2251_his2_wo_fwd_inter'
]

# ---------------------------------------------------------------------------
# 2) 로그 로드
# ---------------------------------------------------------------------------
dfs          = []
valid_paths  = []
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
# 3) 누적 step 계산
# ---------------------------------------------------------------------------
def add_accumulated_step(df, src_col: str, dst_col: str) -> None:
    episodes   = df['episode'].unique()
    accumulated, cur = [], 0
    for ep in episodes:
        seg = df[df['episode'] == ep][src_col] + cur
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
fig, axs = plt.subplots(len(directions), 1, figsize=(10, 12))
if len(directions) == 1:
    axs = [axs]

for i, d in enumerate(directions):
    ax = axs[i]
    for idx, (df, lbl) in enumerate(zip(dfs, valid_paths)):
        x = df['accumulated_train_step']
        y_avg100 = df[f'{d}(avg100env)']
        y_total  = df[f'{d}_total_avg']
        tag = os.path.basename(lbl)
        color = f"C{idx}"

        # 연한 선 먼저, 진한 선 나중에
        ax.plot(x, y_avg100, color=color, alpha=0.3, linewidth=2.0, zorder=1)
        ax.plot(x, y_total,  color=color, linewidth=2.0, label=tag, zorder=2)

    ax.set(title=d, xlabel='Accumulated Training Steps', ylabel='Reward')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
