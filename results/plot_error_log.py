import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# 여러 CSV 파일 경로 설정
csv_paths = [
    '/home/gr/iris_tracker_PID_RL/results/error_logs/error_log_0608_2030_PID.csv',
    '/home/gr/iris_tracker_PID_RL/results/error_logs/error_log_0608_2038_0606_2107_his2.csv',
    '/home/gr/iris_tracker_PID_RL/results/error_logs/error_log_0608_2101_0605_2237_his3.csv',
    '/home/gr/iris_tracker_PID_RL/results/error_logs/error_log_0608_2107_0607_1506_his1.csv',
    '/home/gr/iris_tracker_PID_RL/results/error_logs/error_log_0608_2122_0607_2251_his2_wo_fwd_inter.csv',
    '/home/gr/iris_tracker_PID_RL/results/error_logs/error_log_0608_2146_0607_1807_his2_wo_inter.csv',
    '/home/gr/iris_tracker_PID_RL/results/error_logs/error_log_0608_2200_ver_0607_1807_his2_wo_inter_fwd_0606_2107_his2_lat_0607_2251_his2_wo_fwd_inter.csv'
]

num_files = len(csv_paths)
cols = 2
rows = math.ceil(num_files / cols)

fig, axs = plt.subplots(rows, cols, figsize=(10, 3 * rows), squeeze=False)

for idx, path in enumerate(csv_paths):
    row, col = divmod(idx, cols)
    ax = axs[row][col]

    df = pd.read_csv(path)
    label_prefix = path.split('/')[-1].replace('.csv', '')

    ax.plot(df['env_step'], df['error_vertical'], label='Vertical Error', linewidth=1, alpha=0.7)
    ax.plot(df['env_step'], df['error_forward'], label='Forward Error', linewidth=1, alpha=0.7)
    ax.plot(df['env_step'], df['error_lateral'], label='Lateral Error', linewidth=1, alpha=0.7)

    ax.set_title(f'Tracking Error - {label_prefix}')
    ax.set_xlabel("Env Step")
    ax.set_ylabel("Error")
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks(np.arange(-1.0, 1.1, 0.5))
    ax.grid(True, which='both', axis='y')
    ax.legend(fontsize='small', loc='lower left')

    # 평균 및 분산 계산 및 텍스트 표시
    stats_text = '\n'.join([
        f"Vertical: μ={df['error_vertical'].mean():.3f}, σ={df['error_vertical'].std():.3f}",
        f"Forward : μ={df['error_forward'].mean():.3f}, σ={df['error_forward'].std():.3f}",
        f"Lateral : μ={df['error_lateral'].mean():.3f}, σ={df['error_lateral'].std():.3f}",
    ])
    ax.text(1.0, 1.0, stats_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# 빈 subplot 숨기기 (파일 수가 홀수일 경우)
for i in range(num_files, rows * cols):
    fig.delaxes(axs[i // cols][i % cols])

plt.tight_layout()
plt.show()
