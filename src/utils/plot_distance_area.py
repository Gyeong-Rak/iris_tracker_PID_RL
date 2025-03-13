import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# CSV 파일 로드 및 데이터 변환
csv_path = "/home/gr/RL_controller/data/bbox_area_log.csv"
df = pd.read_csv(csv_path)
time_data = np.array(df["Time (s)"])
area_data = np.array(df["Filtered BBox Area"])

# subplot 3개 (좌측: 시간 기반, 중앙: Exp+Const, 우측: Inverse Square)
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# --------------------- 좌측 subplot: Time-based Plot --------------------- #
ax1 = axs[0]
ax1.plot(time_data, area_data, label="Filtered BBox Area", color="red")

# 시간 기반 세로선과 가로선
x_ticks = np.arange(21.5, 121.5, 10)  # 각 구간의 시작 시간
y_values = np.array([33000, 13500, 7000, 4400, 3000, 2500, 2150, 2000, 1700, 1600])
labels = ["Distance:     "] + [f"{i}m" for i in range(1, 11)]
for x, label in zip(x_ticks, labels):
    ax1.axvline(x=x, color='black', linestyle='--', alpha=0.6)
    ax1.text(x - 5, max(area_data) * 0.95, label, color='black', fontsize=7, ha='center')
for x_start, y in zip(x_ticks, y_values):
    x_end = x_start + 10
    ax1.hlines(y=y, xmin=x_start, xmax=x_end, color='black', linestyle=':', alpha=0.7)
    ax1.text(x_end, y - 2000, str(y), color='black', fontsize=7, va='center', ha='right')
    
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Filtered BBox Area")
ax1.set_title("Bounding Box Area Over Time")
ax1.grid(False)

# --------------------- 중앙 subplot: Exponential + Constant Fitting --------------------- #
# 거리 데이터: 1m ~ 10m
distances = np.arange(1, 11)

# 모델 1: f(d) = a * exp(b * d) + c
def exp_const(d, a, b, c):
    return a * np.exp(b * d) + c

initial_guess = [(y_values[0] - y_values[-1]), -0.5, y_values[-1]]
params, cov = curve_fit(exp_const, distances, y_values, p0=initial_guess)
a_fit, b_fit, c_fit = params

# RMSE 계산
y_pred = exp_const(distances, a_fit, b_fit, c_fit)
rmse = np.sqrt(np.mean((y_values - y_pred)**2))

# 피팅 곡선을 그리기 위한 x값 생성
d_fit = np.linspace(1, 10, 100)
y_fit = exp_const(d_fit, a_fit, b_fit, c_fit)

ax2 = axs[1]
ax2.plot(d_fit, y_fit, color='blue', label='Exp+Const Fit')
ax2.scatter(distances, y_values, color='green', label='Data Points')
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("Filtered BBox Area")
ax2.set_title("Fitting: BBox Area vs Distance\n(Exponential + Constant Model)")
ax2.legend()
ax2.grid(True)

# 피팅 수식과 RMSE를 텍스트로 추가
eq_text = f"f(d) = {a_fit:.2f} exp({b_fit:.2f} d) + {c_fit:.2f}\n\nRMSE = {rmse:.2f}"
ax2.text(6.5, 20000, eq_text, fontsize=10, color='black', ha='center', va='center',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# --------------------- 우측 subplot: Inverse Square Fitting --------------------- #
# 모델 2: f(d) = a / d^2
def inv_sq(d, a):
    return a / (d**2)

initial_guess_inv = [y_values[2] * 9]  # d=2에서 a ~ f(2)*4
params_inv, cov_inv = curve_fit(inv_sq, distances, y_values, p0=initial_guess_inv)
a_fit_inv = params_inv[0]

# RMSE 계산
y_pred_inv = inv_sq(distances, a_fit_inv)
rmse_inv = np.sqrt(np.mean((y_values - y_pred_inv)**2))

# 피팅 곡선을 그리기 위한 x값 생성
d_fit_inv = np.linspace(1, 10, 100)
y_fit_inv = inv_sq(d_fit_inv, a_fit_inv)

ax3 = axs[2]
ax3.plot(d_fit_inv, y_fit_inv, color='blue', label='Inv Sq Fit')
ax3.scatter(distances, y_values, color='green', label='Data Points')
ax3.set_xlabel("Distance (m)")
ax3.set_ylabel("Filtered BBox Area")
ax3.set_title("Fitting: BBox Area vs Distance\n(Inverse Square Model)")
ax3.legend()
ax3.grid(True)

# 피팅 수식과 RMSE 텍스트 추가
eq_text_inv = f"f(d) = {a_fit_inv:.2f} / d²\n\nRMSE = {rmse_inv:.2f}"
ax3.text(6.5, 20000, eq_text_inv, fontsize=10, color='black', ha='center', va='center',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()
plt.show()
