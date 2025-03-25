import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import csv
from scipy.optimize import curve_fit

def main():
    parser = argparse.ArgumentParser(description="Plot distance-area data or pixel-count data, with optional curve fitting.")
    parser.add_argument('--mode', choices=['bbox', 'pixel'], default='bbox',
                         help="Choose 'bbox' to read bbox_area_log.csv or 'pixel' to read pixel_count_log.csv.")
    args = parser.parse_args()

    if args.mode == 'bbox':
        csv_path = "/home/gr/iris_tracker_PID_RL/data/bbox_area_log.csv"
        df = pd.read_csv(csv_path)
        time = np.array(df["Time (s)"])
        value = np.array(df["Filtered BBox Area"])
    else:
        csv_path = "/home/gr/iris_tracker_PID_RL/data/pixel_count_log.csv"
        df = pd.read_csv(csv_path)
        time = np.array(df["Time (s)"]) - 1742900795
        value = np.array(df["Pixel count"])

    # subplot 3개 (좌측: 시간 기반, 중앙: Exp+Const, 우측: Inverse Square)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # --------------------- 좌측 subplot: Time-based Plot --------------------- #
    ax1 = axs[0]
    if args.mode == 'bbox':
        ax1.plot(time, value, label="Filtered BBox Area", color="red")

        # 구간별 참고선
        x_ticks = np.arange(21.5, 121.5, 10)
        y_values = np.array([33000, 13500, 7000, 4400, 3000, 2500, 2150, 2000, 1700, 1600])
        labels = ["Distance:     "] + [f"{i}m" for i in range(1, 11)]
        for x, label in zip(x_ticks, labels):
            ax1.axvline(x=x, color='black', linestyle='--', alpha=0.6)
            ax1.text(x - 5, max(value) * 0.95, label, color='black', fontsize=7, ha='center')
        for x_start, y in zip(x_ticks, y_values):
            x_end = x_start + 10
            ax1.hlines(y=y, xmin=x_start, xmax=x_end, color='black', linestyle=':', alpha=0.7)
            ax1.text(x_end, y - 2000, str(y), color='black', fontsize=7, va='center', ha='right')

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Filtered BBox Area")  
        ax1.set_title("Bounding Box Area Over Time")  # 수정: Over Distance -> Over Time
        ax1.grid(False)

    else:  # pixel 모드
        ax1.plot(time, value, label="Pixel Count", color="red")

        # 구간별 참고선
        x_ticks = np.arange(0, 121.5, 10)
        y_values = np.array([7000, 1500, 730, 480, 290, 205, 155, 120, 105, 90])
        labels = ["Distance:     "] + [f"{i}m" for i in range(1, 11)]
        for x, label in zip(x_ticks, labels):
            ax1.axvline(x=x, color='black', linestyle='--', alpha=0.6)
            ax1.text(x - 5, max(value) * 0.95, label, color='black', fontsize=7, ha='center')
        for x_start, y in zip(x_ticks, y_values):
            x_end = x_start + 10
            ax1.hlines(y=y, xmin=x_start, xmax=x_end, color='black', linestyle=':', alpha=0.7)
            ax1.text(x_end - 2, y + 500, str(y), color='black', fontsize=7, va='center', ha='right')

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Pixel Count")
        ax1.set_title("Pixel Count Over Time")
        ax1.grid(False)

    # --------------------- 중앙 subplot: Exponential + Constant Fitting --------------------- #
    distances = np.arange(1, 11)

    def exp_const(d, a, b, c):
        return a * np.exp(b * d) + c

    if args.mode == 'bbox':
        # bbox 모드일 때 y_values는 Bounding Box Area
        y_values = np.array([33000, 13500, 7000, 4400, 3000, 2500, 2150, 2000, 1700, 1600])
    else:
        # pixel 모드일 때 y_values는 Pixel Count
        y_values = np.array([7000, 1500, 730, 480, 290, 205, 155, 120, 105, 90])

    initial_guess = [(y_values[0] - y_values[-1]), -0.5, y_values[-1]]
    params, cov = curve_fit(exp_const, distances, y_values, p0=initial_guess)
    a_fit, b_fit, c_fit = params

    # RMSE 계산
    y_pred = exp_const(distances, a_fit, b_fit, c_fit)
    rmse = np.sqrt(np.mean((y_values - y_pred) ** 2))

    d_fit = np.linspace(1, 10, 100)
    y_fit = exp_const(d_fit, a_fit, b_fit, c_fit)

    ax2 = axs[1]
    ax2.plot(d_fit, y_fit, color='blue', label='Exp+Const Fit')
    ax2.scatter(distances, y_values, color='green', label='Data Points')
    ax2.set_xlabel("Distance (m)")
    if args.mode == 'bbox':
        ax2.set_ylabel("Filtered BBox Area")
        ax2.set_title("Fitting: BBox Area vs Distance\n(Exponential + Constant Model)")
    else:
        ax2.set_ylabel("Pixel Count")
        ax2.set_title("Fitting: Pixel Count vs Distance\n(Exponential + Constant Model)")
    ax2.legend()
    ax2.grid(True)

    eq_text = f"f(d) = {a_fit:.2f} exp({b_fit:.2f} d) + {c_fit:.2f}\n\nRMSE = {rmse:.2f}"
    if args.mode == 'bbox':
        ax2.text(6.5, 20000, eq_text, fontsize=10, color='black', ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    else:
        ax2.text(6.5, 4000, eq_text, fontsize=10, color='black', ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # --------------------- 우측 subplot: Inverse Square Fitting --------------------- #
    def inv_sq(d, a):
        return a / (d**2)

    initial_guess_inv = [y_values[2] * 9]  # d=2에서의 초기 추정
    params_inv, cov_inv = curve_fit(inv_sq, distances, y_values, p0=initial_guess_inv)
    a_fit_inv = params_inv[0]

    y_pred_inv = inv_sq(distances, a_fit_inv)
    rmse_inv = np.sqrt(np.mean((y_values - y_pred_inv) ** 2))

    d_fit_inv = np.linspace(1, 10, 100)
    y_fit_inv = inv_sq(d_fit_inv, a_fit_inv)

    ax3 = axs[2]
    ax3.plot(d_fit_inv, y_fit_inv, color='blue', label='Inv Sq Fit')
    ax3.scatter(distances, y_values, color='green', label='Data Points')
    ax3.set_xlabel("Distance (m)")
    if args.mode == 'bbox':
        ax3.set_ylabel("Filtered BBox Area")
        ax3.set_title("Fitting: BBox Area vs Distance\n(Inverse Square Model)")
    else:
        ax3.set_ylabel("Pixel Count")
        ax3.set_title("Fitting: Pixel Count vs Distance\n(Inverse Square Model)")
    ax3.legend()
    ax3.grid(True)

    eq_text_inv = f"f(d) = {a_fit_inv:.2f} / d²\n\nRMSE = {rmse_inv:.2f}"
    if args.mode == 'bbox':
        ax3.text(6.5, 20000, eq_text_inv, fontsize=10, color='black', ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    else:
        ax3.text(6.5, 4000, eq_text_inv, fontsize=10, color='black', ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
