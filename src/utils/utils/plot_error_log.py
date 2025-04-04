import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv_path = "/home/gr/iris_tracker_PID_RL/error_log_pixel_6m.csv"
df = pd.read_csv(csv_path)
# df.columns = ["Time (s)", "error_lateral", "error_vertical", "error_forward", "real_forward_error"]

plt.figure(figsize=(8, 4))

plt.plot(df["Time (s)"], df["error_forward"], label="Forward Error")
if "real_forward_error" in df.columns:
    real_error_adjusted = df["real_forward_error"]
    plt.plot(df["Time (s)"], real_error_adjusted, label="Real Forward Error", linestyle="--")
    rmse = np.sqrt(np.mean((df["error_forward"] - real_error_adjusted)**2))

plt.title(f"Tracking Errors Over Time\nmode: pixel, disired distance: 6m")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(rmse)