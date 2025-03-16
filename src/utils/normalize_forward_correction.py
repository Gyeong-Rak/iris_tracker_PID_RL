import numpy as np
import math
import matplotlib.pyplot as plt

# Define functions
def distance2area(distance):
    return 79162.49 * np.exp(-0.94 * distance) + 2034.21

def normalized_forward_error(distance, forward_error):
    dA_dd = -0.94 * 79162.49 * np.exp(-0.94 * distance)  # Derivative of distance2area
    normalized_error = -forward_error / dA_dd  # Normalize forward error
    return normalized_error

# Generate test data
desired_distance = 4
desired_bbox_area = distance2area(desired_distance)
distances = np.linspace(1, 10, 100)
bbox_areas = distance2area(distances)
forward_error = desired_bbox_area - bbox_areas

# Compute normalized errors
normalized_errors = [normalized_forward_error(distance, error) for distance, error in zip(distances, forward_error)]

# Plot results
plt.figure(figsize=(10, 5))

# Plot Forward Error vs. Distance
plt.subplot(1, 2, 1)
plt.plot(distances, forward_error, label="Forward Error")
plt.xlabel("Distance (m)")
plt.ylabel("Forward Error")
plt.title("Forward Error vs. Distance")
plt.legend()
plt.grid()

# Plot Normalized Forward Error vs. Distance
plt.subplot(1, 2, 2)
plt.plot(distances, normalized_errors, label="Normalized Forward Error", color="r")
plt.xlabel("Distance (m)")
plt.ylabel("Normalized Forward Error")
plt.title("Normalized Forward Error vs. Distance")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
