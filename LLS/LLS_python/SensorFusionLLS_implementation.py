import numpy as np
import matplotlib.pyplot as plt

class SensorFusionLLS:
    def __init__(self, initial_window=20, time_step=0.025, default_value=100.0):
        self.time_step = time_step  
        self.default_value = default_value
        self.data = []  # Expanding buffer for fused sensor data
        self.time_data = []  # Expanding buffer for time indices
        self.initial_window = initial_window  

    def reset(self):
        """ Reset the filter by clearing the buffers. """
        self.data.clear()
        self.time_data.clear()

    def update(self, u, enable=True, reset=False):
        """
        Update the model with a new fused sensor value and fit a regression line.
        """
        if reset:
            self.reset()
            return self.default_value

        if not enable:
            return self.default_value

        # Append new fused data point
        self.data.append(u)
        self.time_data.append(len(self.data) * self.time_step)

        if len(self.data) < self.initial_window:  
            return u  # Not enough data to fit regression

        # Convert to numpy arrays
        t_array = np.array(self.time_data, dtype=np.float32)
        u_array = np.array(self.data, dtype=np.float32)

        # Compute sums
        N = len(t_array)
        sum_t = np.sum(t_array)
        sum_t2 = np.sum(t_array ** 2)
        sum_u = np.sum(u_array)
        sum_ut = np.sum(t_array * u_array)

        # Compute determinant
        determinant = (N * sum_t2) - (sum_t * sum_t)

        if abs(determinant) < 1e-6:  
            return self.default_value  # Avoid singular matrix issues

        # Solve normal equations for slope & offset
        offset = ((sum_t2 * sum_u) - (sum_t * sum_ut)) / determinant
        slope = ((-sum_t * sum_u) + (N * sum_ut)) / determinant

        # Predict next value (one time step ahead)
        next_time = (N + 1) * self.time_step
        predicted_value = offset + (slope * next_time)

        return predicted_value

# Simulated Sensor Data (Fusion of LiDAR and Radar Measurements)
np.random.seed(42)
time_steps = np.arange(0, 19, 1)  # Time steps (0 to 20)

# Simulated Sensor Measurements
lidar_distances = 50 - 0.3 * time_steps + np.random.normal(scale=1, size=len(time_steps))  # LiDAR readings
radar_distances = 50 - 0.3 * time_steps + np.random.normal(scale=2, size=len(time_steps))  # Radar readings

# Sensor Fusion: Weighted Average of LiDAR and Radar
fused_distances = 0.7 * lidar_distances + 0.3 * radar_distances  # More weight on LiDAR

# Initialize LLS Filter (Expanding Window)
lls_filter = SensorFusionLLS(initial_window=20, time_step=0.025, default_value=50.0)

# Store results for plotting
estimated_distances = []

# Process data with expanding window LLS
for t, u in zip(time_steps, fused_distances):
    estimated = lls_filter.update(u, enable=True, reset=False)
    estimated_distances.append(estimated)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(time_steps, lidar_distances, 'g.', markersize=5, label="LiDAR Readings")  # Green = LiDAR data
plt.plot(time_steps, radar_distances, 'b.', markersize=5, label="Radar Readings")  # Blue = Radar data
plt.plot(time_steps, fused_distances, 'ko', markersize=5, label="Fused Data")  # Black = Sensor Fusion
plt.plot(time_steps, estimated_distances, 'r-', linewidth=2, label="LLS Predicted")  # Red = LLS Estimate
plt.xlabel("Time Step")
plt.ylabel("Distance Estimate (meters)")
plt.title("Linear Least Squares (LLS) for Sensor Fusion")
plt.legend()
plt.show()
