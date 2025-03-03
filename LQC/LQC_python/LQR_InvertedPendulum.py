import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# System Parameters
M = 1.0  # Cart Mass (kg)
m = 0.1  # Pendulum Mass (kg)
l = 0.5  # Pendulum Length (m)
g = 9.81  # Gravity (m/s²)

# Linearized State-Space Matrices
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -m*g/M, 0],
    [0, 0, 0, 1],
    [0, 0, (M+m)*g/(l*M), 0]
])

B = np.array([
    [0],
    [1/M],
    [0],
    [-1/(l*M)]
])

# Define LQR Weight Matrices
Q = np.diag([10, 1, 10, 1])  # Penalize position & angle deviations
R = np.array([[0.1]])  # Penalize control effort

# Solve Algebraic Riccati Equation for LQR
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P  # Optimal feedback gain

# Compute closed-loop eigenvalues to check stability
eigenvalues = np.linalg.eigvals(A - B @ K)
print("Closed-loop Eigenvalues:", eigenvalues)

# Initial State: Small angle deviation (Unstable Start)
x_init = np.array([[0.0], [0.0], [np.radians(10)], [0.0]])  # 10-degree deviation

# Simulation Parameters
T = 5.0  # Simulation time (seconds)
dt = 0.01  # Time step
time_steps = int(T / dt)

# Initialize state and control input history
x = x_init
trajectory = [x.flatten()]
control_inputs = []

# Simulate system response under LQR control
for t in range(time_steps):
    u = -K @ x  # Compute LQR control input
    control_inputs.append(u.flatten())  # Store control input

    # Update state using Euler method: x_next = x + dx/dt * dt
    x = x + (A @ x + B @ u) * dt
    trajectory.append(x.flatten())

# Convert lists to NumPy arrays for plotting
trajectory = np.array(trajectory)
control_inputs = np.array(control_inputs)

# Print results
print("Feedback Gain Matrix K:")
print(K)
print("\nFinal State Reached:")
print(x)

# Plot state trajectories
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, T, time_steps + 1), trajectory[:, 0], label="Cart Position (x)")
plt.plot(np.linspace(0, T, time_steps + 1), np.degrees(trajectory[:, 2]), label="Pendulum Angle (theta)")
plt.axhline(0, color='k', linestyle='--', linewidth=0.8, label="Target (0)")
plt.title("State Trajectories under LQR Control")
plt.xlabel("Time (s)")
plt.ylabel("State Values")
plt.legend()
plt.grid()

# Plot control input over time
plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, T, time_steps), control_inputs, label="Control Input (Force u)", color="red")
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.title("LQR Control Input Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Force Applied to Cart")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
