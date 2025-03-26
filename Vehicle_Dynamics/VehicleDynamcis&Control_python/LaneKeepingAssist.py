import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

# -----------------------------
#  1. IMPROVED PARAMETERS
# -----------------------------
# Vehicle Parameters (matched with reference)
m = 1778.0     # Mass (kg)
Iz = 3049.0    # Yaw inertia (kg*m^2)
Lf = 1.094     # Distance from CG to front axle (m)
Lr = 1.536     # Distance from CG to rear axle (m)
L = Lf + Lr    # Wheelbase (m)
width = 1.8    # Vehicle width (m)
Cf = 180000.0  # Front cornering stiffness (N/rad)
Cr = 400000.0  # Rear cornering stiffness (N/rad)
Cd = 0.4243    # Aero drag coefficient
Frr = 218.0    # Rolling resistance (N)
g = 9.81       # Gravity (m/s^2)

# Controller Parameters
kp = np.pi / 180 * 2  # Proportional gain
xLA = 40             # Look-ahead distance
delta_max = 0.4712   # Maximum steering angle (rad)

# Simulation parameters
t0, tf = 0.0, 30.0    # Adjusted time to reach 300m
N_steps = 1000
t_eval = np.linspace(t0, tf, N_steps)

# -----------------------------
#  2. IMPROVED LANE DEFINITION
# -----------------------------
def lane_shape(x):
    """Double lane change maneuver over 300m"""
    y_offset = 3.0
    x1, x2 = 100, 200  # Adjusted lane change positions
    return y_offset * (1 - np.cos(np.pi * (x - x1)/(x2 - x1))) * \
           (x > x1) * (x < x2) * 0.5

def lane_heading(x):
    """Lane heading for double lane change"""
    y_offset = 3.0
    x1, x2 = 100, 200
    dy_dx = y_offset * np.pi/(x2 - x1) * np.sin(np.pi * (x - x1)/(x2 - x1)) * \
            (x > x1) * (x < x2) * 0.5
    return np.arctan(dy_dx)

# -----------------------------
#  3. IMPROVED VEHICLE DYNAMICS
# -----------------------------
def vehicle_dynamics(t, state):
    x, y, psi, Ux, Uy, r = state
    
    # Lane following calculations
    y_lane = lane_shape(x)
    psi_lane = lane_heading(x)
    
    # Error calculations with look-ahead
    x_LA = x + xLA * np.cos(psi)
    y_LA = y + xLA * np.sin(psi)
    y_lane_LA = lane_shape(x_LA)
    
    e = y - y_lane
    e_LA = y_LA - y_lane_LA
    e_heading = psi - psi_lane
    
    # Understeering gradient calculation
    FzF = m * g * Lr / L
    FzR = m * g * Lf / L
    Kug = (FzF / Cr - FzR / Cr) / g
    
    # Steady-state sideslip calculation
    if abs(Ux) < 0.1:
        alpha_r = 0
    else:
        alpha_r = np.arctan2((Uy - Lr * r), abs(Ux))
    
    beta_SS = Lr * lane_heading(x) + alpha_r
    
    # Improved steering control
    delta_FB = -kp * (e * 2 + xLA * (e_heading + beta_SS))
    delta_FF = L * lane_heading(x) + Kug * (Ux ** 2) * lane_heading(x)
    delta = np.clip(delta_FB + delta_FF, -delta_max, delta_max)
    
    # Longitudinal force (speed control)
    Fx = 75 * (20 - Ux) + Frr + Cd * Ux**2  # Target speed 20 m/s
    
    # Slip angles
    if abs(Ux) < 0.1:
        alpha_f = 0.0
        alpha_r = 0.0
    else:
        alpha_f = np.arctan2((Uy + Lf * r), abs(Ux)) - delta
        alpha_r = np.arctan2((Uy - Lr * r), abs(Ux))
    
    # Tire forces
    Fy_f = -Cf * alpha_f
    Fy_r = -Cr * alpha_r
    
    # Aerodynamic drag
    Fd = Frr + Cd * Ux**2
    
    # Equations of motion
    dxdt = Ux * np.cos(psi) - Uy * np.sin(psi)
    dydt = Ux * np.sin(psi) + Uy * np.cos(psi)
    dpsidt = r
    dUxdt = (Fx * np.cos(delta) - Fy_f * np.sin(delta) - Fd) / m + r * Uy
    dUydt = (Fy_f * np.cos(delta) + Fy_r) / m - r * Ux
    drdt = (Lf * Fy_f * np.cos(delta) - Lr * Fy_r) / Iz

    return [dxdt, dydt, dpsidt, dUxdt, dUydt, drdt]

# -----------------------------
#  4. SIMULATION
# -----------------------------
# Initial conditions
x0 = 0.0
y0 = 0.0      # Start at center
psi0 = 0.0
Ux0 = 15.0    # Initial speed
Uy0 = 0.0
r0 = 0.0

state0 = [x0, y0, psi0, Ux0, Uy0, r0]
sol = solve_ivp(vehicle_dynamics, (t0, tf), state0, t_eval=t_eval)

x_sol, y_sol, psi_sol, Ux_sol, Uy_sol, r_sol = sol.y

# -----------------------------
#  5. SEPARATED VISUALIZATION
# -----------------------------
# Figure 1: Lane Keeping Trajectory
plt.figure(figsize=(20, 16))  # Increased height from 8 to 16 while keeping width

# Function to plot trajectory
def plot_trajectory(ax):
    # Plot lane boundaries
    X_range = np.linspace(0, 300, 500)
    Y_lane = lane_shape(X_range)
    lane_width = 3.6
    
    # Lane and trajectory
    ax.fill_between(X_range, Y_lane - lane_width/2, Y_lane + lane_width/2, 
                    color='lightgray', alpha=0.3, label='Lane Boundaries')
    ax.plot(X_range, Y_lane, 'k--', alpha=0.8, label='Lane Center')
    ax.plot(x_sol, y_sol, 'b-', label='Vehicle Path', linewidth=2)
    
    # Add vehicle visualization
    num_vehicles = 12
    indices = np.linspace(0, len(x_sol)-1, num_vehicles, dtype=int)
    
    for i in indices:
        x, y, psi = x_sol[i], y_sol[i], psi_sol[i]
        
        # Vehicle body
        vehicle = Rectangle((-Lr, -width/2), L, width, 
                          facecolor='red', alpha=0.3, edgecolor='darkred')
        
        # Wheels
        wheel_width = 0.2
        wheel_length = 0.4
        
        fl_wheel = Rectangle((-wheel_length/2 + Lf, width/2 - wheel_width/2), 
                           wheel_length, wheel_width, color='black')
        fr_wheel = Rectangle((-wheel_length/2 + Lf, -width/2 - wheel_width/2), 
                           wheel_length, wheel_width, color='black')
        rl_wheel = Rectangle((-wheel_length/2 - Lr, width/2 - wheel_width/2), 
                           wheel_length, wheel_width, color='black')
        rr_wheel = Rectangle((-wheel_length/2 - Lr, -width/2 - wheel_width/2), 
                           wheel_length, wheel_width, color='black')
        
        # Transform
        t = transforms.Affine2D().rotate(psi).translate(x, y) + ax.transData
        for patch in [vehicle, fl_wheel, fr_wheel, rl_wheel, rr_wheel]:
            patch.set_transform(t)
            ax.add_patch(patch)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Lane Keeping Trajectory')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set limits with equal scaling
    ax.set_xlim(-10, 310)
    ax.set_ylim(-20, 20)  # Keep the same limits
    ax.set_aspect('equal')

# Create and plot trajectory
ax = plt.gca()
plot_trajectory(ax)

plt.tight_layout()
plt.show()

# Figure 2: Vehicle States (4x1 subplots for better visibility)
fig, axs = plt.subplots(4, 1, figsize=(20, 12))


# Define lane width for plotting
lane_width = 3.6  # Standard lane width (m)

# 1. Lateral Error (top)
e_sol = y_sol - lane_shape(x_sol)
axs[0].plot(x_sol, e_sol, 'r-', label='Lateral Error', linewidth=2)
axs[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[0].fill_between(x_sol, -lane_width/2, lane_width/2, color='green', alpha=0.1)
axs[0].set_ylabel('Lateral Error (m)', fontsize=12)
axs[0].legend(fontsize=10)
axs[0].grid(True, alpha=0.3)
axs[0].set_title('Lateral Error vs Position', fontsize=12)

# 2. Vehicle Speed (second)
speed = np.sqrt(Ux_sol**2 + Uy_sol**2)
axs[1].plot(x_sol, Ux_sol, 'b-', label='Longitudinal Speed', linewidth=2)
axs[1].plot(x_sol, Uy_sol, 'g-', label='Lateral Speed', linewidth=2)
axs[1].set_ylabel('Velocity (m/s)', fontsize=12)
axs[1].legend(fontsize=10)
axs[1].grid(True, alpha=0.3)
axs[1].set_title('Vehicle Velocities vs Position', fontsize=12)

# 3. Yaw Rate (third)
axs[2].plot(x_sol, np.rad2deg(r_sol), 'purple', label='Yaw Rate', linewidth=2)
axs[2].set_ylabel('Yaw Rate (deg/s)', fontsize=12)
axs[2].legend(fontsize=10)
axs[2].grid(True, alpha=0.3)
axs[2].set_title('Yaw Rate vs Position', fontsize=12)

# 4. Heading Error (bottom)
psi_lane = lane_heading(x_sol)
heading_error = np.rad2deg(psi_sol - psi_lane)
axs[3].plot(x_sol, heading_error, 'orange', label='Heading Error', linewidth=2)
axs[3].set_ylabel('Heading Error (deg)', fontsize=12)
axs[3].legend(fontsize=10)
axs[3].grid(True, alpha=0.3)
axs[3].set_title('Heading Error vs Position', fontsize=12)

# Set x-axis limits and labels for all subplots
for i in range(4):
    axs[i].set_xlim(0, 300)
    if i == 3:  # Only bottom plot needs x-label
        axs[i].set_xlabel('X Position (m)', fontsize=12)
    axs[i].tick_params(labelsize=10)

# Adjust layout with more space at the top for the title
plt.subplots_adjust(top=0.92)
plt.show()
