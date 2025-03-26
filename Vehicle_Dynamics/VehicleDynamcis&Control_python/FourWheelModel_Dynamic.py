import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

# Enhanced vehicle parameters
m = 1500  # Vehicle mass (kg)
Iz = 3000  # Moment of inertia about z-axis (kg*m^2)
Lf = 1.2  # Distance from CG to front axle (m)
Lr = 1.4  # Distance from CG to rear axle (m)
L = Lf + Lr  # Wheelbase (m)
track = 1.6  # Vehicle track width (m)
h_cg = 0.5  # CG height (m)
Cx = 10000  # Longitudinal stiffness (N/m)
Cy = 15000  # Cornering stiffness (N/rad)
R = 0.3  # Tire radius (m)

# Aerodynamic parameters
Cd = 0.3
rho = 1.225  # Air density (kg/m³)
A_front = 2.2  # Frontal area (m²)

# Tire parameters (Simplified Pacejka Magic Formula)
B = 10.0  # Stiffness factor
C = 1.9  # Shape factor
D = 1.0  # Peak value
E = 0.97  # Curvature factor

def magic_formula(alpha): # Pacejka Magic Formula for tire forces
    return D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))))

# Initial conditions [x, y, psi, Ux, Uy, yaw_rate]
initial_conditions = [0, 0, 0, 15, 0, 0]
t_span = (0, 10)
t_eval = np.linspace(*t_span, 200)

def vehicle_dynamics(t, state):
    x, y, psi, Ux, Uy, r = state
    
    # Steering input (example: sinusoidal steering)
    delta = 0.1 * np.sin(t)  # Steering angle
    
    # Slip angles with steering
    alpha_FL = delta - np.arctan((Uy + Lf * r) / (Ux + 0.001))
    alpha_FR = delta - np.arctan((Uy + Lf * r) / (Ux + 0.001))
    alpha_RL = -np.arctan((Uy - Lr * r) / (Ux + 0.001))
    alpha_RR = -np.arctan((Uy - Lr * r) / (Ux + 0.001))
    
    # Lateral forces using Magic Formula
    Fy_FL = Cy * magic_formula(alpha_FL)
    Fy_FR = Cy * magic_formula(alpha_FR)
    Fy_RL = Cy * magic_formula(alpha_RL)
    Fy_RR = Cy * magic_formula(alpha_RR)

    # Longitudinal forces with traction control
    Fx_FL = Fx_FR = Cx * 0.3  # Front wheel drive
    Fx_RL = Fx_RR = 0

    # Aerodynamic forces
    Fd = 0.5 * rho * Cd * A_front * Ux**2

    # Load transfer effects
    Fz_transfer_lat = (m * Uy * r * h_cg) / track
    Fz_transfer_long = (m * Ux * r * h_cg) / L

    # System equations
    dx_dt = Ux * np.cos(psi) - Uy * np.sin(psi)
    dy_dt = Ux * np.sin(psi) + Uy * np.cos(psi)
    dpsi_dt = r
    
    dUx_dt = (np.cos(delta)*(Fx_FL + Fx_FR) - np.sin(delta)*(Fy_FL + Fy_FR) + 
              Fx_RL + Fx_RR - Fd) / m + Uy * r
    
    dUy_dt = (np.sin(delta)*(Fx_FL + Fx_FR) + np.cos(delta)*(Fy_FL + Fy_FR) + 
              Fy_RL + Fy_RR) / m - Ux * r
    
    dr_dt = (Lf * (np.cos(delta)*(Fy_FL + Fy_FR) + np.sin(delta)*(Fx_FL + Fx_FR)) - 
             Lr * (Fy_RL + Fy_RR)) / Iz

    return [dx_dt, dy_dt, dpsi_dt, dUx_dt, dUy_dt, dr_dt]

# Solve the system
sol = solve_ivp(vehicle_dynamics, t_span, initial_conditions, t_eval=t_eval)

# -----------------------------
#  1. VEHICLE TRAJECTORY PLOT
# -----------------------------
plt.figure(figsize=(10, 8))
ax_traj = plt.gca()

# Plot vehicle path
ax_traj.plot(sol.y[0], sol.y[1], 'b-', label='Vehicle Path', linewidth=2)

# Add vehicle animation
num_frames = 15
for i in range(0, len(t_eval), len(t_eval)//num_frames):
    x, y, psi = sol.y[0][i], sol.y[1][i], sol.y[2][i]
    
    # Create vehicle rectangle
    vehicle = Rectangle((-Lr, -track/2), L, track, 
                       facecolor='lightblue', alpha=0.3, edgecolor='blue')
    
    # Add wheels (small black rectangles)
    wheel_width = 0.2
    wheel_length = 0.4
    
    # Front left wheel
    fl_wheel = Rectangle((-wheel_length/2 + Lf, track/2 - wheel_width/2), 
                        wheel_length, wheel_width, color='black')
    # Front right wheel
    fr_wheel = Rectangle((-wheel_length/2 + Lf, -track/2 - wheel_width/2), 
                        wheel_length, wheel_width, color='black')
    # Rear left wheel
    rl_wheel = Rectangle((-wheel_length/2 - Lr, track/2 - wheel_width/2), 
                        wheel_length, wheel_width, color='black')
    # Rear right wheel
    rr_wheel = Rectangle((-wheel_length/2 - Lr, -track/2 - wheel_width/2), 
                        wheel_length, wheel_width, color='black')
    
    # Transform vehicle and wheels
    t = transforms.Affine2D().rotate(psi).translate(x, y) + ax_traj.transData
    vehicle.set_transform(t)
    fl_wheel.set_transform(t)
    fr_wheel.set_transform(t)
    rl_wheel.set_transform(t)
    rr_wheel.set_transform(t)
    
    # Add vehicle and wheels to plot
    ax_traj.add_patch(vehicle)
    ax_traj.add_patch(fl_wheel)
    ax_traj.add_patch(fr_wheel)
    ax_traj.add_patch(rl_wheel)
    ax_traj.add_patch(rr_wheel)

# Add velocity vectors at selected points
arrow_scale = 2.0
for i in range(0, len(t_eval), len(t_eval)//8):
    x, y = sol.y[0][i], sol.y[1][i]
    Ux, Uy = sol.y[3][i], sol.y[4][i]
    # Plot velocity vector
    ax_traj.arrow(x, y, Ux*arrow_scale*0.1, Uy*arrow_scale*0.1,
                 head_width=0.3, head_length=0.5, fc='red', ec='red')

ax_traj.set_xlabel('X Position (m)')
ax_traj.set_ylabel('Y Position (m)')
ax_traj.set_title('Vehicle Trajectory with Position and Velocity Visualization')
ax_traj.grid(True)
ax_traj.axis('equal')
plt.show()

# -----------------------------
#  2. VEHICLE DYNAMICS PLOTS
# -----------------------------
fig, axs = plt.subplots(4, 1, figsize=(12, 12))
fig.suptitle('Vehicle Dynamics Analysis', fontsize=16)

# 1. Velocities
axs[0].plot(t_eval, sol.y[3], 'b-', label='Longitudinal (Ux)')
axs[0].plot(t_eval, sol.y[4], 'r-', label='Lateral (Uy)')
axs[0].set_ylabel('Velocity (m/s)')
axs[0].legend()
axs[0].grid(True)
axs[0].set_title('Vehicle Velocities')

# 2. Yaw Rate
axs[1].plot(t_eval, np.rad2deg(sol.y[5]), 'g-', label='Yaw Rate')
axs[1].set_ylabel('Yaw Rate (deg/s)')
axs[1].legend()
axs[1].grid(True)
axs[1].set_title('Vehicle Yaw Rate')

# 3. Steering Input
delta_vals = [0.1 * np.sin(t) for t in t_eval]
axs[2].plot(t_eval, np.rad2deg(delta_vals), 'm-', label='Steering Angle')
axs[2].set_ylabel('Angle (deg)')
axs[2].legend()
axs[2].grid(True)
axs[2].set_title('Steering Input')

# 4. Vehicle Speed
speed = np.sqrt(sol.y[3]**2 + sol.y[4]**2)
axs[3].plot(t_eval, speed, 'k-', label='Total Speed')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Speed (m/s)')
axs[3].legend()
axs[3].grid(True)
axs[3].set_title('Total Vehicle Speed')

plt.tight_layout()
plt.show()
