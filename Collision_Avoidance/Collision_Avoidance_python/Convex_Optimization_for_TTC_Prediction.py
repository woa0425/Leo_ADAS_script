import numpy as np
import matplotlib.pyplot as plt


# ========== Configuration ========== #
MINIMUM_GAP = 2.0       # [m]
MAX_ITERATIONS = 8
INITIAL_TIME = 0.0
STEP_GAIN = 1.0
FALLBACK_ENABLED = True
CONVERGENCE_TOLERANCE = 1e-6
LINE_SEARCH_ITERATIONS = 10


# Vehicle and Target definitions
vehicle = {"x": 0.0, "y": 0.0, "vx": 20.0, "vy": 0.0, "ax": 2.3, "ay": 0.0}
target = {"x": 100.0, "y": 0.0, "vx": -7.0, "vy": 0.0, "ax": -1.7, "ay": 0.0}


def calculate_position(state, t):
    x = state["x"] + state["vx"] * t + 0.5 * state["ax"] * t**2
    y = state["y"] + state["vy"] * t + 0.5 * state["ay"] * t**2
    return x, y


def calculate_velocity(state, t):
    vx = state["vx"] + state["ax"] * t
    vy = state["vy"] + state["ay"] * t
    return vx, vy


def get_state(t):
    xv, yv = calculate_position(vehicle, t)
    xt, yt = calculate_position(target, t)
    dx, dy = xv - xt, yv - yt
    gap = np.sqrt(dx**2 + dy**2)
    vvx, vvy = calculate_velocity(vehicle, t)
    vtx, vty = calculate_velocity(target, t)
    dvx = vvx - vtx
    dvy = vvy - vty
    return gap, dx, dy, dvx, dvy


def compute_gradient(dx, dy, dvx, dvy, gap):
    if gap < 1e-3:
        return 0.0
    return (dx * dvx + dy * dvy) / gap


def compute_hessian(dx, dy, dvx, dvy, gap):
    if gap < 1e-3:
        return 1.0
    rel_speed_sq = dvx**2 + dvy**2
    return rel_speed_sq / gap - (dx * dvx + dy * dvy)**2 / gap**3


def line_search(t_curr, direction, max_iterations=LINE_SEARCH_ITERATIONS):
    alpha = 1.0
    beta = 0.5
    c = 0.1  # Armijo condition constant
    
    for _ in range(max_iterations):
        t_test = t_curr + alpha * direction
        if t_test < 0:
            alpha *= beta
            continue
            
        gap_curr, dx_curr, dy_curr, dvx_curr, dvy_curr = get_state(t_curr)
        gap_test, dx_test, dy_test, dvx_test, dvy_test = get_state(t_test)
        
        grad_curr = compute_gradient(dx_curr, dy_curr, dvx_curr, dvy_curr, gap_curr)
        
        # Armijo condition
        if gap_test <= gap_curr + c * alpha * grad_curr * direction:
            return alpha
            
        alpha *= beta
    
    return alpha


def solve_collision_time():
    t_curr = INITIAL_TIME
    lower_bound, upper_bound = 0.0, 5.0
    history = []
    prev_gap = float('inf')
    min_required_iterations = 2
    
    for i in range(MAX_ITERATIONS):
        gap, dx, dy, dvx, dvy = get_state(t_curr)
        grad = compute_gradient(dx, dy, dvx, dvy, gap)
        hess = compute_hessian(dx, dy, dvx, dvy, gap)
        
        # Check convexity
        if hess < 0:
            print(f"Warning: Non-convex region detected at t = {t_curr:.3f}")
        
        # Newton's method step
        if abs(hess) > 1e-6:
            direction = -grad / hess
        else:
            direction = -grad  # Fallback to gradient descent
            
        # Line search for step size
        step_size = line_search(t_curr, direction)
        t_next = t_curr + step_size * direction
        
        # Project onto feasible region
        t_next = max(lower_bound, min(t_next, upper_bound))
        
        history.append((t_curr, gap, grad, hess))
        
        # Check convergence
        if i >= min_required_iterations:
            if gap <= MINIMUM_GAP:
                return t_curr, history, False  # Solved
            if abs(t_next - t_curr) < CONVERGENCE_TOLERANCE:
                return t_curr, history, False  # Converged
                
        t_curr = t_next
    
    if FALLBACK_ENABLED:
        samples = np.linspace(t_curr - 0.5, t_curr + 0.5, 50)
        min_gap = float('inf')
        best_t = t_curr
        for t in samples:
            if t < 0: continue
            gap, *_ = get_state(t)
            if gap < min_gap:
                min_gap = gap
                best_t = t
        history.append((best_t, min_gap, 0.0, 0.0))
        return best_t, history, True
    
    return t_curr, history, False


def visualize_solution():
    collision_time, hist, fallback = solve_collision_time()
    print(f"\n=== Time to Collision Estimate: {collision_time:.3f} seconds ===\n")
    if fallback:
        print("Note: Fallback sampling was used after iterations.\n")
    times = [h[0] for h in hist]
    gaps = [h[1] for h in hist]
    gradients = [h[2] for h in hist]
    hessians = [h[3] for h in hist]
    
    time_range = np.linspace(0, 5, 300)
    true_gaps = [get_state(t)[0] for t in time_range]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot gap vs time
    ax1.plot(times, gaps, marker='o', label="Solver Iterations")
    for i, (t, g, _, _) in enumerate(hist):
        ax1.text(t, g, f"{i}", fontsize=9)
    ax1.plot(time_range, true_gaps, label="True Gap", alpha=0.6)
    ax1.axhline(MINIMUM_GAP, color='r', linestyle='--', label="Minimum Gap")
    ax1.set_title("Gap vs Time")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Gap [m]")
    ax1.grid(True)
    ax1.legend()
    
    # Plot gradient vs time
    ax2.plot(times, gradients, marker='o', label="Gradient")
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_title("Gradient vs Time")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Gradient")
    ax2.grid(True)
    ax2.legend()
    
    # Plot hessian vs time
    ax3.plot(times, hessians, marker='o', label="Hessian")
    ax3.axhline(0, color='r', linestyle='--')
    ax3.set_title("Hessian vs Time")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Hessian")
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    return collision_time, fallback, len(hist)


visualize_solution()

# ==============================================================================
# Disclaimer:
# This project is an independent, original implementation based on publicly known
# principles of convex optimization, numerical methods, and trajectory planning.
# No proprietary code, calibration, or confidential information from any prior 
# employer is used in this work.
# ==============================================================================
