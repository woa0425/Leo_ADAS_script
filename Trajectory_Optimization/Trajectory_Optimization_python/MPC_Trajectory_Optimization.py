import casadi as ca  # Import CasADi for symbolic computation and optimization
import numpy as np   # Import NumPy for numerical computations
import matplotlib.pyplot as plt  # Import matplotlib for visualization

def define_dynamics():
    """
    Define the vehicle dynamics using a bicycle model.
    States:  [x, y, heading, velocity]
    Control: [omega, acceleration]
    
    The bicycle model is a simplified vehicle model that represents:
    - x, y: position coordinates
    - heading: vehicle orientation angle
    - velocity: vehicle speed
    - omega: steering rate (angular velocity)
    - acceleration: rate of change of velocity
    """
    # Define symbolic variables for state and control
    s = ca.MX.sym("s", 4, 1)  # state vector: [x, y, theta, v]
    c = ca.MX.sym("c", 2, 1)  # control vector: [omega, accel]

    # Define the system dynamics equations
    # dx/dt = v * cos(theta)
    # dy/dt = v * sin(theta)
    # dtheta/dt = omega
    # dv/dt = acceleration
    ds = ca.vertcat(
        s[3] * ca.cos(s[2]),  # x velocity
        s[3] * ca.sin(s[2]),  # y velocity
        c[0],                 # heading rate
        c[1]                  # acceleration
    )
    return s, c, ds


def build_mpc_problem(num_steps=50, horizon=5.0, obstacle_alpha=0.9):
    """
    Construct the Model Predictive Control (MPC) optimization problem.
    
    Parameters:
    - num_steps: Number of discrete time steps in the prediction horizon
    - horizon: Total time horizon in seconds
    - obstacle_alpha: Parameter for obstacle avoidance constraints
    
    The problem includes:
    1. Vehicle dynamics constraints
    2. Obstacle avoidance constraints
    3. Initial state constraints
    4. Final state constraints
    5. Cost function for trajectory optimization
    """
    # Calculate time step size
    dt = horizon / num_steps

    # Get symbolic dynamics
    s_var, c_var, dsdt = define_dynamics()

    # Define one-step dynamics function
    # x(k+1) = x(k) + dt * dx/dt
    step_fn = s_var + dt * dsdt
    step_map = ca.Function("step_map", [s_var, c_var], [step_fn])

    # Define optimization variables
    ctrl_seq  = ca.MX.sym("ctrl_seq",  2, num_steps)    # Control sequence
    state_seq = ca.MX.sym("state_seq", 4, num_steps + 1)  # State sequence
    init_state = ca.MX.sym("init_state", 4, 1)          # Initial state parameter

    # Initialize constraints list
    constraints_expr = []

    # Add dynamic constraints for each time step
    for k in range(num_steps):
        # Calculate next state using dynamics
        next_state = step_map(state_seq[:, k], ctrl_seq[:, k])
        # Add constraints to ensure state evolution follows dynamics
        for idx in range(4):
            constraints_expr.append(state_seq[idx, k+1] - next_state[idx])

    # Add obstacle avoidance constraints
    # Uses control barrier functions to ensure vehicle stays away from obstacles
    for k in range(num_steps):
        if k % 2 == 0:
            # First obstacle at (2,2)
            b_k   = 4 - ((state_seq[0, k]   - 2)**2 + (state_seq[1, k]   - 2)**2)
            b_kp1 = 4 - ((state_seq[0, k+1] - 2)**2 + (state_seq[1, k+1] - 2)**2)
        else:
            # Second obstacle at (4,4)
            b_k   = 4 - ((state_seq[0, k]   - 4)**2 + (state_seq[1, k]   - 4)**2)
            b_kp1 = 4 - ((state_seq[0, k+1] - 4)**2 + (state_seq[1, k+1] - 4)**2)

        # Control barrier function constraint
        constraints_expr.append(b_kp1 - obstacle_alpha*b_k)

    # Add initial state constraint
    init_con = [state_seq[:, 0] - init_state]
    constraints_expr += init_con

    # Add final state constraints
    # Vehicle must end exactly at (6,6) with zero heading and velocity
    final_x = state_seq[0, -1]
    final_y = state_seq[1, -1]
    final_theta = state_seq[2, -1]
    final_v = state_seq[3, -1]

    constraints_expr.append(final_x - 6)   # Final x position
    constraints_expr.append(final_y - 6)   # Final y position
    constraints_expr.append(final_theta)   # Final heading
    constraints_expr.append(final_v)       # Final velocity

    # Define the cost function
    total_cost = 0

    # Terminal state cost
    total_cost += 100*((final_x - 6)**2 + (final_y - 6)**2)  # Position cost
    total_cost += 100*(final_theta**2 + final_v**2)          # State cost

    # Running cost for each time step
    for k in range(num_steps):
        # Extract state and control variables
        x_k     = state_seq[0, k]
        y_k     = state_seq[1, k]
        theta_k = state_seq[2, k]
        v_k     = state_seq[3, k]
        om_k    = ctrl_seq[0, k]
        acc_k   = ctrl_seq[1, k]

        # Add costs for position tracking and state regulation
        total_cost += 10*((x_k - 6)**2 + (y_k - 6)**2)  # Position tracking
        total_cost += 10*(theta_k**2 + v_k**2)          # State regulation

        # Add control effort cost
        total_cost += 1*(om_k**2 + acc_k**2)            # Control effort

    # Construct the complete optimization problem
    all_decision_vars = ca.vertcat(
        ctrl_seq.reshape((-1, 1)),    # Control variables
        state_seq.reshape((-1, 1))    # State variables
    )
    g_expr = ca.vertcat(*constraints_expr)  # All constraints

    return all_decision_vars, g_expr, total_cost, init_state, step_map

def prepare_bounds(num_steps=50):
    """
    Prepare bounds for optimization variables and constraints.
    
    The optimization problem has:
    - 4*N dynamic constraints (equality)
    - N obstacle constraints (inequality)
    - 4 initial state constraints (equality)
    - 4 final state constraints (equality)
    Total constraints: 5*N + 8
    """
    # Define bounds for state variables
    state_lb = np.array([-1e4, -1e4, -1e4, -1e4])  # Lower bounds
    state_ub = np.array([ 1e4,  1e4,  1e4,  1e4])   # Upper bounds
    
    # Define bounds for control variables
    ctrl_lb  = np.array([-1, -1])  # Lower bounds for controls
    ctrl_ub  = np.array([ 1,  1])   # Upper bounds for controls

    # Repeat bounds for entire time horizon
    full_states_lb = np.tile(state_lb, (num_steps+1, 1))
    full_states_ub = np.tile(state_ub, (num_steps+1, 1))
    full_ctrl_lb = np.tile(ctrl_lb, (num_steps, 1))
    full_ctrl_ub = np.tile(ctrl_ub, (num_steps, 1))

    # Flatten arrays for solver
    lb_dec_vars = np.concatenate([
        full_ctrl_lb.reshape((-1, 1)),
        full_states_lb.reshape((-1, 1))
    ])
    ub_dec_vars = np.concatenate([
        full_ctrl_ub.reshape((-1, 1)),
        full_states_ub.reshape((-1, 1))
    ])

    # Calculate constraint bounds lengths
    dyn_len   = num_steps * 4    # Dynamic constraints
    obs_len   = num_steps        # Obstacle constraints
    init_len  = 4                # Initial state constraints
    final_len = 4                # Final state constraints

    # Set lower bounds for constraints
    lower_g = np.concatenate([
        np.zeros(dyn_len),           # Dynamic constraints = 0
        -1e20*np.ones(obs_len),      # Obstacle constraints <= 0
        np.zeros(init_len),          # Initial constraints = 0
        np.zeros(final_len)          # Final constraints = 0
    ])

    # Set upper bounds for constraints
    upper_g = np.concatenate([
        np.zeros(dyn_len),           # Dynamic constraints = 0
        np.zeros(obs_len),           # Obstacle constraints <= 0
        np.zeros(init_len),          # Initial constraints = 0
        np.zeros(final_len)          # Final constraints = 0
    ])

    return lb_dec_vars, ub_dec_vars, lower_g, upper_g

def create_solver(problem_vars, constraints_expr, total_cost_expr, init_state_sym):
    """
    Create and configure the IPOPT solver for the optimization problem.
    
    Parameters:
    - problem_vars: Decision variables
    - constraints_expr: Constraint expressions
    - total_cost_expr: Objective function
    - init_state_sym: Initial state parameter
    """
    # Define the NLP problem structure
    nlp_dict = {
        'x': problem_vars,       # Decision variables
        'p': init_state_sym,     # Parameters (initial state)
        'f': total_cost_expr,    # Objective function
        'g': constraints_expr    # Constraints
    }
    
    # Configure solver options
    solver_opts = {
        'ipopt.print_level': 0,  # Suppress solver output
        'print_time': 0
    }
    
    # Create and return solver object
    solver_obj = ca.nlpsol("zigzag_solver", "ipopt", nlp_dict, solver_opts)
    return solver_obj

def solve_mpc():
    """
    Build and solve the MPC optimization problem.
    
    The solver finds a trajectory that:
    1. Starts from initial state (0,0,0,0)
    2. Avoids obstacles
    3. Reaches final state (6,6,0,0)
    4. Minimizes control effort and state deviations
    """
    # Set problem parameters
    num_steps = 50
    horizon   = 5.0
    alpha_val = 0.9

    # Build the optimization problem
    dec_vars, g_expr, cost_expr, init_sym, step_fn = build_mpc_problem(
        num_steps=num_steps,
        horizon=horizon,
        obstacle_alpha=alpha_val
    )

    # Prepare bounds and create solver
    lbx, ubx, lbg, ubg = prepare_bounds(num_steps)
    solver = create_solver(dec_vars, g_expr, cost_expr, init_sym)

    # Get problem dimensions
    n_dec = dec_vars.size1()
    n_con = g_expr.size1()

    # Initialize solver variables
    x0_init = np.zeros((n_dec, 1))  # Initial guess
    lamx0   = np.zeros((n_dec, 1))  # Lagrange multipliers for bounds
    lamg0   = np.zeros((n_con, 1))  # Lagrange multipliers for constraints

    # Set initial state
    init_val = np.array([0,0,0,0])
    
    # Solve the optimization problem
    sol = solver(
        x0   = x0_init,
        lam_x0 = lamx0,
        lam_g0 = lamg0,
        lbx  = lbx,
        ubx  = ubx,
        lbg  = lbg,
        ubg  = ubg,
        p    = init_val
    )
    return sol, num_steps

def plot_solution(sol, num_steps):
    """
    Visualize the solution trajectory and obstacles.
    
    The plot shows:
    - Vehicle trajectory (blue line)
    - Start point (red dot)
    - Goal point (green dot)
    - Obstacles (black filled circles)
    """
    # Extract solution dimensions
    num_ctrl  = 2
    num_state = 4
    total_ctrl_len = num_ctrl * num_steps

    # Extract state trajectory from solution
    sol_vec = sol["x"].full().flatten()
    state_vec = sol_vec[total_ctrl_len:]  # States are the last part

    # Extract x and y coordinates
    x_vals = state_vec[0::num_state]
    y_vals = state_vec[1::num_state]

    # Create plot
    plt.figure(figsize=(6,6))
    
    # Plot trajectory
    plt.plot(x_vals, y_vals, '-o', label="Trajectory")
    
    # Plot start and end points
    plt.plot(x_vals[0], y_vals[0], 'ro', label="Start")
    plt.plot(x_vals[-1], y_vals[-1], 'go', label="End (6,6)")

    # Define obstacle circles
    theta_arr = np.linspace(0, 2*np.pi, 120)
    theta_arr_2 = np.linspace(0, 2*np.pi, 120)
    obs1_x = 2 + 2*np.cos(theta_arr)
    obs1_y = 2 + 2*np.sin(theta_arr)
    obs2_x = 4 + 1.7*np.cos(theta_arr_2)
    obs2_y = 4 + 1.7*np.sin(theta_arr_2)

    # Plot filled black obstacles
    plt.fill(obs1_x, obs1_y, 'k', label="Obstacle1 (2,2)")
    plt.fill(obs2_x, obs2_y, 'k', label="Obstacle2 (4,4)")

    # Configure plot
    plt.axis("equal")
    plt.title("MPC trajectory optimization for a vehicle navigating")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Solving Zigzag MPC with final state pinned to (6,6,0,0).")
    solution, N = solve_mpc()
    plot_solution(solution, N)
