import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# 1. System Simulation Function
# ------------------------------------------------------------------------
def simulate_system(x0, A, B, controller, n_sim):
    """
    Simulates the discrete-time system:
        x_{k+1} = A x_k + B u_k
    over n_sim time steps, starting from x0.
    """
    dim_x = A.shape[0]
    dim_u = B.shape[1]
    
    x_log = np.zeros((n_sim + 1, dim_x))
    u_log = np.zeros((n_sim, dim_u))
    
    x_log[0, :] = x0.flatten()
    
    for k in range(n_sim):
        u = controller(x_log[k, :])
        x_next = A @ x_log[k, :] + B @ u
        x_log[k+1, :] = x_next
        u_log[k, :] = u
    
    return x_log, u_log

# ------------------------------------------------------------------------
# 2. KKT-Based LQR
# ------------------------------------------------------------------------
def kkt_lqr(A, B, Q=None, R=None, horizon=10):
    """
    Constructs a KKT-based LQR controller for the system
    x_{k+1} = A x_k + B u_k
    over a finite horizon (preview).
    The user can pass in Q, R, or rely on defaults.
    """
    # Default costs encouraging a more aggressive control
    if Q is None:
        Q = np.eye(A.shape[0]) * 10.0   # Heavier penalty on states
    if R is None:
        R = np.eye(B.shape[1]) * 1.0    # Lighter penalty on control

    n_x = A.shape[0]
    n_u = B.shape[1]
    
    # Flatten dimension references
    total_x = (horizon + 1) * n_x  # states over horizon
    total_u = horizon * n_u       # controls over horizon
    total_vars = total_x + total_u
    
    # 2.1 Construct the cost matrix P
    P = np.zeros((total_vars, total_vars))
    # place Q for each x_k
    for k in range(horizon):
        idx_x = k * n_x
        P[idx_x:idx_x + n_x, idx_x:idx_x + n_x] = Q
    # place Q for final x_{horizon}
    P[ horizon * n_x : horizon * n_x + n_x,
        horizon * n_x : horizon * n_x + n_x ] = Q
    # place R for each u_k
    for k in range(horizon):
        idx_u = total_x + k * n_u
        P[idx_u:idx_u + n_u, idx_u:idx_u + n_u] = R

    # 2.2 Build dynamics constraints
    #   x_{k+1} - A x_k - B u_k = 0
    # We'll collect these into a large matrix C * z = d
    # z = [ x_0, x_1, ... x_horizon, u_0, ... u_{horizon-1} ]
    # dims: # of constraints = horizon * n_x
    C = np.zeros((horizon * n_x, total_vars))
    d = np.zeros((horizon * n_x, 1))
    
    # For each k in [0..horizon-1], fill row block
    for k in range(horizon):
        row_start = k * n_x
        # x_{k+1}
        col_xkp1 = (k + 1) * n_x
        # x_k
        col_xk = k * n_x
        # u_k
        col_uk = total_x + k * n_u
        
        # x_{k+1} portion
        C[row_start:row_start + n_x, col_xkp1:col_xkp1 + n_x] = np.eye(n_x)
        # -A x_k portion
        C[row_start:row_start + n_x, col_xk:col_xk + n_x] = -A
        # -B u_k portion
        C[row_start:row_start + n_x, col_uk:col_uk + n_u] = -B

    # 2.3 We also impose x_0 is free (the actual x0), so that doesn't appear in constraints
    # Instead, we'll fix x0 in the cost solution directly via KKT
    # We'll set that in the right-hand side (some approaches do it differently).
    
    # 2.4 KKT matrix
    #     [ P     C^T ] [ z ] = [ -q ]
    #     [ C     0   ] [ λ ]   [  d ]
    # but here q = 0, so left side is just 0 for the cost linear part
    # We do need to set x0 in the constraints manually: x_0 - x_initial = 0
    # But let's do a simpler approach: we'll define a function that solves once at each step.
    
    # We'll store these big matrices to solve on-the-fly in the closure:
    # but let's do a single solve for each state. In a real scenario, you'd do an MPC approach
    # that re-solves with x0 each time. For demonstration, let's do a single-step approach:
    
    # 2.5 We'll define an internal function that solves the KKT system
    # given the current x0
    def controller_kkt(x_current):
        # Build offset for x0 in the first block of variables
        # we want x_0 = x_current => z[0..n_x-1] = x_current
        # We'll add equality constraints for that as well
        # We'll do that by adding rows that enforce x_0 - x_current = 0
        eq_extra = np.zeros((n_x, total_vars))
        eq_extra[0:n_x, 0:n_x] = np.eye(n_x)
        C_aug = np.vstack([C, eq_extra])
        
        d_aug = np.vstack([d, x_current.reshape(-1, 1)])
        
        # KKT: [ P      C_aug^T ] [z]   = [0]
        #      [ C_aug   0      ] [λ]     [d_aug]
        # Construct augmented KKT
        top = np.hstack([P, C_aug.T])
        bot = np.hstack([C_aug, np.zeros((C_aug.shape[0], C_aug.shape[0]))])
        KKT = np.vstack([top, bot])
        
        rhs = np.vstack([ np.zeros((total_vars,1)), d_aug ])
        
        sol = np.linalg.solve(KKT, rhs)
        
        # z is the first portion
        z = sol[:total_vars]
        # Return the first control: z[ total_x : total_x + n_u ]
        # for k=0
        return z[ total_x : total_x + n_u ].flatten()
    
    return controller_kkt

# ------------------------------------------------------------------------
# 3. Traditional LQR (via direct solve of Algebraic Riccati or simpler method)
#    We'll do a single-step control for demonstration
# ------------------------------------------------------------------------
def simple_lqr(A, B, Q=None, R=None):
    """
    A simpler LQR solve that returns a constant gain K, i.e., u_k = -K x_k.
    For demonstration, we assume the pair (A,B) is controllable.
    """
    if Q is None:
        Q = np.eye(A.shape[0]) * 1.0
    if R is None:
        R = np.eye(B.shape[1]) * 20.0
    
    # Solve discrete-time algebraic Riccati if we want a standard LQR
    from scipy.linalg import solve_discrete_are
    P = solve_discrete_are(A, B, Q, R)
    # K = (R + B^T P B)^-1 B^T P A
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    
    def lqr_controller(x):
        # u = -Kx
        return -K @ x
    
    return lqr_controller

# ------------------------------------------------------------------------
# 4. Main script
# ------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dt = 0.1
    A_c = np.array([[0, 1, -5],
                    [-5, 0, 0],
                    [0, 1, 0]])
    B_c = np.array([[0, 0],
                    [1, 0],
                    [0, 1]])
    
    # Discretize
    A_d = np.eye(3) + A_c * dt + (A_c @ A_c) * (dt**2 / 2.0)
    B_d = B_c * dt
    
    x_init = np.random.randn(3)
    N_sim = 100
    
    # 4.1 KKT-based LQR with a horizon
    horizon = 10
    kkt_controller = kkt_lqr(A_d, B_d, Q=10*np.eye(3), R=1*np.eye(2), horizon=horizon)
    x_log_kkt, u_log_kkt = simulate_system(x_init, A_d, B_d, kkt_controller, N_sim)
    
    # 4.2 Traditional (simple) LQR
    # We give it smaller Q, bigger R => more conservative
    simple_lqr_controller = simple_lqr(A_d, B_d, Q=1*np.eye(3), R=20*np.eye(2))
    x_log_lqr, u_log_lqr = simulate_system(x_init, A_d, B_d, simple_lqr_controller, N_sim)
    
    # 4.3 Plot
    plt.figure()
    plt.plot(x_log_kkt[:, 0], '--', label='KKT-LQR, x1')
    plt.plot(x_log_lqr[:, 0], label='Simple LQR, x1')
    plt.title('Comparison of x1: KKT-LQR vs Simple LQR')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.plot(np.linalg.norm(x_log_kkt, axis=1), '--', label='KKT-LQR Norm')
    plt.plot(np.linalg.norm(x_log_lqr, axis=1), label='Simple LQR Norm')
    plt.title('State Norm Over Time')
    plt.legend()
    plt.grid()
    plt.show()
