import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

class MPCVehicleSim:
    def __init__(self, dt=0.05, total_time=20.0, wheelbase=1.5):
        self.dt = dt
        self.total_time = total_time
        self.Ns = int(np.ceil(total_time / dt))
        self.L  = wheelbase

        # Control constraints: narrower bounds to force constraint activation
        self.delta_min, self.delta_max = -0.3, 0.3  # Steering (rad)
        self.a_min,     self.a_max     = -1.0, 1.0  # Accel (m/s^2)

        # "Standard" LQR weighting (time-varying, finite horizon)
        self.Q_lqr      = np.diag([20.0, 20.0, 2.0, 0.5])
        self.R_lqr      = np.diag([0.1, 0.05])
        self.Qf_lqr     = np.diag([25.0, 25.0, 5.0, 1.0])

        # *** More Aggressive MPC Tuning ***
        # Heavier state cost on x,y => tries harder to track
        # Smaller R => can use more control force
        self.Q_mpc       = np.diag([20.0, 20.0, 2.0, 0.5])
        self.R_mpc       = np.diag([0.02, 0.01])
        self.Qf_mpc      = np.diag([120.0, 120.0, 5.0, 2.0])  # bigger terminal cost

    # ------------------- DYNAMICS -------------------- #
    def bicycle_dynamics(self, state, u):
        """
        Kinematic bicycle with slip angle = arctan(0.5*tan(delta)).
        state = [x, y, psi, v], control = [delta, a].
        """
        x, y, psi, v = state
        delta, accel = u

        beta = np.arctan(0.5 * np.tan(delta))
        dx   = v * np.cos(psi + beta)
        dy   = v * np.sin(psi + beta)
        dpsi = (v / (0.5*self.L)) * np.sin(beta)
        dv   = accel
        return np.array([dx, dy, dpsi, dv])

    def step_dynamics(self, state, u):
        """Euler integration for one dt step."""
        return state + self.dt*self.bicycle_dynamics(state, u)

    # ------------------- REFERENCE GENERATION -------------------- #
    def generate_oscillatory_reference(self, alpha=12, beta=16):
        """
        Creates a reference trajectory (X_ref) & controls (U_ref).
        Each step: we pick a steering & accel, saturate them,
        and integrate forward.
        """
        X_ref = np.zeros((self.Ns+1, 4))
        U_ref = np.zeros((self.Ns,   2))

        for k in range(self.Ns):
            # Sine/cos pattern
            steer = np.cos(k / (10 * alpha)) * 0.5 + 0.5 * np.sin(k / (10 * np.sqrt(beta)))
            accel = -1 * (X_ref[k, 3] - 8 + 10 * np.sin(k / 20) + np.sin(k / np.sqrt(7)))

            # saturate reference
            steer = np.clip(steer, self.delta_min, self.delta_max)
            accel = np.clip(accel, self.a_min,     self.a_max)

            U_ref[k,:]   = [steer, accel]
            X_ref[k+1,:] = self.step_dynamics(X_ref[k,:], U_ref[k,:])

        return X_ref, U_ref

    # ------------------- DISCRETE LINEARIZATION -------------------- #
    def linearize_dynamics(self, x_nom, u_nom):
        x, y, psi, v = x_nom
        delta_nom, a_nom = u_nom

        beta = np.arctan(0.5 * np.tan(delta_nom))
        cos_ = np.cos(psi + beta)
        sin_ = np.sin(psi + beta)

        A = np.eye(4)
        # partial wrt psi
        A[0,2] += -self.dt*v*sin_
        A[1,2] +=  self.dt*v*cos_
        # partial wrt v
        A[0,3] += self.dt*cos_
        A[1,3] += self.dt*sin_

        B = np.zeros((4,2))
        B[2,0] = self.dt*(v/(0.5*self.L))*np.cos(beta)
        B[3,1] = self.dt

        # offset
        x_next_nom = self.step_dynamics(x_nom, u_nom)
        c = x_next_nom - A@x_nom - B@u_nom
        return A, B, c

    # ------------------- FINITE-HORIZON LQR (POST-CLIPPING) ------------------- #
    def run_lqr_finitehorizon(self, x_init, alpha=12, beta=16):
        """
        1) Build reference
        2) For each step, linearize about (X_ref[k], U_ref[k]).
        3) Backward Riccati recursion => K_k
        4) forward pass => u_k = U_ref[k] - K_k*(x - x_ref[k]), then clip
        """
        X_ref, U_ref = self.generate_oscillatory_reference(alpha, beta)
        N = self.Ns

        # gather linearizations
        A_list, B_list = [], []
        for k in range(N):
            A_k, B_k, _ = self.linearize_dynamics(X_ref[k,:], U_ref[k,:])
            A_list.append(A_k)
            B_list.append(B_k)

        # S array
        S_array = [None]*(N+1)
        K_array = [None]*N
        S_array[N] = self.Qf_lqr

        # backward recursion
        for k in range(N-1, -1, -1):
            A_k = A_list[k]
            B_k = B_list[k]
            S_kp1 = S_array[k+1]

            M = self.R_lqr + B_k.T @ S_kp1 @ B_k
            K_k = np.linalg.inv(M) @ (B_k.T @ S_kp1 @ A_k)

            S_k = ( self.Q_lqr
                     + A_k.T @ S_kp1 @ A_k
                     - A_k.T @ S_kp1 @ B_k @ K_k )

            K_array[k] = K_k
            S_array[k] = S_k

        # forward pass
        x_log = np.zeros((N+1, 4))
        u_log = np.zeros((N,2))
        x_log[0,:] = x_init

        for k in range(N):
            dx_k = x_log[k,:] - X_ref[k,:]
            du_k = -K_array[k] @ dx_k
            u_k  = U_ref[k,:] + du_k

            # post-clipping
            u_k[0] = np.clip(u_k[0], self.delta_min, self.delta_max)
            u_k[1] = np.clip(u_k[1], self.a_min,     self.a_max)

            x_log[k+1,:] = self.step_dynamics(x_log[k,:], u_k)
            u_log[k,:]   = u_k

        return x_log, u_log, X_ref

    # ------------------- MPC with Constraints ------------------- #
    def solve_mpc(self, x_curr, X_nom, U_nom, horizon):
        nx, nu = 4, 2
        X = cp.Variable((nx, horizon+1))
        U = cp.Variable((nu, horizon))

        cost = 0
        constraints = [X[:,0] == x_curr]

        for k in range(horizon):
            cost += cp.quad_form(X[:,k] - X_nom[k,:], self.Q_mpc)
            cost += cp.quad_form(U[:,k], self.R_mpc)

            A_k, B_k, c_k = self.linearize_dynamics(X_nom[k,:], U_nom[k,:])
            constraints.append( X[:,k+1] == A_k@X[:,k] + B_k@U[:,k] + c_k )

            # input constraints
            constraints += [
                self.delta_min <= U[0,k], U[0,k] <= self.delta_max,
                self.a_min     <= U[1,k], U[1,k] <= self.a_max
            ]

        cost += cp.quad_form(X[:,horizon] - X_nom[horizon,:], self.Qf_mpc)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return np.array([0.0, 0.0]), None

        return U.value[:, 0], (X.value, U.value)

    def run_mpc(self, x_init, horizon=25, alpha=12, beta=16):
        X_ref, U_ref = self.generate_oscillatory_reference(alpha, beta)
        x_log = np.zeros((self.Ns+1, 4))
        u_log = np.zeros((self.Ns,   2))
        x_log[0,:] = x_init

        curr_state = x_init.copy()
        for k in range(self.Ns):
            tail = min(k+horizon, self.Ns)
            # slice
            x_slice = X_ref[k:tail+1,:]
            u_slice = U_ref[k:tail,:]

            # pad
            needed = horizon+1 - x_slice.shape[0]
            if needed>0:
                x_slice = np.vstack([x_slice, np.tile(x_slice[-1,:], (needed,1))])
                u_pad   = np.tile(u_slice[-1,:], (needed,1))
                u_slice = np.vstack([u_slice,   u_pad])

            # solve
            u_opt, _ = self.solve_mpc(curr_state, x_slice, u_slice, horizon)
            # step
            next_st = self.step_dynamics(curr_state, u_opt)
            x_log[k+1,:] = next_st
            u_log[k,:]   = u_opt
            curr_state = next_st

        return x_log, u_log, X_ref


# ----------------- DEMO: Compare LQR vs MPC ---------------
if __name__=="__main__":
    sim = MPCVehicleSim(dt=0.05, total_time=20.0, wheelbase=1.5)
    x_init = np.array([0.0, 0.0, 0.0, 3.0])

    # 1) LQR
    x_lqr, u_lqr, ref_lqr = sim.run_lqr_finitehorizon(
        x_init=x_init, alpha=12, beta=16
    )

    # 2) MPC with bigger horizon & more aggressive cost
    x_mpc, u_mpc, ref_mpc = sim.run_mpc(
        x_init=x_init, horizon=25, alpha=12, beta=16
    )

    # Plot path
    plt.figure(figsize=(6,6))
    plt.plot(ref_lqr[:,0], ref_lqr[:,1], 'k--', label='Reference', linewidth=2)
    plt.plot(x_lqr[:,0], x_lqr[:,1], 'r--', label='LQR (clipped)', linewidth=1.2)
    plt.plot(x_mpc[:,0], x_mpc[:,1], 'b-', label='MPC (constrained)', linewidth=1.2)

    # Start/End
    plt.plot(x_lqr[0,0],  x_lqr[0,1],  'go', label='LQR Start')
    plt.plot(x_lqr[-1,0], x_lqr[-1,1], 'gs', label='LQR End')
    plt.plot(x_mpc[0,0],  x_mpc[0,1],  'bo', label='MPC Start')
    plt.plot(x_mpc[-1,0], x_mpc[-1,1], 'bs', label='MPC End')

    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Time-Varying LQR vs. MPC')
    plt.legend()
    plt.show()

    # Steering/acc comparisons
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(u_lqr[:,0], 'g-', label='LQR Steering')
    plt.plot(u_mpc[:,0], 'b-', label='MPC Steering')
    plt.title('Steering vs Time'); plt.grid(True); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(u_lqr[:,1], 'g-', label='LQR Accel')
    plt.plot(u_mpc[:,1], 'b-', label='MPC Accel')
    plt.title('Acceleration vs Time'); plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()
