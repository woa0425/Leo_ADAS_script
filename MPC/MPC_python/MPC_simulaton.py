import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # for drawing obstacles

class MPCVehicleSim:
    def __init__(self, dt=0.05, total_time=20.0, wheelbase=1.5):
        self.dt         = dt
        self.total_time = total_time
        self.Ns         = int(np.ceil(total_time / dt))
        self.L          = wheelbase

        self.delta_min, self.delta_max = -0.5, 0.5  # Steering (rad)
        self.a_min,     self.a_max     = -2.0, 2.0  # Accel (m/s^2)

        # Cost weights
        self.Q = np.diag([20.0, 20.0, 2.0, 0.5])    
        self.R = np.diag([0.1, 0.05])
        self.Q_terminal = np.diag([25.0, 25.0, 5.0, 1.0])

    def bicycle_dynamics(self, state, u):
        """Kinematic bicycle with a simplified slip angle."""
        x, y, psi, v = state
        delta, a     = u
        beta = np.arctan(0.5 * np.tan(delta))
        dx   = v * np.cos(psi + beta)
        dy   = v * np.sin(psi + beta)
        dpsi = (v / (0.5*self.L)) * np.sin(beta)
        dv   = a
        return np.array([dx, dy, dpsi, dv])

    def step_dynamics(self, state, u):
        """One-step Euler integration."""
        return state + self.dt * self.bicycle_dynamics(state, u)

    def generate_oscillatory_reference(self, alpha=12, beta=16):
        """
        Creates a reference trajectory (X_ref) & control reference (U_ref).
        X_ref: shape (Ns+1, 4) -> [x, y, psi, v]
        U_ref: shape (Ns,   2) -> [steering, accel]
        """
        X_ref = np.zeros((self.Ns+1, 4))
        U_ref = np.zeros((self.Ns,   2))

        for k in range(self.Ns):
            steer = 0.3 * np.sin(k / alpha) + 0.2 * np.cos(k / beta)
            accel = -1.0 * (X_ref[k, 3] - 6.0 + 4.0 * np.sin(k / 25.0))

            steer = np.clip(steer, self.delta_min, self.delta_max)
            accel = np.clip(accel, self.a_min,     self.a_max)

            U_ref[k,:]   = [steer, accel]
            X_ref[k+1,:] = self.step_dynamics(X_ref[k,:], U_ref[k,:])

        return X_ref, U_ref

    def linearize_dynamics(self, x_nom, u_nom):
        """
        A,B,c for local discrete linearization around (x_nom,u_nom).
        X_{k+1} ~ A X_k + B U_k + c
        """
        x, y, psi, v = x_nom
        delta_nom, a_nom = u_nom

        beta = np.arctan(0.5 * np.tan(delta_nom))
        cos_ = np.cos(psi + beta)
        sin_ = np.sin(psi + beta)

        A = np.eye(4)
        # partial wrt psi
        A[0,2] += -self.dt * v * sin_
        A[1,2] +=  self.dt * v * cos_
        # partial wrt v
        A[0,3] += self.dt * cos_
        A[1,3] += self.dt * sin_

        B = np.zeros((4,2))
        B[2,0] = self.dt * (v / (0.5*self.L)) * np.cos(beta)
        B[3,1] = self.dt

        x_next_nom = self.step_dynamics(x_nom, u_nom)
        c = x_next_nom - A @ x_nom - B @ u_nom
        return A, B, c

    def solve_mpc(self, x_curr, X_nom, U_nom, horizon):
        """
        Solve one QP for 'horizon' steps.
        """
        nx, nu = 4, 2
        X = cp.Variable((nx, horizon+1))
        U = cp.Variable((nu, horizon))

        cost = 0
        constr = []
        constr += [X[:, 0] == x_curr]

        for k in range(horizon):
            cost += cp.quad_form(X[:,k] - X_nom[k,:], self.Q)
            cost += cp.quad_form(U[:,k], self.R)

            A_k, B_k, c_k = self.linearize_dynamics(X_nom[k,:], U_nom[k,:])
            constr += [ X[:,k+1] == A_k@X[:,k] + B_k@U[:,k] + c_k ]

            # Input constraints
            constr += [
                self.delta_min <= U[0,k], U[0,k] <= self.delta_max,
                self.a_min     <= U[1,k], U[1,k] <= self.a_max
            ]

        # Terminal cost
        cost += cp.quad_form(X[:,horizon] - X_nom[horizon,:], self.Q_terminal)

        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return np.array([0.0, 0.0]), None

        return U.value[:, 0], (X.value, U.value)

    def run_mpc(self, x_init, horizon=15, alpha=12, beta=16):
        """
        Receding-horizon simulation for Ns steps using the
        oscillatory reference generated above.
        """
        # Build reference
        X_ref, U_ref = self.generate_oscillatory_reference(alpha=alpha, beta=beta)

        x_log = np.zeros((self.Ns+1, 4))
        u_log = np.zeros((self.Ns,   2))
        x_log[0,:] = x_init.copy()

        curr_state = x_init.copy()

        # nominal guesses
        X_nom = np.tile(curr_state, (horizon+1,1))
        U_nom = np.zeros((horizon,2))

        for k in range(self.Ns):
            tail = min(k+horizon, self.Ns)
            x_slice = X_ref[k:tail+1, :]
            u_slice = U_ref[k:tail,   :]

            # pad if needed
            needed = horizon+1 - (x_slice.shape[0])
            if needed > 0:
                x_slice = np.vstack([x_slice, np.tile(x_slice[-1,:], (needed,1))])
                u_pad   = np.tile(u_slice[-1,:], (needed,1))
                u_slice = np.vstack([u_slice,   u_pad])

            u_opt, _ = self.solve_mpc(curr_state, x_slice, u_slice, horizon)

            next_state = self.step_dynamics(curr_state, u_opt)
            x_log[k+1,:] = next_state
            u_log[k,:]   = u_opt
            curr_state   = next_state.copy()

            # shift guesses
            X_nom = np.roll(X_nom, -1, axis=0)
            X_nom[-1,:] = x_slice[-1,:]
            U_nom = np.roll(U_nom, -1, axis=0)
            U_nom[-1,:] = [0.0, 0.0]

        return x_log, u_log, X_ref


########################################################################
#         Main: We define obstacles AWAY from the path
########################################################################
if __name__=="__main__":
    import matplotlib.patches as patches

    sim = MPCVehicleSim(dt=0.05, total_time=20.0, wheelbase=1.5)
    x0  = np.array([0.0, 0.0, 0.0, 3.0])
    horizon = 15

    x_hist, u_hist, X_ref = sim.run_mpc(
        x_init=x0,
        horizon=horizon,
        alpha=12,
        beta=30
    )

    # Define obstacles well above or below the path
    # Just replace the old 'obstacles' definition with this:
    obstacles = []
    # Circle at center (7, -5), radius e.g. 2
    obstacles.append(("circle", (16, -3), 3.0))

    # Rectangle with bottom-left corner at (25, 0), width 4, height 5
    obstacles.append(("rectangle", (23, 3), (4, -12)))

    # Circle at (32.5, -12.5), radius e.g. 3
    obstacles.append(("circle", (32.5, -11.5), 2.0))

    fig, ax = plt.subplots(figsize=(7,7))

    # Plot reference + MPC
    ax.plot(X_ref[:,0], X_ref[:,1], 'r--', label='Reference', linewidth=1.0)
    ax.plot(x_hist[:,0], x_hist[:,1], 'b--', label='MPC Traj', linewidth=1.0)

    # Start/End
    ax.plot(x_hist[0,0],  x_hist[0,1],  'go', label='Start')
    ax.plot(x_hist[-1,0], x_hist[-1,1], 'ks', label='End')

    # Obstacles
    for obs in obstacles:
        obs_type = obs[0]
        if obs_type == "circle":
            center_xy = obs[1]
            radius    = obs[2]
            circle = patches.Circle(center_xy, radius, color='orange', alpha=0.4, label='Obstacle')
            ax.add_patch(circle)
        elif obs_type == "rectangle":
            xy_corner = obs[1]
            w, h      = obs[2]
            rect = patches.Rectangle(xy_corner, w, h, color='orange', alpha=0.4, label='Obstacle')
            ax.add_patch(rect)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            unique_handles.append(h)
            unique_labels.append(l)
            seen.add(l)

    ax.legend(unique_handles, unique_labels)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal', 'box')
    ax.set_title('MPC with Obstacles Placed Off the Reference Path')
    ax.grid(True)
    plt.show()

    # Plot controls
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(u_hist[:,0], 'b-', label='Steering')
    plt.title('Steering vs Time'); plt.grid(True); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(u_hist[:,1], 'r-', label='Acceleration')
    plt.title('Acceleration vs Time'); plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()
