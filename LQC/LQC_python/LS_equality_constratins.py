import numpy as np
import matplotlib.pyplot as plt

def solve_least_squares_with_constraints(A, b, C, d):
    """
    Solves the constrained least squares problem:
        Minimize ||Ax - b||^2 subject to Cx = d
    using the method of Lagrange multipliers.

    Parameters:
        A (ndarray): Design matrix (m x n)
        b (ndarray): Observation vector (m,)
        C (ndarray): Constraint matrix (k x n)
        d (ndarray): Right-hand side constraint vector (k,)

    Returns:
        x (ndarray): Optimal solution vector (n,)
        lambda_ (ndarray): Lagrange multipliers (k,)
    """
    # Compute normal equation components
    ATA = A.T @ A  
    ATb = A.T @ b  

    # Construct augmented system
    augmented_matrix = np.block([
        [ATA, C.T],
        [C, np.zeros((C.shape[0], C.shape[0]))]
    ])
    rhs = np.concatenate([ATb, d])

    # Solve for x and lambda
    solution = np.linalg.solve(augmented_matrix, rhs)
    x_optimal = solution[:A.shape[1]]
    lambda_optimal = solution[A.shape[1]:]

    return x_optimal, lambda_optimal

def verify_solution(A, b, C, d, x_solution):
    """Checks if the solution satisfies the constraints and computes residual error."""
    constraint_satisfied = np.allclose(C @ x_solution, d)
    residual_error = np.linalg.norm(A @ x_solution - b)

    print("**Solution Verification**")
    print("Constraint satisfied:", constraint_satisfied)
    print("Residual norm ||Ax - b||:", residual_error)

    return constraint_satisfied, residual_error

def plot_solution(A, b, x_solution, C, d):
    """Plots the least squares solution and the constraint line for a 2D system."""
    if A.shape[1] != 2:
        print("Plotting is only supported for 2D problems.")
        return
    
    plt.figure(figsize=(8, 6))

    # Generate grid points for visualization
    x_vals = np.linspace(-5, 35, 100)
    
    # Plot the least squares fit
    y_vals = -(A[:, 0][0] * x_vals) / A[:, 1][0] + b[0] / A[:, 1][0]
    plt.plot(x_vals, y_vals, 'b--', label="Least Squares Fit (Unconstrained)")

    # Plot the constraint line
    constraint_y_vals = -(C[:, 0][0] * x_vals) / C[:, 1][0] + d[0] / C[:, 1][0]
    plt.plot(x_vals, constraint_y_vals, 'r-', label="Equality Constraint Cx = d")

    # Mark the optimal solution
    plt.scatter(x_solution[0], x_solution[1], color='green', s=100, label="Optimal Solution (x*)")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Least Squares Solution with Equality Constraint")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Define system parameters
    A = np.array([[1, 2], [3, 4], [5, 6]])  
    b = np.array([7, 8, 9])  
    C = np.array([[1, 1]])  
    d = np.array([10])  

    # Solve constrained least squares
    x_solution, lambda_solution = solve_least_squares_with_constraints(A, b, C, d)

    # Display results
    print("**Computed Solution**")
    print("Optimal x:", x_solution)
    print("Lagrange multiplier (λ):", lambda_solution)

    # Verify the solution
    verify_solution(A, b, C, d, x_solution)

    # Plot solution
    plot_solution(A, b, x_solution, C, d)
