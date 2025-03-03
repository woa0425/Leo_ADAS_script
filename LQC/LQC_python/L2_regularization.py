import numpy as np
import matplotlib.pyplot as plt

def compute_basis(x1, x2, scale=1.5):
    """Compute the basis function using a Gaussian kernel."""
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * scale ** 2))

def generate_dataset():
    """Generate synthetic data for regression."""
    x_values = np.linspace(0, 8, 100)
    true_values = np.exp(-0.1 * x_values) * np.cos(x_values)

    # Select random training points
    num_samples = 15
    indices = np.sort(np.random.choice(len(x_values), num_samples, replace=False))
    x_train = x_values[indices]
    y_train = true_values[indices] + 0.05 * np.random.randn(num_samples)

    # Introduce outliers
    outlier_indices = np.random.choice(num_samples, size=3, replace=False)
    y_train[outlier_indices] += (2 * np.random.rand(len(outlier_indices)) - 1)

    return x_values, true_values, x_train, y_train

def construct_design_matrix(x_data, basis_x):
    """Construct the design matrix using basis functions."""
    num_basis = len(basis_x)
    Phi = np.ones((len(x_data), num_basis + 1))  # Include bias term

    for i in range(len(x_data)):
        for j in range(num_basis):
            Phi[i, j + 1] = compute_basis(x_data[i], basis_x[j])

    return Phi

def solve_ridge_regression(Phi, targets, reg_strength=1e-3):
    """Solve for regression weights using L2 regularization."""
    identity_matrix = reg_strength * np.eye(Phi.shape[1])  # Regularization term
    weights = np.linalg.inv(Phi.T @ Phi + identity_matrix) @ Phi.T @ targets
    return weights

def iterative_reweighted_least_squares(Phi, targets, max_iter=500, threshold=1e-9, alpha=0.1):
    """Perform robust regression using Iteratively Reweighted Least Squares (IRLS)."""
    num_samples = len(targets)
    B = np.eye(num_samples)
    weights = solve_ridge_regression(Phi, targets)

    for iteration in range(max_iter):
        prev_weights = weights.copy()

        # Update weights matrix B
        for i in range(num_samples):
            residual = targets[i] - Phi[i, :] @ weights
            B[i, i] = 1 / (1 + (residual / alpha) ** 2)

        B /= np.sum(B)  # Normalize

        # Solve for new weights
        weights = np.linalg.inv(Phi.T @ B @ Phi + 1e-3 * np.eye(Phi.shape[1])) @ (Phi.T @ B @ targets)

        # Check convergence
        if np.linalg.norm(weights - prev_weights) < threshold:
            print(f'Converged in {iteration} iterations.')
            break

    return weights

def plot_results(x_values, true_values, x_train, y_train, predictions):
    """Plot the true function, training data, and model predictions."""
    plt.figure()
    plt.plot(x_values, true_values, label='True Function', linewidth=3)
    plt.scatter(x_train, y_train, color='red', label='Training Data')
    plt.plot(x_values, predictions, linestyle='--', color='black', linewidth=3, label='Model Prediction')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Predictions by L2 Regularization")
    plt.show()


if __name__ == "__main__":
    # Generate dataset
    x_values, true_values, x_train, y_train = generate_dataset()

    # Construct design matrix
    Phi_train = construct_design_matrix(x_train, x_train)
    Phi_test = construct_design_matrix(x_values, x_train)

    # Solve for weights using robust regression
    learned_weights = iterative_reweighted_least_squares(Phi_train, y_train)

    # Predict outputs
    predictions = Phi_test @ learned_weights

    # Plot results
    plot_results(x_values, true_values, x_train, y_train, predictions)
    plot_weights(learned_weights)

# Reference: 
# Robust Linear Regression via M-Estimation
# Author: Fangtong Liu
# Date: 06/14/2020