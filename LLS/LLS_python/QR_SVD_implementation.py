import numpy as np
import matplotlib.pyplot as plt

# ----- Step 1: Generate Well-Conditioned Data -----
np.random.seed(42)  # For reproducibility
m = 50  # Number of observations
x = np.linspace(0, 10, m)  # Well-spaced x values
true_slope = 2.5
true_intercept = 1.0
noise = np.random.normal(scale=2.0, size=m)  # Random noise
y = true_slope * x + true_intercept + noise  # Generate noisy observations

# Construct A matrix for least squares problem (Ax ≈ b)
A = np.vstack([x, np.ones(m)]).T  # Design matrix with [x, 1] columns
b = y  # Observation vector

# ----- Step 2: Solve Using Normal Equation -----
x_ls_normal = np.linalg.inv(A.T @ A) @ A.T @ b  # Direct inverse method

# ----- Step 3: Solve Using QR Decomposition -----
# QR decomposition factroizes the matrix A into Q and R such that A = QR
# Q is an orthogonal matrix and R is an upper triangular matrix
Q, R = np.linalg.qr(A)  # Compute QR decomposition
x_ls_qr = np.linalg.solve(R, Q.T @ b)  # Solve Rx = Q^T*b

# ----- Step 4: Solve Using SVD(Singular Value Decomposition) -----
# SVD decomposes the matrix A into U, S, and V such that A = U*S*V^T
# U and V are orthogonal matrices and S is a diagonal matrix
U, S, Vt = np.linalg.svd(A, full_matrices=False)  # Compute SVD decomposition
x_ls_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ b  # Solve using SVD

# Generate fitted values for all three methods
y_fit_normal = x_ls_normal[0] * x + x_ls_normal[1]
y_fit_qr = x_ls_qr[0] * x + x_ls_qr[1]
y_fit_svd = x_ls_svd[0] * x + x_ls_svd[1]

# ----- Step 5: Plot results -----
plt.figure(figsize=(9, 5))
plt.scatter(x, y, label="Noisy Data", color="blue", alpha=0.6)  # Raw data points

# Normal Equation Solution
plt.plot(x, y_fit_normal, 
         label=f"Normal Equation: y = {x_ls_normal[0]:.2f}x + {x_ls_normal[1]:.2f}", 
         color="green", linestyle="dashed", linewidth=2, alpha=0.9)

# QR Decomposition Solution
plt.plot(x, y_fit_qr, 
         label=f"QR Decomposition: y = {x_ls_qr[0]:.2f}x + {x_ls_qr[1]:.2f}", 
         color="red", linewidth=2)

# SVD Solution
plt.plot(x, y_fit_svd, 
         label=f"SVD: y = {x_ls_svd[0]:.2f}x + {x_ls_svd[1]:.2f}", 
         color="purple", linestyle="dotted", linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Normal Equation vs QR vs SVD for Least Squares")
plt.grid()
plt.show()

# ----- Step 6: Print solutions -----
print("\nSolution using Normal Equation:")
print(f"Slope: {x_ls_normal[0]:.5f}, Intercept: {x_ls_normal[1]:.5f}")

print("\nSolution using QR Decomposition:")
print(f"Slope: {x_ls_qr[0]:.5f}, Intercept: {x_ls_qr[1]:.5f}")

print("\nSolution using SVD:")
print(f"Slope: {x_ls_svd[0]:.5f}, Intercept: {x_ls_svd[1]:.5f}")

# ----- Step 7: Compare Differences -----
diff_qr_normal = np.abs(x_ls_normal - x_ls_qr)
diff_svd_normal = np.abs(x_ls_normal - x_ls_svd)
diff_svd_qr = np.abs(x_ls_qr - x_ls_svd)

print(f"\n Difference between Normal Equation and QR: {diff_qr_normal}")
print(f" Difference between Normal Equation and SVD: {diff_svd_normal}")
print(f" Difference between QR and SVD: {diff_svd_qr}")
