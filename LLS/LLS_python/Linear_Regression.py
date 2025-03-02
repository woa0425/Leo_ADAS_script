import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate a larger dataset for speed vs. braking distance (200 points)
np.random.seed(42)
speed = np.linspace(10, 150, 200).reshape(-1, 1)  # Speeds from 10 km/h to 150 km/h

# True relationship (quadratic component added to simulate real-world data)
true_braking_distance = 0.05 * speed**2 + 2 * speed + 5  

# Add some noise to simulate real-world variability
braking_distance = true_braking_distance + np.random.normal(scale=10, size=speed.shape)

# Fit Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(speed, braking_distance)

# Fit Quadratic Regression Model
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(speed, braking_distance)

# Generate predictions
predicted_distances_linear = linear_model.predict(speed)
predicted_distances_poly = poly_model.predict(speed)

# Visualization: Linear vs Quadratic Regression
plt.scatter(speed, braking_distance, color='blue', alpha=0.5, s=10, label='Observed Data')  # Smaller dots
plt.plot(speed, predicted_distances_linear, color='red', linestyle='--', label='Linear Regression')
plt.plot(speed, predicted_distances_poly, color='green', linestyle='-', label='Quadratic Regression')
plt.xlabel("Speed (km/h)")
plt.ylabel("Braking Distance (meters)")
plt.title("Linear vs. Quadratic Regression for Braking Distance Prediction")
plt.legend()
plt.show()

# Print Linear Regression equation coefficients
linear_intercept, linear_slope = linear_model.intercept_, linear_model.coef_[0]

# Extract Quadratic Regression coefficients
poly_intercept = poly_model.named_steps['linearregression'].intercept_
poly_coefficients = poly_model.named_steps['linearregression'].coef_

# Display both models' coefficients
print("Linear Regression Equation: d = {:.2f}s + {:.2f}".format(linear_slope, linear_intercept))
print("Quadratic Regression Equation: d = {:.4f}s^2 + {:.2f}s + {:.2f}".format(poly_coefficients[0][2], poly_coefficients[0][1], poly_intercept))
