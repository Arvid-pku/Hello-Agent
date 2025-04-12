# Implementation of Linear Regression
# Import necessary libraries
import numpy as np

# Generate or load data
# For simplicity, we'll create a synthetic dataset
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)

# Define the hypothesis and cost function
def hypothesis(X, theta):
    return np.dot(X, theta)

def cost_function(X, y, theta):
    m = len(y)
    J = (1/(2*m)) * np.sum((hypothesis(X, theta) - y)**2)
    return J

# Implement gradient descent to learn model parameters
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = [0] * iterations
    for iteration in range(iterations):
        h = hypothesis(X, theta)
        error = h - y
        gradient = (1/m) * np.dot(X.T, error)
        theta = theta - learning_rate * gradient
        cost = cost_function(X, y, theta)
        cost_history[iteration] = cost
    return theta, cost_history

# Prepare data and add bias term
X_b = np.c_[np.ones((len(X), 1)), X]
theta = np.zeros((2, 1))

# Train the model
learning_rate = 0.01
iterations = 1000
theta, cost_history = gradient_descent(X_b, y, theta, learning_rate, iterations)

# Print learned parameters
print('Learned Parameters:', theta)

# Predict using the learned model
def predict(X, theta):
    X_b = np.c_[np.ones((len(X), 1)), X]
    return hypothesis(X_b, theta)

# Make predictions on new data
new_X = np.array([[0], [2]])
predictions = predict(new_X, theta)
print('Predictions:', predictions)