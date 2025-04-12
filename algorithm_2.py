# Implementation of Logistic Regression
# Import necessary libraries
import numpy as np

# Generate or load data
# For simplicity, we'll create a synthetic dataset
X = np.random.rand(100, 1)
y = (4 * X + 3 > np.random.randn(100, 1)).astype(int)

# Define the sigmoid function and cost function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))
    return J[0][0]

# Implement gradient descent to learn model parameters
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = [0] * iterations
    for iteration in range(iterations):
        h = sigmoid(np.dot(X, theta))
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
    probabilities = sigmoid(np.dot(X_b, theta))
    return (probabilities >= 0.5).astype(int)

# Make predictions on new data
new_X = np.array([[0], [2]])
predictions = predict(new_X, theta)
print('Predictions:', predictions)