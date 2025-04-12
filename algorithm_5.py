# Implementation of the fifth basic machine learning algorithm, K-Nearest Neighbors (KNN). The code will be implemented and placed here.
# Import necessary libraries
import numpy as np
from collections import Counter

# Generate or load data
# For simplicity, we'll create a synthetic dataset
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Define the distance metric
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, x_test, k=3):
    distances = [euclidean_distance(x_test, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# Make predictions on new data
new_X = np.array([[0.5, 0.5], [1.5, 1.5]])
predictions = [knn_predict(X, y, x, k=3) for x in new_X]
print('Predictions:', predictions)