#Implement a decision regression tree algorithm without using scikit-learn using the diabetes dataset. Fetch the dataset from scikit-learn library.

import numpy as np
from sklearn.datasets import load_diabetes

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Mean Squared Error
def mse(y):
    return np.mean((y - np.mean(y))**2)

# Find best split
def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_error = float("inf")

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])

        for t in thresholds:
            left = y[X[:, feature] <= t]
            right = y[X[:, feature] > t]

            if len(left) == 0 or len(right) == 0:
                continue

            error = (len(left)*mse(left) + len(right)*mse(right)) / len(y)

            if error < best_error:
                best_error = error
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold

# Simple regression tree (single split)
def regression_tree(X, y):
    feature, threshold = best_split(X, y)

    left = y[X[:, feature] <= threshold]
    right = y[X[:, feature] > threshold]

    print("Best Feature:", feature)
    print("Best Threshold:", threshold)
    print("Left Mean Prediction:", np.mean(left))
    print("Right Mean Prediction:", np.mean(right))

def main():
    regression_tree(X, y)

if __name__ == "__main__":
    main()