#Implement decision tree classifier without using scikit-learn using the iris dataset. Fetch the iris dataset from scikit-learn library.

import numpy as np
from sklearn.datasets import load_iris

# Load iris dataset
data = load_iris()
X = data.data
y = data.target

# Entropy function
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

# Information gain
def information_gain(X_column, y, threshold):
    parent_entropy = entropy(y)

    left_mask = X_column <= threshold
    right_mask = X_column > threshold

    if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
        return 0

    n = len(y)
    n_left, n_right = len(y[left_mask]), len(y[right_mask])

    child_entropy = (n_left/n)*entropy(y[left_mask]) + (n_right/n)*entropy(y[right_mask])
    return parent_entropy - child_entropy

# Find best split
def best_split(X, y):
    best_gain = -1
    split_feature, split_threshold = None, None

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            gain = information_gain(X[:, feature], y, t)
            if gain > best_gain:
                best_gain = gain
                split_feature = feature
                split_threshold = t

    return split_feature, split_threshold

# Simple decision tree
def decision_tree(X, y):
    feature, threshold = best_split(X, y)
    print("Best Feature:", feature)
    print("Best Threshold:", threshold)

    left = y[X[:, feature] <= threshold]
    right = y[X[:, feature] > threshold]

    print("Left class:", np.bincount(left).argmax())
    print("Right class:", np.bincount(right).argmax())

# Main
def main():
    decision_tree(X, y)

if __name__ == "__main__":
    main()