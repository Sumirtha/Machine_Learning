import numpy as np

# 1. L1 NORM
#    Formula: ||x||₁ = Σ|xᵢ|

def l1_norm(x):
    """Compute L1 norm of a vector"""
    return sum(abs(xi) for xi in x)


def l1_normalize(X):
    """
    Normalize each ROW of a 2D array using L1 norm
    x_norm = x / ||x||₁
    """
    X = np.array(X, dtype=np.float64)
    X_norm = np.zeros_like(X)

    for i in range(X.shape[0]):
        norm = l1_norm(X[i])
        if norm == 0:
            X_norm[i] = 0.0   # Avoid division by zero
        else:
            X_norm[i] = X[i] / norm
    return X_norm

# 2. L2 NORM
#    Formula: ||x||₂ = √(Σxᵢ²)

def l2_norm(x):
    """Compute L2 norm of a vector"""
    return sum(xi ** 2 for xi in x) ** 0.5

def l2_normalize(X):
    """
    Normalize each ROW of a 2D array using L2 norm
    x_norm = x / ||x||₂
    """
    X = np.array(X, dtype=np.float64)
    X_norm = np.zeros_like(X)

    for i in range(X.shape[0]):
        norm = l2_norm(X[i])
        if norm == 0:
            X_norm[i] = 0.0   # Avoid division by zero
        else:
            X_norm[i] = X[i] / norm
    return X_norm

# TEST 3: SONAR DATASET
print("SONAR DATASET - NORM COMPARISON")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, Normalizer

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
df  = pd.read_csv(url, header=None)
X_sonar = df.iloc[:, :60].values.astype(np.float64)
y_sonar = LabelEncoder().fit_transform(df.iloc[:, 60].values)

# Apply all versions
X_raw    = X_sonar.copy()
X_l1_son = l1_normalize(X_sonar)
X_l2_son = l2_normalize(X_sonar)

# Sklearn Normalizer for verification
X_sk_l1  = Normalizer(norm='l1').fit_transform(X_sonar)
X_sk_l2  = Normalizer(norm='l2').fit_transform(X_sonar)

kf    = KFold(n_splits=10, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000, random_state=42)

datasets = {
    "No Normalization" : X_raw,
    "Custom L1 Norm"   : X_l1_son,
    "Custom L2 Norm"   : X_l2_son,
    "Sklearn L1 Norm"  : X_sk_l1,
    "Sklearn L2 Norm"  : X_sk_l2,
}

print(f"\n  {'Method':<22} {'Mean Acc':>10} {'Std Dev':>10} {'Min':>8} {'Max':>8}")

for name, X_data in datasets.items():
    scores = cross_val_score(model, X_data, y_sonar, cv=kf, scoring='accuracy')
    print(f"  {name:<22} {np.mean(scores):>10.7f} {np.std(scores):>10.7f}"
          f" {np.min(scores):>8.7f} {np.max(scores):>8.7f}")