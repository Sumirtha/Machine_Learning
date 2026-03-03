#Compute SONAR classification results with and without data pre-processing (data normalization).
#Perform data pre-processing with your implementation and with scikit-learn methods and compare the results.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. LOAD SONAR DATASET

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
df = pd.read_csv(url, header=None)

X = df.iloc[:, :60].values
y = df.iloc[:, 60].values

le = LabelEncoder()
y = le.fit_transform(y)  # M → 1, R → 0

print(f"Dataset Shape : {df.shape}")
print(f"Classes       : {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

# 2. CUSTOM MIN-MAX NORMALIZATION (From Scratch)

def normalize_minmax(X):
    """Normalize each column between 0 and 1"""
    X_norm = np.zeros_like(X, dtype=np.float64)
    for col in range(X.shape[1]):
        col_min = X[:, col].min()
        col_max = X[:, col].max()
        if col_max - col_min == 0:
            X_norm[:, col] = 0.0  # Avoid division by zero
        else:
            X_norm[:, col] = (X[:, col] - col_min) / (col_max - col_min)
    return X_norm

# 3. PREPARE ALL VERSIONS OF DATA

# Version 1: No normalization (raw data)
X_raw = X.copy().astype(np.float64)

# Version 2: Custom Min-Max normalization
X_custom = normalize_minmax(X)

# Version 3: Scikit-learn MinMaxScaler
scaler = MinMaxScaler()
X_sklearn = scaler.fit_transform(X)

# 4. 10-FOLD CROSS VALIDATION FOR ALL 3
kf    = KFold(n_splits=10, shuffle=True, random_state=69)
model = LogisticRegression(max_iter=1000, random_state=69)

datasets = {
    "Without Normalization  ": X_raw,
    "Custom Normalization   ": X_custom,
    "Sklearn Normalization  ": X_sklearn,
}

results = {}
print(f"  {'Method':<26} {'Mean Acc':>10} {'Std Dev':>10} {'Min':>8} {'Max':>8}")

for name, X_data in datasets.items():
    scores = cross_val_score(model, X_data, y, cv=kf, scoring='accuracy')
    results[name] = scores
    print(f"  {name:<26} {np.mean(scores):>10.7f} {np.std(scores):>10.7f} "
          f"{np.min(scores):>8.7f} {np.max(scores):>8.7f}")

# 5. FOLD-BY-FOLD COMPARISON

print(f"  {'Fold':<6} {'No Normalization':>18} {'Custom Norm':>14} {'Sklearn Norm':>14}")

for i in range(10):
    print(f"  Fold {i+1:<2}  "
          f"{results['Without Normalization  '][i]:>18.7f}  "
          f"{results['Custom Normalization   '][i]:>12.7f}  "
          f"{results['Sklearn Normalization  '][i]:>12.7f}")

print(f"  {'Mean':<6}  "
      f"{np.mean(results['Without Normalization  ']):>18.7f}  "
      f"{np.mean(results['Custom Normalization   ']):>12.7f}  "
      f"{np.mean(results['Sklearn Normalization  ']):>12.7f}")
# 6. VERIFY: Custom vs Sklearn are same
diff = np.max(np.abs(X_custom - X_sklearn))
print(f"\n Max difference between Custom & Sklearn normalization: {diff:.05f}")
#print("   (Should be ~0.0 confirming both implementations are identical)")