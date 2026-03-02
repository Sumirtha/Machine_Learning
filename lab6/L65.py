#Use validation set to do feature and model selection.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from itertools import combinations
# 1. LOAD & SPLIT DATA (Train / Val / Test)
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))

    train_end = int(len(X) * train_ratio)
    val_end = int(len(X) * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# 2. FEATURE SELECTION USING VALIDATION SET
# Try all combinations of features, pick best

feature_names = data.feature_names
num_features = X.shape[1]

best_feature_acc = 0
best_feature_set = None

print("\n Feature Selection (Logistic Regression as base model)")

for r in range(1, num_features + 1):
    for combo in combinations(range(num_features), r):
        model = LogisticRegression(max_iter=200)
        model.fit(X_train[:, combo], y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val[:, combo]))

        if val_acc > best_feature_acc:
            best_feature_acc = val_acc
            best_feature_set = combo

        print(f"  Features {combo} → Val Accuracy: {val_acc:.4f}")

print(f"\n Best Feature Set : {best_feature_set}")
print(f"   Feature Names    : {[feature_names[i] for i in best_feature_set]}")
print(f"   Val Accuracy     : {best_feature_acc:.4f}")

# 3. MODEL SELECTION USING VALIDATION SET
# Use the best features found above

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

X_train_best = X_train[:, best_feature_set]
X_val_best = X_val[:, best_feature_set]
X_test_best = X_test[:, best_feature_set]

best_model_acc = 0
best_model_name = None
best_model = None

print("\n Model Selection")

for name, model in models.items():
    model.fit(X_train_best, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val_best))
    print(f"  {name:25s} → Val Accuracy: {val_acc:.4f}")

    if val_acc > best_model_acc:
        best_model_acc = val_acc
        best_model_name = name
        best_model = model

print(f"\n Best Model : {best_model_name}")
print(f"   Val Accuracy: {best_model_acc:.4f}")

# 4. FINAL EVALUATION ON TEST SET
# Only done ONCE after all selections are made

test_acc = accuracy_score(y_test, best_model.predict(X_test_best))
print(f"\n Final Test Accuracy")
print(f"  Model    : {best_model_name}")
print(f"  Features : {[feature_names[i] for i in best_feature_set]}")
print(f"  Test Acc : {test_acc:.4f}")