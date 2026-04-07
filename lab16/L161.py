import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class RandomForestRegressor:
    def __init__(self, n_trees=100, max_depth=5, random_state=69):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.rng = np.random.RandomState(random_state)
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n = X.shape[0]
        for _ in range(self.n_trees):
            idx = self.rng.choice(n, size=n, replace=True)  # bootstrap sample
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        # Average predictions across all trees
        preds = np.array([tree.predict(X) for tree in self.trees])
        return preds.mean(axis=0)


X, y = make_regression(n_samples=500, n_features=10, noise=20, random_state=69)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

rf = RandomForestRegressor(n_trees=100, max_depth=5)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.7f}")
print(f"R²   : {r2_score(y_test, y_pred):.7f}")