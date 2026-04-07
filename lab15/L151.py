# # Sample predictions from 3 trees for 5 data points
# tree1 = [10.0, 12.0, 14.0, 16.0, 18.0]
# tree2 = [11.0, 13.0, 15.0, 17.0, 19.0]
# tree3 = [9.0, 11.0, 13.0, 15.0, 17.0]
#
# # Combine all trees
# trees = [tree1, tree2, tree3]
#
# # Aggregate predictions (average)
# final_prediction = []
#
# for i in range(len(tree1)):
#     sum_pred = 0
#     for tree in trees:
#         sum_pred += tree[i]
#     avg = sum_pred / len(trees)
#     final_prediction.append(avg)
#
# # Output
# print("Final Predictions:", final_prediction)

#from islp import load_data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# --- Regression: Boston Housing ---
boston = pd.read_csv("/home/ibab/PyCharmMiscProject/ML/ALL+CSV+FILES+-+2nd+Edition+-+corrected/ALL CSV FILES - 2nd Edition/Boston.csv")
X, y = boston.drop("medv", axis=1), boston["medv"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=69)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"Regression  — RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.7f}  R²: {r2_score(y_test, y_pred):.7f}")

# --- Classification: Weekly / Direction ---
# weekly = load_data("Weekly")
# X = weekly.drop("Direction", axis=1).select_dtypes(include="number")
# y = LabelEncoder().fit_transform(weekly["Direction"])   # Down=0, Up=1
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
# clf.fit(X_train, y_train)
# print(f"Classification — Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.4f}")