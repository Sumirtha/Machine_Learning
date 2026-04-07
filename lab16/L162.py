from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

#Classifier

X, y = make_classification(n_samples=500, n_features=10, random_state=69)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                    eval_metric="logloss", random_state=69)
clf.fit(X_train, y_train)
print(f"Classifier Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.7f}")

#Regressor

X, y = make_regression(n_samples=500, n_features=10, noise=20, random_state=69)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

reg = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                   objective="reg:squarederror", random_state=69)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"Regressor R²  : {r2_score(y_test, y_pred):.7f}")