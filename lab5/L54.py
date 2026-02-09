from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

reg = LogisticRegression(max_iter=10000, random_state=0)
reg.fit(X_train, y_train)

acc = accuracy_score(y_test, reg.predict(X_test)) * 100
print(f"Logistic Regression model accuracy: {acc:.4f}%")