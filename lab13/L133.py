#Implement Random Forest algorithm for regression and classification using scikit-learn. Use diabetes and iris datasets.

from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

def regression_rf():
    data = load_diabetes()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Regression MSE:", mean_squared_error(y_test, y_pred))


def classification_rf():
    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Accuracy:", accuracy_score(y_test, y_pred))


def main():
    regression_rf()
    classification_rf()


if __name__ == "__main__":
    main()