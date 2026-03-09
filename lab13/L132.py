#Implement bagging regressor without using scikit-learn
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def bagging_regressor(X_train, y_train, X_test, n_estimators=10):
    predictions = []

    for i in range(n_estimators):
        # Bootstrap sampling
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_sample = X_train[indices]
        y_sample = y_train[indices]

        # Train base model
        model = DecisionTreeRegressor()
        model.fit(X_sample, y_sample)

        # Store predictions
        pred = model.predict(X_test)
        predictions.append(pred)

    # Average predictions
    final_pred = np.mean(predictions, axis=0)
    return final_pred


def main():
    data = load_diabetes()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    y_pred = bagging_regressor(X_train, y_train, X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


if __name__ == "__main__":
    main()