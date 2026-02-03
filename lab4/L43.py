import numpy as np
import pandas as pd

# Read the file
def read_data(filepath):
    return pd.read_csv('/home/ibab/PyCharmMiscProject/ML/lab4/Admission_Predict.csv')

# Form x(features) and y(target)
def form_x_y(df_data):
    X = df_data.iloc[:, :-1].values
    y = df_data.iloc[:, -1].values
    return X, y

def fit(X, y):
    X_b = np.c_[np.ones(len(X)), X]

    # Normal equation: theta = (X^T X)^-1 X^T y
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta

def predict(X, theta):
    X_b = np.c_[np.ones(len(X)), X]

    # Prediction: y = X * theta
    return X_b.dot(theta)

def main():
    # Read data
    df_data = read_data('/home/ibab/PyCharmMiscProject/ML/lab4/Admission_Predict.csv')

    # Form X and y
    X, y = form_x_y(df_data)

    # Fit model
    theta = fit(X, y)

    # Make predictions
    prediction = predict(X, theta)

    print("Predictions:")
    print(prediction)
    print(f"\nTheta (coefficients): {theta}")

if __name__ == "__main__":
    main()