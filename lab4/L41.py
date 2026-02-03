import pandas as pd
import numpy as np

#Read the file
def read_data():
    return pd.read_csv()

#form x(features)and y (target)
def form_x_y(df_data):
    X=df_data.drop(axis=1).values
    y=df_data.values
    return X,y

#computing hypothesis
def hypothesis(X,theta):
    return np.dot(X,theta)

#computing cost function
def compute_cost(X, y, theta):
    predictions = hypothesis(X,theta)
    cost=(1/2)*np.sum((predictions-y)**2)
    return cost

#computing derivative
def compute_derivative(X, y, theta):
    predictions = hypothesis(X,theta)
    derivative = np.dot(X.T,(predictions-y))
    return derivative

#gradient_descent
def gradient_descent(X, y, theta, alpha, num_iters):
    cost_history = []

    for i in range(num_iters):

        #compute derivative
        derivative = compute_derivative(X, y, theta)

        #update theta
        theta = theta - alpha * derivative

        #computing cost
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        if i % 1000 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return theta, cost_history

def main():
    #read data
    df_data = pd.read_csv()
    X, y = form_x_y(df_data)

    # Add intercept term (column of ones)
    X = np.c_[np.ones(X.shape[0]), X]

    # Normalize features
    X_mean = np.mean(X[:, 1:], axis=0)
    X_std = np.std(X[:, 1:], axis=0)
    X[:, 1:] = (X[:, 1:] - X_mean) / X_std
    #initializing parameter
    theta = np.zeros(X.shape[1])
    alpha = 0.00001
    num_iters = 25000

    #running gradient descent
    print("Starting Gradient Descent:")
    theta_final, cost_history = gradient_descent(X, y, theta, alpha, num_iters)

    print(f"\nFinal parameters (theta): {theta_final}")

    print(f"Final cost: {cost_history[-1]:.7f}")

    # making predication
    predictions = hypothesis(X, theta_final)

    print(f"Initial predictions: {predictions[:50]}")
    print(f"y values: {y[:5]}")

    # Calculating R² score
    ss = np.sum((y - predictions) ** 2)
    s = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss / s)

    print(f"r² Score: {r2:.7f}")

if __name__ == '__main__':
    main()