import numpy as np


def hypothesis(X, theta):
    return np.dot(X, theta)

def compute_cost(X, y, theta):
    predictions = hypothesis(X, theta)
    cost = (1 / 2) * np.sum((predictions - y) ** 2)
    return cost


def sgd(X, y, theta, alpha, num_iterations):
    cost_history = []
    m = len(y)

    for i in range(num_iterations):
        # Randomly select the index of ONE sample only as it is Stochastic selection
        random_index = np.random.randint(0, m)

        xi = X[random_index, :].reshape(1, -1)
        yi = y[random_index]

        # Computing prediction for this single instance
        prediction = hypothesis(xi, theta)

        # Computing gradient for this single instance
        # Derivative: (prediction - yi) * xi
        derivative = (prediction - yi) * xi.flatten()

        # parameter update
        theta = theta - alpha * derivative

        # cost
        if i % 100 == 0:
            current_cost = compute_cost(X, y, theta)
            cost_history.append(current_cost)
            print(f"Iteration {i}: Cost = {current_cost:.4f}")

    return theta, cost_history


def main():
    # Example dataset (y = 2x)
    X_raw = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    # Add intercept term
    X = np.c_[np.ones(X_raw.shape[0]), X_raw]

    # Initialize parameters
    theta = np.zeros(X.shape[1])
    alpha = 0.01

    num_iterations = 1000

    print(f"Starting SGD for {num_iterations} iterations:")
    theta_final, cost_history = sgd(X, y, theta, alpha, num_iterations)

    print(f"\nFinal parameters (theta): {theta_final}")

    # r2 score
    predictions = hypothesis(X, theta_final)
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"R² Score: {r2:.7f}")

if __name__ == '__main__':
    main()