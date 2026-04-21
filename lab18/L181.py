import numpy as np

# Data
X = np.array([
    [6,5],[6,9],[8,6],[8,8],[8,10],[9,2],[9,5],[10,10],
    [10,13],[11,5],[11,8],[12,6],[12,11],[13,4],[14,8]
])

# Labels (Blue=0, Red=1)
y = np.array([
    0,0,1,1,1,0,1,1,
    0,1,1,1,0,0,0
])

from sklearn import svm
import matplotlib.pyplot as plt

# Models
rbf_model = svm.SVC(kernel='rbf', gamma=0.1)
poly_model = svm.SVC(kernel='poly', degree=2)

# Train
rbf_model.fit(X, y)
poly_model.fit(X, y)

def plot_boundary(model, title):
    plt.figure()

    # Plot points
    for i in range(len(X)):
        if y[i] == 0:
            plt.scatter(X[i,0], X[i,1])
        else:
            plt.scatter(X[i,0], X[i,1])

    # Grid
    xx, yy = np.meshgrid(
        np.linspace(5,15,100),
        np.linspace(2,15,100)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

# Plot both
plot_boundary(rbf_model, "RBF Kernel")
plot_boundary(poly_model, "Polynomial Kernel")

print("RBF Accuracy:", rbf_model.score(X, y))
print("Polynomial Accuracy:", poly_model.score(X, y))