import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Keep only classes 1 and 2
X = X[y != 0, :2]   # first 2 features
y = y[y != 0]

# Convert labels to 0 and 1
y = (y == 2).astype(int)

# Split data (90% train, 10% test)
# Ensures both classes present
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=69
)

# Train SVM models
linear_svm = svm.SVC(kernel='linear')
rbf_svm = svm.SVC(kernel='rbf', gamma=0.5)

linear_svm.fit(X_train, y_train)
rbf_svm.fit(X_train, y_train)

# Evaluate performance

print("Linear SVM Accuracy:", linear_svm.score(X_test, y_test))
print("RBF SVM Accuracy:", rbf_svm.score(X_test, y_test))

def plot_model(model, title):
    plt.figure()

    # Plot data
    for i in range(len(X)):
        if y[i] == 0:
            plt.scatter(X[i,0], X[i,1])
        else:
            plt.scatter(X[i,0], X[i,1])

    # Grid
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100),
        np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_model(linear_svm, "Linear SVM")
plot_model(rbf_svm, "RBF SVM")
