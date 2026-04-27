import numpy as np

def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        points = X[labels == i]
        if len(points) > 0:
            new_centroids.append(points.mean(axis=0))
        else:
            # handle empty cluster
            new_centroids.append(X[np.random.randint(0, X.shape[0])])
    return np.array(new_centroids)

def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)

    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


if __name__ == "__main__":
 
    X = np.array([
        [1, 2], [1, 1], [5, 8],
        [8, 2], [1, 6], [5, 11]
    ])

    k = 7
    centroids, labels = kmeans(X, k)

    print("Centroids:\n", centroids)
    print("Labels:\n", labels)
