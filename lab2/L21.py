# Defining X 
X = [
 [1,0,2],
 [0,1,1],
 [2,1,0],
 [1,1,1],
 [0,2,1]
]

n = len(X)        # samples = 5
m = len(X[0])     # features = 3

# Column means 
mean = [0]*m

for j in range(m):
    s = 0
    for i in range(n):
        s += X[i][j]
    mean[j] = s / n

#print("Mean of each feature:", mean)

# Center the data (X - mean)
Xc = []

for i in range(n):
    row = []
    for j in range(m):
        row.append(X[i][j] - mean[j])
    Xc.append(row)

#print("\nCentered Matrix:")
#for r in Xc:
#    print(r)

# Compute covariance = (1/(n-1)) * Xc^T Xc
cov = []
for i in range(m):
    row = []
    for j in range(m):
        s = 0
        for k in range(n):
            s += Xc[k][i] * Xc[k][j]
        row.append(s/(n-1))
    cov.append(row)

print("\nCovariance Matrix:")
for r in cov:
    print(r)
#verifying using numpy
import numpy as np

X = np.array([[1, 0, 2],
              [0, 1, 1],
              [2, 1, 0],
              [1, 1, 1],
              [0, 2, 1]])

# rowvar=False because columns are the features
cov_matrix = np.cov(X, rowvar=False)
print("\nCov matrix using numpy", cov_matrix)