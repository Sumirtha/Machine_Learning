import numpy as np

x = np.array([1,2])
z = np.array([3,4])

kernel = (np.dot(x, z))**2
print("Result:", kernel)