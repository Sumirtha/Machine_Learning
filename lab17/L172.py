import numpy as np

# Given vectors
x1 = np.array([3, 6])
x2 = np.array([10, 10])

# Transform function
def phi(v):
    return np.array([v[0]**2, np.sqrt(2)*v[0]*v[1], v[1]**2])

# Transform both vectors
phi_x1 = phi(x1)
phi_x2 = phi(x2)

print("phi(x1):", phi_x1)
print("phi(x2):", phi_x2)

# Dot product in higher dimension
dot_product = np.dot(phi_x1, phi_x2)
print("Dot product in higher dimension:", dot_product)