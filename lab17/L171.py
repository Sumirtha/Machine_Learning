import numpy as np
import matplotlib.pyplot as plt

# Data
data = [
    (1,13,'Blue'), (1,18,'Blue'), (2,9,'Blue'), (3,6,'Blue'),
    (6,3,'Blue'), (9,2,'Blue'), (13,1,'Blue'), (18,1,'Blue'),
    (3,15,'Red'), (6,6,'Red'), (6,11,'Red'), (9,5,'Red'),
    (10,10,'Red'), (11,5,'Red'), (12,6,'Red'), (16,3,'Red')
]

x1 = np.array([d[0] for d in data])
x2 = np.array([d[1] for d in data])
labels = np.array([d[2] for d in data])

# Transform function
def transform(x1, x2):
    z1 = x1**2
    z2 = np.sqrt(2)*x1*x2
    z3 = x2**2
    return z1, z2, z3

z1, z2, z3 = transform(x1, x2)

# 2D Plot
plt.figure()
for label in ['Blue', 'Red']:
    idx = labels == label
    plt.scatter(x1[idx], x2[idx], label=label)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Original Space")
plt.legend()
plt.show()

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for label in ['Blue', 'Red']:
    idx = labels == label
    ax.scatter(z1[idx], z2[idx], z3[idx], label=label)

ax.set_xlabel("x1^2")
ax.set_ylabel("sqrt(2)x1x2")
ax.set_zlabel("x2^2")
ax.set_title("Transformed Space")
ax.legend()
plt.show()