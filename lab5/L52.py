import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# def sigmoidDeriv(x):
#     return sigmoid(x) * (1 - sigmoid(x))

x= np.linspace(-10, 10, 1000)
y = sigmoid(x)

# for visualization
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=3, label='σ(x) = 1/(1+e⁻ˣ)')
plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='y = 0.5')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.scatter([0], [0.5], color='red', s=100, zorder=5)
#
# plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=14)
plt.ylabel('σ(x)', fontsize=14)
plt.title('SIGMOID FUNCTION', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.ylim(-0.1, 1.1)
plt.show()