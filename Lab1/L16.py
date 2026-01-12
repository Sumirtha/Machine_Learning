#Implement y = 2x1 + 3x2 + 3x3 + 4, where x1, x2 and x3 are three independent variables.
# Compute the gradient of y and print the values.
def gradient(x1, x2, x3):
    return (2, 3, 3)

print("Gradient at (1,2,3):", gradient(1,2,3))
print("Gradient at (0,0,0):", gradient(0,0,0))
