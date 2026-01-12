#Implement y = x1^2, plot x1, y in the range [start=--10, stop=10, num=100].
#Compute the value of derivatives at these points, x1 = -5, -3, 0, 3, 5.
#What is the value of x1 at which the function value (y) is zero. What do you infer from this?

def derivative(x):
    return 2*x

points = [-5, -3, 0, 3, 5]

for p in points:
    print("x =", p, "dy/dx =", derivative(p))

# zero of function
print("y = x^2 is zero at x = 0")
