#Implement Gaussian PDF - mean = 0, sigma = 15 in the range[start=-100, stop=100, num=100]

import math

mu = 0
sigma = 15

x = []
y = []

start = -100
stop = 100
num = 100

step = (stop - start) / (num - 1)

for i in range(num):
    xi = start + i * step

    exponent = -((xi - mu) ** 2) / (2 * sigma * sigma)
    yi = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(exponent)

    x.append(xi)
    y.append(yi)

# print first 10 values
print("x        Gaussian(x)")
for i in range(10):
    print(x[i], " ", y[i])
