#Implement y = 2x12 + 3x1 + 4 and plot x1, y in the range [start=-10, stop=10, num=100]
#step=start-stop/num-1
#-10-10/99=-20/99=0.2
#step=0.2
import matplotlib as mpl
import matplotlib.pyplot as plt

# y = 2x^2 + 3x + 4

x = []
y = []

start = -10
stop = 10
num = 100

step = (stop - start) / (num - 1)

for i in range(num):
    xi = start + i * step          # generate x value
    yi = 2*xi*xi + 3*xi + 4        # apply formula
    x.append(xi)
    y.append(yi)

# print first 10 values
print("x        y")
for i in range(10):
    print(x[i], " ", y[i])
