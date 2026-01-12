#Implement y = 2x1 + 3 and plot x1, y [start=-100, stop=100, num=100]
import matplotlib as mpl
import matplotlib.pyplot as plt
#start= -100
#stop= 100
#num=100

#step = (start-stop)/(num -1)
#for i in range(num):
 #        x1=append
#plt.subplot(num,1,i+1)
#y= 2*x1+3
#plt.subplot(num,1,1)

# define empty lists
x = []
y = []

start = -100
stop = 100
num = 100

step = (stop - start) / (num - 1)

for i in range(num):
    xi = start + i * step     # generate x value
    yi = 2 * xi + 3           # y = 2x + 3
    x.append(xi)
    y.append(yi)

# print first 10 values to verify
print("x        y")
for i in range(10):
    print(x[i], " ", y[i])

