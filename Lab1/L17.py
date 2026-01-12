X = [
 [1,0,2],
 [0,1,1],
 [2,1,0],
 [1,1,1],
 [0,2,1]
]

theta = [2,3,3]

y = []

for row in X:
    s = 0
    for i in range(3):
        s += row[i] * theta[i]
    y.append(s)

print("Xθ =", y)
