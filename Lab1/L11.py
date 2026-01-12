#A=[1,2,3
#4,5,6]
#import numpy as np
#A=np.array([[1,2,3],[4,5,6]])
#B=A.T
#print(B)

A = [
    [1, 2, 3],
    [4, 5, 6]
]

rows = len(A)
cols = len(A[0])

# create empty transpose matrix (3x2)
AT = []
for j in range(cols):
    row = []
    for i in range(rows):
        row.append(A[i][j])
    AT.append(row)

print("Transpose A^T =")
for r in AT:
    print(r)
# A^T is 3x2 and A is 2x3
# result will be 3x3

result = []
for i in range(len(AT)):
    row = []
    for j in range(len(A[0])):
        s = 0
        for k in range(len(A)):
            s += AT[i][k] * A[k][j]
        row.append(s)
    result.append(row)

print("A^T A =")
for r in result:
    print(r)
