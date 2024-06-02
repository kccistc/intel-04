import numpy as  np


n = 6
tmp = 1

A = np.zeros((n, n))

for i in range(n):
    for k in range(n):
        A[i][k] = tmp
        tmp += 1

A = A.reshape(A.size)

print(A)